"""B512 val 결과표 페이지용 데이터와 사진을 만든다 (점수는 저장된 결과 파일에서 읽기만 한다, 모델을 돌리지 않는다).

만드는 것 (--out 폴더):
    val_conditions.json   원본 + 33조건 한 줄씩 (점수, 변화, 구간, 클래스별 recall, 예측 비율, 필터 설명)   ← reports/equipment_v2/val_B512/shift_results.csv
    val_confusion.json    원본 val 혼동행렬 (행 = 정답)                                                      ← runs/results/b512-.../val_predictions.csv
    photos-N.js           같은 정상 흉부 X선 한 장에 33조건을 건 사진 (전체 144px + 실제 픽셀 128px 확대), data: URL, 파일당 200KB 안팎

사진은 예시 한 장이다 (train 정상, 점수를 낸 val 사진이 아니다). 필터는 모델 시험과 같은 코드(training/equipment_direct.py,
training/shift_filters.py)로, 노이즈 seed는 42.

사용 예 (p1 폴더에서):
    .venv/bin/python src/equipment_results_page.py --out <폴더>
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from equipment_direct import EQUIP, apply_equipment  # noqa: E402
from shift_filters import SHIFTS, apply_shift, to_gray  # noqa: E402

SIZE = 512
EXAMPLE = "train/NORMAL/IM-0487-0001.jpeg"
THUMB, CROP = 144, 128
SEED = 42
CHUNK = 190_000  # 한 JS 파일의 글자 수 상한 (저장소 한도 256KB 아래)

ORDER = ["dose", "motion", "scatter", "kvp", "sensor", "sharpen", "equalize", "jpeg", "blur", "bright", "dark"]
META = {  # 이름, 분류, 장치 원리, 구분
    "dose": ("선량 (mAs)", "기기·촬영 조건", "선량이 줄면 광자가 줄어 양자 노이즈가 늘어난다", "본편"),
    "motion": ("조사시간 (움직임)", "기기·촬영 조건", "오래 찍으면 아이가 움직여 세로로 번진다. 아래쪽일수록 더 번진다", "본편"),
    "scatter": ("산란 (격자 없음)", "기기·촬영 조건", "산란선이 안개처럼 깔려 작은 구조의 대비가 줄어든다", "본편"),
    "kvp": ("관전압 kVp ↑", "기기·촬영 조건", "관전압이 높으면 대비가 줄고 밝은 부위(뼈)가 덜 하얘진다. 강도 = 대비 배율 · 밝은 쪽 기울기", "본편"),
    "sensor": ("센서 화소 크기", "기기·촬영 조건", "화소가 커지면 미세한 무늬가 뭉개진다", "본편"),
    "sharpen": ("경계 강조", "제조사 후처리", "경계와 미세 무늬를 또렷하게 만드는 후처리. 언샤프 마스킹으로 대신했다", "본편"),
    "equalize": ("대비 보정", "제조사 후처리", "구역별 밝기 분포를 다시 맞추는 후처리. CLAHE와 히스토그램 평활화(HE)로 대신했다", "본편"),
    "jpeg": ("JPEG 압축", "전송", "재압축하면 8×8 블록 무늬가 생기고 미세 무늬가 사라진다", "본편"),
    "blur": ("흐림", "초점·촬영 거리", "초점 크기와 촬영 거리로 생기는 번짐. 처음에 일반 전처리로 정해 본편에서 뺐다", "부록"),
    "bright": ("밝게", "표시·노출 설정", "과다 노출이나 밝게 맞춘 표시 설정. 처음에 일반 전처리로 정해 본편에서 뺐다", "부록"),
    "dark": ("어둡게", "표시·노출 설정", "노출 부족이나 어둡게 맞춘 표시 설정. 처음에 일반 전처리로 정해 본편에서 뺐다", "부록"),
}
VALUE = {
    "dose": ["1/4", "1/16", "1/64"], "motion": ["2 mm", "4 mm", "8 mm"], "scatter": ["k 0.25", "k 0.5", "k 1.0"],
    "kvp": ["0.85 · 0.8", "0.70 · 0.6", "0.60 · 0.5"], "sensor": ["0.5 mm", "1.0 mm", "2.0 mm"],
    "sharpen": ["α 0.5", "α 1.0", "α 2.0"], "equalize": ["CLAHE 1.5", "CLAHE 3.0", "HE"], "jpeg": ["품질 75", "품질 40", "품질 20"],
    "blur": ["σ 0.5", "σ 1.0", "σ 2.0"], "bright": ["감마 0.85", "감마 0.7", "감마 0.55"], "dark": ["감마 1.2", "감마 1.45", "감마 1.8"],
}
LEVEL_LABEL = {0: "원본", 1: "약", 2: "중", 3: "강"}
# 확대 부위 (512 사진의 행 시작, 열 시작; 128×128): 폐 윗부분 / 폐 아래·횡격막 / 가운데(척추·심장)
REGION = {"upper": (160, 69), "lower": (320, 69), "center": (206, 174)}
REGION_OF = {"motion": "lower", "scatter": "center", "kvp": "center"}  # 나머지는 upper


def minus(s):
    return s.replace("-", "−")


def make(gray, name, level):
    if name in EQUIP:
        return apply_equipment(gray, name, level, seed=SEED)
    return apply_shift(gray, name, level, seed=SEED)


def jpeg_url(arr, quality):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def thumb(arr):
    small = np.asarray(Image.fromarray(arr).resize((THUMB, THUMB), Image.LANCZOS))
    return jpeg_url(small, 85)


def crop(arr, region):
    r, c = REGION[region]
    return jpeg_url(arr[r:r + CROP, c:c + CROP], 90)


def build_rows(csv_path):
    d = pd.read_csv(csv_path)
    n_images = 1030
    clean = d[d["shift"] == "clean"].iloc[0]
    rows = []

    def row(key, name, level, r):
        sig = None
        if level:
            sig = int(r.delta_f1_ci_high < 0 or r.delta_f1_ci_low > 0)
        title, cat, cause, group = META[name] if name else ("원본 (필터 없음)", "기준", "필터를 걸지 않은 val 사진", "원본")
        return {
            "key": key, "group": group, "filter": name or "clean", "filter_title": title, "category": cat, "cause": cause,
            "level": level, "level_label": LEVEL_LABEL[level], "value_label": VALUE[name][level - 1] if level else "—",
            "macro_f1": round(float(r.macro_f1), 3), "auroc": round(float(r.macro_auroc), 3),
            "delta_f1": round(float(r.delta_f1), 3) if level else None,
            "ci_text": minus(f"[{r.delta_f1_ci_low:+.3f}, {r.delta_f1_ci_high:+.3f}]") if level else "—", "significant": sig,
            "recall_normal": round(float(r.recall_normal), 3), "recall_bacterial": round(float(r.recall_bacterial), 3),
            "recall_viral": round(float(r.recall_viral), 3),
            "share_normal": round(float(r.pred_share_normal), 3), "share_bacterial": round(float(r.pred_share_bacterial), 3),
            "share_viral": round(float(r.pred_share_viral), 3),
            "share_text": f"{r.pred_share_normal * 100:.0f} / {r.pred_share_bacterial * 100:.0f} / {r.pred_share_viral * 100:.0f}",
            "changed_pct": round(float(r.changed_pred_pct), 1) if level else None, "n_images": n_images,
        }

    rows.append(row("clean-0", None, 0, clean))
    idx = d.set_index(["shift", "level"])
    for name in ORDER:
        for level in (1, 2, 3):
            rows.append(row(f"{name}-{level}", name, level, idx.loc[(name, level)]))
    return rows


def build_confusion(pred_path):
    v = pd.read_csv(pred_path)
    cm = np.zeros((3, 3), int)
    for t, p in zip(v.true_label, v.pred_label):
        cm[t, p] += 1
    names = ["정상", "세균성", "바이러스성"]
    keys = ["normal", "bacterial", "viral"]
    return [{"key": keys[i], "true_label": names[i], "pred_normal": int(cm[i, 0]), "pred_bacterial": int(cm[i, 1]),
             "pred_viral": int(cm[i, 2]), "n_true": int(cm[i].sum())} for i in range(3)]


def build_photos(images_root):
    with Image.open(images_root / EXAMPLE) as im:
        base = to_gray(im, SIZE)
    t, c = {"clean-0": thumb(base)}, {}
    for region in REGION:
        c[f"clean@{region}"] = crop(base, region)
    for name in ORDER:
        region = REGION_OF.get(name, "upper")
        for level in (1, 2, 3):
            arr = make(base, name, level)
            t[f"{name}-{level}"] = thumb(arr)
            c[f"{name}-{level}"] = crop(arr, region)
    return t, c


def write_js(out, thumbs, crops):
    items = [("t", k, v) for k, v in thumbs.items()] + [("c", k, v) for k, v in crops.items()]
    files, cur, size = [], [], 0
    for kind, k, v in items:
        line = f'PH.{kind}[{json.dumps(k)}]={json.dumps(v)};\n'
        if size + len(line) > CHUNK and cur:
            files.append(cur)
            cur, size = [], 0
        cur.append(line)
        size += len(line)
    files.append(cur)
    names = []
    for i, lines in enumerate(files, 1):
        name = f"photos-{i}.js"
        (out / name).write_text("window.PH=window.PH||{t:{},c:{}};\n" + "".join(lines), encoding="utf-8")
        names.append(name)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", default=str(Path.home() / "data/chest_xray"))
    ap.add_argument("--csv", default=str(ROOT / "reports/equipment_v2/val_B512/shift_results.csv"))
    ap.add_argument("--val-pred", default=str(ROOT / "runs/results/b512-20261022-111844/val_predictions.csv"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.csv)
    (out / "val_conditions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "val_confusion.json").write_text(json.dumps(build_confusion(args.val_pred), ensure_ascii=False, indent=1), encoding="utf-8")
    thumbs, crops = build_photos(Path(args.images))
    names = write_js(out, thumbs, crops)
    print(f"행 {len(rows)}개, 사진 {len(thumbs)}장 + 확대 {len(crops)}장")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}: {p.stat().st_size / 1024:.0f} KB")
    print("JS 파일:", names)


if __name__ == "__main__":
    main()
