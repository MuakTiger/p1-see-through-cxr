"""분할 균형 + D 학습 seed 반복 비교 보고서 생성 (val 만 사용, test 는 열지 않는다).

보고서에 들어가는 숫자는 전부 파일에서 직접 읽어 계산한다 (손으로 옮겨 적지 않는다).
    1. train/val 분할 균형: 클래스별 장수·그룹 수·val 비율, split 안 클래스 비중, split × 클래스별 파일 속성
    2. D 를 seed 42/43/44 로 학습한 결과: val 점수, 장비 차이 시험 요약 (B seed 42 와 비교)

사용 예 (p1 폴더에서):
    .venv/bin/python src/split_seed_report.py --manifest manifests/split_manifest.csv \
        --images ~/data/chest_xray --out reports/seed_check_and_split_balance.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, recall_score

from normal_shift_analysis import estimate_jpeg_quality

ROOT = Path(__file__).resolve().parents[1]
CLASSES = [(0, "정상"), (1, "세균성"), (2, "바이러스성")]
HIGH_Q = 90  # 이 값 이상이면 "고품질 저장" (train/val 정상이 이쪽, 폐렴과 test 정상은 약 75)

# 모델 이름 → (학습 결과 폴더, 장비 차이 시험 폴더)
RUNS = {
    "B seed 42": ("runs/results/b-20261019-103853", "reports/shift_test"),
    "D seed 42 (최종)": ("runs/results/d-20261019-134621", "reports/shift_test_D"),
    "D seed 43": ("runs/results/d-s43-20261021-143018", "reports/shift_test_D_s43"),
    "D seed 44": ("runs/results/d-s44-20261021-143019", "reports/shift_test_D_s44"),
}


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return lines


# ----------------------------------------------------------------------------
# 1. 분할 균형
# ----------------------------------------------------------------------------
def class_table(pool):
    rows = []
    for lab, name in CLASSES:
        s = pool[pool.label_id == lab]
        tr, va = s[s.split == "train"], s[s.split == "val"]
        per_tr, per_va = tr.groupby("split_group").size(), va.groupby("split_group").size()
        rows.append([name, f"{len(tr):,} / {len(va):,}", f"{100 * len(va) / len(s):.1f}%",
                     f"{per_tr.size:,} / {per_va.size:,}", f"{per_tr.mean():.2f} / {per_va.mean():.2f}",
                     f"{per_tr.max()} / {per_va.max()}"])
    tr_all, va_all = pool[pool.split == "train"], pool[pool.split == "val"]
    rows.append(["전체", f"{len(tr_all):,} / {len(va_all):,}", f"{100 * len(va_all) / len(pool):.1f}%",
                 f"{tr_all.split_group.nunique():,} / {va_all.split_group.nunique():,}", "", ""])
    return md_table(["클래스", "사진 수 (train / val)", "사진 중 val 비율", "그룹 수 (train / val)",
                     "그룹당 평균 사진 수", "한 그룹 최대 사진 수"], rows)


def share_table(pool):
    rows = []
    for sp in ["train", "val"]:
        c = pool[pool.split == sp].label_id.value_counts().sort_index()
        rows.append([sp, f"{100 * c[0] / c.sum():.1f}%", f"{100 * c[1] / c.sum():.1f}%",
                     f"{100 * c[2] / c.sum():.1f}%", f"{c[1] / c[2]:.2f}배"])
    return md_table(["split", "정상", "세균성", "바이러스성", "세균성 ÷ 바이러스성"], rows)


def mixed_viral(pool):
    labels = pool.groupby("split_group").label_id.agg(set)
    out = []
    for sp in ["train", "val"]:
        sgs = pool[(pool.split == sp) & (pool.label_id == 2)].split_group.unique()
        mixed = sum(1 for g in sgs if 1 in labels[g])
        out.append(f"{sp} {len(sgs)}개 중 {mixed}개 ({100 * mixed / len(sgs):.0f}%)")
    return ", ".join(out)


def file_attributes(m, images_root):
    """헤더만 읽는다 (픽셀 디코딩 없음). 저장 품질은 양자화표로 거꾸로 추정한 근사값."""
    q = []
    for rel in m.relative_path:
        with Image.open(images_root / rel) as im:
            q.append(estimate_jpeg_quality(im))
    m = m.assign(q=q, aspect=m.width / m.height, mpix=m.width * m.height / 1e6)
    m["high_q"] = (m.q >= HIGH_Q).astype(int)
    rows = []
    for sp in ["train", "val", "test"]:
        for lab, name in CLASSES:
            s = m[(m.split == sp) & (m.label_id == lab)]
            rows.append([sp, name, f"{len(s):,}", f"{s.q.median():.1f}", f"{100 * s.high_q.mean():.1f}%",
                         f"{s.aspect.median():.3f}", f"{s.mpix.median():.2f}"])
    return md_table(["split", "클래스", "사진 수", "저장 품질 추정(중앙값)", f"품질 {HIGH_Q} 이상 비율",
                     "가로/세로(중앙값)", "픽셀 수(백만, 중앙값)"], rows)


# ----------------------------------------------------------------------------
# 2. seed 반복
# ----------------------------------------------------------------------------
def run_row(name, results_dir, shift_dir):
    s = json.load(open(ROOT / results_dir / "experiment_summary.json", encoding="utf-8"))
    p = pd.read_csv(ROOT / results_dir / "val_predictions.csv")
    f1 = f1_score(p.true_label, p.pred_label, average="macro")
    if abs(f1 - s["best_val_macro_f1"]) > 1e-6:
        raise SystemExit(f"{name}: val_predictions 로 다시 계산한 F1 {f1:.6f} 이 기록 {s['best_val_macro_f1']:.6f} 과 다릅니다.")
    viral = recall_score(p.true_label, p.pred_label, average=None, labels=[0, 1, 2])[2]
    d = pd.read_csv(ROOT / shift_dir / "shift_results.csv")
    cond = d[d["shift"] != "clean"]
    noise = d[d["shift"] == "noise"].set_index("level")
    return {"name": name, "epoch": s["selected_epoch"], "f1": f1, "viral": viral,
            "all24": cond.macro_f1.mean(), "non_noise": cond[cond["shift"] != "noise"].macro_f1.mean(),
            "noise3": noise.macro_f1.mean(), "noise_std16": noise.loc[3, "macro_f1"],
            "noise_viral_share": noise.loc[3, "pred_share_viral"], "split_sha": s["split_manifest_sha256"]}


def seed_tables(rows):
    val = md_table(["모델", "val Macro-F1", "바이러스성 recall", "선택 에폭"],
                   [[r["name"], f"{r['f1']:.3f}", f"{r['viral']:.3f}", r["epoch"]] for r in rows])
    shift = md_table(["모델", "24조건 평균", "노이즈 제외 21조건 평균", "노이즈 3조건 평균", "노이즈 std16 F1",
                      "std16 바이러스성 예측 비율"],
                     [[r["name"], f"{r['all24']:.3f}", f"{r['non_noise']:.3f}", f"{r['noise3']:.3f}",
                       f"{r['noise_std16']:.3f}", f"{100 * r['noise_viral_share']:.1f}%"] for r in rows])
    return val, shift


def build_report(m, pool, images_root):
    rows = [run_row(n, *paths) for n, paths in RUNS.items()]
    if len({r["split_sha"] for r in rows}) != 1:
        raise SystemExit("모델마다 쓴 split 파일이 다릅니다.")
    d_rows = [r for r in rows if r["name"].startswith("D")]
    b = rows[0]
    f1s = [r["f1"] for r in d_rows]
    val_t, shift_t = seed_tables(rows)

    L = ["# 분할 균형과 D 의 seed 반복 확인", "",
         "val 만 사용했고 test 는 열지 않았다. 숫자는 `src/split_seed_report.py` 가 파일에서 직접 계산했다.", "",
         "## 1. train/val 분할은 균형이 맞는가", "", "### 1-1. 클래스별 비교", ""]
    L += class_table(pool)
    L += ["", "- 같은 환자 그룹(`split_group`)은 한쪽에만 있다. 클래스별 그룹 수를 더하면 전체 그룹 수와 맞지 않는데, "
          "한 번호에 세균성과 바이러스성이 같이 있으면 양쪽에 한 번씩 세어지기 때문이다.",
          f"- 바이러스성이 들어 있는 그룹 중 세균성도 섞인 그룹: {mixed_viral(pool)}.",
          "- 그룹 단위로 나누므로 val 비율을 정확히 20% 로 맞출 수 없다 (바이러스성은 목표 20% 보다 약간 높음).", "",
          "### 1-2. split 안에서 클래스가 차지하는 비중", ""]
    L += share_table(pool)
    L += ["", f"### 1-3. 파일 속성 (헤더만 읽은 값, 저장 품질은 근사 추정, 품질 {HIGH_Q} 이상을 '고품질 저장'으로 셈)", ""]
    L += file_attributes(m, images_root)
    L += ["", "- train 과 val 은 모든 클래스에서 거의 같다. 즉 val 은 train 과 같은 분포의 사진으로 이루어져 있다.",
          "- test 정상은 train/val 정상과 저장 방식·모양이 다르다. 풀(train+val) 안에는 test 정상 같은 사진이 없어서 "
          "풀 안을 다시 나눠도 이 차이는 줄지 않는다. val 점수가 test 점수를 예측하지 못하는 이유 중 하나로 보이지만, "
          "원인으로 확정한 것은 아니다 (저장 화질은 D 의 오분류 원인이 아님을 앞서 확인).",
          "- 따라서 분할 자체에는 문제가 없고, 분할을 더 손볼 여지도 가이드 안에서는 크지 않다.", "",
          "## 2. D 를 seed 3개로 학습한 결과", "",
          "설정은 모두 같다 (layer4 미세조정, 노이즈 증강 p=0.5·std 0~10, 30에폭, 배치 16). 바꾼 것은 학습 seed 뿐이고 "
          "분할 파일(해시 동일)은 seed 42 로 만든 하나를 그대로 썼다.", "", "### 2-1. 원본 val 점수", ""]
    L += val_t
    L += ["", f"- D 의 val Macro-F1 은 seed 에 따라 {min(f1s):.3f}~{max(f1s):.3f} (평균 {np.mean(f1s):.3f}). "
          f"B seed 42 의 {b['f1']:.3f} 이 이 범위 안에 있다. 원본 val 에서 D 가 B 보다 낫다고 말할 근거는 없다. "
          f"(B 는 seed 가 하나뿐이라 B 자신의 흔들림은 모른다.)",
          f"- seed 42 의 {d_rows[0]['f1']:.3f} 은 세 개 중 가장 높다. 선택 에폭도 {', '.join(str(r['epoch']) for r in d_rows)} 로 크게 흔들린다.",
          "", "### 2-2. 장비 차이 시험 (val 1,030장, 필터 8종 × 3단계)", ""]
    L += shift_t
    L += ["", f"- 노이즈 약점 보완은 seed 3개 모두에서 재현된다. 노이즈 std16 에서 B 는 {b['noise_std16']:.3f} 까지 무너졌고 "
          f"바이러스성 예측 비율이 {100 * b['noise_viral_share']:.1f}% 로 쏠렸지만, D 는 "
          f"{min(r['noise_std16'] for r in d_rows):.3f}~{max(r['noise_std16'] for r in d_rows):.3f}, "
          f"쏠림 {100 * min(r['noise_viral_share'] for r in d_rows):.1f}~{100 * max(r['noise_viral_share'] for r in d_rows):.1f}% 였다.",
          f"- 24조건 평균의 향상은 대부분 노이즈에서 나온다. 노이즈를 뺀 21조건 평균은 B {b['non_noise']:.3f}, "
          f"D {min(r['non_noise'] for r in d_rows):.3f}~{max(r['non_noise'] for r in d_rows):.3f} 로 차이가 작고 D 끼리의 차이와 비슷하다.",
          "", "## 3. 한계", "",
          "- 학습 seed 만 바꿨고 분할은 하나다. 다른 분할에서의 흔들림은 보지 않았다.",
          "- B 는 seed 가 하나뿐이다. D 가 B 보다 낫다는 비교(특히 원본 점수)는 오차 범위를 말할 수 없다.",
          "- 장비 차이는 필터로 흉내 낸 것이다. 노이즈 증강과 노이즈 시험이 같은 종류(가우시안)라서, 이 개선은 "
          "'배운 유형으로 시험한 결과'다. 흐림 등 다른 변화에 대한 강인함을 뜻하지 않는다.",
          "- seed 43, 44 는 최종 모델이 아니다. 최종 모델은 seed 42 의 D 로 유지했고 test 는 다시 열지 않았다.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="split_manifest.csv 경로")
    ap.add_argument("--images", required=True, help="이미지 루트 (예: ~/data/chest_xray)")
    ap.add_argument("--out", required=True, help="보고서(.md) 저장 경로")
    args = ap.parse_args()

    m = pd.read_csv(Path(args.manifest).expanduser())
    pool = m[m.split.isin(["train", "val"])]
    report = build_report(m, pool, Path(args.images).expanduser())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
