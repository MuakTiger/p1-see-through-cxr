"""폐 상자의 구도(폐가 사진 세로를 얼마나 채우나)가 분할·클래스마다 다른지, test 정상 recall 과 연관이 있는지 본다.

해석용 분석이다. 모델·필터·강도를 고르는 데 쓰지 않는다. 폐 마스크는 PSPNet(runs/lung_masks, 사진 512 기준)이고
면적 5% 미만인 분할 실패 사진은 뺀다. test 는 같은 사진 사본 6장을 뺀 618장이다.
모델은 B512 seed 42 (재학습 없음). 폐 상자로 사진을 잘라 다시 학습하는 실험은 하지 않았다 (연관만 본다).

사용 예 (p1 폴더에서):
    .venv/bin/python src/lung_box_composition.py
결과: reports/equipment_v2/lung_box_composition.md
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))
import lung_prep as LP  # noqa: E402
import train as T  # noqa: E402
from shift_eval import load_model  # noqa: E402

SIZE = 512
IMAGES = Path.home() / "data/chest_xray"


def box_stats(m):
    rows = np.where(m.any(1))[0]
    cols = np.where(m.any(0))[0]
    return (rows.max() - rows.min() + 1) / SIZE, (cols.max() - cols.min() + 1) / SIZE, float(m.mean())


def main():
    rows = list(csv.DictReader(open(ROOT / "manifests/split_manifest.csv", encoding="utf-8")))
    meta = {r["relative_path"]: (int(r["label_id"]), r["split"]) for r in rows}
    dup = set(re.findall(r"'(test/[^']+)'", open(ROOT / "reports/equipment_v2/test_B512/shift_report.md", encoding="utf-8").read()))
    assert len(dup) == 6, dup

    stat = {}   # relative_path -> (세로 점유, 가로 점유, 면적)
    n_fail = {}
    for sp in ("train", "val", "test"):
        masks = LP.load_masks(ROOT / f"runs/lung_masks/{sp}_lung_masks.npz", SIZE)
        n_fail[sp] = 0
        for rel, m in masks.items():
            if rel in dup:
                continue
            if m.mean() < LP.MIN_AREA:
                n_fail[sp] += 1
                continue
            stat[rel] = box_stats(m)

    def group(name, pred):
        v = np.array([stat[p] for p in stat if pred(*meta[p])])
        return name, len(v), np.median(v[:, 0]), np.median(v[:, 1]), np.median(v[:, 2])

    groups = [
        group("train·val 정상", lambda l, s: s in ("train", "val") and l == 0),
        group("train·val 폐렴", lambda l, s: s in ("train", "val") and l != 0),
        group("test 정상", lambda l, s: s == "test" and l == 0),
        group("test 폐렴", lambda l, s: s == "test" and l != 0),
    ]

    # B512 seed 42 가 test 정상 사진을 얼마나 맞히는지 (폐 상자 세로 점유와 비교)
    torch.set_num_threads(4)
    model = load_model(str(ROOT / "runs/results/b512-20261022-111844/model"))
    tf = T.build_transform("P0", SIZE)
    normals = [p for p in stat if meta[p] == (0, "test")]
    pn = []
    for i in range(0, len(normals), 16):
        batch = torch.stack([tf(Image.open(IMAGES / p).convert("RGB")) for p in normals[i:i + 16]])
        with torch.no_grad():
            pr = torch.softmax(model(batch), 1)
        pn += [(float(x[0]), int(x.argmax()) == 0) for x in pr]
    p_normal = np.array([a for a, _ in pn])
    correct = np.array([b for _, b in pn])
    vocc = np.array([stat[p][0] for p in normals])
    hocc = np.array([stat[p][1] for p in normals])
    area = np.array([stat[p][2] for p in normals])
    order = np.argsort(vocc)
    terc = np.array_split(order, 3)

    L = ["# 폐 상자의 구도와 test 정상 recall (해석용, 모델·필터·강도 선택에 쓰지 않음)", "",
         "생성: `src/lung_box_composition.py`. 폐 마스크는 PSPNet(`runs/lung_masks`, 입력 512 기준), 면적 5% 미만인 분할 실패는 뺐다 "
         f"(train {n_fail['train']}장, val {n_fail['val']}장, test {n_fail['test']}장). test 는 같은 사진 사본 6장을 뺀 618장. "
         "모델은 B512 seed 42, 재학습 없음, test 는 정상 사진 예측에만 썼다.", "",
         "## 1. 폐 상자가 사진에서 차지하는 비율 (중앙값)", "",
         "| 묶음 | 장수 | 세로 점유 | 가로 점유 | 폐 면적 비율 |", "|---|---|---|---|---|"]
    for name, n, v, h, a in groups:
        L.append(f"| {name} | {n} | {v:.3f} | {h:.3f} | {a:.3f} |")
    L += ["", "세로 점유 = 폐 마스크의 위아래 끝 사이 길이 ÷ 사진 세로(512). 가로 점유도 같은 방식이다. 사진은 모델 입력과 같이 가로세로를 정사각형으로 눌렀다.", "",
          "## 2. test 정상 사진의 recall 을 폐 상자 세로 점유 3분위로 나눠 본 값 (B512 seed 42)", "",
          "| 세로 점유 | 장수 | 점유 중앙값 | 정상으로 맞힌 비율 |", "|---|---|---|---|"]
    for name, idx in zip(("낮음", "중간", "높음"), terc):
        L.append(f"| {name} | {len(idx)} | {np.median(vocc[idx]):.3f} | {correct[idx].mean():.3f} |")
    r_v = spearmanr(vocc, p_normal)
    r_h = spearmanr(hocc, p_normal)
    r_a = spearmanr(area, p_normal)
    L += ["", f"- 스피어만 상관 (정상 확률 p_normal 과): 세로 점유 {r_v.statistic:+.2f} (p={r_v.pvalue:.1e}), 가로 점유 {r_h.statistic:+.2f} (p={r_h.pvalue:.1e}), 폐 면적 {r_a.statistic:+.2f} (p={r_a.pvalue:.1e}).",
          f"- test 정상 {len(normals)}장 중 정상으로 맞힌 사진 {int(correct.sum())}장 ({correct.mean():.3f}).", "",
          "## 읽는 법과 한계", "",
          "- 폐가 사진 세로를 꽉 채운 정상 사진일수록 모델이 폐렴으로 판정하는 경향이 있다. 이것은 **연관**이며 원인이 아니다. 같은 사진은 다른 특성(저장 방식, 촬영 시기 등)도 함께 다르다.",
          "- train·val 정상은 폐 위아래에 여백이 많고, 폐렴과 test 정상은 폐가 세로를 더 채운다. 강사 의견(폐 영역 상자로 잘라서 구도를 맞추면 달라질 수 있다)과 방향이 맞는 정황이지만, **폐 상자 크롭으로 다시 학습하는 실험은 아직 하지 않았다.**",
          "- 폐 마스크는 사진 12장만 눈으로 확인했고, 폐렴으로 뿌옇게 된 폐는 놓칠 수 있다 (그러면 폐렴 쪽 점유가 작게 나온다).",
          "- 모델 1개(seed 1개), 정상 사진 한 묶음(test 정상)이다."]
    out = ROOT / "reports/equipment_v2/lung_box_composition.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
