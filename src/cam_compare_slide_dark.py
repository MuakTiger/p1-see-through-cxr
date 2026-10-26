"""발표 슬라이드용 Grad-CAM 비교 그림 (검정 바탕 판). 위 줄 = 기준 모델(B512), 아래 줄 = 폐 영역 모델(l-s42).

같은 val 사진 3장(정상·세균성·바이러스성, 두 모델이 모두 맞힌 예시 — reports/equipment_v2/gradcam/B512/examples.csv 의 0·4·8번째).
모델 입력은 512×512 로 눌러 맞춘 모양이라, 그림은 원본 가로세로 비율로 되돌려 그린다 (그림용, 점수와 무관).
빨강 = 모델이 크게 반응한 곳, 흰 선 = 폐 마스크(PSPNet). 이전 판(흰 바탕)은 src/scratch_1022 가 아니라 세션 임시 폴더의 cam_compare_slide.py.

사용 예 (p1 폴더에서):
    .venv/bin/python src/cam_compare_slide_dark.py
"""
import csv
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))
import gradcam as G  # noqa: E402
import lung_prep as LP  # noqa: E402
import shift_eval as SE  # noqa: E402
import train as T  # noqa: E402
from shift_filters import to_gray  # noqa: E402

S = 512
G.SIZE = S
torch.set_num_threads(6)
BG = "#0A0A0A"
IMG = Path.home() / "data/chest_xray"

ex = list(csv.DictReader(open(ROOT / "reports/equipment_v2/gradcam/B512/examples.csv", encoding="utf-8")))
sel = [0, 4, 8]  # 정상 맞힘, 세균성 맞힘, 바이러스성 맞힘
raw = LP.load_masks(ROOT / "runs/lung_masks/val_lung_masks.npz", S)
cfb = G.GradCAM(SE.load_model(ROOT / "runs/results/b512-20261022-111844/model"))
cfl = G.GradCAM(SE.load_model(glob.glob(str(ROOT / "runs/results/l-s42-*"))[0] + "/model"))
tf = T.build_transform("P0", S)

aspects = []
for k in sel:
    with Image.open(IMG / ex[k]["image"]) as im:
        aspects.append(im.size[0] / im.size[1])

plt = G.setup_matplotlib()
fig, ax = plt.subplots(2, 3, figsize=(sum(aspects) * 3.3 + 0.9, 7.6), gridspec_kw={"width_ratios": aspects}, facecolor=BG)
names = ["정상", "세균성 폐렴", "바이러스성 폐렴"]
for j, k in enumerate(sel):
    rel = ex[k]["image"]
    m = raw[rel]
    h, _ = LP.hull_mask(m)
    xb = tf(Image.open(IMG / rel).convert("RGB")).unsqueeze(0)
    _, cb, _ = cfb(xb, [None])
    grayb = G.denorm_gray(xb)[0]
    gl = np.where(h, to_gray(Image.open(IMG / rel), S), 0).astype(np.uint8)
    xl = SE.gray_to_tensor(gl[None])
    _, cl, _ = cfl(xl, [None])
    for r, (gray, cam, vis) in enumerate([(grayb, cb[0][0], np.ones_like(h)), (gl, cl[0][0], h)]):
        a = ax[r, j]
        a.set_facecolor(BG)
        a.imshow(gray, cmap="gray", vmin=0, vmax=255, aspect=1.0 / aspects[j])
        a.imshow(np.ma.masked_where(~vis, cam), cmap="jet", alpha=0.45, vmin=0, vmax=1, aspect=1.0 / aspects[j])
        a.contour(m.astype(float), levels=[0.5], colors="white", linewidths=1.8)
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_edgecolor("#363A3F")
            sp.set_linewidth(1.2)
    ax[0, j].set_title(names[j], fontsize=19, color="white", pad=10)
fig.text(0.012, 0.72, "기준 모델", rotation=90, fontsize=20, color="#FFC285", va="center")
fig.text(0.012, 0.28, "폐 영역 모델", rotation=90, fontsize=20, color="#FFC285", va="center")
fig.subplots_adjust(left=0.05, right=0.995, bottom=0.01, top=0.92, hspace=0.04, wspace=0.03)
out = ROOT / "reports/equipment_v2/lung_model/gradcam_compare_dark.png"
fig.savefig(out, dpi=110, facecolor=BG)
print("저장", out, out.stat().st_size // 1024, "KB", [round(a, 2) for a in aspects])
