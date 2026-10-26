"""train_lung.py 의 입력 처리가 (1) 폐 영역 PNG(학습용 l 모델 입력)와 (2) 평가(필터 → 고정 마스크)와 같은지 확인한다."""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("~/Documents/p1")
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))
import train_lung as TL  # noqa: E402
import lung_prep as LP  # noqa: E402
import equipment_direct as ED  # noqa: E402
import train as T  # noqa: E402
from shift_filters import to_gray, apply_shift  # noqa: E402

IMG = Path.home() / "data/chest_xray"
rows = [r for r in csv.DictReader(open(ROOT / "manifests/split_manifest.csv", encoding="utf-8")) if r["split"] == "val"]
rng = np.random.default_rng(0)
rows = [rows[i] for i in rng.permutation(len(rows))[:8]]
hulls = TL.load_hulls(ROOT / "runs/lung_masks/lung_hull_masks.npz", 512)
raw = LP.load_masks(ROOT / "runs/lung_masks/val_lung_masks.npz", 512)

# (1) 평가용 PNG(학습 l 모델의 입력)와 같은 텐서인가
ds = TL.LungDataset(rows, IMG, hulls, None, 512)
base = T.build_transform("P0", 512)
d1 = max(float((ds[i][0] - base(Image.open(ROOT / "runs/lung512/images" / Path(r["relative_path"]).with_suffix(".png")).convert("RGB"))).abs().max()) for i, r in enumerate(rows))
print("(1) LungDataset(val) 대 폐 영역 PNG 텐서 최대 차이:", d1)

# (2) 학습 증강(필터 → 폐 처리)이 평가 경로(필터 → 고정 마스크)와 같은가


class Fixed:
    def __init__(self, kind, u, seed):
        self.kind, self.u, self.seed, self.re = kind, u, seed, TL.RandomEquipment(1.0, 512)

    def __call__(self, img):
        return self.re.apply(img, self.kind, self.u, self.seed)


for kind in ("dose", "kvp", "scatter", "motion", "sensor"):
    d2 = 0.0
    for i, r in enumerate(rows):
        t_train = TL.LungDataset([r], IMG, hulls, Fixed(kind, 1.0, 42 + i), 512)[0][0]
        g = to_gray(Image.open(IMG / r["relative_path"]), 512)
        ev = LP.apply_lung(ED.apply_equipment(g, kind, 3, seed=42 + i), raw[r["relative_path"]])[0]
        t_eval = base(Image.fromarray(ev).convert("RGB"))
        d2 = max(d2, float((t_train - t_eval).abs().max()))
    print(f"(2) 필터 {kind:8s} 학습 입력 대 평가 입력 최대 차이: {d2}")
