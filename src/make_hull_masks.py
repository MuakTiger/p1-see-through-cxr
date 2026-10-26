"""폐 영역 학습(train_lung.py)용 껍질 마스크 묶음을 만든다: 원본 폐 마스크(xrv_mask_split.py)의 볼록 껍질 + 여백(lung_prep.hull_mask).

분할 실패(면적 5% 미만)인 사진은 껍질이 사진 전체(모두 참)다. 비트로 묶어(npz) 저장한다.

사용 예 (p1 폴더에서):
    .venv/bin/python src/make_hull_masks.py            # runs/lung_masks/lung_hull_masks.npz (train + val)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import lung_prep as LP  # noqa: E402

MASKS = ROOT / "runs/lung_masks"


def main(splits=("train", "val")):
    paths, packed, fails = [], [], 0
    for sp in splits:
        for rel, m in LP.load_masks(MASKS / f"{sp}_lung_masks.npz", 512).items():
            hull, ok = LP.hull_mask(m)
            fails += (not ok)
            paths.append(rel)
            packed.append(np.packbits(hull.reshape(-1)))
    out = MASKS / "lung_hull_masks.npz"
    np.savez_compressed(out, masks=np.stack(packed), paths=np.array(paths))
    print(f"저장: {out.relative_to(ROOT)} ({len(paths)}장, 분할 실패 {fails}장, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
