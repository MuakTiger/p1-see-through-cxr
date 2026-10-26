"""발표 슬라이드용 X선 사진 9장(원본 + 필터 8종, 가장 센 단계)을 다시 만든다. src/equipment_tiles.py 의 개선판.

바뀐 점 (모델·점수와 무관한 그림용):
  - 사진: train 정상 한 장(NORMAL2-IM-0537-0001, 폐·심장·횡격막이 한눈에 보이고 글자 표식이 잘리지 않는 사진).
  - 모양: 모델 입력처럼 찌그러뜨리지 않고, 원본 가운데를 정사각형으로 잘라 512×512로 줄인다(LANCZOS).
    모델 입력(전체를 512×512로 눌러 맞춤)과는 모양이 다르지만 필터 코드와 강도는 같다.
필터와 강도는 reports/equipment_v2/strength_fixed_512.md 와 같다 (선량 1/64, 노출시간 8mm, 산란 k=1.0,
관전압 (0.60, 0.5), 센서 2.0mm, 경계 강조 α 2.0, 대비 보정 HE, JPEG 품질 20). 노이즈 seed 42.
val·test 사진은 열지 않는다 (train 사진만).

사용 예 (p1 폴더에서):
    .venv/bin/python src/slide_xray_tiles.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from equipment_direct import EQUIP, apply_equipment  # noqa: E402
from shift_filters import SHIFTS, apply_shift  # noqa: E402

SIZE = 512
NEW = ["dose", "motion", "scatter", "kvp", "sensor"]
REUSED = ["sharpen", "equalize", "jpeg"]


def square_gray(path, size=SIZE):
    """원본 가운데를 정사각형으로 잘라 회색조 size×size 로 줄인다 (찌그러뜨리지 않음)."""
    with Image.open(path) as im:
        g = im.convert("L")
        w, h = g.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        g = g.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)
    return np.asarray(g, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(Path.home() / "data/chest_xray"))
    ap.add_argument("--image", default="train/NORMAL/NORMAL2-IM-0537-0001.jpeg", help="이미지 루트 기준 상대 경로 (train 사진만)")
    ap.add_argument("--out", default=str(ROOT / "reports/equipment_v2/tiles_slide"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.image.startswith("train/"):
        raise SystemExit("슬라이드 사진은 train 폴더의 사진만 쓴다.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = square_gray(Path(args.images).expanduser() / args.image)
    tiles = {"original": g}
    for name in NEW:
        tiles[name] = apply_equipment(g, name, 3, seed=args.seed)
    for name in REUSED:
        tiles[name] = apply_shift(g, name, 3, seed=args.seed)
    for name, arr in tiles.items():
        Image.fromarray(arr).save(out / f"{name}.png")
        label = "원본" if name == "original" else (EQUIP[name]["labels"][2] if name in EQUIP else SHIFTS[name]["levels"][2])
        print(f"저장: {out / (name + '.png')}  (강 = {label})")


if __name__ == "__main__":
    main()
