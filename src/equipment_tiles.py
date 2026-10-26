"""발표 슬라이드용 타일: 같은 정상 흉부 X선 한 장에 장치 설정 필터 8종을 가장 센 단계로 걸어 글자 없는 512×512 PNG 9장(원본 + 8종)으로 저장한다.

필터와 강도는 reports/equipment_v2/strength_fixed_512.md 와 같다 (모델·점수와 무관, 가벼운 CPU 계산만):
    dose    선량 1/64            (training/equipment_direct.py)
    motion  조사시간 번짐 8mm    (training/equipment_direct.py)
    scatter 산란 k = 1.0         (training/equipment_direct.py)
    kvp     관전압 (0.60, 0.5)   (training/equipment_direct.py)
    sensor  센서 화소 2.0mm      (training/equipment_direct.py)
    sharpen 경계 강조 α 2.0      (training/shift_filters.py, 3단계)
    equalize 대비 보정 HE        (training/shift_filters.py, 3단계)
    jpeg    JPEG 품질 20         (training/shift_filters.py, 3단계)
사진은 train 폴더의 정상 사진 한 장이다 (val·test 사진은 열지 않는다). 노이즈 seed 는 42.

사용 예 (p1 폴더에서):
    .venv/bin/python src/equipment_tiles.py
"""
import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from equipment_direct import EQUIP, apply_equipment  # noqa: E402
from shift_filters import SHIFTS, apply_shift, to_gray  # noqa: E402

SIZE = 512
NEW = ["dose", "motion", "scatter", "kvp", "sensor"]
REUSED = ["sharpen", "equalize", "jpeg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(Path.home() / "data/chest_xray"))
    ap.add_argument("--image", default="train/NORMAL/IM-0487-0001.jpeg", help="이미지 루트 기준 상대 경로 (train 사진만)")
    ap.add_argument("--out", default=str(ROOT / "reports/equipment_v2/tiles_512"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.image.startswith("train/"):
        raise SystemExit("슬라이드 타일은 train 폴더의 사진만 쓴다.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with Image.open(Path(args.images).expanduser() / args.image) as im:
        g = to_gray(im, SIZE)
    tiles = {"original": g}
    for name in NEW:
        tiles[name] = apply_equipment(g, name, 3, seed=args.seed)
    for name in REUSED:
        tiles[name] = apply_shift(g, name, 3, seed=args.seed)
    for name, arr in tiles.items():
        Image.fromarray(arr).save(out / f"{name}.png")
        label = "원본" if name == "original" else (EQUIP[name]["labels"][2] if name in EQUIP else SHIFTS[name]["levels"][2])
        print(f"저장: {out / (name + '.png')}  ({'원본' if name == 'original' else name} 강 = {label})")


if __name__ == "__main__":
    main()
