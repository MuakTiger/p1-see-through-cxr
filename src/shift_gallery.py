"""장비 차이 필터를 적용한 사진을 구경하는 갤러리를 만든다 (iMac 전용, 계산 없음).

train split 사진만 쓴다 (val·check·test는 열지 않는다). 클래스마다 1장씩 seed로 고정해 고른다.

사용 예:
    python src/shift_gallery.py --out reports/shift_gallery

결과 (모두 PNG):
    00_overview.png   필터마다 가장 강한 단계를 한눈에 (사진 1장 기준)
    <필터이름>.png    필터 하나: 행 = 정상·세균성·바이러스성, 열 = 원본·약·중·강
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from shift_filters import SHIFTS, apply_shift, to_gray_224  # noqa: E402

CLASS_NAMES = ["NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA"]
CLASS_KO = {"NORMAL": "정상", "BACTERIAL_PNEUMONIA": "세균성", "VIRAL_PNEUMONIA": "바이러스성"}
LEVEL_KO = ["원본", "약", "중", "강"]
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
SCALE = 2  # 224 픽셀 타일을 화면에서 보기 좋게 2배로 키운다 (모델이 보는 건 224 그대로)
TILE = 224 * SCALE
PAD = 12


def font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def pick_train_images(manifest_path, seed):
    """train split에서 클래스마다 1장씩 고른다."""
    rows = [r for r in csv.DictReader(open(manifest_path, encoding="utf-8")) if r["split"] == "train"]
    rng = random.Random(seed)
    picked = []
    for name in CLASS_NAMES:
        cands = sorted(r["relative_path"] for r in rows if r["label_name"] == name)
        picked.append((name, rng.choice(cands)))
    return picked


def tile(arr):
    return Image.fromarray(arr).convert("RGB").resize((TILE, TILE), Image.BICUBIC)


def level_label(name, level):
    if level == 0:
        return "원본"
    return f"{LEVEL_KO[level]} ({SHIFTS[name]['param'].split(' ')[0]}={SHIFTS[name]['levels'][level - 1]})"


def draw_grid(title, subtitle, row_labels, col_labels, images):
    """images[r][c] = numpy 영상. 글자와 함께 격자 그림을 만든다."""
    rows, cols = len(images), len(images[0])
    left, top = 250, 120  # 왼쪽 여백: 파일명이 잘리지 않게
    canvas = Image.new("RGB", (left + cols * (TILE + PAD) + PAD, top + rows * (TILE + PAD) + PAD), "white")
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 14), title, fill="black", font=font(34))
    d.text((PAD, 60), subtitle, fill="#555555", font=font(20))
    for c, label in enumerate(col_labels):
        d.text((left + c * (TILE + PAD), top - 32), label, fill="black", font=font(22))
    for r, label in enumerate(row_labels):
        y = top + r * (TILE + PAD)
        for i, line in enumerate(label.split("\n")):
            d.text((PAD, y + TILE // 2 - 30 + i * 28), line, fill="black", font=font(22 if i == 0 else 15))
        for c in range(cols):
            canvas.paste(tile(images[r][c]), (left + c * (TILE + PAD), y))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/split_manifest.csv")
    ap.add_argument("--images-dir", default=str(Path.home() / "data/chest_xray"))
    ap.add_argument("--out", default="reports/shift_gallery")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    picked = pick_train_images(args.manifest, args.seed)
    grays = [to_gray_224(Image.open(Path(args.images_dir) / rel)) for _, rel in picked]
    row_labels = [f"{CLASS_KO[n]}\n{Path(rel).name}" for n, rel in picked]

    # 필터마다 한 장씩: 행 = 클래스, 열 = 원본·약·중·강
    for name, spec in SHIFTS.items():
        imgs = [[apply_shift(g, name, lv, seed=args.seed + i) for lv in range(4)] for i, g in enumerate(grays)]
        cols = [level_label(name, lv) for lv in range(4)]
        draw_grid(spec["title"], f"흉내 내는 상황: {spec['situation']}", row_labels, cols, imgs).save(out / f"{name}.png")
        print("저장:", out / f"{name}.png", flush=True)

    # 한눈에 보기: 사진마다 원본 + 필터별 가장 강한 단계
    names = list(SHIFTS)
    imgs = [[g] + [apply_shift(g, n, 3, seed=args.seed + i) for n in names] for i, g in enumerate(grays)]
    cols = ["원본"] + [SHIFTS[n]["title"].split(" (")[0] + " 강" for n in names]
    draw_grid("장비 차이 필터: 가장 강한 단계 한눈에 보기", "모든 필터는 224×224 회색조에서 적용 (그림은 보기 좋게 2배 확대)",
              row_labels, cols, imgs).save(out / "00_overview.png")
    print("저장:", out / "00_overview.png", flush=True)

    # 확인: 같은 입력 → 같은 출력, 크기·형식
    for name in names:
        for lv in range(4):
            a = apply_shift(grays[0], name, lv, seed=1)
            b = apply_shift(grays[0], name, lv, seed=1)
            assert np.array_equal(a, b) and a.shape == (224, 224) and a.dtype == np.uint8, (name, lv)
    print("확인: 모든 필터·단계가 결정적이고 224×224 uint8을 돌려줌", flush=True)


if __name__ == "__main__":
    main()
