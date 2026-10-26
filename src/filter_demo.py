"""장비 차이 필터를 실제 흉부 X선 한 장에 걸어 크게 보여 주는 그림 (발표·설명용, 모델과 무관).

그림 1 demo_grid.png  원본 + 필터 8종(가장 센 단계)을 한 화면에 (3×3)
그림 2 demo_zoom.png  흐림·경계 강조·노이즈·JPEG 는 전체 사진에서 구분하기 어려워서 64×64 영역을 확대해 원본·약·중·강을 나란히

필터와 강도는 training/shift_filters.py 의 SHIFTS 그대로다 (시험 전에 고정한 값). 224×224 회색조에서 적용한다.

사용 예 (p1 폴더에서):
    .venv/bin/python src/filter_demo.py --images ~/data/chest_xray --image train/NORMAL/IM-0487-0001.jpeg --out reports/shift_gallery
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))
from shift_filters import SHIFTS, apply_shift, to_gray_224  # noqa: E402
from gradcam import setup_matplotlib  # noqa: E402

# 3×3 배치 순서 (원본 포함 9칸)
GRID = ["original", "blur", "sharpen", "noise", "bright", "dark", "contrast", "equalize", "jpeg"]
ZOOM_FILTERS = ["blur", "sharpen", "noise", "jpeg"]
CROP = (slice(70, 134), slice(30, 94))  # 확대할 64×64 영역 (왼쪽 폐 윗부분, 갈비뼈와 폐 무늬가 보이는 곳)


def level_text(name, lv):
    v = SHIFTS[name]["levels"][lv - 1]
    return f"{v}"


def fig_grid(plt, g, out):
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 14.4))
    for ax, key in zip(axes.ravel(), GRID):
        if key == "original":
            img, title, sub = g, "원본", "필터를 걸기 전의 사진"
        else:
            img = apply_shift(g, key, 3, seed=42)
            spec = SHIFTS[key]
            title = f"{spec['title']} · 강 ({level_text(key, 3)})"
            sub = spec["situation"]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=13, pad=22)
        ax.text(0.5, 1.015, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5, color="#555555")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("같은 폐 사진에 장비 차이 필터를 가장 센 단계로 걸었을 때 (224×224)", fontsize=15)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.915, bottom=0.015, hspace=0.24, wspace=0.05)  # 제목이 위 사진과 겹치지 않게 줄 간격을 넓힘
    fig.savefig(out / "demo_grid.png", dpi=130)
    plt.close(fig)


def fig_zoom(plt, g, out):
    cols = ["원본", "약", "중", "강"]
    fig, axes = plt.subplots(len(ZOOM_FILTERS), 5, figsize=(17, 13), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.5]})
    for r, key in enumerate(ZOOM_FILTERS):
        spec = SHIFTS[key]
        for c in range(4):
            img = apply_shift(g, key, c, seed=42)
            ax = axes[r, c]
            ax.imshow(img[CROP], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=13)
            lab = "" if c == 0 else f"{spec['levels'][c - 1]}"
            ax.set_xlabel(lab, fontsize=11)
        ax = axes[r, 4]
        ax.axis("off")
        ax.text(0, 0.62, spec["title"], fontsize=14, fontweight="bold", va="center")
        ax.text(0, 0.38, f"흉내 내는 상황\n{spec['situation']}\n\n강도 ({spec['param']})\n" + " / ".join(str(v) for v in spec["levels"]),
                fontsize=10.5, va="center", color="#333333", linespacing=1.5)
    fig.suptitle("확대해서 보면 (64×64 픽셀 영역, 픽셀 하나하나가 보이도록 확대)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "demo_zoom.png", dpi=120)
    plt.close(fig)


def save_tiles(g, out):
    """슬라이드에 직접 넣는 정사각형 타일 (448×448, 글자 없음). 필터는 가장 센 단계. 글자는 슬라이드에서 붙인다."""
    tiles = out / "tiles"
    tiles.mkdir(exist_ok=True)
    for key in GRID:
        img = g if key == "original" else apply_shift(g, key, 3, seed=42)
        Image.fromarray(img).resize((448, 448), Image.LANCZOS).save(tiles / f"{key}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="이미지 루트 (예: ~/data/chest_xray)")
    ap.add_argument("--image", required=True, help="이미지 루트 기준 상대 경로")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with Image.open(Path(args.images).expanduser() / args.image) as im:
        g = to_gray_224(im)
    plt = setup_matplotlib()
    fig_grid(plt, g, out)
    fig_zoom(plt, g, out)
    save_tiles(g, out)
    print(f"저장: {out / 'demo_grid.png'}, {out / 'demo_zoom.png'}, {out / 'tiles'}/*.png")


if __name__ == "__main__":
    main()
