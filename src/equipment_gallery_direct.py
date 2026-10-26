"""장치 설정 필터 v2.1(입력 크기 S에 직접 거는 판, training/equipment_direct.py)의 갤러리와 실측표 (모델·점수와 무관).

그림 feel_S.png   새 5종의 가장 센 단계를 크게 + 같은 부위 확대, 흰 막대 = 10mm (224 사진에서 1픽셀 ≈ 1mm 기준)
그림 zoom_S.png  새 5종의 원본 / 약 / 중 / 강을 같은 부위에서 확대
effects_S.md      val 200장에 걸어 잰 실제 변화량 (밝기·대비·경계·노이즈·원본과의 차이) + 선량 노이즈가 원리값과 맞는지

사용 예 (p1 폴더에서):
    .venv/bin/python src/equipment_gallery_direct.py --size 512 --image train/NORMAL/IM-0487-0001.jpeg --out reports/equipment_v2/direct --table
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))
from equipment_direct import EQUIP, SIGMA0, apply_equipment, px_per_mm  # noqa: E402
from filter_effects import measures  # noqa: E402
from gradcam import setup_matplotlib  # noqa: E402
from shift_filters import SHIFTS, apply_shift, to_gray  # noqa: E402

NEW = list(EQUIP)
REUSED = ["blur", "sharpen", "equalize", "jpeg", "bright", "dark"]
CROP224 = {"dose": (70, 134, 30, 94), "motion": (140, 204, 30, 94), "scatter": (90, 154, 76, 140),
           "kvp": (90, 154, 76, 140), "sensor": (70, 134, 30, 94)}  # (행 시작, 끝, 열 시작, 끝), 224 사진 기준 64×64 영역
N_IMAGES = 200


def title_of(name):
    return EQUIP[name]["title"] if name in EQUIP else SHIFTS[name]["title"] + " (v1)"


def value_of(name, level):
    return EQUIP[name]["labels"][level - 1] if name in EQUIP else str(SHIFTS[name]["levels"][level - 1])


def make(gray, name, level, seed):
    if name in EQUIP:
        return apply_equipment(gray, name, level, seed=seed)
    return apply_shift(gray, name, level, seed=seed)


def crop_of(name, size):
    s = size / 224
    r0, r1, c0, c1 = (int(round(v * s)) for v in CROP224[name])
    return slice(r0, r1), slice(c0, c1)


def measure_table(manifest, images_root, size):
    m = pd.read_csv(manifest)
    val = m[m.split == "val"]
    rows = val.iloc[np.random.default_rng(42).choice(len(val), size=N_IMAGES, replace=False)]
    acc = {}
    for i, rel in enumerate(rows.relative_path):
        with Image.open(images_root / rel) as im:
            g = to_gray(im, size)
        base = measures(g)
        for name in NEW + REUSED:
            for lv in (1, 2, 3):
                out = make(g, name, lv, 42 + i)
                ms = measures(out)
                diff = out.astype(np.float32) - g.astype(np.float32)
                acc.setdefault((name, lv), []).append(
                    {"d_mean": ms["mean"] - base["mean"], "std_x": ms["std"] / base["std"], "edge_x": ms["edge"] / base["edge"],
                     "noise_x": ms["noise"] / max(base["noise"], 1e-6), "mad": float(np.abs(diff).mean())})
    return {k: pd.DataFrame(v).mean() for k, v in acc.items()}


def build_markdown(table, size):
    def cell(name, col, fmt):
        return " / ".join(format(table[(name, lv)][col], fmt) for lv in (1, 2, 3))

    s = px_per_mm(size)
    L = [f"# 장치 설정 필터 v2.1 (입력 {size}×{size}): 임시가 아닌 고정 강도가 사진을 얼마나 바꾸나", "",
         f"val 사진 {N_IMAGES}장(seed 42)을 {size}×{size} 회색조로 만든 뒤 필터를 걸어 잰 평균이다 (`src/equipment_gallery_direct.py`). "
         f"모델 점수와는 무관하다. 길이는 224 사진에서 1픽셀 ≈ 1mm 기준이라 {size} 사진에서는 1mm = {s:.2f}픽셀이다. 값은 0~255 기준.", "",
         "| 필터 | 장치 원리 | 강도 (약 / 중 / 강) | 평균 밝기 변화 | 대비(밝기 폭) 배율 | 경계 세기 배율 | 노이즈 추정 배율 | 원본과 평균 차이 |",
         "|---|---|---|---|---|---|---|---|"]
    for name in NEW + REUSED:
        sit = EQUIP[name]["situation"] if name in EQUIP else SHIFTS[name]["situation"]
        param = EQUIP[name]["param"] if name in EQUIP else SHIFTS[name]["param"]
        L.append(f"| {title_of(name)} | {sit} | {' / '.join(value_of(name, lv) for lv in (1, 2, 3))} ({param}) | "
                 f"{cell(name, 'd_mean', '+.1f')} | {cell(name, 'std_x', '.2f')} | {cell(name, 'edge_x', '.2f')} | "
                 f"{cell(name, 'noise_x', '.2f')} | {cell(name, 'mad', '.1f')} |")
    exp = [1 / np.sqrt(d) for d in EQUIP["dose"]["levels"]]
    meas = [table[("dose", lv)]["noise_x"] for lv in (1, 2, 3)]
    L += ["", "## 확인한 것", "",
          f"- 선량: 원리대로면 노이즈는 선량의 제곱근에 반비례하므로 선량 {', '.join(EQUIP['dose']['labels'])}에서 "
          f"{exp[0]:.2f} / {exp[1]:.2f} / {exp[2]:.2f}배가 되어야 한다. 실제 사진에서 잰 노이즈 추정은 {meas[0]:.2f} / {meas[1]:.2f} / {meas[2]:.2f}배다 "
          f"(기준 노이즈 SIGMA0[{size}] = {SIGMA0.get(size, float('nan')):.3f}).",
          f"- 센서 화소 0.5 / 1.0 / 2.0mm는 {size} 사진에서 {0.5 * s:.1f} / {1.0 * s:.1f} / {2.0 * s:.1f} 픽셀이다. "
          "224 사진에서는 0.5 / 1.0 / 2.0 픽셀이라 앞의 둘은 거의 보이지 않았다.",
          "- 평균 밝기를 유지하도록 만든 필터(조사시간·산란·kVp·센서)의 평균 밝기 변화는 절댓값 최대 "
          f"{max(abs(table[(k, lv)]['d_mean']) for k in ('motion', 'scatter', 'kvp', 'sensor') for lv in (1, 2, 3)):.2f}다 (0~255 기준)."]
    return "\n".join(L) + "\n"


def fig_feel(plt, g, path, size, seed=42):
    from matplotlib.patches import Rectangle
    n = len(NEW)
    fig_h = 4.0 * n + 0.9
    fig, axes = plt.subplots(n, 4, figsize=(15.5, fig_h), gridspec_kw={"width_ratios": [1.5, 1.5, 1, 1]})
    s = size / 224
    for r, name in enumerate(NEW):
        crop = crop_of(name, size)
        out = make(g, name, 3, seed)
        views = [(g, "원본"), (out, f"강 · {value_of(name, 3)}"), (g[crop], "확대: 원본"), (out[crop], "확대: 강")]
        for c, (img, head) in enumerate(views):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(head, fontsize=12)
            if c < 2:
                ax.add_patch(Rectangle((crop[1].start - 0.5, crop[0].start - 0.5), crop[1].stop - crop[1].start,
                                       crop[0].stop - crop[0].start, fill=False, ec="#FFD400", lw=1.6))
                ax.plot([10 * s, 20 * s], [212 * s, 212 * s], color="white", lw=4, solid_capstyle="butt")
                ax.text(15 * s, 206 * s, "10 mm", color="white", fontsize=9, ha="center", va="bottom",
                        bbox=dict(facecolor="black", alpha=0.55, pad=1.5, lw=0))
        axes[r, 0].text(0, 1.15, f"{EQUIP[name]['title']}  [{EQUIP[name]['param']}: {' / '.join(EQUIP[name]['labels'])}]",
                        transform=axes[r, 0].transAxes, fontsize=14, fontweight="bold", va="bottom")
        axes[r, 0].text(0, 1.075, EQUIP[name]["situation"], transform=axes[r, 0].transAxes, fontsize=10.5, color="#555555", va="bottom")
    fig.suptitle(f"입력 {size}×{size}: 가장 센 단계 (흰 막대 = 10mm, 노란 상자 = 오른쪽에서 확대한 부위)", fontsize=15, y=1 - 0.15 / fig_h)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 1.4 / fig_h, bottom=0.01, hspace=0.5, wspace=0.05)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fig_zoom(plt, g, path, size, seed=42):
    fig, axes = plt.subplots(len(NEW), 5, figsize=(16.5, 3.5 * len(NEW)), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.45]})
    for r, name in enumerate(NEW):
        crop = crop_of(name, size)
        for c in range(4):
            ax = axes[r, c]
            ax.imshow(make(g, name, c, seed)[crop], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(["원본", "약", "중", "강"][c], fontsize=13)
            if c:
                ax.set_xlabel(value_of(name, c), fontsize=11)
        ax = axes[r, 4]
        ax.axis("off")
        ax.text(0, 0.62, EQUIP[name]["title"], fontsize=14, fontweight="bold", va="center")
        ax.text(0, 0.36, EQUIP[name]["situation"] + f"\n\n강도 ({EQUIP[name]['param']})\n" + " / ".join(EQUIP[name]["labels"]),
                fontsize=10.5, va="center", color="#333333", linespacing=1.5)
    fig.suptitle(f"입력 {size}×{size}에서 확대해서 보면 (64×64mm 영역, 224 기준 64픽셀에 해당)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--images", default=str(Path.home() / "data/chest_xray"))
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--manifest", default=str(ROOT / "manifests/split_manifest.csv"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.images).expanduser()
    with Image.open(root / args.image) as im:
        g = to_gray(im, args.size)
    plt = setup_matplotlib()
    fig_feel(plt, g, out / f"feel_{args.size}{args.tag}.png", args.size)
    fig_zoom(plt, g, out / f"zoom_{args.size}{args.tag}.png", args.size)
    print("그림 저장:", out)
    if args.table:
        table = measure_table(args.manifest, root, args.size)
        (out / f"effects_{args.size}.md").write_text(build_markdown(table, args.size), encoding="utf-8")
        print("표 저장:", out / f"effects_{args.size}.md")


if __name__ == "__main__":
    main()
