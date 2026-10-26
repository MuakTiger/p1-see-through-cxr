"""장치 설정 필터 v2를 실제 흉부 X선에 걸어 "물리량이 사진에서 어느 정도인지" 보여 주는 갤러리 (모델·점수와 무관, 강도는 임시).

그림 gallery_new.png     새로 만든 5종(선량·조사시간·산란·kVp·센서): 원본 / 약 / 중 / 강 / 강-원본 차이(증폭)
그림 gallery_reused.png  v1에서 그대로 쓰는 6종(흐림·경계 강조·대비 보정·JPEG·밝게·어둡게), 같은 배치
그림 zoom.png            눈에 안 띄는 변화를 64×64 영역으로 확대 (+ 참고로 v1 노이즈 std 4/8/16)
그림 feel.png            새 5종의 가장 센 단계를 크게 + 같은 부위 확대, 10mm 눈금자로 크기 감 잡기
effects.md (--table)     val 사진 200장에 걸어 잰 실제 변화량 (밝기·대비·경계·노이즈·원본과의 차이)

사용 예 (p1 폴더에서):
    .venv/bin/python src/equipment_gallery.py --images ~/data/chest_xray --image train/NORMAL/IM-0487-0001.jpeg \
        --out reports/equipment_v2 --table --manifest manifests/split_manifest.csv
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
from equipment_filters import EQUIP, apply_equipment, to_input, to_work  # noqa: E402
from shift_filters import SHIFTS, apply_shift, to_gray_224  # noqa: E402
from filter_effects import measures  # noqa: E402
from gradcam import setup_matplotlib  # noqa: E402

NEW = list(EQUIP)
REUSED = ["blur", "sharpen", "equalize", "jpeg", "bright", "dark"]
SITUATION = {"blur": "초점 크기·촬영 거리로 생기는 번짐 (센서 화소 차이는 아님)"}  # v1 설명 중 고친 것
KEY_METRIC = {"dose": "noise_x", "motion": "edge_x", "scatter": "std_x", "kvp": "std_x", "sensor": "edge_x",
              "blur": "edge_x", "sharpen": "edge_x", "equalize": "std_x", "jpeg": "noise_x",
              "bright": "d_mean", "dark": "d_mean"}
METRIC_TEXT = {"noise_x": "노이즈 ×{:.2f}", "edge_x": "경계 ×{:.2f}", "std_x": "대비 ×{:.2f}", "d_mean": "밝기 {:+.0f}"}
CROP = (slice(70, 134), slice(30, 94))        # 왼쪽 폐 윗부분 (갈비뼈와 폐 무늬)
CROP_LOW = (slice(140, 204), slice(30, 94))   # 왼쪽 폐 아래와 횡격막 근처
CROP_MID = (slice(90, 154), slice(76, 140))   # 가운데: 척추·심장 경계 (산란·kVp가 대비를 바꾸는 곳)
FEEL_CROPS = {"dose": CROP, "motion": CROP_LOW, "scatter": CROP_MID, "kvp": CROP_MID, "sensor": CROP}
N_IMAGES = 200
SEED = 42


def title_of(key):
    return EQUIP[key]["title"] if key in EQUIP else SHIFTS[key]["title"]


def situation_of(key):
    return EQUIP[key]["situation"] if key in EQUIP else SITUATION.get(key, SHIFTS[key]["situation"])


def label_of(key, lv):
    return EQUIP[key]["labels"][lv - 1] if key in EQUIP else str(SHIFTS[key]["levels"][lv - 1])


def param_of(key):
    return EQUIP[key]["param"] if key in EQUIP else SHIFTS[key]["param"]


def make(work, g224, key, lv, seed):
    """필터 key를 단계 lv로 걸어 224×224 uint8을 돌려준다. 새 필터는 작업 해상도(896)에서, v1 필터는 224에서 건다."""
    if key in EQUIP:
        return to_input(work, apply_equipment(work, key, lv, seed), g224)
    return apply_shift(g224, key, lv, seed=seed)


def measure_table(manifest, images_root):
    m = pd.read_csv(manifest)
    val = m[m.split == "val"]
    rng = np.random.default_rng(SEED)
    rows = val.iloc[rng.choice(len(val), size=N_IMAGES, replace=False)]
    acc = {}
    for i, rel in enumerate(rows.relative_path):
        with Image.open(images_root / rel) as im:
            work, g = to_work(im), to_gray_224(im)
        base = measures(g)
        for key in NEW + REUSED:
            for lv in range(1, 4):
                out = make(work, g, key, lv, i)
                ms = measures(out)
                diff = out.astype(np.float32) - g.astype(np.float32)
                acc.setdefault((key, lv), []).append(
                    {"d_mean": ms["mean"] - base["mean"], "std_x": ms["std"] / base["std"],
                     "edge_x": ms["edge"] / base["edge"], "noise_x": ms["noise"] / max(base["noise"], 1e-6),
                     "mad": float(np.abs(diff).mean())})
    return {k: pd.DataFrame(v).mean() for k, v in acc.items()}


def fig_gallery(plt, work, g, keys, title, path, seed=42):
    n = len(keys)
    fig_h = 3.55 * n + 0.8
    fig, axes = plt.subplots(n, 5, figsize=(14.8, fig_h))
    for r, key in enumerate(keys):
        outs = [g] + [make(work, g, key, lv, seed) for lv in (1, 2, 3)]
        diff = outs[3].astype(np.float32) - g.astype(np.float32)
        gain = float(np.clip(100.0 / max(np.percentile(np.abs(diff), 99), 1e-3), 1.0, 60.0))
        shown = outs + [np.clip(128.0 + diff * gain, 0, 255)]
        heads = ["원본"] + [f"{n_} · {label_of(key, lv)}" for lv, n_ in zip((1, 2, 3), ("약", "중", "강"))] + [f"강 - 원본 (x{gain:.0f} 증폭)"]
        for c in range(5):
            ax = axes[r, c]
            ax.imshow(shown[c], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(heads[c], fontsize=11.5)
        axes[r, 0].text(0, 1.2, f"{title_of(key)}  [{param_of(key)}]", transform=axes[r, 0].transAxes, fontsize=14,
                        fontweight="bold", va="bottom")
        axes[r, 0].text(0, 1.115, situation_of(key), transform=axes[r, 0].transAxes, fontsize=10.5, color="#555555", va="bottom")
    fig.suptitle(title, fontsize=15, y=1 - 0.15 / fig_h)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 1.45 / fig_h, bottom=0.01, hspace=0.62, wspace=0.04)  # 위쪽 1.45인치: 그림 제목 + 첫 줄 제목
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fig_feel(plt, work, g, path, seed=42):
    """"이 강도가 어느 정도인지" 감을 잡는 그림: 원본과 가장 센 단계를 크게, 같은 부위를 확대해서 나란히.
    큰 사진 왼쪽 아래의 흰 막대가 10mm (가슴 폭 약 20cm = 224픽셀 가정, 1픽셀 ≈ 1mm)."""
    from matplotlib.patches import Rectangle
    n = len(NEW)
    fig_h = 4.0 * n + 0.9
    fig, axes = plt.subplots(n, 4, figsize=(15.5, fig_h), gridspec_kw={"width_ratios": [1.5, 1.5, 1, 1]})
    for r, key in enumerate(NEW):
        crop = FEEL_CROPS[key]
        out = make(work, g, key, 3, seed)
        r0, c0 = crop[0].start, crop[1].start
        views = [(g, "원본"), (out, f"강 · {label_of(key, 3)}"), (g[crop], "확대: 원본"), (out[crop], "확대: 강")]
        for c, (img, head) in enumerate(views):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(head, fontsize=12)
            if c < 2:
                ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), 64, 64, fill=False, ec="#FFD400", lw=1.6))
                ax.plot([10, 20], [212, 212], color="white", lw=4, solid_capstyle="butt")
                ax.text(15, 206, "10 mm", color="white", fontsize=9, ha="center", va="bottom",
                        bbox=dict(facecolor="black", alpha=0.55, pad=1.5, lw=0))
        axes[r, 0].text(0, 1.15, f"{title_of(key)}  [{param_of(key)}: {' / '.join(EQUIP[key]['labels'])}]",
                        transform=axes[r, 0].transAxes, fontsize=14, fontweight="bold", va="bottom")
        axes[r, 0].text(0, 1.075, situation_of(key), transform=axes[r, 0].transAxes, fontsize=10.5, color="#555555", va="bottom")
    fig.suptitle("가장 센 단계가 어느 정도인가 (흰 막대 = 10mm, 노란 상자 = 오른쪽에서 확대한 부위)", fontsize=15, y=1 - 0.15 / fig_h)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 1.4 / fig_h, bottom=0.01, hspace=0.5, wspace=0.05)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fig_zoom(plt, work, g, path, seed=42):
    rows = [("dose", CROP, "dose"), ("motion", CROP_LOW, "motion"), ("sensor", CROP, "sensor"), ("noise", CROP, "v1 noise")]
    fig, axes = plt.subplots(len(rows), 5, figsize=(16.5, 3.5 * len(rows)), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.45]})
    for r, (key, crop, _) in enumerate(rows):
        for c in range(4):
            out = make(work, g, key, c, seed) if key in EQUIP else apply_shift(g, key, c, seed=seed)
            ax = axes[r, c]
            ax.imshow(out[crop], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(["원본", "약", "중", "강"][c], fontsize=13)
            if c:
                ax.set_xlabel(label_of(key, c), fontsize=11)
        ax = axes[r, 4]
        ax.axis("off")
        if key == "noise":
            head, body = "(참고) v1 노이즈", "지난번 장비 차이 시험에서 쓴 가우시안 노이즈.\n표준편차 4 / 8 / 16.\n위 선량 노이즈와 같은 크기로 비교해 보기 위한 줄."
        else:
            head, body = title_of(key), situation_of(key) + f"\n\n강도 ({param_of(key)})\n" + " / ".join(EQUIP[key]["labels"])
            if key == "motion":
                body += "\n\n(아래쪽 영역을 확대)"
        ax.text(0, 0.62, head, fontsize=14, fontweight="bold", va="center")
        ax.text(0, 0.36, body, fontsize=10.5, va="center", color="#333333", linespacing=1.5)
    fig.suptitle("확대해서 보면 (64×64 픽셀 영역, 224 사진 기준)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def build_markdown(table, v1_noise):
    def cell(key, col, fmt):
        return " / ".join(format(table[(key, lv)][col], fmt) for lv in (1, 2, 3))

    L = ["# 장치 설정 필터 v2: 임시 강도가 사진을 얼마나 바꾸나", "",
         f"val 사진 {N_IMAGES}장(seed {SEED})에 필터를 걸어 잰 평균이다 (`src/equipment_gallery.py`, 필터는 `training/equipment_filters.py`). "
         "**강도는 임시이며 갤러리 사진을 보고 확정한다.** 모델 점수와는 무관하다. 224×224 회색조, 값은 0~255 기준. "
         "새 필터 5종은 896×896 작업 해상도에서 걸고 224로 줄이며, v1 필터 6종은 224에서 그대로 건다.", "",
         "| 필터 | 장치 원리 | 강도 (약 / 중 / 강) | 평균 밝기 변화 | 대비(밝기 폭) 배율 | 경계 세기 배율 | 노이즈 추정 배율 | 원본과 평균 차이 |",
         "|---|---|---|---|---|---|---|---|"]
    for key in NEW + REUSED:
        L.append(f"| {title_of(key)}{'' if key in EQUIP else ' (v1)'} | {situation_of(key)} | "
                 f"{' / '.join(label_of(key, lv) for lv in (1, 2, 3))} ({param_of(key)}) | {cell(key, 'd_mean', '+.1f')} | "
                 f"{cell(key, 'std_x', '.2f')} | {cell(key, 'edge_x', '.2f')} | {cell(key, 'noise_x', '.2f')} | {cell(key, 'mad', '.1f')} |")
    exp = [1 / np.sqrt(d) for d in EQUIP["dose"]["levels"]]
    meas = [table[("dose", lv)]["noise_x"] for lv in (1, 2, 3)]
    L += ["", "## 확인한 것", "",
          f"- 선량: 원리대로면 노이즈는 선량의 제곱근에 반비례하므로 선량 {', '.join(EQUIP['dose']['labels'])}에서 {exp[0]:.2f} / {exp[1]:.2f} / {exp[2]:.2f}배가 되어야 한다. "
          f"실제 사진에서 잰 노이즈 추정(Immerkaer)은 {meas[0]:.2f} / {meas[1]:.2f} / {meas[2]:.2f}배다.",
          f"- v1 가우시안 노이즈(std 4 / 8 / 16)의 노이즈 추정은 {v1_noise[0]:.2f} / {v1_noise[1]:.2f} / {v1_noise[2]:.2f}배였다. "
          "같은 원리(노이즈 ∝ 1/√선량)로 환산하면 선량 "
          + " / ".join(f"1/{x ** 2:.0f}" for x in v1_noise)
          + " 에 해당한다. 원본의 노이즈를 모두 양자 노이즈로 가정한 환산이라 정확하지 않다.",
          f"- 평균 밝기를 유지하도록 만든 필터(조사시간·산란·kVp·센서)의 평균 밝기 변화는 절댓값 최대 "
          f"{max(abs(table[(k, lv)]['d_mean']) for k in ('motion', 'scatter', 'kvp', 'sensor') for lv in (1, 2, 3)):.2f}다 (0~255 기준).",
          "- v1 필터 6종의 숫자는 `reports/shift_test/filter_effects.md`와 같아야 한다 (같은 필터, 같은 사진 200장, 같은 원본 224). 같으면 기준이 이어진 것이다.", "",
          "## 가정과 한계", "",
          "- 사진은 이미 8비트 JPEG로 처리된 표시 값이다. 표시 값이 노출량과 선형이라고 가정했고 실제 톤 곡선은 모른다.",
          "- 노이즈 기준(`SIGMA_REF`)은 원본 224 사진의 노이즈 추정 약 0.9에 맞춘 값이다. 이 값에는 JPEG와 질감이 섞여 있어 순수한 양자 노이즈가 아니다.",
          "- 선량 노이즈는 서로 독립인 가우시안으로 단순화했고 검출기의 노이즈 상관, 전자 노이즈는 넣지 않았다.",
          "- 움직임 번짐의 길이, 산란 비율 k, 산란 폭(25mm), kVp의 대비·기울기는 실측값이 아니라 눈으로 정할 임시 설정이다.",
          "- kVp는 뼈와 연부조직의 대비를 에너지에 따라 서로 다르게 바꾸지만, 한 장의 표시 사진에서는 둘을 나눌 수 없어 전체 대비 감소와 밝은 쪽 눌러 주기로 근사했다.",
          "- 센서 화소 크기는 224로 줄이면 가려지는 것이 핵심이다. 화소 0.9mm는 224 사진의 1픽셀과 같은 크기다."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--image", required=True, help="이미지 루트 기준 상대 경로 (그림에 쓸 사진)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="", help="파일 이름 뒤에 붙일 말 (예: _pneumonia)")
    ap.add_argument("--table", action="store_true", help="val 200장에 걸어 effects.md를 만든다")
    ap.add_argument("--manifest")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.images).expanduser()
    with Image.open(root / args.image) as im:
        work, g = to_work(im), to_gray_224(im)
    plt = setup_matplotlib()
    fig_gallery(plt, work, g, NEW, "장치 설정 필터 5종 (새로 만든 것): 같은 폐 사진, 224×224, 강도는 임시", out / f"gallery_new{args.tag}.png")
    fig_gallery(plt, work, g, REUSED, "v1에서 그대로 쓰는 필터 6종: 같은 폐 사진, 224×224", out / f"gallery_reused{args.tag}.png")
    fig_zoom(plt, work, g, out / f"zoom{args.tag}.png")
    fig_feel(plt, work, g, out / f"feel{args.tag}.png")
    print("그림 저장:", out)
    if args.table:
        table = measure_table(Path(args.manifest), root)
        # v1 노이즈(std 4/8/16)의 추정 배율은 같은 사진 200장에서 따로 잰다 (선량으로 환산해 비교하려고)
        m = pd.read_csv(args.manifest)
        val = m[m.split == "val"]
        rng = np.random.default_rng(SEED)
        rows = val.iloc[rng.choice(len(val), size=N_IMAGES, replace=False)]
        acc = [[], [], []]
        for i, rel in enumerate(rows.relative_path):
            with Image.open(root / rel) as im:
                g_i = to_gray_224(im)
            base = measures(g_i)["noise"]
            for j, lv in enumerate((1, 2, 3)):
                acc[j].append(measures(apply_shift(g_i, "noise", lv, seed=i))["noise"] / max(base, 1e-6))
        v1_noise = [float(np.mean(a)) for a in acc]
        (out / "effects.md").write_text(build_markdown(table, v1_noise), encoding="utf-8")
        print("표 저장:", out / "effects.md")


if __name__ == "__main__":
    main()
