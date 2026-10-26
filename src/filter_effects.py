"""장비 차이 필터 8종 × 3단계가 사진을 실제로 얼마나 바꾸는지 재는 스크립트 (val 사진 N장, 모델·점수와 무관).

필터와 강도는 training/shift_filters.py 의 SHIFTS 그대로이며, 시험 전에 고정한 값이다 (근거는 reports/shift_test/filter_rationale.md).
여기서는 "그 강도가 사진에서 어떤 변화인가"를 숫자와 히스토그램으로 보여 준다.

재는 것 (224×224 회색조, 0~255 기준, 사진마다 잰 뒤 평균):
    평균 밝기 변화      필터 후 평균 − 원본 평균
    대비 배율           밝기 표준편차의 배율 (원본 = 1)
    경계 세기 배율      Sobel 경계 세기 평균의 배율 (흐리면 작아지고 강조·노이즈면 커진다)
    노이즈 추정 배율    Immerkaer 노이즈 추정의 배율
    원본과의 평균 차이  |필터 후 − 원본| 평균 (0~255), PSNR(dB, 클수록 원본에 가깝다)

사용 예 (p1 폴더에서):
    .venv/bin/python src/filter_effects.py --manifest manifests/split_manifest.csv --images ~/data/chest_xray \
        --out reports/shift_test/filter_effects
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))
from shift_filters import SHIFTS, apply_shift, to_gray_224  # noqa: E402
from normal_shift_analysis import immerkaer_noise  # noqa: E402
from gradcam import setup_matplotlib  # noqa: E402

SEED = 42
N_IMAGES = 200
BINS = 64


def measures(g):
    f = g.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return {"mean": float(f.mean()), "std": float(f.std()),
            "edge": float(np.sqrt(gx ** 2 + gy ** 2).mean()), "noise": immerkaer_noise(g)}


def run(manifest, images_root):
    m = pd.read_csv(manifest)
    val = m[m.split == "val"]
    rng = np.random.default_rng(SEED)
    rows = val.iloc[rng.choice(len(val), size=N_IMAGES, replace=False)]
    acc = {}   # (필터, 단계) → 사진별 측정값 목록
    hist = {}  # (필터, 단계) → 평균 히스토그램
    for i, rel in enumerate(rows.relative_path):
        with Image.open(images_root / rel) as im:
            g = to_gray_224(im)
        base = measures(g)
        for name in SHIFTS:
            for lv in range(0, 4):
                out = apply_shift(g, name, lv, seed=i)
                ms = measures(out)
                diff = out.astype(np.float32) - g.astype(np.float32)
                mse = float((diff ** 2).mean())
                rec = {"d_mean": ms["mean"] - base["mean"], "std_x": ms["std"] / base["std"],
                       "edge_x": ms["edge"] / base["edge"], "noise_x": ms["noise"] / max(base["noise"], 1e-6),
                       "mad": float(np.abs(diff).mean()), "psnr": 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)}
                acc.setdefault((name, lv), []).append(rec)
                h = np.histogram(out, bins=BINS, range=(0, 256))[0] / out.size
                hist[(name, lv)] = hist.get((name, lv), 0) + h / N_IMAGES
    table = {k: pd.DataFrame(v).mean() for k, v in acc.items()}
    return table, hist


def level_label(name, lv):
    v = SHIFTS[name]["levels"][lv - 1]
    return str(v)


def build_markdown(table):
    L = ["# 장비 차이 필터 8종 × 3단계: 실제로 사진이 얼마나 바뀌나", "",
         f"val 사진 {N_IMAGES}장(seed {SEED})에 필터를 걸어 잰 평균이다 (`src/filter_effects.py`). 필터와 강도는 시험 전에 정한 값이고, "
         "원인과 선택 근거는 `reports/shift_test/filter_rationale.md`에 있다. 224×224 회색조, 값은 0~255 기준이다.", "",
         "## 1. 측정표", "",
         "| 필터 | 흉내 내는 상황 | 강도 (약 / 중 / 강) | 평균 밝기 변화 | 대비(밝기 폭) 배율 | 경계 세기 배율 | 노이즈 추정 배율 | 원본과 평균 차이 | PSNR (dB) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, spec in SHIFTS.items():
        rs = [table[(name, lv)] for lv in (1, 2, 3)]
        cell = lambda col, fmt: " / ".join(format(r[col], fmt) for r in rs)  # noqa: E731
        L.append(f"| {spec['title']} | {spec['situation']} | {' / '.join(str(v) for v in spec['levels'])} ({spec['param']}) | "
                 f"{cell('d_mean', '+.1f')} | {cell('std_x', '.2f')} | {cell('edge_x', '.2f')} | {cell('noise_x', '.2f')} | "
                 f"{cell('mad', '.1f')} | {cell('psnr', '.1f')} |")
    t = lambda name, lv, col: table[(name, lv)][col]  # noqa: E731

    hist_group = [("bright", "밝게"), ("dark", "어둡게"), ("contrast", "대비 감소"), ("equalize", "대비 보정")]
    other_group = [("blur", "흐림"), ("sharpen", "경계 강조"), ("jpeg", "JPEG 압축"), ("noise", "노이즈")]
    L += ["", "## 2. 읽는 법", "",
          "**히스토그램(밝기 분포)에서 바로 보이는 필터 4개**", "",
          f"- 밝게(감마 < 1): 평균 밝기가 약 / 중 / 강에서 {t('bright', 1, 'd_mean'):+.1f} / {t('bright', 2, 'd_mean'):+.1f} / {t('bright', 3, 'd_mean'):+.1f}. 분포 전체가 오른쪽으로 밀린다.",
          f"- 어둡게(감마 > 1): {t('dark', 1, 'd_mean'):+.1f} / {t('dark', 2, 'd_mean'):+.1f} / {t('dark', 3, 'd_mean'):+.1f}. 왼쪽으로 밀린다.",
          f"- 대비 감소: 평균은 그대로(변화 {t('contrast', 3, 'd_mean'):+.1f})이고 폭만 줄어든다 (강에서 {t('contrast', 3, 'std_x'):.2f}배). 분포가 평균 쪽으로 모인다.",
          f"- 대비 보정(CLAHE → HE): 폭이 약 / 중 / 강에서 {t('equalize', 1, 'std_x'):.2f} / {t('equalize', 2, 'std_x'):.2f} / {t('equalize', 3, 'std_x'):.2f}배. 분포를 고르게 펴는 방향이고, 같은 사진도 구역마다 다르게 바뀐다.",
          f"  - CLAHE(약·중)는 평균 밝기도 {t('equalize', 1, 'd_mean'):+.1f} / {t('equalize', 2, 'd_mean'):+.1f} 올리고, 노이즈 추정을 {t('equalize', 1, 'noise_x'):.2f} / {t('equalize', 2, 'noise_x'):.2f}배로 키운다. "
          "구역별로 대비를 올리면서 알갱이도 같이 키운 것이다. 이 전처리 모델이 노이즈에 가장 약했던 것과 방향이 맞는 정황이지만, 원인이라고 확인한 것은 아니다.", "",
          "**히스토그램에는 거의 안 나타나고 경계·질감에서 보이는 필터 4개**", ""]
    quiet = max(abs(t(n, 3, "d_mean")) for n, _ in other_group)
    quiet_std = [t(n, 3, "std_x") for n, _ in other_group]
    L += [f"- 이 네 가지는 강한 단계에서도 평균 밝기 변화가 최대 {quiet:.1f}, 대비 배율이 {min(quiet_std):.2f}~{max(quiet_std):.2f}라서 히스토그램만 보면 거의 같다. "
          "그래서 아래 지표로 본다.",
          f"- 흐림: 경계 세기가 강에서 원본의 {t('blur', 3, 'edge_x'):.2f}배로 줄고 노이즈 추정도 {t('blur', 3, 'noise_x'):.2f}배로 줄어든다. 미세한 무늬가 지워진다.",
          f"- 경계 강조: 경계 세기가 강에서 {t('sharpen', 3, 'edge_x'):.2f}배, 노이즈 추정이 {t('sharpen', 3, 'noise_x'):.2f}배로 늘어난다. 경계와 미세 무늬가 날카로워진다.",
          f"- 노이즈: 노이즈 추정이 강에서 {t('noise', 3, 'noise_x'):.1f}배, 경계 세기도 {t('noise', 3, 'edge_x'):.2f}배로 늘어난다. 단, 노이즈가 만드는 '경계'는 진짜 경계가 아니라 알갱이다.",
          f"- JPEG 압축: 평균 밝기는 그대로이고 원본과의 평균 차이는 약 / 중 / 강에서 {t('jpeg', 1, 'mad'):.1f} / {t('jpeg', 2, 'mad'):.1f} / {t('jpeg', 3, 'mad'):.1f}, 경계 세기는 {t('jpeg', 3, 'edge_x'):.2f}배(강)로 거의 그대로이지만 노이즈 추정은 {t('jpeg', 1, 'noise_x'):.2f} / {t('jpeg', 2, 'noise_x'):.2f} / {t('jpeg', 3, 'noise_x'):.2f}배로 줄어든다. 미세한 무늬가 사라진다는 뜻이고, 8×8 블록 무늬는 숫자보다 사진에서 잘 보인다 (`reports/shift_gallery/jpeg.png`).", "",
          "## 3. 강도를 어떻게 읽을까", "",
          f"- 같은 '강'이라도 사진이 바뀌는 크기는 필터마다 다르다. 원본과의 평균 차이(0~255)로 보면 강에서 가장 큰 것은 "
          f"{SHIFTS[max(SHIFTS, key=lambda n: t(n, 3, 'mad'))]['title']}({max(t(n, 3, 'mad') for n in SHIFTS):.1f}), 가장 작은 것은 "
          f"{SHIFTS[min(SHIFTS, key=lambda n: t(n, 3, 'mad'))]['title']}({min(t(n, 3, 'mad') for n in SHIFTS):.1f})이다.",
          "- 그러므로 점수가 떨어진 정도를 필터끼리 직접 비교하면 안 된다. '노이즈 16에서 크게 떨어졌다'는 말은 그 필터의 그 강도에서 그렇다는 뜻이다.",
          "- 강도는 특정 장비의 실측이 아니라, 원인별 변화 방향과 대략의 크기를 반영해 시험 전에 정한 설정이다.", "",
          "## 4. 한계", "",
          "- 평균은 val 사진 일부(200장)에서 쟀다. 사진마다 밝기와 노이즈가 달라 표준편차는 표에 넣지 않았다.",
          "- 경계 세기와 노이즈 추정은 근사 지표다 (Sobel, Immerkaer). 실제 장비의 노이즈 특성(밝기에 따라 달라지는 양자 노이즈)과는 다르다."]
    return "\n".join(L)


def figure(hist, table, out):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.2))
    xs = (np.arange(BINS) + 0.5) * 256 / BINS
    colors = ["#222222", "#F2B84B", "#E0772D", "#B8291F"]
    for ax, (name, spec) in zip(axes.ravel(), SHIFTS.items()):
        for lv in range(4):
            lab = "원본" if lv == 0 else f"{['약', '중', '강'][lv - 1]} ({spec['levels'][lv - 1]})"
            ax.plot(xs, hist[(name, lv)], color=colors[lv], lw=2.2 if lv == 0 else 1.6, label=lab)
        ax.set_title(f"{spec['title']}", fontsize=12)
        ax.set_xlim(0, 255)
        ax.set_xlabel("밝기 (0 검정 ~ 255 흰색)", fontsize=9)
        quiet = abs(table[(name, 3)]["d_mean"]) < 2 and 0.95 < table[(name, 3)]["std_x"] < 1.05
        if quiet:
            ax.text(0.5, 0.62, "히스토그램은 거의 그대로\n(경계·질감이 바뀜)", transform=ax.transAxes, ha="center",
                    fontsize=10, color="#A3500A")
        ax.legend(fontsize=8, loc="upper right")
    axes[0, 0].set_ylabel("사진 한 장당 비율 (200장 평균)")
    axes[1, 0].set_ylabel("사진 한 장당 비율 (200장 평균)")
    fig.suptitle("장비 차이 필터가 만든 밝기 분포(히스토그램) 변화 (val 사진 200장 평균, 224×224)", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out) + ".png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True, help="저장 경로(확장자 없음). .md 와 .png 가 만들어진다")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table, hist = run(Path(args.manifest).expanduser(), Path(args.images).expanduser())
    Path(str(out) + ".md").write_text(build_markdown(table), encoding="utf-8")
    figure(hist, table, out)
    print(f"저장: {out}.md, {out}.png")


if __name__ == "__main__":
    main()
