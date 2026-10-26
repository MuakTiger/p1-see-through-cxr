"""test 정상 사진은 train/val 정상 사진과 무엇이 다른가 (사후 설명용, iMac 전용, 비용 없음).

왜 하는가
    최종 test 에서 모델 D 는 정상 사진 231장 중 129장을 폐렴으로 틀렸다 (정상 recall: val 0.981 → test 0.442).
    val 은 원래 train 폴더 안에서 나눴고, test 는 데이터 제공자가 처음부터 따로 둔 폴더다.
    "test 정상 사진이 학습 때 본 정상 사진과 생김새·파일 성질이 다르다"는 가설을 숫자로 확인한다.
    모델·임계값·test 점수는 바꾸지 않는다. 결과는 설명에만 쓰고 모델 선택에는 쓰지 않는다.

무엇을 하는가
    1. 사진별 속성 표 (normal_features.csv)
       - 대상: 정상 전체 (train 1,075 / val 266 / test 234) + 비교용 폐렴 (split·클래스마다 최대 300장, seed 42)
       - 파일 속성: 크기, 종횡비, 픽셀 수, 파일 크기, 바이트/픽셀, 채널 수, JPEG 품질 추정, 메타데이터 표식
       - 영상 통계: 모델과 같은 224×224 회색조(training/shift_filters.to_gray_224)에서 밝기·대비·엔트로피·경계,
         노이즈는 원본 해상도와 224 두 곳에서 추정
    2. 분포 차이: split 별 중앙값·IQR, Cliff's delta (train vs test, train vs val, val vs test)
    3. 식별 가능성: 속성만으로 "train 풀(train+val)인가 test 인가"를 맞히는 로지스틱 회귀, 5-fold AUROC
       (정상과 폐렴 각각, 같은 환자는 같은 fold 에 둔다)
    4. 오분류와의 관련: test 정상 맞힘 vs 틀림, train 정상 분포에서 벗어난 정도와 p_normal 의 상관
    5. 그림 3장, 표 모음 (normal_shift_tables.md), 숫자 모음 (normal_shift_summary.json)

열어 보는 영상: 위 대상 사진만. 학습·평가는 하지 않는다.

사용 예:
    .venv/bin/python src/normal_shift_analysis.py
    .venv/bin/python src/normal_shift_analysis.py --reuse-features   # 속성 표를 다시 계산하지 않고 분석만
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from shift_filters import to_gray_224  # noqa: E402  (모델 입력과 같은 회색조 224 축소)

ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
SEED = 42
SPLITS = ["train", "val", "test"]

# JPEG 표준(ITU-T T.81 부록 K) 밝기 양자화표. libjpeg 는 품질 Q 에 따라 이 표를 배율 S 로 키우거나 줄인다.
#   Q < 50 이면 S = 5000 / Q,  Q ≥ 50 이면 S = 200 − 2Q  (값 = 표 × S / 100, 최소 1)
STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)

# 특징 묶음 (파일명·번호는 넣지 않는다: 번호 범위가 split 마다 달라서 넣으면 답을 알려 주는 셈)
FILE_FEATS = ["width", "height", "aspect", "pixels", "file_bytes", "bytes_per_pixel", "channels", "jpeg_quality"]
META_FEATS = ["has_exif", "has_icc", "has_photoshop"]
IMAGE_FEATS = ["mean", "std", "p5", "p95", "entropy", "sobel_mean", "noise_orig", "noise_224", "border_mean"]
IMAGE224_FEATS = [f for f in IMAGE_FEATS if f != "noise_orig"]  # 모델이 실제로 보는 224 영상에서만 잰 것
SHAPE_FEATS = ["width", "height", "aspect", "pixels"]  # 크기·모양만 (test 정상 안에서 값이 제각각인 파일 속성)
FEATURE_SETS = {
    "a_file": ("(a) 파일 속성", FILE_FEATS),
    "a2_file_meta": ("(a+) 파일 속성\n+ 메타데이터 표식", FILE_FEATS + META_FEATS),
    "b_image": ("(b) 밝기·대비·노이즈", IMAGE_FEATS),
    "b2_image224": ("(b224) 224 영상 통계만\n(모델이 보는 것)", IMAGE224_FEATS),
    "c_all": ("(c) 전부", FILE_FEATS + META_FEATS + IMAGE_FEATS),
}
FIG_FEATS = ["jpeg_quality", "bytes_per_pixel", "noise_orig", "noise_224", "aspect", "pixels"]
KO = {
    "width": "가로(px)", "height": "세로(px)", "aspect": "종횡비(가로/세로)", "pixels": "픽셀 수(백만)",
    "file_bytes": "파일 크기(KB)", "bytes_per_pixel": "바이트/픽셀", "channels": "채널 수",
    "jpeg_quality": "JPEG 품질 추정", "has_exif": "EXIF 있음", "has_icc": "ICC 색 프로필 있음",
    "has_photoshop": "Photoshop 표식 있음", "mean": "평균 밝기", "std": "대비(밝기 표준편차)",
    "p5": "어두운 쪽 5% 밝기", "p95": "밝은 쪽 95% 밝기", "entropy": "엔트로피(bit)",
    "sobel_mean": "경계 세기(Sobel 평균)", "noise_orig": "노이즈 추정(원본 해상도)",
    "noise_224": "노이즈 추정(224)", "border_mean": "테두리 10% 평균 밝기",
}
SCALE_FOR_TABLE = {"pixels": 1e-6, "file_bytes": 1 / 1024}  # 표에 보기 좋게 단위만 바꾼다 (분석은 원래 값)


# ----------------------------------------------------------------------------
# 1. 사진별 속성
# ----------------------------------------------------------------------------
def prefix_of(name):
    """파일명 접두어와 번호. 정상은 IM-XXXX / NORMAL2-IM-XXXX, 폐렴은 personN."""
    m = re.match(r"^(NORMAL2-IM-|IM-)(\d+)", name)
    if m:
        return m.group(1), int(m.group(2))
    m = re.match(r"^person(\d+)_", name)
    if m:
        return "person", int(m.group(1))
    return "other", -1


def estimate_jpeg_quality(im):
    """PIL 의 밝기 양자화표(0번)를 표준표와 비교해 libjpeg 품질 Q 를 거꾸로 추정한다 (근사).

    배율 S = 100 × (표 값의 합) / (표준표의 합). S ≤ 100 이면 Q = (200 − S)/2, 아니면 Q = 5000/S.
    표 값이 1 로 잘리는 고품질(≈ 95 이상)이나 libjpeg 가 아닌 프로그램(예: Photoshop)의 표는
    정확히 맞지 않는다. 그래도 "같은 방식으로 저장됐나"를 비교하는 데는 충분하다.
    """
    q = getattr(im, "quantization", None)
    if not q or 0 not in q:
        return np.nan
    s = 100.0 * float(np.sum(q[0])) / STD_LUMA.sum()
    quality = (200.0 - s) / 2.0 if s <= 100 else 5000.0 / s
    return float(np.clip(quality, 1, 100))


def immerkaer_noise(gray):
    """Immerkær(1996) 빠른 노이즈 추정. 3×3 라플라시안 차분 마스크의 절댓값 평균으로 노이즈 표준편차를 근사.

    sigma = sqrt(pi/2) / (6 (W−2)(H−2)) × Σ|I * M|,  M = [[1,−2,1],[−2,4,−2],[1,−2,1]]
    마스크가 평평한 면과 기울기를 지우므로 남는 값은 주로 미세한 알갱이(노이즈)다.
    단, 아주 가는 구조(뼈 경계 등)와 JPEG 압축으로 사라진 노이즈도 영향을 준다 → '추정'으로만 읽는다.
    """
    m = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    g = gray.astype(np.float32)
    conv = cv2.filter2D(g, -1, m, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    h, w = g.shape
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6.0 * (w - 2) * (h - 2)))


def image_features(path):
    with Image.open(path) as im:
        info = im.info
        feats = {
            "pil_mode": im.mode,
            "jpeg_quality": estimate_jpeg_quality(im),
            "has_exif": int("exif" in info),
            "has_icc": int("icc_profile" in info),
            "has_photoshop": int("photoshop" in info),
        }
        gray_orig = np.asarray(im.convert("L"), dtype=np.uint8)  # 원본 해상도 회색조 (노이즈 추정용)
        g = to_gray_224(im)  # 모델이 보는 것과 같은 224×224 (train.py P0 도 같은 크기로 줄인다)
    gf = g.astype(np.float32)
    hist = np.bincount(g.ravel(), minlength=256) / g.size
    nz = hist[hist > 0]
    sx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    b = int(round(224 * 0.10))  # 가장자리 10% = 22픽셀 띠
    border = np.ones_like(g, dtype=bool)
    border[b:-b, b:-b] = False
    feats.update({
        "mean": float(gf.mean()),
        "std": float(gf.std()),  # RMS 대비
        "p5": float(np.percentile(g, 5)),
        "p95": float(np.percentile(g, 95)),
        "entropy": float(-(nz * np.log2(nz)).sum()),
        "sobel_mean": float(np.sqrt(sx ** 2 + sy ** 2).mean()),
        "noise_orig": immerkaer_noise(gray_orig),
        "noise_224": immerkaer_noise(g),
        "border_mean": float(gf[border].mean()),
    })
    return feats


def select_rows(manifest):
    """정상은 train/val/test 전부, 폐렴은 split·클래스마다 최대 300장 (seed 42)."""
    rows = [manifest[(manifest.label_name == "NORMAL") & manifest.split.isin(SPLITS)]]
    rng = random.Random(SEED)
    for split in SPLITS:
        for cls in ["BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA"]:
            cand = sorted(manifest[(manifest.split == split) & (manifest.label_name == cls)].relative_path)
            pick = set(rng.sample(cand, min(300, len(cand))))
            rows.append(manifest[manifest.relative_path.isin(pick)])
    return pd.concat(rows).reset_index(drop=True)


def build_features(args):
    manifest = pd.read_csv(args.manifest)
    sel = select_rows(manifest)
    out = []
    for i, r in enumerate(sel.itertuples(index=False), 1):
        path = Path(args.images_dir) / r.relative_path
        name = Path(r.relative_path).name
        pre, num = prefix_of(name)
        f = image_features(path)
        w, h = int(r.width), int(r.height)
        size = path.stat().st_size
        out.append({
            "relative_path": r.relative_path, "split": r.split, "label": r.label_name,
            "patient_group": r.patient_group, "prefix": pre, "number": num,
            "width": w, "height": h, "aspect": w / h, "pixels": w * h,
            "file_bytes": size, "bytes_per_pixel": size / (w * h), "channels": int(r.channels), **f,
        })
        if i % 300 == 0:
            print(f"  속성 계산 {i}/{len(sel)}", flush=True)
    df = pd.DataFrame(out)

    # D 의 예측을 붙인다 (val·test 만 있음). test 는 중복 6장 표시(excluded)도 같이.
    test = pd.read_csv(args.test_pred)
    val = pd.read_csv(args.val_pred)
    pred = pd.concat([
        test[["relative_path", "pred_label", "p_normal", "excluded"]],
        val[["relative_path", "pred_label", "p_normal"]].assign(excluded=False),
    ])
    df = df.merge(pred, on="relative_path", how="left")
    df.loc[df.split == "train", "excluded"] = False
    df["excluded"] = df["excluded"].astype(bool)
    label_id = df.label.map({"NORMAL": 0, "BACTERIAL_PNEUMONIA": 1, "VIRAL_PNEUMONIA": 2})
    df["correct"] = np.where(df.pred_label.notna(), (df.pred_label == label_id).astype(float), np.nan)
    return df


# ----------------------------------------------------------------------------
# 2~4. 분석 도구
# ----------------------------------------------------------------------------
def cliffs_delta(x, y):
    """Cliff's delta = P(x > y) − P(x < y). −1~1. 0 이면 두 분포가 겹침, |d| ≥ 0.474 를 흔히 '큰 차이'로 본다."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic  # 동점은 0.5 로 센다
    return float(2 * u / (len(x) * len(y)) - 1)


def mw_p(x, y):
    return float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)


def med_iqr(s, feat):
    k = SCALE_FOR_TABLE.get(feat, 1.0)
    q1, q2, q3 = np.percentile(s * k, [25, 50, 75])
    return q1, q2, q3


def fmt(v, feat):
    if feat in ("width", "height"):
        return f"{v:.0f}"
    if feat in ("file_bytes", "jpeg_quality"):
        return f"{v:.0f}"
    if feat in ("aspect", "pixels", "bytes_per_pixel"):
        return f"{v:.2f}" if feat != "bytes_per_pixel" else f"{v:.3f}"
    return f"{v:.2f}"


def cv_auroc(X, y, groups):
    """표준화 + 로지스틱 회귀, 같은 환자는 같은 fold (StratifiedGroupKFold 5, seed 42).

    돌려주는 것: fold 별 AUROC 평균·표준편차, 그리고 모든 사진의 out-of-fold 확률 (자기 자신을 보지 않은 예측).
    """
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    aucs = []
    for tr, te in skf.split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], oof[te]))
    return float(np.mean(aucs)), float(np.std(aucs)), float(roc_auc_score(y, oof)), oof


def train_percentile(ref, values):
    """values 가 ref(train 정상) 분포에서 몇 백분위인지 (동점은 절반). 50 이면 한가운데."""
    ref = np.sort(np.asarray(ref, float))
    lo = np.searchsorted(ref, values, side="left")
    hi = np.searchsorted(ref, values, side="right")
    return 100.0 * (lo + hi) / 2.0 / len(ref)


# ----------------------------------------------------------------------------
# 그림
# ----------------------------------------------------------------------------
def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = "Apple SD Gothic Neo"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_distributions(plt, normal, pneu, feats, deltas, out):
    groups = [
        ("train 정상", normal[normal.split == "train"], "#7f7f7f"),
        ("val 정상", normal[normal.split == "val"], "#1f77b4"),
        ("test 정상\n맞힘", normal[(normal.split == "test") & (normal.correct == 1)], "#2ca02c"),
        ("test 정상\n틀림", normal[(normal.split == "test") & (normal.correct == 0)], "#d62728"),
        ("(참고) train\n폐렴 표본", pneu[pneu.split == "train"], "#9467bd"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    rng = np.random.default_rng(SEED)
    for ax, feat in zip(axes.ravel(), feats):
        k = SCALE_FOR_TABLE.get(feat, 1.0)
        data = [g[feat].to_numpy() * k for _, g, _ in groups]
        bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True)
        for patch, (_, _, c) in zip(bp["boxes"], groups):
            patch.set_facecolor(c)
            patch.set_alpha(0.25)
        for i, (d, (_, _, c)) in enumerate(zip(data, groups), 1):
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d, s=5, color=c, alpha=0.35, linewidths=0)
        ax.set_xticks(range(1, len(groups) + 1), [f"{n}\n({len(g)}장)" for n, g, _ in groups], fontsize=9)
        ax.set_title(f"{KO[feat]}\n정상 Cliff's δ train→test {deltas[feat]:+.2f}", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("정상 사진 속성 분포: train / val / test (test 는 D 가 맞힌 사진과 틀린 사진으로 나눔), 오른쪽 끝은 비교용 train 폐렴\n"
                 "δ = P(test > train) - P(test < train). 0 이면 겹침, ±1 이면 완전히 갈림", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_auroc(plt, auroc, out):
    keys = list(FEATURE_SETS)
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for j, (cls, color, label) in enumerate([("normal", "#d62728", "정상 사진"), ("pneumonia", "#1f77b4", "폐렴 사진")]):
        m = [auroc[cls][k]["mean"] for k in keys]
        s = [auroc[cls][k]["sd"] for k in keys]
        bars = ax.bar(x + (j - 0.5) * 0.38, m, 0.36, yerr=s, capsize=4, color=color, alpha=0.8, label=label)
        for b_, v, e in zip(bars, m, s):
            ax.text(b_.get_x() + b_.get_width() / 2, v + e + 0.015, f"{v:.3f}", ha="center", fontsize=10)
    ctrl = auroc["normal_train_vs_val"]["c_all"]["mean"]
    ax.axhline(0.5, color="black", lw=1, ls="--")
    ax.text(len(keys) - 0.5, 0.515, "0.5 = 동전 던지기", ha="right", fontsize=10)
    ax.set_xticks(x, [FEATURE_SETS[k][0] for k in keys], fontsize=10)
    ax.set_ylim(0.3, 1.2)
    ax.set_ylabel("5-fold AUROC (막대 = fold 표준편차)")
    ax.set_title("속성만으로 'train 풀(train+val)인가 test 인가'를 얼마나 맞히나\n"
                 f"(대조: 정상 train vs val, 특징 전부 = {ctrl:.3f})", fontsize=13)
    ax.legend(loc="upper center", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def plot_examples(normal, images_dir, out):
    """train 정상 4장 / test 맞힘 4장 / test 틀림 4장 (seed 42 로 고름), 모델 입력과 같은 224 회색조."""
    rng = random.Random(SEED)
    rows = [
        ("train 정상", normal[normal.split == "train"]),
        ("test 정상 · D 맞힘", normal[(normal.split == "test") & (normal.correct == 1)]),
        ("test 정상 · D 틀림", normal[(normal.split == "test") & (normal.correct == 0)]),
    ]
    tile, pad, left, top, cap = 224 * 2, 14, 240, 110, 58
    cols = 4
    canvas = Image.new("RGB", (left + cols * (tile + pad) + pad, top + len(rows) * (tile + cap + pad) + pad), "white")
    d = ImageDraw.Draw(canvas)
    d.text((pad, 14), "정상 사진 예시: train vs test (모델이 보는 224×224 회색조, 화면용 2배 확대)", fill="black", font=font(30))
    d.text((pad, 58), "칸 아래: 원본 크기 · 파일 크기 · JPEG 품질 추정 · D 의 정상 확률(p_normal). 사진은 seed 42 로 무작위 선택",
           fill="#555555", font=font(19))
    for r, (label, pool) in enumerate(rows):
        picks = rng.sample(sorted(pool.relative_path), cols)
        y = top + r * (tile + cap + pad)
        d.text((pad, y + tile // 2 - 14), label, fill="black", font=font(24))
        for c, rel in enumerate(picks):
            row = pool[pool.relative_path == rel].iloc[0]
            with Image.open(Path(images_dir) / rel) as im:
                g = to_gray_224(im)
            x = left + c * (tile + pad)
            canvas.paste(Image.fromarray(g).convert("RGB").resize((tile, tile), Image.BICUBIC), (x, y))
            p = "" if pd.isna(row.p_normal) else f" · p_normal {row.p_normal:.3f}"
            if row.split == "test" and row.correct == 0:
                p += " → " + ("세균성" if row.pred_label == 1 else "바이러스성")
            d.text((x, y + tile + 4), f"{Path(rel).name}", fill="#333333", font=font(15))
            d.text((x, y + tile + 24), f"{row.width}×{row.height} · {row.file_bytes / 1024:.0f}KB · Q≈{row.jpeg_quality:.0f}{p}",
                   fill="#333333", font=font(15))
    canvas.save(out)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "manifests/split_manifest.csv"))
    ap.add_argument("--images-dir", default=str(Path.home() / "data/chest_xray"))
    ap.add_argument("--test-pred", default=str(ROOT / "reports/test_eval/aws/test_predictions.csv"))
    ap.add_argument("--val-pred", default=str(ROOT / "runs/results/d-20261019-134621/val_predictions.csv"))
    ap.add_argument("--out", default=str(ROOT / "reports/test_eval/normal_shift"))
    ap.add_argument("--reuse-features", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    feat_csv = out / "normal_features.csv"

    if args.reuse_features and feat_csv.exists():
        df = pd.read_csv(feat_csv)
    else:
        print("1. 사진별 속성 계산", flush=True)
        df = build_features(args)
        df.to_csv(feat_csv, index=False)
        print("저장:", feat_csv, flush=True)

    counts = df.groupby(["split", "label"]).size().unstack()
    print(counts)
    assert counts.loc["train", "NORMAL"] == 1075 and counts.loc["val", "NORMAL"] == 266 and counts.loc["test", "NORMAL"] == 234

    # 분석은 test 중복 사본 6장을 뺀 기준 (최종 평가와 같은 618장 기준)
    ana = df[~df.excluded].copy()
    normal = ana[ana.label == "NORMAL"]
    pneu = ana[ana.label != "NORMAL"]
    tn = normal[normal.split == "test"]
    assert len(tn) == 231 and int((tn.correct == 1).sum()) == 102 and int((tn.correct == 0).sum()) == 129
    S = {"counts_analysis": {s: {k: int(v) for k, v in g.label.value_counts().items()} for s, g in ana.groupby("split")}}
    md = []

    # --- 2. 분포 차이 -------------------------------------------------------
    all_feats = FILE_FEATS + META_FEATS + IMAGE_FEATS
    dist_rows, deltas = [], {}
    for feat in all_feats:
        tr, va, te = (normal[normal.split == s][feat].to_numpy(float) for s in SPLITS)
        row = {"feature": feat}
        for s, x in zip(SPLITS, (tr, va, te)):
            row[f"{s}_q1"], row[f"{s}_median"], row[f"{s}_q3"] = med_iqr(x, feat)
            row[f"{s}_mean"] = float(np.mean(x)) * SCALE_FOR_TABLE.get(feat, 1.0)
        row["delta_train_test"] = cliffs_delta(te, tr)
        row["delta_train_val"] = cliffs_delta(va, tr)
        row["delta_val_test"] = cliffs_delta(te, va)
        row["p_train_test"] = mw_p(te, tr)
        deltas[feat] = row["delta_train_test"]
        dist_rows.append(row)
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(out / "distribution_normal.csv", index=False)
    S["distribution_normal"] = dist.set_index("feature").to_dict(orient="index")

    md.append("## 분포 차이 (정상, 중앙값 [Q1, Q3], Cliff's δ)\n")
    md.append("| 속성 | train (1,075) | val (266) | test (231) | δ train→test | δ train→val | δ val→test |")
    md.append("|---|---|---|---|---|---|---|")
    for r in dist_rows:
        f = r["feature"]
        if f in META_FEATS + ["channels"]:
            cells = [f"{r[f'{s}_mean']:.1%}" for s in SPLITS]  # 0/1 속성은 비율로
        else:
            cells = [f"{fmt(r[f'{s}_median'], f)} [{fmt(r[f'{s}_q1'], f)}, {fmt(r[f'{s}_q3'], f)}]" for s in SPLITS]
        md.append(f"| {KO[f]} | " + " | ".join(cells)
                  + f" | {r['delta_train_test']:+.2f} | {r['delta_train_val']:+.2f} | {r['delta_val_test']:+.2f} |")
    md.append("\n채널 수·메타데이터 표식은 비율(평균)로 적었다. 채널 수 열의 값은 평균 채널 수다.\n")

    # 폐렴도 같은 표 (train vs test δ 만)
    md.append("## 참고: 폐렴 사진의 train→test δ (같은 속성)\n")
    md.append("| 속성 | 정상 δ train→test | 폐렴 δ train→test | 폐렴 δ train→val |")
    md.append("|---|---|---|---|")
    S["delta_pneumonia"] = {}
    for f in all_feats:
        tr, va, te = (pneu[pneu.split == s][f].to_numpy(float) for s in SPLITS)
        dp, dv = cliffs_delta(te, tr), cliffs_delta(va, tr)
        S["delta_pneumonia"][f] = {"train_test": dp, "train_val": dv}
        md.append(f"| {KO[f]} | {deltas[f]:+.2f} | {dp:+.2f} | {dv:+.2f} |")

    # test 정상은 train 의 어느 쪽을 닮았나: train 정상 vs train 폐렴 vs test 정상 (중앙값, 0/1 속성은 비율)
    md.append("\n## test 정상은 train 의 정상과 폐렴 중 어느 쪽을 닮았나 (중앙값, 0/1 속성은 비율)\n")
    groups = [("train 정상", normal[normal.split == "train"]),
              ("train 세균성(표본)", pneu[(pneu.split == "train") & (pneu.label == "BACTERIAL_PNEUMONIA")]),
              ("train 바이러스성(표본)", pneu[(pneu.split == "train") & (pneu.label == "VIRAL_PNEUMONIA")]),
              ("test 정상", tn),
              ("test 정상 맞힘", tn[tn.correct == 1]), ("test 정상 틀림", tn[tn.correct == 0])]
    md.append("| 속성 | " + " | ".join(f"{n} ({len(g)})" for n, g in groups) + " |")
    md.append("|---|" + "---|" * len(groups))
    S["lookalike"] = {}
    for f in ["aspect", "pixels", "file_bytes", "bytes_per_pixel", "jpeg_quality", "has_photoshop",
              "noise_orig", "noise_224", "sobel_mean"]:
        k = SCALE_FOR_TABLE.get(f, 1.0)
        vals = [(float(g[f].mean()) if f in META_FEATS else float(g[f].median() * k)) for _, g in groups]
        S["lookalike"][f] = dict(zip([n for n, _ in groups], vals))
        cells = [f"{v:.1%}" if f in META_FEATS else fmt(v, f) for v in vals]
        md.append(f"| {KO[f]} | " + " | ".join(cells) + " |")

    # 파일명 접두어·번호 범위
    md.append("\n## 파일명 접두어와 번호 범위 (정상)\n")
    md.append("| split | IM- 장수 (번호 범위) | NORMAL2-IM- 장수 (번호 범위) | IM- 비율 |")
    md.append("|---|---|---|---|")
    S["prefix"] = {}
    for s in SPLITS:
        g = df[(df.label == "NORMAL") & (df.split == s)]  # 범위는 중복 포함 전체로 본다
        rec = {}
        for p in ["IM-", "NORMAL2-IM-"]:
            gp = g[g.prefix == p]
            rec[p] = {"n": int(len(gp)), "min": int(gp.number.min()), "max": int(gp.number.max())}
        rec["im_ratio"] = rec["IM-"]["n"] / len(g)
        S["prefix"][s] = rec
        md.append(f"| {s} ({len(g)}) | {rec['IM-']['n']} ({rec['IM-']['min']}~{rec['IM-']['max']}) | "
                  f"{rec['NORMAL2-IM-']['n']} ({rec['NORMAL2-IM-']['min']}~{rec['NORMAL2-IM-']['max']}) | {rec['im_ratio']:.1%} |")
    pool_nums = {p: set(df[(df.label == "NORMAL") & df.split.isin(["train", "val"]) & (df.prefix == p)].number) for p in ["IM-", "NORMAL2-IM-"]}
    test_nums = {p: set(df[(df.label == "NORMAL") & (df.split == "test") & (df.prefix == p)].number) for p in ["IM-", "NORMAL2-IM-"]}
    S["prefix_number_overlap_pool_test"] = {p: len(pool_nums[p] & test_nums[p]) for p in pool_nums}
    md.append(f"\ntrain 풀과 test 에서 같은 접두어·같은 번호인 정상 환자 수: "
              f"IM- {S['prefix_number_overlap_pool_test']['IM-']}, NORMAL2-IM- {S['prefix_number_overlap_pool_test']['NORMAL2-IM-']}\n")

    # --- 3. 식별 가능성 ---------------------------------------------------------
    print("3. 식별 가능성 (5-fold 로지스틱 회귀)", flush=True)
    auroc = {"normal": {}, "pneumonia": {}, "normal_train_vs_val": {}}
    oof_normal = None
    for name, sub, target in [
        ("normal", normal, normal.split == "test"),
        ("pneumonia", pneu, pneu.split == "test"),
        ("normal_train_vs_val", normal[normal.split != "test"], normal[normal.split != "test"].split == "val"),
    ]:
        y = target.to_numpy().astype(int)
        for key, (_, feats) in FEATURE_SETS.items():
            m, sd, pooled, oof = cv_auroc(sub[feats].to_numpy(float), y, sub.patient_group.to_numpy())
            auroc[name][key] = {"mean": m, "sd": sd, "pooled": pooled, "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
            if name == "normal" and key == "c_all":
                oof_normal = pd.Series(oof, index=sub.index)
    S["auroc"] = auroc
    md.append("## 식별 가능성 AUROC (5-fold, 환자 단위로 fold 를 나눔, 평균 ± 표준편차)\n")
    md.append("| 특징 묶음 | 정상: train 풀 vs test | 폐렴: train 풀 vs test | 대조: 정상 train vs val |")
    md.append("|---|---|---|---|")
    for key, (label, _) in FEATURE_SETS.items():
        cells = [f"{auroc[n][key]['mean']:.3f} ± {auroc[n][key]['sd']:.3f}" for n in ["normal", "pneumonia", "normal_train_vs_val"]]
        md.append(f"| {label.replace(chr(10), ' ')} | " + " | ".join(cells) + " |")
    a = auroc["normal"]["a_file"]
    b = auroc["pneumonia"]["a_file"]
    md.append(f"\n표본: 정상 train 풀 {a['n_neg']} / test {a['n_pos']}, 폐렴 train 풀 {b['n_neg']} (표본) / test {b['n_pos']}.\n")

    # 보조: 파일 속성만으로 정상 vs 폐렴 라벨을 맞히나 (train 풀에서 학습 → test 에서 확인)
    pool = ana[ana.split != "test"]
    test_all = ana[ana.split == "test"]
    label_sets = {k: FEATURE_SETS[k] for k in ["a_file", "a2_file_meta", "b_image", "b2_image224"]}
    label_sets["shape"] = ("(s) 크기·모양만 (가로·세로·종횡비·픽셀 수)", SHAPE_FEATS)
    label_auc, normal_score = {}, {}
    y_pool = (pool.label == "NORMAL").to_numpy().astype(int)
    y_test = (test_all.label == "NORMAL").to_numpy().astype(int)
    for key, (_, feats) in label_sets.items():
        m, sd, _, _ = cv_auroc(pool[feats].to_numpy(float), y_pool, pool.patient_group.to_numpy())
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(pool[feats].to_numpy(float), y_pool)
        p_test = clf.predict_proba(test_all[feats].to_numpy(float))[:, 1]
        label_auc[key] = {"pool_cv_mean": m, "pool_cv_sd": sd, "test": float(roc_auc_score(y_test, p_test))}
        normal_score[key] = pd.Series(p_test, index=test_all.index)  # "train 기준 정상다움" (test 사진마다)
    S["label_from_attributes"] = label_auc
    md.append("## 보조: 속성만으로 '정상 vs 폐렴' 라벨을 맞히나 (train 풀에서 학습 → test 에 적용)\n")
    md.append("| 특징 묶음 | train 풀 안 5-fold AUROC | test AUROC |")
    md.append("|---|---|---|")
    for key, v in label_auc.items():
        md.append(f"| {label_sets[key][0].replace(chr(10), ' ')} | {v['pool_cv_mean']:.3f} ± {v['pool_cv_sd']:.3f} | {v['test']:.3f} |")
    md.append("\n폐렴은 split·클래스마다 최대 300장 표본이라 정상:폐렴 비율이 실제와 다르다 (AUROC 는 비율에 덜 민감).\n")

    # --- 4. 오분류와의 관련 ------------------------------------------------------
    print("4. 오분류와의 관련", flush=True)
    trn = normal[normal.split == "train"]
    ok, bad = tn[tn.correct == 1], tn[tn.correct == 0]
    mis_rows = []
    for feat in all_feats:
        pct = train_percentile(trn[feat], tn[feat].to_numpy(float))
        dev = np.abs(pct - 50.0)  # train 정상 한가운데(50 백분위)에서 얼마나 떨어졌나
        if tn[feat].nunique() > 1:
            rho_v = stats.spearmanr(tn[feat], tn.p_normal)
            rho_d = stats.spearmanr(dev, tn.p_normal)
        else:  # test 정상에서 값이 하나뿐 (JPEG 품질·메타데이터·채널): 상관을 정의할 수 없음
            rho_v = rho_d = SimpleNamespace(statistic=np.nan, pvalue=np.nan)
        mis_rows.append({
            "feature": feat,
            "ok_median": float(np.median(ok[feat])) * SCALE_FOR_TABLE.get(feat, 1.0),
            "bad_median": float(np.median(bad[feat])) * SCALE_FOR_TABLE.get(feat, 1.0),
            "ok_mean": float(np.mean(ok[feat])), "bad_mean": float(np.mean(bad[feat])),
            "delta_bad_vs_ok": cliffs_delta(bad[feat], ok[feat]),
            "p_bad_vs_ok": mw_p(bad[feat], ok[feat]) if ok[feat].nunique() + bad[feat].nunique() > 2 else np.nan,
            "ok_train_pct_median": float(np.median(pct[(tn.correct == 1).to_numpy()])),
            "bad_train_pct_median": float(np.median(pct[(tn.correct == 0).to_numpy()])),
            "spearman_value_pnormal": float(rho_v.statistic), "p_value_pnormal": float(rho_v.pvalue),
            "spearman_dev_pnormal": float(rho_d.statistic), "p_dev_pnormal": float(rho_d.pvalue),
        })
    mis = pd.DataFrame(mis_rows)
    mis.to_csv(out / "misclassification_normal_test.csv", index=False)
    S["misclassification"] = mis.set_index("feature").to_dict(orient="index")
    md.append("## test 정상: D 가 맞힌 102장 vs 틀린 129장\n")
    md.append("δ 는 P(틀림 > 맞힘) − P(틀림 < 맞힘). 'train 백분위'는 train 정상 1,075장 중 그 값보다 작은 사진의 비율(%). "
              "ρ 는 test 정상 231장에서 p_normal 과의 스피어먼 상관 (값 자체 / train 한가운데에서 떨어진 정도 |백분위−50|).\n")
    md.append("| 속성 | 맞힘 중앙값 | 틀림 중앙값 | δ 틀림 vs 맞힘 (p) | train 백분위 중앙값 맞힘 / 틀림 | ρ(값, p_normal) | ρ(거리, p_normal) |")
    md.append("|---|---|---|---|---|---|---|")
    for r in mis_rows:
        f = r["feature"]
        if f in META_FEATS + ["channels"]:
            o, b_ = f"{r['ok_mean']:.1%}", f"{r['bad_mean']:.1%}"
        else:
            o, b_ = fmt(r["ok_median"], f), fmt(r["bad_median"], f)
        p = "" if np.isnan(r["p_bad_vs_ok"]) else f" ({r['p_bad_vs_ok']:.3g})"
        if np.isnan(r["spearman_value_pnormal"]):
            rho_txt = "값이 하나뿐 | 값이 하나뿐"
        else:
            rho_txt = (f"{r['spearman_value_pnormal']:+.2f} ({r['p_value_pnormal']:.3g}) | "
                       f"{r['spearman_dev_pnormal']:+.2f} ({r['p_dev_pnormal']:.3g})")
        md.append(f"| {KO[f]} | {o} | {b_} | {r['delta_bad_vs_ok']:+.2f}{p} | "
                  f"{r['ok_train_pct_median']:.0f} / {r['bad_train_pct_median']:.0f} | {rho_txt} |")

    # 종합 거리 1: 식별 분류기(c)의 out-of-fold "test 다움" 확률
    tl = oof_normal.loc[tn.index]
    rho = stats.spearmanr(tl, tn.p_normal)
    auc_wrong = roc_auc_score((tn.correct == 0).astype(int), tl)
    S["test_likeness"] = {"spearman_pnormal": float(rho.statistic), "p": float(rho.pvalue),
                          "auroc_predict_wrong": float(auc_wrong),
                          "median_ok": float(tl[tn.correct == 1].median()), "median_bad": float(tl[tn.correct == 0].median())}
    md.append(f"\n종합 거리 1 (train 풀과 얼마나 다른가): 식별 분류기 (c) 의 out-of-fold 'test 다움' 확률. "
              f"맞힘 중앙값 {S['test_likeness']['median_ok']:.3f}, 틀림 중앙값 {S['test_likeness']['median_bad']:.3f}, "
              f"틀림을 가려내는 AUROC {auc_wrong:.3f}, p_normal 과 ρ {rho.statistic:+.2f} (p={rho.pvalue:.3g}). "
              "test 정상이 거의 모두 1 에 붙어 있어(완전 구분) 맞힘·틀림을 가르지 못한다.\n")

    # 종합 거리 2: train 풀에서 '정상 vs 폐렴'을 배운 속성 분류기가 test 정상을 얼마나 '정상답다'고 보나
    md.append("종합 거리 2 (train 의 정상과 폐렴 중 어느 쪽을 닮았나): 위 보조 분류기의 정상 확률을 test 정상 231장에 적용.\n")
    md.append("| 특징 묶음 | 정상다움 중앙값 맞힘 / 틀림 | 틀림을 가려내는 AUROC (정상다움이 낮을수록 틀림) | ρ(정상다움, p_normal) |")
    md.append("|---|---|---|---|")
    S["normal_likeness_test_normal"] = {}
    for key, sc in normal_score.items():
        s_tn = sc.loc[tn.index]
        a_w = roc_auc_score((tn.correct == 0).astype(int), -s_tn)
        r_ = stats.spearmanr(s_tn, tn.p_normal)
        S["normal_likeness_test_normal"][key] = {
            "median_ok": float(s_tn[tn.correct == 1].median()), "median_bad": float(s_tn[tn.correct == 0].median()),
            "auroc_wrong": float(a_w), "spearman_pnormal": float(r_.statistic), "p": float(r_.pvalue)}
        md.append(f"| {label_sets[key][0].replace(chr(10), ' ')} | {s_tn[tn.correct == 1].median():.3f} / {s_tn[tn.correct == 0].median():.3f} | "
                  f"{a_w:.3f} | {r_.statistic:+.2f} (p={r_.pvalue:.3g}) |")

    # 종횡비 구간별 정상 recall (구간 경계는 train 에서 정함: train 정상 Q3, train 폐렴 중앙값)
    cut1 = float(np.percentile(trn.aspect, 75))
    cut2 = float(pneu[pneu.split == "train"].aspect.median())
    bins = [("train 정상 Q3 이하", tn.aspect <= cut1), ("그 사이", (tn.aspect > cut1) & (tn.aspect < cut2)),
            ("train 폐렴 중앙값 이상", tn.aspect >= cut2)]
    S["recall_by_aspect"] = {"cut_train_normal_q3": cut1, "cut_train_pneumonia_median": cut2, "bins": {}}
    md.append(f"\n종횡비 구간별 test 정상 recall (경계: train 정상 Q3 = {cut1:.3f}, train 폐렴 표본 중앙값 = {cut2:.3f})\n")
    md.append("| 구간 | 장수 | recall |")
    md.append("|---|---|---|")
    for name, mask in bins:
        g = tn[mask]
        S["recall_by_aspect"]["bins"][name] = {"n": int(len(g)), "recall": float(g.correct.mean())}
        md.append(f"| {name} | {len(g)} | {g.correct.mean():.3f} ({int(g.correct.sum())}/{len(g)}) |")

    # 접두어 안에서도 같은 경향인가
    md.append("\n접두어 안에서 맞힘 / 틀림 중앙값 (종횡비, 픽셀 수 백만)\n")
    md.append("| 접두어 | 맞힘 종횡비 | 틀림 종횡비 | 맞힘 픽셀 수 | 틀림 픽셀 수 |")
    md.append("|---|---|---|---|---|")
    S["within_prefix"] = {}
    for p in ["IM-", "NORMAL2-IM-"]:
        g = tn[tn.prefix == p]
        o, b_ = g[g.correct == 1], g[g.correct == 0]
        S["within_prefix"][p] = {"ok_aspect": float(o.aspect.median()), "bad_aspect": float(b_.aspect.median()),
                                 "ok_mp": float(o.pixels.median() / 1e6), "bad_mp": float(b_.pixels.median() / 1e6)}
        md.append(f"| {p} | {o.aspect.median():.2f} | {b_.aspect.median():.2f} | {o.pixels.median() / 1e6:.2f} | {b_.pixels.median() / 1e6:.2f} |")
    md.append("")

    # 접두어별 정상 recall (val·test)
    md.append("## 접두어별 정상 recall (D)\n")
    md.append("| split | IM- | NORMAL2-IM- |")
    md.append("|---|---|---|")
    S["recall_by_prefix"] = {}
    for s in ["val", "test"]:
        g = normal[normal.split == s]
        rec = {}
        for p in ["IM-", "NORMAL2-IM-"]:
            gp = g[g.prefix == p]
            rec[p] = {"n": int(len(gp)), "correct": int(gp.correct.sum()), "recall": float(gp.correct.mean())}
        S["recall_by_prefix"][s] = rec
        md.append(f"| {s} | {rec['IM-']['recall']:.3f} ({rec['IM-']['correct']}/{rec['IM-']['n']}) | "
                  f"{rec['NORMAL2-IM-']['recall']:.3f} ({rec['NORMAL2-IM-']['correct']}/{rec['NORMAL2-IM-']['n']}) |")

    (out / "normal_shift_tables.md").write_text("# 자동 생성 표 (src/normal_shift_analysis.py)\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    (out / "normal_shift_summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("저장:", out / "normal_shift_tables.md", flush=True)

    # --- 5. 그림 ----------------------------------------------------------------
    plt = setup_plt()
    # 그림 속성: 저장 방식 3개(품질·바이트/픽셀·원본 노이즈) + 모델이 보는 224 노이즈 + 오분류와 관련된 크기·모양 2개
    plot_distributions(plt, normal, pneu, FIG_FEATS, deltas, out / "normal_shift_distributions.png")
    plot_auroc(plt, auroc, out / "normal_shift_auroc.png")
    plot_examples(normal, args.images_dir, out / "normal_shift_examples.png")
    print("그림 저장 완료", flush=True)


if __name__ == "__main__":
    main()
