"""P 시리즈 1단계: 전처리 P0~P4 를 적용한 사진 갤러리와 영상 통계 (iMac 전용, 학습 없음).

train split 사진만 쓴다 (val·check·test 는 열지 않는다). 사진 선택은 seed 로 고정한다.
전처리 정의는 training/preprocess.py (학습 코드와 같은 함수)를 그대로 쓴다.

사용 예 (p1 폴더에서):
    .venv/bin/python src/preprocess_gallery.py --out reports/preprocess

결과:
    preprocess_gallery.png  행 = 정상·세균성·바이러스성 각 2장, 열 = P0~P4 (모델이 보는 224×224 회색조)
    preprocess_stats.csv    train 사진 클래스당 50장, 전처리별·클래스별 평균과 표준편차
        rms_contrast  RMS 대비 = 픽셀 표준편차 / 255 (클수록 밝고 어두운 차이가 큼)
        entropy_bits  엔트로피 (8비트 히스토그램 256칸, 단위 비트. 클수록 밝기 값이 고르게 퍼짐)
        sobel_mean    Sobel 기울기 크기의 평균 (0~255 기준. 클수록 경계·미세 무늬가 많음)
        brightness    평균 밝기 (0~255)
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))
from preprocess import PREPROCESS, apply_preprocess  # noqa: E402
from shift_filters import to_gray_224  # noqa: E402
from shift_gallery import CLASS_KO, CLASS_NAMES, draw_grid  # noqa: E402  (그림 스타일·한글 폰트 재사용)

NAMES = list(PREPROCESS)  # P0~P4
STATS = ["rms_contrast", "entropy_bits", "sobel_mean", "brightness"]


def train_rows(manifest_path):
    """split_manifest.csv 에서 train 행만 남긴다 (val·check·test 는 여기서 버린다)."""
    rows = [r for r in csv.DictReader(open(manifest_path, encoding="utf-8")) if r["split"] == "train"]
    assert rows and all(r["split"] == "train" for r in rows)
    return rows


def pick(rows, n_per_class, seed):
    """클래스마다 n 장을 seed 로 고정해 고른다."""
    rng = random.Random(seed)
    out = []
    for name in CLASS_NAMES:
        cands = sorted(r["relative_path"] for r in rows if r["label_name"] == name)
        out += [(name, rel) for rel in rng.sample(cands, n_per_class)]
    return out


def image_stats(img):
    x = img.astype(np.float32)
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    return {
        "rms_contrast": float(x.std() / 255.0),
        "entropy_bits": float(-(p * np.log2(p)).sum()),
        "sobel_mean": float(np.sqrt(gx ** 2 + gy ** 2).mean()),
        "brightness": float(x.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "manifests/split_manifest.csv"))
    ap.add_argument("--images-dir", default=str(Path.home() / "data/chest_xray"))
    ap.add_argument("--out", default=str(ROOT / "reports/preprocess"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-stats", type=int, default=50, help="통계용 클래스당 사진 수")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = train_rows(args.manifest)

    def load(rel):
        with Image.open(Path(args.images_dir) / rel) as im:
            return to_gray_224(im)  # 회색조 → 224×224 (전처리는 이 해상도에서 적용)

    # 1) 갤러리: 클래스당 2장 × P0~P4
    picked = pick(rows, 2, args.seed)
    grays = [load(rel) for _, rel in picked]
    imgs = [[apply_preprocess(g, n) for n in NAMES] for g in grays]
    row_labels = [f"{CLASS_KO[c]}\n{Path(rel).name}" for c, rel in picked]
    col_labels = [PREPROCESS[n]["title"] for n in NAMES]
    draw_grid("전처리 P0~P4 비교 (train 사진)",
              "모두 224×224 회색조에서 적용 (그림은 보기 좋게 2배 확대). "
              "P1 HE / P2 CLAHE clip 2.0, 8×8 / P3 가우시안 σ 1.5 / P4 언샤프 σ 2, α 1.0",
              row_labels, col_labels, imgs).save(out / "preprocess_gallery.png")
    print("저장:", out / "preprocess_gallery.png", flush=True)

    # 2) 통계: 클래스당 n 장 (갤러리와 다른 seed 로 따로 뽑음)
    stat_pick = pick(rows, args.n_stats, args.seed + 1)
    values = {(n, c): [] for n in NAMES for c in CLASS_NAMES}
    for c, rel in stat_pick:
        g = load(rel)
        for n in NAMES:
            values[(n, c)].append(image_stats(apply_preprocess(g, n)))
    with open(out / "preprocess_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["preprocess", "preprocess_desc", "class", "n"] + [f"{s}_{k}" for s in STATS for k in ("mean", "std")])
        for n in NAMES:
            for c in CLASS_NAMES:
                vs = values[(n, c)]
                cells = []
                for s in STATS:
                    arr = np.array([v[s] for v in vs])
                    cells += [f"{arr.mean():.4f}", f"{arr.std(ddof=1):.4f}"]
                w.writerow([n, PREPROCESS[n]["desc"], c, len(vs)] + cells)
                print(f"  {n} {CLASS_KO[c]:5s} " + "  ".join(
                    f"{s} {np.mean([v[s] for v in vs]):.3f}±{np.std([v[s] for v in vs], ddof=1):.3f}" for s in STATS),
                    flush=True)
    print("저장:", out / "preprocess_stats.csv", f"(train 사진 클래스당 {args.n_stats}장)", flush=True)

    # 확인: 같은 입력 → 같은 출력, 크기·형식
    for n in NAMES:
        a, b = apply_preprocess(grays[0], n), apply_preprocess(grays[0], n)
        assert np.array_equal(a, b) and a.shape == (224, 224) and a.dtype == np.uint8, n
    print("확인: P0~P4 모두 결정적이고 224×224 uint8 을 돌려줌", flush=True)


if __name__ == "__main__":
    main()
