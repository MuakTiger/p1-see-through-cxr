"""입력 크기별로 원본 val 사진의 노이즈 추정(Immerkaer, 0~255 기준) 평균을 잰다 (선량 필터의 기준 노이즈 SIGMA0 근거).

선량 필터는 "기준 선량(=원본)의 노이즈가 이 값이고, 선량이 d배가 되면 총 노이즈가 1/√d배" 라는 식으로 만든다.
그래서 사진 크기마다 원본의 노이즈 추정을 같은 방법으로 재야 한다. val 사진 200장 (src/filter_effects.py 와 같은 선택, seed 42).

사용 예 (p1 폴더에서):
    .venv/bin/python src/noise_anchor.py --sizes 224 448 512
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
from normal_shift_analysis import immerkaer_noise  # noqa: E402
from shift_filters import to_gray  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 448, 512])
    ap.add_argument("--manifest", default=str(ROOT / "manifests/split_manifest.csv"))
    ap.add_argument("--images", default=str(Path.home() / "data/chest_xray"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--calibrate", action="store_true", help="선량 필터의 노이즈 배율이 원리값에 맞도록 보정계수를 계산")
    args = ap.parse_args()
    m = pd.read_csv(args.manifest)
    val = m[m.split == "val"]
    rows = val.iloc[np.random.default_rng(42).choice(len(val), size=args.n, replace=False)]
    acc = {s: [] for s in args.sizes}
    for rel in rows.relative_path:
        with Image.open(Path(args.images).expanduser() / rel) as im:
            for s in args.sizes:
                acc[s].append(immerkaer_noise(to_gray(im, s)))
    for s in args.sizes:
        print(f"입력 {s}: 원본 노이즈 추정 평균 {np.mean(acc[s]):.3f} (중앙값 {np.median(acc[s]):.3f}, 사진 {args.n}장)")
    if args.calibrate:
        # 선량 1/16에서 노이즈 추정 배율이 원리값 4.0(=1/√(1/16))이 되도록 기준 노이즈를 보정한다.
        # 분산이 더해진다고 보면 배율² = 1 + f²·(보정 전 배율² - 1) 이므로 f = √(15 / (보정 전 배율² - 1)). 모델 점수는 쓰지 않는다.
        from equipment_direct import dose
        for s in args.sizes:
            ratios = []
            for k, rel in enumerate(rows.relative_path):
                with Image.open(Path(args.images).expanduser() / rel) as im:
                    g = to_gray(im, s)
                d = g.astype(np.float32) / 255.0
                noisy = np.clip(np.rint(dose(d, 1 / 16, 42 + k, float(np.mean(acc[s]))) * 255.0), 0, 255).astype(np.uint8)
                ratios.append(immerkaer_noise(noisy) / max(immerkaer_noise(g), 1e-6))
            m0 = float(np.mean(ratios))
            f = float(np.sqrt(15.0 / (m0 ** 2 - 1.0)))
            print(f"입력 {s}: 선량 1/16 보정 전 노이즈 배율 {m0:.2f} → 보정계수 f = {f:.3f}, 사용할 기준 노이즈 = {f * np.mean(acc[s]):.3f}")


if __name__ == "__main__":
    main()
