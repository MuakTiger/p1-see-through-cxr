"""배포용 묶음(deploy/b512)을 로컬에서 Endpoint와 같은 함수 순서로 돌려, 학습 평가 결과와 확률이 같은지 검산한다.

model_fn → input_fn → predict_fn 을 val 사진 몇 장에 그대로 호출하고, runs/results/<실험>/val_predictions.csv 의 확률과 비교한다.
AWS 자원은 만들지 않는다.

사용 예 (p1 폴더에서): .venv/bin/python deploy/verify_local.py --n 60
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(ROOT / "deploy/b512"))
    ap.add_argument("--val-pred", default=str(ROOT / "runs/results/b512-20261022-111844/val_predictions.csv"))
    ap.add_argument("--images", default=str(Path.home() / "data/chest_xray"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("inference", Path(args.bundle) / "code/inference.py")
    inf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inf)
    bundle = inf.model_fn(args.bundle)

    v = pd.read_csv(args.val_pred)
    pick = v.groupby("true_label", group_keys=False).apply(lambda g: g.sample(args.n // 3, random_state=args.seed))
    diffs, same, skipped = [], 0, 0
    for r in pick.itertuples():
        body = (Path(args.images) / r.relative_path).read_bytes()
        try:
            tensor = inf.input_fn(body, "image/jpeg")
        except ValueError as e:
            skipped += 1
            print("건너뜀:", r.relative_path, e)
            continue
        out = inf.predict_fn(tensor, bundle)
        p = np.array([out["probabilities"][c] for c in inf.CLASSES])
        ref = np.array([r.p_normal, r.p_bacterial, r.p_viral])
        diffs.append(np.abs(p - ref).max())
        same += int(np.argmax(p) == r.pred_label)
    n = len(diffs)
    print(f"검산 {n}장 (건너뜀 {skipped}장): 확률 최대 차이 {max(diffs):.2e}, 평균 {np.mean(diffs):.2e}, 예측 일치 {same}/{n}")
    print("입력 크기:", inf.INPUT_SIZE, "/ 모델 ID:", bundle["metadata"]["exp_id"], "/ 선택 에폭:", bundle["metadata"]["selected_epoch"])


if __name__ == "__main__":
    main()
