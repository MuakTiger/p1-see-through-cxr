"""Grad-CAM 해상도 비교: B(입력 224, 지도 7×7) vs B448(입력 448, 지도 14×14). val 사진만 쓴다.

질문
  1. 지도가 더 세밀해졌나?   → 활성 면적(지도 ≥ 0.5 인 픽셀 비율), 흉부 상자 안 비율
  2. 두 모델이 같은 곳을 보나? → 같은 사진에서 두 지도의 상관
  3. 노이즈(std 16)를 넣어도 지도가 유지되나? → 원본↔노이즈 지도 상관 (gradcam.py Q2 와 같은 방식: B 0.279, D 0.644)

규칙: 재학습·임계값 조정·AWS 없음. test 는 열지 않는다. 지도는 둘 다 224×224 로 키워서 같은 기준(gradcam.py 의 영역 정의)으로 잰다.
지도의 "정확도"(병변 위치와 맞는가)는 병변 표시가 없어서 잴 수 없다. 여기서 잰 것은 세밀함과 안정성이다.
구현은 gradcam.py 의 GradCAM 을 그대로 쓴다 (입력이 448 이면 layer4 출력이 14×14 가 된다).

사용 예 (p1 폴더에서):
    .venv/bin/python src/gradcam_resolution.py
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))
from gradcam import (GradCAM, SIZE, overlay, pearson_rows, region_fractions,  # noqa: E402
                     setup_matplotlib)
from shift_eval import gray_to_tensor, load_model  # noqa: E402
from shift_filters import apply_shift, to_gray  # noqa: E402

MODELS = {
    "B224": (ROOT / "runs/results/b-20261019-103853/model", 224),
    "B448": (ROOT / "runs/results/b448-20261020-161517/model", 448),
}
NOISE_LEVEL = 3  # std 16


def run_model(cam_model, size, rows, images_dir, batch=16):
    """val 전체의 (원본 지도, 노이즈 지도, 예측 클래스) 를 돌려준다. 지도는 224×224."""
    n = len(rows)
    cams = {"clean": np.zeros((n, SIZE, SIZE), np.float32), "noise": np.zeros((n, SIZE, SIZE), np.float32)}
    pred = {"clean": np.zeros(n, int), "noise": np.zeros(n, int)}
    for s0 in range(0, n, batch):
        part = rows[s0:s0 + batch]
        grays = []
        for r in part:
            with Image.open(Path(images_dir) / r["relative_path"]) as im:
                grays.append(to_gray(im, size))
        grays = np.stack(grays)
        noisy = np.stack([apply_shift(g, "noise", NOISE_LEVEL, seed=42 + s0 + j) for j, g in enumerate(grays)])
        for key, arr in (("clean", grays), ("noise", noisy)):
            logits, (cam,), _ = cam_model(gray_to_tensor(arr), [None])
            cams[key][s0:s0 + len(part)] = cam
            pred[key][s0:s0 + len(part)] = logits.argmax(1).numpy()
        print(f"    {min(s0 + batch, n)}/{n}", flush=True)
    return cams, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "manifests/split_manifest.csv"))
    ap.add_argument("--images-dir", default=str(Path.home() / "data/chest_xray"))
    ap.add_argument("--out", default=str(ROOT / "reports/gradcam_resolution"))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [r for r in csv.DictReader(open(args.manifest, encoding="utf-8")) if r["split"] == "val"]
    assert rows and all(r["split"] == "val" for r in rows)
    y = np.array([int(r["label_id"]) for r in rows])
    print(f"val {len(rows)}장", flush=True)

    res = {}
    for name, (mdir, size) in MODELS.items():
        print(f"  {name} (입력 {size})", flush=True)
        res[name] = run_model(GradCAM(load_model(mdir)), size, rows, args.images_dir)

    summary = {}
    for name, (cams, pred) in res.items():
        fc, fn = region_fractions(cams["clean"]), region_fractions(cams["noise"])
        corr = pearson_rows(cams["clean"], cams["noise"])
        summary[name] = {
            "활성면적_원본": float(np.nanmean(fc["active"])),
            "흉부상자안비율_원본": float(np.nanmean(fc["center"])),
            "가장자리비율_원본": float(np.nanmean(fc["border"])),
            "원본_노이즈_지도상관_평균": float(np.nanmean(corr)),
            "원본_노이즈_지도상관_중앙값": float(np.nanmedian(corr)),
            "노이즈에서_바이러스예측_비율": float((pred["noise"] == 2).mean()),
            "원본_바이러스예측_비율": float((pred["clean"] == 2).mean()),
            "원본에서_바이러스로_바뀐_사진_상관": float(np.nanmean(corr[(pred["noise"] == 2) & (pred["clean"] != 2)])),
        }
    cross = pearson_rows(res["B224"][0]["clean"], res["B448"][0]["clean"])
    summary["B224_vs_B448_원본_지도상관_평균"] = float(np.nanmean(cross))
    summary["지도 대비 평균 최대-최소 간격(원본, 사진별 지도 표준편차 평균)"] = {
        k: float(res[k][0]["clean"].reshape(len(rows), -1).std(1).mean()) for k in res}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 예시: 클래스마다 무작위 1장씩 + 하나 더 (seed 42)
    rng = np.random.default_rng(42)
    idx = [int(rng.choice(np.where(y == c)[0])) for c in (0, 1, 2)] + [int(rng.integers(len(rows)))]
    plt = setup_matplotlib()
    fig, axes = plt.subplots(len(idx), 5, figsize=(17, 3.5 * len(idx)))
    for row, i in enumerate(idx):
        with Image.open(Path(args.images_dir) / rows[i]["relative_path"]) as im:
            g = to_gray(im, SIZE)
        axes[row, 0].imshow(g, cmap="gray"); axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        axes[row, 0].set_title(f"val 사진 {i} ({['정상', '세균성', '바이러스성'][y[i]]})", fontsize=10)
        for col, (name, key, lab) in enumerate([("B224", "clean", "B 224 · 7×7 · 원본"), ("B448", "clean", "B448 · 14×14 · 원본"),
                                                ("B224", "noise", "B 224 · 노이즈 std16"), ("B448", "noise", "B448 · 노이즈 std16")], 1):
            cams, pred = res[name]
            overlay(axes[row, col], g, cams[key][i], f"{lab}\n예측: {['정상', '세균성', '바이러스성'][pred[key][i]]}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "gradcam_resolution_examples.png", dpi=120)
    print("저장:", out)


if __name__ == "__main__":
    main()
