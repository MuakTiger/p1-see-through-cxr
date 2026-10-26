"""B512 Grad-CAM 지도가 믿을 만한지 두 가지로 점검한다. val 사진만 연다 (test 는 열지 않는다), 재학습 없음, AWS 사용 없음.

(1) 구현 검증: src/gradcam.py 의 GradCAM 지도와, 훅 없이 layer4 출력과 fc 가중치로 직접 만든 지도(CAM)의 상관.
(2) 가리기 시험: 입력 512 사진을 16×16 칸(칸 하나 32×32 픽셀)으로 나눠, 지도 점수가 높은/무작위/낮은 칸 40개를
    그 사진의 평균 밝기로 덮고 예측 확률이 얼마나 변하는지 본다. 지도가 모델이 실제로 쓰는 곳을 가리키는지 보는 시험이다.
    "무엇을 단서로 쓰는지"나 "폐를 보는지"는 이 시험으로 알 수 없다.

사진은 val 에서 seed 42 로 무작위 300장.

사용 예 (p1 폴더에서):
    .venv/bin/python src/gradcam_b512_checks.py
결과: reports/equipment_v2/gradcam/B512/gradcam_checks.md
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))
import train as T  # noqa: E402
import gradcam as G  # noqa: E402
from shift_eval import load_model  # noqa: E402

SIZE = 512
GRID = 16
PATCH = SIZE // GRID


def direct_cam(model, x):
    """훅 없이 만든 CAM: relu(Σ_k fc.weight[c,k] · layer4 출력_k), c = 예측 클래스. (N,16,16) 원본 격자와 512 확대본을 돌려준다."""
    m = model
    with torch.no_grad():
        h = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        a = m.layer4(m.layer3(m.layer2(m.layer1(h))))        # (N,512,16,16)
        logits = m.fc(torch.flatten(m.avgpool(a), 1))
        c = logits.argmax(1)
        w = m.fc.weight[c]                                    # (N,512)
        grid = F.relu((w[:, :, None, None] * a).sum(1))       # (N,16,16)
        big = F.interpolate(grid[:, None], size=(SIZE, SIZE), mode="bilinear", align_corners=False)[:, 0]
        lo = big.amin(dim=(1, 2), keepdim=True)
        span = big.amax(dim=(1, 2), keepdim=True) - lo
        big = torch.where(span > 0, (big - lo) / span.clamp_min(1e-12), torch.zeros_like(big))
    return logits, grid, big


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "runs/results/b512-20261022-111844/model"))
    ap.add_argument("--images-dir", default=str(Path.home() / "data/chest_xray"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=40, help="가릴 칸 수 (256칸 중)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fill", choices=["mean", "zero"], default="mean",
                    help="가리는 값: mean = 그 사진 입력 텐서의 평균, zero = 정규화 후 0 (ImageNet 평균색)")
    ap.add_argument("--out", default=str(ROOT / "reports/equipment_v2/gradcam/B512/gradcam_checks.md"))
    args = ap.parse_args()

    G.SIZE = SIZE
    torch.set_num_threads(4)
    model = load_model(args.model_dir)
    cam_fn = G.GradCAM(model)
    tf = T.build_transform("P0", SIZE)

    rows = [r for r in csv.DictReader(open(ROOT / "manifests/split_manifest.csv", encoding="utf-8")) if r["split"] == "val"]
    rng = np.random.default_rng(args.seed)
    pick = [rows[i] for i in rng.permutation(len(rows))[: args.n]]

    corr, max_abs = [], []
    drop = {"높은 칸": [], "무작위 칸": [], "낮은 칸": []}
    flip = {k: [] for k in drop}
    top_in_ring = []
    ring = np.zeros((GRID, GRID), bool)
    ring[:2, :] = ring[-2:, :] = True
    ring[:, :2] = ring[:, -2:] = True

    for n, r in enumerate(pick, 1):
        img = Image.open(Path(args.images_dir) / r["relative_path"]).convert("RGB")
        x = tf(img).unsqueeze(0)
        logits_g, cams, valid = cam_fn(x, [None])
        cam_hook = cams[0][0]
        logits_d, grid, big = direct_cam(model, x)
        assert torch.allclose(logits_g, logits_d, atol=1e-4)
        a, b = cam_hook.ravel(), big[0].numpy().ravel()
        if valid[0][0] and b.std() > 0:
            corr.append(float(np.corrcoef(a, b)[0, 1]))
            max_abs.append(float(np.abs(a - b).max()))

        score = grid[0].numpy().ravel()                       # 칸 점수 (16×16 = 256)
        p0 = torch.softmax(logits_d, 1)[0]
        c = int(p0.argmax())
        order = np.argsort(-score, kind="stable")
        top_in_ring.append(bool(ring.ravel()[order[0]]))
        sets = {
            "높은 칸": order[: args.k],
            "무작위 칸": rng.choice(GRID * GRID, args.k, replace=False),
            "낮은 칸": order[::-1][: args.k],
        }
        fill = float(x.mean()) if args.fill == "mean" else 0.0
        for name, idx in sets.items():
            xm = x.clone()
            for i in idx:
                y0, x0 = (int(i) // GRID) * PATCH, (int(i) % GRID) * PATCH
                xm[:, :, y0:y0 + PATCH, x0:x0 + PATCH] = fill
            with torch.no_grad():
                p1 = torch.softmax(model(xm), 1)[0]
            drop[name].append(float(p1[c] - p0[c]))
            flip[name].append(int(p1.argmax()) != c)
        if n % 50 == 0:
            print(f"{n}/{len(pick)}", flush=True)

    ring_share = float(ring.mean())
    lines = [
        "# B512 Grad-CAM 점검 (val 무작위 %d장, seed %d)" % (len(pick), args.seed),
        "",
        "생성: `src/gradcam_b512_checks.py`. val 사진만 열었고 test·재학습·AWS는 쓰지 않았다. 모델은 `runs/results/b512-20261022-111844/model` (B512, seed 42).",
        "",
        "## 1. 구현 검증",
        "",
        f"- `src/gradcam.py`의 Grad-CAM 지도와, 훅 없이 layer4 출력·fc 가중치로 직접 만든 지도의 상관: 평균 {np.mean(corr):.6f} (최소 {np.min(corr):.6f}), 최대 절대 차이 {np.max(max_abs):.2e} ({len(corr)}장).",
        "- 두 경로의 logit은 모두 같았다 (최대 허용 오차 1e-4).",
        "",
        "## 2. 가리기 시험 (칸 %d개 / 256칸을 %s로 덮음)" % (args.k, "그 사진의 평균 밝기" if args.fill == "mean" else "정규화 후 0(ImageNet 평균색)"),
        "",
        "| 가린 칸 | 원래 예측 클래스의 확률 변화(평균) | 예측이 바뀐 사진 |",
        "|---|---|---|",
    ]
    for name in drop:
        lines.append(f"| 지도 점수 {name} | {np.mean(drop[name]):+.3f} | {100 * np.mean(flip[name]):.1f}% |")
    lines += [
        "",
        f"- 지도 점수가 가장 높은 칸이 바깥 2칸 띠에 있는 사진: {100 * np.mean(top_in_ring):.1f}% (띠의 면적 비율 {100 * ring_share:.1f}%)",
        "",
        "## 읽는 법과 한계",
        "",
        "- 지도가 높은 칸을 가렸을 때 확률이 더 많이 떨어지면, 지도는 모델이 실제로 쓰는 곳을 가리킨다고 볼 수 있다. 무작위 칸도 크게 떨어지므로(칸 40개는 사진의 16%) '지도가 정확한 근거'라고 말해서는 안 된다.",
        "- 이 시험은 '어디'만 말해 준다. 그곳에서 모델이 무엇에 반응하는지(글자 표식, 어깨, 촬영 방식, 폐 소견 등)는 알 수 없다.",
        "- 가리는 값(`--fill`)과 칸 크기(32픽셀)는 임의로 정했고, 사진 300장·모델 1개(seed 1개)다. 가리는 값을 바꾸면 결과가 달라질 수 있어 두 가지(mean, zero)를 따로 돌렸다.",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
