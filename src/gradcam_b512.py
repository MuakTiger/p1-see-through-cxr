"""B512(입력 512)에 Grad-CAM 을 적용해 본다. val 사진만 연다 (test 는 열지 않는다), 재학습 없음, AWS 사용 없음.

src/gradcam.py 의 GradCAM(model.layer4[-1] 출력 기준)을 그대로 쓰고 지도 크기만 512 로 바꾼다 (layer4 출력은 16×16).
입력은 학습·평가와 같은 경로(P0: RGB → Resize((512,512)) → ToTensor → Normalize)다.
사진은 val 에서 정답 클래스마다 맞힌 것 2장, 틀린 것 2장(없으면 있는 만큼)을 seed 42 로 무작위로 뽑는다.

사용 예 (p1 폴더에서):
    .venv/bin/python src/gradcam_b512.py
    .venv/bin/python src/gradcam_b512.py --model-dir runs/results/v-s42-20261022-jlab/model --tag v-s42
결과: reports/equipment_v2/gradcam/<tag>/val_examples.png, examples.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))
import train as T  # noqa: E402
import gradcam as G  # noqa: E402
from shift_eval import load_model  # noqa: E402

SIZE = 512
CLASS_KO = G.CLASS_KO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "runs/results/b512-20261022-111844/model"))
    ap.add_argument("--images-dir", default=str(Path.home() / "data/chest_xray"))
    ap.add_argument("--tag", default="B512")
    ap.add_argument("--per-group", type=int, default=2, help="정답 클래스마다 맞힌 것·틀린 것 각각 몇 장")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    G.SIZE = SIZE  # GradCAM.__call__ 이 지도를 이 크기로 늘린다
    torch.set_num_threads(4)
    model = load_model(args.model_dir)
    cam_fn = G.GradCAM(model)
    tf = T.build_transform("P0", SIZE)

    rows = list(csv.DictReader(open(ROOT / "manifests/split_manifest.csv", encoding="utf-8")))
    val = [r for r in rows if r["split"] == "val"]
    rng = np.random.default_rng(args.seed)

    def predict(r):
        img = Image.open(Path(args.images_dir) / r["relative_path"]).convert("RGB")
        x = tf(img).unsqueeze(0)
        with torch.no_grad():
            p = torch.softmax(model(x), 1)[0].numpy()
        return x, p

    # 클래스마다 후보를 섞어서, 맞힌 것과 틀린 것이 각각 per-group 장 모일 때까지 예측한다.
    chosen = []  # (true, row, x, p)
    for t in range(3):
        pool = [r for r in val if int(r["label_id"]) == t]
        order = rng.permutation(len(pool))
        ok, bad = [], []
        for i in order:
            if len(ok) >= args.per_group and len(bad) >= args.per_group:
                break
            r = pool[i]
            x, p = predict(r)
            (ok if p.argmax() == t else bad).append((t, r, x, p))
            ok, bad = ok[: args.per_group], bad[: args.per_group]
        chosen += ok + bad

    plt = G.setup_matplotlib()
    cols = 2 * args.per_group
    fig, axes = plt.subplots(3, cols, figsize=(3.0 * cols, 3.55 * 3))
    axes = np.asarray(axes).reshape(3, cols)
    out_rows = []
    col_of = {0: 0, 1: 0, 2: 0}
    for t, r, x, p in chosen:
        logits, cams, valid = cam_fn(x, [None])
        assert np.allclose(torch.softmax(logits, 1)[0].numpy(), p, atol=1e-5)  # 훅이 값을 바꾸지 않았다
        cam = cams[0][0]
        gray = G.denorm_gray(x)[0]
        pred = int(p.argmax())
        ok = pred == t
        c = col_of[t]
        col_of[t] += 1
        ax = axes[t, c]
        G.overlay(ax, gray, cam if valid[0][0] else None,
                  f"정답 {CLASS_KO[t]} → 예측 {CLASS_KO[pred]} ({'맞음' if ok else '틀림'})\n{G.prob_text(p)}", fontsize=8.5)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a7a3b" if ok else "#c0392b")
            sp.set_linewidth(2.5)
        out_rows.append({"image": r["relative_path"], "true": CLASS_KO[t], "pred": CLASS_KO[pred], "correct": ok,
                         "p_normal": f"{p[0]:.4f}", "p_bacterial": f"{p[1]:.4f}", "p_viral": f"{p[2]:.4f}"})
    for t in range(3):
        for c in range(col_of[t], cols):
            axes[t, c].axis("off")
    fig.suptitle(f"{args.tag} Grad-CAM (val 사진, 입력 512, 예측 클래스 기준, layer4 출력 16×16)", fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.9, hspace=0.22, wspace=0.04)
    out = ROOT / "reports/equipment_v2/gradcam" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "val_examples.png", dpi=110)
    with open(out / "examples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"저장: {out.relative_to(ROOT)}  ({len(out_rows)}장)")


if __name__ == "__main__":
    main()
