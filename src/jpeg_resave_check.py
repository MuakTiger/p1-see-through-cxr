"""저장 화질 지름길 확인: val 사진을 JPEG 품질 75로 다시 저장해서 모델에 넣으면 예측이 바뀌는가.

배경: train/val 정상 사진은 JPEG 품질 약 96, test 정상 사진은 품질 약 75 로 저장되어 있다 (폐렴은 대부분 75).
224 로 줄이면 이 차이가 거의 사라져서(픽셀 차이 1.70 → 0.19) 모델 D 는 이 흔적을 쓰지 않았다.
448 에서는 흔적이 더 남을 수 있어서, 같은 확인을 B(224) 와 B448 에 한다.

세 경우: (a) 원본 그대로 (b) 정상 사진만 품질 75 로 재저장 (c) 전부 재저장.
재저장은 원본 해상도에서 하고, 그다음 모델의 입력 크기로 줄인다. val 사진만 읽는다.
재학습·임계값 조정 없음. 결과는 "예측이 바뀐 사진 수", Macro-F1, 정상 recall.

사용 예 (p1 폴더에서):
    .venv/bin/python src/jpeg_resave_check.py --out reports/shift_test_B448/jpeg_resave_check.md
"""
import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))
import train as T  # noqa: E402
from shift_eval import load_model  # noqa: E402
from shift_filters import to_gray  # noqa: E402

MODELS = {
    "B 224": (ROOT / "runs/results/b-20261019-103853/model", 224),
    "D 224": (ROOT / "runs/results/d-20261019-134621/model", 224),
    "B448": (ROOT / "runs/results/b448-20261020-161517/model", 448),
}


def resave(im, quality=75):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


@torch.no_grad()
def predict(model, tensors):
    return torch.softmax(model(torch.stack(tensors)), 1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "manifests/split_manifest.csv"))
    ap.add_argument("--images-dir", default=str(Path.home() / "data/chest_xray"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--quality", type=int, default=75)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.manifest, encoding="utf-8")) if r["split"] == "val"]
    assert rows and all(r["split"] == "val" for r in rows)
    y = np.array([int(r["label_id"]) for r in rows])
    models = {k: load_model(d) for k, (d, _) in MODELS.items()}
    sizes = {k: s for k, (_, s) in MODELS.items()}
    tfm = {s: T.build_transform("P0", s) for s in set(sizes.values())}

    probs = {k: {"orig": [], "resaved": []} for k in MODELS}
    pix_diff = {s: [] for s in tfm}  # 정상 사진 20장에서 재저장 전후 회색조 픽셀 차이 (0~255)
    chunk = 32
    for s0 in range(0, len(rows), chunk):
        part = rows[s0:s0 + chunk]
        ims = []
        for r in part:
            with Image.open(Path(args.images_dir) / r["relative_path"]) as im:
                im = im.convert("RGB")
                ims.append((im.copy(), resave(im, args.quality)))
        for size, t in tfm.items():
            a = [t(o) for o, _ in ims]
            b = [t(c) for _, c in ims]
            for j, r in enumerate(part):
                if r["label_id"] == "0" and len(pix_diff[size]) < 20:
                    pix_diff[size].append(float(np.abs(to_gray(ims[j][0], size).astype(float)
                                                       - to_gray(ims[j][1], size).astype(float)).mean()))
            for k in MODELS:
                if sizes[k] == size:
                    probs[k]["orig"].append(predict(models[k], a))
                    probs[k]["resaved"].append(predict(models[k], b))
        print(f"  {min(s0 + chunk, len(rows))}/{len(rows)}", flush=True)

    lines = ["# JPEG 재저장 확인 (val 1,030장)", "",
             f"정상 사진만 품질 {args.quality}로 다시 저장하거나 전부 다시 저장해서 모델에 넣었다. 재학습 없음, val 사진만 사용.", "",
             "| 모델 | 경우 | Macro-F1 | 정상 recall | 예측이 바뀐 사진 | 정상 사진 중 예측이 바뀐 수 |", "|---|---|---|---|---|---|"]
    for k in MODELS:
        po = np.concatenate(probs[k]["orig"]); pr = np.concatenate(probs[k]["resaved"])
        base = po.argmax(1)
        for label, mask in [("(a) 원본 그대로", np.zeros(len(y), bool)), ("(b) 정상만 재저장", y == 0),
                            ("(c) 전부 재저장", np.ones(len(y), bool))]:
            pred = np.where(mask, pr.argmax(1), base)
            f1, rec, _ = T.macro_f1_and_recall(y.tolist(), pred.tolist())
            changed = int((pred != base).sum())
            ch_norm = int(((pred != base) & (y == 0)).sum())
            lines.append(f"| {k} | {label} | {f1:.4f} | {rec[0]:.3f} | {changed} | {ch_norm} |")
    lines += ["", "재저장 전후 회색조 픽셀 평균 차이 (val 정상 사진 앞 20장, 0~255, 입력 크기별): "
              + ", ".join(f"{s}: {np.mean(v):.2f}" for s, v in pix_diff.items())]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
