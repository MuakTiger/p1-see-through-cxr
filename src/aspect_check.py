"""가로세로 비율 의존 점검 (해석용, val만, 재학습 없음, 설계·선택에 쓰지 않음).

질문: B512는 사진의 가로세로 비율(눌린 정도)을 단서로 쓰는가?
방법: val 사진의 병변·질감은 그대로 두고, 512×512로 눌린 사진에서 내용을 가로로 f배 키우거나 줄여(가운데 기준, 가장자리는
      복제로 채움/잘라냄) "원본 사진의 가로세로 비율이 달랐다면" 생기는 모양 변화를 흉내 낸다.
      원본 비율 a의 사진을 정사각형으로 누르면 내용은 가로로 1/a배 눌린다. 비율이 a'였다면 내용 가로 배율은 a/a' 만큼 달라진다.
      f = 0.905 → 정상(1.225)이 test 정상 비율(1.353)로 찍혔다면, f = 1.22 → 폐렴(1.50)이 정상 비율(1.225)로 찍혔다면.
      f = 0.8, 1.1 은 같은 크기의 왜곡이 모든 클래스에 어떻게 작용하는지 보는 대조.
      레터박스(비율 유지 + 검은 패딩)는 눌림 자체를 없앤 입력 (학습 때 보지 못한 입력이라 해석에 주의).
지표: 클래스별 recall, 정답 클래스별 예측 분포, 예측이 바뀐 사진 비율. 같은 모델·같은 사진이라 짝지은 비교.

사용 예: .venv/bin/python src/aspect_check.py --device mps
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))
import shift_eval as SE  # noqa: E402
from equipment_eval_direct import predict_grays  # noqa: E402
from shift_filters import to_gray  # noqa: E402

NAMES = ["정상", "세균성", "바이러스성"]
FACTORS = [0.80, 0.905, 1.00, 1.10, 1.22]


def hscale(gray, f):
    """512×512 회색조의 내용을 가로로 f배 (가운데 기준). f<1 이면 양옆을 가장자리 복제로 채우고, f>1 이면 양옆이 잘린다."""
    if f == 1.0:
        return gray.copy()
    s = gray.shape[0]
    m = np.array([[f, 0, s / 2 * (1 - f)], [0, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(gray, m, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def letterbox(image, size):
    g = np.asarray(image.convert("L"))
    h, w = g.shape
    k = size / max(h, w)
    nh, nw = max(1, round(h * k)), max(1, round(w * k))
    r = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    out = np.zeros((size, size), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "runs/results/b512-20261022-111844/model"))
    ap.add_argument("--manifest", default=str(ROOT / "manifests/split_manifest.csv"))
    ap.add_argument("--images-dir", default=str(Path.home() / "data/chest_xray"))
    ap.add_argument("--out", default=str(ROOT / "reports/equipment_v2/aspect_check"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    rows = [r for r in csv.DictReader(open(args.manifest, encoding="utf-8")) if r["split"] == "val"]
    if args.limit:
        rows = rows[:args.limit]
    n = len(rows)
    y = np.array([int(r["label_id"]) for r in rows])
    groups = np.array([r["split_group"] for r in rows])
    aspect = np.array([int(r["width"]) / int(r["height"]) for r in rows])
    preprocess, hist_f1 = SE.read_metadata(args.model_dir)
    size = SE.read_input_size(args.model_dir)
    assert preprocess == "P0" and size == 512, (preprocess, size)
    model = SE.load_model(args.model_dir).to(args.device)

    grays = np.zeros((n, size, size), np.uint8)
    boxed = np.zeros((n, size, size), np.uint8)
    for i, r in enumerate(rows):
        with Image.open(Path(args.images_dir) / r["relative_path"]) as im:
            grays[i] = to_gray(im, size)
            boxed[i] = letterbox(im, size)
    print(f"val {n}장 읽음 ({time.time() - t0:.0f}초), 클래스별 {np.bincount(y, minlength=3).tolist()}, 장치 {args.device}", flush=True)

    variants = [("원본", grays)] + [(f"가로 {f:g}배", np.stack([hscale(g, f) for g in grays])) for f in FACTORS] + [("레터박스", boxed)]
    results = {}
    for name, arr in variants:
        probs = predict_grays(model, arr, args.device)
        f1, auroc, rec, _, pred = SE.metrics(y, probs)
        results[name] = (f1, rec, pred, probs)
        print(f"  {name}: Macro-F1 {f1:.4f}, recall {rec[0]:.3f}/{rec[1]:.3f}/{rec[2]:.3f} ({time.time() - t0:.0f}초)", flush=True)

    f1_c, rec_c, pred_c, probs_c = results["원본"]
    ident = bool(np.array_equal(pred_c, results["가로 1배"][2]))
    hist = f"{hist_f1:.4f}" if hist_f1 is not None else "기록 없음"

    lines = ["# 가로세로 비율 의존 점검 (B512, val, 해석용)", "",
             f"- 모델 B512 (재학습 없음), val {n}장 (정상/세균성/바이러스성 {np.bincount(y, minlength=3).tolist()}), 계산 장치 {args.device}",
             f"- 검증: 원본 Macro-F1 {f1_c:.4f} (학습 기록 베스트 {hist}), 가로 1배 = 원본 예측 동일: {ident}",
             "- 내용을 가로로 f배 (가운데 기준, f<1은 가장자리 복제로 채움, f>1은 양옆 잘림). f = 0.905: 정상이 test 정상 비율(1.353)로 찍혔다면, "
             "f = 1.22: 폐렴이 정상 비율(1.225)로 찍혔다면. 0.8, 1.1은 같은 크기 왜곡이 모든 클래스에 주는 영향을 보는 대조.",
             "- 이 시험은 해석용이다. test 결과나 모델·필터·강도 선택에 쓰지 않는다. 인과를 확정하지 못한다 "
             "(내용을 가로로 늘이는 것은 실제 다른 비율의 촬영과 같지 않다).", "",
             "## 정답 클래스별 recall (괄호: 원본 대비)", "",
             "| 변형 | Macro-F1 | 정상 recall | 세균성 recall | 바이러스성 recall |", "|---|---|---|---|---|"]
    for name, (f1, rec, pred, _) in results.items():
        cells = " | ".join(f"{rec[k]:.3f}" + ("" if name == "원본" else f" ({rec[k] - rec_c[k]:+.3f})") for k in range(3))
        lines.append(f"| {name} | {f1:.3f}" + ("" if name == "원본" else f" ({f1 - f1_c:+.3f})") + f" | {cells} |")

    lines += ["", "## 예측이 바뀐 사진 비율 (정답 클래스별, 원본 예측 대비)", "",
              "| 변형 | 정상 사진 | 세균성 사진 | 바이러스성 사진 | 전체 |", "|---|---|---|---|---|"]
    for name, (_, _, pred, _) in results.items():
        if name == "원본":
            continue
        ch = pred != pred_c
        lines.append(f"| {name} | " + " | ".join(f"{ch[y == k].mean() * 100:.1f}%" for k in range(3)) + f" | {ch.mean() * 100:.1f}% |")

    lines += ["", "## 정상 사진을 모델이 무엇으로 보나 (예측 분포, 정상/세균성/바이러스성)", "",
              "| 변형 | 정상 사진 → | 세균성 사진 → | 바이러스성 사진 → |", "|---|---|---|---|"]
    for name, (_, _, pred, _) in results.items():
        cells = " | ".join("/".join(f"{(pred[y == k] == c).mean() * 100:.0f}" for c in range(3)) + "%" for k in range(3))
        lines.append(f"| {name} | {cells} |")

    # 정상 사진 안에서: 원본 비율이 넓은 사진일수록 가로 0.905배에 더 민감한가는 보지 않는다 (표본 부족). 대신 평균 정상 확률만.
    lines += ["", "## 정상 사진의 평균 '정상' 확률", "", "| 변형 | 정상 사진 | 세균성 사진 | 바이러스성 사진 |", "|---|---|---|---|"]
    for name, (_, _, _, probs) in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{probs[y == k, 0].mean():.3f}" for k in range(3)) + " |")

    lines += ["", "## 95% 구간 (Macro-F1 변화, 환자 묶음 단위 짝지은 bootstrap)", "", "| 변형 | 변화 | 95% 구간 |", "|---|---|---|"]
    for name, (f1, _, pred, _) in results.items():
        if name == "원본":
            continue
        lo, hi = SE.bootstrap_delta(y, pred_c, pred, groups, args.n_boot, args.seed)
        lines.append(f"| {name} | {f1 - f1_c:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")

    lines += ["", f"참고: val 사진의 원본 가로세로 비율 중앙값 — 정상 {np.median(aspect[y == 0]):.3f}, 세균성 {np.median(aspect[y == 1]):.3f}, "
                  f"바이러스성 {np.median(aspect[y == 2]):.3f}", f"실행 시간 {time.time() - t0:.0f}초", ""]
    (out / "aspect_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"저장: {out / 'aspect_report.md'} ({time.time() - t0:.0f}초)", flush=True)


if __name__ == "__main__":
    main()
