"""장비 차이 민감도 시험: 학습된 모델에 val 사진을 필터별로 바꿔 넣고 점수를 다시 잰다.

재학습 없음. 필터는 메모리에서만 적용하고 사진을 저장하지 않는다. val 행만 읽는다
(train·check·test 영상은 열지 않는다). CPU에서만 실행한다 (MPS 사용 안 함).

사용 예:
    python src/shift_eval.py --model-dir runs/results/b-20261019-103853/model --out reports/shift_test

결과:
    shift_results.csv   필터 × 강도마다 Macro-F1, AUROC, 클래스별 recall, 원본 대비 변화와 95% 구간
    shift_curves.png    필터별 강도에 따른 점수 곡선
    shift_report.md     요약 표와 확인 결과

95% 구간: val 환자 묶음(split_group) 단위로 1,000번 복원 추출해서 "원본 대비 Macro-F1 변화"의
분포를 만든다. 같은 사진 묶음으로 원본과 필터 결과를 함께 계산하는 짝지은 방식이다.

전처리 모델(P 시리즈, P1~P4): model-dir 의 metadata.json 에서 preprocess 를 읽어 자동으로 적용한다.
순서는 "224 회색조 → 장비 필터(촬영 단계에서 생기는 차이) → 모델의 전처리(병원이 적용하는 처리) → 텐서".
비유: 다른 병원 장비로 찍힌 사진(장비 필터)이 들어와도, 병원은 늘 하던 처리(전처리)를 거친 뒤 모델에 넣는다.
metadata.json 이 없거나 preprocess 가 P0 이면 예전과 완전히 같은 동작이다.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import train as T  # noqa: E402  (지표 함수와 P0 전처리를 학습 코드와 똑같이 쓰기 위해)
from shift_filters import SHIFTS, apply_shift, to_gray  # noqa: E402
import preprocess as PP  # noqa: E402  (P1~P4 전처리, 학습 코드와 같은 정의)

MEAN = torch.tensor(T.IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(T.IMAGENET_STD).view(1, 3, 1, 1)
CLASS_KO = ["정상", "세균성", "바이러스성"]


def load_model(model_dir):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, T.NUM_CLASSES)
    state = torch.load(Path(model_dir) / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval()


def read_metadata(model_dir):
    """metadata.json 에서 전처리 이름과 학습 기록의 베스트 val Macro-F1 을 읽는다 (없으면 P0, None)."""
    path = Path(model_dir) / "metadata.json"
    if not path.is_file():
        return "P0", None
    meta = json.loads(path.read_text(encoding="utf-8"))
    return meta.get("preprocess", "P0"), meta.get("best_val_macro_f1")


def read_input_size(model_dir):
    """metadata.json 의 입력 크기 (없으면 224, 예전 실험 A~D, P1~P4 와 같다)."""
    path = Path(model_dir) / "metadata.json"
    if not path.is_file():
        return 224
    return int(json.loads(path.read_text(encoding="utf-8")).get("input_size", [224, 224])[0])


def apply_model_preprocess(grays, preprocess):
    """장비 필터를 거친 (N,224,224) 회색조에 모델의 전처리를 적용한다. P0 이면 그대로 돌려준다."""
    if preprocess == "P0":
        return grays
    return np.stack([PP.apply_preprocess(g, preprocess) for g in grays])


def gray_to_tensor(grays):
    """(N,224,224) uint8 회색조 → 3채널 복제 → 0~1 → ImageNet 정규화. train.py P0와 같은 순서."""
    x = torch.from_numpy(grays).float().div_(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
    return (x - MEAN) / STD


def predict_grays(model, grays, preprocess, chunk=32):
    """(N,S,S) 회색조 → 모델 전처리 → 텐서 → 확률. 큰 입력(448)에서 메모리를 아끼려고 chunk 장씩 처리한다."""
    return np.concatenate([predict(model, gray_to_tensor(apply_model_preprocess(grays[s:s + chunk], preprocess)))
                           for s in range(0, len(grays), chunk)])


@torch.no_grad()
def predict(model, x, batch=128):
    probs = []
    for i in range(0, len(x), batch):
        probs.append(torch.softmax(model(x[i:i + batch]), dim=1))
    return torch.cat(probs).numpy()


def macro_f1_fast(y, p):
    """bootstrap용 빠른 Macro-F1. train.py 의 macro_f1_and_recall 과 같은 정의 (zero_division=0)."""
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3)
    tp = np.diag(cm)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean()


def bootstrap_delta(y, p_clean, p_shift, groups, n_boot, seed):
    """환자 묶음 단위 짝지은 bootstrap으로 ΔMacro-F1(필터 − 원본)의 95% 구간."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        deltas[b] = macro_f1_fast(y[idx], p_shift[idx]) - macro_f1_fast(y[idx], p_clean[idx])
    return np.percentile(deltas, [2.5, 97.5])


def metrics(y, probs):
    pred = probs.argmax(1)
    f1, recalls, warns = T.macro_f1_and_recall(y.tolist(), pred.tolist())
    auroc, _, w2 = T.ovr_auroc(y, probs)
    return f1, auroc, recalls, warns + w2, pred


def plot(rows, clean, out_path, label="실험 B"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    plt.rcParams["font.family"] = "Apple SD Gothic Neo"
    plt.rcParams["axes.unicode_minus"] = False

    names = list(SHIFTS)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), sharey=True)
    for ax, name in zip(axes.flat, names):
        rs = [r for r in rows if r["shift"] == name]
        xs = [0] + [r["level"] for r in rs]
        labels = ["원본"] + [str(r["param_value"]) for r in rs]
        f1 = [clean["macro_f1"]] + [r["macro_f1"] for r in rs]
        lo = [0] + [r["delta_f1_ci_low"] for r in rs]
        hi = [0] + [r["delta_f1_ci_high"] for r in rs]
        ax.fill_between(xs, [clean["macro_f1"] + v for v in lo], [clean["macro_f1"] + v for v in hi],
                        color="black", alpha=0.10, label="Macro-F1 변화 95% 구간")
        ax.plot(xs, f1, "k-o", lw=2.5, label="Macro-F1")
        for key, ko, c in [("recall_normal", "정상 recall", "tab:green"),
                           ("recall_bacterial", "세균성 recall", "tab:orange"),
                           ("recall_viral", "바이러스성 recall", "tab:red")]:
            ax.plot(xs, [clean[key]] + [r[key] for r in rs], "--o", color=c, lw=1.5, ms=4, label=ko)
        ax.set_title(SHIFTS[name]["title"], fontsize=13)
        ax.set_xticks(xs, labels, fontsize=9)
        ax.set_xlabel(SHIFTS[name]["param"].split(" ")[0], fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes.flat[0].set_ylabel("점수")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=11, frameon=False)
    fig.suptitle(f"장비 차이 민감도 시험 ({label} 모델, val 1,030장, 재학습 없음)", fontsize=16)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="best_model.pt 가 있는 폴더")
    ap.add_argument("--manifest", default=str(ROOT / "manifests/split_manifest.csv"))
    ap.add_argument("--images-dir", default=str(Path.home() / "data/chest_xray"))
    ap.add_argument("--out", default=str(ROOT / "reports/shift_test"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="실험 B", help="그래프·리포트에 적을 모델 이름")
    ap.add_argument("--label-desc", default="layer4 미세조정", help="리포트에 적을 모델 설명")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # val 행만 읽는다
    rows = [r for r in csv.DictReader(open(args.manifest, encoding="utf-8")) if r["split"] == "val"]
    assert rows and all(r["split"] == "val" for r in rows)
    y = np.array([int(r["label_id"]) for r in rows])
    groups = np.array([r["split_group"] for r in rows])
    print(f"val {len(rows)}장 (클래스별 {np.bincount(y).tolist()}), 환자 묶음 {len(set(groups))}개", flush=True)

    preprocess, history_best_f1 = read_metadata(args.model_dir)
    PP.check_preprocess(preprocess)
    print(f"모델 전처리: {PP.PREPROCESS[preprocess]['desc']}"
          + ("" if preprocess == "P0" else " (순서: 224 회색조 → 장비 필터 → 모델 전처리 → 텐서)"), flush=True)

    size = read_input_size(args.model_dir)
    if size != 224:
        print(f"입력 크기 {size}×{size} (장비 필터의 σ 는 {size / 224:g}배로 키워 같은 해부학적 크기에 적용)", flush=True)

    # 사진을 한 번만 읽어서 (1) 학습과 똑같은 변환의 텐서, (2) 회색조(size×size)를 메모리에 둔다.
    # 448 에서는 전체를 텐서로 만들면 메모리가 모자라서 chunk 장씩 읽고 바로 예측한다.
    model = load_model(args.model_dir)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    p0 = T.build_transform(preprocess, size)  # P0 이면 예전과 같은 P0 변환
    chunk = 32
    grays = np.zeros((len(rows), size, size), np.uint8)
    probs_p0, probs_clean_parts, max_diff = [], [], 0.0
    for s0 in range(0, len(rows), chunk):
        part = rows[s0:s0 + chunk]
        tensors = []
        for j, r in enumerate(part):
            with Image.open(Path(args.images_dir) / r["relative_path"]) as im:
                tensors.append(p0(im.convert("RGB")))
                grays[s0 + j] = to_gray(im, size)
        x0 = torch.stack(tensors)
        gx = gray_to_tensor(apply_model_preprocess(grays[s0:s0 + len(part)], preprocess))
        max_diff = max(max_diff, float((x0 - gx).abs().max()))
        probs_p0.append(predict(model, x0))
        probs_clean_parts.append(predict(model, gx))
    print(f"사진 읽기와 원본 예측 완료 ({time.time() - t0:.0f}초)", flush=True)

    # 확인 1: 학습과 똑같은 변환 입력의 점수 = 학습 기록의 베스트 val Macro-F1
    f1_p0, auroc_p0, rec_p0, _, pred_p0 = metrics(y, np.concatenate(probs_p0))
    # 확인 2: 필터 경로(회색조 → 크기 → 모델 전처리)의 원본 = level 0
    probs_clean = np.concatenate(probs_clean_parts)
    f1_c, auroc_c, rec_c, _, pred_clean = metrics(y, probs_clean)
    same_pred = float((pred_p0 == pred_clean).mean())
    hist_txt = f"{history_best_f1:.4f}" if history_best_f1 is not None else "기록 없음"
    print(f"확인: 학습 변환 경로 Macro-F1 {f1_p0:.4f} (학습 기록 베스트 {hist_txt}) / 필터 경로 원본 {f1_c:.4f}, "
          f"예측 일치 {same_pred:.2%}, 입력 최대 차이 {max_diff:.4f}", flush=True)

    clean = {"macro_f1": f1_c, "macro_auroc": auroc_c, "recall_normal": rec_c[0],
             "recall_bacterial": rec_c[1], "recall_viral": rec_c[2]}
    results = []
    for name, spec in SHIFTS.items():
        for level in (1, 2, 3):
            shifted = np.stack([apply_shift(g, name, level, seed=args.seed + i) for i, g in enumerate(grays)])
            # 장비 필터(촬영 단계) 다음에 모델의 전처리(병원 처리)를 적용한다. P0 이면 그대로.
            probs = predict_grays(model, shifted, preprocess)
            f1, auroc, rec, warns, pred = metrics(y, probs)
            lo, hi = bootstrap_delta(y, pred_clean, pred, groups, args.n_boot, args.seed)
            viral = y == 2
            results.append({
                "shift": name, "level": level, "param_value": spec["levels"][level - 1],
                "macro_f1": f1, "macro_auroc": auroc,
                "recall_normal": rec[0], "recall_bacterial": rec[1], "recall_viral": rec[2],
                "delta_f1": f1 - f1_c, "delta_f1_ci_low": lo, "delta_f1_ci_high": hi,
                "changed_pred_pct": float((pred != pred_clean).mean() * 100),
                "viral_to_bacterial_pct": float(((pred == 1) & viral).sum() / viral.sum() * 100),
                "pred_share_normal": float((pred == 0).mean()), "pred_share_bacterial": float((pred == 1).mean()),
                "pred_share_viral": float((pred == 2).mean()),
                "warnings": " | ".join(warns),
            })
            print(f"  {name:9s} 강도{level} ({spec['levels'][level - 1]}): Macro-F1 {f1:.4f} "
                  f"(변화 {f1 - f1_c:+.4f}, 95% [{lo:+.4f}, {hi:+.4f}]), 바이러스 recall {rec[2]:.3f}", flush=True)

    # 저장
    fields = list(results[0].keys())
    with open(out / "shift_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({"shift": "clean", "level": 0, "param_value": "", "macro_f1": f1_c, "macro_auroc": auroc_c,
                    "recall_normal": rec_c[0], "recall_bacterial": rec_c[1], "recall_viral": rec_c[2],
                    "delta_f1": 0, "delta_f1_ci_low": 0, "delta_f1_ci_high": 0, "changed_pred_pct": 0,
                    "viral_to_bacterial_pct": float(((pred_clean == 1) & (y == 2)).sum() / (y == 2).sum() * 100),
                    "pred_share_normal": float((pred_clean == 0).mean()),
                    "pred_share_bacterial": float((pred_clean == 1).mean()),
                    "pred_share_viral": float((pred_clean == 2).mean()), "warnings": ""})
        for r in results:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    plot(results, clean, out / "shift_curves.png", args.label)

    lines = ["# 장비 차이 민감도 시험 결과", "",
             f"- 모델: `{args.model_dir}` ({args.label}, {args.label_desc})",
             *([] if preprocess == "P0" else [
                 f"- 모델 전처리: {PP.PREPROCESS[preprocess]['desc']}. 입력 순서: 224 회색조 → 장비 필터(촬영 단계 차이) "
                 "→ 모델 전처리(병원이 늘 하는 처리) → 텐서"]),
             f"- 데이터: val {len(rows)}장 (정상/세균성/바이러스성 {np.bincount(y).tolist()}), 환자 묶음 {len(set(groups))}개",
             "- 필터 정의와 강도의 근거: `filter_rationale.md`. 강도는 시험 전에 고정했다.",
             f"- 95% 구간: 환자 묶음 단위 짝지은 bootstrap {args.n_boot}회",
             "", "## 확인", "",
             f"- 학습과 똑같은 {preprocess} 변환 경로의 Macro-F1: {f1_p0:.4f} (학습 기록의 베스트 val Macro-F1 {hist_txt} 과 같아야 함)",
             f"- 필터 경로(회색조 변환 후 224{'' if preprocess == 'P0' else ' → ' + preprocess})의 원본 Macro-F1: {f1_c:.4f}, 두 경로의 예측 일치율 {same_pred:.2%}, 입력 최대 차이 {max_diff:.4f}",
             "- 아래 변화량은 모두 필터 경로의 원본 기준이다.",
             "", "## 결과", "",
             "| 필터 | 강도 | 값 | Macro-F1 | 변화 | 95% 구간 | 정상 recall | 세균성 recall | 바이러스성 recall | 예측이 바뀐 사진 |",
             "|---|---|---|---|---|---|---|---|---|---|",
             f"| 원본 | 0 | - | {f1_c:.3f} | - | - | {rec_c[0]:.3f} | {rec_c[1]:.3f} | {rec_c[2]:.3f} | - |"]
    for r in results:
        lines.append(f"| {SHIFTS[r['shift']]['title']} | {r['level']} | {r['param_value']} | {r['macro_f1']:.3f} | "
                     f"{r['delta_f1']:+.3f} | [{r['delta_f1_ci_low']:+.3f}, {r['delta_f1_ci_high']:+.3f}] | "
                     f"{r['recall_normal']:.3f} | {r['recall_bacterial']:.3f} | {r['recall_viral']:.3f} | {r['changed_pred_pct']:.1f}% |")
    lines += ["", f"실행 시간: {time.time() - t0:.0f}초 (CPU)", ""]
    (out / "shift_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"저장: {out}/shift_results.csv, shift_curves.png, shift_report.md ({time.time() - t0:.0f}초)", flush=True)


if __name__ == "__main__":
    main()
