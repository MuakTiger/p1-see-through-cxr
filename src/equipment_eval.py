"""장치 설정 필터 v2 시험: 학습된 모델에 val 사진을 필터별로 바꿔 넣고 점수를 다시 잰다 (재학습 없음, val만, CPU).

필터는 새 5종(선량·조사시간·산란·kVp·센서: training/equipment_filters.py)과 v1에서 그대로 쓰는 6종
(흐림·경계 강조·대비 보정·JPEG·밝게·어둡게: training/shift_filters.py), 모두 × 3단계. 강도는 시험 전에
reports/equipment_v2/strength_fixed.md 로 고정했다. 지표와 bootstrap 은 src/shift_eval.py 의 함수를 그대로 쓴다.

검증 (스크립트가 스스로 확인해서 리포트에 적는다)
    (a) 학습과 같은 변환 경로의 원본 Macro-F1 = 학습 기록의 베스트 val Macro-F1
    (b) v1 에서 그대로 쓰는 6종을 같은 경로로 다시 돌린 점수 = reports/shift_test 의 B 결과
    (c) 같은 입력·seed 면 같은 출력, (d) 필터 0단계 = 원본 (사진 1장으로 확인)
    (e) 이 시험의 코드 해시 = 시험 전에 적어 둔 해시

사용 예 (p1 폴더에서):
    .venv/bin/python src/equipment_eval.py --model-dir runs/results/b-20261019-103853/model --out reports/equipment_v2/val_B
"""
import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))
import shift_eval as SE  # noqa: E402
import train as T  # noqa: E402
from equipment_filters import EQUIP, apply_equipment, to_input, to_work  # noqa: E402
from gradcam import setup_matplotlib  # noqa: E402
from shift_filters import SHIFTS, apply_shift, to_gray  # noqa: E402

NEW = list(EQUIP)
REUSED = ["blur", "sharpen", "equalize", "jpeg", "bright", "dark"]
ALL = NEW + REUSED
LEVELS = (1, 2, 3)


def title_of(name):
    return EQUIP[name]["title"] if name in EQUIP else SHIFTS[name]["title"] + " (v1)"


def value_of(name, level):
    return EQUIP[name]["labels"][level - 1] if name in EQUIP else str(SHIFTS[name]["levels"][level - 1])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(results, clean, path, label):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(3, 4, figsize=(18, 12.5), sharey=True)
    for ax, name in zip(axes.flat, ALL):
        rs = [r for r in results if r["shift"] == name]
        xs = [0] + [r["level"] for r in rs]
        ax.fill_between(xs, [clean["macro_f1"] + v for v in [0] + [r["delta_f1_ci_low"] for r in rs]],
                        [clean["macro_f1"] + v for v in [0] + [r["delta_f1_ci_high"] for r in rs]],
                        color="black", alpha=0.10, label="Macro-F1 변화 95% 구간")
        ax.plot(xs, [clean["macro_f1"]] + [r["macro_f1"] for r in rs], "k-o", lw=2.5, label="Macro-F1")
        for key, ko, c in [("recall_normal", "정상 recall", "tab:green"), ("recall_bacterial", "세균성 recall", "tab:orange"),
                           ("recall_viral", "바이러스성 recall", "tab:red")]:
            ax.plot(xs, [clean[key]] + [r[key] for r in rs], "--o", color=c, lw=1.5, ms=4, label=ko)
        ax.set_title(title_of(name), fontsize=13, fontweight="bold" if name in EQUIP else "normal")
        ax.set_xticks(xs, ["원본"] + [value_of(name, r["level"]) for r in rs], fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    ax = axes.flat[11]
    ax.axis("off")
    ax.legend(handles, labels, loc="center", fontsize=12, frameon=False)
    axes.flat[0].set_ylabel("점수")
    fig.suptitle(f"장치 설정 필터 v2 시험 ({label} 모델, val 1,030장, 재학습 없음). 굵은 제목 = 새로 만든 5종", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "manifests/split_manifest.csv"))
    ap.add_argument("--images-dir", default=str(Path.home() / "data/chest_xray"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="B")
    ap.add_argument("--v1-csv", default=str(ROOT / "reports/shift_test/shift_results.csv"), help="같은 모델의 v1 시험 결과 (검증 b)")
    ap.add_argument("--hash-file", default=str(ROOT / "reports/equipment_v2/code_sha256_before_val.txt"))
    ap.add_argument("--limit", type=int, default=0, help="속도 확인용: 앞의 N장만 (결과는 의미 없음)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rows = [r for r in csv.DictReader(open(args.manifest, encoding="utf-8")) if r["split"] == "val"]
    assert rows and all(r["split"] == "val" for r in rows)
    if args.limit:
        rows = rows[:args.limit]
    n = len(rows)
    y = np.array([int(r["label_id"]) for r in rows])
    groups = np.array([r["split_group"] for r in rows])
    print(f"val {n}장 (클래스별 {np.bincount(y, minlength=3).tolist()}), 환자 묶음 {len(set(groups))}개", flush=True)

    preprocess, hist_f1 = SE.read_metadata(args.model_dir)
    assert preprocess == "P0" and SE.read_input_size(args.model_dir) == 224, "이 시험은 P0·224 모델만 지원한다"
    model = SE.load_model(args.model_dir)
    p0 = T.build_transform("P0", 224)

    grays = np.zeros((n, 224, 224), np.uint8)
    conds = {(name, lv): np.zeros((n, 224, 224), np.uint8) for name in ALL for lv in LEVELS}
    probs_p0, tensors = [], []
    checks = {"deterministic": True, "level0_identity": True}
    for i, r in enumerate(rows):
        with Image.open(Path(args.images_dir) / r["relative_path"]) as im:
            work, g = to_work(im), to_gray(im, 224)
            tensors.append(p0(im.convert("RGB")))
        grays[i] = g
        for name in NEW:
            for lv in LEVELS:
                conds[(name, lv)][i] = to_input(work, apply_equipment(work, name, lv, seed=args.seed + i), g)
        for name in REUSED:
            for lv in LEVELS:
                conds[(name, lv)][i] = apply_shift(g, name, lv, seed=args.seed + i)
        if i == 0:  # (c) 같은 입력·seed → 같은 출력, (d) 0단계 = 원본
            for name in NEW:
                again = to_input(work, apply_equipment(work, name, 3, seed=args.seed), g)
                checks["deterministic"] &= bool(np.array_equal(again, conds[(name, 3)][0]))
                checks["level0_identity"] &= bool(np.array_equal(to_input(work, apply_equipment(work, name, 0), g), g))
        if len(tensors) == 32 or i == n - 1:
            probs_p0.append(SE.predict(model, torch.stack(tensors)))
            tensors = []
        if (i + 1) % 100 == 0:
            print(f"  사진 {i + 1}/{n} ({time.time() - t0:.0f}초)", flush=True)

    # (a) 학습과 같은 변환 경로 vs 필터 경로의 원본
    f1_p0, _, _, _, pred_p0 = SE.metrics(y, np.concatenate(probs_p0))
    probs_clean = SE.predict_grays(model, grays, "P0")
    f1_c, auroc_c, rec_c, _, pred_clean = SE.metrics(y, probs_clean)
    same_pred = float((pred_p0 == pred_clean).mean())
    hist_txt = f"{hist_f1:.4f}" if hist_f1 is not None else "기록 없음"
    print(f"확인 (a): 학습 변환 경로 Macro-F1 {f1_p0:.4f} (학습 기록 베스트 {hist_txt}), 필터 경로 원본 {f1_c:.4f}, 예측 일치 {same_pred:.2%}", flush=True)
    clean = {"macro_f1": f1_c, "macro_auroc": auroc_c, "recall_normal": rec_c[0], "recall_bacterial": rec_c[1], "recall_viral": rec_c[2]}

    results = []
    for name in ALL:
        for lv in LEVELS:
            probs = SE.predict_grays(model, conds[(name, lv)], "P0")
            f1, auroc, rec, warns, pred = SE.metrics(y, probs)
            lo, hi = SE.bootstrap_delta(y, pred_clean, pred, groups, args.n_boot, args.seed)
            viral = y == 2
            results.append({
                "shift": name, "level": lv, "param_value": value_of(name, lv), "macro_f1": f1, "macro_auroc": auroc,
                "recall_normal": rec[0], "recall_bacterial": rec[1], "recall_viral": rec[2],
                "delta_f1": f1 - f1_c, "delta_f1_ci_low": lo, "delta_f1_ci_high": hi,
                "changed_pred_pct": float((pred != pred_clean).mean() * 100),
                "viral_to_bacterial_pct": float(((pred == 1) & viral).sum() / max(viral.sum(), 1) * 100),
                "pred_share_normal": float((pred == 0).mean()), "pred_share_bacterial": float((pred == 1).mean()),
                "pred_share_viral": float((pred == 2).mean()), "warnings": " | ".join(warns)})
            print(f"  {name:9s} 강도{lv} ({value_of(name, lv)}): Macro-F1 {f1:.4f} (변화 {f1 - f1_c:+.4f}, 95% [{lo:+.4f}, {hi:+.4f}]), "
                  f"바이러스 recall {rec[2]:.3f}", flush=True)

    # (b) v1 에서 그대로 쓰는 6종: 같은 경로로 다시 돌린 점수 = 지난번 B 결과
    v1_diff, v1_note = None, "v1 결과 파일 없음"
    if Path(args.v1_csv).is_file() and not args.limit:
        v1 = {(r["shift"], int(r["level"])): float(r["macro_f1"]) for r in csv.DictReader(open(args.v1_csv, encoding="utf-8"))}
        diffs = [abs(round(r["macro_f1"], 4) - v1[(r["shift"], r["level"])]) for r in results if r["shift"] in REUSED]
        diffs.append(abs(round(f1_c, 4) - v1[("clean", 0)]))
        v1_diff = max(diffs)
        v1_note = f"6종 × 3단계 + 원본 {len(diffs)}개의 Macro-F1 최대 차이 {v1_diff:.4f}"
    # (e) 코드 해시
    now_hash = sha256(ROOT / "training/equipment_filters.py")
    hash_note = "해시 파일 없음"
    if Path(args.hash_file).is_file():
        ref = {Path(l.split()[1]).name: l.split()[0] for l in open(args.hash_file, encoding="utf-8") if l.strip()}
        hash_note = ("시험 전 해시와 같음" if ref.get("equipment_filters.py") == now_hash else "**시험 전 해시와 다름**") + f" ({now_hash[:12]}…)"
    print(f"확인 (b): {v1_note} / (c) 같은 seed 같은 출력: {checks['deterministic']} / (d) 0단계=원본: {checks['level0_identity']} / (e) {hash_note}", flush=True)

    fields = list(results[0].keys())
    with open(out / "shift_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({"shift": "clean", "level": 0, "param_value": "", "macro_f1": f1_c, "macro_auroc": auroc_c,
                    "recall_normal": rec_c[0], "recall_bacterial": rec_c[1], "recall_viral": rec_c[2], "delta_f1": 0,
                    "delta_f1_ci_low": 0, "delta_f1_ci_high": 0, "changed_pred_pct": 0,
                    "viral_to_bacterial_pct": float(((pred_clean == 1) & (y == 2)).sum() / max((y == 2).sum(), 1) * 100),
                    "pred_share_normal": float((pred_clean == 0).mean()), "pred_share_bacterial": float((pred_clean == 1).mean()),
                    "pred_share_viral": float((pred_clean == 2).mean()), "warnings": ""})
        for r in results:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    plot(results, clean, out / "shift_curves.png", args.label)

    lines = ["# 장치 설정 필터 v2 시험 결과 (val)", "",
             f"- 모델: `{args.model_dir}` ({args.label}, 증강·전처리 없음, 재학습 없음)",
             f"- 데이터: val {n}장 (정상/세균성/바이러스성 {np.bincount(y, minlength=3).tolist()}), 환자 묶음 {len(set(groups))}개. train·check·test는 열지 않음",
             "- 필터와 강도: `strength_fixed.md` (시험 전에 고정). 굵게 표시한 행이 새로 만든 5종, (v1)은 지난번과 같은 필터",
             f"- 95% 구간: 환자 묶음 단위 짝지은 bootstrap {args.n_boot}회, 노이즈 seed 42 + 사진 번호", "",
             "## 검증", "",
             f"- (a) 학습과 같은 변환 경로의 Macro-F1 {f1_p0:.4f} (학습 기록 베스트 {hist_txt}), 필터 경로의 원본 {f1_c:.4f}, 두 경로의 예측 일치율 {same_pred:.2%}",
             f"- (b) v1에서 그대로 쓰는 6종을 같은 경로로 다시 돌린 결과와 `reports/shift_test`(B)의 비교: {v1_note}",
             f"- (c) 같은 입력·seed에서 같은 출력: {checks['deterministic']}, (d) 필터 0단계 = 원본: {checks['level0_identity']}",
             f"- (e) 코드 해시: {hash_note}", "", "## 결과", "",
             "| 필터 | 강도 | 값 | Macro-F1 | 변화 | 95% 구간 | 정상 recall | 세균성 recall | 바이러스성 recall | 예측이 바뀐 사진 | 예측 비율 (정상/세균/바이러스) |",
             "|---|---|---|---|---|---|---|---|---|---|---|",
             f"| 원본 | 0 | - | {f1_c:.3f} | - | - | {rec_c[0]:.3f} | {rec_c[1]:.3f} | {rec_c[2]:.3f} | - | "
             f"{(pred_clean == 0).mean() * 100:.0f}/{(pred_clean == 1).mean() * 100:.0f}/{(pred_clean == 2).mean() * 100:.0f}% |"]
    for r in results:
        t = f"**{title_of(r['shift'])}**" if r["shift"] in EQUIP else title_of(r["shift"])
        lines.append(f"| {t} | {r['level']} | {r['param_value']} | {r['macro_f1']:.3f} | {r['delta_f1']:+.3f} | "
                     f"[{r['delta_f1_ci_low']:+.3f}, {r['delta_f1_ci_high']:+.3f}] | {r['recall_normal']:.3f} | {r['recall_bacterial']:.3f} | "
                     f"{r['recall_viral']:.3f} | {r['changed_pred_pct']:.1f}% | "
                     f"{r['pred_share_normal'] * 100:.0f}/{r['pred_share_bacterial'] * 100:.0f}/{r['pred_share_viral'] * 100:.0f}% |")
    sig = [f"{title_of(r['shift'])} {r['param_value']} ({r['delta_f1']:+.3f})" for r in results if r["delta_f1_ci_high"] < 0 or r["delta_f1_ci_low"] > 0]
    lines += ["", f"95% 구간이 0을 포함하지 않는 조건 {len(sig)}개 / 33개: " + (", ".join(sig) if sig else "없음"),
              "", f"실행 시간: {time.time() - t0:.0f}초 (CPU)", ""]
    (out / "shift_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"저장: {out} ({time.time() - t0:.0f}초)", flush=True)


if __name__ == "__main__":
    main()
