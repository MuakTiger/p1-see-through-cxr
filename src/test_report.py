"""최종 test 평가 결과 정리: AWS vs iMac 재현 비교, 그림 2장, 통합 요약 JSON, 보고서 표.

입력 (이미 만들어진 결과만 읽는다. 영상은 열지 않는다):
    reports/test_eval/aws/    SageMaker 평가 작업 산출물 (주 결과)
    reports/test_eval/local/  같은 evaluate.py 를 iMac CPU 에서 돌린 산출물 (재현 확인용)
    runs/results/*/history.csv, runs/test_eval_models/*/metadata.json  val 점수 (베스트 에폭)
    runs/submissions/<평가 작업>.json  작업 이름·상태

출력 (reports/test_eval/):
    test_confusion.png        주 모델 618장 혼동행렬 (개수 + 행 비율)
    test_vs_val.png           모델 8개의 val Macro-F1 vs test Macro-F1
    reproducibility.csv       AWS vs iMac 비교 (주 모델 예측 일치율, 확률 최대 차이, 모든 지표 최대 차이)
    evaluation_summary.json   통합 요약 (작업 이름·상태, 주 결과, 재현 확인)
    test_report_tables.md     보고서에 넣을 숫자 표 (test_report.md 는 이 표를 바탕으로 사람이 해석을 붙여 쓴다)

사용 예 (p1 폴더에서):
    .venv/bin/python src/test_report.py --job p1-cxr-<학생ID>-testeval-20261019-171158
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/test_eval"
RESULT_DIRS = {"A": "a-20261019-101748", "B": "b-20261019-103853", "C": "c-20261019-143150",
               "D": "d-20261019-134621", "P1": "p1-20261019-153257", "P2": "p2-20261019-153259",
               "P3": "p3-20261019-153301", "P4": "p4-20261019-153303"}
CLASS_KO = ["정상", "세균성", "바이러스성"]
LABEL = {"A": "A 특징 고정", "B": "B layer4", "C": "C 클래스 가중치", "D": "D 노이즈 증강",
         "P1": "P1 HE", "P2": "P2 CLAHE", "P3": "P3 LPF", "P4": "P4 HPF"}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def val_best(mid):
    """history.csv 의 베스트 에폭(metadata 의 selected_epoch) 행. 그 값이 history 최댓값·metadata 값과 같은지 확인."""
    meta = json.loads((ROOT / "runs/test_eval_models" / mid / "metadata.json").read_text(encoding="utf-8"))
    hist = read_csv(ROOT / "runs/results" / RESULT_DIRS[mid] / "history.csv")
    row = next(h for h in hist if int(h["epoch"]) == meta["selected_epoch"])
    f1 = float(row["val_macro_f1"])
    assert abs(f1 - max(float(h["val_macro_f1"]) for h in hist)) < 1e-9, mid
    assert abs(f1 - meta["best_val_macro_f1"]) < 1e-5, mid
    return {"epoch": int(row["epoch"]), "macro_f1": f1, "recall": [float(row[f"val_recall_{s}"]) for s in
                                                                  ("normal", "bacterial", "viral")]}


def compare_runs():
    """AWS 와 iMac 결과 비교. 주 모델은 사진별 확률까지, 나머지는 metrics.csv 의 모든 숫자."""
    pa = {r["relative_path"]: r for r in read_csv(OUT / "aws/test_predictions.csv")}
    pl = {r["relative_path"]: r for r in read_csv(OUT / "local/test_predictions.csv")}
    assert pa.keys() == pl.keys() and len(pa) == 624
    out = {}
    for name, keep in (("618", lambda r: r["excluded"] == "False"), ("624", lambda r: True)):
        paths = [p for p in pa if keep(pa[p])]
        same = sum(pa[p]["pred_label"] == pl[p]["pred_label"] for p in paths)
        diff = max(abs(float(pa[p][k]) - float(pl[p][k])) for p in paths for k in ("p_normal", "p_bacterial", "p_viral"))
        out[name] = {"n": len(paths), "pred_agree": same, "pred_agree_rate": same / len(paths),
                     "max_abs_prob_diff": diff}
    ma, ml = read_csv(OUT / "aws/metrics.csv"), read_csv(OUT / "local/metrics.csv")
    assert len(ma) == len(ml)
    num_cols = ["precision", "recall", "recall_ci_low", "recall_ci_high", "f1", "auroc", "accuracy", "macro_f1",
                "macro_f1_ci_low", "macro_f1_ci_high", "macro_auroc", "macro_auroc_ci_low", "macro_auroc_ci_high"]
    worst = (0.0, None)
    for a, b in zip(ma, ml):
        assert (a["model_id"], a["eval_set"], a["scope"]) == (b["model_id"], b["eval_set"], b["scope"])
        for c in num_cols:
            if a[c] and b[c]:
                d = abs(float(a[c]) - float(b[c]))
                if d > worst[0]:
                    worst = (d, f"{a['model_id']}/{a['eval_set']}/{a['scope']}/{c}")
    out["metrics_max_abs_diff"] = {"value": worst[0], "where": worst[1]}
    # 오분류 목록은 확률(소수 6자리)이 아주 조금 달라 파일 바이트는 다를 수 있으므로, 사진과 오류 종류가 같은지 본다
    mis = [{(r["relative_path"], r["error_type"]) for r in read_csv(OUT / f"{k}/misclassified.csv")} for k in ("aws", "local")]
    out["misclassified_same_images_and_types"] = mis[0] == mis[1]
    out["misclassified_count"] = len(mis[0])
    out["confusion_identical"] = (OUT / "aws/confusion_matrix.csv").read_bytes() == (OUT / "local/confusion_matrix.csv").read_bytes()
    return out


def compare_with_first_run():
    """재실행 전 첫 실행(superseded_run1/aws, 확률 소수 8자리 저장)과 지금 결과의 지표가 같은지 본다.

    재실행에서 바뀐 것은 확률 저장 형식뿐이어야 한다. metrics.csv 는 메모리의 확률로 계산하므로 같아야 하고,
    예측 라벨도 같아야 한다.
    """
    old_dir = OUT / "superseded_run1/aws"
    if not old_dir.is_dir():
        return None
    ma, mo = read_csv(OUT / "aws/metrics.csv"), read_csv(old_dir / "metrics.csv")
    same_metrics = ma == mo
    pa = read_csv(OUT / "aws/test_predictions.csv")
    po = {r["relative_path"]: r for r in read_csv(old_dir / "test_predictions.csv")}
    same_pred = all(po[r["relative_path"]]["pred_label"] == r["pred_label"] for r in pa)
    return {"first_run_dir": "reports/test_eval/superseded_run1/aws", "metrics_csv_identical": same_metrics,
            "pred_labels_identical_624": same_pred}


def confidence_analysis():
    """주 모델 618장의 확신도 분석 (test_predictions.csv, 전체 정밀도 확률로 계산).

    - 정상→폐렴 오류의 p_normal 분포와 예측 확률(가장 큰 확률) 분포
    - 정상 AUROC 가 아주 작은 확률들의 순서에 얼마나 기대는지: 확률을 소수 8자리로 반올림하면 AUROC 가 얼마나 바뀌나
    """
    import sys
    sys.path.insert(0, str(ROOT / "training"))
    import train as T  # 지표 정의는 train.py 와 같게

    rows = [r for r in read_csv(OUT / "aws/test_predictions.csv") if r["excluded"] == "False"]
    y = np.array([int(r["true_label"]) for r in rows])
    p = np.array([[float(r["p_normal"]), float(r["p_bacterial"]), float(r["p_viral"])] for r in rows])
    pred = p.argmax(1)
    pmax = p.max(1)
    fn = (y == 0) & (pred != 0)  # 정상인데 폐렴으로 예측
    pn = p[fn, 0]
    q = {f"q{int(k * 100)}": float(np.quantile(pn, k)) for k in (0.10, 0.25, 0.50, 0.75, 0.90)}
    wrong = pred != y
    rounded = np.round(p, 8)
    auc_full = [T.binary_auroc(y == c, p[:, c]) for c in range(3)]
    auc_round = [T.binary_auroc(y == c, rounded[:, c]) for c in range(3)]
    return {
        "normal_to_pneumonia": {
            "n": int(fn.sum()), "p_normal_quantiles": q,
            "n_p_normal_lt_0.01": int((pn < 0.01).sum()), "n_p_normal_lt_0.1": int((pn < 0.1).sum()),
            "pred_prob_median": float(np.median(pmax[fn])),
            "n_pred_prob_ge_0.70": int((pmax[fn] >= 0.70).sum()),
            "n_pred_prob_ge_0.90": int((pmax[fn] >= 0.90).sum()),
        },
        "all_errors_618": {"n": int(wrong.sum()), "n_pred_prob_ge_0.70": int((pmax[wrong] >= 0.70).sum()),
                           "pred_prob_median": float(np.median(pmax[wrong]))},
        "correct_618": {"n": int((~wrong).sum()), "pred_prob_median": float(np.median(pmax[~wrong]))},
        "auroc_rounding_sensitivity": {
            "note": "확률을 소수 8자리로 반올림하면 생기는 동점 때문에 AUROC 가 바뀌는 정도 (첫 실행 CSV 의 문제)",
            "auroc_full_precision": auc_full, "auroc_rounded_8": auc_round,
            "macro_full": float(np.mean(auc_full)), "macro_rounded_8": float(np.mean(auc_round)),
            "n_p_normal_lt_5e-9": int((p[:, 0] < 5e-9).sum()),
            "n_true_normal_with_p_normal_lt_5e-9": int(((p[:, 0] < 5e-9) & (y == 0)).sum()),
        },
    }


def setup_font():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    plt.rcParams["font.family"] = "Apple SD Gothic Neo"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_confusion(plt, cm, m618):
    cm = np.array(cm)
    rate = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}장\n({rate[i, j]:.1%})", ha="center", va="center", fontsize=14,
                    color="white" if rate[i, j] > 0.55 else "black")
    ax.set_xticks(range(3), [f"예측 {k}" for k in CLASS_KO], fontsize=12)
    ax.set_yticks(range(3), [f"실제 {k}\n({n}장)" for k, n in zip(CLASS_KO, cm.sum(1))], fontsize=12)
    ax.set_title(f"실험 D, test 618장 혼동행렬 (행 = 실제, 괄호 = 행 비율)\n"
                 f"Macro-F1 {m618['macro_f1']:.3f} [95% {m618['ci']['macro_f1'][0]:.3f}~{m618['ci']['macro_f1'][1]:.3f}]",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "test_confusion.png", dpi=140)
    plt.close(fig)


def plot_val_test(plt, rows):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    lo, hi = 0.60, 0.88
    ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1, label="val = test 선")
    for r in rows:
        prim = r["id"] == "D"
        yerr = [[r["test"] - r["ci"][0]], [r["ci"][1] - r["test"]]]
        ax.errorbar(r["val"], r["test"], yerr=yerr, fmt="o", ms=11 if prim else 7,
                    color="tab:red" if prim else "tab:blue", ecolor="tab:red" if prim else "lightsteelblue",
                    capsize=3, zorder=3 if prim else 2)
        ax.annotate(LABEL[r["id"]] + (" (주 모델)" if prim else ""), (r["val"], r["test"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=10,
                    color="tab:red" if prim else "black", fontweight="bold" if prim else "normal")
    ax.set_xlim(0.79, 0.845)
    ax.set_ylim(lo, 0.82)
    ax.set_xlabel("val Macro-F1 (베스트 에폭, 1,030장)", fontsize=12)
    ax.set_ylabel("test Macro-F1 (618장, 막대 = 환자 단위 95% 구간)", fontsize=12)
    ax.set_title("val 점수와 test 점수 (모델 8개, test 는 선택에 쓰지 않음)", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "test_vs_val.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="SageMaker 평가 작업 이름 (주 결과)")
    ap.add_argument("--first-job", default=None, help="재실행이면 첫 작업 이름 (기록용)")
    args = ap.parse_args()

    aws = json.loads((OUT / "aws/evaluation_summary.json").read_text(encoding="utf-8"))
    local = json.loads((OUT / "local/evaluation_summary.json").read_text(encoding="utf-8"))
    sub = json.loads((ROOT / "runs/submissions" / f"{args.job}.json").read_text(encoding="utf-8"))
    assert aws["sagemaker_job_name"] == args.job and sub["status"] == "Completed"
    first = None
    if args.first_job:
        f = json.loads((ROOT / "runs/submissions" / f"{args.first_job}.json").read_text(encoding="utf-8"))
        first = {"name": args.first_job, "status": f["status"], "billable_time_seconds": f.get("billable_time_seconds"),
                 "evaluate_py_sha256": f.get("evaluate_py_sha256")}
    repro = compare_runs()
    rerun_check = compare_with_first_run()
    conf = confidence_analysis()

    vals = {mid: val_best(mid) for mid in RESULT_DIRS}
    allm = {r["model_id"]: r for r in read_csv(OUT / "aws/all_models_test.csv")}
    rows = [{"id": mid, "val": vals[mid]["macro_f1"], "test": float(allm[mid]["test_macro_f1"]),
             "ci": [float(allm[mid]["test_macro_f1_ci_low"]), float(allm[mid]["test_macro_f1_ci_high"])]}
            for mid in RESULT_DIRS]

    plt = setup_font()
    m618 = aws["primary_618"]
    plot_confusion(plt, m618["confusion"], m618)
    plot_val_test(plt, rows)

    with open(OUT / "reproducibility.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "value", "note"])
        for name in ("618", "624"):
            r = repro[name]
            w.writerow([f"primary_D_{name}_pred_agree", f"{r['pred_agree']}/{r['n']}", f"{r['pred_agree_rate']:.4%}"])
            w.writerow([f"primary_D_{name}_max_abs_prob_diff", f"{r['max_abs_prob_diff']:.2e}", "test_predictions.csv 는 전체 정밀도(.17g)"])
        w.writerow(["metrics_csv_max_abs_diff", f"{repro['metrics_max_abs_diff']['value']:.2e}", repro["metrics_max_abs_diff"]["where"]])
        w.writerow(["misclassified_same_images_and_types", repro["misclassified_same_images_and_types"],
                    f"오분류 {repro['misclassified_count']}장"])
        w.writerow(["confusion_matrix_csv_identical", repro["confusion_identical"], ""])

    summary = {
        "note": "최종 test 평가 통합 요약. 주 결과는 AWS(SageMaker) 실행, iMac 실행은 재현 확인용.",
        "evaluation_plan_sha256": aws["evaluation_plan"]["sha256"],
        "evaluation_plan_sha256_now": sha256_file(OUT / "evaluation_plan.md"),
        "split_manifest_sha256": aws["split_manifest_sha256"],
        "sagemaker_job": {"name": args.job, "status": sub["status"], "instance_type": sub["instance_type"],
                          "training_time_seconds": sub.get("training_time_seconds"),
                          "billable_time_seconds": sub.get("billable_time_seconds"),
                          "evaluate_py_sha256": sub.get("evaluate_py_sha256"),
                          "retry": first is not None, "retry_of": first, "rerun_reason": sub.get("rerun_reason")},
        "rerun_check_vs_first_run": rerun_check,
        "code_history_note": ("첫 실행(AWS 제출 전): iMac 에서 evaluate.py 를 한 번 실행했고, 그 뒤 보조 분석 5-2 에 '폐렴 사진만의 정답 비율' "
                              "열(계획 밖, 결과를 본 뒤 추가)과 클래스가 없는 묶음의 구간 NaN 처리를 추가해 AWS 작업 1을 실행했다. "
                              "재실행(리뷰 지적): test_predictions.csv 확률을 소수 8자리에서 전체 정밀도(.17g)로 바꾸고, 저장 후 CSV 로 "
                              "지표를 재계산해 확인하는 검사와, __getitem__ 이 실제로 돌려준 행으로 연 영상을 세는 기록을 추가했다. "
                              "이번에는 AWS 를 먼저 실행하고 iMac 은 그 뒤 재현용으로 실행했다. 모델·규칙·전처리·지표 정의는 바꾸지 않았다."),
        "model_provenance": {
            "B,C,D,P1,P2,P3,P4": "runs/results/<실험>/model/best_model.pt 와 바이트 단위 동일 (리뷰에서 확인)",
            "A": ("s3://aws-say3-student-dataset/students/<학생ID>/P1/cxr/experiments/"
                  "p1-cxr-<학생ID>-a-20261019-101748/output/model.tar.gz 안의 best_model.pt·metadata.json 과 동일 "
                  "(2026-10-19 재실행 전 내려받아 sha256 7c3a14d1… 일치, metadata 바이트 동일 확인; 로컬 runs/results/a-…/ 에는 model/ 이 없음)"),
        },
        "confidence_analysis_primary_618": conf,
        "aws_run": {k: aws[k] for k in ("device", "versions", "started_at_kst", "finished_at_kst", "elapsed_seconds")},
        "local_run": {k: local[k] for k in ("device", "versions", "started_at_kst", "finished_at_kst", "elapsed_seconds")},
        "checks": aws["checks"], "image_counts": aws["image_counts"], "model_sha256": aws["model_sha256"],
        "primary_model": "D",
        "primary_618": {k: m618[k] for k in ("macro_f1", "macro_auroc", "accuracy", "precision", "recall", "f1",
                                              "auroc", "confusion", "n_images", "n_patients", "ci", "zero_division_classes")},
        "primary_624": {k: aws["primary_624"][k] for k in ("macro_f1", "macro_auroc", "accuracy", "recall", "ci")},
        "val_vs_test": {mid: {"val_macro_f1": vals[mid]["macro_f1"], "val_epoch": vals[mid]["epoch"],
                              "val_recall": vals[mid]["recall"], "test_macro_f1": float(allm[mid]["test_macro_f1"]),
                              "test_recall": [float(allm[mid][f"test_recall_{s}"]) for s in ("normal", "bacterial", "viral")],
                              "val_rank": int(allm[mid]["val_rank"]), "test_rank": int(allm[mid]["test_rank"])}
                        for mid in RESULT_DIRS},
        "subgroup_overlap": aws["subgroup_overlap"],
        "misclassified_counts_618": aws["misclassified_counts_618"],
        "reproducibility_aws_vs_local": repro,
        "sklearn_crosscheck": {"aws": aws["sklearn_crosscheck_primary_618"], "local": local["sklearn_crosscheck_primary_618"]},
    }
    with open(OUT / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 숫자 표 (보고서 초안용)
    L = ["| 모델 | val Macro-F1 (에폭) | test Macro-F1 [95%] | val→test | val 정상 recall | test 정상 recall | test 세균 recall | test 바이러스 recall | val 순위 | test 순위 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for mid in RESULT_DIRS:
        v, a = vals[mid], allm[mid]
        L.append(f"| {LABEL[mid]}{' (주)' if mid == 'D' else ''} | {v['macro_f1']:.3f} ({v['epoch']}) | {float(a['test_macro_f1']):.3f} "
                 f"[{float(a['test_macro_f1_ci_low']):.3f}, {float(a['test_macro_f1_ci_high']):.3f}] | "
                 f"{float(a['test_macro_f1']) - v['macro_f1']:+.3f} | {v['recall'][0]:.3f} | {float(a['test_recall_normal']):.3f} | "
                 f"{float(a['test_recall_bacterial']):.3f} | {float(a['test_recall_viral']):.3f} | {a['val_rank']} | {a['test_rank']} |")
    (OUT / "test_report_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps(repro, ensure_ascii=False, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
