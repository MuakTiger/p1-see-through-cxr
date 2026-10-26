"""실험 A·B·C·D 종합 비교표를 만든다 (val 기준, test 는 쓰지 않는다).

입력 (모두 이미 내려받은 파일):
    runs/results/<실험>/history.csv          에폭별 val 지표
    runs/results/<실험>/val_predictions.csv  베스트 에폭의 val 예측 (예측 비율, 혼동행렬 계산용)
    reports/shift_test*/shift_results.csv    장비 차이 민감도 시험 결과 (B·C·D)
    SageMaker describe_training_job          학습 시간 TrainingTimeInSeconds (읽기 전용 조회)

사용 예 (p1 폴더에서):
    .venv/bin/python src/experiments_summary.py

결과:
    reports/experiments_summary.csv   실험당 한 줄
    reports/experiments_summary.md    표 (해석 절은 이 스크립트가 쓰지 않는다. 숫자를 보고 사람이 덧붙인다)
"""
import csv
import json
import sys
from pathlib import Path

import boto3
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import train as T  # noqa: E402  (Macro-F1 정의를 학습 코드와 똑같이 쓰기 위해)

PROFILE, REGION = "<학생ID>", "ap-northeast-2"
JOB_PREFIX = "p1-cxr-<학생ID>-"
CLASS_KO = ["정상", "세균성", "바이러스성"]
# (실험, 결과 폴더, 바꾼 조건, 민감도 결과 폴더)
EXPERIMENTS = [
    ("A", "a-20261019-101748", "ResNet18 몸통 고정, fc 만 학습 (기준선)", None),
    ("B", "b-20261019-103853", "A + layer4 미세조정 (lr 1e-4)", "reports/shift_test"),
    ("C", None, "B + 클래스 가중치 손실 (inverse, train 빈도로 계산)", "reports/shift_test_C"),
    ("D", "d-20261019-134621", "B + 노이즈 증강 (학습에만, p 0.5, std 0~10)", "reports/shift_test_D"),
]
# 민감도 핵심 숫자: (열 이름, shift, level, 표 제목)
SHIFT_KEYS = [
    ("shift_clean", "clean", 0, "원본"),
    ("shift_noise8", "noise", 2, "노이즈 8"),
    ("shift_noise16", "noise", 3, "노이즈 16"),
    ("shift_blur2", "blur", 3, "흐림 σ2"),
    ("shift_contrast055", "contrast", 3, "대비 0.55"),
    ("shift_jpeg20", "jpeg", 3, "JPEG 20"),
]
LAST_N = 5  # 마지막 몇 에폭의 val Macro-F1 흔들림을 볼지


def find_c_dir():
    dirs = sorted((ROOT / "runs/results").glob("c-2*"))
    if len(dirs) != 1:
        raise SystemExit(f"runs/results/c-* 폴더가 정확히 1개여야 합니다: {dirs}")
    return dirs[0].name


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(exp, res_dir, desc, shift_dir, sm):
    hist = read_csv(ROOT / "runs/results" / res_dir / "history.csv")
    best = [h for h in hist if h["is_best"] == "True"][-1]  # 마지막으로 갱신된 베스트 = 선택된 에폭
    last = np.array([float(h["val_macro_f1"]) for h in hist[-LAST_N:]])
    preds = read_csv(ROOT / "runs/results" / res_dir / "val_predictions.csv")
    y = np.array([int(r["true_label"]) for r in preds])
    p = np.array([int(r["pred_label"]) for r in preds])
    cm = T.confusion_counts(y, p)
    f1_check, _, _ = T.macro_f1_and_recall(y.tolist(), p.tolist())
    if abs(f1_check - float(best["val_macro_f1"])) > 1e-5:  # 예측 파일과 기록이 같은 에폭인지 확인
        raise SystemExit(f"{exp}: val_predictions 의 Macro-F1 {f1_check} 이 history 베스트와 다릅니다")
    job = JOB_PREFIX + res_dir  # 결과 폴더 이름 = 작업 이름에서 앞부분을 뺀 것
    d = sm.describe_training_job(TrainingJobName=job)
    row = {
        "experiment": exp, "change": desc, "job_name": job, "status": d["TrainingJobStatus"],
        "training_time_seconds": d.get("TrainingTimeInSeconds"),
        "best_epoch": int(best["epoch"]), "epochs": len(hist),
        "val_macro_f1": float(best["val_macro_f1"]), "val_macro_auroc": float(best["val_macro_auroc"]),
        "recall_normal": float(best["val_recall_normal"]), "recall_bacterial": float(best["val_recall_bacterial"]),
        "recall_viral": float(best["val_recall_viral"]),
        f"last{LAST_N}_f1_mean": float(last.mean()), f"last{LAST_N}_f1_std": float(last.std(ddof=1)),
        f"last{LAST_N}_f1_min": float(last.min()), f"last{LAST_N}_f1_max": float(last.max()),
        "pred_share_normal": float((p == 0).mean()), "pred_share_bacterial": float((p == 1).mean()),
        "pred_share_viral": float((p == 2).mean()),
        "n_val": len(y),
        "val_counts": np.bincount(y, minlength=3).tolist(),
    }
    for i in range(3):
        for j in range(3):
            row[f"cm_true{i}_pred{j}"] = int(cm[i, j])
    for key, _, _, _ in SHIFT_KEYS:
        row[key] = None
    if shift_dir:
        sr = {(r["shift"], int(r["level"])): float(r["macro_f1"]) for r in read_csv(ROOT / shift_dir / "shift_results.csv")}
        for key, shift, level, _ in SHIFT_KEYS:
            row[key] = sr[(shift, level)]
    return row


def f3(v):
    return "-" if v is None else f"{v:.3f}"


def main():
    sm = boto3.Session(profile_name=PROFILE, region_name=REGION).client("sagemaker")
    rows = []
    for exp, res_dir, desc, shift_dir in EXPERIMENTS:
        rows.append(summarize(exp, res_dir or find_c_dir(), desc, shift_dir, sm))
    vc = rows[0].pop("val_counts")
    for r in rows[1:]:
        if r.pop("val_counts") != vc:
            raise SystemExit("실험마다 val 클래스별 수가 다릅니다")
    n_val = sum(vc)

    out_csv = ROOT / "reports/experiments_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else ("" if v is None else v)) for k, v in r.items()})

    L = ["# 실험 A·B·C·D 종합 비교 (val 기준)", "",
         f"- 모든 숫자는 val {n_val:,}장 (정상 {vc[0]} / 세균성 {vc[1]} / 바이러스성 {vc[2]}) 기준이다. test 는 아직 쓰지 않았다.",
         "- 네 실험 모두 seed 42 단일 실행, ml.g4dn.xlarge, 30에폭, 베스트 에폭은 val Macro-F1 로 고름.",
         "- 표는 `src/experiments_summary.py` 가 history.csv, val_predictions.csv, shift_results.csv 에서 만들었다. "
         "CSV 는 `reports/experiments_summary.csv`.",
         f"- '마지막 {LAST_N}에폭'은 에폭 {rows[0]['epochs'] - LAST_N + 1}~{rows[0]['epochs']} 의 val Macro-F1 (표준편차는 표본 표준편차).",
         "", "## 1. 기본 지표", "",
         "| 실험 | 바꾼 조건 | 작업 이름 | 학습 시간(초) | 베스트 에폭 | val Macro-F1 | val Macro AUROC | 정상 recall | 세균성 recall | 바이러스성 recall |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['experiment']} | {r['change']} | `{r['job_name']}` | {r['training_time_seconds']} | {r['best_epoch']} | "
                 f"{r['val_macro_f1']:.4f} | {r['val_macro_auroc']:.4f} | {r['recall_normal']:.3f} | "
                 f"{r['recall_bacterial']:.3f} | {r['recall_viral']:.3f} |")
    L += ["", f"## 2. 에폭 간 흔들림 (마지막 {LAST_N}에폭 val Macro-F1)", "",
          "| 실험 | 베스트 | 평균 | 표준편차 | 최소 | 최대 |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['experiment']} | {r['val_macro_f1']:.4f} | {r[f'last{LAST_N}_f1_mean']:.4f} | "
                 f"{r[f'last{LAST_N}_f1_std']:.4f} | {r[f'last{LAST_N}_f1_min']:.4f} | {r[f'last{LAST_N}_f1_max']:.4f} |")
    L += ["", "## 3. 예측 비율과 혼동행렬 (베스트 에폭, 행 = 실제, 열 = 예측)", "",
          "실제 val 비율: " + " / ".join(f"{k} {100 * n / n_val:.1f}%" for k, n in zip(CLASS_KO, vc)), "",
          "| 실험 | 예측 비율 정상/세균성/바이러스성 (%) | 실제 정상 → 정상/세균/바이러스 | 실제 세균성 → | 실제 바이러스성 → |",
          "|---|---|---|---|---|"]
    for r in rows:
        share = "/".join(f"{100 * r[k]:.1f}" for k in ("pred_share_normal", "pred_share_bacterial", "pred_share_viral"))
        cms = [" / ".join(str(r[f"cm_true{i}_pred{j}"]) for j in range(3)) for i in range(3)]
        L.append(f"| {r['experiment']} | {share} | {cms[0]} | {cms[1]} | {cms[2]} |")
    L += ["", "## 4. 장비 차이 민감도 핵심 숫자 (Macro-F1, 재학습 없이 val 에 필터 적용)", "",
          "| 실험 | " + " | ".join(t for _, _, _, t in SHIFT_KEYS) + " |",
          "|---|" + "---|" * len(SHIFT_KEYS)]
    for r in rows:
        if r["shift_clean"] is None:
            continue
        L.append(f"| {r['experiment']} | " + " | ".join(f3(r[k]) for k, _, _, _ in SHIFT_KEYS) + " |")
    L += ["", "A 는 민감도 시험을 하지 않았다. 원본 열은 필터 경로(회색조 → 224) 기준이며 학습 경로 점수와 같다.", ""]
    out_md = ROOT / "reports/experiments_summary.md"
    out_md.write_text("\n".join(L), encoding="utf-8")
    print(json.dumps([{k: r[k] for k in ("experiment", "best_epoch", "val_macro_f1", "recall_viral", "training_time_seconds")}
                      for r in rows], ensure_ascii=False))
    print(f"저장: {out_csv.relative_to(ROOT)}, {out_md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
