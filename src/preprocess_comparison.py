"""P 시리즈 2단계 비교표: 전처리 P0~P4 로 각각 학습한 모델을 val 기준으로 나란히 놓는다 (test 는 쓰지 않는다).

P0 = 실험 B (같은 설정에 전처리만 원본). P1~P4 는 B 설정에서 입력 전처리만 바꿨다.
실험 D(노이즈 증강)는 전처리가 아니라 "학습 때만 쓰는 증강" 이므로 참고 행으로만 붙인다.

입력 (모두 이미 내려받은 파일):
    runs/results/<폴더>/history.csv, val_predictions.csv
    reports/shift_test*/shift_results.csv     장비 차이 민감도 시험 결과
    SageMaker describe_training_job           학습 시간 TrainingTimeInSeconds, 상태 (읽기 전용 조회)

사용 예 (p1 폴더에서):
    .venv/bin/python src/preprocess_comparison.py

결과:
    reports/preprocess/preprocess_comparison.csv   모델당 한 줄 (기획안 파일 이름)
    reports/preprocess/preprocess_comparison.md    표 (해석 절은 이 스크립트가 쓰지 않는다. 숫자를 보고 사람이 덧붙인다.
                                                   다시 실행하면 손으로 쓴 해석 절이 지워지므로 먼저 따로 옮겨 둘 것)
    reports/preprocess/preprocess_comparison.png   그림 (왼쪽: val 점수, 오른쪽: 장비 차이별 Macro-F1)
"""
import csv
import sys
from pathlib import Path

import boto3
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import train as T  # noqa: E402  (Macro-F1 정의를 학습 코드와 똑같이 쓰기 위해)
from preprocess import PREPROCESS  # noqa: E402

PROFILE, REGION = "<학생ID>", "ap-northeast-2"
JOB_PREFIX = "p1-cxr-<학생ID>-"
OUT = ROOT / "reports/preprocess"
LAST_N = 5
LFR_TEXT = "미측정(Grad-CAM 단계에서)"
# (이름, 결과 폴더 glob, 설명, 민감도 결과 폴더)
MODELS = [
    ("P0", "b-20261019-103853", PREPROCESS["P0"]["desc"] + " = 실험 B", "reports/shift_test"),
    ("P1", "p1-2*", PREPROCESS["P1"]["desc"], "reports/shift_test_P1"),
    ("P2", "p2-2*", PREPROCESS["P2"]["desc"], "reports/shift_test_P2"),
    ("P3", "p3-2*", PREPROCESS["P3"]["desc"], "reports/shift_test_P3"),
    ("P4", "p4-2*", PREPROCESS["P4"]["desc"], "reports/shift_test_P4"),
]
REFERENCE = ("D(참고)", "d-20261019-134621", "P0 + 노이즈 증강 (전처리 아님, 학습에만 쓰는 증강)", "reports/shift_test_D")
# 민감도 핵심 숫자: (열 이름, shift, level, 표 제목)
SHIFT_KEYS = [
    ("shift_clean", "clean", 0, "원본"),
    ("shift_noise8", "noise", 2, "노이즈 8"),
    ("shift_noise16", "noise", 3, "노이즈 16"),
    ("shift_blur2", "blur", 3, "흐림 σ2"),
    ("shift_sharpen2", "sharpen", 3, "경계 강조 α2"),
    ("shift_bright055", "bright", 3, "밝게 γ0.55"),
    ("shift_dark18", "dark", 3, "어둡게 γ1.8"),
    ("shift_contrast055", "contrast", 3, "대비 0.55"),
    ("shift_jpeg20", "jpeg", 3, "JPEG 20"),
]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_dir(pattern):
    dirs = sorted((ROOT / "runs/results").glob(pattern))
    if len(dirs) != 1:
        raise SystemExit(f"runs/results/{pattern} 폴더가 정확히 1개여야 합니다: {[d.name for d in dirs]}")
    return dirs[0].name


def summarize(name, pattern, desc, shift_dir, sm):
    res_dir = find_dir(pattern)
    hist = read_csv(ROOT / "runs/results" / res_dir / "history.csv")
    best = [h for h in hist if h["is_best"] == "True"][-1]  # 마지막으로 갱신된 베스트 = 선택된 에폭
    last = np.array([float(h["val_macro_f1"]) for h in hist[-LAST_N:]])
    preds = read_csv(ROOT / "runs/results" / res_dir / "val_predictions.csv")
    y = np.array([int(r["true_label"]) for r in preds])
    p = np.array([int(r["pred_label"]) for r in preds])
    f1_check, _, _ = T.macro_f1_and_recall(y.tolist(), p.tolist())
    if abs(f1_check - float(best["val_macro_f1"])) > 1e-5:
        raise SystemExit(f"{name}: val_predictions 의 Macro-F1 {f1_check} 이 history 베스트와 다릅니다")
    job = JOB_PREFIX + res_dir  # 결과 폴더 이름 = 작업 이름에서 앞부분을 뺀 것
    d = sm.describe_training_job(TrainingJobName=job)
    row = {
        "model": name, "preprocess_desc": desc, "job_name": job, "status": d["TrainingJobStatus"],
        "training_time_seconds": d.get("TrainingTimeInSeconds"),
        "best_epoch": int(best["epoch"]), "epochs": len(hist),
        "val_macro_f1": float(best["val_macro_f1"]), "val_macro_auroc": float(best["val_macro_auroc"]),
        "recall_normal": float(best["val_recall_normal"]), "recall_bacterial": float(best["val_recall_bacterial"]),
        "recall_viral": float(best["val_recall_viral"]),
        f"last{LAST_N}_f1_mean": float(last.mean()), f"last{LAST_N}_f1_std": float(last.std(ddof=1)),
        "pred_share_normal": float((p == 0).mean()), "pred_share_bacterial": float((p == 1).mean()),
        "pred_share_viral": float((p == 2).mean()),
        "lfr": LFR_TEXT,
    }
    sr = read_csv(ROOT / shift_dir / "shift_results.csv")
    by_key = {(r["shift"], int(r["level"])): float(r["macro_f1"]) for r in sr}
    for key, shift, level, _ in SHIFT_KEYS:
        row[key] = by_key[(shift, level)]
    shifted = [float(r["macro_f1"]) for r in sr if r["shift"] != "clean"]
    if len(shifted) != 24:
        raise SystemExit(f"{name}: 민감도 조건이 24개가 아닙니다 ({len(shifted)}개)")
    row["shift_mean24"] = float(np.mean(shifted))
    row["shift_dir"] = shift_dir
    return row


def plot(rows, ref, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    plt.rcParams["font.family"] = "Apple SD Gothic Neo"
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 7), gridspec_kw={"width_ratios": [1, 1.6]})
    names = [r["model"] for r in rows]
    x = np.arange(len(rows))
    w = 0.38
    ax1.bar(x - w / 2, [r["val_macro_f1"] for r in rows], w, color="#4C72B0", label="val Macro-F1 (베스트 에폭)")
    ax1.errorbar(x - w / 2, [r[f"last{LAST_N}_f1_mean"] for r in rows], yerr=[r[f"last{LAST_N}_f1_std"] for r in rows],
                 fmt="o", color="black", capsize=5, label=f"마지막 {LAST_N}에폭 평균 ± 표준편차")
    ax1.bar(x + w / 2, [r["recall_viral"] for r in rows], w, color="#DD8452", label="바이러스성 recall (베스트 에폭)")
    for i, r in enumerate(rows):
        ax1.text(i - w / 2, r["val_macro_f1"] + 0.012, f"{r['val_macro_f1']:.3f}", ha="center", fontsize=10)
        ax1.text(i + w / 2, r["recall_viral"] + 0.012, f"{r['recall_viral']:.3f}", ha="center", fontsize=10)
    ax1.set_xticks(x, [PREPROCESS[n]["title"].replace(" ", "\n", 1) for n in names], fontsize=11)
    ax1.set_ylim(0.4, 1.0)
    ax1.set_ylabel("점수")
    ax1.set_title("원본 val 점수 (val 1,030장)", fontsize=14)
    ax1.legend(loc="upper center", fontsize=10, ncol=2, frameon=False)
    ax1.grid(axis="y", alpha=0.3)

    keys = [k for k, _, _, _ in SHIFT_KEYS] + ["shift_mean24"]
    titles = [t for _, _, _, t in SHIFT_KEYS] + ["24조건\n평균"]
    all_rows = rows + [ref]
    mat = np.array([[r[k] for k in keys] for r in all_rows])
    im = ax2.imshow(mat, cmap="RdYlGn", vmin=0.3, vmax=0.9, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax2.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=11,
                     color="white" if mat[i, j] < 0.4 else "black", fontweight="bold" if j == 0 else "normal")
    ax2.set_xticks(range(len(keys)), [t.replace(" ", "\n", 1) for t in titles], fontsize=10)
    ax2.set_yticks(range(len(all_rows)), [PREPROCESS[r["model"]]["title"] if r["model"] in PREPROCESS
                                          else "D 노이즈 증강\n(참고, 전처리 아님)" for r in all_rows], fontsize=11)
    ax2.axhline(len(rows) - 0.5, color="black", lw=2)
    ax2.set_title("장비 차이 민감도: 필터를 건 val 의 Macro-F1 (재학습 없음, 가장 강한 단계 위주)", fontsize=14)
    fig.colorbar(im, ax=ax2, fraction=0.03, pad=0.02, label="Macro-F1")
    fig.suptitle("P 시리즈: 입력 전처리만 바꿔 학습한 모델 비교 (실험 B 설정, seed 42 단일 실행)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    sm = boto3.Session(profile_name=PROFILE, region_name=REGION).client("sagemaker")
    rows = [summarize(*m, sm) for m in MODELS]
    ref = summarize(*REFERENCE, sm)
    OUT.mkdir(parents=True, exist_ok=True)

    fields = list(rows[0].keys())
    with open(OUT / "preprocess_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows + [ref]:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else ("" if v is None else v)) for k, v in r.items()})
    plot(rows, ref, OUT / "preprocess_comparison.png")

    L = ["# P 시리즈: 전처리별 모델 비교 (val 기준)", "",
         "- 모든 모델은 실험 B 설정(ResNet18 layer4 미세조정, Adam lr 0.001/0.0001, 배치 16, 30에폭, seed 42, 증강 없음, "
         "클래스 가중치 없음)이고 **입력 전처리 하나만** 다르다. P0 = 실험 B.",
         "- 전처리는 224×224 회색조에서 적용하는 결정적 변환이며 train·val 모두에 똑같이 적용했다 (정의: `training/preprocess.py`).",
         "- 숫자는 val 1,030장 (정상 266 / 세균성 488 / 바이러스성 276) 기준. test 는 쓰지 않았다. "
         f"'마지막 {LAST_N}에폭'은 에폭 26~30 의 val Macro-F1 (표본 표준편차).",
         "- 민감도 시험 순서: 224 회색조 → 장비 필터(촬영 단계 차이) → 모델의 전처리 → 텐서. 필터 강도는 기존 시험과 같다.",
         "- 표는 `src/preprocess_comparison.py` 가 만들었다. CSV 는 `preprocess_comparison.csv`, 그림은 `preprocess_comparison.png`.",
         "", "## 1. 기본 지표", "",
         "| 모델 | 전처리 | 작업 이름 | 상태 | 학습 시간(초) | 베스트 에폭 | val Macro-F1 | val Macro AUROC | 정상 recall | 세균성 recall | 바이러스성 recall | LFR |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows + [ref]:
        L.append(f"| {r['model']} | {r['preprocess_desc']} | `{r['job_name']}` | {r['status']} | {r['training_time_seconds']} | "
                 f"{r['best_epoch']} | {r['val_macro_f1']:.4f} | {r['val_macro_auroc']:.4f} | {r['recall_normal']:.3f} | "
                 f"{r['recall_bacterial']:.3f} | {r['recall_viral']:.3f} | {r['lfr']} |")
    L += ["", f"## 2. 에폭 간 흔들림과 예측 비율", "",
          "실제 val 비율: 정상 25.8% / 세균성 47.4% / 바이러스성 26.8%", "",
          f"| 모델 | 베스트 val Macro-F1 | 마지막 {LAST_N}에폭 평균 | 표준편차 | 예측 비율 정상/세균성/바이러스성 (%) |",
          "|---|---|---|---|---|"]
    for r in rows + [ref]:
        share = "/".join(f"{100 * r[k]:.1f}" for k in ("pred_share_normal", "pred_share_bacterial", "pred_share_viral"))
        L.append(f"| {r['model']} | {r['val_macro_f1']:.4f} | {r[f'last{LAST_N}_f1_mean']:.4f} | "
                 f"{r[f'last{LAST_N}_f1_std']:.4f} | {share} |")
    L += ["", "## 3. 장비 차이 민감도 핵심 숫자 (Macro-F1, 재학습 없이 val 에 필터 적용)", "",
          "| 모델 | " + " | ".join(t for _, _, _, t in SHIFT_KEYS) + " | 24조건 평균 |",
          "|---|" + "---|" * (len(SHIFT_KEYS) + 1)]
    for r in rows + [ref]:
        L.append(f"| {r['model']} | " + " | ".join(f"{r[k]:.3f}" for k, _, _, _ in SHIFT_KEYS) + f" | {r['shift_mean24']:.3f} |")
    L += ["", "원본 열은 필터 경로(회색조 → 224 → 모델 전처리) 기준이며 학습 경로 점수와 같다. "
          "D 는 전처리가 아니라 학습 때만 쓰는 노이즈 증강(p 0.5, std 0~10)이며 참고로 붙였다.", ""]
    (OUT / "preprocess_comparison.md").write_text("\n".join(L), encoding="utf-8")
    for r in rows + [ref]:
        print(f"{r['model']}: F1 {r['val_macro_f1']:.4f}, 바이러스 recall {r['recall_viral']:.3f}, "
              f"마지막5 {r[f'last{LAST_N}_f1_mean']:.4f}±{r[f'last{LAST_N}_f1_std']:.4f}, 24조건 평균 {r['shift_mean24']:.3f}")
    print(f"저장: {OUT.relative_to(ROOT)}/preprocess_comparison.csv, .md, .png")


if __name__ == "__main__":
    main()
