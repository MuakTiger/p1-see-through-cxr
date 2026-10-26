"""결과보고서에 쓰는 숫자를 저장된 결과 파일(shift_results.csv 등)에서 읽는다.

손으로 옮겨 적은 숫자는 쓰지 않는다. 계산한 값은 아래 assert 로 `reports/equipment_v2/lung/lung_summary.md`,
`vaccine_summary.md` 에 적힌 값과 맞는지 확인한다 (다르면 보고서 만들기가 멈춘다).
"""
import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EQ = ROOT / "reports/equipment_v2"

DEVICE = ["dose", "motion", "scatter", "kvp", "sensor"]       # 장비 설정 5종 (연습에서 본 필터)
POST = ["sharpen", "equalize", "jpeg"]                          # 제조사 후처리·전송 3종 (학습에 쓰지 않음)
APPENDIX = ["blur", "bright", "dark"]                           # 부록 3종
MAIN = DEVICE + POST
NAME_KO = {
    "dose": "관전류↓ (선량)", "motion": "노출시간 ↑", "scatter": "산란", "kvp": "관전압 ↑",
    "sensor": "센서 해상도", "sharpen": "경계 강조", "equalize": "대비 보정", "jpeg": "JPEG 압축",
    "blur": "흐림", "bright": "밝게", "dark": "어둡게",
}
SPLIT_N = {"val": (1030, 266), "test": (618, 231)}             # (전체 장수, 정상 장수)
SEEDS = (42, 43, 44)


def model_dirs(split):
    """모델 묶음 -> {seed: 결과 폴더}. 같은 환경끼리만 비교한다 (AWS 학습 작업 대 JupyterLab)."""
    return {
        "aws_b": {42: EQ / f"{split}_B512", 43: EQ / f"lung/{split}_b512-s43", 44: EQ / f"lung/{split}_b512-s44"},
        "lung": {s: EQ / f"lung/{split}_l-s{s}" for s in SEEDS},
        "lung_aug": {s: EQ / f"lung/{split}_lv-s{s}" for s in SEEDS},
        "jl_b": {s: EQ / f"vac/{split}_b-s{s}" for s in SEEDS},
        "jl_v": {s: EQ / f"vac/{split}_v-s{s}" for s in SEEDS},
    }


def load_rows(folder):
    rows = {}
    with open(Path(folder) / "shift_results.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["shift"], int(r["level"]))
            rows[key] = {
                "param": r["param_value"],
                "f1": float(r["macro_f1"]), "auroc": float(r["macro_auroc"]),
                "rec_n": float(r["recall_normal"]), "rec_b": float(r["recall_bacterial"]), "rec_v": float(r["recall_viral"]),
                "delta": float(r["delta_f1"]), "lo": float(r["delta_f1_ci_low"]), "hi": float(r["delta_f1_ci_high"]),
                "changed": float(r["changed_pred_pct"]),
                "share_n": float(r["pred_share_normal"]), "share_b": float(r["pred_share_bacterial"]),
                "share_v": float(r["pred_share_viral"]),
            }
    return rows


def summarize(rows, split):
    """모델 하나의 요약: 원본 점수, 필터 묶음 평균, 강한 단계 점수, 폐렴을 정상으로 놓친 장수."""
    n_all, n_norm = SPLIT_N[split]
    clean = rows[("clean", 0)]
    mean = lambda shifts: float(np.mean([rows[(s, l)]["f1"] for s in shifts for l in (1, 2, 3)]))
    missed = round(clean["share_n"] * n_all - clean["rec_n"] * n_norm)    # 폐렴인데 정상으로 예측한 장수
    return {
        "orig_f1": clean["f1"], "auroc": clean["auroc"], "rec_n": clean["rec_n"], "rec_b": clean["rec_b"], "rec_v": clean["rec_v"],
        "main24": mean(MAIN), "learned5": mean(DEVICE), "unseen3": mean(POST), "appendix9": mean(APPENDIX),
        "drop24": mean(MAIN) - clean["f1"],
        "strong": {s: rows[(s, 3)]["f1"] for s in DEVICE + POST + APPENDIX},
        "strong_drop": {s: rows[(s, 3)]["f1"] - clean["f1"] for s in DEVICE + POST + APPENDIX},
        "pneu_missed": missed, "pneu_sens": 1 - missed / (n_all - n_norm),
    }


def load_all():
    """{묶음: {split: {seed: summary}}} 와 seed 42 AWS 기준 모델의 원자료 {split: rows}."""
    out, raw = {}, {}
    for split in ("val", "test"):
        for fam, per_seed in model_dirs(split).items():
            for s, folder in per_seed.items():
                rows = load_rows(folder)
                out.setdefault(fam, {}).setdefault(split, {})[s] = summarize(rows, split)
                if fam == "aws_b" and s == 42:
                    raw[split] = rows
    return out, raw


def agg(summaries, key):
    """seed 별 요약 -> (평균, 표준편차(ddof=1), 값 목록)."""
    v = [summaries[s][key] for s in sorted(summaries)]
    return float(np.mean(v)), float(np.std(v, ddof=1)), v


def agg_strong(summaries, shift, drop=False):
    k = "strong_drop" if drop else "strong"
    v = [summaries[s][k][shift] for s in sorted(summaries)]
    return float(np.mean(v)), v


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def val_test_changes(raw):
    """seed 42 AWS 기준 모델의 본편 24조건에서 val 대비 test 의 F1 변화 (원본 대비)."""
    pts = []
    for s in MAIN:
        for l in (1, 2, 3):
            pts.append((s, l, raw["val"][(s, l)]["delta"], raw["test"][(s, l)]["delta"]))
    return pts


def check(data):
    """lung_summary.md / vaccine_summary.md 에 적힌 값과 맞는지 확인한다."""
    def near(a, b, tol=6e-4):
        assert abs(a - b) <= tol, (a, b)

    A, L, LV, JB, JV = (data[k] for k in ("aws_b", "lung", "lung_aug", "jl_b", "jl_v"))
    near(agg(A["val"], "orig_f1")[0], 0.8394); near(agg(A["val"], "orig_f1")[1], 0.0067)
    near(agg(A["test"], "orig_f1")[0], 0.705); near(agg(A["test"], "orig_f1")[1], 0.033)
    near(agg(L["val"], "orig_f1")[0], 0.826); near(agg(L["test"], "orig_f1")[0], 0.775)
    near(agg(A["val"], "main24")[0], 0.795); near(agg(A["test"], "main24")[0], 0.639)
    near(agg(L["val"], "main24")[0], 0.772); near(agg(L["test"], "main24")[0], 0.704)
    near(agg(A["val"], "drop24")[0], -0.044); near(agg(L["val"], "drop24")[0], -0.054)
    near(agg(A["test"], "drop24")[0], -0.066); near(agg(L["test"], "drop24")[0], -0.070)
    near(agg(JB["val"], "learned5")[0], 0.774); near(agg(JV["val"], "learned5")[0], 0.815)
    near(agg(JB["test"], "learned5")[0], 0.580); near(agg(JV["test"], "learned5")[0], 0.700)
    near(agg(L["test"], "learned5")[0], 0.660); near(agg(LV["test"], "learned5")[0], 0.721)
    near(agg(JB["test"], "unseen3")[0], 0.694); near(agg(JV["test"], "unseen3")[0], 0.706)
    near(agg(L["test"], "unseen3")[0], 0.779); near(agg(LV["test"], "unseen3")[0], 0.734)
    near(agg(A["test"], "rec_n")[0], 0.483); near(agg(L["test"], "rec_n")[0], 0.605)


if __name__ == "__main__":
    d, raw = load_all()
    check(d)
    A = d["aws_b"]
    print("AWS 기준 val 원본", agg(A["val"], "orig_f1"), "test", agg(A["test"], "orig_f1"))
    print("폐렴을 정상으로 놓친 장수(test)", {s: A["test"][s]["pneu_missed"] for s in SEEDS},
          "민감도", {s: round(A["test"][s]["pneu_sens"], 4) for s in SEEDS})
    print("폐 모델(test)", {s: d["lung"]["test"][s]["pneu_missed"] for s in SEEDS})
    pts = val_test_changes(raw)
    print("스피어만(24조건, seed 42)", round(spearman([p[2] for p in pts], [p[3] for p in pts]), 3),
          "피어슨", round(float(np.corrcoef([p[2] for p in pts], [p[3] for p in pts])[0, 1]), 3))
    both = sum(1 for p in pts if p[2] < 0 and p[3] < 0)
    print("둘 다 하락한 조건", both)
    for s in MAIN:
        print(s, "val 강 하락 평균", round(agg_strong(A["val"], s, True)[0], 3), "test", round(agg_strong(A["test"], s, True)[0], 3))
    print("검증 통과")
