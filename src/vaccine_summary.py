"""예방접종(v) 3개와 기준(b) 3개의 val·test 필터 시험 결과를 한 표로 모으고, 계획 문서의 판정 규칙을 적용한다 (저장된 결과 파일만 읽음, 모델을 돌리지 않음).

계획: reports/equipment_v2/vaccine_plan.md 3·4장(val), reports/equipment_v2/test_plan_vaccine.md 3장(test).
  - 묶음: 학습에 쓴 5종(선량·조사시간·산란·kVp·센서, 15조건), 쓰지 않은 3종(경계 강조·대비 보정·JPEG, 9조건), 부록 3종(흐림·밝게·어둡게, 9조건).
  - 판정: 예방접종 − 기준의 평균 차이가 기준 3개의 seed 표준편차(같은 지표, ddof=1)의 2배 이상일 때만 "차이가 있다", 아니면 "차이를 말할 수 없다".
  - 원본 F1이 기준보다 표준편차 이상 낮으면 "원본 성능 손실"로 함께 적는다.
  - 학습에 쓴 5종은 배운 유형이라 방어력의 증거로 쓰지 않고, 쓰지 않은 3종이 방어력을 보는 부분이다.

사용 예 (p1 폴더에서):
    .venv/bin/python src/vaccine_summary.py
결과: reports/equipment_v2/vaccine_summary.md, vaccine_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VAC = ROOT / "reports/equipment_v2/vac"
SEEDS = [42, 43, 44]
GROUPS = {
    "학습에 쓴 5종 (15조건)": ["dose", "motion", "scatter", "kvp", "sensor"],
    "쓰지 않은 3종 (9조건)": ["sharpen", "equalize", "jpeg"],
    "부록: 흐림·밝게·어둡게 (9조건)": ["blur", "bright", "dark"],
}
KO = {"dose": "선량", "motion": "조사시간", "scatter": "산란", "kvp": "kVp", "sensor": "센서",
      "sharpen": "경계 강조", "equalize": "대비 보정", "jpeg": "JPEG", "blur": "흐림", "bright": "밝게", "dark": "어둡게"}
REF = {"val": ROOT / "reports/equipment_v2/val_B512", "test": ROOT / "reports/equipment_v2/test_B512"}


def load(split, kind, seed):
    p = VAC / f"{split}_{kind}-s{seed}" / "shift_results.csv"
    return pd.read_csv(p) if p.is_file() else None


def metrics(df):
    clean = df[df["shift"] == "clean"].iloc[0]
    s = df[df["shift"] != "clean"]
    out = {"원본 F1": float(clean.macro_f1), "AUROC": float(clean.macro_auroc), "정상 recall": float(clean.recall_normal)}
    for g, k in GROUPS.items():
        out[g] = float(s[s["shift"].isin(k)].macro_f1.mean())
        out[g + " 하락"] = out[g] - out["원본 F1"]
    return out


def fmt(x):
    return f"{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}" if len(x) > 1 else f"{np.mean(x):.3f}"


def verdict(v, b, higher_is_better=True):
    d = np.mean(v) - np.mean(b)
    sd = np.std(b, ddof=1)
    k = d / sd if sd > 0 else float("nan")
    if abs(d) >= 2 * sd:
        txt = "차이 있음 (예방접종이 " + ("높음" if d > 0 else "낮음") + ")"
    else:
        txt = "차이를 말할 수 없음"
    return d, sd, k, txt


def main():
    L = ["# 예방접종 대 기준 요약 (seed 42·43·44, 계획 문서의 판정 규칙 적용)", "",
         "생성: `src/vaccine_summary.py` (저장된 결과 파일만 읽음). 판정: 예방접종 − 기준의 평균 차이가 **기준 3개의 seed 표준편차(ddof=1)의 2배 이상**일 때만 \"차이 있음\". "
         "학습에 쓴 5종은 배운 유형이라 방어력의 증거로 쓰지 않는다. 표준편차는 seed 3개에서 나온 거친 추정이다.", ""]
    rows = []
    for split in ("val", "test"):
        M = {}
        for kind in ("b", "v"):
            for sd in SEEDS:
                df = load(split, kind, sd)
                if df is not None:
                    M[(kind, sd)] = metrics(df)
        b = [M[("b", s)] for s in SEEDS if ("b", s) in M]
        v = [M[("v", s)] for s in SEEDS if ("v", s) in M]
        L += [f"## {split}", "", f"모델 수: 기준 {len(b)}개, 예방접종 {len(v)}개." + ("" if len(b) == len(v) == 3 else " **3개씩이 아니라 판정 규칙을 쓸 수 없다 (참고 수준).**"), ""]
        keys = ["원본 F1", "AUROC", "정상 recall"] + list(GROUPS) + [g + " 하락" for g in GROUPS]
        L += ["### seed별 값", "", "| 모델 | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for (kind, sd), m in sorted(M.items()):
            L.append(f"| {kind}-s{sd} | " + " | ".join(f"{m[k]:.3f}" for k in keys) + " |")
        ref = REF[split] / "shift_results.csv"
        if ref.is_file():
            m = metrics(pd.read_csv(ref))
            L.append("| (참고) AWS B512 seed 42 | " + " | ".join(f"{m[k]:.3f}" for k in keys) + " |")
        L += ["", "### 묶음별 평균 ± 표준편차와 판정", "",
              "| 지표 | 기준 | 예방접종 | 차이 (예방접종 − 기준) | 기준 표준편차 | 차이 ÷ 표준편차 | 판정 |", "|---|---|---|---|---|---|---|"]
        if len(b) >= 2 and len(v) >= 1:
            for k in keys:
                bb, vv = [m[k] for m in b], [m[k] for m in v]
                d, sdv, ratio, txt = verdict(vv, bb)
                if len(b) < 3 or len(v) < 3:
                    txt = "참고 수준 (seed 3개 미만)"
                note = " (배운 유형, 방어력 증거 아님)" if k.startswith("학습에 쓴") else ""
                L.append(f"| {k} | {fmt(bb)} | {fmt(vv)} | {d:+.3f} | {sdv:.3f} | {ratio:+.1f} | {txt}{note} |")
                rows.append({"split": split, "metric": k, "base_mean": np.mean(bb), "base_sd": np.std(bb, ddof=1), "vac_mean": np.mean(vv),
                             "diff": d, "ratio": ratio, "verdict": txt})
        # 강 조건별
        L += ["", "### 강(3단계) 조건별 Macro-F1 (seed 평균)", "", "| 필터 | 기준 | 예방접종 | 차이 | 기준 seed 범위 |", "|---|---|---|---|---|"]
        for c in sum(GROUPS.values(), []):
            def lv(kind, sd):
                d = load(split, kind, sd)
                return float(d[(d["shift"] == c) & (d.level == 3)].macro_f1.iloc[0]) if d is not None else np.nan
            bb = [lv("b", s) for s in SEEDS]
            vv = [lv("v", s) for s in SEEDS]
            L.append(f"| {KO[c]} | {np.nanmean(bb):.3f} | {np.nanmean(vv):.3f} | {np.nanmean(vv) - np.nanmean(bb):+.3f} | {np.nanmin(bb):.3f}~{np.nanmax(bb):.3f} |")
        L.append("")
    L += ["## 읽을 때 주의", "",
          "- 환경(JupyterLab torch 2.8)이 AWS 학습 작업(torch 2.2)의 B512와 달라 참고 행은 환경 차이가 섞인다.",
          "- 기준 3개의 표준편차가 커서(에폭 선택 규칙의 흔들림 포함) 판정이 엄격해진다. 규칙은 결과를 보고 바꾸지 않았다.",
          "- 필터는 실제 장비가 아닌 흉내이고 필터끼리 강도가 보정되지 않았다. test 절대 점수는 낮고 seed에 따라 크게 흔들리므로 변화량과 방향을 중심으로 해석한다.",
          "- test는 이미 열린 적이 있는 데이터에서의 사전 지정 추가 분석이다(`test_plan_vaccine.md` 1장)."]
    out = ROOT / "reports/equipment_v2/vaccine_summary.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    pd.DataFrame(rows).round(4).to_csv(ROOT / "reports/equipment_v2/vaccine_summary.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
