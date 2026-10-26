"""과적합 줄이기 실험(r0~r4 × seed 3개)의 val 결과를 한 표로 모은다 (저장된 결과 파일만 읽음, 모델을 돌리지 않음, test 안 봄).

기준은 reports/equipment_v2/regularization_plan.md 3장: (원본 val F1 + 필터 24조건 평균 F1) / 2 를 seed 3개 평균으로 비교하고,
같은 설정의 seed 표준편차와 비교해 차이를 말할 수 있는지 본다. 과신 지표(마지막 에폭 val 손실, 틀린 예측 중 최대 확률 ≥ 0.9 비율)도 같이 적는다.

사용 예 (p1 폴더에서):
    .venv/bin/python src/reg_summary.py
결과: reports/equipment_v2/reg_summary.md
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "reports/equipment_v2/reg"
MAIN = ["dose", "motion", "scatter", "kvp", "sensor", "sharpen", "equalize", "jpeg"]
NAMES = {"r0": "r0 기준(바꾼 것 없음, 마지막 에폭)", "r1": "r1 에폭 5", "r2": "r2 라벨 스무딩 0.1", "r3": "r3 가중치 감쇠 0.1", "r4": "r4 특징 고정"}


def overconfidence(pred_path):
    v = pd.read_csv(pred_path)
    p = v[["p_normal", "p_bacterial", "p_viral"]].to_numpy()
    conf, pred = p.max(1), p.argmax(1)
    wrong = pred != v.true_label.to_numpy()
    bins = np.clip((conf * 10).astype(int), 0, 9)
    ece = 0.0
    for b in range(10):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - (~wrong[m]).mean())
    return {"mean_conf": conf.mean(), "err_conf90": (conf[wrong] >= 0.9).mean() if wrong.any() else float("nan"),
            "n_err": int(wrong.sum()), "ece": ece}


def collect():
    rows = []
    for d in sorted(REG.iterdir()):
        csv = d / "shift_results.csv"
        if not d.is_dir() or not csv.is_file():
            continue
        exp = d.name.rsplit("-", 2)[0]
        m = re.search(r"-s(\d+)$", exp)
        res_dir = next((ROOT / "runs/results").glob(f"{exp}-*"))
        s = pd.read_csv(csv)
        clean = float(s[s["shift"] == "clean"].macro_f1.iloc[0])
        stress = s[s["shift"].isin(MAIN)].macro_f1
        hist = pd.read_csv(res_dir / "history.csv")
        import json
        meta = json.loads((res_dir / "model/metadata.json").read_text(encoding="utf-8"))
        rows.append({"exp": exp, "config": exp.split("-")[0], "seed": int(m.group(1)), "clean": clean, "stress": float(stress.mean()),
                     "crit": (clean + float(stress.mean())) / 2, "peak": meta.get("peak_val_macro_f1"), "n_cond": len(stress),
                     "train_loss": float(hist.train_loss.iloc[-1]), "val_loss": float(hist.val_loss.iloc[-1]),
                     **overconfidence(res_dir / "val_predictions.csv")})
    return pd.DataFrame(rows)


def main():
    df = collect()
    ref = pd.read_csv(ROOT / "reports/equipment_v2/val_B512/shift_results.csv")
    ref_clean = float(ref[ref["shift"] == "clean"].macro_f1.iloc[0])
    ref_stress = float(ref[ref["shift"].isin(MAIN)].macro_f1.mean())
    ref_oc = overconfidence(ROOT / "runs/results/b512-20261022-111844/val_predictions.csv")
    L = ["# 과적합 줄이기 실험 결과 (val만, test는 열지 않음)", "",
         "설정은 `regularization_plan.md`에서 결과를 보기 전에 고정했다 (`src/reg_summary.py`로 계산). 모든 설정은 마지막 에폭의 모델이고 seed 42·43·44를 돌렸다. "
         "점수는 원본 val Macro-F1과 필터 24조건(새 5종 + 경계 강조·대비 보정·JPEG) 평균 Macro-F1이다. 선택 기준 = 두 값의 평균(seed 평균).", "",
         f"참고: 기존 B512(에폭 중 val 최댓값으로 선택, seed 1개)는 원본 {ref_clean:.3f}, 필터 24조건 평균 {ref_stress:.3f}, 기준값 {(ref_clean + ref_stress) / 2:.3f}, "
         f"틀린 예측 중 확률 0.9 이상 {ref_oc['err_conf90']:.1%}.", "",
         "## 설정별 요약 (seed 3개 평균 ± 표준편차)", "",
         "| 설정 | seed 수 | 원본 val F1 | 필터 24조건 평균 F1 | **기준값** | 에폭 중 최댓값(참고) | 마지막 에폭 val 손실 | 평균 확신도 | 틀린 예측 중 확률 ≥ 0.9 | ECE |", "|---|---|---|---|---|---|---|---|---|---|"]
    agg = {}
    for cfg, g in df.groupby("config"):
        f = lambda c, fmt=".3f": f"{g[c].mean():{fmt}} ± {g[c].std(ddof=1):{fmt}}" if len(g) > 1 else f"{g[c].mean():{fmt}}"
        agg[cfg] = g
        L.append(f"| {NAMES.get(cfg, cfg)} | {len(g)} | {f('clean')} | {f('stress')} | **{f('crit')}** | {f('peak')} | {f('val_loss', '.2f')} | "
                 f"{f('mean_conf')} | {g['err_conf90'].mean():.1%} | {f('ece')} |")
    L += ["", "## seed별 값", "", "| 실험 | 원본 val F1 | 필터 24조건 평균 | 기준값 | 에폭 중 최댓값 | 학습 손실 | val 손실 | 틀린 예측 수 | 틀린 예측 중 확률 ≥ 0.9 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in df.sort_values(["config", "seed"]).itertuples():
        L.append(f"| {r.exp} | {r.clean:.3f} | {r.stress:.3f} | {r.crit:.3f} | {r.peak:.3f} | {r.train_loss:.3f} | {r.val_loss:.2f} | {r.n_err} | {r.err_conf90:.1%} |")
    if "r0" in agg and len(agg["r0"]) > 1:
        sd0 = agg["r0"].crit.std(ddof=1)
        base = agg["r0"].crit.mean()
        L += ["", "## 기준(r0) 대비", "", f"r0의 기준값 평균 {base:.3f}, seed 표준편차 {sd0:.3f}. 설정 평균과의 차이가 이 표준편차의 몇 배인지 적는다 (계획: 표준편차보다 작으면 차이를 말할 수 없다).", "",
              "| 설정 | 기준값 평균 | r0와의 차이 | 차이 ÷ r0 표준편차 | seed 3개 중 r0 최고값을 넘은 개수 |", "|---|---|---|---|---|"]
        for cfg in sorted(agg):
            if cfg == "r0":
                continue
            g = agg[cfg]
            diff = g.crit.mean() - base
            L.append(f"| {NAMES.get(cfg, cfg)} | {g.crit.mean():.3f} | {diff:+.3f} | {diff / sd0:+.1f} | {(g.crit > agg['r0'].crit.max()).sum()} / {len(g)} |")
    out = ROOT / "reports/equipment_v2/reg_summary.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
