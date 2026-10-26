"""B512의 필터 시험을 val과 test에서 나란히 놓는다 (저장된 결과 파일만 읽음, 모델을 돌리지 않음).

사용 예 (p1 폴더에서):
    .venv/bin/python src/compare_val_test.py
결과: reports/equipment_v2/compare_val_test_B512.md

원본 대비 변화(Macro-F1)가 val과 test에서 같은 방향·비슷한 크기인지 본다. test의 원본 점수 자체가 val보다 훨씬 낮아서
(정상 사진 분포 차이) 절대 점수가 아니라 변화량을 비교한다. 필터끼리 강도가 보정되지 않아 서로 직접 비교하지 않는다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/equipment_v2"
TITLE = {"dose": "선량", "motion": "조사시간", "scatter": "산란", "kvp": "kVp", "sensor": "센서 화소", "sharpen": "경계 강조",
         "equalize": "대비 보정", "jpeg": "JPEG", "blur": "흐림 (부록)", "bright": "밝게 (부록)", "dark": "어둡게 (부록)"}
MAIN = ["dose", "motion", "scatter", "kvp", "sensor", "sharpen", "equalize", "jpeg"]
LEVEL = {1: "약", 2: "중", 3: "강"}


def load(path):
    d = pd.read_csv(path)
    return d[d["shift"] != "clean"].set_index(["shift", "level"]), d[d["shift"] == "clean"].iloc[0]


def sig(r):
    return r.delta_f1_ci_low > 0 or r.delta_f1_ci_high < 0


def main():
    v, cv = load(OUT / "val_B512/shift_results.csv")
    t, ct = load(OUT / "test_B512/shift_results.csv")
    L = ["# B512 필터 시험: val과 test 비교", "",
         "원본 대비 Macro-F1 변화를 val(1,030장)과 test(618장)에서 나란히 놓았다 (`src/compare_val_test.py`). `*` = 95% 구간이 0을 포함하지 않음. "
         "test의 원본 점수 자체가 val보다 훨씬 낮아서 절대 점수가 아니라 변화량을 본다. 필터끼리 강도가 보정되지 않았으므로 필터 사이의 직접 비교는 하지 않는다.", "",
         f"원본 Macro-F1: val {cv.macro_f1:.3f}, test {ct.macro_f1:.3f}. 정상 recall: val {cv.recall_normal:.3f}, test {ct.recall_normal:.3f}.", "",
         "| 필터 | 단계 (값) | val Macro-F1 | val 변화 | test Macro-F1 | test 변화 | 같은 방향 |", "|---|---|---|---|---|---|---|"]
    rows = []
    for name in MAIN + ["blur", "bright", "dark"]:
        for lv in (1, 2, 3):
            a, b = v.loc[(name, lv)], t.loc[(name, lv)]
            same = (a.delta_f1 < 0) == (b.delta_f1 < 0)
            rows.append((name, lv, a.delta_f1, b.delta_f1, sig(a), sig(b)))
            L.append(f"| {TITLE[name]} | {LEVEL[lv]} ({a.param_value}) | {a.macro_f1:.3f} | {a.delta_f1:+.3f}{'*' if sig(a) else ''} | "
                     f"{b.macro_f1:.3f} | {b.delta_f1:+.3f}{'*' if sig(b) else ''} | {'예' if same else '아니오'} |")
    df = pd.DataFrame(rows, columns=["name", "lv", "dv", "dt", "sv", "st"])
    main = df[df.name.isin(MAIN)]
    rho = main.dv.rank().corr(main.dt.rank())
    pear = main.dv.corr(main.dt)
    both_down = int(((main.dv < 0) & (main.dt < 0)).sum())
    sig_both = int((main.sv & main.st & (main.dv < 0) & (main.dt < 0)).sum())
    top = lambda col, k=6: ", ".join(f"{TITLE[r.name]} {LEVEL[r.lv]} ({getattr(r, col):+.3f})" for r in main.sort_values(col).head(k).itertuples())
    L += ["", "## 요약 (본편 8종 24조건)", "",
          f"- 변화의 순위 상관(Spearman) {rho:.2f}, 피어슨 {pear:.2f}.",
          f"- val과 test 둘 다 점수가 떨어진 조건 {both_down}개 / 24개, 그중 둘 다 구간이 0을 포함하지 않은 조건 {sig_both}개.",
          f"- val에서 가장 크게 떨어진 6개: {top('dv')}.",
          f"- test에서 가장 크게 떨어진 6개: {top('dt')}.",
          "- test에서 점수가 오른 조건(구간이 0을 포함하지 않음): " + (", ".join(f"{TITLE[r.name]} {LEVEL[r.lv]} ({r.dt:+.3f})" for r in main[(main.dt > 0) & main.st].itertuples()) or "없음") + ".",
          ""]
    (OUT / "compare_val_test_B512.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[-9:]))


if __name__ == "__main__":
    main()
