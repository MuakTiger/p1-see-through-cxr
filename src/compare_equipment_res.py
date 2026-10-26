"""장치 설정 필터 시험의 입력 해상도 비교표: B(224)와 B512의 val 결과를 필터 × 단계별로 나란히 놓는다 (저장된 결과 파일만 읽음).

사용 예 (p1 폴더에서):
    .venv/bin/python src/compare_equipment_res.py
결과: reports/equipment_v2/compare_224_512.md

주의: 두 모델은 학습 입력 크기가 다른 서로 다른 모델이고(seed 1개씩), 224 시험은 896에서 걸고 줄인 v2.0 필터, 512 시험은 512에 직접 건
v2.1 필터라서 "해상도의 효과"만 떼어 낸 비교가 아니다. 센서 화소 라벨은 224판 0.45 / 0.9 / 1.8, 512판 0.5 / 1.0 / 2.0mm로 다르다.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/equipment_v2"
TITLE = {"dose": "선량", "motion": "조사시간", "scatter": "산란", "kvp": "kVp", "sensor": "센서 화소", "blur": "흐림 (v1)",
         "sharpen": "경계 강조 (v1)", "equalize": "대비 보정 (v1)", "jpeg": "JPEG (v1)", "bright": "밝게 (v1)", "dark": "어둡게 (v1)"}
LEVEL = ["약", "중", "강"]


def load(path):
    d = pd.read_csv(path)
    return d[d["shift"] != "clean"].set_index(["shift", "level"]), d[d["shift"] == "clean"].iloc[0]


def share(r):
    return f"{r.pred_share_normal * 100:.0f}/{r.pred_share_bacterial * 100:.0f}/{r.pred_share_viral * 100:.0f}"


def main():
    a, ca = load(OUT / "val_B/shift_results.csv")
    b, cb = load(OUT / "val_B512/shift_results.csv")
    sig = lambda r: r.delta_f1_ci_low > 0 or r.delta_f1_ci_high < 0
    L = ["# 입력 해상도 비교: B(224) vs B512, 장치 설정 필터 33조건 (val 1,030장)", "",
         "두 모델은 서로 다른 모델(seed 1개씩)이고 224 시험은 v2.0 필터(896에서 걸고 줄임), 512 시험은 v2.1 필터(512에 직접 적용)이다. "
         "해상도의 효과만 떼어 낸 비교가 아니다. `*` = 95% 구간이 0을 포함하지 않는 변화. 예측 비율은 정상/세균성/바이러스성 %.", "",
         f"원본 Macro-F1: B(224) {ca.macro_f1:.3f} (예측 {share(ca)}), B512 {cb.macro_f1:.3f} (예측 {share(cb)})", "",
         "| 필터 | 단계 | B224 Macro-F1 | B224 변화 | B224 예측 비율 | B512 Macro-F1 | B512 변화 | B512 예측 비율 |", "|---|---|---|---|---|---|---|---|"]
    n224 = n512 = 0
    for name in TITLE:
        for lv in (1, 2, 3):
            ra, rb = a.loc[(name, lv)], b.loc[(name, lv)]
            n224 += sig(ra)
            n512 += sig(rb)
            L.append(f"| {TITLE[name]} | {LEVEL[lv - 1]} ({ra.param_value} / {rb.param_value}) | {ra.macro_f1:.3f} | {ra.delta_f1:+.3f}{'*' if sig(ra) else ''} | {share(ra)} | "
                     f"{rb.macro_f1:.3f} | {rb.delta_f1:+.3f}{'*' if sig(rb) else ''} | {share(rb)} |")
    L += ["", f"구간이 0을 포함하지 않는 조건: B224 {n224}개 / B512 {n512}개 (33개 중).", ""]
    (OUT / "compare_224_512.md").write_text("\n".join(L), encoding="utf-8")
    print(f"저장: {OUT / 'compare_224_512.md'}")
    print("\n".join(L[6:]))


if __name__ == "__main__":
    main()
