"""폐 영역 모델(l), 폐 영역 + 예방접종(lv)과 기준 모델의 val·test 필터 시험 결과를 한 표로 모으고 계획 문서의 판정 규칙을 적용한다
(저장된 결과 파일만 읽음, 모델을 돌리지 않음).

계획: reports/equipment_v2/lung_plan.md 3장, lung_plan_note.md 1장, lung_vaccine_plan.md 3장, test_plan_lung.md, test_plan_lung_vaccine.md.
  - 비교 (A − B): (가) l − AWS 기준 [주], (나) l − JupyterLab 기준 [보조], (다) lv − l [lv의 주 비교], (라) lv − AWS 기준 [참고].
    AWS 기준 3개 = B512 seed 42(기존 val_B512·test_B512) + b512-s43·44, JupyterLab 기준 3개 = b-s42·43·44.
  - 묶음: 본편 24조건 = 새 5종 + 경계 강조·대비 보정·JPEG (각 3단계), 학습에 쓴 5종(lv의 증강) 15조건, 쓰지 않은 3종 9조건, 부록 9조건 = 흐림·밝게·어둡게.
  - 판정: A − B의 평균 차이가 B 3개의 seed 표준편차(같은 지표, ddof=1)의 2배 이상일 때만 "차이 있음", 아니면 "차이를 말할 수 없음".
  - 참고로 일반 예방접종 v 3개(JupyterLab)도 같은 지표로 적는다 (판정 없음).

사용 예 (p1 폴더에서):
    .venv/bin/python src/lung_summary.py
결과: reports/equipment_v2/lung/lung_summary.md, lung_summary.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import lung_prep as LP  # noqa: E402

EQ = ROOT / "reports/equipment_v2"
SEEDS = [42, 43, 44]
TRAINED = ["dose", "motion", "scatter", "kvp", "sensor"]
UNSEEN = ["sharpen", "equalize", "jpeg"]
MAIN = TRAINED + UNSEEN
APP = ["blur", "bright", "dark"]
KO = {"dose": "선량", "motion": "조사시간", "scatter": "산란", "kvp": "kVp", "sensor": "센서", "sharpen": "경계 강조",
      "equalize": "대비 보정", "jpeg": "JPEG", "blur": "흐림", "bright": "밝게", "dark": "어둡게"}
NAME = {"lung": "폐 영역 l", "lv": "폐 영역 + 예방접종 lv", "aws": "기준 AWS b512", "jup": "기준 JupyterLab b", "vac": "(참고) 예방접종 v"}
GROUPS = ["lung", "lv", "aws", "jup", "vac"]
COMPARE = [("lung", "aws", "(가) l 대 AWS 기준 [주]"), ("lung", "jup", "(나) l 대 JupyterLab 기준 [보조]"),
           ("lv", "lung", "(다) lv 대 l [lv의 주 비교]"), ("lv", "aws", "(라) lv 대 AWS 기준 [참고]")]


def path_of(group, split, seed):
    if group == "lung":
        return EQ / f"lung/{split}_l-s{seed}"
    if group == "lv":
        return EQ / f"lung/{split}_lv-s{seed}"
    if group == "aws":
        return (EQ / ("val_B512" if split == "val" else "test_B512")) if seed == 42 else EQ / f"lung/{split}_b512-s{seed}"
    if group == "jup":
        return EQ / f"vac/{split}_b-s{seed}"
    return EQ / f"vac/{split}_v-s{seed}"


def load(group, split, seed):
    p = path_of(group, split, seed) / "shift_results.csv"
    return pd.read_csv(p) if p.is_file() else None


def metrics(df):
    c = df[df["shift"] == "clean"].iloc[0]
    s = df[df["shift"] != "clean"]
    f = lambda keys: float(s[s["shift"].isin(keys)].macro_f1.mean())
    out = {"원본 F1": float(c.macro_f1), "AUROC": float(c.macro_auroc), "정상 recall": float(c.recall_normal),
           "본편 24조건 평균": f(MAIN), "배운 5종 평균": f(TRAINED), "안 배운 3종 평균": f(UNSEEN), "부록 9조건 평균": f(APP)}
    out["본편 24조건 하락"] = out["본편 24조건 평균"] - out["원본 F1"]
    out["안 배운 3종 하락"] = out["안 배운 3종 평균"] - out["원본 F1"]
    return out


def fmt(x):
    return f"{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}" if len(x) > 1 else f"{np.mean(x):.3f}"


def verdict(a, b):
    d = np.mean(a) - np.mean(b)
    sd = np.std(b, ddof=1)
    return d, sd, (d / sd if sd > 0 else float("nan")), ("차이 있음 (" + ("앞 모델이 높음" if d > 0 else "앞 모델이 낮음") + ")" if abs(d) >= 2 * sd else "차이를 말할 수 없음")


def segmentation_note():
    rows = {r["relative_path"]: r for r in csv.DictReader(open(ROOT / "manifests/split_manifest.csv", encoding="utf-8"))}
    lines = []
    for sp in ("train", "val", "test"):
        p = ROOT / f"runs/lung_masks/{sp}_lung_masks.npz"
        if not p.is_file():
            continue
        m = LP.load_masks(p)
        bad = [k for k, v in m.items() if v.mean() < LP.MIN_AREA]
        cls = {}
        for k in bad:
            cls[rows[k]["label_name"]] = cls.get(rows[k]["label_name"], 0) + 1
        lines.append(f"- {sp}: {len(m)}장 중 분할 실패(면적<{LP.MIN_AREA:.0%}) {len(bad)}장 {cls}, 폐 면적 평균 {np.mean([v.mean() for v in m.values()]):.3f}")
    return lines


def main():
    L = ["# 폐 영역 모델(l)·폐+예방접종(lv) 대 기준 요약 (seed 42·43·44, 계획 문서의 판정 규칙 적용)", "",
         "생성: `src/lung_summary.py` (저장된 결과 파일만 읽음). 판정: 앞 모델 − 뒤 모델(기준)의 평균 차이가 **기준 3개의 seed 표준편차(ddof=1)의 2배 이상**일 때만 \"차이 있음\". "
         "폐 모델의 필터 시험은 필터를 먼저 걸고 고정 마스크로 폐 처리한 입력이다 (분할이 항상 맞다는 가정). "
         "표준편차는 seed 3개에서 나온 거친 추정이고, 하락 행은 기준 3개가 우연히 거의 같으면 표준편차가 아주 작아 판정이 부풀 수 있다. "
         "'배운 5종'은 lv가 학습에서 본 필터라 방어력의 증거가 아니고, '안 배운 3종'이 방어력을 보는 부분이다.", "",
         "## 폐 분할 현황", ""] + segmentation_note() + [""]
    rows = []
    for split in ("val", "test"):
        M = {(g, s): metrics(load(g, split, s)) for g in GROUPS for s in SEEDS if load(g, split, s) is not None}
        gm = {g: [M[(g, s)] for s in SEEDS if (g, s) in M] for g in GROUPS}
        keys = list(next(iter(M.values())).keys()) if M else []
        L += [f"## {split}", "", "모델 수: " + ", ".join(f"{NAME[g]} {len(gm[g])}개" for g in GROUPS), "",
              "### seed별 값", "", "| 모델 | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for (g, s), m in sorted(M.items(), key=lambda kv: (GROUPS.index(kv[0][0]), kv[0][1])):
            L.append(f"| {NAME[g]} s{s} | " + " | ".join(f"{m[k]:.3f}" for k in keys) + " |")
        L += ["", "### 묶음 평균 ± 표준편차", "", "| 묶음 | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for g in GROUPS:
            if gm[g]:
                L.append(f"| {NAME[g]} ({len(gm[g])}개) | " + " | ".join(fmt([m[k] for m in gm[g]]) for k in keys) + " |")
        for a, b, title in COMPARE:
            L += ["", f"### 판정 {title}", "", "| 지표 | 뒤(기준) | 앞 모델 | 차이 (앞 − 뒤) | 뒤 표준편차 | 차이 ÷ 표준편차 | 판정 |", "|---|---|---|---|---|---|---|"]
            if gm[a] and len(gm[b]) >= 2:
                for k in keys:
                    bb, aa = [m[k] for m in gm[b]], [m[k] for m in gm[a]]
                    d, sd, ratio, txt = verdict(aa, bb)
                    if len(gm[b]) < 3 or len(gm[a]) < 3:
                        txt = "참고 수준 (seed 3개 미만)"
                    if k in ("배운 5종 평균",) and a == "lv":
                        txt += " (배운 유형, 방어력 증거 아님)"
                    L.append(f"| {k} | {fmt(bb)} | {fmt(aa)} | {d:+.3f} | {sd:.3f} | {ratio:+.1f} | {txt} |")
                    rows.append({"split": split, "comparison": f"{a}-{b}", "metric": k, "base_mean": np.mean(bb), "base_sd": np.std(bb, ddof=1),
                                 "model_mean": np.mean(aa), "diff": d, "ratio": ratio, "verdict": txt})
        L += ["", "### 강(3단계) 조건별 Macro-F1 (seed 평균)", "", "| 필터 | l | lv | AWS 기준 | JupyterLab 기준 | l − AWS | lv − l | AWS 기준 seed 범위 |", "|---|---|---|---|---|---|---|---|"]
        for c in MAIN + APP:
            def lv(g, s):
                d = load(g, split, s)
                return float(d[(d["shift"] == c) & (d.level == 3)].macro_f1.iloc[0]) if d is not None else np.nan
            vals = {g: [lv(g, s) for s in SEEDS] for g in ("lung", "lv", "aws", "jup")}
            mn = {g: (np.nanmean(v) if not np.all(np.isnan(v)) else np.nan) for g, v in vals.items()}
            L.append(f"| {KO[c]} | {mn['lung']:.3f} | {mn['lv']:.3f} | {mn['aws']:.3f} | {mn['jup']:.3f} | {mn['lung'] - mn['aws']:+.3f} | {mn['lv'] - mn['lung']:+.3f} | "
                     f"{np.nanmin(vals['aws']):.3f}~{np.nanmax(vals['aws']):.3f} |")
        L.append("")
    L += ["## 읽을 때 주의", "",
          "- l·lv는 PSPNet + ResNet18의 두 단계 시스템이다. 필터 시험의 마스크는 필터 없는 사진에서 만든 것이라 분할 모델 자체의 강건성은 들어 있지 않다.",
          "- 폐 분할은 병이 심한 쪽에서 틀린다(실패는 전부 세균성, 폐가 하얗게 찬 사진). 그런 사진에서는 병변이 껍질 밖으로 잘릴 수 있다.",
          "- val은 train과 같은 단서를 공유한다. 폐 밖을 지워도 val이 안 변하는 것이 곧 지름길이 없다는 뜻은 아니다.",
          "- 기준 3개의 표준편차가 커서 판정이 엄격하다(에폭 선택 규칙의 흔들림 포함). 규칙은 결과를 보고 바꾸지 않았다.",
          "- test는 이미 열린 적이 있는 데이터에서의 사전 지정 추가 분석이고, 설계 당시 B512 계열의 test 결과를 알고 있었다 (`test_plan_lung.md`, `test_plan_lung_vaccine.md` 1장)."]
    out = EQ / "lung/lung_summary.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    pd.DataFrame(rows).round(4).to_csv(EQ / "lung/lung_summary.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
