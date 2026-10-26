"""원본 학습(B) 대비, 학습 방식을 바꾼 모델들의 val · test · 장비 차이 비교표와 그림.

저장된 결과 파일만 읽는다 (test 를 다시 평가하지 않는다).
    val / 장비 차이 : reports/shift_test*/shift_results.csv 의 clean 행과 24조건, A 는 reports/experiments_summary.csv
    test            : reports/test_eval/aws/all_models_test.csv (사전에 정한 계획대로 한 번 낸 결과)
D 의 seed 43·44 와 B448 은 val·장비 차이만 있다 (test 를 열지 않았다).

사용 예 (p1 폴더에서):
    .venv/bin/python src/model_compare.py --out reports/model_comparison
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gradcam import setup_matplotlib  # noqa: E402

# (이름, 바꾼 것, test 모델 id 또는 None, 장비 차이 폴더 또는 None)
MODELS = [
    ("B", "기준: layer4 미세조정, 증강·전처리 없음", "B", "reports/shift_test"),
    ("D", "노이즈 증강 (학습 사진 50%, std 0~10), seed 42", "D", "reports/shift_test_D"),
    ("D seed 43", "D 와 같은 설정, seed 만 43", None, "reports/shift_test_D_s43"),
    ("D seed 44", "D 와 같은 설정, seed 만 44", None, "reports/shift_test_D_s44"),
    ("P1 HE", "히스토그램 평활화 전처리 (밝기·대비 정규화)", "P1", "reports/shift_test_P1"),
    ("P2 CLAHE", "국소 대비 평활화 전처리 (밝기·대비 정규화)", "P2", "reports/shift_test_P2"),
    ("P3 블러", "가우시안 블러 σ1.5 전처리", "P3", "reports/shift_test_P3"),
    ("P4 샤프닝", "언샤프 마스크 전처리", "P4", "reports/shift_test_P4"),
    ("C 가중치", "클래스 가중치 손실 (역빈도)", "C", "reports/shift_test_C"),
    ("A 몸통 고정", "layer4 도 얼림, fc 만 학습", "A", None),
    ("B448", "입력 해상도 448", None, "reports/shift_test_B448"),
]
FILTERS = [("blur", "흐림"), ("sharpen", "경계 강조"), ("noise", "노이즈"), ("bright", "밝게"), ("dark", "어둡게"),
           ("contrast", "대비 감소"), ("equalize", "대비 보정"), ("jpeg", "JPEG 압축")]


def read_shift(folder):
    s = pd.read_csv(ROOT / folder / "shift_results.csv")
    clean = s[s["shift"] == "clean"].iloc[0]
    cond = s[s["shift"] != "clean"]
    noise = s[s["shift"] == "noise"].set_index("level")
    return {"val_f1": float(clean.macro_f1), "val_normal": float(clean.recall_normal),
            "val_viral": float(clean.recall_viral), "all24": float(cond.macro_f1.mean()),
            "noise16": float(noise.loc[3, "macro_f1"]),
            "by_filter": cond.groupby("shift").macro_f1.mean().to_dict()}


def collect():
    test = pd.read_csv(ROOT / "reports/test_eval/aws/all_models_test.csv").set_index("model_id")
    summ = pd.read_csv(ROOT / "reports/experiments_summary.csv").set_index("experiment")
    rows = []
    for name, change, tid, folder in MODELS:
        r = {"name": name, "change": change, "test": None, "shift": None}
        if folder:
            r["shift"] = read_shift(folder)
            r.update(val_f1=r["shift"]["val_f1"], val_normal=r["shift"]["val_normal"], val_viral=r["shift"]["val_viral"])
        else:  # A: 장비 차이 시험을 하지 않았다
            a = summ.loc["A"]
            r.update(val_f1=float(a.val_macro_f1), val_normal=float(a.recall_normal), val_viral=float(a.recall_viral))
        if tid:
            t = test.loc[tid]
            if abs(float(t.val_macro_f1) - r["val_f1"]) > 1e-3:
                raise SystemExit(f"{name}: val F1 이 파일마다 다릅니다 ({t.val_macro_f1:.4f} / {r['val_f1']:.4f})")
            r["test"] = {"f1": float(t.test_macro_f1), "lo": float(t.test_macro_f1_ci_low), "hi": float(t.test_macro_f1_ci_high),
                         "normal": float(t.test_recall_normal), "viral": float(t.test_recall_viral)}
        rows.append(r)
    return rows


def sgn(v, base):
    return f"{v - base:+.3f}"


def md_table(headers, body):
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + ["| " + " | ".join(map(str, r)) + " |" for r in body]


def build_markdown(rows):
    b = rows[0]
    tested = [r for r in rows if r["test"]]
    d_rows = [r for r in rows if r["name"].startswith("D")]
    t1 = []
    for r in rows:
        base = r is b
        t = r["test"]
        t1.append([r["name"], r["change"], f"{r['val_f1']:.3f}", "—" if base else sgn(r["val_f1"], b["val_f1"]),
                   f"{t['f1']:.3f} [{t['lo']:.3f}, {t['hi']:.3f}]" if t else "열지 않음",
                   "—" if base or not t else sgn(t["f1"], b["test"]["f1"]),
                   f"{t['f1'] - r['val_f1']:+.3f}" if t else "—"])
    t2 = []
    for r in rows:
        t, s = r["test"], r["shift"]
        t2.append([r["name"], f"{r['val_normal']:.3f} / " + (f"{t['normal']:.3f}" if t else "—"),
                   f"{r['val_viral']:.3f} / " + (f"{t['viral']:.3f}" if t else "—"),
                   f"{s['all24']:.3f}" if s else "—", f"{s['noise16']:.3f}" if s else "—"])
    t3 = [[r["name"]] + [f"{r['shift']['by_filter'][k]:.3f}" for k, _ in FILTERS] for r in rows if r["shift"]]

    val_all = [r["val_f1"] for r in tested]
    test_all = [r["test"]["f1"] for r in tested]
    drops = [r["test"]["f1"] - r["val_f1"] for r in tested]
    rho = pd.Series(val_all).corr(pd.Series(test_all), method="spearman")
    overlap_free = [r["name"] for r in tested if r is not b and (r["test"]["lo"] > b["test"]["hi"] or r["test"]["hi"] < b["test"]["lo"])]
    d_val = [r["val_f1"] for r in d_rows]
    spread = sorted(((max(r["shift"]["by_filter"][k] for r in rows if r["shift"]) - min(r["shift"]["by_filter"][k] for r in rows if r["shift"]), ko)
                     for k, ko in FILTERS), reverse=True)
    spread_txt = ", ".join(f"{ko} {v:.3f}" for v, ko in spread)

    L = ["# 원본 학습(B) 대비 학습 방식을 바꾼 모델 비교", "",
         "저장된 결과 파일에서 직접 읽어 만들었다 (`src/model_compare.py`). test 는 다시 열지 않았고, 사전에 정한 계획대로 한 번 낸 결과를 그대로 썼다. "
         "모든 값은 Macro-F1 이고 recall 은 그 병인 사진 중 맞힌 비율이다.", "",
         "**먼저 알아 둘 것:** 밝기·대비를 학습 때 무작위로 바꿔 주는 증강 모델은 만든 적이 없다. 밝기·대비 쪽 변화는 정규화형 전처리 P1(HE)·P2(CLAHE)가 전부이고, "
         "'밝게·어둡게·대비 감소'는 학습이 아니라 **시험 조건**으로만 썼다. 노이즈는 D 에서 학습 때 넣었다.", "",
         "## 1. 점수 (val 1,030장 / test 618장)", ""]
    L += md_table(["모델", "바꾼 것", "val F1", "B 대비", "test F1 [95% 구간]", "B 대비", "val→test"], t1)
    L += ["", f"- val 은 {min(val_all):.3f}~{max(val_all):.3f}로 모델 사이 차이가 {max(val_all) - min(val_all):.3f} 뿐이다. D 만 해도 seed 에 따라 {min(d_val):.3f}~{max(d_val):.3f}라서, "
          "이 차이는 학습을 다시 돌릴 때 생기는 흔들림과 구분되지 않는다.",
          f"- test 는 {min(test_all):.3f}~{max(test_all):.3f}이고, B 의 95% 구간과 겹치지 않는 모델은 {'없다' if not overlap_free else ', '.join(overlap_free)}. "
          "구간이 겹친다고 같다는 뜻은 아니지만, 이 자료로 어느 변화가 test 에서 낫다고 말하기는 어렵다.",
          f"- 모든 모델이 test 에서 val 보다 낮다 ({min(drops):+.3f}~{max(drops):+.3f}). val 순위와 test 순위의 순위상관(Spearman)은 {rho:+.2f}다. 모델이 {len(tested)}개뿐이라 우연과 구분되지 않고, 'val 순위가 test 순위를 예측하지 못했다'까지만 말할 수 있다.", "",
          "## 2. 클래스별 recall (val / test)와 장비 차이", ""]
    L += md_table(["모델", "정상 recall (val / test)", "바이러스성 recall (val / test)", "장비 차이 24조건 평균 (val)", "노이즈 16 F1 (val)"], t2)
    t_normals = [r["test"]["normal"] for r in tested]
    v_normals = [r["val_normal"] for r in tested]
    L += ["", f"- 정상 recall 은 val 에서 {min(v_normals):.3f}~{max(v_normals):.3f}인데 test 에서는 {min(t_normals):.3f}~{max(t_normals):.3f}로 모델 전부가 크게 낮아진다. "
          "어떤 변화를 줘도 이 낙폭은 사라지지 않았다.",
          f"- 장비 차이에서 확실히 달라진 것은 노이즈 하나다: B {b['shift']['noise16']:.3f} → D {min(r['shift']['noise16'] for r in d_rows):.3f}~"
          f"{max(r['shift']['noise16'] for r in d_rows):.3f} (seed 3개 모두). 24조건 평균은 B {b['shift']['all24']:.3f}, D "
          f"{min(r['shift']['all24'] for r in d_rows):.3f}~{max(r['shift']['all24'] for r in d_rows):.3f}.", "",
          "## 3. 장비 차이 시험, 필터별 평균 F1 (val, 단계 3개 평균)", ""]
    L += md_table(["모델"] + [ko for _, ko in FILTERS], t3)
    L += ["", "- 표의 값은 '그 필터를 val 에 걸었을 때' 점수다. 전처리 모델(P1~P4)은 학습·시험 모두 자기 전처리를 거친다.",
          f"- 필터별로 모델 사이 차이(가장 높은 모델 − 가장 낮은 모델)는 큰 순서로 {spread_txt}.", "",
          "## 4. 한계", "",
          "- 학습은 대부분 seed 1개다 (D 만 3개). test 는 한 번만 열었고, 그 결과를 보고 설정을 바꾸지 않았다.",
          "- 노이즈 증강과 노이즈 시험은 같은 종류(가우시안)다. 흐림 등 다른 변화에 강해졌다는 뜻이 아니다.",
          "- 장비 차이는 필터로 흉내 낸 것이고, 실제 다른 병원 사진이 아니다.",
          "- A 는 장비 차이 시험을 하지 않았고, B448 은 test 를 열지 않았다. D seed 43·44 도 test 를 열지 않았다."]
    return "\n".join(L)


def figure(rows, out):
    plt = setup_matplotlib()
    b = rows[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5.8))
    tested = [r for r in rows if r["test"]]
    x = np.arange(len(tested))
    w = 0.38
    val_c, test_c = "#0E6C76", "#A3500A"
    a1.bar(x - w / 2, [r["val_f1"] for r in tested], w, color=val_c, label="val")
    a1.bar(x + w / 2, [r["test"]["f1"] for r in tested], w, color=test_c, label="test (95% 구간)",
           yerr=[[r["test"]["f1"] - r["test"]["lo"] for r in tested], [r["test"]["hi"] - r["test"]["f1"] for r in tested]],
           error_kw={"ecolor": "#3C2A12", "capsize": 3, "lw": 1})
    xd = [i for i, r in enumerate(tested) if r["name"] == "D"][0]
    seeds = [r["val_f1"] for r in rows if r["name"].startswith("D seed")]
    a1.scatter([xd - w / 2] * len(seeds), seeds, s=46, facecolor="white", edgecolor="black", zorder=5, label="D 의 다른 seed (val)")
    a1.axhline(b["val_f1"], color=val_c, ls="--", lw=1)
    a1.axhline(b["test"]["f1"], color=test_c, ls="--", lw=1)
    for i, r in enumerate(tested):
        a1.text(i - w / 2, r["val_f1"] + 0.004, f"{r['val_f1']:.3f}", ha="center", fontsize=8, color=val_c)
        a1.text(i + w / 2, r["test"]["hi"] + 0.004, f"{r['test']['f1']:.3f}", ha="center", fontsize=8, color=test_c)
    a1.set_xticks(x)
    a1.set_xticklabels([r["name"].replace(" ", "\n", 1) for r in tested], fontsize=9)
    a1.set_ylim(0.6, 0.9)
    a1.set_ylabel("Macro-F1")
    a1.set_title("val 과 test (점선 = B 의 값, 세로축은 0.6 부터)", fontsize=12)
    a1.legend(loc="upper right", fontsize=9)

    sh = [r for r in rows if r["shift"]]
    x2 = np.arange(len(sh))
    a2.bar(x2 - w / 2, [r["shift"]["all24"] for r in sh], w, color="#5B8DB8", label="장비 차이 24조건 평균")
    a2.bar(x2 + w / 2, [r["shift"]["noise16"] for r in sh], w, color="#C9472B", label="노이즈 16 에서 F1")
    a2.axhline(b["shift"]["all24"], color="#5B8DB8", ls="--", lw=1)
    a2.axhline(b["shift"]["noise16"], color="#C9472B", ls="--", lw=1)
    for i, r in enumerate(sh):
        a2.text(i - w / 2, r["shift"]["all24"] + 0.01, f"{r['shift']['all24']:.2f}", ha="center", fontsize=8)
        a2.text(i + w / 2, r["shift"]["noise16"] + 0.01, f"{r['shift']['noise16']:.2f}", ha="center", fontsize=8)
    a2.set_xticks(x2)
    a2.set_xticklabels([r["name"].replace(" ", "\n", 1) for r in sh], fontsize=9)
    a2.set_ylim(0, 0.95)
    a2.set_title("장비 차이 시험 (val, 점선 = B 의 값)", fontsize=12)
    a2.legend(loc="lower right", fontsize=9)
    fig.suptitle("원본 학습(B) 대비 학습 방식을 바꾼 모델 (단일 seed, D 만 seed 3개)", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out) + ".png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="저장 경로(확장자 없음). .md 와 .png 가 만들어진다")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = collect()
    Path(str(out) + ".md").write_text(build_markdown(rows), encoding="utf-8")
    figure(rows, out)
    print(f"저장: {out}.md, {out}.png")


if __name__ == "__main__":
    main()
