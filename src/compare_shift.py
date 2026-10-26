"""장비 차이 민감도 시험 결과 두 개(B, D)를 나란히 비교하는 표와 그림을 만든다.

입력은 src/shift_eval.py 가 만든 shift_results.csv 두 개뿐이다 (사진을 다시 읽지 않는다).

사용 예 (p1 폴더에서):
    .venv/bin/python src/compare_shift.py \\
        --b reports/shift_test/shift_results.csv --d reports/shift_test_D/shift_results.csv \\
        --out reports/shift_test_D

결과:
    compare_B_vs_D.png   필터별 Macro-F1(실선)과 바이러스성 recall(점선), B 회색 / D 파랑
    compare_B_vs_D.md    필터 × 강도마다 B 와 D 의 Macro-F1, 원본 대비 변화, 바이러스성 recall, 예측 비율
                         (해석 문단은 이 스크립트가 쓰지 않는다. 표 아래에 사람이 덧붙인다.)
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from shift_filters import SHIFTS  # noqa: E402

SHARE_KEYS = ["pred_share_normal", "pred_share_bacterial", "pred_share_viral"]
CLASS_KO = ["정상", "세균성", "바이러스성"]


def read(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return {(r["shift"], int(r["level"])): r for r in rows}


def f(r, k):
    return float(r[k])


def shares(r):
    return "/".join(f"{100 * f(r, k):.0f}" for k in SHARE_KEYS)


def tilt(r, clean):
    """원본 대비 예측 비율이 가장 많이 늘어난 클래스 (5%p 미만이면 '-')."""
    gains = [f(r, k) - f(clean, k) for k in SHARE_KEYS]
    c = max(range(3), key=lambda i: gains[i])
    return f"{CLASS_KO[c]} +{100 * gains[c]:.0f}%p" if gains[c] >= 0.05 else "-"


def plot(b, d, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    plt.rcParams["font.family"] = "Apple SD Gothic Neo"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), sharey=True)
    for ax, name in zip(axes.flat, SHIFTS):
        xs = [0, 1, 2, 3]
        labels = ["원본"] + [str(v) for v in SHIFTS[name]["levels"]]
        for res, tag, color in [(b, "B", "gray"), (d, "D", "tab:blue")]:
            keys = [("clean", 0)] + [(name, lv) for lv in (1, 2, 3)]
            ax.plot(xs, [f(res[k], "macro_f1") for k in keys], "-o", color=color, lw=2.5, label=f"{tag} Macro-F1")
            ax.plot(xs, [f(res[k], "recall_viral") for k in keys], "--s", color=color, lw=1.5, ms=4,
                    label=f"{tag} 바이러스성 recall")
        ax.set_title(SHIFTS[name]["title"], fontsize=13)
        ax.set_xticks(xs, labels, fontsize=9)
        ax.set_xlabel(SHIFTS[name]["param"].split(" ")[0], fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes.flat[0].set_ylabel("점수")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11, frameon=False)
    fig.suptitle("장비 차이 민감도: 실험 B(증강 없음) vs 실험 D(노이즈 증강), val 1,030장", fontsize=16)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", default=str(ROOT / "reports/shift_test/shift_results.csv"))
    ap.add_argument("--d", default=str(ROOT / "reports/shift_test_D/shift_results.csv"))
    ap.add_argument("--out", default=str(ROOT / "reports/shift_test_D"))
    args = ap.parse_args()
    b, d = read(args.b), read(args.d)
    if set(b) != set(d):
        raise SystemExit("두 결과의 필터·강도 목록이 다릅니다.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot(b, d, out / "compare_B_vs_D.png")

    bc, dc = b[("clean", 0)], d[("clean", 0)]
    lines = [
        "# 장비 차이 민감도: 실험 B vs 실험 D", "",
        "- B: layer4 미세조정, 증강 없음 (`reports/shift_test/shift_results.csv`)",
        "- D: B 와 같은 설정 + 학습 사진에만 회색 가우시안 노이즈 증강 (확률 0.5, std ~ Uniform(0, 10), 0~255 기준)"
        " (`reports/shift_test_D/shift_results.csv`)",
        "- 같은 val 1,030장, 같은 필터·강도·seed. 재학습 없이 필터만 바꿔 넣었다. 그림: `compare_B_vs_D.png`",
        "- 노이즈 시험 강도 std 16 은 D 의 학습 범위(0~10) 밖이다. 학습하지 않은 세기로도 버티는지 보려는 의도된 설정이다.",
        "- '변화'는 각 모델 자신의 원본 점수 대비. '예측 비율'은 정상/세균성/바이러스성 예측 비율(%)."
        " '쏠림'은 원본 대비 예측 비율이 가장 많이 늘어난 클래스(5%p 미만이면 -).",
        "", "## 표", "",
        "| 필터 | 강도 | 값 | Macro-F1 B | Macro-F1 D | D−B | 변화 B | 변화 D | 바이러스성 recall B | 바이러스성 recall D "
        "| 예측 비율 B | 예측 비율 D | 쏠림 B | 쏠림 D |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        f"| 원본 | 0 | - | {f(bc, 'macro_f1'):.3f} | {f(dc, 'macro_f1'):.3f} | {f(dc, 'macro_f1') - f(bc, 'macro_f1'):+.3f} "
        f"| - | - | {f(bc, 'recall_viral'):.3f} | {f(dc, 'recall_viral'):.3f} | {shares(bc)} | {shares(dc)} | - | - |",
    ]
    for name in SHIFTS:
        for lv in (1, 2, 3):
            rb, rd = b[(name, lv)], d[(name, lv)]
            lines.append(
                f"| {SHIFTS[name]['title']} | {lv} | {SHIFTS[name]['levels'][lv - 1]} | {f(rb, 'macro_f1'):.3f} | {f(rd, 'macro_f1'):.3f} "
                f"| {f(rd, 'macro_f1') - f(rb, 'macro_f1'):+.3f} "
                f"| {f(rb, 'delta_f1'):+.3f} [{f(rb, 'delta_f1_ci_low'):+.3f}, {f(rb, 'delta_f1_ci_high'):+.3f}] "
                f"| {f(rd, 'delta_f1'):+.3f} [{f(rd, 'delta_f1_ci_low'):+.3f}, {f(rd, 'delta_f1_ci_high'):+.3f}] "
                f"| {f(rb, 'recall_viral'):.3f} | {f(rd, 'recall_viral'):.3f} | {shares(rb)} | {shares(rd)} "
                f"| {tilt(rb, bc)} | {tilt(rd, dc)} |"
            )
    lines.append("")
    (out / "compare_B_vs_D.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"저장: {out}/compare_B_vs_D.md, compare_B_vs_D.png", flush=True)


if __name__ == "__main__":
    main()
