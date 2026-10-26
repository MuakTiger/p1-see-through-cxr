"""규칙 기반 설명문: 사진 한 장 → "어느 부위를 참고해, 어떤 확률로, 어떻게 판단했는지" 문장.

비유: 의사 대신 "채점 기록표"를 읽어 주는 것이다. 모델이 낸 확률과 Grad-CAM 형광펜 위치(어느 구역에
몇 %)를 정해 둔 문장 틀에 끼워 넣을 뿐, 새로운 판단을 하거나 LLM(Bedrock)을 쓰지 않는다.
같은 입력이면 항상 같은 문장이 나온다.

규칙은 아래 상수에 모아 두었다. 이 값들은 test를 보고 정한 것이 아니라 기획안의 τ 0.70과
gradcam.py 의 영역 정의를 그대로 가져온 것이다.

한계 (문장에도 그대로 들어간다)
  - Grad-CAM 은 "모델이 참고한 곳"이지 병변이 있는 곳이라는 증명이 아니다.
  - 폐 분할 모델이 없어서 구역은 사진의 흉부 상자(가로 15~85%, 세로 10~85%) 근사다.
  - 사진 왼쪽 = 환자의 오른쪽 폐라고 가정한다 (표준 후전면 촬영 기준, 확인하지 않음).
  - test 에서 정상 사진 129장을 확신 높게 폐렴이라 했다. 확신도가 높아도 틀릴 수 있다.

사용 예 (p1 폴더에서):
    .venv/bin/python src/explain.py --image ~/data/chest_xray/test/NORMAL/IM-0001-0001.jpeg
    .venv/bin/python src/explain.py --examples        # test 그룹별 무작위 예시 → reports/explain/
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))
import train as T  # noqa: E402
from gradcam import (GROUP_KO, GROUPS, MASKS, SIZE, GradCAM, denorm_gray,  # noqa: E402
                     overlay, setup_matplotlib)
from shift_eval import load_model  # noqa: E402

D_DIR = ROOT / "runs/results/d-20261019-134621/model"
CLASS_KO = ["정상", "세균성 폐렴", "바이러스성 폐렴"]

# ---- 규칙 상수 ---------------------------------------------------------------
TAU = 0.70            # 기획안의 판단 보류 기준: 최대 확률이 이보다 작으면 판정하지 않는다
HIGH = 0.90           # 이상이면 확신도 "높음", 그 사이는 "보통"
MAIN_SHARE = 0.30     # 흉부 상자 안 6구역 중 한 구역이 안쪽 지도의 30% 이상이면 "주로" (6구역이 고르면 각 17%)
SECOND_SHARE = 0.20   # 두 번째 구역은 20% 이상일 때만 같이 말한다
# 아래 두 값은 모델 D 가 val 1,030장(원본)에서 보인 분포의 95번째 백분위수다 (가장자리 중앙값 0.36, 위 모서리 중앙값 0.04).
# 즉 "평소 val 사진 20장 중 1장 정도만 이보다 가장자리를 많이 본다". test 는 보고 정하지 않았다.
OUTSIDE_FLAG = 0.55   # 흉부 상자 밖 비율
CORNER_FLAG = 0.12    # 위 모서리(글자 표식 위치) 비율

# 흉부 상자 안 6구역: 사진 좌/우 × 위/가운데/아래 (상자 세로를 삼등분)
X0, X1, Y0, Y1 = 0.15, 0.85, 0.10, 0.85
ZONE_NAMES = {
    ("L", 0): "사진 왼쪽(환자 오른쪽) 폐 위쪽", ("L", 1): "사진 왼쪽(환자 오른쪽) 폐 가운데",
    ("L", 2): "사진 왼쪽(환자 오른쪽) 폐 아래쪽",
    ("R", 0): "사진 오른쪽(환자 왼쪽) 폐 위쪽", ("R", 1): "사진 오른쪽(환자 왼쪽) 폐 가운데",
    ("R", 2): "사진 오른쪽(환자 왼쪽) 폐 아래쪽",
}


def zone_masks():
    c = (np.arange(SIZE) + 0.5) / SIZE
    X, Y = np.meshgrid(c, c)
    inside = (X >= X0) & (X < X1) & (Y >= Y0) & (Y < Y1)
    side = np.where(X < 0.5, "L", "R")
    third = np.clip(((Y - Y0) / (Y1 - Y0) * 3).astype(int), 0, 2)
    masks = {(s, t): inside & (side == s) & (third == t) for s in "LR" for t in range(3)}
    return masks, ~inside


ZONES, OUTSIDE = zone_masks()


def cam_shares(cam):
    """지도 한 장 (224,224) → {구역: 상자 안 지도 총합 대비 비율}, 흉부 밖 비율(전체 대비), 위 모서리 비율(전체 대비)."""
    total = float(cam.sum())
    if total <= 0:
        return None
    inside = sum(float(cam[m].sum()) for m in ZONES.values())
    if inside <= 0:
        return None
    shares = {z: float(cam[m].sum()) / inside for z, m in ZONES.items()}  # 흉부 상자 안만으로 다시 계산한 비율
    return shares, float(cam[OUTSIDE].sum()) / total, float(cam[MASKS["corner"]].sum()) / total


def confidence_tier(p_max):
    if p_max < TAU:
        return "보류"
    return "높음" if p_max >= HIGH else "보통"


def build_text(p, shares_info):
    """확률 (3,) 과 지도 비율 → 설명문 목록 (줄 단위). 규칙은 위 상수와 아래 if 문이 전부다."""
    order = np.argsort(-p)
    top, second = int(order[0]), int(order[1])
    tier = confidence_tier(p[top])
    lines = []

    if tier == "보류":
        lines.append(f"[판정] 보류. 가장 높은 확률이 {CLASS_KO[top]} {p[top]*100:.1f}%로 기준({TAU*100:.0f}%)에 못 미쳐 "
                     f"판단하지 않습니다. 의료진 확인이 필요합니다.")
    else:
        lines.append(f"[판정] {CLASS_KO[top]} 가능성 {p[top]*100:.1f}% (확신도: {tier}). "
                     f"그다음은 {CLASS_KO[second]} {p[second]*100:.1f}%입니다.")

    warnings = []
    if shares_info is None:
        lines.append("[참고 부위] 이 사진에서는 열지도가 만들어지지 않아 참고 부위를 말할 수 없습니다.")
    else:
        shares, outside, corner = shares_info
        ranked = sorted(shares.items(), key=lambda kv: -kv[1])
        (z1, s1), (z2, s2) = ranked[0], ranked[1]
        if outside >= OUTSIDE_FLAG:
            lines.append(f"[참고 부위] 모델이 참고한 곳의 {outside*100:.0f}%가 흉부 상자 밖(가장자리)입니다.")
            warnings.append("가장자리 비중이 평소(val 사진 95%)보다 큽니다. 폐 밖을 많이 본 판정이라 근거가 약할 수 있습니다.")
        elif s1 >= MAIN_SHARE:
            where = ZONE_NAMES[z1] + f"({s1*100:.0f}%)"
            if s2 >= SECOND_SHARE:
                where += f" 및 {ZONE_NAMES[z2]}({s2*100:.0f}%)"
            lines.append(f"[참고 부위] 모델은 주로 {where} 쪽을 참고해 "
                         + (f"{CLASS_KO[top]}으로 판단했습니다." if tier != "보류" else "판단을 시도했으나 확신하지 못했습니다."))
        else:
            lines.append(f"[참고 부위] 모델이 참고한 곳이 한 구역에 몰리지 않고 폐 전체에 넓게 퍼져 있습니다 "
                         f"(흉부 안에서 가장 큰 구역 {s1*100:.0f}%).")
        if corner >= CORNER_FLAG:
            warnings.append(f"위 모서리(글자 표식이 있는 곳)를 {corner*100:.0f}% 참고했습니다. 평소(val 사진 95%)보다 큽니다.")

    if tier != "보류" and top != 0:
        warnings.append("확신도가 높아도 틀릴 수 있습니다. 이 모델은 test에서 정상 사진 231장 중 129장을 폐렴으로 잘못 판정했고, "
                        "그중 113장은 확률 70% 이상이었습니다.")
    if tier != "보류" and top == 0:
        warnings.append("이 모델은 test에서 정상 사진의 44%만 정상으로 맞혔습니다. 정상 판정이 나와도 의료진 확인이 필요합니다.")
    lines.append("[주의] 열지도는 모델이 참고한 곳일 뿐 병변이 있는 곳이라는 증거가 아니며, 구역은 흉부 상자를 근사한 것입니다. "
                 "이 결과는 연구용이며 진단이 아닙니다.")
    lines.extend("[경고] " + w for w in warnings)
    return lines, tier


def explain_batch(cam_model, x):
    """정규화된 입력 (N,3,224,224) → [(확률, 지도, 설명 줄, 확신도)]"""
    logits, (cams,), valid = cam_model(x, [None])
    probs = torch.softmax(logits, 1).numpy()
    out = []
    for i in range(len(x)):
        info = cam_shares(cams[i]) if valid[0][i] else None
        lines, tier = build_text(probs[i], info)
        out.append((probs[i], cams[i], lines, tier))
    return out


def load_input(path):
    with Image.open(path) as im:
        return T.build_transform("P0")(im.convert("RGB"))  # 학습·평가와 똑같은 전처리


def draw_zone_lines(ax):
    ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
        (X0 * SIZE - .5, Y0 * SIZE - .5), (X1 - X0) * SIZE, (Y1 - Y0) * SIZE, fill=False, ec="white", ls="--", lw=1))
    ax.axvline(0.5 * SIZE - .5, ymin=1 - Y1, ymax=1 - Y0, color="white", ls=":", lw=1)
    for t in (1, 2):
        y = (Y0 + (Y1 - Y0) * t / 3) * SIZE - .5
        ax.plot([X0 * SIZE - .5, X1 * SIZE - .5], [y, y], color="white", ls=":", lw=1)


def run_examples(args):
    """test 결과 그룹마다 무작위(seed 42) 1장 + 판단 보류가 되는 1장. 고른 기준은 무작위이며 보기 좋은 사진을 고르지 않았다."""
    rng = np.random.default_rng(args.seed)
    with open(ROOT / "reports/test_eval/aws/test_predictions.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["excluded"] == "False"]
    picks = []
    for name, t, p_, ko in GROUPS[:5] + GROUPS[5:6]:
        pool = [r for r in rows if int(r["true_label"]) == t and int(r["pred_label"]) == p_
                and max(float(r["p_normal"]), float(r["p_bacterial"]), float(r["p_viral"])) >= TAU]
        if pool:
            picks.append((GROUP_KO[name], pool[int(rng.integers(len(pool)))]))
    hold = [r for r in rows if max(float(r["p_normal"]), float(r["p_bacterial"]), float(r["p_viral"])) < TAU]
    picks.append(("판단 보류(확률 70% 미만) 무작위", hold[int(rng.integers(len(hold)))]))

    cam_model = GradCAM(load_model(D_DIR))
    x = torch.stack([load_input(Path(args.images_dir) / r["relative_path"]) for _, r in picks])
    res = explain_batch(cam_model, x)
    grays = denorm_gray(x)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 4, figsize=(20, 11))
    md = ["# 규칙 기반 설명문 예시 (모델 D, test)", "",
          "test 결과 그룹마다 무작위(seed 42)로 1장씩 골랐다. 보기 좋은 사진을 고르지 않았다. 문장은 `src/explain.py`의 "
          "규칙만으로 만들었고 LLM을 쓰지 않았다.", ""]
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    for ax, (label, r), (p, cam, lines, tier), g in zip(axes.ravel(), picks, res, grays):
        overlay(ax, g, cam, f"{label}\n{r['relative_path'].split('/', 1)[1]}", fontsize=9)
        draw_zone_lines(ax)
        md += [f"## {label}", f"- 파일: `{r['relative_path']}` (정답: {CLASS_KO[int(r['true_label'])]})", ""]
        md += [f"> {ln}" for ln in lines]
        md.append("")
    fig.tight_layout()
    fig.savefig(out / "explain_examples.png", dpi=130)
    (out / "explain_examples.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"저장: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="사진 한 장의 경로")
    ap.add_argument("--examples", action="store_true", help="test 그룹별 무작위 예시를 reports/explain/ 에 만든다")
    ap.add_argument("--images-dir", default=str(Path.home() / "data/chest_xray"))
    ap.add_argument("--out", default=str(ROOT / "reports/explain"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.examples:
        run_examples(args)
    elif args.image:
        cam_model = GradCAM(load_model(D_DIR))
        (_, _, lines, _), = explain_batch(cam_model, load_input(args.image).unsqueeze(0))
        print("\n".join(lines))
    else:
        ap.error("--image 또는 --examples 중 하나를 주세요")


if __name__ == "__main__":
    main()
