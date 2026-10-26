"""Grad-CAM: 모델이 사진의 어디를 보고 판단했는지 열지도로 보여 준다 (가이드 12장).

비유: 학생이 답을 고를 때 시험지의 어느 줄에 형광펜을 그었는지 보는 것과 같다.
형광펜 자국은 "참고한 곳"일 뿐, 그 줄이 정답의 근거라는 증명은 아니다.
그래서 이 열지도는 모델의 참고 영역이지 병변 경계나 임상적 원인의 증거가 아니다.

두 가지 질문
  Q1 (test, 최종 모델 D): 정상 사진을 폐렴으로 틀릴 때 폐(중앙)를 보는가, 테두리·모서리(표식·배경)를 보는가.
  Q2 (val, 실험 B vs D): 노이즈(std 16)를 넣으면 B 는 바이러스성으로 쏠리고 D 는 버틴다.
      같은 사진에서 두 모델의 열지도가 노이즈 전후로 어떻게 바뀌는가.

규칙
  - 재학습 없음, 임계값 조정 없음, AWS 사용 없음. CPU 만 쓴다 (MPS 사용 안 함). 스레드는 5개로 제한.
  - Q1 은 test 예측표(reports/test_eval/aws/test_predictions.csv)에서 excluded==False 인 618장만 연다.
    입력 전처리는 학습·평가와 똑같은 P0 (train.py 의 build_transform("P0")).
  - Q2 는 val 행만 연다. 입력 경로는 장비 차이 시험(shift_eval.py)과 똑같이
    "회색조 → 224 → (노이즈) → 3채널 → 정규화", 노이즈 seed 는 42 + (val 안에서의 순번).
  - 외부 Grad-CAM 라이브러리를 쓰지 않고 PyTorch 훅으로 직접 구현한다.

사용 예 (p1 폴더에서):
    .venv/bin/python src/gradcam.py

결과 (reports/gradcam/):
    cam_region_stats.csv     사진·지도별 영역 비율 (테두리 / 위 모서리 / 중앙 흉부 상자)
    gradcam_summary.json     보고서에 쓴 숫자 전부
    gradcam_tables.md        자동으로 만든 표 (보고서 원본 숫자)
    q1_test_examples.png     test 그룹별 예시 4장씩 (D, 예측 클래스 지도)
    q1_misclassified_pred_vs_true.png  정상 오분류 예시의 예측 클래스 지도 vs 정답(정상) 클래스 지도
    q1_mean_maps.png         test 그룹별 평균 지도 + 영역 정의 그림
    q2_noise_B_vs_D.png      val 사진의 [원본 B / 노이즈 B / 원본 D / 노이즈 D]
    q2_mean_maps.png         val 평균 지도 (원본·노이즈, B·D)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))
import train as T  # noqa: E402  (P0 전처리·지표를 학습 코드와 똑같이 쓰기 위해)
from shift_filters import SHIFTS, apply_shift, to_gray_224  # noqa: E402
from shift_eval import gray_to_tensor, load_model  # noqa: E402  (장비 차이 시험과 같은 모델 로드·입력 경로)

CLASS_KO = ["정상", "세균성", "바이러스성"]
SIZE = 224
NOISE_LEVEL = 3  # shift_filters 의 noise 강도 3 = std 16 (0~255 기준)
MODEL_DIRS = {
    "B": ROOT / "runs/results/b-20261019-103853/model",
    "D": ROOT / "runs/results/d-20261019-134621/model",
}
JOB_NAMES = {
    "B": "p1-cxr-<학생ID>-b-20261019-103853",
    "D": "p1-cxr-<학생ID>-d-20261019-134621",
}
# test 결과 그룹: (이름, 정답, 예측, 한국어 설명)
GROUPS = [
    ("N→N", 0, 0, "정상→정상 (맞힘)"),
    ("N→V", 0, 2, "정상→바이러스성 (틀림)"),
    ("N→B", 0, 1, "정상→세균성 (틀림)"),
    ("B→B", 1, 1, "세균성→세균성 (맞힘)"),
    ("V→V", 2, 2, "바이러스성→바이러스성 (맞힘)"),
    ("B→V", 1, 2, "세균성→바이러스성 (틀림)"),
    ("V→B", 2, 1, "바이러스성→세균성 (틀림)"),
]
GROUP_KO = {g[0]: g[3] for g in GROUPS}
REGIONS = ["center", "border", "corner"]  # 주 지표 세 개 (표의 열 순서도 이 순서)
SUB_BANDS = ["band_top", "band_bottom", "band_sides"]  # 테두리 띠를 위·아래·좌우로 나눈 것 (어느 쪽 테두리인지)
METRICS = REGIONS + SUB_BANDS + ["active", "cy"]
METRIC_KO = {"center": "중앙 흉부 상자", "border": "테두리 띠", "corner": "위 모서리",
             "band_top": "테두리-위", "band_bottom": "테두리-아래", "band_sides": "테두리-좌우",
             "active": "활성 면적(≥0.5)", "cy": "무게중심 세로(0 위~1 아래)"}


# ----------------------------------------------------------------------------
# Grad-CAM
# ----------------------------------------------------------------------------
class GradCAM:
    """model.layer4[-1] (출력 512채널 × 7×7) 기준 표준 Grad-CAM.

    순서: (1) 대상 클래스 점수(softmax 전 logit)를 layer4 출력으로 미분한다.
          (2) 채널마다 기울기를 7×7 평균 → 그 채널의 "중요도" 가중치.
          (3) 가중치 × 특징 지도를 채널 방향으로 더한다 → ReLU (그 클래스 쪽으로 미는 곳만 남김).
          (4) 224×224 로 쌍선형 확대 → 사진마다 0~1 로 정규화(최소-최대).
    속도: 몸통(conv1~layer4) 파라미터의 기울기는 필요 없으므로, 훅에서 layer4 출력을 떼어 내
          그 지점부터만 미분한다. 그래서 역전파는 avgpool + fc 만 지나간다.
    """

    def __init__(self, model):
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.acts = None
        self.handle = self.model.layer4[-1].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        # layer4 출력을 그래프의 출발점(잎)으로 만든다. 값은 그대로라 logits 도 그대로다 (검증 (a)).
        out = output.detach().requires_grad_(True)
        self.acts = out
        return out

    def __call__(self, x, targets_list):
        """x: (N,3,224,224). targets_list: 대상 클래스 배열 목록. 각 원소는 None(=예측 클래스) 또는 (N,) 정수.

        반환: logits (N,3), [지도 (N,224,224) float32 0~1], [지도가 0 이 아닌지 (N,) bool]
        """
        with torch.enable_grad():
            logits = self.model(x)
            acts = self.acts
            cams, valid = [], []
            for k, tgt in enumerate(targets_list):
                tgt = logits.argmax(1) if tgt is None else torch.as_tensor(tgt, dtype=torch.long)
                score = logits.gather(1, tgt.view(-1, 1)).sum()  # 사진끼리는 서로 영향이 없어서 합쳐서 미분해도 된다
                grads, = torch.autograd.grad(score, acts, retain_graph=k < len(targets_list) - 1)
                weights = grads.mean(dim=(2, 3), keepdim=True)            # (N,512,1,1) 채널 중요도
                cam = F.relu((weights * acts.detach()).sum(1, keepdim=True))  # (N,1,7,7)
                cam = F.interpolate(cam, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[:, 0]
                lo = cam.amin(dim=(1, 2), keepdim=True)
                span = cam.amax(dim=(1, 2), keepdim=True) - lo
                ok = span[:, 0, 0] > 0
                cam = torch.where(span > 0, (cam - lo) / span.clamp_min(1e-12), torch.zeros_like(cam))
                cams.append(cam.numpy().astype(np.float32))
                valid.append(ok.numpy())
        return logits.detach(), cams, valid


# ----------------------------------------------------------------------------
# 영역 지표 (폐 마스크가 없어서 쓰는 상자 근사)
# ----------------------------------------------------------------------------
def region_masks(size=SIZE):
    """224×224 위의 세 영역. 픽셀 중심 좌표를 0~1 로 보고 정한다."""
    c = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(c, c)  # X: 가로(왼→오), Y: 세로(위→아래)
    border = (X < 0.15) | (X > 0.85) | (Y < 0.15) | (Y > 0.85)
    corner = (Y < 0.20) & ((X < 0.20) | (X > 0.80))
    center = (X >= 0.15) & (X < 0.85) & (Y >= 0.10) & (Y < 0.85)
    # 테두리 띠 세분: 위(세로 0~15%), 아래(85~100%), 좌우(나머지 높이에서 가로 바깥 15%). 세 개를 더하면 테두리 띠.
    band_top = Y < 0.15
    band_bottom = Y > 0.85
    band_sides = ~band_top & ~band_bottom & ((X < 0.15) | (X > 0.85))
    return {"center": center, "border": border, "corner": corner,
            "band_top": band_top, "band_bottom": band_bottom, "band_sides": band_sides}


MASKS = region_masks()
AREA = {k: float(m.mean()) for k, m in MASKS.items()}  # 지도가 고르게 퍼져 있을 때의 기대 비율
_YC = ((np.arange(SIZE) + 0.5) / SIZE)[:, None].repeat(SIZE, 1).ravel()  # 픽셀 세로 좌표 0~1


def region_fractions(cams):
    """(N,224,224) 지도 → 영역별 "CAM 총합 대비 비율", 활성 면적(지도 ≥ 0.5 인 픽셀 비율),
    지도 무게중심의 세로 위치 cy (0 = 사진 맨 위, 1 = 맨 아래)."""
    flat = cams.reshape(len(cams), -1).astype(np.float64)
    total = flat.sum(1)
    out = {}
    for k, m in MASKS.items():
        s = flat[:, m.ravel()].sum(1)
        out[k] = np.where(total > 0, s / np.maximum(total, 1e-12), np.nan)
    out["active"] = (flat >= 0.5).mean(1)
    out["cy"] = np.where(total > 0, flat @ _YC / np.maximum(total, 1e-12), np.nan)
    return out


def metric_cols(f, i):
    """CSV 한 줄에 넣을 영역 지표 문자열."""
    return {f"{k}_frac" if k not in ("active", "cy") else ("active_area" if k == "active" else "cam_cy"):
            f"{f[k][i]:.4f}" for k in METRICS}


def pearson_rows(a, b):
    """두 지도 묶음 (N,224,224) 의 사진별 피어슨 상관. 한쪽이 상수이면 NaN."""
    a = a.reshape(len(a), -1).astype(np.float64)
    b = b.reshape(len(b), -1).astype(np.float64)
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(den > 0, (a * b).sum(1) / np.maximum(den, 1e-12), np.nan)


def bootstrap_diff(a, b, n_boot=2000, seed=42):
    """평균 차이(a − b)의 95% 구간. 두 그룹 안에서 사진 단위로 따로 복원 추출."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)]
    return np.percentile(d, [2.5, 97.5]).tolist()


def bootstrap_mean(a, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(a)
    return np.percentile([rng.choice(a, len(a)).mean() for _ in range(n_boot)], [2.5, 97.5]).tolist()


def denorm_gray(x):
    """정규화된 입력 (N,3,224,224) → 화면용 회색조 (N,224,224) uint8. 모델이 실제로 본 사진."""
    mean = torch.tensor(T.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(T.IMAGENET_STD).view(1, 3, 1, 1)
    g = (x * std + mean)[:, 0].clamp(0, 1).mul(255).round()
    return g.byte().numpy()


# ----------------------------------------------------------------------------
# 그림
# ----------------------------------------------------------------------------
def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    plt.rcParams["font.family"] = "Apple SD Gothic Neo"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def overlay(ax, gray, cam, title=None, fontsize=10):
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    if cam is not None:
        ax.imshow(cam, cmap="jet", alpha=0.42, vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=fontsize, linespacing=1.25)


def prob_text(p):
    return f"정상 {p[0]:.2f} · 세균 {p[1]:.2f} · 바이러스 {p[2]:.2f}"


def draw_regions(ax, lw=1.4, shade_border=False):
    """영역 정의를 사진 위에 그린다 (픽셀 좌표). 중앙 상자(흰 점선), 위 모서리(노랑), 필요하면 테두리 띠 칠하기."""
    from matplotlib.patches import Rectangle
    s = SIZE
    if shade_border:
        ax.imshow(np.where(MASKS["border"], 1.0, np.nan), cmap="cool", alpha=0.30, vmin=0, vmax=1)
    ax.add_patch(Rectangle((0.15 * s - 0.5, 0.10 * s - 0.5), 0.70 * s, 0.75 * s, fill=False,
                           ec="white", ls="--", lw=lw))
    for x0 in (0, 0.8 * s):
        ax.add_patch(Rectangle((x0 - 0.5, -0.5), 0.2 * s, 0.2 * s, fill=False, ec="yellow", lw=lw))


# ----------------------------------------------------------------------------
# 검증 (일회용 확인)
# ----------------------------------------------------------------------------
def sanity_checks(model_dir, x):
    """(a) 훅 logits = 일반 forward, (b) 같은 입력 두 번 → 같은 지도,
    (c) layer4·fc(X선으로 학습한 부분)를 무작위로 되돌린 모델과 지도 비교."""
    plain = load_model(model_dir)
    with torch.no_grad():
        ref = plain(x)
    cam_model = GradCAM(load_model(model_dir))
    logits, (c1,), _ = cam_model(x, [None])
    _, (c2,), _ = cam_model(x, [None])
    res = {"a_logit_max_abs_diff": float((logits - ref).abs().max()),
           "b_repeat_map_max_abs_diff": float(np.abs(c1 - c2).max())}

    torch.manual_seed(42)
    rnd = load_model(model_dir)
    for m in list(rnd.layer4.modules()) + [rnd.fc]:
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()  # BatchNorm 은 이동 평균도 초기화된다
    rnd_cam = GradCAM(rnd.eval())
    _, (cr,), _ = rnd_cam(x, [None])
    corr = pearson_rows(c1, cr)
    fr_t, fr_r = region_fractions(c1), region_fractions(cr)
    # 평균 지도의 구조: 평균 지도의 픽셀 간 표준편차 (0 이면 평균적으로 아무 데도 안 보는 것)
    res.update({
        "c_n_images": int(len(x)),
        "c_corr_trained_vs_random_mean": float(np.nanmean(corr)),
        "c_center_trained_mean": float(np.nanmean(fr_t["center"])),
        "c_center_random_mean": float(np.nanmean(fr_r["center"])),
        "c_meanmap_pixel_std_trained": float(c1.mean(0).std()),
        "c_meanmap_pixel_std_random": float(cr.mean(0).std()),
        "c_mean_map_corr_trained_vs_random": float(pearson_rows(c1.mean(0)[None], cr.mean(0)[None])[0]),
    })
    return res


# ----------------------------------------------------------------------------
# Q1: test, 모델 D
# ----------------------------------------------------------------------------
def run_q1(args, cam_d, writer, summary, plt):
    t0 = time.time()
    with open(args.test_predictions, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    rows = [r for r in all_rows if r["excluded"] == "False"]
    assert len(rows) == 618, len(rows)
    assert all(r["relative_path"].startswith("test/") for r in rows)
    y = np.array([int(r["true_label"]) for r in rows])
    pred_csv = np.array([int(r["pred_label"]) for r in rows])
    p_csv = np.array([[float(r["p_normal"]), float(r["p_bacterial"]), float(r["p_viral"])] for r in rows])

    p0 = T.build_transform("P0")  # 학습·평가와 똑같은 전처리
    n = len(rows)
    cams_pred = np.zeros((n, SIZE, SIZE), np.float32)
    cams_true = {}  # 오분류 사진만: 정답 클래스 지도
    grays = np.zeros((n, SIZE, SIZE), np.uint8)
    probs = np.zeros((n, 3))
    valid_pred = np.zeros(n, bool)
    for s in range(0, n, args.batch):
        batch = rows[s:s + args.batch]
        xs = []
        for r in batch:
            with Image.open(Path(args.images_dir) / r["relative_path"]) as im:
                xs.append(p0(im.convert("RGB")))
        x = torch.stack(xs)
        yt = y[s:s + len(batch)]
        logits, (cp, ct), (vp, vt) = cam_d(x, [None, yt])
        pr = torch.softmax(logits, 1).numpy()
        probs[s:s + len(batch)] = pr
        cams_pred[s:s + len(batch)] = cp
        valid_pred[s:s + len(batch)] = vp
        grays[s:s + len(batch)] = denorm_gray(x)
        for j in range(len(batch)):
            if pr[j].argmax() != yt[j]:
                cams_true[s + j] = (ct[j], bool(vt[j]))
        print(f"  Q1 test {min(s + args.batch, n)}/{n} ({time.time() - t0:.0f}초)", flush=True)
    pred = probs.argmax(1)
    agree = int((pred == pred_csv).sum())
    max_pdiff = float(np.abs(probs - p_csv).max())
    print(f"  확인: 예측 일치 {agree}/{n}, 확률 최대 차이 {max_pdiff:.2e}", flush=True)

    group = np.array(["?"] * n, dtype=object)
    for name, t, p, _ in GROUPS:
        group[(y == t) & (pred == p)] = name
    fr = region_fractions(cams_pred)
    for i, r in enumerate(rows):
        writer.writerow(dict(
            split="test", model="D", condition="clean", target_kind="pred", target_class=int(pred[i]),
            relative_path=r["relative_path"], true_label=int(y[i]), pred_label=int(pred[i]),
            p_normal=f"{probs[i, 0]:.6f}", p_bacterial=f"{probs[i, 1]:.6f}", p_viral=f"{probs[i, 2]:.6f}",
            outcome_group=group[i], **metric_cols(fr, i), cam_nonzero=bool(valid_pred[i]), corr_with_clean=""))
    mis_idx = sorted(cams_true)
    fr_true = region_fractions(np.stack([cams_true[i][0] for i in mis_idx])) if mis_idx else None
    for k, i in enumerate(mis_idx):
        writer.writerow(dict(
            split="test", model="D", condition="clean", target_kind="true", target_class=int(y[i]),
            relative_path=rows[i]["relative_path"], true_label=int(y[i]), pred_label=int(pred[i]),
            p_normal=f"{probs[i, 0]:.6f}", p_bacterial=f"{probs[i, 1]:.6f}", p_viral=f"{probs[i, 2]:.6f}",
            outcome_group=group[i], **metric_cols(fr_true, k), cam_nonzero=cams_true[i][1], corr_with_clean=""))

    # 그룹별 요약
    gstats = {}
    for name, *_ in GROUPS:
        m = group == name
        d = {"n": int(m.sum())}
        for k in METRICS:
            v = fr[k][m]
            v = v[~np.isnan(v)]
            d[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                    "median": float(np.median(v))} if len(v) else None
        gstats[name] = d
    # 그룹 평균 지도끼리의 상관: "틀린 정상의 지도가 어느 그룹의 지도와 닮았나"
    gmean = np.stack([cams_pred[group == name].mean(0) for name, *_ in GROUPS])
    gnames = [g[0] for g in GROUPS]
    gcorr = {a: {b: float(pearson_rows(gmean[i][None], gmean[j][None])[0]) for j, b in enumerate(gnames)}
             for i, a in enumerate(gnames)}
    # 정답(정상) 클래스 지도 요약 (정상 오분류만)
    true_stats = {}
    for name in ("N→V", "N→B"):
        ks = [k for k, i in enumerate(mis_idx) if group[i] == name]
        true_stats[name] = {k: float(np.nanmean(fr_true[k][ks])) for k in METRICS}
        true_stats[name]["corr_pred_vs_true_map"] = float(np.nanmean(pearson_rows(
            np.stack([cams_pred[mis_idx[k]] for k in ks]), np.stack([cams_true[mis_idx[k]][0] for k in ks]))))

    # 맞힌 정상 vs 틀린 정상: 순위 검정 + 평균 차이 bootstrap 구간
    ok = group == "N→N"
    wrong = (y == 0) & (pred != 0)
    viral_wrong = group == "N→V"
    tests = {}
    for label, mb in (("N→N vs 정상 오분류 전체", wrong), ("N→N vs N→V", viral_wrong)):
        tests[label] = {}
        for k in METRICS:
            a, b = fr[k][mb], fr[k][ok]
            a, b = a[~np.isnan(a)], b[~np.isnan(b)]
            u = stats.mannwhitneyu(a, b, alternative="two-sided")
            tests[label][k] = {"mean_wrong": float(a.mean()), "mean_correct": float(b.mean()),
                               "diff_wrong_minus_correct": float(a.mean() - b.mean()),
                               "diff_ci95": bootstrap_diff(a, b), "mannwhitney_p": float(u.pvalue),
                               "n_wrong": int(len(a)), "n_correct": int(len(b))}
    summary["q1"] = {"n": n, "pred_agree_with_csv": agree, "prob_max_abs_diff": max_pdiff,
                     "n_zero_maps_pred": int((~valid_pred).sum()), "group_stats": gstats,
                     "true_class_map_stats": true_stats, "normal_tests": tests, "group_mean_map_corr": gcorr,
                     "seconds": round(time.time() - t0, 1)}

    # --- 그림 1: 그룹별 예시 4장 ---
    rng = np.random.default_rng(42)
    picks = []
    for name, ko, k in [("N→N", GROUP_KO["N→N"], 4), ("N→V", GROUP_KO["N→V"], 4), ("N→B", GROUP_KO["N→B"], 4),
                        ("B→B", GROUP_KO["B→B"], 4), ("V→V", GROUP_KO["V→V"], 4)]:
        idx = np.where(group == name)[0]
        picks.append((ko, sorted(rng.choice(idx, min(k, len(idx)), replace=False).tolist())))
    bv = sorted(rng.choice(np.where(group == "B→V")[0], 2, replace=False).tolist())
    vb = sorted(rng.choice(np.where(group == "V→B")[0], 2, replace=False).tolist())
    picks.append(("세균성↔바이러스성 혼동 (틀림)", bv + vb))
    summary["q1"]["example_files"] = {ko: [rows[i]["relative_path"] for i in ids] for ko, ids in picks}

    h = 4.9 * len(picks)
    fig, axes = plt.subplots(len(picks), 4, figsize=(15, h))
    for r_i, (ko, ids) in enumerate(picks):
        for c_i in range(4):
            ax = axes[r_i, c_i]
            if c_i >= len(ids):
                ax.axis("off")
                continue
            i = ids[c_i]
            t = (f"정답 {CLASS_KO[y[i]]} → 예측 {CLASS_KO[pred[i]]}\n{prob_text(probs[i])}\n"
                 f"중앙 {fr['center'][i]:.2f} · 테두리 {fr['border'][i]:.2f}(아래 {fr['band_bottom'][i]:.2f}) · "
                 f"모서리 {fr['corner'][i]:.2f}")
            overlay(ax, grays[i], cams_pred[i], t, fontsize=10.5)
            ax.text(3, 218, Path(rows[i]["relative_path"]).name, color="white", fontsize=8,
                    bbox=dict(fc="black", alpha=0.5, pad=1, lw=0))
        axes[r_i, 0].set_ylabel(ko, fontsize=13)
    fig.suptitle("Q1. test 사진에서 최종 모델 D 가 본 곳 (Grad-CAM, 예측 클래스 기준, 그룹당 4장, seed 42)\n"
                 "빨강일수록 그 예측 쪽으로 강하게 민 곳. 사진은 모델 입력 그대로(224×224, 종횡비 변경). "
                 "숫자는 지도 총합 중 영역 비율 (영역 정의는 q1_mean_maps.png 마지막 칸)", fontsize=14)
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.005, top=1 - 1.7 / h, hspace=0.24, wspace=0.05)
    fig.savefig(args.out / "q1_test_examples.png", dpi=110)
    plt.close(fig)

    # --- 그림 2: 정상 오분류 예시의 예측 클래스 지도 vs 정답(정상) 클래스 지도 ---
    mis_examples = picks[1][1] + picks[2][1]  # N→V 4장 + N→B 4장
    fig, axes = plt.subplots(4, 4, figsize=(15, 18))
    for k, i in enumerate(mis_examples):
        r_i, c_i = k // 2, (k % 2) * 2
        overlay(axes[r_i, c_i], grays[i], cams_pred[i],
                f"예측 클래스({CLASS_KO[pred[i]]}) 지도\n{prob_text(probs[i])}", fontsize=10.5)
        overlay(axes[r_i, c_i + 1], grays[i], cams_true[i][0],
                f"정답 클래스(정상) 지도\n{Path(rows[i]['relative_path']).name}", fontsize=10.5)
    fig.suptitle("정상인데 폐렴으로 틀린 test 사진: '폐렴' 쪽으로 민 곳(왼쪽) vs '정상' 쪽으로 민 곳(오른쪽)\n"
                 "위 2줄: 정상→바이러스성, 아래 2줄: 정상→세균성 (모델 D)", fontsize=14)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.005, top=0.94, hspace=0.22, wspace=0.05)
    fig.savefig(args.out / "q1_misclassified_pred_vs_true.png", dpi=110)
    plt.close(fig)

    # --- 그림 3: 그룹별 평균 지도 ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 10.5))
    for ax, (name, _, _, ko) in zip(axes.flat, GROUPS):
        m = group == name
        overlay(ax, grays[m].mean(0), cams_pred[m].mean(0) / max(cams_pred[m].mean(0).max(), 1e-12),
                f"{ko}, {int(m.sum())}장\n중앙 {np.nanmean(fr['center'][m]):.2f} · 테두리 "
                f"{np.nanmean(fr['border'][m]):.2f}(아래 {np.nanmean(fr['band_bottom'][m]):.2f}) · "
                f"모서리 {np.nanmean(fr['corner'][m]):.2f}", fontsize=11.5)
        draw_regions(ax, lw=1.0)
    ax = axes.flat[-1]
    overlay(ax, grays.mean(0), None, "영역 정의 (test 618장 평균 사진)\n흰 점선 상자: 중앙 흉부 상자\n"
            "분홍색 칠: 테두리 띠(바깥 15%), 노란 칸: 위 모서리", fontsize=11)
    draw_regions(ax, lw=2.0, shade_border=True)
    fig.suptitle("Q1. test 그룹별 평균 Grad-CAM (모델 D, 예측 클래스 기준; 평균 지도는 최댓값 1로 다시 맞춤)\n"
                 f"고르게 퍼진 지도라면 비율은 중앙 {AREA['center']:.2f} · 테두리 {AREA['border']:.2f} · "
                 f"모서리 {AREA['corner']:.2f} (면적 비율)", fontsize=14)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.86, hspace=0.28, wspace=0.05)
    fig.savefig(args.out / "q1_mean_maps.png", dpi=110)
    plt.close(fig)
    return rows, grays


# ----------------------------------------------------------------------------
# Q2: val, 모델 B vs D, 노이즈 std 16
# ----------------------------------------------------------------------------
def run_q2(args, cams, writer, summary, plt):
    t0 = time.time()
    with open(args.manifest, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val"]  # val 행만 (순서 = shift_eval 과 같음)
    assert rows and all(r["split"] == "val" for r in rows)
    n = len(rows)
    y = np.array([int(r["label_id"]) for r in rows])
    grays = []
    for r in rows:
        with Image.open(Path(args.images_dir) / r["relative_path"]) as im:
            grays.append(to_gray_224(im))  # shift_eval 과 같은 회색조 경로
    grays = np.stack(grays)
    noisy = np.stack([apply_shift(g, "noise", NOISE_LEVEL, seed=args.seed + i) for i, g in enumerate(grays)])
    print(f"  Q2 val {n}장 읽기·노이즈 완료 ({time.time() - t0:.0f}초)", flush=True)

    res = {m: {"p_clean": np.zeros((n, 3)), "p_noise": np.zeros((n, 3)),
               "fr_clean": {k: np.zeros(n) for k in METRICS},
               "fr_noise": {k: np.zeros(n) for k in METRICS},
               "fr_same": {k: np.zeros(n) for k in METRICS},
               "corr_pred": np.zeros(n), "corr_same": np.zeros(n), "zero_maps": 0} for m in cams}
    mean_maps = {m: {"clean": np.zeros((SIZE, SIZE)), "noise": np.zeros((SIZE, SIZE))} for m in cams}
    flip_maps = {m: {"clean": np.zeros((SIZE, SIZE)), "noise": np.zeros((SIZE, SIZE))} for m in cams}
    n_flip_acc = 0
    for s in range(0, n, args.batch):
        e = min(s + args.batch, n)
        xc, xn = gray_to_tensor(grays[s:e]), gray_to_tensor(noisy[s:e])
        batch_out = {}
        for m, cam in cams.items():
            lc, (cc,), (vc,) = cam(xc, [None])
            pc = lc.argmax(1).numpy()
            ln, (cn, cs), (vn, vs) = cam(xn, [None, pc])  # 노이즈: 예측 클래스 지도 + 원본 예측과 같은 클래스 지도
            r = res[m]
            r["p_clean"][s:e] = torch.softmax(lc, 1).numpy()
            r["p_noise"][s:e] = torch.softmax(ln, 1).numpy()
            for key, c in (("fr_clean", cc), ("fr_noise", cn), ("fr_same", cs)):
                f = region_fractions(c)
                for k in f:
                    r[key][k][s:e] = f[k]
            r["corr_pred"][s:e] = pearson_rows(cc, cn)
            r["corr_same"][s:e] = pearson_rows(cc, cs)
            r["zero_maps"] += int((~vc).sum() + (~vn).sum() + (~vs).sum())
            mean_maps[m]["clean"] += cc.sum(0)
            mean_maps[m]["noise"] += cn.sum(0)
            batch_out[m] = (pc, ln.argmax(1).numpy(), cc, cn)
        # B 가 노이즈로 바이러스성으로 바뀐 사진 (원본 예측은 바이러스성이 아님)
        flip = (batch_out["B"][0] != 2) & (batch_out["B"][1] == 2)
        n_flip_acc += int(flip.sum())
        for m in cams:
            flip_maps[m]["clean"] += batch_out[m][2][flip].sum(0)
            flip_maps[m]["noise"] += batch_out[m][3][flip].sum(0)
        print(f"  Q2 val {e}/{n} ({time.time() - t0:.0f}초)", flush=True)

    # 확인: 장비 차이 시험(shift_eval)의 점수와 같아야 한다
    check = {}
    for m in cams:
        f1c, *_ = T.macro_f1_and_recall(y.tolist(), res[m]["p_clean"].argmax(1).tolist())
        f1n, *_ = T.macro_f1_and_recall(y.tolist(), res[m]["p_noise"].argmax(1).tolist())
        check[m] = {"macro_f1_clean": f1c, "macro_f1_noise_std16": f1n,
                    "pred_share_viral_clean": float((res[m]["p_clean"].argmax(1) == 2).mean()),
                    "pred_share_viral_noise": float((res[m]["p_noise"].argmax(1) == 2).mean())}
    print(f"  확인 (shift_test 와 비교): {json.dumps(check, ensure_ascii=False)}", flush=True)

    pB_c, pB_n = res["B"]["p_clean"].argmax(1), res["B"]["p_noise"].argmax(1)
    pD_c, pD_n = res["D"]["p_clean"].argmax(1), res["D"]["p_noise"].argmax(1)
    flip_b = (pB_c != 2) & (pB_n == 2)
    assert int(flip_b.sum()) == n_flip_acc

    for m in cams:
        r = res[m]
        pc, pn = r["p_clean"].argmax(1), r["p_noise"].argmax(1)
        for i, row in enumerate(rows):
            for cond, kind, tgt, pr, key, corr in (
                    ("clean", "pred", pc[i], r["p_clean"][i], "fr_clean", ""),
                    ("noise16", "pred", pn[i], r["p_noise"][i], "fr_noise", f"{r['corr_pred'][i]:.4f}"),
                    ("noise16", "clean_pred", pc[i], r["p_noise"][i], "fr_same", f"{r['corr_same'][i]:.4f}")):
                f = r[key]
                writer.writerow(dict(
                    split="val", model=m, condition=cond, target_kind=kind, target_class=int(tgt),
                    relative_path=row["relative_path"], true_label=int(y[i]), pred_label=int(pr.argmax()),
                    p_normal=f"{pr[0]:.6f}", p_bacterial=f"{pr[1]:.6f}", p_viral=f"{pr[2]:.6f}",
                    outcome_group=f"{'NBV'[y[i]]}→{'NBV'[pr.argmax()]}", **metric_cols(f, i),
                    cam_nonzero="", corr_with_clean=corr))

    # 요약: 전체 val 과 "B 가 바이러스성으로 바뀐 사진"
    def summarize(mask):
        out = {}
        for m in cams:
            r = res[m]
            d = {"n": int(mask.sum()),
                 "corr_clean_vs_noise_pred_mean": float(np.nanmean(r["corr_pred"][mask])),
                 "corr_clean_vs_noise_pred_median": float(np.nanmedian(r["corr_pred"][mask])),
                 "corr_clean_vs_noise_pred_ci95": bootstrap_mean(r["corr_pred"][mask][~np.isnan(r["corr_pred"][mask])]),
                 "corr_clean_vs_noise_same_mean": float(np.nanmean(r["corr_same"][mask])),
                 "pred_changed_pct": float(((r["p_clean"].argmax(1) != r["p_noise"].argmax(1))[mask]).mean() * 100)}
            for k in METRICS:
                d[f"{k}_clean"] = float(np.nanmean(r["fr_clean"][k][mask]))
                d[f"{k}_noise"] = float(np.nanmean(r["fr_noise"][k][mask]))
                d[f"{k}_change"] = float(np.nanmean(r["fr_noise"][k][mask] - r["fr_clean"][k][mask]))
            out[m] = d
        # B vs D 짝지은 비교 (같은 사진): 상관
        a, b = res["B"]["corr_pred"][mask], res["D"]["corr_pred"][mask]
        ok = ~np.isnan(a) & ~np.isnan(b)
        out["B_minus_D_corr_pred_mean"] = float((a[ok] - b[ok]).mean())
        out["wilcoxon_p_corr_pred"] = float(stats.wilcoxon(a[ok], b[ok]).pvalue) if ok.sum() > 10 else None
        return out

    summary["q2"] = {"n_val": n, "noise": f"shift_filters noise 강도 {NOISE_LEVEL} (std {SHIFTS['noise']['levels'][NOISE_LEVEL - 1]:g}), "
                                          f"seed = {args.seed} + val 순번 (shift_eval 과 같음)",
                     "check_vs_shift_eval": check, "zero_maps": {m: res[m]["zero_maps"] for m in cams},
                     "all_val": summarize(np.ones(n, bool)), "B_flipped_to_viral": summarize(flip_b),
                     "D_flipped_to_viral_n": int(((pD_c != 2) & (pD_n == 2)).sum())}

    # 예시 고르기: 두 모델 모두 원본에서 정상 또는 세균성을 맞히고, 노이즈에서 B 는 바이러스성, D 는 그대로
    cand = (y != 2) & (pB_c == y) & (pD_c == y) & (pB_n == 2) & (pD_n == y)
    rng = np.random.default_rng(args.seed)
    pick = []
    for c in (0, 1):
        idx = np.where(cand & (y == c))[0]
        pick += rng.choice(idx, min(3, len(idx)), replace=False).tolist() if len(idx) else []
    if len(pick) < 6:  # 한쪽 클래스가 모자라면 남은 후보에서 채운다
        rest = np.setdiff1d(np.where(cand)[0], pick)
        pick += rng.choice(rest, min(6 - len(pick), len(rest)), replace=False).tolist() if len(rest) else []
    pick = sorted(pick)
    summary["q2"]["n_candidates"] = {"total": int(cand.sum()), "normal": int((cand & (y == 0)).sum()),
                                     "bacterial": int((cand & (y == 1)).sum())}
    summary["q2"]["example_files"] = [rows[i]["relative_path"] for i in pick]

    # 고른 사진만 지도를 다시 계산 (검증 (b) 로 같은 입력 → 같은 지도임을 확인)
    sel_c, sel_n = gray_to_tensor(grays[pick]), gray_to_tensor(noisy[pick])
    sel = {}
    for m, cam in cams.items():
        lc, (cc,), _ = cam(sel_c, [None])
        ln, (cn,), _ = cam(sel_n, [None])
        sel[m] = (torch.softmax(lc, 1).numpy(), cc, torch.softmax(ln, 1).numpy(), cn)

    if pick:
        h = 4.9 * len(pick)
        fig, axes = plt.subplots(len(pick), 4, figsize=(15, h), squeeze=False)
        for r_i, i in enumerate(pick):
            cells = [("B 원본", grays[i], sel["B"][0][r_i], sel["B"][1][r_i]),
                     ("B 노이즈", noisy[i], sel["B"][2][r_i], sel["B"][3][r_i]),
                     ("D 원본", grays[i], sel["D"][0][r_i], sel["D"][1][r_i]),
                     ("D 노이즈", noisy[i], sel["D"][2][r_i], sel["D"][3][r_i])]
            for c_i, (name, g, p, c) in enumerate(cells):
                f = region_fractions(c[None])
                overlay(axes[r_i, c_i], g, c, f"{name} → 예측 {CLASS_KO[p.argmax()]} ({p.max():.2f})\n{prob_text(p)}\n"
                        f"중앙 {f['center'][0]:.2f} · 테두리 {f['border'][0]:.2f}", fontsize=10.5)
            axes[r_i, 0].set_ylabel(f"정답 {CLASS_KO[y[i]]}\n{Path(rows[i]['relative_path']).name}", fontsize=11)
        fig.suptitle("Q2. 같은 val 사진에 노이즈(std 16)를 넣었을 때 B(노이즈에 약함)와 D(노이즈 증강)가 본 곳\n"
                     f"조건: 원본은 두 모델 모두 맞힘 → 노이즈에서 B 는 바이러스성, D 는 그대로 "
                     f"(후보 {int(cand.sum())}장 중 seed 42 로 {len(pick)}장). 지도는 각 칸의 예측 클래스 기준", fontsize=14)
        fig.subplots_adjust(left=0.05, right=0.99, bottom=0.005, top=1 - 1.7 / h, hspace=0.24, wspace=0.05)
        fig.savefig(args.out / "q2_noise_B_vs_D.png", dpi=110)
        plt.close(fig)

    # 평균 지도: 전체 val, B 가 바이러스성으로 바뀐 사진
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.6))
    nf = max(int(flip_b.sum()), 1)
    for r_i, (title, maps, cnt, base_c, base_n) in enumerate((
            (f"val 전체 {n}장", mean_maps, n, grays.mean(0), noisy.mean(0)),
            (f"B 가 바이러스성으로 바뀐 {int(flip_b.sum())}장", flip_maps, nf,
             grays[flip_b].mean(0), noisy[flip_b].mean(0)))):
        for c_i, (m, cond, base) in enumerate((("B", "clean", base_c), ("B", "noise", base_n),
                                               ("D", "clean", base_c), ("D", "noise", base_n))):
            mm = maps[m][cond] / cnt
            overlay(axes[r_i, c_i], base, mm / max(mm.max(), 1e-12),
                    f"{m} {'원본' if cond == 'clean' else '노이즈 std 16'}\n({title})", fontsize=11.5)
            draw_regions(axes[r_i, c_i], lw=1.0)
    fig.suptitle("Q2. val 평균 Grad-CAM (각 사진의 예측 클래스 기준, 평균 지도는 최댓값 1로 다시 맞춤)\n"
                 "흰 점선: 중앙 흉부 상자, 노란 칸: 위 모서리", fontsize=14)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.87, hspace=0.20, wspace=0.05)
    fig.savefig(args.out / "q2_mean_maps.png", dpi=110)
    plt.close(fig)
    summary["q2"]["seconds"] = round(time.time() - t0, 1)
    return rows, grays


# ----------------------------------------------------------------------------
# 표 (보고서 원본 숫자)
# ----------------------------------------------------------------------------
def write_tables(summary, path):
    q1, q2 = summary["q1"], summary["q2"]
    L = ["# Grad-CAM 자동 표 (gradcam.py 가 만듦)", "",
         f"면적 기준선 (지도가 고르게 퍼졌을 때의 비율): 중앙 {AREA['center']:.3f} · 테두리 {AREA['border']:.3f} · "
         f"모서리 {AREA['corner']:.3f}", "", "## 검증", ""]
    s = summary["sanity"]
    L += [f"- (a) 훅 logits vs 일반 forward 최대 차이: {s['a_logit_max_abs_diff']:.2e}",
          f"- (b) 같은 입력 두 번, 지도 최대 차이: {s['b_repeat_map_max_abs_diff']:.2e}",
          f"- (c) val {s['c_n_images']}장, layer4·fc 무작위화 모델과 비교: 사진별 지도 상관 평균 "
          f"{s['c_corr_trained_vs_random_mean']:.3f}, 중앙 비율 학습 {s['c_center_trained_mean']:.3f} vs 무작위 "
          f"{s['c_center_random_mean']:.3f}, 평균 지도 픽셀 표준편차 학습 {s['c_meanmap_pixel_std_trained']:.3f} vs 무작위 "
          f"{s['c_meanmap_pixel_std_random']:.3f}, 두 평균 지도의 상관 {s['c_mean_map_corr_trained_vs_random']:.3f}",
          f"- Q1 D 예측 vs test_predictions.csv: {q1['pred_agree_with_csv']}/{q1['n']} 일치, 확률 최대 차이 {q1['prob_max_abs_diff']:.1e}",
          "- Q2 val 점수 (shift_test 와 같아야 함): " + ", ".join(
              f"{m} 원본 {c['macro_f1_clean']:.4f} / 노이즈 {c['macro_f1_noise_std16']:.4f} "
              f"(바이러스성 예측 비율 {c['pred_share_viral_clean']:.1%} → {c['pred_share_viral_noise']:.1%})"
              for m, c in q2["check_vs_shift_eval"].items()),
          f"- 0 지도(ReLU 뒤 모두 0) 수: Q1 {q1['n_zero_maps_pred']}, Q2 {q2['zero_maps']}", "",
          "## Q1 test 그룹별 영역 비율 (모델 D, 예측 클래스 지도, 평균 ± 표준편차)", "",
          "| 그룹 | 장수 | " + " | ".join(METRIC_KO[k] for k in METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for name, *_ in GROUPS:
        g = q1["group_stats"][name]
        L.append(f"| {GROUP_KO[name]} | {g['n']} | " + " | ".join(
            f"{g[k]['mean']:.3f} ± {g[k]['sd']:.3f}" for k in METRICS) + " |")
    L += ["", "## Q1 맞힌 정상 vs 틀린 정상 (평균 차이 = 틀림 − 맞힘, 95% bootstrap 구간, Mann-Whitney p)", "",
          "| 비교 | 지표 | 틀림 평균 | 맞힘 평균 | 차이 [95%] | p |", "|---|---|---|---|---|---|"]
    for label, d in q1["normal_tests"].items():
        for k, v in d.items():
            L.append(f"| {label} ({v['n_wrong']} vs {v['n_correct']}) | {METRIC_KO[k]} | {v['mean_wrong']:.3f} | {v['mean_correct']:.3f} | "
                     f"{v['diff_wrong_minus_correct']:+.3f} [{v['diff_ci95'][0]:+.3f}, {v['diff_ci95'][1]:+.3f}] | "
                     f"{v['mannwhitney_p']:.2g} |")
    gc = q1["group_mean_map_corr"]
    names = list(gc)
    L += ["", "## Q1 그룹 평균 지도끼리의 상관 (피어슨, 224×224 평균 지도)", "",
          "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for a in names:
        L.append(f"| {a} | " + " | ".join(f"{gc[a][b]:.2f}" for b in names) + " |")
    L += ["", "## Q1 정상 오분류의 정답(정상) 클래스 지도", ""]
    for name, d in q1["true_class_map_stats"].items():
        L.append(f"- {GROUP_KO[name]}: 중앙 {d['center']:.3f}, 테두리 {d['border']:.3f} (아래 {d['band_bottom']:.3f}), "
                 f"모서리 {d['corner']:.3f}, 무게중심 세로 {d['cy']:.3f}, "
                 f"예측 지도와 정답 지도의 상관 평균 {d['corr_pred_vs_true_map']:.3f}")
    L += ["", f"## Q2 val 노이즈 전후 (모델별, {q2['noise']})", "",
          "| 묶음 | 모델 | 장수 | 예측 바뀜 | 지도 상관 (예측 클래스) 평균 [95%] | 중앙값 | 지도 상관 (같은 클래스) | "
          + " | ".join(f"{METRIC_KO[k]} 원본→노이즈" for k in Q2_COLS) + " |",
          "|---|---|---|---|---|---|---|" + "---|" * len(Q2_COLS)]
    for key, label in (("all_val", "val 전체"), ("B_flipped_to_viral", "B 가 바이러스성으로 바뀐 사진")):
        for m in ("B", "D"):
            d = q2[key][m]
            L.append(f"| {label} | {m} | {d['n']} | {d['pred_changed_pct']:.1f}% | {d['corr_clean_vs_noise_pred_mean']:.3f} "
                     f"[{d['corr_clean_vs_noise_pred_ci95'][0]:.3f}, {d['corr_clean_vs_noise_pred_ci95'][1]:.3f}] | "
                     f"{d['corr_clean_vs_noise_pred_median']:.3f} | {d['corr_clean_vs_noise_same_mean']:.3f} | "
                     + " | ".join(f"{d[k + '_clean']:.3f} → {d[k + '_noise']:.3f}" for k in Q2_COLS) + " |")
        L.append(f"| {label} | B − D 상관 차이 (짝지음) | | | {q2[key]['B_minus_D_corr_pred_mean']:+.3f} "
                 f"(Wilcoxon p {q2[key]['wilcoxon_p_corr_pred']:.2g}) | | |" + " |" * len(Q2_COLS))
    L += ["", f"Q2 예시 후보 {q2['n_candidates']}, D 가 바이러스성으로 바뀐 사진 {q2['D_flipped_to_viral_n']}장", ""]
    path.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
Q2_COLS = ["center", "border", "band_bottom", "corner", "active", "cy"]
FIELDS = ["split", "model", "condition", "target_kind", "target_class", "relative_path", "true_label", "pred_label",
          "p_normal", "p_bacterial", "p_viral", "outcome_group", "center_frac", "border_frac", "corner_frac",
          "band_top_frac", "band_bottom_frac", "band_sides_frac", "active_area", "cam_cy", "cam_nonzero",
          "corr_with_clean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "manifests/split_manifest.csv"))
    ap.add_argument("--test-predictions", default=str(ROOT / "reports/test_eval/aws/test_predictions.csv"))
    ap.add_argument("--images-dir", default=str(Path.home() / "data/chest_xray"))
    ap.add_argument("--out", default=str(ROOT / "reports/gradcam"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=5, help="다른 작업과 CPU 를 나눠 쓰기 위한 스레드 제한")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    plt = setup_matplotlib()
    t0 = time.time()
    summary = {"models": {m: {"dir": str(d.relative_to(ROOT)), "job": JOB_NAMES[m]} for m, d in MODEL_DIRS.items()},
               "target_layer": "model.layer4[-1] (512×7×7)", "region_area": AREA, "threads": args.threads,
               "torch": torch.__version__}

    # 검증용 사진: val 에서 클래스별 20장 (seed 42)
    with open(args.manifest, encoding="utf-8") as f:
        val_rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    rng = np.random.default_rng(args.seed)
    chk = []
    for c in range(3):
        idx = [i for i, r in enumerate(val_rows) if int(r["label_id"]) == c]
        chk += rng.choice(idx, 20, replace=False).tolist()
    p0 = T.build_transform("P0")
    xs = []
    for i in chk:
        with Image.open(Path(args.images_dir) / val_rows[i]["relative_path"]) as im:
            xs.append(p0(im.convert("RGB")))
    summary["sanity"] = sanity_checks(MODEL_DIRS["D"], torch.stack(xs))
    print(f"검증: {json.dumps(summary['sanity'], ensure_ascii=False)}", flush=True)

    cams = {m: GradCAM(load_model(d)) for m, d in MODEL_DIRS.items()}
    with open(args.out / "cam_region_stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        run_q1(args, cams["D"], writer, summary, plt)
        run_q2(args, cams, writer, summary, plt)
    summary["seconds_total"] = round(time.time() - t0, 1)
    (args.out / "gradcam_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_tables(summary, args.out / "gradcam_tables.md")
    print(f"완료 ({time.time() - t0:.0f}초): {args.out}", flush=True)


if __name__ == "__main__":
    main()
