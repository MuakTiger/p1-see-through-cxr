"""장치 설정 필터 v2.1 시험 (입력 크기 S에 직접 거는 판): 학습된 모델에 val 사진을 필터별로 바꿔 넣고 점수를 다시 잰다.

재학습 없음, val만. 모델의 입력 크기는 model-dir/metadata.json 에서 읽는다 (512, 448 등; P0 모델만).
새 5종은 training/equipment_direct.py, v1에서 그대로 쓰는 6종은 training/shift_filters.py (σ는 크기에 비례). 강도는 시험 전에
reports/equipment_v2/strength_fixed_512.md 로 고정했다. 지표와 bootstrap 은 src/shift_eval.py 의 함수를 그대로 쓴다.
맥에서는 CPU, GPU가 있으면(SageMaker g4dn) GPU로 계산한다. 같은 스크립트가 SageMaker 작업의 진입점도 된다 (submit_shift_eval.py).

검증 (스크립트가 스스로 확인해서 리포트에 적는다)
    (a) 학습과 같은 변환 경로의 원본 Macro-F1 = 학습 기록의 베스트 val Macro-F1
    (b) --v1-csv 를 주면, v1에서 그대로 쓰는 6종의 점수 = 그 모델의 기존 장비 차이 시험 결과 (파이프라인 회귀 검증)
    (c) 같은 입력·seed 면 같은 출력, (d) 필터 0단계 = 원본, (e) 코드 해시 = 시험 전에 적어 둔 해시

사용 예 (p1 폴더에서, 맥):
    .venv/bin/python src/equipment_eval_direct.py --model-dir runs/results/b512-20261022-111844/model --out reports/equipment_v2/val_B512
SageMaker 에서는 환경변수 SM_CHANNEL_IMAGES / SM_CHANNEL_MANIFESTS / SM_CHANNEL_MODELS / SM_OUTPUT_DATA_DIR 를 읽고
--model-ids B512,B 처럼 모델 이름을 준다 (모델 폴더 = SM_CHANNEL_MODELS/<이름>, 결과 = SM_OUTPUT_DATA_DIR/<이름>).
"""
import argparse
import csv
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))
import shift_eval as SE  # noqa: E402
import train as T  # noqa: E402
import equipment_direct as ED  # noqa: E402
from equipment_direct import EQUIP, SIGMA0, apply_equipment  # noqa: E402
from shift_filters import SHIFTS, apply_shift, to_gray  # noqa: E402

NEW = list(EQUIP)
REUSED = ["blur", "sharpen", "equalize", "jpeg", "bright", "dark"]
ALL = NEW + REUSED
LEVELS = (1, 2, 3)


def title_of(name):
    return EQUIP[name]["title"] if name in EQUIP else SHIFTS[name]["title"] + " (v1)"


def value_of(name, level):
    return EQUIP[name]["labels"][level - 1] if name in EQUIP else str(SHIFTS[name]["levels"][level - 1])


def make(gray, name, level, seed):
    if name in EQUIP:
        return apply_equipment(gray, name, level, seed=seed)
    return apply_shift(gray, name, level, seed=seed)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def predict_tensor(model, x, device, batch=64):
    return torch.cat([torch.softmax(model(x[i:i + batch].to(device)), dim=1).cpu() for i in range(0, len(x), batch)]).numpy()


def predict_grays(model, grays, device, chunk=64):
    """(N,S,S) uint8 회색조 → 3채널 복제 → ImageNet 정규화 → 확률. 메모리를 아끼려고 chunk 장씩 처리한다."""
    return np.concatenate([predict_tensor(model, SE.gray_to_tensor(grays[s:s + chunk]), device) for s in range(0, len(grays), chunk)])


def plot(results, clean, path, label, size, split="val", n=1030):
    from gradcam import setup_matplotlib  # 맥 전용 글꼴이라 SageMaker 에서는 그림을 건너뛴다 (--plot 0)
    plt = setup_matplotlib()
    fig, axes = plt.subplots(3, 4, figsize=(18, 12.5), sharey=True)
    for ax, name in zip(axes.flat, ALL):
        rs = [r for r in results if r["shift"] == name]
        xs = [0] + [r["level"] for r in rs]
        ax.fill_between(xs, [clean["macro_f1"] + v for v in [0] + [r["delta_f1_ci_low"] for r in rs]],
                        [clean["macro_f1"] + v for v in [0] + [r["delta_f1_ci_high"] for r in rs]],
                        color="black", alpha=0.10, label="Macro-F1 변화 95% 구간")
        ax.plot(xs, [clean["macro_f1"]] + [r["macro_f1"] for r in rs], "k-o", lw=2.5, label="Macro-F1")
        for key, ko, c in [("recall_normal", "정상 recall", "tab:green"), ("recall_bacterial", "세균성 recall", "tab:orange"),
                           ("recall_viral", "바이러스성 recall", "tab:red")]:
            ax.plot(xs, [clean[key]] + [r[key] for r in rs], "--o", color=c, lw=1.5, ms=4, label=ko)
        ax.set_title(title_of(name), fontsize=13, fontweight="bold" if name in EQUIP else "normal")
        ax.set_xticks(xs, ["원본"] + [value_of(name, r["level"]) for r in rs], fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    ax = axes.flat[11]
    ax.axis("off")
    ax.legend(handles, labels, loc="center", fontsize=12, frameon=False)
    axes.flat[0].set_ylabel("점수")
    fig.suptitle(f"장치 설정 필터 v2.1 시험 ({label}, 입력 {size}, {split} {n:,}장, 재학습 없음). 굵은 제목 = 새로 만든 5종", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def find_duplicates_to_exclude(rows):
    """같은 sha256 묶음에서 relative_path 사전순 첫 파일만 남기고 나머지를 제외 목록으로 돌려준다 (training/evaluate.py 와 같은 규칙)."""
    by_hash = {}
    for r in rows:
        by_hash.setdefault(r["sha256"], []).append(r["relative_path"])
    return sorted(p for paths in by_hash.values() if len(paths) > 1 for p in sorted(paths)[1:])


def bootstrap_f1_ci(y, pred, groups, n_boot, seed):
    """환자 묶음 단위 bootstrap 으로 Macro-F1 의 95% 구간."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        vals[b] = SE.macro_f1_fast(y[idx], pred[idx])
    return np.percentile(vals, [2.5, 97.5])


def evaluate(model_dir, out, manifest, images_dir, label, n_boot=1000, seed=42, v1_csv=None, hash_file=None,
             expected_hash=None, limit=0, do_plot=True, device="cpu", split="val", dedupe=False):
    t0 = time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(manifest, encoding="utf-8")) if r["split"] == split]
    assert rows and all(r["split"] == split for r in rows)
    excluded = []
    if dedupe:  # test: 같은 사진(sha256)이 두 번 들어 있는 6쌍은 사본을 평가에서 뺀다 (618장)
        rows.sort(key=lambda r: r["relative_path"])
        excluded = find_duplicates_to_exclude(rows)
        rows = [r for r in rows if r["relative_path"] not in set(excluded)]
    if limit:
        rows = rows[:limit]
    n = len(rows)
    y = np.array([int(r["label_id"]) for r in rows])
    groups = np.array([r["split_group"] for r in rows])
    print(f"[{label}] {split} {n}장 (클래스별 {np.bincount(y, minlength=3).tolist()}), 환자 묶음 {len(set(groups))}개"
          f"{f', 중복 사본 {len(excluded)}장 제외' if dedupe else ''}, 계산 장치 {device}", flush=True)

    preprocess, hist_f1 = SE.read_metadata(model_dir)
    size = SE.read_input_size(model_dir)
    assert preprocess == "P0" and size in SIGMA0, f"P0 모델과 입력 크기 {sorted(SIGMA0)}만 지원한다 (이 모델: {preprocess}, {size})"
    print(f"[{label}] 입력 {size}×{size}, 1mm = {size / 224:.2f}픽셀, 선량 기준 노이즈 {SIGMA0[size]}", flush=True)
    model = SE.load_model(model_dir).to(device)
    p0 = T.build_transform("P0", size)

    # 사진을 한 번만 읽어 회색조(size×size)를 메모리에 둔다. 필터는 이 회색조에 직접 건다 (사진을 다시 열지 않는다).
    grays = np.zeros((n, size, size), np.uint8)
    probs_p0, tensors = [], []
    for i, r in enumerate(rows):
        with Image.open(Path(images_dir) / r["relative_path"]) as im:
            grays[i] = to_gray(im, size)
            tensors.append(p0(im.convert("RGB")))
        if len(tensors) == 32 or i == n - 1:
            probs_p0.append(predict_tensor(model, torch.stack(tensors), device))
            tensors = []
        if (i + 1) % 200 == 0:
            print(f"  사진 읽기 {i + 1}/{n} ({time.time() - t0:.0f}초)", flush=True)

    checks = {"deterministic": True, "level0_identity": True}
    for name in NEW:  # (c) 같은 입력·seed → 같은 출력, (d) 0단계 = 원본 (사진 1장)
        checks["deterministic"] &= bool(np.array_equal(make(grays[0], name, 3, seed), make(grays[0], name, 3, seed)))
        checks["level0_identity"] &= bool(np.array_equal(make(grays[0], name, 0, seed), grays[0]))

    f1_p0, _, _, _, pred_p0 = SE.metrics(y, np.concatenate(probs_p0))
    probs_clean = predict_grays(model, grays, device)
    f1_c, auroc_c, rec_c, _, pred_clean = SE.metrics(y, probs_clean)
    same_pred = float((pred_p0 == pred_clean).mean())
    hist_txt = (f"{hist_f1:.4f}" if hist_f1 is not None else "기록 없음") if split == "val" else "해당 없음 (학습 기록은 val 점수)"
    print(f"[{label}] 확인 (a): 학습 변환 경로 Macro-F1 {f1_p0:.4f} (학습 기록 베스트 {hist_txt}), 필터 경로 원본 {f1_c:.4f}, "
          f"예측 일치 {same_pred:.2%} ({time.time() - t0:.0f}초)", flush=True)
    clean = {"macro_f1": f1_c, "macro_auroc": auroc_c, "recall_normal": rec_c[0], "recall_bacterial": rec_c[1], "recall_viral": rec_c[2]}

    results = []
    for name in ALL:
        for lv in LEVELS:
            shifted = np.stack([make(g, name, lv, seed + i) for i, g in enumerate(grays)])
            probs = predict_grays(model, shifted, device)
            f1, auroc, rec, warns, pred = SE.metrics(y, probs)
            lo, hi = SE.bootstrap_delta(y, pred_clean, pred, groups, n_boot, seed)
            viral = y == 2
            results.append({
                "shift": name, "level": lv, "param_value": value_of(name, lv), "macro_f1": f1, "macro_auroc": auroc,
                "recall_normal": rec[0], "recall_bacterial": rec[1], "recall_viral": rec[2],
                "delta_f1": f1 - f1_c, "delta_f1_ci_low": lo, "delta_f1_ci_high": hi,
                "changed_pred_pct": float((pred != pred_clean).mean() * 100),
                "viral_to_bacterial_pct": float(((pred == 1) & viral).sum() / max(viral.sum(), 1) * 100),
                "pred_share_normal": float((pred == 0).mean()), "pred_share_bacterial": float((pred == 1).mean()),
                "pred_share_viral": float((pred == 2).mean()), "warnings": " | ".join(warns)})
            print(f"  [{label}] {name:9s} 강도{lv} ({value_of(name, lv)}): Macro-F1 {f1:.4f} (변화 {f1 - f1_c:+.4f}, 95% [{lo:+.4f}, {hi:+.4f}]), "
                  f"바이러스 recall {rec[2]:.3f} ({time.time() - t0:.0f}초)", flush=True)

    # (b) v1에서 그대로 쓰는 6종: 이 모델의 기존 장비 차이 시험 결과와 같은지
    v1_note = "해당 없음 (이 모델의 v1 결과 파일을 주지 않음)"
    if v1_csv and Path(v1_csv).is_file() and not limit:
        v1 = {(r["shift"], int(r["level"])): float(r["macro_f1"]) for r in csv.DictReader(open(v1_csv, encoding="utf-8"))}
        diffs = [abs(round(r["macro_f1"], 4) - v1[(r["shift"], r["level"])]) for r in results if r["shift"] in REUSED]
        diffs.append(abs(round(f1_c, 4) - v1[("clean", 0)]))
        v1_note = f"6종 × 3단계 + 원본 {len(diffs)}개의 Macro-F1 최대 차이 {max(diffs):.4f} (`{v1_csv}`와 비교)"
    now_hash = sha256(ED.__file__)
    ref_hash = expected_hash
    if ref_hash is None and hash_file and Path(hash_file).is_file():
        ref_hash = {Path(l.split()[1]).name: l.split()[0] for l in open(hash_file, encoding="utf-8") if l.strip()}.get("equipment_direct.py")
    hash_note = "기준 해시 없음" if ref_hash is None else ("시험 전 해시와 같음" if ref_hash == now_hash else "**시험 전 해시와 다름**")
    hash_note += f" ({now_hash[:12]}…)"
    print(f"[{label}] 확인 (b): {v1_note} / (c) 같은 seed 같은 출력: {checks['deterministic']} / (d) 0단계=원본: {checks['level0_identity']} / (e) {hash_note}", flush=True)

    fields = list(results[0].keys())
    with open(out / "shift_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({"shift": "clean", "level": 0, "param_value": "", "macro_f1": f1_c, "macro_auroc": auroc_c,
                    "recall_normal": rec_c[0], "recall_bacterial": rec_c[1], "recall_viral": rec_c[2], "delta_f1": 0,
                    "delta_f1_ci_low": 0, "delta_f1_ci_high": 0, "changed_pred_pct": 0,
                    "viral_to_bacterial_pct": float(((pred_clean == 1) & (y == 2)).sum() / max((y == 2).sum(), 1) * 100),
                    "pred_share_normal": float((pred_clean == 0).mean()), "pred_share_bacterial": float((pred_clean == 1).mean()),
                    "pred_share_viral": float((pred_clean == 2).mean()), "warnings": ""})
        for r in results:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    if do_plot:
        plot(results, clean, out / "shift_curves.png", label, size, split, n)

    cm = np.bincount(y * 3 + pred_clean, minlength=9).reshape(3, 3)
    f1_lo, f1_hi = bootstrap_f1_ci(y, pred_clean, groups, n_boot, seed)
    clean_lines = ["", "## 원본(필터 없음) 성능", "",
                   f"- Macro-F1 {f1_c:.4f} (95% 구간 [{f1_lo:.3f}, {f1_hi:.3f}], 환자 묶음 단위 bootstrap {n_boot}회), Macro-AUROC {auroc_c:.4f}",
                   f"- 클래스별 recall: 정상 {rec_c[0]:.3f} / 세균성 {rec_c[1]:.3f} / 바이러스성 {rec_c[2]:.3f}",
                   "- 혼동행렬 (행 = 정답, 열 = 예측; 정상 / 세균성 / 바이러스성):", "",
                   "| 정답＼예측 | 정상 | 세균성 | 바이러스성 |", "|---|---|---|---|"]
    clean_lines += [f"| {['정상', '세균성', '바이러스성'][i]} | {cm[i, 0]} | {cm[i, 1]} | {cm[i, 2]} |" for i in range(3)]
    print(f"[{label}] 원본 {split} Macro-F1 {f1_c:.4f} [{f1_lo:.3f}, {f1_hi:.3f}], AUROC {auroc_c:.4f}, 혼동행렬 {cm.tolist()}", flush=True)
    lines = [f"# 장치 설정 필터 v2.1 시험 결과 ({split}, 입력 {size})", "",
             f"- 모델: {label} (증강·전처리 없음, 입력 {size}×{size}, 재학습 없음), 계산 장치 {device}",
             f"- 데이터: {split} {n}장 (정상/세균성/바이러스성 {np.bincount(y, minlength=3).tolist()}), 환자 묶음 {len(set(groups))}개"
             f"{f', 같은 사진의 사본 {len(excluded)}장 제외 ({excluded})' if dedupe else ''}. {split} 사진만 읽음",
             "- 필터와 강도: `strength_fixed_512.md` (시험 전에 고정). 굵게 표시한 행이 새로 만든 5종, (v1)은 기존 필터",
             f"- 95% 구간: 환자 묶음 단위 짝지은 bootstrap {n_boot}회, 노이즈 seed 42 + 사진 번호", "",
             "## 검증", "",
             f"- (a) 학습과 같은 변환 경로의 Macro-F1 {f1_p0:.4f} (학습 기록 베스트 {hist_txt}), 필터 경로의 원본 {f1_c:.4f}, 두 경로의 예측 일치율 {same_pred:.2%}",
             f"- (b) {v1_note}",
             f"- (c) 같은 입력·seed에서 같은 출력: {checks['deterministic']}, (d) 필터 0단계 = 원본: {checks['level0_identity']}",
             f"- (e) 코드 해시: {hash_note}", *clean_lines, "", "## 필터 결과", "",
             "| 필터 | 강도 | 값 | Macro-F1 | 변화 | 95% 구간 | 정상 recall | 세균성 recall | 바이러스성 recall | 예측이 바뀐 사진 | 예측 비율 (정상/세균/바이러스) |",
             "|---|---|---|---|---|---|---|---|---|---|---|",
             f"| 원본 | 0 | - | {f1_c:.3f} | - | - | {rec_c[0]:.3f} | {rec_c[1]:.3f} | {rec_c[2]:.3f} | - | "
             f"{(pred_clean == 0).mean() * 100:.0f}/{(pred_clean == 1).mean() * 100:.0f}/{(pred_clean == 2).mean() * 100:.0f}% |"]
    for r in results:
        t = f"**{title_of(r['shift'])}**" if r["shift"] in EQUIP else title_of(r["shift"])
        lines.append(f"| {t} | {r['level']} | {r['param_value']} | {r['macro_f1']:.3f} | {r['delta_f1']:+.3f} | "
                     f"[{r['delta_f1_ci_low']:+.3f}, {r['delta_f1_ci_high']:+.3f}] | {r['recall_normal']:.3f} | {r['recall_bacterial']:.3f} | "
                     f"{r['recall_viral']:.3f} | {r['changed_pred_pct']:.1f}% | "
                     f"{r['pred_share_normal'] * 100:.0f}/{r['pred_share_bacterial'] * 100:.0f}/{r['pred_share_viral'] * 100:.0f}% |")
    sig = [f"{title_of(r['shift'])} {r['param_value']} ({r['delta_f1']:+.3f})" for r in results if r["delta_f1_ci_high"] < 0 or r["delta_f1_ci_low"] > 0]
    lines += ["", f"95% 구간이 0을 포함하지 않는 조건 {len(sig)}개 / 33개: " + (", ".join(sig) if sig else "없음"),
              "", f"실행 시간: {time.time() - t0:.0f}초 ({device})", ""]
    (out / "shift_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[{label}] 저장: {out} ({time.time() - t0:.0f}초)", flush=True)
    return results, clean


def main():
    env = os.environ.get
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None, help="모델 폴더 하나 (맥)")
    ap.add_argument("--model-ids", default="", help="모델 이름 목록 쉼표 구분 (SageMaker: 폴더 = models-dir/<이름>)")
    ap.add_argument("--models-dir", default=env("SM_CHANNEL_MODELS"))
    ap.add_argument("--manifest", default=(str(Path(env("SM_CHANNEL_MANIFESTS")) / "split_manifest.csv") if env("SM_CHANNEL_MANIFESTS")
                                           else str(ROOT / "manifests/split_manifest.csv")))
    ap.add_argument("--images-dir", default=env("SM_CHANNEL_IMAGES") or str(Path.home() / "data/chest_xray"))
    ap.add_argument("--out", default=env("SM_OUTPUT_DATA_DIR"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="B512")
    ap.add_argument("--v1-csv", default=None, help="같은 모델의 기존 장비 차이 시험 결과 (검증 b, 없으면 건너뜀)")
    ap.add_argument("--hash-file", default=str(ROOT / "reports/equipment_v2/code_sha256_before_val_direct.txt"))
    ap.add_argument("--expected-hash", default=None, help="equipment_direct.py 의 시험 전 sha256 (SageMaker 용)")
    ap.add_argument("--limit", type=int, default=0, help="속도 확인용: 앞의 N장만 (결과는 의미 없음)")
    ap.add_argument("--plot", type=int, default=1, help="그림을 그릴지 (SageMaker 에서는 0)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="계산 장치. auto = cuda 가 있으면 cuda, 아니면 cpu. 맥 GPU 는 mps 를 직접 지정 (추론에만 사용, 학습은 안 함)")
    ap.add_argument("--split", default="val", choices=["val", "test"], help="평가할 사진 묶음. test 는 --plan-file 과 --plan-sha256 이 필요하다")
    ap.add_argument("--plan-file", default=None, help="test 평가의 사전 계획 문서 경로")
    ap.add_argument("--plan-sha256", default=None, help="사전 계획 문서의 sha256 (고정 시점에 적어 둔 값). 다르면 test 를 열지 않는다")
    args = ap.parse_known_args()[0]

    if args.split == "test":  # test 는 사전 계획이 고정된 뒤에만 연다 (사진을 읽기 전에 막는다)
        if args.limit or not args.plan_file or not args.plan_sha256 or sha256(args.plan_file) != args.plan_sha256:
            raise SystemExit("test 평가는 --plan-file 과 그 문서의 --plan-sha256 이 맞아야 하고 --limit 을 쓸 수 없다. test 를 열지 않았다.")
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    common = dict(manifest=args.manifest, images_dir=args.images_dir, n_boot=args.n_boot, seed=args.seed, v1_csv=args.v1_csv,
                  hash_file=args.hash_file, expected_hash=args.expected_hash, limit=args.limit, do_plot=bool(args.plot), device=device,
                  split=args.split, dedupe=(args.split == "test"))
    if args.model_ids:
        for mid in [m for m in args.model_ids.split(",") if m]:
            evaluate(Path(args.models_dir) / mid, Path(args.out) / mid, label=mid, **common)
    else:
        evaluate(args.model_dir, args.out, label=args.label, **common)


if __name__ == "__main__":
    main()
