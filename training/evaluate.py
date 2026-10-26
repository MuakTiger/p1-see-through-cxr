"""최종 test 평가 코드 (사전 고정 계획 reports/test_eval/evaluation_plan.md 를 그대로 따른다).

test 는 이 프로젝트에서 처음이자 마지막으로 쓰는 데이터다. 이 코드는 점수를 보고 무엇을 고르지 않는다.
- 임계값 조정 없음 (예측 = 확률이 가장 큰 클래스)
- test 기반 보정·추가 학습 없음
- 주 모델은 계획에서 이미 정한 D. 다른 모델 7개는 "val 순위가 test 에서도 유지되나" 참고용

이 파일 하나가 SageMaker 학습 컨테이너(entry_point)와 iMac(CPU) 양쪽에서 쓰인다.
지표는 train.py 의 함수(macro_f1_and_recall, ovr_auroc, confusion_counts)를 그대로 써서 학습 때와 정의가 같다.
scikit-learn 은 있으면 재확인용으로만 쓴다 (결과 숫자는 train.py 함수 값).

[데이터 사용 규칙]
    1) split_manifest.csv 는 전체를 읽지만, 영상을 여는 행은 split=="test" 인 624행뿐이다.
       (train 행은 "원래 train 과 같은 번호" 보조 분석을 위해 patient_group 문자열만 읽는다. 영상은 열지 않는다.)
    2) 영상을 여는 곳은 TestDataset.__getitem__ 하나뿐이다. "test 영상만 연다"는 보장은 두 필터에서 온다:
       read_manifest 가 split=="test" 이고 test/ 로 시작하는 행만 남기고, TestDataset 이 행마다 다시 검사한다
       (__getitem__ 도 열기 직전에 경로를 확인). 추가로 __getitem__ 이 실제로 돌려준 행 번호를 모아서,
       연 영상 수와 경로가 test 행과 같은지 마지막에 확인한다.

[확률 저장 - 재실행 이유]
    test_predictions.csv 의 확률은 전체 정밀도(.17g)로 저장한다. 처음에는 소수 8자리로 반올림해서
    아주 작은 확률들이 0.00000000 동점이 되었고, CSV 로 AUROC 를 다시 계산하면 보고 값과 달랐다.
    저장한 뒤 CSV 를 다시 읽어 지표를 재계산하고, 보고 값과 다르면 멈춘다.

[중복 제외 규칙 - 계획 2장]
    sha256 이 같은 파일 묶음에서 relative_path 사전순 첫 파일만 남기고 나머지를 뺀다 → 618장.
    코드로 계산한 제외 목록이 계획에 적힌 6장과 다르면 SystemExit 로 멈춘다.

[로컬 실행 예시 - iMac 터미널, p1 폴더에서]
    .venv/bin/python training/evaluate.py \\
        --images-dir ~/data/chest_xray --manifests-dir manifests --models-dir runs/test_eval_models \\
        --output-dir reports/test_eval/local --plan-path reports/test_eval/evaluation_plan.md \\
        --num-workers 0 --run-label local

[SageMaker 환경변수 - 있으면 CLI 인자보다 우선한다]
    SM_CHANNEL_IMAGES     이미지 루트 (이 아래에 test/ 폴더)
    SM_CHANNEL_MANIFESTS  split_manifest.csv 폴더
    SM_CHANNEL_MODELS     모델 폴더들 (하위 폴더 이름 = 모델 ID, 각 폴더에 best_model.pt + metadata.json)
    SM_OUTPUT_DATA_DIR    결과 저장 폴더 (작업이 끝나면 output.tar.gz 로 묶인다)
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models

import train as T  # 같은 폴더(training/)의 train.py. 지표 함수·전처리·상수를 학습과 똑같이 쓰기 위해

KST = timezone(timedelta(hours=9))
CLASS_NAMES = T.CLASS_NAMES
CLASS_KO = ["정상", "세균성", "바이러스성"]
SHORT = ["normal", "bacterial", "viral"]
NUM_CLASSES = T.NUM_CLASSES

# ---- 계획(evaluation_plan.md)에 적힌 값. 실행 중 이 값과 다르면 멈춘다 ----
PLAN_SHA256 = "ba614c0a63711e11e47b7a5d3bfc38679c6b3532cc5b6642d38dc46eec0edd33"
SPLIT_MANIFEST_SHA256 = "205286f33fe23aacde35de43aa5170a318f64eeae44e05402dae01fc83df0268"
PRIMARY_MODEL_SHA256 = "a8bb8a96275344988935af29aeade0167e49cb77a751cb1a483e737fcce47254"  # 실험 D
EXPECTED_EXCLUDED = [  # 계획 2장의 제외 6장
    "test/NORMAL/NORMAL2-IM-0096-0001.jpeg",
    "test/NORMAL/NORMAL2-IM-0173-0001-0002.jpeg",
    "test/NORMAL/NORMAL2-IM-0246-0001-0002.jpeg",
    "test/PNEUMONIA/person128_bacteria_607.jpeg",
    "test/PNEUMONIA/person134_bacteria_644.jpeg",
    "test/PNEUMONIA/person1661_virus_2873.jpeg",
]
EXPECTED_COUNTS = {  # 세트 이름: (클래스별 영상 수, 환자 그룹 수)
    "618": ([231, 240, 147], 426),
    "624": ([234, 242, 148], 427),
}
EXPECTED_MODELS = ["A", "B", "C", "D", "P1", "P2", "P3", "P4"]


# ----------------------------------------------------------------------------
# 인자와 경로
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="최종 test 평가 (계획 고정, 임계값 조정 없음)")
    p.add_argument("--images-dir", default=None, help="이미지 루트 (SM_CHANNEL_IMAGES 우선)")
    p.add_argument("--manifests-dir", default=None, help="split_manifest.csv 폴더 (SM_CHANNEL_MANIFESTS 우선)")
    p.add_argument("--models-dir", default=None, help="모델 폴더들의 상위 폴더 (SM_CHANNEL_MODELS 우선)")
    p.add_argument("--output-dir", default=None, help="결과 폴더 (SM_OUTPUT_DATA_DIR 우선)")
    p.add_argument("--primary", default="D", help="주 모델 ID (계획 1장: D)")
    p.add_argument("--models", default=",".join(EXPECTED_MODELS), help="평가할 모델 ID 목록 (쉼표 구분)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0, help="iMac 은 0 (피클 오류), SageMaker 는 4")
    p.add_argument("--plan-path", default=None, help="evaluation_plan.md 경로 (로컬에서 해시 직접 확인)")
    p.add_argument("--plan-sha256", default=None, help="제출 스크립트가 계산해 넘긴 계획 해시 (SageMaker 용)")
    p.add_argument("--run-label", default="local", help="결과에 적을 실행 환경 이름 (aws / local)")
    args, unknown = p.parse_known_args(argv)
    if unknown:
        print(f"[경고] 알 수 없는 인자를 무시합니다: {unknown}", flush=True)
    if args.primary != "D" or args.n_boot != 2000 or args.seed != 42:
        raise SystemExit("계획 1·4장과 다릅니다: 주 모델 D, bootstrap 2,000회, seed 42 로만 실행합니다.")
    return args


def resolve_output_dir(cli_value):
    """결과 폴더: SageMaker 면 SM_OUTPUT_DATA_DIR (output.tar.gz 로 묶임), 아니면 --output-dir."""
    for env in ("SM_OUTPUT_DATA_DIR", "SM_MODEL_DIR"):
        if os.environ.get(env):
            return Path(os.environ[env]), f"env:{env}"
    if cli_value:
        return Path(cli_value).expanduser(), "cli:--output-dir"
    raise SystemExit("결과 폴더가 없습니다. SM_OUTPUT_DATA_DIR 또는 --output-dir 를 주세요.")


def check_plan_hash(args):
    """계획 파일이 고정된 그 문서인지 확인한다. 로컬은 파일을 직접, SageMaker 는 제출 스크립트가 넘긴 해시로."""
    if args.plan_path:
        got, how = T.sha256_of_file(Path(args.plan_path).expanduser()), "로컬 파일 직접 계산"
    elif args.plan_sha256:
        got, how = args.plan_sha256, "제출 스크립트가 제출 직전에 계산해 넘긴 값"
    else:
        raise SystemExit("--plan-path 또는 --plan-sha256 이 필요합니다 (계획 고정 확인).")
    if got != PLAN_SHA256:
        raise SystemExit(f"[중단] evaluation_plan.md 해시가 고정 값과 다릅니다: {got}")
    print(f"확인: 계획 sha256 {got[:12]}… 고정 값과 같음 ({how})", flush=True)
    return {"sha256": got, "checked_by": how}


# ----------------------------------------------------------------------------
# manifest 와 중복 제외
# ----------------------------------------------------------------------------
def read_manifest(path):
    """test 행(영상을 열 행)과, 원래 train 행의 patient_group 집합(문자열만)을 돌려준다."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = {"relative_path", "label_id", "label_name", "source_split", "split", "patient_group", "sha256"}
        missing = need - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"split_manifest.csv 에 필요한 열이 없습니다: {sorted(missing)}")
        all_rows = list(reader)
    # 보조 분석 5-2용: 원래 train 폴더에 있던 사진의 번호(patient_group) 집합. 영상은 열지 않는다.
    train_groups = {r["patient_group"] for r in all_rows if r["source_split"] == "train"}
    test_rows = [dict(r) for r in all_rows if r["split"] == "test"]
    del all_rows
    for r in test_rows:
        if r["split"] != "test" or not r["relative_path"].startswith("test/"):  # 이중 확인
            raise SystemExit(f"test 가 아닌 행이 섞였습니다: {r['relative_path']}")
        c = int(r["label_id"])
        if c not in range(NUM_CLASSES) or CLASS_NAMES[c] != r["label_name"]:
            raise SystemExit(f"라벨 번호와 이름이 맞지 않습니다: {r['relative_path']}")
        r["label_id"] = c
    test_rows.sort(key=lambda r: r["relative_path"])  # 순서를 결정적으로
    return test_rows, train_groups


def find_duplicates_to_exclude(test_rows):
    """같은 sha256 묶음에서 relative_path 사전순 첫 파일만 남기고 나머지를 제외 목록으로 돌려준다."""
    by_hash = {}
    for r in test_rows:
        by_hash.setdefault(r["sha256"], []).append(r["relative_path"])
    excluded, pairs = [], []
    for paths in by_hash.values():
        if len(paths) > 1:
            paths = sorted(paths)
            excluded += paths[1:]
            pairs.append({"kept": paths[0], "excluded": paths[1:]})
    return sorted(excluded), sorted(pairs, key=lambda d: d["kept"])


def count_set(y, groups):
    """클래스별 영상 수, 클래스별 환자 수(그 클래스 사진이 있는 그룹 수), 전체 환자 수."""
    n_img = [int((y == c).sum()) for c in range(NUM_CLASSES)]
    n_pat = [len(set(groups[y == c])) for c in range(NUM_CLASSES)]
    return n_img, n_pat, len(set(groups))


# ----------------------------------------------------------------------------
# Dataset / 모델
# ----------------------------------------------------------------------------
class TestDataset(Dataset):
    """test 행만 받는 Dataset. 이 평가 코드에서 영상을 여는 유일한 곳이다."""

    def __init__(self, rows, images_root, transform):
        bad = [r["relative_path"] for r in rows if r["split"] != "test"]
        if bad:
            raise SystemExit(f"test 가 아닌 행이 Dataset 에 들어왔습니다: {bad[:3]}")
        self.rows = rows
        self.root = Path(images_root)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        rel = self.rows[idx]["relative_path"]
        if not rel.startswith("test/"):  # 열기 직전 마지막 확인
            raise RuntimeError(f"test 가 아닌 영상을 열려고 했습니다: {rel}")
        path = self.root / rel
        try:
            with Image.open(path) as im:
                img = im.convert("RGB")  # 학습과 같은 방식으로 3채널 통일
        except Exception as e:
            raise RuntimeError(f"영상을 읽지 못했습니다: {path} ({type(e).__name__}: {e})") from e
        return self.transform(img), idx


def load_model(model_dir, device):
    """resnet18 + fc(512→3). A(특징 고정)도 구조는 같다. strict 로드라 구조가 다르면 바로 오류."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    state = torch.load(Path(model_dir) / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def read_model_meta(model_dir, model_id):
    path = Path(model_dir) / "metadata.json"
    if not path.is_file() or not (Path(model_dir) / "best_model.pt").is_file():
        raise SystemExit(f"모델 {model_id}: best_model.pt 또는 metadata.json 이 없습니다 ({model_dir})")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("class_names") != CLASS_NAMES:
        raise SystemExit(f"모델 {model_id}: 클래스 순서가 {CLASS_NAMES} 와 다릅니다: {meta.get('class_names')}")
    meta.setdefault("preprocess", "P0")
    return meta


@torch.no_grad()
def predict(models_by_id, loader, device, n):
    """같은 전처리를 쓰는 모델 여러 개에 같은 입력 묶음을 넣는다. 확률 = softmax, 순서 = 행 순서.

    반환: (모델별 확률, __getitem__ 이 실제로 돌려준 행 번호 목록). 행 번호는 worker 프로세스가 연
    영상까지 포함한다 (worker 안에서 목록에 적으면 본 프로세스에 전달되지 않으므로 돌려받은 값으로 센다).
    """
    probs = {mid: np.zeros((n, NUM_CLASSES), dtype=np.float64) for mid in models_by_id}
    returned = []
    for x, idx in loader:
        returned += idx.tolist()
        x = x.to(device)
        for mid, model in models_by_id.items():
            probs[mid][idx.numpy()] = torch.softmax(model(x), dim=1).cpu().numpy().astype(np.float64)
    return probs, returned


# ----------------------------------------------------------------------------
# 지표 (train.py 함수 재사용)
# ----------------------------------------------------------------------------
def compute_metrics(y, probs, groups):
    pred = probs.argmax(axis=1)
    cm = T.confusion_counts(y.tolist(), pred.tolist())  # 행 = 실제, 열 = 예측
    macro_f1, recalls, warns = T.macro_f1_and_recall(y.tolist(), pred.tolist())
    macro_auroc, aurocs, warns_auc = T.ovr_auroc(y, probs)
    precisions, f1s, zero_div = [], [], []
    for c in range(NUM_CLASSES):
        tp, n_pred, n_true = cm[c, c], cm[:, c].sum(), cm[c, :].sum()
        if n_pred == 0:
            zero_div.append(CLASS_NAMES[c])  # 정밀도 분모 0 → precision 은 NaN 으로 두고 F1 = 0
        precisions.append(tp / n_pred if n_pred > 0 else float("nan"))
        denom = 2 * tp + (n_pred - tp) + (n_true - tp)
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    if abs(float(np.mean(f1s)) - macro_f1) > 1e-12:  # train.py 정의와 같은지 확인
        raise SystemExit("클래스별 F1 평균이 train.py Macro-F1 과 다릅니다 (지표 구현 오류).")
    n_img, n_pat, n_pat_total = count_set(y, groups)
    return {
        "macro_f1": macro_f1, "macro_auroc": macro_auroc,
        "accuracy": float((pred == y).mean()),
        "precision": [float(v) for v in precisions], "recall": [float(v) for v in recalls],
        "f1": [float(v) for v in f1s], "auroc": [float(v) for v in aurocs],
        "macro_precision": float(np.nanmean(precisions)), "macro_recall": float(np.nanmean(recalls)),
        "confusion": cm.tolist(), "n_images": n_img, "n_patients": n_pat,
        "n_images_total": int(len(y)), "n_patients_total": n_pat_total,
        # train.py 경고 문구의 "val 에" 는 학습용 표현이라 "이 세트에" 로 바꿔 기록한다 (내용은 같음)
        "zero_division_classes": zero_div, "warnings": [w.replace("val 에", "이 세트에") for w in warns + warns_auc],
        "pred": pred,
    }


def bootstrap_ci(y, probs, groups, n_boot, seed):
    """환자 그룹 단위 bootstrap: 그룹을 복원 추출해서 그 그룹의 사진을 모두 넣는다 (계획 4장).

    seed 를 매번 새로 고정하므로 같은 세트라면 모든 모델이 똑같은 재추출 표본으로 계산된다.
    반환: {지표: [2.5%, 97.5%]}, 구간 계산에서 NaN(재추출 표본에 클래스가 빠짐)이 나온 횟수
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    pred = probs.argmax(axis=1)
    vals = {"macro_f1": [], "macro_auroc": [], **{f"recall_{s}": [] for s in SHORT}}
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        f1, recalls, _ = T.macro_f1_and_recall(y[idx].tolist(), pred[idx].tolist())
        auc, _, _ = T.ovr_auroc(y[idx], probs[idx])
        vals["macro_f1"].append(f1)
        vals["macro_auroc"].append(auc)
        for c, s in enumerate(SHORT):
            vals[f"recall_{s}"].append(recalls[c])
    ci, nan_counts = {}, {}
    for k, v in vals.items():
        v = np.asarray(v, dtype=np.float64)
        nan_counts[k] = int(np.isnan(v).sum())
        # 그 클래스가 세트에 아예 없으면(예: 정상 사진이 없는 보조 묶음) 구간도 NaN
        ci[k] = [float("nan")] * 2 if nan_counts[k] == len(v) else [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]
    return ci, nan_counts


def bootstrap_scalar(values, groups, n_boot, seed):
    """사진마다 0/1 값(맞음/틀림 등)의 평균에 대한 환자 그룹 단위 bootstrap 95% 구간."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    stats = [values[np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])].mean()
             for _ in range(n_boot)]
    return [float(x) for x in np.percentile(stats, [2.5, 97.5])]


def sklearn_crosscheck(y, probs, m):
    """scikit-learn 이 있으면 같은 숫자가 나오는지만 확인한다 (결과에는 train.py 값을 쓴다)."""
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    except Exception as e:
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}
    import sklearn
    pred = probs.argmax(axis=1)
    diffs = {
        "macro_f1": abs(f1_score(y, pred, average="macro", zero_division=0) - m["macro_f1"]),
        "precision": float(np.nanmax(np.abs(precision_score(y, pred, average=None, zero_division=np.nan) - np.array(m["precision"])))),
        "recall": float(np.max(np.abs(recall_score(y, pred, average=None, zero_division=0) - np.array(m["recall"])))),
        "auroc": float(np.max(np.abs(np.array([roc_auc_score(y == c, probs[:, c]) for c in range(NUM_CLASSES)])
                                     - np.array(m["auroc"])))),
    }
    return {"available": True, "sklearn_version": sklearn.__version__,
            "max_abs_diff": {k: float(v) for k, v in diffs.items()},
            "all_match_1e-9": all(v < 1e-9 for v in diffs.values())}


def recompute_from_csv(path, reported):
    """저장한 test_predictions.csv 만 다시 읽어서 지표를 재계산하고, 보고 값과 다르면 멈춘다 (가이드 12장).

    reported: {"618": 지표, "624": 지표} (compute_metrics 결과). 확률을 반올림하면 동점이 생겨 AUROC 가
    달라지므로, 이 확인으로 "CSV 에서 같은 숫자가 다시 나온다"를 실행할 때마다 보장한다.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out, worst = {}, 0.0
    sk = None
    try:
        from sklearn.metrics import roc_auc_score
        sk = roc_auc_score
    except Exception:
        pass
    sk_ok = None if sk is None else True
    for name, rep in reported.items():
        sel = [r for r in rows if name == "624" or r["excluded"] == "False"]
        y = np.array([int(r["true_label"]) for r in sel])
        pr = np.array([[float(r["p_normal"]), float(r["p_bacterial"]), float(r["p_viral"])] for r in sel])
        if any(int(r["pred_label"]) != int(p.argmax()) for r, p in zip(sel, pr)):
            raise SystemExit(f"[중단] {name}장: CSV 의 pred_label 이 확률 argmax 와 다릅니다.")
        m = compute_metrics(y, pr, np.array([r["group"] for r in sel]))
        diffs = [abs(m["macro_f1"] - rep["macro_f1"]), abs(m["macro_auroc"] - rep["macro_auroc"]),
                 abs(m["accuracy"] - rep["accuracy"])]
        for k in ("recall", "f1", "auroc"):
            diffs += [abs(a - b) for a, b in zip(m[k], rep[k])]
        if m["confusion"] != rep["confusion"]:
            raise SystemExit(f"[중단] {name}장: CSV 로 다시 계산한 혼동행렬이 다릅니다.")
        d = max(diffs)
        worst = max(worst, d)
        if d > 1e-12:
            raise SystemExit(f"[중단] {name}장: CSV 로 다시 계산한 지표가 보고 값과 다릅니다 (최대 차이 {d:.3e}).")
        if sk is not None:
            sk_auc = [float(sk(y == c, pr[:, c])) for c in range(NUM_CLASSES)]
            sk_d = max(abs(a - b) for a, b in zip(sk_auc, rep["auroc"]))
            sk_ok = sk_ok and sk_d < 1e-9
            out[f"sklearn_auroc_from_csv_{name}"] = sk_auc
        out[f"n_rows_{name}"] = len(sel)
    out["max_abs_diff"] = worst
    out["sklearn_auroc"] = "없음(재확인 생략)" if sk is None else ("일치" if sk_ok else "불일치")
    if sk_ok is False:
        raise SystemExit("[중단] CSV 로 sklearn 이 계산한 AUROC 가 보고 값과 다릅니다.")
    out["probability_format"] = ".17g (전체 정밀도)"
    return out


# ----------------------------------------------------------------------------
# 저장
# ----------------------------------------------------------------------------
def f6(v):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.6f}"


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def error_type(t, p):
    if t == 0:
        return "정상→폐렴"
    if p == 0:
        return "폐렴→정상"
    return "세균성→바이러스성" if t == 1 else "바이러스성→세균성"


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(argv=None):
    t0 = time.time()
    started = datetime.now(KST).isoformat(timespec="seconds")
    args = parse_args(argv)
    images_dir, images_src = T.resolve_path("SM_CHANNEL_IMAGES", args.images_dir, "--images-dir")
    manifests_dir, manifests_src = T.resolve_path("SM_CHANNEL_MANIFESTS", args.manifests_dir, "--manifests-dir")
    models_dir, models_src = T.resolve_path("SM_CHANNEL_MODELS", args.models_dir, "--models-dir")
    output_dir, output_src = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # MPS 는 쓰지 않는다
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else (platform.processor() or "cpu")
    print(f"torch {torch.__version__}, torchvision {torchvision.__version__}, python {platform.python_version()}", flush=True)
    print(f"cuda 사용 가능: {torch.cuda.is_available()}, 장치: {device.type} ({device_name})", flush=True)
    print(f"실행 환경: {args.run_label}, 결과 폴더: {output_dir} ({output_src})", flush=True)

    # ---- 1. 계획·분할 파일 고정 확인 ----
    plan_info = check_plan_hash(args)
    manifest_path = manifests_dir / "split_manifest.csv"
    if not manifest_path.is_file():
        raise SystemExit(f"파일이 없습니다: {manifest_path}")
    manifest_sha = T.sha256_of_file(manifest_path)
    if manifest_sha != SPLIT_MANIFEST_SHA256:
        raise SystemExit(f"[중단] split_manifest.csv 해시가 계획과 다릅니다: {manifest_sha}")
    print(f"확인: split_manifest sha256 {manifest_sha[:12]}… 계획과 같음", flush=True)

    # ---- 2. test 행과 중복 제외 (계획 2장) ----
    test_rows, train_groups = read_manifest(manifest_path)
    excluded, dup_pairs = find_duplicates_to_exclude(test_rows)
    if excluded != sorted(EXPECTED_EXCLUDED):
        raise SystemExit(f"[중단] 코드로 계산한 제외 목록이 계획 2장과 다릅니다:\n  계산: {excluded}\n  계획: {sorted(EXPECTED_EXCLUDED)}")
    print(f"확인: 중복 제외 {len(excluded)}장이 계획 2장 목록과 정확히 같음", flush=True)
    for d in dup_pairs:
        print(f"  남김 {d['kept']} / 제외 {', '.join(d['excluded'])}", flush=True)

    y_all = np.array([r["label_id"] for r in test_rows])
    g_all = np.array([r["patient_group"] for r in test_rows])
    excl_set = set(excluded)
    keep = np.array([r["relative_path"] not in excl_set for r in test_rows])
    overlap = np.array([r["patient_group"] in train_groups for r in test_rows])  # 원래 train 과 같은 번호
    sets = {"618": keep, "624": np.ones(len(test_rows), dtype=bool)}
    for name, mask in sets.items():
        n_img, n_pat, n_pat_total = count_set(y_all[mask], g_all[mask])
        exp_img, exp_pat = EXPECTED_COUNTS[name]
        print(f"{name}장 세트: 영상 {int(mask.sum())}장, 클래스별 {n_img} (정상/세균/바이러스), "
              f"클래스별 환자 {n_pat}, 환자 그룹 {n_pat_total}개", flush=True)
        if n_img != exp_img or n_pat_total != exp_pat:
            raise SystemExit(f"[중단] {name}장 세트 수가 계획과 다릅니다: {n_img}/{n_pat_total} (계획 {exp_img}/{exp_pat})")
    print(f"확인: 618/624 세트의 클래스별 수와 환자 수가 계획과 같음", flush=True)
    print(f"원래 train 과 같은 번호를 가진 test 사진: 624장 중 {int(overlap.sum())}장, 618장 중 {int((overlap & keep).sum())}장",
          flush=True)

    # ---- 3. 모델 읽기 ----
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.primary not in model_ids:
        raise SystemExit(f"주 모델 {args.primary} 가 모델 목록에 없습니다: {model_ids}")
    metas, model_sha = {}, {}
    for mid in model_ids:
        metas[mid] = read_model_meta(models_dir / mid, mid)
        model_sha[mid] = T.sha256_of_file(models_dir / mid / "best_model.pt")
        print(f"모델 {mid}: 실험 {metas[mid].get('experiment_id')}, 전처리 {metas[mid]['preprocess']}, "
              f"베스트 에폭 {metas[mid].get('selected_epoch')}, val Macro-F1 {metas[mid].get('best_val_macro_f1'):.4f}, "
              f"sha256 {model_sha[mid][:12]}…", flush=True)
    if model_sha[args.primary] != PRIMARY_MODEL_SHA256:
        raise SystemExit(f"[중단] 주 모델 가중치 해시가 계획 1장과 다릅니다: {model_sha[args.primary]}")
    if metas[args.primary]["preprocess"] != "P0":
        raise SystemExit("[중단] 주 모델 전처리가 계획 3장(P0)과 다릅니다.")
    print(f"확인: 주 모델 {args.primary} 가중치 sha256 이 계획 1장과 같음, 전처리 P0", flush=True)

    # ---- 4. 예측 (전처리가 같은 모델끼리 한 번에) ----
    opened = []  # __getitem__ 이 실제로 돌려준 행의 경로 (전처리 묶음마다 test 행 전체를 한 번씩 연다)
    probs = {}
    by_pre = {}
    for mid in model_ids:
        by_pre.setdefault(metas[mid]["preprocess"], []).append(mid)
    loader_kwargs = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                         pin_memory=(device.type == "cuda"))
    for pre, mids in by_pre.items():
        t1 = time.time()
        ds = TestDataset(test_rows, images_dir, T.build_transform(pre))  # 학습과 같은 전처리 정의
        loaded = {mid: load_model(models_dir / mid, device) for mid in mids}
        pre_probs, returned = predict(loaded, DataLoader(ds, **loader_kwargs), device, len(test_rows))
        probs.update(pre_probs)
        if sorted(returned) != list(range(len(test_rows))):  # 모든 test 행을 정확히 한 번씩 열었는지
            raise SystemExit(f"[중단] 전처리 {pre}: 연 영상 행이 test 행과 다릅니다 ({len(returned)}개).")
        opened += [test_rows[i]["relative_path"] for i in returned]
        print(f"예측 완료: 전처리 {pre}, 모델 {mids}, 영상 {len(returned)}장 ({time.time() - t1:.0f}초)", flush=True)
        del loaded
    if any(not p.startswith("test/") for p in opened):
        raise SystemExit("[중단] test 가 아닌 영상이 열렸습니다.")
    print(f"확인: 실제로 연 영상 {len(opened)}회 (전처리 {len(by_pre)}묶음 × {len(test_rows)}장), "
          f"서로 다른 경로 {len(set(opened))}개, 모두 test/ 아래", flush=True)
    for mid in model_ids:
        if not np.allclose(probs[mid].sum(1), 1.0, atol=1e-5):
            raise SystemExit(f"모델 {mid}: 확률 합이 1 이 아닙니다 (예측이 비어 있는 행이 있음).")

    # ---- 5. 지표와 bootstrap 구간 ----
    results = {}  # (모델, 세트) → 지표
    for mid in model_ids:
        for name, mask in sets.items():
            t1 = time.time()
            m = compute_metrics(y_all[mask], probs[mid][mask], g_all[mask])
            m["ci"], m["ci_nan_counts"] = bootstrap_ci(y_all[mask], probs[mid][mask], g_all[mask], args.n_boot, args.seed)
            results[(mid, name)] = m
            print(f"[{mid} / {name}장] Macro-F1 {m['macro_f1']:.4f} [95% {m['ci']['macro_f1'][0]:.4f}, "
                  f"{m['ci']['macro_f1'][1]:.4f}], Macro AUROC {m['macro_auroc']:.4f}, 정확도 {m['accuracy']:.4f}, "
                  f"recall 정상/세균/바이러스 {m['recall'][0]:.3f}/{m['recall'][1]:.3f}/{m['recall'][2]:.3f} "
                  f"({time.time() - t1:.0f}초)", flush=True)
            if m["zero_division_classes"]:
                print(f"  [기록] 정밀도 분모 0 → F1 = 0 처리: {m['zero_division_classes']}", flush=True)

    P = args.primary
    pm = results[(P, "618")]
    crosscheck = sklearn_crosscheck(y_all[keep], probs[P][keep], pm)
    print(f"scikit-learn 재확인: {crosscheck}", flush=True)

    # ---- 6. 저장 ----
    # 6-1 test_predictions.csv: 주 모델, 624행 (제외 6장은 excluded=True)
    write_csv(output_dir / "test_predictions.csv",
              ["relative_path", "group", "true_label", "pred_label", "p_normal", "p_bacterial", "p_viral", "excluded"],
              [[r["relative_path"], r["patient_group"], int(y_all[i]), int(probs[P][i].argmax()),
                *[f"{v:.17g}" for v in probs[P][i]], str(not keep[i])] for i, r in enumerate(test_rows)])
    csv_check = recompute_from_csv(output_dir / "test_predictions.csv", {n: results[(P, n)] for n in sets})
    print(f"확인: test_predictions.csv 만으로 다시 계산한 지표가 보고 값과 같음 "
          f"(최대 차이 {csv_check['max_abs_diff']:.1e}, sklearn AUROC 재계산 {csv_check['sklearn_auroc']})", flush=True)

    # 6-2 metrics.csv: 모델별 · 세트별 · 클래스별 + 평균
    header = ["model_id", "experiment_id", "preprocess", "eval_set", "scope", "n_images", "n_patients",
              "precision", "recall", "recall_ci_low", "recall_ci_high", "f1", "auroc", "accuracy",
              "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "macro_auroc", "macro_auroc_ci_low",
              "macro_auroc_ci_high", "n_boot", "zero_division", "warnings"]
    rows = []
    for (mid, name), m in results.items():
        base = [mid, metas[mid].get("experiment_id"), metas[mid]["preprocess"], name]
        for c in range(NUM_CLASSES):
            ci = m["ci"][f"recall_{SHORT[c]}"]
            rows.append(base + [CLASS_NAMES[c], m["n_images"][c], m["n_patients"][c], f6(m["precision"][c]),
                                f6(m["recall"][c]), f6(ci[0]), f6(ci[1]), f6(m["f1"][c]), f6(m["auroc"][c]),
                                "", "", "", "", "", "", "", args.n_boot,
                                str(CLASS_NAMES[c] in m["zero_division_classes"]), ""])
        rows.append(base + ["macro", m["n_images_total"], m["n_patients_total"], f6(m["macro_precision"]),
                            f6(m["macro_recall"]), "", "", "", "", f6(m["accuracy"]), f6(m["macro_f1"]),
                            f6(m["ci"]["macro_f1"][0]), f6(m["ci"]["macro_f1"][1]), f6(m["macro_auroc"]),
                            f6(m["ci"]["macro_auroc"][0]), f6(m["ci"]["macro_auroc"][1]), args.n_boot,
                            "|".join(m["zero_division_classes"]), " | ".join(m["warnings"])])
    write_csv(output_dir / "metrics.csv", header, rows)

    # 6-3 confusion_matrix.csv: 주 모델, 618/624 (행 = 실제, 열 = 예측)
    write_csv(output_dir / "confusion_matrix.csv",
              ["model_id", "eval_set", "true_label", "true_name", *[f"pred_{n}" for n in CLASS_NAMES], "row_total"],
              [[P, name, c, CLASS_NAMES[c], *results[(P, name)]["confusion"][c], sum(results[(P, name)]["confusion"][c])]
               for name in sets for c in range(NUM_CLASSES)])

    # 6-4 misclassified.csv: 주 모델, 618장에서 틀린 사진 (오류 종류별)
    mis = []
    for i, r in enumerate(test_rows):
        t, p = int(y_all[i]), int(probs[P][i].argmax())
        if keep[i] and t != p:
            mis.append([error_type(t, p), r["relative_path"], r["patient_group"], t, CLASS_NAMES[t], p, CLASS_NAMES[p],
                        *[f"{v:.6f}" for v in probs[P][i]], f"{probs[P][i][t]:.6f}", f"{probs[P][i][p]:.6f}",
                        str(bool(overlap[i]))])
    order = {"정상→폐렴": 0, "폐렴→정상": 1, "세균성→바이러스성": 2, "바이러스성→세균성": 3}
    mis.sort(key=lambda row: (order[row[0]], -float(row[11])))
    write_csv(output_dir / "misclassified.csv",
              ["error_type", "relative_path", "group", "true_label", "true_name", "pred_label", "pred_name",
               "p_normal", "p_bacterial", "p_viral", "p_true", "p_pred", "group_in_original_train"], mis)
    mis_counts = {k: sum(1 for row in mis if row[0] == k) for k in order}
    print(f"오분류 (주 모델, 618장): {len(mis)}장 {mis_counts}", flush=True)

    # 6-5 subgroup_overlap.csv: 원래 train 과 같은 번호 그룹 vs 나머지 (주 모델)
    sub_rows, sub_summary = [], {}
    for name, mask in sets.items():
        for sub, smask in (("same_number_as_original_train", mask & overlap), ("rest", mask & ~overlap)):
            m = compute_metrics(y_all[smask], probs[P][smask], g_all[smask])
            ci, _ = bootstrap_ci(y_all[smask], probs[P][smask], g_all[smask], args.n_boot, args.seed)
            # 추가 비교(계획 밖, 설명용): 번호 겹침 묶음에는 정상 사진이 구조적으로 없어서(정상 번호는 train 과 겹치지 않음)
            # 3클래스 Macro-F1 은 두 묶음의 클래스 구성 차이를 그대로 반영한다. 그래서 폐렴 사진만 놓고
            # "세균성/바이러스성 중 맞힌 비율"을 함께 적는다 (정상으로 예측한 폐렴도 틀림으로 센다).
            pmask = smask & (y_all > 0)
            correct = (probs[P][pmask].argmax(1) == y_all[pmask]).astype(np.float64)
            pn_ci = bootstrap_scalar(correct, g_all[pmask], args.n_boot, args.seed)
            sub_rows.append([P, name, sub, m["n_images_total"], m["n_patients_total"], *m["n_images"],
                             f6(m["macro_f1"]), f6(ci["macro_f1"][0]), f6(ci["macro_f1"][1]), f6(m["macro_auroc"]),
                             f6(m["accuracy"]), *[f6(v) for v in m["recall"]], int(pmask.sum()),
                             f6(float(correct.mean())), f6(pn_ci[0]), f6(pn_ci[1]), " | ".join(m["warnings"])])
            sub_summary[f"{name}/{sub}"] = {"n_images": m["n_images_total"], "n_patients": m["n_patients_total"],
                                            "n_images_by_class": m["n_images"], "macro_f1": m["macro_f1"],
                                            "macro_f1_ci95": ci["macro_f1"], "recall": m["recall"],
                                            "pneumonia_only_n": int(pmask.sum()),
                                            "pneumonia_only_accuracy": float(correct.mean()),
                                            "pneumonia_only_accuracy_ci95": pn_ci, "warnings": m["warnings"]}
            print(f"[보조 {name}장 / {sub}] 영상 {m['n_images_total']}장 {m['n_images']}, 환자 {m['n_patients_total']}, "
                  f"Macro-F1 {m['macro_f1']:.4f} [{ci['macro_f1'][0]:.4f}, {ci['macro_f1'][1]:.4f}], "
                  f"폐렴 {int(pmask.sum())}장 중 맞힌 비율 {correct.mean():.4f} [{pn_ci[0]:.4f}, {pn_ci[1]:.4f}]", flush=True)
    write_csv(output_dir / "subgroup_overlap.csv",
              ["model_id", "eval_set", "subgroup", "n_images", "n_patients", "n_normal", "n_bacterial", "n_viral",
               "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "macro_auroc", "accuracy",
               "recall_normal", "recall_bacterial", "recall_viral", "pneumonia_only_n", "pneumonia_only_accuracy",
               "pneumonia_only_accuracy_ci_low", "pneumonia_only_accuracy_ci_high", "warnings"], sub_rows)

    # 6-6 all_models_test.csv: 모델 8개의 618장 결과 (참고용, 선택에 쓰지 않음)
    val_f1 = {mid: metas[mid].get("best_val_macro_f1") for mid in model_ids}
    val_rank = {mid: k + 1 for k, mid in enumerate(sorted(model_ids, key=lambda m: -val_f1[m]))}
    test_rank = {mid: k + 1 for k, mid in enumerate(sorted(model_ids, key=lambda m: -results[(m, "618")]["macro_f1"]))}
    am_rows = []
    for mid in model_ids:
        m = results[(mid, "618")]
        am_rows.append([mid, metas[mid].get("experiment_id"), metas[mid]["preprocess"], metas[mid].get("train_mode"),
                        metas[mid].get("augment", "none"), metas[mid].get("selected_epoch"), f6(val_f1[mid]),
                        f6(m["macro_f1"]), f6(m["ci"]["macro_f1"][0]), f6(m["ci"]["macro_f1"][1]), f6(m["macro_auroc"]),
                        f6(m["accuracy"]), *[f6(v) for v in m["recall"]], val_rank[mid], test_rank[mid],
                        str(mid == P), model_sha[mid]])
    write_csv(output_dir / "all_models_test.csv",
              ["model_id", "experiment_id", "preprocess", "train_mode", "augment", "selected_epoch", "val_macro_f1",
               "test_macro_f1", "test_macro_f1_ci_low", "test_macro_f1_ci_high", "test_macro_auroc", "test_accuracy",
               "test_recall_normal", "test_recall_bacterial", "test_recall_viral", "val_rank", "test_rank",
               "is_primary", "model_sha256"], am_rows)

    # 6-7 evaluation_summary.json
    def short(m):
        return {k: m[k] for k in ("macro_f1", "macro_auroc", "accuracy", "precision", "recall", "f1", "auroc",
                                  "confusion", "n_images", "n_patients", "n_images_total", "n_patients_total",
                                  "zero_division_classes", "warnings", "ci", "ci_nan_counts")}
    summary = {
        "run_label": args.run_label,
        "sagemaker_job_name": os.environ.get("TRAINING_JOB_NAME"),
        "started_at_kst": started, "finished_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - t0, 1),
        "device": f"{device.type} ({device_name})",
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "python": platform.python_version(), "numpy": np.__version__},
        "paths": {"images": images_src, "manifests": manifests_src, "models": models_src, "output": output_src},
        "evaluation_plan": plan_info,
        "split_manifest_sha256": manifest_sha,
        "model_sha256": model_sha,
        "models": {mid: {"experiment_id": metas[mid].get("experiment_id"), "preprocess": metas[mid]["preprocess"],
                         "selected_epoch": metas[mid].get("selected_epoch"),
                         "best_val_macro_f1": metas[mid].get("best_val_macro_f1")} for mid in model_ids},
        "primary_model": P,
        "rules": {"prediction": "argmax(softmax), 임계값 조정 없음", "bootstrap": f"patient_group 단위 {args.n_boot}회, seed {args.seed}, 95% 백분위",
                  "duplicates": "같은 sha256 묶음에서 relative_path 사전순 첫 파일만 남김", "zero_division": "정밀도 분모 0 → F1 = 0"},
        "checks": {
            "plan_sha256_matches": True, "split_manifest_sha256_matches": True, "primary_model_sha256_matches": True,
            "excluded_matches_plan": True, "excluded": excluded, "duplicate_pairs": dup_pairs,
            "counts_match_plan": True,
            "opened_images_all_test": all(p.startswith("test/") for p in opened),
            "opened_images_basis": ("read_manifest 필터(split==test, test/ 경로) + TestDataset 행 검사 + __getitem__ 열기 직전 검사. "
                                    "개수는 __getitem__ 이 실제로 돌려준 행 번호로 셈"),
            "opened_image_count": len(opened), "unique_opened_images": len(set(opened)),
            "test_predictions_csv_recompute": csv_check,
        },
        "image_counts": {name: {"n_images": int(mask.sum()), "by_class": count_set(y_all[mask], g_all[mask])[0],
                                "patients_by_class": count_set(y_all[mask], g_all[mask])[1],
                                "patients_total": count_set(y_all[mask], g_all[mask])[2]} for name, mask in sets.items()},
        "primary_618": short(pm), "primary_624": short(results[(P, "624")]),
        "misclassified_counts_618": mis_counts,
        "subgroup_overlap": sub_summary,
        "all_models_618": {mid: {"macro_f1": results[(mid, "618")]["macro_f1"],
                                 "macro_f1_ci95": results[(mid, "618")]["ci"]["macro_f1"],
                                 "macro_auroc": results[(mid, "618")]["macro_auroc"],
                                 "val_macro_f1": val_f1[mid], "val_rank": val_rank[mid], "test_rank": test_rank[mid]}
                           for mid in model_ids},
        "sklearn_crosscheck_primary_618": crosscheck,
        "outputs": ["test_predictions.csv", "metrics.csv", "confusion_matrix.csv", "misclassified.csv",
                    "subgroup_overlap.csv", "all_models_test.csv", "evaluation_summary.json"],
    }
    write_json(output_dir / "evaluation_summary.json", summary)
    print(f"저장 완료: {output_dir} ({', '.join(summary['outputs'])}), 전체 {time.time() - t0:.0f}초", flush=True)


if __name__ == "__main__":
    main()
