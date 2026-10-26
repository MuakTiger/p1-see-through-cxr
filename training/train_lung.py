"""[train_lung.py] train_aug.py 에 폐 영역 입력 하나만 더한 복사본 (입력 = 회색 S×S → [학습만 --augment equip 필터] → 폐 껍질 밖을 0으로).
기준 train_aug.py 의 sha256: acb3c7e800dad079fad1af0b9a5ce83fcd345c11b54034956fc4b5e3dc42c65c
  폐 마스크의 볼록 껍질은 --lung-masks (기본: weights 폴더의 lung_hull_masks.npz, src/make_hull_masks.py 로 만듦)에서 읽는다.
  필터를 먼저 걸고 그다음 폐 처리를 하는 순서다 (평가 src/equipment_eval_lung.py 와 같은 순서, 같은 lung_prep 규칙). --augment none/equip 만 지원한다.
  [train_aug.py] train.py 에 장치 설정 필터 증강 하나(--augment equip)만 더한 복사본. --augment none/noise 는 train.py 와 같은 동작이다.
기준 train.py 의 sha256: 42b42903dc1a6e3cad0083c9abb25df011c5731ff9e2bbf16f611cbec3d8f5af
  --augment equip      학습 사진의 --equip-prob 에 기기 설정 필터 5종(선량·조사시간·산란·kVp·센서) 중 하나를 무작위로 건다.
                       세기는 사진마다 u ~ Uniform(0, 1) 이고 u=0 이 원본, u=1 이 시험의 "강"과 같은 세기다 (training/equipment_direct.py).
                       val(에폭 선택용)은 증강 없이 원본이다. 에폭 선택 규칙은 train.py 와 같다.

흉부 X선 3분류 학습 코드 (실험 A: ResNet18 특징 고정 / 실험 B: layer4 미세조정 /
실험 C: B + 클래스 가중치 / 실험 D: B + 노이즈 증강).

클래스: 0 NORMAL / 1 BACTERIAL_PNEUMONIA / 2 VIRAL_PNEUMONIA
학습 모드(--train-mode):
    frozen (기본, 실험 A)  몸통은 ImageNet 그대로 얼리고 새 fc 만 학습 (학습률 --lr)
    layer4 (실험 B)        마지막 잔차 블록 layer4 와 fc 를 학습 (fc 는 --lr, layer4 는 --lr-finetune)
증강(--augment):
    none  (기본)           증강 없음. A·B 와 완전히 같은 동작
    noise (실험 D)         학습 사진에만 회색 가우시안 노이즈를 확률 --noise-prob 로 더한다.
                           세기 std 는 사진마다 0 ~ --noise-std-max (0~255 기준) 에서 고르게 뽑는다.
                           val(검증)과 추론 전처리는 P0 그대로다.
클래스 가중치(--class-weight):
    none    (기본)         가중치 없음. A·B·D 와 완전히 같은 동작
    inverse (실험 C)       실제로 학습에 쓰는 train 영상의 클래스별 수 n_c 로 w_c = N / (K · n_c) 를 계산해
                           학습 손실에만 곱한다 (N = train 영상 수, K = 3). val 은 계산에 쓰지 않는다.
                           val_loss 는 실험끼리 비교할 수 있게 가중치 없는 CrossEntropy 로 잰다.
입력 전처리(--preprocess, P 시리즈, 정의는 preprocess.py):
    P0 (기본)              원본. A·B·C·D 와 완전히 같은 경로
    P1 HE / P2 CLAHE(clip 2.0, 8x8) / P3 LPF 가우시안(σ 1.5) / P4 HPF 언샤프(σ 2, α 1.0)
                           224 회색조에서 적용하는 결정적 변환. train 과 val 모두에 똑같이 적용한다 (증강 아님).
이 파일 하나가 iMac(CPU) 시운전과 SageMaker PyTorch 컨테이너(entry_point) 양쪽에서 쓰인다.
쓰는 라이브러리: torch, torchvision, numpy, PIL, 파이썬 표준 라이브러리, 그리고 전처리용 opencv(cv2)
(cv2 는 preprocess.py / shift_filters.py 에서만 쓴다. 학습 컨테이너에 opencv-python 4.9.0.80 이 있음을 확인했다.
SageMaker 학습 인스턴스는 인터넷이 없다고 가정하므로 scikit-learn, pandas 는 쓰지 않는다).

[로컬 시운전 명령 예시 - iMac 터미널, p1 폴더에서]
    .venv/bin/python training/train.py \\
        --images-dir ~/data/chest_xray --manifests-dir manifests --weights-dir weights \\
        --model-dir runs/dryrun_A/model --output-dir runs/dryrun_A/output \\
        --epochs 2 --batch-size 16 --num-workers 0 \\
        --max-train-samples 96 --max-val-samples 48
    실험 B 는 위 명령에 --train-mode layer4 를 붙이고 출력 폴더를 runs/dryrun_B 로 바꾼다.
    실험 D 는 실험 B 명령에 --augment noise 를 더 붙이고 출력 폴더를 runs/dryrun_D 로 바꾼다.
    실험 C 는 실험 B 명령에 --class-weight inverse 를 더 붙이고 출력 폴더를 runs/dryrun_C 로 바꾼다.
    P 시리즈는 실험 B 명령에 --preprocess P2 (P1~P4) 를 더 붙이고 출력 폴더를 runs/dryrun_P2 로 바꾼다.

[SageMaker 환경변수 - 있으면 CLI 인자보다 우선한다]
    SM_CHANNEL_IMAGES     이미지 루트 (이 아래에 train/ val/ test/ 폴더)
    SM_CHANNEL_MANIFESTS  split_manifest.csv 가 있는 폴더
    SM_CHANNEL_WEIGHTS    resnet18_imagenet.pt 가 있는 폴더
    SM_MODEL_DIR          best_model.pt, metadata.json 저장 폴더
    SM_OUTPUT_DATA_DIR    history.csv, val_predictions.csv, config.json, experiment_summary.json 저장 폴더
    하이퍼파라미터는 SageMaker 가 --epochs 30 같은 인자로 넘긴다.

[데이터 사용 규칙: test/check 영상은 절대 열지 않는다]
    1) split_manifest.csv 를 읽을 때 split 열이 "train" 또는 "val" 인 행만 남긴다.
       test/check 행은 이 단계에서 버려지므로 Dataset, 통계 계산, 파일 존재 확인 어디에도 전달되지 않는다.
    2) 남긴 행의 split 값이 {train, val} 밖이면 SystemExit 로 중단한다(이중 확인).
    3) 영상을 여는 곳은 TrainValDataset.__getitem__ 하나뿐이고, 이 Dataset 은 위에서 남긴 행만 받는다.
    4) 파일명은 모델 입력으로 쓰지 않는다. 경로는 영상 로딩과 기록용이다.

[검증 기록] 시운전 결과(지표 직접 구현 vs scikit-learn 비교, test/check 비접근 증거,
    best_model.pt 로드 재현)는 runs/dryrun_A/output/verification.md 에 적어 두었다.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import preprocess as PP  # 같은 폴더(training/)의 preprocess.py. P1~P4 전처리 정의
import equipment_direct as ED  # 같은 폴더(training/)의 equipment_direct.py. 장치 설정 필터 5종
import lung_prep as LP  # 같은 폴더(training/)의 lung_prep.py. 폐 껍질 밖을 0으로 (규칙은 평가와 같음)
from shift_filters import to_gray as _to_gray  # 평가의 필터 경로와 같은 회색 S×S 변환

CLASS_NAMES = ["NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA"]
NUM_CLASSES = 3
ALLOWED_SPLITS = {"train", "val"}  # 이 학습 코드가 읽어도 되는 split 값 (test, check 는 제외)
INPUT_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SELECTION_RULE = (
    "val_macro_f1 가 가장 높은 에폭. 동률이면 val_loss 가 낮은 쪽, "
    "그래도 같으면 더 이른 에폭. 조기 종료 없음(최대 에폭까지 실행)."
)
HISTORY_COLUMNS = [
    "epoch", "train_loss", "val_loss", "val_macro_f1", "val_macro_auroc",
    "val_recall_normal", "val_recall_bacterial", "val_recall_viral",
    "val_auroc_normal", "val_auroc_bacterial", "val_auroc_viral",
    "elapsed_seconds", "is_best", "warnings",
]
SHORT_NAMES = ["normal", "bacterial", "viral"]  # history.csv 열 이름용
TRAIN_MODES = ["frozen", "layer4"]  # frozen = 실험 A, layer4 = 실험 B
PREPROCESS_CHOICES = list(PP.PREPROCESS)  # P0~P4 (P5 는 이번 범위에서 구현하지 않음)
TRAINABLE_DESC = {
    "frozen": "fc only (실험 A: 특징 고정)",
    "layer4": "layer4 + fc (실험 B: layer4 미세조정)",
}
OPTIMIZER_DESC = {
    "frozen": "Adam (fc 파라미터만)",
    "layer4": "Adam (파라미터 그룹 2개: fc=lr, layer4=lr_finetune)",
}
AUGMENT_CHOICES = ["none", "noise", "equip"]  # none = 증강 없음(A·B), noise = 실험 D, equip = 장치 설정 필터 증강
AUGMENT_NOTE = "증강은 학습(train)에만 적용, val 과 추론 전처리는 P0 와 동일"
CLASS_WEIGHT_CHOICES = ["none", "inverse"]  # none = 가중치 없음(A·B·D), inverse = 실험 C
CLASS_WEIGHT_NOTE = (
    "w_c = N / (K * n_c), 실제로 학습에 쓴 train 영상 수로만 계산 (val 사용 안 함). "
    "학습 손실에만 적용, val_loss 는 가중치 없는 CrossEntropy"
)


# ----------------------------------------------------------------------------
# 경로 해석
# ----------------------------------------------------------------------------
def resolve_path(env_name, cli_value, cli_flag):
    """SageMaker 환경변수가 있으면 그것을, 없으면 CLI 인자를 쓴다. 둘 다 없으면 중단."""
    env_value = os.environ.get(env_name)
    if env_value:
        return Path(env_value), f"env:{env_name}"
    if cli_value:
        return Path(cli_value).expanduser(), f"cli:{cli_flag}"
    raise SystemExit(f"경로가 없습니다. 환경변수 {env_name} 또는 인자 {cli_flag} 중 하나를 주세요.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ResNet18 학습 (실험 A: 특징 고정 / B: layer4 미세조정 / C: 클래스 가중치 / D: 노이즈 증강)")
    p.add_argument("--images-dir", default=None, help="이미지 루트 (SM_CHANNEL_IMAGES 가 있으면 무시)")
    p.add_argument("--manifests-dir", default=None, help="split_manifest.csv 폴더 (SM_CHANNEL_MANIFESTS 우선)")
    p.add_argument("--weights-dir", default=None, help="resnet18_imagenet.pt 폴더 (SM_CHANNEL_WEIGHTS 우선)")
    p.add_argument("--model-dir", default=None, help="모델 저장 폴더 (SM_MODEL_DIR 우선)")
    p.add_argument("--output-dir", default=None, help="기타 결과 폴더 (SM_OUTPUT_DATA_DIR 우선)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.001, help="분류층(fc) 학습률")
    p.add_argument("--train-mode", choices=TRAIN_MODES, default="frozen",
                   help="frozen=실험 A(fc만 학습, 기본), layer4=실험 B(layer4+fc 학습)")
    p.add_argument("--lr-finetune", type=float, default=0.0001, help="layer4 학습률 (--train-mode layer4 일 때만 사용)")
    p.add_argument("--input-size", type=int, default=224,
                   help="입력 한 변의 픽셀 수 (기본 224=기존 실험 전부). 448 이면 마지막 특징 지도가 7×7 이 아니라 14×14 가 된다. P0 에서만 사용")
    p.add_argument("--preprocess", default="P0",
                   help="입력 전처리: P0=원본(기본), P1=HE, P2=CLAHE, P3=LPF 가우시안, P4=HPF 언샤프 (train·val 모두 적용)")
    p.add_argument("--augment", choices=AUGMENT_CHOICES, default="none",
                   help="학습 증강: none=없음(기본, A·B), noise=회색 가우시안 노이즈(실험 D, 학습에만)")
    p.add_argument("--noise-std-max", type=float, default=10.0,
                   help="노이즈 세기 상한 (0~255 기준). 사진마다 std ~ Uniform(0, 이 값)")
    p.add_argument("--noise-prob", type=float, default=0.5, help="학습 사진 한 장에 노이즈를 넣을 확률")
    p.add_argument("--equip-prob", type=float, default=0.5, help="--augment equip 일 때 사진 한 장에 필터를 걸 확률 (필터 종류는 5종 중 하나를 고르게)")
    p.add_argument("--lung-masks", default=None, help="폐 껍질 마스크 묶음(npz). 기본: weights 폴더의 lung_hull_masks.npz")
    p.add_argument("--class-weight", choices=CLASS_WEIGHT_CHOICES, default="none",
                   help="학습 손실 클래스 가중치: none=없음(기본, A·B·D), inverse=train 빈도의 역수(실험 C)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--experiment-id", default="expA")
    p.add_argument("--max-train-samples", type=int, default=None, help="시운전용: train 영상 수 제한")
    p.add_argument("--max-val-samples", type=int, default=None, help="시운전용: val 영상 수 제한")
    # SageMaker 가 --model_dir 같은 인자를 추가로 넘길 수 있어서 모르는 인자는 경고만 한다.
    args, unknown = p.parse_known_args(argv)
    if unknown:
        print(f"[경고] 알 수 없는 인자를 무시합니다: {unknown}", flush=True)
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs 와 --batch-size 는 1 이상이어야 합니다.")
    if not 0.0 <= args.equip_prob <= 1.0:
        raise SystemExit("--equip-prob 는 0~1 사이여야 합니다.")
    if args.augment == "equip" and args.input_size not in ED.SIGMA0:
        raise SystemExit(f"--augment equip 은 입력 크기 {sorted(ED.SIGMA0)} 에서만 지원합니다 (선량 기준 노이즈가 크기마다 정해져 있음).")
    if args.noise_std_max < 0 or not 0.0 <= args.noise_prob <= 1.0:
        raise SystemExit("--noise-std-max 는 0 이상, --noise-prob 는 0~1 사이여야 합니다.")
    if args.preprocess not in PREPROCESS_CHOICES:
        raise SystemExit(f"--preprocess 는 {', '.join(PREPROCESS_CHOICES)} 중 하나여야 합니다 (받은 값: {args.preprocess}). "
                         "P5(3채널 조합)는 이번 범위에서 구현하지 않았습니다.")
    if args.input_size < 64:
        raise SystemExit("--input-size 는 64 이상이어야 합니다.")
    if args.input_size != 224 and args.preprocess != "P0":
        raise SystemExit("--input-size 를 바꾸는 것은 P0(원본) 에서만 지원합니다 (P1~P4 의 필터 강도가 224 기준이라서).")
    if args.preprocess != "P0" and args.augment != "none":
        raise SystemExit("전처리 P1~P4 와 증강(--augment noise)을 함께 쓰는 것은 지원하지 않습니다 (P 시리즈는 증강 없음).")
    return args


# ----------------------------------------------------------------------------
# 재현성
# ----------------------------------------------------------------------------
def set_seed(seed):
    """random, numpy, torch 시드를 고정한다.

    주의: GPU 연산이나 DataLoader 병렬 처리 때문에 완전히 똑같은 결과까지는 보장하지 않는다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_worker_init_fn(seed):
    def worker_init(worker_id):
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
    return worker_init


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# manifest 읽기
# ----------------------------------------------------------------------------
def read_split_manifest(csv_path):
    """split_manifest.csv 에서 train/val 행만 읽는다. test/check 행은 여기서 버린다."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = {"relative_path", "label_id", "label_name", "split", "patient_group"}
        missing = need - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"split_manifest.csv 에 필요한 열이 없습니다: {sorted(missing)}")
        rows = [r for r in reader if r["split"] in ALLOWED_SPLITS]  # test/check 는 여기서 걸러진다

    for r in rows:
        if r["split"] not in ALLOWED_SPLITS:  # 이중 확인
            raise SystemExit(f"허용되지 않은 split 값입니다: {r['split']} ({r['relative_path']})")
        label_id = int(r["label_id"])
        if label_id not in range(NUM_CLASSES) or CLASS_NAMES[label_id] != r["label_name"]:
            raise SystemExit(
                f"label_id 와 label_name 이 정해진 순서 {CLASS_NAMES} 와 맞지 않습니다: "
                f"{r['label_id']}, {r['label_name']} ({r['relative_path']})"
            )
        r["label_id"] = label_id
    return rows


def split_rows(rows):
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    if not train or not val:
        raise SystemExit(f"train({len(train)}장) 또는 val({len(val)}장) 이 비어 있습니다.")
    return train, val


def limit_rows_balanced(rows, max_samples, seed):
    """시운전용: 클래스별로 균등하게(부족하면 있는 만큼) 결정적으로 뽑는다. None 이면 전체."""
    if max_samples is None or max_samples >= len(rows):
        return rows
    if max_samples < NUM_CLASSES:
        raise SystemExit("--max-*-samples 는 3 이상이어야 세 클래스가 모두 들어갑니다.")
    rng = random.Random(seed)
    by_class = {c: [r for r in rows if r["label_id"] == c] for c in range(NUM_CLASSES)}
    # 클래스당 몫을 같게 나누고, 나머지는 앞 클래스부터 하나씩 더 준다. 부족한 클래스의 남는 몫은 다른 클래스가 가져간다.
    quota = {c: max_samples // NUM_CLASSES + (1 if c < max_samples % NUM_CLASSES else 0) for c in range(NUM_CLASSES)}
    take = {c: min(quota[c], len(by_class[c])) for c in range(NUM_CLASSES)}
    leftover = max_samples - sum(take.values())
    for c in range(NUM_CLASSES):
        extra = min(leftover, len(by_class[c]) - take[c])
        take[c] += extra
        leftover -= extra
    chosen = []
    for c in range(NUM_CLASSES):
        chosen += rng.sample(by_class[c], take[c])
    chosen.sort(key=lambda r: r["relative_path"])  # 순서도 결정적으로
    return chosen


def class_counts(rows):
    return {CLASS_NAMES[c]: sum(1 for r in rows if r["label_id"] == c) for c in range(NUM_CLASSES)}


def compute_class_weights(train_rows):
    """실험 C: train 영상의 클래스별 수 n_c 로 w_c = N / (K * n_c) 를 계산한다.

    비유: 시험 문제 중 세균성 문제가 두 배 많으면, 세균성 한 문제의 배점을 절반으로 줄여서
    세 클래스가 손실 전체에 같은 몫을 갖게 한다. 클래스가 고르게 있으면 세 값이 모두 1 이 된다.
    반드시 학습에 실제로 쓰는 train 행만 넘긴다 (val 을 넣으면 검증 정보가 학습에 새어 들어간다).
    반환: (가중치 3개 list, 클래스별 수 3개 list)
    """
    counts = [sum(1 for r in train_rows if r["label_id"] == c) for c in range(NUM_CLASSES)]
    if min(counts) == 0:
        raise SystemExit(f"train 에 영상이 0장인 클래스가 있어 가중치를 계산할 수 없습니다: {counts}")
    n_total = sum(counts)
    weights = [n_total / (NUM_CLASSES * n) for n in counts]
    return weights, counts


# ----------------------------------------------------------------------------
# Dataset / 전처리
# ----------------------------------------------------------------------------
def build_transform(preprocess="P0", size=224):
    """입력 전처리 (train·val 공통, 결정적). 입력은 Dataset 이 convert("RGB") 한 PIL 영상.

    P0: 기존 경로 그대로 (A·B·C·D 와 같음). 이 분기는 바꾸지 않는다.
    P1~P4: 회색조(L) → PIL Resize (224,224) BILINEAR → 전처리 → 3채널 복제 → ToTensor → Normalize.
           원본 크기가 384~2916 으로 제각각이라 224 로 줄인 뒤 적용해야 σ·CLAHE 타일의 의미가
           모든 사진에서 같아진다. 정의는 preprocess.py 한 곳에만 있다.
    """
    if preprocess not in PREPROCESS_CHOICES:
        raise ValueError(f"알 수 없는 전처리: {preprocess} (가능: {', '.join(PREPROCESS_CHOICES)})")
    if preprocess == "P0":
        return transforms.Compose([
            transforms.Resize((size, size)),  # 자르기 없이 전체 영상을 줄임 (종횡비는 바뀜)
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        PP.GrayPreprocess(preprocess),  # 224 회색조 → 전처리 → 3채널 PIL 영상
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class RandomGrayNoise:
    """학습용 노이즈 증강 (실험 D). ToTensor 직후(0~1 범위, Normalize 전)에 쓴다.

    비유: 같은 사진을 저선량 장비로 다시 찍은 것처럼 모래알 노이즈를 뿌려서,
    모델이 "노이즈 양" 이 아니라 폐의 모양을 보도록 연습시킨다.
    - 확률 prob 로만 넣는다 (나머지 사진은 원본 그대로).
    - 세기 std 는 사진마다 Uniform(0, std_max) 에서 뽑고 0~255 기준이라 /255 로 바꾼다.
    - 입력이 회색 사진을 3채널로 복제한 것이므로 노이즈도 (1,H,W) 한 장을 세 채널에 똑같이 더한다.
    - 난수는 torch 기본 생성기를 쓴다. DataLoader 가 워커마다, 에폭마다 다른 seed 를 주므로 매번 다르게 뽑힌다.
    """

    def __init__(self, std_max, prob):
        self.std_max = float(std_max)
        self.prob = float(prob)

    def __call__(self, x):
        if torch.rand(()).item() >= self.prob:
            return x
        std = torch.rand(()).item() * self.std_max / 255.0
        noise = torch.randn(1, x.shape[1], x.shape[2]) * std
        return (x + noise).clamp_(0.0, 1.0)


EQUIP_KINDS = ["dose", "motion", "scatter", "kvp", "sensor"]
# u=1 이 시험의 "강"과 같은 세기 (reports/equipment_v2/strength_fixed_512.md): 선량 1/64, 번짐 8mm, 산란 k 1.0, kVp (0.60, 0.5), 센서 2.0mm
EQUIP_MAX = {"dose": "상대 선량 2^(-6u) (u=1 → 1/64)", "motion": "번짐 8u mm", "scatter": "산란 k = u",
             "kvp": "대비 1-0.40u, 밝은 쪽 기울기 1-0.50u", "sensor": "화소 2u mm"}


class RandomEquipment:
    """학습용 장치 설정 필터 증강 (--augment equip). Resize 직후의 PIL 영상(회색을 3채널로 복제한 것)에 쓴다.

    확률 prob 로 5종(선량·조사시간·산란·kVp·센서) 중 하나를 고르게 뽑아 u ~ Uniform(0, 1) 세기로 건다 (u=0 원본, u=1 시험의 "강").
    필터 구현은 시험과 같은 training/equipment_direct.py 의 함수를 그대로 쓴다. 난수는 torch 기본 생성기를 쓰므로
    DataLoader 가 워커마다, 에폭마다 다른 seed 를 주어 매번 다르게 뽑힌다 (RandomGrayNoise 와 같은 방식).
    """

    def __init__(self, prob, size):
        self.prob = float(prob)
        self.sigma0 = ED.SIGMA0[size]

    def apply(self, img, kind, u, seed):
        g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        if kind == "dose":
            out = ED.dose(g, 2.0 ** (-6.0 * u), seed, self.sigma0)
        elif kind == "motion":
            out = ED.motion(g, 8.0 * u)
        elif kind == "scatter":
            out = ED.scatter(g, 1.0 * u)
        elif kind == "kvp":
            out = ED.kvp(g, 1.0 - 0.40 * u, 1.0 - 0.50 * u)
        else:
            out = ED.sensor(g, 2.0 * u)
        arr = np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    def __call__(self, img):
        if torch.rand(()).item() >= self.prob:
            return img
        kind = EQUIP_KINDS[int(torch.randint(0, len(EQUIP_KINDS), ()).item())]
        u = torch.rand(()).item()
        seed = int(torch.randint(0, 2 ** 31 - 1, ()).item())
        return self.apply(img, kind, u, seed)


def build_train_transform(preprocess="P0", augment="none", noise_std_max=10.0, noise_prob=0.5, size=224):
    """학습(train) 전용 변환. augment=none 이면 build_transform 과 똑같다."""
    if augment == "none":
        return build_transform(preprocess, size)
    if preprocess != "P0":
        raise ValueError(f"전처리 {preprocess} 와 노이즈 증강은 함께 쓸 수 없습니다 (노이즈 증강은 P0 에서만)")
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        RandomGrayNoise(noise_std_max, noise_prob),  # 0~1 범위에서 더하고 자른 뒤 정규화
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class TrainValDataset(Dataset):
    """train 또는 val 행만 받는 Dataset. 영상을 여는 유일한 곳이다."""

    def __init__(self, rows, images_root, transform):
        bad = [r["relative_path"] for r in rows if r["split"] not in ALLOWED_SPLITS]
        if bad:
            raise SystemExit(f"train/val 이 아닌 행이 Dataset 에 들어왔습니다: {bad[:3]}")
        self.rows = rows
        self.root = Path(images_root)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = self.root / row["relative_path"]
        try:
            with Image.open(path) as im:
                # 회색(1채널) 영상은 3채널로 복제, 원래 3채널 영상도 같은 방식으로 통일
                img = im.convert("RGB")
        except Exception as e:  # 조용히 건너뛰지 않고 어느 파일인지 알려 준다
            raise RuntimeError(f"영상을 읽지 못했습니다: {path} ({type(e).__name__}: {e})") from e
        return self.transform(img), row["label_id"], idx


def load_hulls(path, size):
    """lung_hull_masks.npz → {relative_path: 비트로 묶은 껍질 마스크}. 쓸 때 풀어서 쓴다 (메모리 절약)."""
    z = np.load(path)
    masks, paths = z["masks"], z["paths"]  # npz 는 접근할 때마다 통째로 풀리므로 한 번만 꺼낸다
    if masks.shape[1] != size * size // 8:
        raise SystemExit(f"마스크 크기가 입력 크기 {size} 와 다릅니다: {masks.shape}")
    return {str(p): masks[i] for i, p in enumerate(paths)}


class LungDataset(Dataset):
    """train/val 행만 받는 Dataset (폐 영역 판). 원본 → 회색 S×S(평가와 같은 to_gray) → [학습만 장치 필터] → 폐 껍질 밖 0 → RGB 3채널 → ToTensor → Normalize."""

    def __init__(self, rows, images_root, hulls, equip, size):
        bad = [r["relative_path"] for r in rows if r["split"] not in ALLOWED_SPLITS]
        if bad:
            raise SystemExit(f"train/val 이 아닌 행이 Dataset 에 들어왔습니다: {bad[:3]}")
        self.rows, self.root, self.hulls, self.equip, self.size = rows, Path(images_root), hulls, equip, size
        self.tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = self.root / row["relative_path"]
        try:
            with Image.open(path) as im:
                gray = _to_gray(im, self.size)
        except Exception as e:
            raise RuntimeError(f"영상을 읽지 못했습니다: {path} ({type(e).__name__}: {e})") from e
        img = Image.fromarray(gray).convert("RGB")
        if self.equip is not None:
            img = self.equip(img)  # 확률 prob 로 5종 중 하나 (회색 복제 → 필터 → 회색 복제)
        g = np.asarray(img.convert("L"))
        hull = np.unpackbits(self.hulls[row["relative_path"]])[: self.size * self.size].reshape(self.size, self.size).astype(bool)
        out = np.where(hull, g, 0).astype(np.uint8)
        return self.tf(Image.fromarray(out).convert("RGB")), row["label_id"], idx


# ----------------------------------------------------------------------------
# 모델
# ----------------------------------------------------------------------------
def build_model(weights_path, train_mode="frozen"):
    """ImageNet 사전학습 ResNet18 에 새 fc(3 출력)를 붙인다.

    frozen (실험 A): 몸통 전체를 얼리고 새 fc 만 학습한다.
    layer4 (실험 B): conv1~layer3 은 얼리고, 마지막 잔차 블록 layer4 와 fc 만 학습한다.
    """
    model = models.resnet18(weights=None)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # 새로 만든 층은 requires_grad=True
    if train_mode == "layer4":
        for p in model.layer4.parameters():
            p.requires_grad = True
    return model


def build_optimizer(model, train_mode, lr, lr_finetune):
    """Adam 하나. layer4 모드는 파라미터 그룹을 2개로 나눠 학습률을 따로 준다.

    비유: 새로 붙인 fc 는 백지라 크게(lr) 배우고, 이미 잘 배운 layer4 는 조금만(lr_finetune) 고친다.
    """
    if train_mode == "frozen":
        return torch.optim.Adam(model.fc.parameters(), lr=lr)
    return torch.optim.Adam([
        {"params": model.fc.parameters(), "lr": lr},
        {"params": model.layer4.parameters(), "lr": lr_finetune},
    ])


def set_train_bn_mode(model, train_mode):
    """학습 중 BatchNorm 상태를 정한다.

    BatchNorm 은 train() 상태에서 배치마다 이동 평균(running mean/var)을 갱신한다.
    - frozen: 모델 전체를 eval() 로 둔다. 몸통 BN 이동 평균이 ImageNet 값 그대로 남는다.
    - layer4: 먼저 전체 eval() 로 고정된 층(conv1~layer3)의 BN 을 멈추고, layer4 만 train() 으로 켠다.
      그래야 얼려 둔 층은 "ImageNet 그대로" 이고, 학습하는 layer4 의 BN 만 X선 영상에 맞게 갱신된다.
    fc 에는 dropout/BN 이 없어서 eval() 이어도 학습에 지장 없다.
    """
    model.eval()
    if train_mode == "layer4":
        model.layer4.train()


# ----------------------------------------------------------------------------
# 지표 (numpy 직접 구현, scikit-learn 사용 안 함)
# ----------------------------------------------------------------------------
def confusion_counts(y_true, y_pred):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm  # 행: 실제, 열: 예측


def macro_f1_and_recall(y_true, y_pred):
    """Macro-F1(3클래스 평균)과 클래스별 recall. 분모가 0 이면 F1=0 (zero_division=0 과 동일).

    반환: (macro_f1, recalls[3], warnings[list])
    recall 은 val 에 그 클래스가 없으면 NaN.
    """
    cm = confusion_counts(y_true, y_pred)
    warns, f1s, recalls = [], [], []
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        n_true = cm[c, :].sum()
        n_pred = cm[:, c].sum()
        if n_pred == 0:
            warns.append(f"{CLASS_NAMES[c]}: 예측된 적 없음(정밀도 분모 0, F1=0 처리)")
        if n_true == 0:
            warns.append(f"{CLASS_NAMES[c]}: val 에 없음(recall 계산 불가)")
        recalls.append(tp / n_true if n_true > 0 else float("nan"))
        denom = 2 * tp + (n_pred - tp) + (n_true - tp)  # F1 = 2TP / (2TP + FP + FN)
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s)), recalls, warns


def binary_auroc(y_bin, score):
    """AUROC = 순위합(Mann-Whitney U) 방식. 동점은 평균 순위. 한쪽 클래스가 없으면 NaN."""
    y_bin = np.asarray(y_bin).astype(bool)
    score = np.asarray(score, dtype=np.float64)
    n_pos = int(y_bin.sum())
    n_neg = len(y_bin) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    avg_rank = ends - (counts - 1) / 2.0  # 1부터 시작하는 평균 순위
    ranks = avg_rank[inverse]
    return float((ranks[y_bin].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ovr_auroc(y_true, probs):
    """클래스별 One-vs-Rest AUROC, Macro(계산 가능한 클래스만 평균), 경고 목록."""
    y_true = np.asarray(y_true)
    aurocs, warns = [], []
    for c in range(NUM_CLASSES):
        a = binary_auroc(y_true == c, probs[:, c])
        if math.isnan(a):
            warns.append(f"{CLASS_NAMES[c]}: val 에 없어서 AUROC 계산 불가(NaN, Macro 평균에서 제외)")
        aurocs.append(a)
    valid = [a for a in aurocs if not math.isnan(a)]
    macro = float(np.mean(valid)) if valid else float("nan")
    return macro, aurocs, warns


# ----------------------------------------------------------------------------
# 학습 / 검증 루프
# ----------------------------------------------------------------------------
def check_finite(value, what, epoch):
    if not math.isfinite(value):
        raise SystemExit(f"[중단] 에폭 {epoch}: {what} 가 {value} 입니다. 학습률이나 데이터를 확인하세요.")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs, train_mode):
    # 얼린 층의 BatchNorm 이동 평균이 바뀌지 않도록 모드에 맞춰 eval()/train() 을 정한다 (set_train_bn_mode 참고).
    # 검증(evaluate) 때는 다시 전체 eval() 로 바뀐다.
    set_train_bn_mode(model, train_mode)
    total_loss, n_seen = 0.0, 0
    n_batches = len(loader)
    for b, (x, y, _) in enumerate(loader, start=1):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        check_finite(loss.item(), "train loss", epoch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n_seen += x.size(0)
        if b % 20 == 0 or b == n_batches:
            print(f"  [에폭 {epoch}/{epochs}] 배치 {b}/{n_batches} train_loss(누적 평균)={total_loss / n_seen:.4f}", flush=True)
    return total_loss / n_seen


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss, n_seen = 0.0, 0
    all_logits, all_labels, all_idx = [], [], []
    for x, y, idx in loader:
        logits = model(x.to(device))
        loss = criterion(logits, y.to(device))
        check_finite(loss.item(), "val loss", epoch)
        total_loss += loss.item() * x.size(0)
        n_seen += x.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(y)
        all_idx.append(idx)
    logits = torch.cat(all_logits)
    probs = torch.softmax(logits, dim=1).numpy().astype(np.float64)
    y_true = torch.cat(all_labels).numpy()
    idx = torch.cat(all_idx).numpy()
    y_pred = probs.argmax(axis=1)
    macro_f1, recalls, warns_f1 = macro_f1_and_recall(y_true, y_pred)
    macro_auc, aurocs, warns_auc = ovr_auroc(y_true, probs)
    return {
        "val_loss": total_loss / n_seen,
        "val_macro_f1": macro_f1,
        "val_macro_auroc": macro_auc,
        "recalls": recalls,
        "aurocs": aurocs,
        "warnings": warns_f1 + warns_auc,
        "y_true": y_true, "y_pred": y_pred, "probs": probs, "idx": idx,
    }


def is_better(f1, loss, best_f1, best_loss):
    """베스트 선택 규칙: F1 이 더 높으면 갱신, F1 이 같으면 val_loss 가 더 낮을 때만 갱신.
    둘 다 같으면 갱신하지 않으므로 더 이른 에폭이 남는다."""
    if best_f1 is None:
        return True
    if f1 > best_f1:
        return True
    return f1 == best_f1 and loss < best_loss


# ----------------------------------------------------------------------------
# 저장
# ----------------------------------------------------------------------------
def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def fmt(v):
    return f"{v:.6f}"


def write_history(path, history):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HISTORY_COLUMNS)
        for h in history:
            w.writerow([
                h["epoch"], fmt(h["train_loss"]), fmt(h["val_loss"]), fmt(h["val_macro_f1"]), fmt(h["val_macro_auroc"]),
                *[fmt(v) for v in h["recalls"]], *[fmt(v) for v in h["aurocs"]],
                f"{h['elapsed_seconds']:.1f}", h["is_best"], " | ".join(h["warnings"]),
            ])


def write_val_predictions(path, val_rows, best):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relative_path", "patient_group", "true_label", "pred_label", "p_normal", "p_bacterial", "p_viral"])
        for k, idx in enumerate(best["idx"]):
            r = val_rows[int(idx)]
            w.writerow([
                r["relative_path"], r["patient_group"], int(best["y_true"][k]), int(best["y_pred"][k]),
                *[f"{p:.8f}" for p in best["probs"][k]],
            ])


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(argv=None):
    t_start = time.time()
    args = parse_args(argv)
    images_dir, images_src = resolve_path("SM_CHANNEL_IMAGES", args.images_dir, "--images-dir")
    manifests_dir, manifests_src = resolve_path("SM_CHANNEL_MANIFESTS", args.manifests_dir, "--manifests-dir")
    weights_dir, weights_src = resolve_path("SM_CHANNEL_WEIGHTS", args.weights_dir, "--weights-dir")
    model_dir, model_src = resolve_path("SM_MODEL_DIR", args.model_dir, "--model-dir")
    output_dir, output_src = resolve_path("SM_OUTPUT_DATA_DIR", args.output_dir, "--output-dir")
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    # 장치: cuda 가 있으면 cuda, 아니면 cpu. (MPS 는 쓰지 않는다)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "cpu"
    print(f"torch {torch.__version__}, torchvision {torchvision.__version__}, python {platform.python_version()}", flush=True)
    print(f"cuda 사용 가능: {torch.cuda.is_available()}, 장치: {device.type} ({device_name})", flush=True)

    # 데이터 (train/val 행만)
    manifest_path = manifests_dir / "split_manifest.csv"
    weights_path = weights_dir / "resnet18_imagenet.pt"
    for p in (manifest_path, weights_path):
        if not p.is_file():
            raise SystemExit(f"파일이 없습니다: {p}")
    train_all, val_all = split_rows(read_split_manifest(manifest_path))
    train_rows = limit_rows_balanced(train_all, args.max_train_samples, args.seed)
    val_rows = limit_rows_balanced(val_all, args.max_val_samples, args.seed)
    train_counts, val_counts = class_counts(train_rows), class_counts(val_rows)
    print(f"train {len(train_rows)}장 {train_counts}, val {len(val_rows)}장 {val_counts}", flush=True)

    preprocess_record = PP.describe(args.preprocess)  # 이름·파라미터·처리 순서 (서비스가 똑같이 적용할 수 있게 기록)
    if args.input_size != 224:  # P0 설명 문자열의 224 를 실제 입력 크기로 바꿔 기록
        preprocess_record = {k: (v.replace("224", str(args.input_size)) if isinstance(v, str) else v)
                             for k, v in preprocess_record.items()}
    print(f"전처리: {preprocess_record['preprocess_desc']} (train·val 모두 적용, 순서: {preprocess_record['preprocess_order']})",
          flush=True)
    transform = build_transform(args.preprocess, args.input_size)  # val 용 (선택한 전처리만, 증강 없음)
    if args.augment == "equip":
        train_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)), RandomEquipment(args.equip_prob, args.input_size),
            transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    else:
        train_transform = build_train_transform(args.preprocess, args.augment, args.noise_std_max, args.noise_prob, args.input_size)
    # 증강을 쓰지 않을 때는 기록에 None 을 남긴다 (lr_finetune 과 같은 방식)
    noise_std_max = args.noise_std_max if args.augment == "noise" else None
    noise_prob = args.noise_prob if args.augment == "noise" else None
    if args.augment == "noise":
        augment_desc = f"noise (p={args.noise_prob:g}, std 0~{args.noise_std_max:g})"
        augmentation_text = f"회색 가우시안 노이즈 {augment_desc}, 학습에만 (좌우 반전 금지)"
    elif args.augment == "equip":
        augment_desc = f"equip (p={args.equip_prob:g}, 5종 중 하나, 세기 0~강)"
        augmentation_text = f"장치 설정 필터 {augment_desc}: " + ", ".join(EQUIP_MAX.values()) + ", 학습에만 (좌우 반전 금지)"
    else:
        augment_desc = "없음"
        augmentation_text = "없음 (좌우 반전 금지)"
    print(f"증강: {augment_desc} ({AUGMENT_NOTE})", flush=True)
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.num_workers,
        worker_init_fn=make_worker_init_fn(args.seed), pin_memory=(device.type == "cuda"),
    )
    if args.augment not in ("none", "equip"):
        raise SystemExit("train_lung.py 는 --augment none / equip 만 지원합니다.")
    if args.input_size != 512:
        raise SystemExit("폐 마스크가 512 기준이라 --input-size 512 만 지원합니다.")
    lung_path = Path(args.lung_masks) if args.lung_masks else weights_dir / "lung_hull_masks.npz"
    hulls = load_hulls(lung_path, args.input_size)
    missing = [r["relative_path"] for r in train_rows + val_rows if r["relative_path"] not in hulls]
    if missing:
        raise SystemExit(f"폐 마스크가 없는 사진이 {len(missing)}장 있습니다: {missing[:3]}")
    full = sum(1 for r in train_rows + val_rows if np.unpackbits(hulls[r["relative_path"]]).all())
    print(f"폐 영역 입력: 껍질 마스크 {len(hulls)}장 읽음 ({lung_path.name}), 분할 실패로 전체를 남기는 사진 {full}장, 여백 {LP.DILATE_PX}px", flush=True)
    equip = RandomEquipment(args.equip_prob, args.input_size) if args.augment == "equip" else None
    train_loader = DataLoader(LungDataset(train_rows, images_dir, hulls, equip, args.input_size), shuffle=True, generator=g, **loader_kwargs)
    val_loader = DataLoader(LungDataset(val_rows, images_dir, hulls, None, args.input_size), shuffle=False, **loader_kwargs)

    model = build_model(weights_path, args.train_mode).to(device)
    # val_loss 용 손실: 항상 가중치 없음. 실험 C 에서도 val_loss 를 A·B·D 와 같은 잣대로 비교하기 위해서다.
    # (가중치를 넣으면 손실 값의 기준 자체가 바뀌어 실험끼리 숫자를 나란히 놓을 수 없다.)
    # 입력: softmax 이전 점수 + 정수 라벨
    criterion = nn.CrossEntropyLoss()
    if args.class_weight == "inverse":
        class_weight_values, class_weight_counts = compute_class_weights(train_rows)  # train 행만 (val 사용 안 함)
        train_criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight_values, dtype=torch.float32, device=device))
        loss_desc = "CrossEntropyLoss (클래스 가중치 inverse, 학습 손실에만; val_loss 는 가중치 없음)"
        print(f"클래스 가중치: inverse [{', '.join(f'{w:.4f}' for w in class_weight_values)}] "
              f"(train 클래스별 수 {class_weight_counts}, {CLASS_NAMES} 순)", flush=True)
    else:
        class_weight_values, class_weight_counts = None, None
        train_criterion = criterion  # A·B·D: 학습과 검증이 같은 손실
        loss_desc = "CrossEntropyLoss (클래스 가중치 없음)"
        print("클래스 가중치: none (없음)", flush=True)
    class_weight_record = {
        "class_weight": args.class_weight, "class_weight_values": class_weight_values,
        "class_weight_counts": class_weight_counts,
        "class_weight_note": CLASS_WEIGHT_NOTE if args.class_weight == "inverse" else None,
    }
    optimizer = build_optimizer(model, args.train_mode, args.lr, args.lr_finetune)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lr_finetune = args.lr_finetune if args.train_mode == "layer4" else None  # frozen 에서는 쓰지 않음
    print(f"학습 모드: {args.train_mode}, 학습 파라미터 {n_trainable}개 ({TRAINABLE_DESC[args.train_mode]})", flush=True)
    print(f"학습률: fc={args.lr}" + (f", layer4={lr_finetune}" if lr_finetune is not None else ""), flush=True)

    config = {
        "experiment_id": args.experiment_id, **preprocess_record, "model": "resnet18",
        "train_mode": args.train_mode, "trainable": TRAINABLE_DESC[args.train_mode], "trainable_params": n_trainable,
        "optimizer": OPTIMIZER_DESC[args.train_mode], "loss": loss_desc, **class_weight_record,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "lr_finetune": lr_finetune, "seed": args.seed,
        "num_workers": args.num_workers, "augmentation": augmentation_text,
        "augment": args.augment, "noise_std_max": noise_std_max, "noise_prob": noise_prob,
        "equip_prob": args.equip_prob if args.augment == "equip" else None,
        "lung_prep": {"masks_file": lung_path.name, "masks_sha256": sha256_of_file(lung_path), "dilate_px": LP.DILATE_PX, "min_area": LP.MIN_AREA,
                      "order": "회색 512 (평가와 같은 to_gray) → (학습만 장치 필터) → 폐 볼록 껍질 밖을 0으로 → RGB → Normalize"},
        "equip_kinds": EQUIP_KINDS if args.augment == "equip" else None, "equip_max": EQUIP_MAX if args.augment == "equip" else None,
        "max_train_samples": args.max_train_samples, "max_val_samples": args.max_val_samples,
        "path_sources": {  # 경로 값은 적지 않고 어디서 왔는지만 기록
            "images": images_src, "manifests": manifests_src, "weights": weights_src,
            "model_dir": model_src, "output_dir": output_src,
        },
    }
    write_json(output_dir / "config.json", config)

    history, best, best_epoch = [], None, None
    best_f1, best_loss = None, None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, train_criterion, optimizer, device, epoch, args.epochs, args.train_mode)
        res = evaluate(model, val_loader, criterion, device, epoch)
        better = is_better(res["val_macro_f1"], res["val_loss"], best_f1, best_loss)
        if better:
            best, best_epoch = res, epoch
            best_f1, best_loss = res["val_macro_f1"], res["val_loss"]
            torch.save(model.state_dict(), model_dir / "best_model.pt")  # 베스트 에폭마다 덮어쓴다
        warns = [f"에폭 {epoch} {w}" for w in res["warnings"]]
        elapsed = time.time() - t_start
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": res["val_loss"],
            "val_macro_f1": res["val_macro_f1"], "val_macro_auroc": res["val_macro_auroc"],
            "recalls": res["recalls"], "aurocs": res["aurocs"],
            "elapsed_seconds": elapsed, "is_best": better, "warnings": warns,
        })
        write_history(output_dir / "history.csv", history)  # 매 에폭 갱신(중간에 멈춰도 기록이 남음)
        print(
            f"에폭 {epoch}/{args.epochs} ({100 * epoch / args.epochs:.0f}%) "
            f"train_loss={train_loss:.4f} val_loss={res['val_loss']:.4f} "
            f"val_macro_f1={res['val_macro_f1']:.4f} val_macro_auroc={res['val_macro_auroc']:.4f} "
            f"경과 {elapsed:.0f}초{' *best' if better else ''}",
            flush=True,
        )
        for w in warns:
            print(f"  [경고] {w}", flush=True)

    write_val_predictions(output_dir / "val_predictions.csv", val_rows, best)
    total_seconds = time.time() - t_start

    metadata = {
        "experiment_id": args.experiment_id,
        "class_names": CLASS_NAMES,
        "label_mapping": {str(i): n for i, n in enumerate(CLASS_NAMES)},
        "input_size": [args.input_size, args.input_size], "channels": 3,
        "resize": f"전체 영상을 Resize(({args.input_size},{args.input_size})), 자르기 없음, 종횡비 변경됨",
        "normalize": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        **preprocess_record, "model": "resnet18", "pretrained_weights": "torchvision IMAGENET1K_V1",
        "train_mode": args.train_mode, "lr_finetune": lr_finetune, "trainable_params": n_trainable,
        "augment": args.augment, "noise_std_max": noise_std_max, "noise_prob": noise_prob,
        "augment_note": AUGMENT_NOTE, **class_weight_record,
        "num_classes": NUM_CLASSES, "selected_epoch": best_epoch,
        "selection_rule": SELECTION_RULE, "best_val_macro_f1": best_f1,
    }
    write_json(model_dir / "metadata.json", metadata)

    summary = {
        "config": config, "seed": args.seed,
        "train_mode": args.train_mode, "lr_finetune": lr_finetune, "trainable_params": n_trainable,
        "augment": args.augment, "noise_std_max": noise_std_max, "noise_prob": noise_prob,
        **class_weight_record,
        "device": {"type": device.type, "name": device_name},
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__, "python": platform.python_version()},
        "selected_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "split_manifest_sha256": sha256_of_file(manifest_path),
        "resnet18_imagenet_pt_sha256": sha256_of_file(weights_path),
        "num_train_images": len(train_rows), "num_val_images": len(val_rows),
        "train_class_counts": train_counts, "val_class_counts": val_counts,
        "total_training_seconds": round(total_seconds, 1),
        "status": "Completed",  # 여기까지 오류 없이 왔을 때만 기록된다
    }
    write_json(output_dir / "experiment_summary.json", summary)
    print(f"완료: 베스트 에폭 {best_epoch}, val_macro_f1={best_f1:.4f}, 총 {total_seconds:.0f}초", flush=True)


if __name__ == "__main__":
    main()
