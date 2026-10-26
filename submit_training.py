"""SageMaker Training Job 제출 스크립트 (가이드 9장).

training/train.py 를 SageMaker 학습 컨테이너에서 실행하는 작업 1개를 제출한다.
설정은 aws_config.json 에서 읽는다 (비밀 정보가 있어서 git 에 올리지 않는다).

사용 예 (p1 폴더에서):
    # 스모크 작업: 작은 샘플(160/60장) 2에폭으로 환경만 확인
    .venv/bin/python submit_training.py --experiment-id smoke --smoke --wait

    # 실험 A 본 학습 (30에폭)
    .venv/bin/python submit_training.py --experiment-id A --wait

    # train.py 에 옵션이 새로 생기면 --extra KEY=VALUE 로 hyperparameters 에 추가
    .venv/bin/python submit_training.py --experiment-id B --extra train-mode=layer4

결과:
    runs/submissions/<작업이름>.json   제출 기록 (submission.json). --wait 이면 최종 상태까지 기록
    S3 output_s3 아래                  model.tar.gz, output.tar.gz (SageMaker 가 저장)
"""
import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "aws_config.json"
TRAIN_PY = ROOT / "training" / "train.py"
SPLIT_MANIFEST = ROOT / "manifests" / "split_manifest.csv"
SUBMISSION_DIR = ROOT / "runs" / "submissions"
KST = timezone(timedelta(hours=9))
POLL_SECONDS = 30
TERMINAL_STATUSES = {"Completed", "Failed", "Stopped"}

# train.py 가 출력하는 에폭 로그 한 줄에서 숫자를 뽑는 정규식. 예:
# 에폭 1/2 (50%) train_loss=0.9082 val_loss=0.8189 val_macro_f1=0.7002 val_macro_auroc=0.8724 경과 5초
# 배치 로그의 "train_loss(누적 평균)=" 는 '=' 앞에 괄호가 있어서 아래 패턴과 겹치지 않는다.
METRIC_DEFINITIONS = [
    {"Name": "train_loss", "Regex": r"train_loss=([0-9.]+) val_loss="},
    {"Name": "val_loss", "Regex": r"val_loss=([0-9.]+) val_macro_f1="},
    {"Name": "val_macro_f1", "Regex": r"val_macro_f1=([0-9.]+) val_macro_auroc="},
    {"Name": "val_macro_auroc", "Regex": r"val_macro_auroc=([0-9.]+) 경과"},
]


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description="SageMaker Training Job 제출")
    p.add_argument("--experiment-id", required=True, help="실험 이름 (예: A, smoke). 작업 이름에 들어간다")
    p.add_argument("--smoke", action="store_true", help="스모크: 160/60장, 2에폭, 시간 제한 짧게")
    p.add_argument("--epochs", type=int, default=None, help="기본 30 (스모크는 2)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extra", action="append", default=[], metavar="KEY=VALUE",
                   help="train.py 에 그대로 넘길 추가 hyperparameter (여러 번 가능)")
    p.add_argument("--instance-type", default=None, help="학습 인스턴스 종류 (기본: aws_config.json 의 training_instance). 예: ml.g4dn.2xlarge")
    p.add_argument("--entry-point", default="train.py", help="training/ 폴더 안의 학습 스크립트 (기본 train.py, 과적합 줄이기 옵션은 train_reg.py)")
    p.add_argument("--wait", action="store_true", help="완료될 때까지 30초마다 상태를 확인")
    p.add_argument("--images-uri", default=None, help="images 채널의 S3 주소 (기본: aws_config.json 의 source_s3). 폐 영역 사진처럼 따로 만든 사진을 쓸 때")
    p.add_argument("--manifests-uri", default=None, help="manifests 채널의 S3 주소 (기본: aws_config.json 의 manifest_s3)")
    p.add_argument("--manifest-file", default=None, help="manifests 채널에 올린 split_manifest.csv 의 로컬 경로 (제출 기록의 해시용, 기본 manifests/split_manifest.csv)")
    return p.parse_args()


def make_job_name(student_id, experiment_id):
    """작업 이름 규칙: 소문자·숫자·하이픈, 63자 이내."""
    exp = re.sub(r"[^a-z0-9-]", "-", experiment_id.lower()).strip("-")
    if not exp:
        raise SystemExit("--experiment-id 에 영문·숫자가 하나는 있어야 합니다.")
    stamp = datetime.now(KST).strftime("%Y%m%d-%H%M%S")
    name = f"p1-cxr-{student_id}-{exp}-{stamp}"
    if len(name) > 63:
        raise SystemExit(f"작업 이름이 63자를 넘습니다({len(name)}자): --experiment-id 를 줄여 주세요.")
    return name


def build_hyperparameters(args):
    hp = {
        "epochs": args.epochs if args.epochs is not None else (2 if args.smoke else 30),
        "batch-size": args.batch_size,
        "lr": args.lr,
        "num-workers": args.num_workers,
        "seed": args.seed,
        "experiment-id": args.experiment_id,
    }
    if args.smoke:
        hp["max-train-samples"] = 160
        hp["max-val-samples"] = 60
    for item in args.extra:
        if "=" not in item:
            raise SystemExit(f"--extra 는 KEY=VALUE 형태여야 합니다: {item}")
        key, value = item.split("=", 1)
        hp[key] = value
    return hp


def write_submission(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def wait_for_job(sm_client, job_name):
    """끝날 때까지 30초마다 상태를 출력한다. 마지막 describe 결과를 돌려준다."""
    while True:
        desc = sm_client.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        print(f"[{datetime.now(KST):%H:%M:%S}] 상태: {status} / {desc.get('SecondaryStatus', '')}", flush=True)
        if status in TERMINAL_STATUSES:
            return desc
        time.sleep(POLL_SECONDS)


def main():
    args = parse_args()
    if not CONFIG_PATH.is_file():
        raise SystemExit(f"aws_config.json 이 없습니다: {CONFIG_PATH}")
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    train_py = ROOT / "training" / args.entry_point  # 기본은 TRAIN_PY(train.py)
    for p in (train_py, SPLIT_MANIFEST):
        if not p.is_file():
            raise SystemExit(f"파일이 없습니다: {p}")

    job_name = make_job_name(cfg["student_id"], args.experiment_id)
    hyperparameters = build_hyperparameters(args)
    max_run = cfg["max_run_seconds"]["smoke" if args.smoke else "full"]
    channels = {"images": args.images_uri or cfg["source_s3"], "manifests": args.manifests_uri or cfg["manifest_s3"], "weights": cfg["weights_s3"]}

    boto_session = boto3.Session(profile_name=cfg["profile"], region_name=cfg["region"])
    # 기본 버킷을 만들 권한이 없을 수 있어서 본인 학생 버킷을 직접 지정한다.
    sm_session = sagemaker.Session(boto_session=boto_session, default_bucket=cfg["default_bucket"])

    estimator = PyTorch(
        entry_point=args.entry_point,
        source_dir=str(ROOT / "training"),
        role=cfg["role_arn"],
        instance_count=1,
        instance_type=args.instance_type or cfg["training_instance"],
        framework_version=cfg["framework_version"],
        py_version=cfg["py_version"],
        image_uri=cfg["training_image_uri"],
        output_path=cfg["output_s3"],
        code_location=cfg["code_s3"],
        max_run=max_run,
        base_job_name=f"p1-cxr-{cfg['student_id']}-{args.experiment_id}",
        hyperparameters=hyperparameters,
        metric_definitions=METRIC_DEFINITIONS,
        sagemaker_session=sm_session,
    )

    print(f"작업 제출: {job_name} (실험 {args.experiment_id}, 최대 {max_run}초)", flush=True)
    print(f"hyperparameters: {hyperparameters}", flush=True)
    estimator.fit(channels, wait=False, job_name=job_name)

    submission_path = SUBMISSION_DIR / f"{job_name}.json"
    record = {
        "job_name": job_name,
        "experiment_id": args.experiment_id,
        "submitted_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "input_channels": channels,
        "output_s3": cfg["output_s3"],
        "instance_type": args.instance_type or cfg["training_instance"],
        "framework_version": cfg["framework_version"],
        "py_version": cfg["py_version"],
        "max_run_seconds": max_run,
        "hyperparameters": hyperparameters,
        # 비밀 값은 원문 대신 해시만 남긴다
        "aws_config_sha256": {
            "file": sha256_file(CONFIG_PATH),
            "role_arn": sha256_text(cfg["role_arn"]),
            "training_image_uri": sha256_text(cfg["training_image_uri"]),
        },
        "entry_point": args.entry_point,
        "train_py_sha256": sha256_file(train_py),
        "split_manifest_sha256": sha256_file(Path(args.manifest_file) if args.manifest_file else SPLIT_MANIFEST),
        "status": "Submitted",
    }
    write_submission(submission_path, record)
    print(f"제출 기록 저장: {submission_path.relative_to(ROOT)}", flush=True)

    if not args.wait:
        print("--wait 없이 제출만 했습니다. 상태는 SageMaker 콘솔에서 확인하세요.")
        return

    desc = wait_for_job(sm_session.sagemaker_client, job_name)
    status = desc["TrainingJobStatus"]
    record["status"] = status  # 실패·중단을 Completed 로 적지 않는다
    record["secondary_status"] = desc.get("SecondaryStatus")
    record["training_time_seconds"] = desc.get("TrainingTimeInSeconds")
    record["billable_time_seconds"] = desc.get("BillableTimeInSeconds")
    record["failure_reason"] = desc.get("FailureReason")
    record["model_artifacts_s3"] = desc.get("ModelArtifacts", {}).get("S3ModelArtifacts")
    record["finished_at_kst"] = datetime.now(KST).isoformat(timespec="seconds")
    write_submission(submission_path, record)
    print(f"최종 상태: {status}, 학습 시간 {record['training_time_seconds']}초", flush=True)
    if status != "Completed":
        print(f"실패 이유: {record['failure_reason']}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
