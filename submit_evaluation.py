"""최종 test 평가 SageMaker 작업 제출 스크립트 (계획 6장: 학습과 같은 컨테이너에서 평가 1회).

training/evaluate.py 를 학습과 같은 컨테이너·인스턴스에서 실행한다. 설정은 aws_config.json 에서 읽는다
(비밀 정보가 있어서 git 에 올리지 않고, 화면·기록에는 해시만 남긴다).

하는 일:
    1) runs/test_eval_models/<ID>/ (best_model.pt + metadata.json, ID = A,B,C,D,P1,P2,P3,P4) 를
       학생 S3 experiments/test_eval_models/<ID>/ 에 올리고, 올린 파일 크기를 다시 확인한다.
    2) evaluation_plan.md 의 sha256 을 계산해 evaluate.py 에 넘긴다 (컨테이너 안에서 고정 값과 비교).
    3) 작업 p1-cxr-<학생ID>-testeval-<시각> 을 제출하고 (--wait) 끝날 때까지 기다린다.
    4) output.tar.gz 를 내려받아 reports/test_eval/aws/ 에 푼다.

사용 예 (p1 폴더에서):
    .venv/bin/python submit_evaluation.py --wait

결과:
    runs/submissions/<작업이름>.json   제출 기록 (최종 상태까지)
    reports/test_eval/aws/             evaluate.py 산출물 7종
"""
import argparse
import hashlib
import json
import sys
import tarfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "aws_config.json"
EVALUATE_PY = ROOT / "training" / "evaluate.py"
PLAN_PATH = ROOT / "reports" / "test_eval" / "evaluation_plan.md"
SPLIT_MANIFEST = ROOT / "manifests" / "split_manifest.csv"
MODELS_LOCAL = ROOT / "runs" / "test_eval_models"
AWS_OUT = ROOT / "reports" / "test_eval" / "aws"
SUBMISSION_DIR = ROOT / "runs" / "submissions"
MODEL_IDS = ["A", "B", "C", "D", "P1", "P2", "P3", "P4"]
MAX_RUN_SECONDS = 1800
KST = timezone(timedelta(hours=9))
POLL_SECONDS = 30
TERMINAL_STATUSES = {"Completed", "Failed", "Stopped"}


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_s3(uri):
    bucket, _, key = uri.replace("s3://", "", 1).partition("/")
    return bucket, key


def upload_models(s3, models_s3):
    """모델 8개를 올리고 S3 쪽 파일 크기가 로컬과 같은지 확인한다. 반환: {ID: sha256}"""
    bucket, prefix = split_s3(models_s3)
    shas = {}
    for mid in MODEL_IDS:
        for fname in ("best_model.pt", "metadata.json"):
            local = MODELS_LOCAL / mid / fname
            if not local.is_file():
                raise SystemExit(f"파일이 없습니다: {local}")
            key = f"{prefix}{mid}/{fname}"
            s3.upload_file(str(local), bucket, key)
            size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
            if size != local.stat().st_size:
                raise SystemExit(f"업로드 크기가 다릅니다: {key} ({size} vs {local.stat().st_size})")
        shas[mid] = sha256_file(MODELS_LOCAL / mid / "best_model.pt")
        print(f"  모델 {mid} 업로드 확인 (sha256 {shas[mid][:12]}…)", flush=True)
    return shas


def write_submission(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def wait_for_job(sm_client, job_name):
    while True:
        desc = sm_client.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        print(f"[{datetime.now(KST):%H:%M:%S}] 상태: {status} / {desc.get('SecondaryStatus', '')}", flush=True)
        if status in TERMINAL_STATUSES:
            return desc
        time.sleep(POLL_SECONDS)


def download_output(s3, output_s3, job_name):
    """output.tar.gz 를 내려받아 reports/test_eval/aws/ 에 푼다."""
    bucket, prefix = split_s3(output_s3)
    key = f"{prefix}{job_name}/output/output.tar.gz"
    AWS_OUT.mkdir(parents=True, exist_ok=True)
    tar_path = AWS_OUT / "output.tar.gz"
    s3.download_file(bucket, key, str(tar_path))
    with tarfile.open(tar_path) as tar:
        for m in tar.getmembers():  # 압축 안의 경로가 폴더 밖으로 나가지 않는지 확인
            if m.name.startswith("/") or ".." in Path(m.name).parts:
                raise SystemExit(f"압축 안에 이상한 경로가 있습니다: {m.name}")
        tar.extractall(AWS_OUT)
    tar_path.unlink()
    return f"s3://{bucket}/{key}"


def main():
    ap = argparse.ArgumentParser(description="최종 test 평가 SageMaker 작업 제출")
    ap.add_argument("--wait", action="store_true", help="끝날 때까지 기다리고 결과를 내려받는다")
    ap.add_argument("--retry-of", default=None, help="재실행이면 앞 작업 이름 (기록용)")
    ap.add_argument("--rerun-reason", default=None, help="재실행 이유 (기록용)")
    args = ap.parse_args()

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for p in (EVALUATE_PY, PLAN_PATH, SPLIT_MANIFEST):
        if not p.is_file():
            raise SystemExit(f"파일이 없습니다: {p}")
    plan_sha = sha256_file(PLAN_PATH)
    stamp = datetime.now(KST).strftime("%Y%m%d-%H%M%S")
    job_name = f"p1-cxr-{cfg['student_id']}-testeval-{stamp}"
    models_s3 = f"{cfg['output_s3']}test_eval_models/"
    channels = {"images": cfg["source_s3"], "manifests": cfg["manifest_s3"], "models": models_s3}

    boto_session = boto3.Session(profile_name=cfg["profile"], region_name=cfg["region"])
    s3 = boto_session.client("s3")
    print(f"모델 업로드: {models_s3}", flush=True)
    model_shas = upload_models(s3, models_s3)

    hyperparameters = {"primary": "D", "n-boot": 2000, "seed": 42, "batch-size": 64, "num-workers": 4,
                       "plan-sha256": plan_sha, "run-label": "aws"}
    sm_session = sagemaker.Session(boto_session=boto_session, default_bucket=cfg["default_bucket"])
    estimator = PyTorch(
        entry_point="evaluate.py",
        source_dir=str(ROOT / "training"),
        role=cfg["role_arn"],
        instance_count=1,
        instance_type=cfg["training_instance"],
        framework_version=cfg["framework_version"],
        py_version=cfg["py_version"],
        image_uri=cfg["training_image_uri"],
        output_path=cfg["output_s3"],
        code_location=cfg["code_s3"],
        max_run=MAX_RUN_SECONDS,
        base_job_name=f"p1-cxr-{cfg['student_id']}-testeval",
        hyperparameters=hyperparameters,
        sagemaker_session=sm_session,
    )
    print(f"작업 제출: {job_name} (최대 {MAX_RUN_SECONDS}초)", flush=True)
    estimator.fit(channels, wait=False, job_name=job_name)

    submission_path = SUBMISSION_DIR / f"{job_name}.json"
    record = {
        "job_name": job_name, "experiment_id": "testeval", "retry_of": args.retry_of,
        "rerun_reason": args.rerun_reason,
        "submitted_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "input_channels": channels, "output_s3": cfg["output_s3"], "instance_type": cfg["training_instance"],
        "framework_version": cfg["framework_version"], "py_version": cfg["py_version"],
        "max_run_seconds": MAX_RUN_SECONDS, "hyperparameters": hyperparameters,
        "aws_config_sha256": {"file": sha256_file(CONFIG_PATH), "role_arn": sha256_text(cfg["role_arn"]),
                              "training_image_uri": sha256_text(cfg["training_image_uri"])},
        "evaluate_py_sha256": sha256_file(EVALUATE_PY), "train_py_sha256": sha256_file(ROOT / "training" / "train.py"),
        "evaluation_plan_sha256": plan_sha, "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
        "model_sha256": model_shas, "status": "Submitted",
    }
    write_submission(submission_path, record)
    print(f"제출 기록 저장: {submission_path.relative_to(ROOT)}", flush=True)
    if not args.wait:
        return

    desc = wait_for_job(sm_session.sagemaker_client, job_name)
    status = desc["TrainingJobStatus"]
    record.update({"status": status, "secondary_status": desc.get("SecondaryStatus"),
                   "training_time_seconds": desc.get("TrainingTimeInSeconds"),
                   "billable_time_seconds": desc.get("BillableTimeInSeconds"),
                   "failure_reason": desc.get("FailureReason"),
                   "finished_at_kst": datetime.now(KST).isoformat(timespec="seconds")})
    if status == "Completed":
        record["output_tar_s3"] = download_output(s3, cfg["output_s3"], job_name)
        record["local_output_dir"] = str(AWS_OUT.relative_to(ROOT))
    write_submission(submission_path, record)
    print(f"최종 상태: {status}, 실행 시간 {record['training_time_seconds']}초", flush=True)
    if status != "Completed":
        print(f"실패 이유: {record['failure_reason']}", file=sys.stderr)
        raise SystemExit(1)
    print(f"결과를 풀어 둔 곳: {AWS_OUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
