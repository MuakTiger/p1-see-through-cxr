"""장치 설정 필터 v2.1 val 시험 SageMaker 작업 제출 스크립트 (GPU, 재학습 없음, val만).

맥 CPU에서는 모델 하나에 약 45분 걸리는 시험(src/equipment_eval_direct.py)을 학습과 같은 컨테이너·GPU 인스턴스에서 돌린다.
submit_evaluation.py(test 평가)와 같은 방식이다. 설정은 aws_config.json 에서 읽고 화면·기록에는 해시만 남긴다.

하는 일:
    1) 모델(best_model.pt + metadata.json)을 학생 S3 experiments/shift_eval_models/<이름>/ 에 올리고 크기를 확인한다.
    2) 진입점과 필요한 코드 6개를 runs/bundles/shift_eval/ 에 모아 SageMaker 작업으로 제출한다.
       equipment_direct.py 의 시험 전 해시(reports/equipment_v2/code_sha256_before_val_direct.txt)를 넘겨 컨테이너 안에서 비교한다.
    3) --wait 이면 끝날 때까지 기다린 뒤 output.tar.gz 를 내려받아 reports/equipment_v2/aws_val/ 에 푼다.

사용 예 (p1 폴더에서):
    .venv/bin/python submit_shift_eval.py --models B512 --wait
"""
import argparse
import hashlib
import json
import shutil
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
BUNDLE = ROOT / "runs" / "bundles" / "shift_eval"
ENTRY = "equipment_eval_direct.py"
CODE_FILES = ["src/equipment_eval_direct.py", "src/shift_eval.py", "training/equipment_direct.py", "training/shift_filters.py",
              "training/train.py", "training/preprocess.py"]
HASH_FILE = ROOT / "reports" / "equipment_v2" / "code_sha256_before_val_direct.txt"
AWS_OUT = ROOT / "reports" / "equipment_v2" / "aws_val"
SUBMISSION_DIR = ROOT / "runs" / "submissions"
MODELS_LOCAL = {"B512": ROOT / "runs/results/b512-20261022-111844/model"}  # 이름 → 모델 폴더 (새 모델은 여기에 추가)
KST = timezone(timedelta(hours=9))
POLL_SECONDS = 30
TERMINAL_STATUSES = {"Completed", "Failed", "Stopped"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_s3(uri):
    bucket, _, key = uri.replace("s3://", "", 1).partition("/")
    return bucket, key


def upload_models(s3, models_s3, names):
    bucket, prefix = split_s3(models_s3)
    shas = {}
    for name in names:
        for fname in ("best_model.pt", "metadata.json"):
            local = MODELS_LOCAL[name] / fname
            if not local.is_file():
                raise SystemExit(f"파일이 없습니다: {local}")
            key = f"{prefix}{name}/{fname}"
            s3.upload_file(str(local), bucket, key)
            size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
            if size != local.stat().st_size:
                raise SystemExit(f"업로드 크기가 다릅니다: {key} ({size} vs {local.stat().st_size})")
        shas[name] = sha256_file(MODELS_LOCAL[name] / "best_model.pt")
        print(f"  모델 {name} 업로드 확인 (sha256 {shas[name][:12]}…)", flush=True)
    return shas


def make_bundle():
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)
    for rel in CODE_FILES:
        shutil.copy2(ROOT / rel, BUNDLE / Path(rel).name)


def wait_for_job(sm_client, job_name):
    while True:
        desc = sm_client.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        print(f"[{datetime.now(KST):%H:%M:%S}] 상태: {status} / {desc.get('SecondaryStatus', '')}", flush=True)
        if status in TERMINAL_STATUSES:
            return desc
        time.sleep(POLL_SECONDS)


def download_output(s3, output_s3, job_name, dest):
    bucket, prefix = split_s3(output_s3)
    key = f"{prefix}{job_name}/output/output.tar.gz"
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest / "output.tar.gz"
    s3.download_file(bucket, key, str(tar_path))
    with tarfile.open(tar_path) as tar:
        for m in tar.getmembers():  # 압축 안의 경로가 폴더 밖으로 나가지 않는지 확인
            if m.name.startswith("/") or ".." in Path(m.name).parts:
                raise SystemExit(f"압축 안에 이상한 경로가 있습니다: {m.name}")
        tar.extractall(dest)
    tar_path.unlink()
    return f"s3://{bucket}/{key}"


def main():
    ap = argparse.ArgumentParser(description="장치 설정 필터 v2.1 val 시험 SageMaker 작업 제출")
    ap.add_argument("--models", default="B512", help=f"평가할 모델 이름 (쉼표 구분, 가능: {', '.join(MODELS_LOCAL)})")
    ap.add_argument("--instance-type", default="ml.g4dn.2xlarge")
    ap.add_argument("--max-run", type=int, default=3600)
    ap.add_argument("--out-name", default="aws_val", help="결과를 풀 폴더 (reports/equipment_v2/ 아래)")
    ap.add_argument("--wait", action="store_true", help="끝날 때까지 기다리고 결과를 내려받는다")
    args = ap.parse_args()

    names = [n for n in args.models.split(",") if n]
    for n in names:
        if n not in MODELS_LOCAL:
            raise SystemExit(f"알 수 없는 모델 이름: {n}")
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected_hash = {Path(l.split()[1]).name: l.split()[0] for l in open(HASH_FILE, encoding="utf-8") if l.strip()}["equipment_direct.py"]
    if sha256_file(ROOT / "training/equipment_direct.py") != expected_hash:
        raise SystemExit("training/equipment_direct.py 가 시험 전에 적어 둔 해시와 다릅니다. 제출하지 않습니다.")
    stamp = datetime.now(KST).strftime("%Y%m%d-%H%M%S")
    job_name = f"p1-cxr-{cfg['student_id']}-shifteval-{stamp}"
    models_s3 = f"{cfg['output_s3']}shift_eval_models/"
    channels = {"images": cfg["source_s3"], "manifests": cfg["manifest_s3"], "models": models_s3}

    boto_session = boto3.Session(profile_name=cfg["profile"], region_name=cfg["region"])
    s3 = boto_session.client("s3")
    print(f"모델 업로드: {models_s3}", flush=True)
    model_shas = upload_models(s3, models_s3, names)
    make_bundle()

    hyperparameters = {"model-ids": ",".join(names), "n-boot": 1000, "seed": 42, "plot": 0, "expected-hash": expected_hash}
    sm_session = sagemaker.Session(boto_session=boto_session, default_bucket=cfg["default_bucket"])
    estimator = PyTorch(
        entry_point=ENTRY, source_dir=str(BUNDLE), role=cfg["role_arn"], instance_count=1, instance_type=args.instance_type,
        framework_version=cfg["framework_version"], py_version=cfg["py_version"], image_uri=cfg["training_image_uri"],
        output_path=cfg["output_s3"], code_location=cfg["code_s3"], max_run=args.max_run,
        base_job_name=f"p1-cxr-{cfg['student_id']}-shifteval", hyperparameters=hyperparameters, sagemaker_session=sm_session)
    print(f"작업 제출: {job_name} ({args.instance_type}, 최대 {args.max_run}초)", flush=True)
    estimator.fit(channels, wait=False, job_name=job_name)

    submission_path = SUBMISSION_DIR / f"{job_name}.json"
    record = {
        "job_name": job_name, "experiment_id": "shifteval", "submitted_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "input_channels": channels, "output_s3": cfg["output_s3"], "instance_type": args.instance_type,
        "framework_version": cfg["framework_version"], "py_version": cfg["py_version"], "max_run_seconds": args.max_run,
        "hyperparameters": hyperparameters,
        "aws_config_sha256": {"file": sha256_file(CONFIG_PATH), "role_arn": sha256_text(cfg["role_arn"]),
                              "training_image_uri": sha256_text(cfg["training_image_uri"])},
        "code_sha256": {rel: sha256_file(ROOT / rel) for rel in CODE_FILES}, "model_sha256": model_shas, "status": "Submitted"}
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"제출 기록 저장: {submission_path.relative_to(ROOT)}", flush=True)
    if not args.wait:
        print("--wait 없이 제출만 했습니다.", flush=True)
        return

    desc = wait_for_job(sm_session.sagemaker_client, job_name)
    status = desc["TrainingJobStatus"]
    record.update({"status": status, "secondary_status": desc.get("SecondaryStatus"),
                   "training_time_seconds": desc.get("TrainingTimeInSeconds"), "billable_time_seconds": desc.get("BillableTimeInSeconds"),
                   "failure_reason": desc.get("FailureReason"), "finished_at_kst": datetime.now(KST).isoformat(timespec="seconds")})
    if status == "Completed":
        dest = ROOT / "reports" / "equipment_v2" / args.out_name
        record["output_tar_s3"] = download_output(s3, cfg["output_s3"], job_name, dest)
        record["local_output_dir"] = str(dest.relative_to(ROOT))
    submission_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"최종 상태: {status}, 실행 시간 {record.get('training_time_seconds')}초", flush=True)
    if status != "Completed":
        print(f"실패 이유: {record['failure_reason']}", file=sys.stderr)
        raise SystemExit(1)
    print(f"결과를 풀어 둔 곳: {record['local_output_dir']}", flush=True)


if __name__ == "__main__":
    main()
