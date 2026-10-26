"""끝난 SageMaker 학습 작업의 결과(model.tar.gz, output.tar.gz)를 내려받아 runs/results/<실험ID>-<작업 시각>/ 에 푼다.

runs/submissions/*.json (제출 기록)에서 실험 ID와 작업 이름을 읽고, 상태가 Completed 인 것만 받는다. 이미 받은 것은 건너뛴다.
비밀 정보(계정 번호, 학생 ID, 역할)는 화면에 출력하지 않는다.

사용 예 (p1 폴더에서):
    .venv/bin/python src/fetch_results.py --pattern '^r[0-4]-'
"""
import argparse
import json
import re
import tarfile
import tempfile
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]


def split_s3(uri):
    bucket, _, key = uri.replace("s3://", "", 1).partition("/")
    return bucket, key


def safe_extract(tar_path, dest):
    with tarfile.open(tar_path) as tar:
        for m in tar.getmembers():
            if m.name.startswith("/") or ".." in Path(m.name).parts:
                raise SystemExit(f"압축 안에 이상한 경로가 있습니다: {m.name}")
        tar.extractall(dest)


def fetch(pattern, quiet=False):
    """반환: {실험ID: 상태 또는 'downloaded'/'have'}"""
    cfg = json.loads((ROOT / "aws_config.json").read_text(encoding="utf-8"))
    session = boto3.Session(profile_name=cfg["profile"], region_name=cfg["region"])
    sm, s3 = session.client("sagemaker"), session.client("s3")
    rx = re.compile(pattern)
    out = {}
    for path in sorted((ROOT / "runs/submissions").glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        exp = rec.get("experiment_id", "")
        if not rx.search(exp):
            continue
        job = rec["job_name"]
        stamp = re.search(r"(\d{8}-\d{6})$", job).group(1)
        dest = ROOT / "runs/results" / f"{exp}-{stamp}"
        if (dest / "experiment_summary.json").is_file() and (dest / "model/best_model.pt").is_file():
            out[exp] = "have"
            continue
        status = sm.describe_training_job(TrainingJobName=job)["TrainingJobStatus"]
        if status != "Completed":
            out[exp] = status
            continue
        bucket, prefix = split_s3(rec["output_s3"])
        with tempfile.TemporaryDirectory() as tmp:
            for name, sub in (("model.tar.gz", dest / "model"), ("output.tar.gz", dest)):
                local = Path(tmp) / name
                s3.download_file(bucket, f"{prefix}{job}/output/{name}", str(local))
                sub.mkdir(parents=True, exist_ok=True)
                safe_extract(local, sub)
        out[exp] = "downloaded"
        if not quiet:
            print(f"내려받음: {exp} -> {dest.relative_to(ROOT)}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True, help="실험 ID 정규식 (예: '^r[0-4]-')")
    res = fetch(ap.parse_args().pattern)
    from collections import Counter
    print(dict(Counter(res.values())))
