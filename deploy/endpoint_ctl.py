"""B512 SageMaker Endpoint 만들기·확인·검산·삭제 (강사 실습서 v4의 4절을 boto3로 한 것).

하위 명령:
    build    deploy/b512 → deploy/out/model_with_code.tar.gz (최상위에 best_model.pt, metadata.json, code/inference.py)
    upload   S3 deployment/b512/ 에 올림
    create   Model → EndpointConfig → Endpoint 생성 요청 (ml.m5.large 1대, 켜 두는 동안 시간당 과금)
    status   Endpoint 상태
    invoke   val 사진을 Endpoint에 보내 로컬 평가(val_predictions.csv)의 확률과 비교 (가이드의 "Endpoint 예측 확률 검산")
    delete   Endpoint, EndpointConfig, Model 삭제 (과금 중지)

자격 증명은 aws_config.json 의 프로필을 쓰고 아무 값도 출력하지 않는다 (12자리 숫자는 가려서 출력).
"""
import argparse
import json
import re
import sys
import tarfile
import time
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEP = ROOT / "deploy"
OUT = DEP / "out"
IMAGE_URI = "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:2.6.0-cpu-py312"  # 실습서 v4 2절
INSTANCE = "ml.m5.large"
CLASSES = ["NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA"]


def mask(s):
    return re.sub(r"\d{12}", "<ACCOUNT>", str(s))


def cfg():
    c = json.load(open(ROOT / "aws_config.json", encoding="utf-8"))
    m = re.match(r"s3://([^/]+)/(students/[^/]+/P1/cxr/)", c["manifest_s3"])
    c["bucket"], c["prefix"] = m.group(1), m.group(2)
    base = f"p1-cxr-{c['student_id']}"
    c["names"] = {"model": f"{base}-b-model", "config": f"{base}-b-config-v1", "endpoint": f"{base}-b-endpoint"}
    c["model_key"] = c["prefix"] + "deployment/b512/model_with_code.tar.gz"
    return c


def session(c):
    return boto3.Session(profile_name=c["profile"], region_name=c["region"])


def build(c):
    OUT.mkdir(exist_ok=True)
    src = DEP / "b512"
    dest = OUT / "model_with_code.tar.gz"
    with tarfile.open(dest, "w:gz") as t:
        t.add(src / "best_model.pt", arcname="best_model.pt")
        t.add(src / "metadata.json", arcname="metadata.json")
        t.add(src / "code/inference.py", arcname="code/inference.py")
    with tarfile.open(dest) as t:
        print("묶음 내용:", t.getnames(), f"{dest.stat().st_size / 1e6:.1f} MB")


def upload(c):
    s3 = session(c).client("s3")
    s3.upload_file(str(OUT / "model_with_code.tar.gz"), c["bucket"], c["model_key"])
    print("업로드 완료: deployment/b512/model_with_code.tar.gz")


def create(c):
    sm = session(c).client("sagemaker")
    n = c["names"]
    uri = f"s3://{c['bucket']}/{c['model_key']}"
    print("1/3 Model", flush=True)
    sm.create_model(ModelName=n["model"], ExecutionRoleArn=c["role_arn"],
                    PrimaryContainer={"Image": IMAGE_URI, "ModelDataUrl": uri,
                                      "Environment": {"SAGEMAKER_PROGRAM": "inference.py",
                                                      "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                                                      "SAGEMAKER_MODEL_SERVER_WORKERS": "1"}})
    print("2/3 EndpointConfig", flush=True)
    sm.create_endpoint_config(EndpointConfigName=n["config"], ProductionVariants=[{
        "VariantName": "AllTraffic", "ModelName": n["model"], "InitialInstanceCount": 1,
        "InstanceType": INSTANCE, "InitialVariantWeight": 1.0}])
    print("3/3 Endpoint", flush=True)
    sm.create_endpoint(EndpointName=n["endpoint"], EndpointConfigName=n["config"])
    print("생성 요청 완료. `status`로 InService까지 확인. 켜 두는 동안 호스팅 요금이 발생한다.")


def status(c, wait=False):
    sm = session(c).client("sagemaker")
    while True:
        try:
            d = sm.describe_endpoint(EndpointName=c["names"]["endpoint"])
        except sm.exceptions.ClientError as e:
            print("Endpoint 없음:", mask(e.response["Error"]["Message"]))
            return None
        s = d["EndpointStatus"]
        print(time.strftime("%H:%M:%S"), s, mask(d.get("FailureReason", "")), flush=True)
        if not wait or s in ("InService", "Failed"):
            return s
        time.sleep(30)


def invoke(c, n=30, seed=42):
    rt = session(c).client("sagemaker-runtime")
    v = pd.read_csv(ROOT / "runs/results/b512-20261022-111844/val_predictions.csv")
    pick = v.groupby("true_label", group_keys=False).apply(lambda g: g.sample(n // 3, random_state=seed))
    diffs, same, lat = [], 0, []
    for r in pick.itertuples():
        body = (Path.home() / "data/chest_xray" / r.relative_path).read_bytes()
        if len(body) > 2 * 1024 * 1024:
            continue
        t0 = time.time()
        resp = rt.invoke_endpoint(EndpointName=c["names"]["endpoint"], ContentType="image/jpeg", Accept="application/json", Body=body)
        lat.append(time.time() - t0)
        out = json.loads(resp["Body"].read())
        p = np.array([out["probabilities"][k] for k in CLASSES])
        ref = np.array([r.p_normal, r.p_bacterial, r.p_viral])
        diffs.append(np.abs(p - ref).max())
        same += int(np.argmax(p) == r.pred_label)
    print(f"Endpoint 검산 {len(diffs)}장: 확률 최대 차이 {max(diffs):.2e}, 평균 {np.mean(diffs):.2e}, 예측 일치 {same}/{len(diffs)}, "
          f"왕복 시간 평균 {np.mean(lat):.2f}초")
    print("응답 예:", json.dumps({k: out[k] for k in ("model_id", "selected_epoch", "predicted_label", "model_elapsed_ms")}, ensure_ascii=False))


def delete(c):
    sm = session(c).client("sagemaker")
    n = c["names"]
    for fn, kw in ((sm.delete_endpoint, {"EndpointName": n["endpoint"]}), (sm.delete_endpoint_config, {"EndpointConfigName": n["config"]}),
                   (sm.delete_model, {"ModelName": n["model"]})):
        try:
            fn(**kw)
            print("삭제:", fn.__name__, list(kw.values())[0].replace(c["student_id"], "<ME>"))
        except Exception as e:  # noqa: BLE001
            print("건너뜀:", fn.__name__, mask(str(e))[:120])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "upload", "create", "status", "wait", "invoke", "delete"])
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    c = cfg()
    {"build": build, "upload": upload, "create": create, "status": status,
     "wait": lambda c: status(c, wait=True), "invoke": lambda c: invoke(c, a.n), "delete": delete}[a.cmd](c)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print("오류:", mask(e))
        sys.exit(1)
