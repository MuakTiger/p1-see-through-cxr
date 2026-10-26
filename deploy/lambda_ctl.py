"""Lambda 함수 만들기·테스트·삭제 (실습서 v4 5절을 boto3로). 코드는 강사 예시 lambda_function.py 그대로.

하위 명령: create / test / delete.  자격 증명 값과 12자리 숫자는 출력하지 않는다.
"""
import argparse
import base64
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
SRC = Path.home() / "Downloads/학생실습코드_v3/cxr_service/lambda/lambda_function.py"


def mask(s):
    return re.sub(r"\d{12}", "<ACCOUNT>", str(s))


def setup():
    c = json.load(open(ROOT / "aws_config.json", encoding="utf-8"))
    s = boto3.Session(profile_name=c["profile"], region_name=c["region"])
    base = f"p1-cxr-{c['student_id']}"
    acc = s.client("sts").get_caller_identity()["Account"]
    return c, s, {"fn": f"{base}-predict", "role": f"arn:aws:iam::{acc}:role/{base}-lambda", "endpoint": f"{base}-b-endpoint"}


def create():
    c, s, n = setup()
    lam = s.client("lambda")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(SRC, "lambda_function.py")
    for attempt in range(6):  # 새 역할이 퍼질 때까지 몇 번 기다린다
        try:
            lam.create_function(FunctionName=n["fn"], Runtime="python3.12", Role=n["role"], Handler="lambda_function.lambda_handler",
                                Code={"ZipFile": buf.getvalue()}, Timeout=85, MemorySize=256, Architectures=["x86_64"],
                                Environment={"Variables": {"ENDPOINT_NAME": n["endpoint"]}})
            break
        except lam.exceptions.InvalidParameterValueException as e:
            print("역할 전파 대기:", mask(e)[:120])
            time.sleep(8)
    lam.get_waiter("function_active_v2").wait(FunctionName=n["fn"])
    d = lam.get_function_configuration(FunctionName=n["fn"])
    print("Lambda", d["State"], d["LastUpdateStatus"], "timeout", d["Timeout"], "memory", d["MemorySize"], "handler", d["Handler"])


def test():
    c, s, n = setup()
    lam = s.client("lambda")
    img = Path.home() / "Downloads/학생실습코드_v3/cxr_service/sample/test_image.jpg"
    payload = {"content_type": "image/jpeg", "image_base64": base64.b64encode(img.read_bytes()).decode("ascii")}
    r = lam.invoke(FunctionName=n["fn"], InvocationType="RequestResponse", Payload=json.dumps(payload).encode())
    out = json.loads(r["Payload"].read())
    print(mask(json.dumps(out, ensure_ascii=False))[:600])


def delete():
    c, s, n = setup()
    try:
        s.client("lambda").delete_function(FunctionName=n["fn"])
        print("Lambda 삭제 완료")
    except Exception as e:  # noqa: BLE001
        print("건너뜀:", mask(e)[:150])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["create", "test", "delete"])
    a = ap.parse_args()
    try:
        {"create": create, "test": test, "delete": delete}[a.cmd]()
    except Exception as e:  # noqa: BLE001
        print("오류:", mask(e)[:400])
        sys.exit(1)
