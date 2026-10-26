"""EC2 웹 서버(Flask) 만들기·확인·삭제 (실습서 v4 6절을 boto3와 user-data로).

create : 역할/인스턴스 프로필(lambda:InvokeFunction만), 키 페어, 보안 그룹(SSH 22·웹 8000, 내 공인 IP/32만), t3.small 시작.
         웹 코드(강사 예시 web/ 폴더)는 user-data로 심어 첫 부팅 때 자동 설치한다. 모델·PyTorch는 올리지 않는다.
status : 인스턴스 상태와 공인 IP, /health 응답
add-ip : 현재 공인 IP/32를 보안 그룹에 추가 (발표장 네트워크가 바뀌었을 때)
delete : 인스턴스 종료 → 보안 그룹·키 페어·인스턴스 프로필·역할 삭제

자격 증명 값과 12자리 숫자는 출력하지 않는다. 개인키(.pem)는 deploy/out/ 에만 저장하고 출력하지 않는다.
"""
import argparse
import base64
import io
import json
import os
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy/out"
WEB = Path.home() / "Downloads/학생실습코드_v3/cxr_service/web"


def mask(s):
    return re.sub(r"\d{12}", "<ACCOUNT>", str(s))


def setup():
    c = json.load(open(ROOT / "aws_config.json", encoding="utf-8"))
    s = boto3.Session(profile_name=c["profile"], region_name=c["region"])
    base = f"p1-cxr-{c['student_id']}"
    acc = s.client("sts").get_caller_identity()["Account"]
    n = {"base": base, "web": f"{base}-web", "sg": f"{base}-web-sg", "fn": f"{base}-predict",
         "fn_arn": f"arn:aws:lambda:{c['region']}:{acc}:function:{base}-predict"}
    return c, s, n


def my_ip():
    return urllib.request.urlopen("https://checkip.amazonaws.com", timeout=10).read().decode().strip()


def user_data(fn_name, region):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for p in (WEB / "app.py", WEB / "requirements.txt", WEB / "templates/index.html"):
            t.add(p, arcname=str(p.relative_to(WEB)))
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"""#!/bin/bash
set -x
exec > /var/log/cxr-setup.log 2>&1
dnf install -y python3.11 python3.11-pip
mkdir -p /opt/cxr
echo '{b64}' | base64 -d | tar xz -C /opt/cxr
chown -R ec2-user:ec2-user /opt/cxr
python3.11 -m venv /opt/cxr/.venv
/opt/cxr/.venv/bin/python -m pip install -r /opt/cxr/requirements.txt
cat > /etc/systemd/system/cxr-web.service <<'UNIT'
[Unit]
Description=CXR Flask Web Service
After=network-online.target
Wants=network-online.target

[Service]
User=ec2-user
WorkingDirectory=/opt/cxr
Environment=AWS_DEFAULT_REGION={region}
Environment=LAMBDA_FUNCTION_NAME={fn_name}
ExecStart=/opt/cxr/.venv/bin/gunicorn --bind 0.0.0.0:8000 --workers 1 --threads 2 --timeout 100 --access-logfile - --error-logfile - "app:create_app()"
Restart=on-failure

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now cxr-web
echo SETUP_DONE
"""


def create():
    c, s, n = setup()
    iam, ec2 = s.client("iam"), s.client("ec2")
    OUT.mkdir(exist_ok=True)
    state = {}
    ip = my_ip()
    # 1) 역할 + 인스턴스 프로필
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    try:
        iam.create_role(RoleName=n["web"], AssumeRolePolicyDocument=json.dumps(trust))
    except iam.exceptions.EntityAlreadyExistsException:
        pass
    iam.put_role_policy(RoleName=n["web"], PolicyName="invoke-own-lambda", PolicyDocument=json.dumps(
        {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": "lambda:InvokeFunction", "Resource": n["fn_arn"]}]}))
    try:
        iam.create_instance_profile(InstanceProfileName=n["web"])
        iam.add_role_to_instance_profile(InstanceProfileName=n["web"], RoleName=n["web"])
    except iam.exceptions.EntityAlreadyExistsException:
        pass
    print("역할/인스턴스 프로필 준비")
    # 2) 키 페어
    pem = OUT / f"{n['web']}.pem"
    if not pem.exists():
        kp = ec2.create_key_pair(KeyName=n["web"], KeyType="rsa", KeyFormat="pem")
        pem.write_text(kp["KeyMaterial"])
        os.chmod(pem, 0o400)
    print("키 페어 준비 (개인키는 deploy/out/ 에 저장)")
    # 3) 보안 그룹: 내 IP/32 만
    vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    try:
        sg = ec2.create_security_group(GroupName=n["sg"], Description="CXR web: SSH and 8000 from my IP only", VpcId=vpc)["GroupId"]
        ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=[
            {"IpProtocol": "tcp", "FromPort": p, "ToPort": p, "IpRanges": [{"CidrIp": f"{ip}/32", "Description": "my ip"}]} for p in (22, 8000)])
    except ClientError as e:
        if e.response["Error"]["Code"] != "InvalidGroup.Duplicate":
            raise
        sg = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [n["sg"]]}])["SecurityGroups"][0]["GroupId"]
    print(f"보안 그룹 준비: 22·8000 은 {ip}/32 에서만 허용")
    # 4) AMI (Amazon Linux 2023, x86_64)
    imgs = ec2.describe_images(Owners=["amazon"], Filters=[{"Name": "name", "Values": ["al2023-ami-2023*-x86_64"]},
                                                       {"Name": "state", "Values": ["available"]}, {"Name": "architecture", "Values": ["x86_64"]}])["Images"]
    ami = sorted(imgs, key=lambda i: i["CreationDate"])[-1]["ImageId"]
    subnet = ec2.describe_subnets(Filters=[{"Name": "default-for-az", "Values": ["true"]}, {"Name": "availability-zone", "Values": [c["region"] + "a"]}])["Subnets"][0]["SubnetId"]
    # 5) 인스턴스
    for attempt in range(8):
        try:
            r = ec2.run_instances(
                ImageId=ami, InstanceType="t3.small", MinCount=1, MaxCount=1, KeyName=n["web"],
                IamInstanceProfile={"Name": n["web"]}, UserData=user_data(n["fn"], c["region"]),
                NetworkInterfaces=[{"DeviceIndex": 0, "SubnetId": subnet, "Groups": [sg], "AssociatePublicIpAddress": True}],
                BlockDeviceMappings=[{"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 15, "VolumeType": "gp3", "Encrypted": True, "DeleteOnTermination": True}}],
                MetadataOptions={"HttpTokens": "required"},
                TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": n["web"]}]}])
            break
        except ClientError as e:
            if "Invalid IAM Instance Profile" in str(e) or "iamInstanceProfile" in str(e):
                print("인스턴스 프로필 전파 대기...")
                time.sleep(8)
                continue
            raise
    iid = r["Instances"][0]["InstanceId"]
    json.dump({"instance_id": iid, "sg_id": sg}, open(OUT / "ec2.json", "w"))
    print("인스턴스 시작 요청:", iid)


def state_of(s, n):
    ec2 = s.client("ec2")
    d = json.load(open(OUT / "ec2.json"))
    inst = ec2.describe_instances(InstanceIds=[d["instance_id"]])["Reservations"][0]["Instances"][0]
    return d, inst


def status():
    c, s, n = setup()
    d, inst = state_of(s, n)
    ip = inst.get("PublicIpAddress")
    print("상태:", inst["State"]["Name"], "| 공인 IP:", ip)
    if ip:
        try:
            print("/health:", urllib.request.urlopen(f"http://{ip}:8000/health", timeout=8).read().decode())
        except Exception as e:  # noqa: BLE001
            print("/health 아직 응답 없음:", str(e)[:80])


def add_ip(ip=None):
    c, s, n = setup()
    d = json.load(open(OUT / "ec2.json"))
    ip = ip or my_ip()
    if not re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", ip):
        raise ValueError("IPv4 주소 형식이 아닙니다: " + ip)
    try:
        s.client("ec2").authorize_security_group_ingress(GroupId=d["sg_id"], IpPermissions=[
            {"IpProtocol": "tcp", "FromPort": p, "ToPort": p, "IpRanges": [{"CidrIp": f"{ip}/32", "Description": "my ip (added)"}]} for p in (22, 8000, 8001)])
        print(f"{ip}/32 추가")
    except ClientError as e:
        print(mask(e)[:150])


def delete():
    c, s, n = setup()
    ec2, iam = s.client("ec2"), s.client("iam")
    f = OUT / "ec2.json"
    if f.exists():
        d = json.load(open(f))
        try:
            ec2.terminate_instances(InstanceIds=[d["instance_id"]])
            ec2.get_waiter("instance_terminated").wait(InstanceIds=[d["instance_id"]])
            print("인스턴스 종료 확인")
        except Exception as e:  # noqa: BLE001
            print("인스턴스:", mask(e)[:120])
        try:
            ec2.delete_security_group(GroupId=d["sg_id"])
            print("보안 그룹 삭제")
        except Exception as e:  # noqa: BLE001
            print("보안 그룹:", mask(e)[:120])
    for fn, kw in ((ec2.delete_key_pair, {"KeyName": n["web"]}),
                   (iam.remove_role_from_instance_profile, {"InstanceProfileName": n["web"], "RoleName": n["web"]}),
                   (iam.delete_instance_profile, {"InstanceProfileName": n["web"]}),
                   (iam.delete_role_policy, {"RoleName": n["web"], "PolicyName": "invoke-own-lambda"}),
                   (iam.delete_role, {"RoleName": n["web"]})):
        try:
            fn(**kw)
            print("삭제:", fn.__name__)
        except Exception as e:  # noqa: BLE001
            print("건너뜀:", fn.__name__, mask(e)[:100])
    pem = OUT / f"{n['web']}.pem"
    if pem.exists():
        os.chmod(pem, 0o600)
        pem.unlink()
        print("개인키 파일 삭제")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["create", "status", "add-ip", "delete"])
    ap.add_argument("--ip", default=None, help="add-ip 에서 허용할 공인 IPv4 (없으면 이 컴퓨터의 공인 IP). 발표용 노트북의 IP를 넣는다")
    a = ap.parse_args()
    try:
        {"create": create, "status": status, "add-ip": lambda: add_ip(a.ip), "delete": delete}[a.cmd]()
    except Exception as e:  # noqa: BLE001
        print("오류:", mask(e)[:500])
        sys.exit(1)
