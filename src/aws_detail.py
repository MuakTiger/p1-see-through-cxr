"""우리 배포 자원의 상세를 읽기 전용으로 본다 (describe/get 만, 생성·변경·호출 없음). 계정 번호 등은 출력 뒤 sed 로 가린다."""
import json
from pathlib import Path

import boto3

CFG = json.load(open(Path(__file__).resolve().parents[1] / "aws_config.json"))
PROFILE, REGION, SID = CFG["profile"], CFG["region"], CFG["student_id"]
s = boto3.Session(profile_name=PROFILE, region_name=REGION)
sm, lam, ec2 = s.client("sagemaker"), s.client("lambda"), s.client("ec2")

print("== 엔드포인트")
e = sm.describe_endpoint(EndpointName=f"p1-cxr-{SID}-b-endpoint")
print(" 상태", e["EndpointStatus"], "· 생성", e["CreationTime"].strftime("%m-%d %H:%M"), "· 수정", e["LastModifiedTime"].strftime("%m-%d %H:%M"))
cfg = sm.describe_endpoint_config(EndpointConfigName=e["EndpointConfigName"])
for v in cfg["ProductionVariants"]:
    print(" 구성", e["EndpointConfigName"], v["VariantName"], v.get("InstanceType"), "×", v.get("InitialInstanceCount"))
print(" 서버리스" if cfg["ProductionVariants"][0].get("ServerlessConfig") else " 상시 인스턴스")

print("== 모델")
m = sm.describe_model(ModelName=f"p1-cxr-{SID}-b-model")
c = m["PrimaryContainer"]
print(" 이미지", c["Image"].split("/")[-1])
print(" 모델 파일", c.get("ModelDataUrl", "").split("/", 3)[-1])
print(" 환경변수 키", sorted((c.get("Environment") or {}).keys()))
for k, v in (c.get("Environment") or {}).items():
    if k.startswith("SAGEMAKER_PROGRAM") or k.startswith("SAGEMAKER_SUBMIT"):
        print("   ", k, "=", v)

print("== Lambda")
f = lam.get_function_configuration(FunctionName=f"p1-cxr-{SID}-predict")
print(" ", f["Runtime"], f["Handler"], f["MemorySize"], "MB", f["Timeout"], "초", "· 수정", f["LastModified"][:19])
for k, v in (f.get("Environment", {}).get("Variables") or {}).items():
    low = k.lower()
    print("   env", k, "=", "<가림>" if any(x in low for x in ("key", "secret", "token", "password")) else v)
try:
    pol = lam.get_policy(FunctionName=f"p1-cxr-{SID}-predict")["Policy"]
    print(" 호출 정책 있음 (길이", len(pol), ")")
except Exception as ex:  # noqa: BLE001
    print(" 호출 정책:", type(ex).__name__)
try:
    fu = lam.get_function_url_config(FunctionName=f"p1-cxr-{SID}-predict")
    print(" 함수 URL 있음, 인증", fu["AuthType"])
except Exception as ex:  # noqa: BLE001
    print(" 함수 URL: 없음 또는 조회 불가", type(ex).__name__)

print("== EC2")
for r in ec2.describe_instances(Filters=[{"Name": "tag:Name", "Values": [f"p1-cxr-{SID}-web*"]}])["Reservations"]:
    for i in r["Instances"]:
        print(" ", i["InstanceId"][:6] + "…", i["InstanceType"], i["State"]["Name"], "· 시작", i["LaunchTime"].strftime("%m-%d %H:%M"), "· 공인 IP", "있음" if i.get("PublicIpAddress") else "없음")
        for g in i["SecurityGroups"]:
            sg = ec2.describe_security_groups(GroupIds=[g["GroupId"]])["SecurityGroups"][0]
            for p in sg["IpPermissions"]:
                cidrs = [x["CidrIp"] for x in p.get("IpRanges", [])]
                shown = ["전체 공개(0.0.0.0/0)" if x == "0.0.0.0/0" else ("내 IP 하나(/32)" if x.endswith("/32") else x) for x in cidrs]
                print("   보안 그룹 인바운드", p.get("IpProtocol"), p.get("FromPort"), "-", p.get("ToPort"), shown)
        print("  인스턴스 프로파일", (i.get("IamInstanceProfile") or {}).get("Arn", "없음").split("/")[-1])
