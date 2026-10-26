"""AWS 자원 종료 상태와 학습 작업 청구 시간을 읽기 전용으로 조회해 마크다운 기록(종료 기록)을 만든다.

- 조회 API만 호출한다 (생성·삭제·수정 없음).
- 출력은 mask()를 거쳐 학생 ID는 <학생ID>, 12자리 숫자는 <ACCOUNT>로 가린다.
- aws_config.json (프로필·리전·학생 ID·버킷)은 p1 폴더에 있어야 한다. 이 파일은 제출물에 넣지 않는다.

사용 예 (p1 폴더에서):
    .venv/bin/python tools/aws_teardown_check.py reports/aws_teardown_check_20261026.md
"""
import collections
import datetime as dt
import json
import re
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / "aws_config.json"))
SID = CFG["student_id"]
REGION = CFG["region"]
OUT = Path(sys.argv[1])

sess = boto3.Session(profile_name=CFG["profile"], region_name=REGION)
sm = sess.client("sagemaker")
lam = sess.client("lambda")
ec2 = sess.client("ec2")
iam = sess.client("iam")
s3 = sess.client("s3")

KST = dt.timezone(dt.timedelta(hours=9))
now = dt.datetime.now(KST)


def mask(text):
    text = str(text).replace(SID, "<학생ID>")
    return re.sub(r"(?<!\d)\d{12}(?!\d)", "<ACCOUNT>", text)


def safe(fn, default="조회 불가"):
    try:
        return fn()
    except ClientError as e:
        return f"{default} ({e.response['Error']['Code']})"


def n(x):
    return x if isinstance(x, str) else len(x)


pref = f"p1-cxr-{SID}"
lines = []
add = lines.append

add("# AWS 종료 기록 (읽기 전용 조회)")
add("")
add(f"- 조회 시각: {now.strftime('%Y-%m-%d %H:%M')} KST (서울 리전 ap-northeast-2)")
add("- 방법: 학생 계정으로 조회 API만 호출했다 (생성·삭제·수정 없음). 학생 ID는 `<학생ID>`, 12자리 계정 번호는 `<ACCOUNT>`로 가렸다.")
add("- 이 계정은 다른 수강생과 함께 쓰는 공용 계정이라, 아래는 이름에 `p1-cxr-<학생ID>`가 들어간 내 자원만 센 것이다.")
add("- 만든 스크립트: `tools/aws_teardown_check.py`")
add("")


def listed(fn, key):
    items = []
    for page in sm.get_paginator(fn).paginate(NameContains=pref):
        items += page[key]
    return items


endpoints = safe(lambda: listed("list_endpoints", "Endpoints"))
configs = safe(lambda: listed("list_endpoint_configs", "EndpointConfigs"))
models = safe(lambda: listed("list_models", "Models"))
nbs = safe(lambda: listed("list_notebook_instances", "NotebookInstances"))
funcs = safe(lambda: [f for f in lam.list_functions()["Functions"] if pref in f["FunctionName"]])


def live_instances():
    out, states = [], collections.Counter()
    pages = ec2.get_paginator("describe_instances").paginate(Filters=[{"Name": "tag:Name", "Values": [f"*{SID}*"]}])
    for page in pages:
        for r in page["Reservations"]:
            for i in r["Instances"]:
                states[i["State"]["Name"]] += 1
                if i["State"]["Name"] not in ("terminated", "shutting-down"):
                    out.append(i["InstanceId"])
    return out, states


inst = safe(live_instances)
volumes = safe(lambda: ec2.describe_volumes(Filters=[{"Name": "tag:Name", "Values": [f"*{SID}*"]}])["Volumes"])
sgs = safe(lambda: ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [f"*{SID}*"]}])["SecurityGroups"])
keys = safe(lambda: ec2.describe_key_pairs(Filters=[{"Name": "key-name", "Values": [f"*{SID}*"]}])["KeyPairs"])
eips = safe(lambda: [a for a in ec2.describe_addresses()["Addresses"] if SID in json.dumps(a.get("Tags", []))])


def running_jobs():
    out = []
    for st in ("InProgress", "Stopping"):
        for page in sm.get_paginator("list_training_jobs").paginate(NameContains=pref, StatusEquals=st):
            out += page["TrainingJobSummaries"]
    return out


run_jobs = safe(running_jobs)

rows = [
    ("SageMaker Endpoint", n(endpoints), "삭제됨" if n(endpoints) == 0 else "남아 있음 — 확인 필요"),
    ("SageMaker Endpoint 구성", n(configs), "삭제됨" if n(configs) == 0 else "남아 있음 — 확인 필요"),
    ("SageMaker 모델", n(models), "삭제됨" if n(models) == 0 else "남아 있음 — 확인 필요"),
    ("SageMaker 노트북 인스턴스", n(nbs), "없음" if n(nbs) == 0 else "남아 있음 — 확인 필요"),
    ("Lambda 함수", n(funcs), "삭제됨" if n(funcs) == 0 else "남아 있음 — 확인 필요"),
]
if isinstance(inst, str):
    rows.append(("EC2 인스턴스 (종료되지 않은 것)", inst, ""))
else:
    rows.append(("EC2 인스턴스 (종료되지 않은 것)", len(inst[0]), "전부 종료됨" if not inst[0] else "남아 있음 — 확인 필요"))
rows += [
    ("EBS 볼륨", n(volumes), "없음" if n(volumes) == 0 else "남아 있음 — 확인 필요"),
    ("보안 그룹", n(sgs), "삭제됨" if n(sgs) == 0 else "남아 있음 — 확인 필요"),
    ("키 페어", n(keys), "삭제됨" if n(keys) == 0 else "남아 있음 — 확인 필요"),
    ("탄력적 IP", n(eips), "없음" if n(eips) == 0 else "남아 있음 — 확인 필요"),
    ("진행 중인 학습 작업", n(run_jobs), "없음" if n(run_jobs) == 0 else "남아 있음 — 확인 필요"),
]
add("## 1. 비용이 나가는 자원이 남아 있는가")
add("")
add("| 자원 | 남은 개수 | 판정 |")
add("|---|---|---|")
for name, cnt, verdict in rows:
    add(f"| {name} | {cnt} | {verdict} |")
add("")
if not isinstance(inst, str):
    add("EC2 인스턴스 상태 분포(이름 태그 기준): " + (", ".join(f"{k} {v}대" for k, v in sorted(inst[1].items())) if inst[1]
                                              else "기록 없음 (종료된 인스턴스는 한 시간쯤 뒤 목록에서 사라진다)"))
    add("")

add("## 2. 남아 있는 것 (비용은 거의 없음)")
add("")
for suffix in ("lambda", "web"):
    rn = f"{pref}-{suffix}"
    r = safe(lambda rn=rn: iam.get_role(RoleName=rn)["Role"]["RoleName"])
    add(f"- IAM 역할 `{mask(rn)}`: {'존재함' if r == rn else mask(r)} (인라인 정책만 있음, 비용 없음). 이 계정에는 삭제 권한이 없어 지우지 못했다 → 필요하면 강사에게 삭제를 요청한다.")
prof = safe(lambda: iam.get_instance_profile(InstanceProfileName=f"{pref}-web")["InstanceProfile"]["InstanceProfileName"])
add(f"- IAM 인스턴스 프로필 `{mask(pref)}-web`: {'존재함' if prof == f'{pref}-web' else mask(prof)}")
objs = safe(lambda: s3.list_objects_v2(Bucket=CFG["default_bucket"], Prefix=f"students/{SID}/P1/cxr/deployment/").get("Contents", []))
if isinstance(objs, str):
    add(f"- S3 `deployment/` 경로: {mask(objs)}")
else:
    for o in objs:
        add(f"- S3 `s3://<버킷>/students/<학생ID>/P1/cxr/{mask(o['Key'].split('P1/cxr/')[-1])}` ({o['Size'] / 1e6:.0f} MB, 보관 비용은 월 몇 센트 수준) — 배포용 모델 묶음. 다시 배포할 때 쓰려고 남겨 두었다.")
add("")

add("## 3. SageMaker 학습 작업과 청구 시간")
add("")
jobs = []
for page in sm.get_paginator("list_training_jobs").paginate(NameContains=pref, SortBy="CreationTime", SortOrder="Ascending"):
    jobs += page["TrainingJobSummaries"]
group_rx = re.compile(rf"^{re.escape(pref)}-(.+?)-(\d{{8}}-\d{{6}})$")
rec = []
for j in jobs:
    name = j["TrainingJobName"]
    d = safe(lambda name=name: sm.describe_training_job(TrainingJobName=name), default=None)
    if d is None or isinstance(d, str):
        continue
    m = group_rx.match(name)
    exp = m.group(1) if m else name
    first = exp.split("-")[0]
    if "smoke" in exp:
        grp = "smoke"
    elif first in ("r0", "r1", "r2", "r3", "r4"):
        grp = "r (과적합 줄이기, 취소)"
    else:
        grp = first
    rec.append({"grp": grp, "status": d["TrainingJobStatus"], "inst": d["ResourceConfig"]["InstanceType"],
                "billable": d.get("BillableTimeInSeconds", 0) or 0})
by_grp = collections.OrderedDict()
for r in rec:
    g = by_grp.setdefault(r["grp"], {"n": 0, "st": collections.Counter(), "inst": collections.Counter(), "bill": 0})
    g["n"] += 1
    g["st"][r["status"]] += 1
    g["inst"][r["inst"]] += 1
    g["bill"] += r["billable"]
add(f"이름에 `p1-cxr-<학생ID>`가 들어간 학습 작업 {len(rec)}개를 조회했다. 청구 시간은 AWS가 기록한 `BillableTimeInSeconds`다.")
add("")
add("| 실험 묶음 | 작업 수 | 상태 | 인스턴스 | 청구 시간(분) |")
add("|---|---|---|---|---|")
for g, v in by_grp.items():
    add(f"| {g} | {v['n']} | {', '.join(f'{k} {c}' for k, c in v['st'].items())} | {', '.join(f'{k} {c}' for k, c in v['inst'].items())} | {v['bill'] / 60:.0f} |")
add("")
tot = collections.Counter()
for r in rec:
    tot[r["inst"]] += r["billable"]
add("인스턴스 종류별 청구 시간 합계:")
add("")
add("| 인스턴스 | 청구 시간(시간) |")
add("|---|---|")
for k, v in sorted(tot.items()):
    add(f"| {k} | {v / 3600:.2f} |")
add(f"| 합계 | {sum(tot.values()) / 3600:.2f} |")
add("")
add("단가는 AWS 요금표를 확인하지 못해 금액으로 바꾸지 않았다. 위 시간에 해당 리전의 시간당 요금을 곱하면 학습 비용이 된다. "
    "JupyterLab(Studio) 공간에서 돌린 모델 6개(기준 b 3개, 장비 효과 증강 v 3개)는 학습 작업이 아니라서 위 표에 없다. "
    "학습 기록(`experiment_summary.json`)의 학습 시간 합계가 10,187초(약 2.83시간)이고, 인스턴스는 ml.g4dn.xlarge였다. "
    "학습이 끝난 뒤 공간의 앱은 삭제했다 (공간이 켜져 있던 시간은 학습 시간보다 길 수 있다).")
add("")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(mask("\n".join(lines)) + "\n", encoding="utf-8")
print(mask("\n".join(lines)))
