"""(10/23 배포 확인) Endpoint 예측 확률 검산: val 사진 20장을 엔드포인트에 보내 로컬 B512 출력과 비교한다 (val만, test는 열지 않음)."""
import csv
import glob
import json
import random
import sys
import time
from pathlib import Path

import boto3
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / "aws_config.json"))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))
import shift_eval as SE  # noqa: E402
import train as T  # noqa: E402

S = 512
IMG = Path.home() / "data/chest_xray"
EP = f"p1-cxr-{CFG['student_id']}-b-endpoint"
CLASSES = ["NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA"]
sess = boto3.Session(profile_name=CFG["profile"], region_name=CFG["region"])
rt = sess.client("sagemaker-runtime")

rows = [r for r in csv.DictReader(open(ROOT / "manifests/split_manifest.csv", encoding="utf-8")) if r["split"] == "val"]
ex = [r["image"] for r in csv.DictReader(open(ROOT / "reports/equipment_v2/gradcam/B512/examples.csv", encoding="utf-8"))]
rest = [r["relative_path"] for r in rows if r["relative_path"] not in ex]
random.Random(0).shuffle(rest)
paths = ex + rest[:8]

model = SE.load_model(ROOT / "runs/results/b512-20261022-111844/model")
model.eval()
tf = T.build_transform("P0", S)

diffs, same, times = [], 0, []
print(f"{'#':>2} {'로컬 예측':<9} {'엔드포인트 예측':<9} {'최대 확률 차이':>12}  ms")
for k, rel in enumerate(paths):
    p = IMG / rel
    with Image.open(p) as im:
        x = tf(im.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        lp = torch.softmax(model(x).float(), 1)[0].numpy()
    body = p.read_bytes()
    ctype = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    t0 = time.perf_counter()
    resp = rt.invoke_endpoint(EndpointName=EP, ContentType=ctype, Accept="application/json", Body=body)
    times.append((time.perf_counter() - t0) * 1000)
    out = json.loads(resp["Body"].read())
    ep = np.array([out["probabilities"][c] for c in CLASSES])
    d = float(np.abs(lp - ep).max())
    diffs.append(d)
    same += int(lp.argmax() == ep.argmax())
    print(f"{k:>2} {CLASSES[lp.argmax()][:8]:<9} {out['predicted_label'][:8]:<9} {d:>12.2e}  {times[-1]:.0f}")
print(f"\n사진 {len(paths)}장 · 예측 클래스 일치 {same}/{len(paths)} · 확률 최대 차이 {max(diffs):.2e} · 평균 {np.mean(diffs):.2e}")
print(f"응답 시간(첫 호출 포함) 중앙값 {np.median(times):.0f} ms, 최대 {max(times):.0f} ms")
print("모델 정보:", {k: out[k] for k in ("model_id", "selected_epoch")})
