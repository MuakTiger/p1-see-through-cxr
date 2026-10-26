#!/bin/bash
# 폐 영역 모델 3개(l-s42·43·44)와 같은 환경의 새 기준 2개(b512-s43·44)를 SageMaker 에 제출한다 (제출만 하고 기다리지 않는다).
set -u
cd ~/Documents/p1
BASE=$(.venv/bin/python -c "import json;print(json.load(open('aws_config.json'))['manifest_s3'].rsplit('manifests/',1)[0])")
IMG="${BASE}lung512/images/"; MAN="${BASE}lung512/manifests/"
mask() { sed -E 's/[0-9]{12}/<ACCOUNT>/g' | grep -E "작업 제출|제출 기록|Error|error|Exception|ResourceLimit|limit" | cut -c1-200; }
for s in 42 43 44; do
  echo "== 제출: l-s$s"
  SAGEMAKER_SUPPRESS_V2_WARNING=1 .venv/bin/python submit_training.py --experiment-id "l-s$s" --entry-point train_aug.py --seed "$s" --num-workers 8 \
    --instance-type ml.g4dn.2xlarge --images-uri "$IMG" --manifests-uri "$MAN" --manifest-file runs/lung512/manifests/split_manifest.csv \
    --extra train-mode=layer4 --extra input-size=512 --extra augment=none 2>&1 | mask
done
for s in 43 44; do
  echo "== 제출: b512-s$s"
  SAGEMAKER_SUPPRESS_V2_WARNING=1 .venv/bin/python submit_training.py --experiment-id "b512-s$s" --seed "$s" --num-workers 8 \
    --instance-type ml.g4dn.2xlarge --extra train-mode=layer4 --extra input-size=512 2>&1 | mask
done
