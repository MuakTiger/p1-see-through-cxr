#!/bin/bash
# 과적합 줄이기 실험 5가지(r0~r4)를 주어진 seed로 SageMaker 에 제출한다 (설정은 reports/equipment_v2/regularization_plan.md).
# 사용: bash src/submit_reg_wave.sh 42        (p1 폴더에서; 제출만 하고 기다리지 않는다)
# 공통: train_reg.py, 입력 512, 마지막 에폭 모델(select=last), ml.g4dn.2xlarge, 배치 16, num-workers 8.
set -u
SEED="$1"
cd "$(dirname "$0")/.."
COMMON=(--entry-point train_reg.py --seed "$SEED" --num-workers 8 --instance-type ml.g4dn.2xlarge
        --extra input-size=512 --extra select=last)
submit() {  # 이름, 나머지 인자
  local name="$1"; shift
  echo "== 제출: $name"
  SAGEMAKER_SUPPRESS_V2_WARNING=1 .venv/bin/python submit_training.py --experiment-id "$name" "${COMMON[@]}" "$@" 2>&1 \
    | sed -E 's/[0-9]{12}/<ACCOUNT>/g' | grep -E "작업 제출|제출 기록|Error|error|Exception|ResourceLimit|limit" | cut -c1-220
}
submit "r0-s$SEED"     --extra train-mode=layer4
submit "r1-ep5-s$SEED" --extra train-mode=layer4 --epochs 5
submit "r2-ls-s$SEED"  --extra train-mode=layer4 --extra label-smoothing=0.1
submit "r3-wd-s$SEED"  --extra train-mode=layer4 --extra weight-decay=0.1
submit "r4-frz-s$SEED" --extra train-mode=frozen
