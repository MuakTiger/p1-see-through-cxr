#!/bin/bash
# lv-s42·43·44 학습이 끝나면 받아 val·test 로 평가한다 (폐 모델 평가와 같은 방식, 줄기 2개).
cd ~/Documents/p1
SC=<임시 폴더>
H=$(shasum -a 256 reports/equipment_v2/test_plan_lung_vaccine.md | cut -d' ' -f1)
OUTD=reports/equipment_v2/lung; HF=reports/equipment_v2/test_plan_lung_vaccine.sha256
echo "[$(date +%H:%M)] lv 학습 끝나기를 기다림"
for i in $(seq 1 120); do
  .venv/bin/python src/fetch_results.py --pattern '^lv-s4[234]$' > $SC/fetch_lv.log 2>&1
  n=$(ls -d runs/results/lv-s4[234]-* 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -ge 3 ] && break
  sleep 60
done
echo "[$(date +%H:%M)] lv 결과 $n 개 받음"
[ "$n" -ge 3 ] || exit 1
md() { ls -d runs/results/$1-* | head -1; }
evalval() { local n="$1" M="$(md $1)/model"; .venv/bin/python src/equipment_eval_lung.py --masks-dir runs/lung_masks --model-dir "$M" --out $OUTD/val_$n --device mps --label $n > $OUTD/eval_val_$n.log 2>&1; echo "[$(date +%H:%M)] val 끝: $n"; }
evaltest() { local n="$1" M="$(md $1)/model"
  { echo "# 평가 직전 덧붙임 ($(date '+%Y-%m-%d %H:%M'), $n 의 test 점수를 보기 전)"; shasum -a 256 "$M/best_model.pt" "$M/metadata.json"; } >> $HF
  .venv/bin/python src/equipment_eval_lung.py --masks-dir runs/lung_masks --model-dir "$M" --out $OUTD/test_$n --device mps --label $n --split test --plan-file reports/equipment_v2/test_plan_lung_vaccine.md --plan-sha256 $H --hash-file $HF > $OUTD/eval_test_$n.log 2>&1
  echo "[$(date +%H:%M)] test 끝: $n"; }
( for n in lv-s42 lv-s43 lv-s44; do evalval $n; done ) &
(
  for i in $(seq 1 90); do [ -f runs/lung_masks/test_lung_masks.npz ] && break; sleep 10; done
  for n in lv-s42 lv-s43 lv-s44; do evaltest $n; done
) &
wait
echo "[$(date +%H:%M)] lv 모두 끝"
