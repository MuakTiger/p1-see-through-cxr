#!/bin/bash
# 학습이 끝나면 결과를 받아 폐 모델 3개와 새 기준 2개를 val·test로 평가한다 (두 줄기를 동시에: val 줄기, test 줄기).
cd ~/Documents/p1
SC=<임시 폴더>
H=5a954f7ae81502fbba7fdf0e163892f24c5975859adf572b5cb879e99367a166
OUTD=reports/equipment_v2/lung; mkdir -p $OUTD
HF=reports/equipment_v2/test_plan_lung.sha256
echo "[$(date +%H:%M)] 학습 끝나기를 기다림"
for i in $(seq 1 120); do
  .venv/bin/python src/fetch_results.py --pattern '^(l-s4[234]|b512-s4[34])$' > $SC/fetch.log 2>&1
  n=$(ls -d runs/results/l-s4[234]-* runs/results/b512-s4[34]-* 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -ge 5 ] && break
  sleep 60
done
echo "[$(date +%H:%M)] 결과 $n 개 받음"
[ "$n" -ge 5 ] || { echo "5개가 다 안 받아짐: $(cat $SC/fetch.log | sed -E 's/[0-9]{12}/<ACCOUNT>/g' | tail -3)"; exit 1; }
md() { ls -d runs/results/$1-* | head -1; }
evalval() {  # 이름 종류(lung|direct)
  local n="$1" kind="$2" M="$(md $1)/model"
  if [ "$kind" = lung ]; then
    .venv/bin/python src/equipment_eval_lung.py --masks-dir runs/lung_masks --model-dir "$M" --out $OUTD/val_$n --device mps --label $n > $OUTD/eval_val_$n.log 2>&1
  else
    .venv/bin/python src/equipment_eval_direct.py --model-dir "$M" --out $OUTD/val_$n --device mps --label $n > $OUTD/eval_val_$n.log 2>&1
  fi
  echo "[$(date +%H:%M)] val 끝: $n"
}
evaltest() {
  local n="$1" kind="$2" M="$(md $1)/model"
  { echo "# 평가 직전 덧붙임 ($(date '+%Y-%m-%d %H:%M'), $n 의 test 점수를 보기 전)"; shasum -a 256 "$M/best_model.pt" "$M/metadata.json"; } >> $HF
  if [ "$kind" = lung ]; then
    .venv/bin/python src/equipment_eval_lung.py --masks-dir runs/lung_masks --model-dir "$M" --out $OUTD/test_$n --device mps --label $n --split test --plan-file reports/equipment_v2/test_plan_lung.md --plan-sha256 $H --hash-file $HF > $OUTD/eval_test_$n.log 2>&1
  else
    .venv/bin/python src/equipment_eval_direct.py --model-dir "$M" --out $OUTD/test_$n --device mps --label $n --split test --plan-file reports/equipment_v2/test_plan_lung.md --plan-sha256 $H --hash-file $HF > $OUTD/eval_test_$n.log 2>&1
  fi
  echo "[$(date +%H:%M)] test 끝: $n"
}
( for p in "l-s42 lung" "l-s43 lung" "l-s44 lung" "b512-s43 direct" "b512-s44 direct"; do evalval $p; done ) &
(
  for i in $(seq 1 60); do grep -q "저장 완료" $SC/xrv_out/mask_test.log 2>/dev/null && break; sleep 10; done
  cp $SC/xrv_out/test_lung_masks.npz runs/lung_masks/test_lung_masks.npz
  for p in "l-s42 lung" "l-s43 lung" "l-s44 lung" "b512-s43 direct" "b512-s44 direct"; do evaltest $p; done
) &
wait
echo "[$(date +%H:%M)] 모두 끝"
