#!/bin/bash
# 폐 영역 계열(l, lv)과 새 기준(b512-s43·44)의 val·test 평가를 이어서 한다. 이미 끝난 것(shift_results.csv 가 있는 것)은 건너뛴다.
# 메모리(16GB) 때문에 동시에 2줄기(val 줄기, test 줄기)만 돌린다. 평가가 이미 돌고 있으면(pgrep) 멈춘다.
# 사용 (p1 폴더에서):  bash src/resume_lung_evals.sh        끝난 뒤:  .venv/bin/python src/lung_summary.py
cd "$(dirname "$0")/.."
if pgrep -f "equipment_eval_(lung|direct).py" > /dev/null; then echo "평가가 이미 돌고 있습니다. 끝난 뒤에 다시 실행하세요."; exit 1; fi
OUTD=reports/equipment_v2/lung; mkdir -p $OUTD
.venv/bin/python src/fetch_results.py --pattern '^(l-s4[234]|b512-s4[34]|lv-s4[234])$' | tail -1
planhash() { grep -E "reports/equipment_v2/$1\$" reports/equipment_v2/$2 | head -1 | cut -d' ' -f1; }   # 고정 해시 기록에서 읽는다
md() { ls -d runs/results/$1-* 2>/dev/null | head -1; }
run_eval() {  # 이름 split
  local n="$1" sp="$2" M; M="$(md $n)/model"
  [ -f "$OUTD/${sp}_$n/shift_results.csv" ] && { echo "건너뜀(이미 끝남): $sp $n"; return; }
  [ -d "$M" ] || { echo "모델이 없어 건너뜀: $n"; return; }
  local script=src/equipment_eval_lung.py extra=(--masks-dir runs/lung_masks)
  case "$n" in b512*) script=src/equipment_eval_direct.py; extra=();; esac
  local args=()
  if [ "$sp" = test ]; then
    local plan=test_plan_lung.md rec=test_plan_lung.sha256
    case "$n" in lv*) plan=test_plan_lung_vaccine.md; rec=test_plan_lung_vaccine.sha256;; esac
    { echo "# 평가 직전 덧붙임 ($(date '+%Y-%m-%d %H:%M'), $n 의 test 를 이어서 평가, resume_lung_evals.sh)"; shasum -a 256 "$M/best_model.pt" "$M/metadata.json"; } >> reports/equipment_v2/$rec
    args=(--split test --plan-file reports/equipment_v2/$plan --plan-sha256 "$(planhash $plan $rec)" --hash-file reports/equipment_v2/$rec)
  fi
  echo "[$(date +%H:%M)] 평가 시작: $sp $n"
  .venv/bin/python $script "${extra[@]}" --model-dir "$M" --out $OUTD/${sp}_$n --device mps --label $n "${args[@]}" > $OUTD/eval_${sp}_$n.log 2>&1
  echo "[$(date +%H:%M)] 평가 끝: $sp $n"
}
MODELS=(l-s42 l-s43 l-s44 b512-s43 b512-s44 lv-s42 lv-s43 lv-s44)
( for n in "${MODELS[@]}"; do run_eval $n val; done ) &
( for n in "${MODELS[@]}"; do run_eval $n test; done ) &
wait
echo "[$(date +%H:%M)] 모두 끝. 요약: .venv/bin/python src/lung_summary.py"
