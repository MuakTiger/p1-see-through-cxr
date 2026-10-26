#!/bin/bash
# 메모리 부족(스왑 18.9/19.5GB)으로 일시 정지한 lv 평가 2개를, 폐 모델 평가 줄기가 끝나는 대로 차례로 이어서 돌린다 (동시에 2개만 돌도록).
LOG=<임시 폴더>
until grep -q "test 끝: b512-s44" $LOG; do sleep 20; done
kill -CONT 48680; echo "[$(date +%H:%M)] lv test 재개"
until grep -q "모두 끝" $LOG; do sleep 20; done
kill -CONT 48677; echo "[$(date +%H:%M)] lv val 재개"
