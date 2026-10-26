10/22 밤 작업에서 임시 폴더(scratchpad)에 있던 스크립트의 보관본. 경로가 스크래치패드(/private/tmp/...)로 적혀 있는 것은 새 환경에서 고쳐 써야 한다.
  xrv_mask_split.py   PSPNet(TorchXRayVision) 폐 마스크 생성 (임시 가상환경 xrv_venv 에서 실행: python -m venv xrv_venv && xrv_venv/bin/pip install torchxrayvision; MPS)
  lung_gradcam.py / lung_overlay.py / lung_focus_val.py / cam_check512.py / cam_region_val.py   Grad-CAM 분석 (폐 영역 모델, 폐 안 비율, 가리기 시험)
  seg_failures.py / seg_stats.py   폐 분할 실패·의심 사례 그림과 통계
  order_check.py   필터 → 마스크 대 마스크 → 필터 순서 점검
  aspect_sweep.py / res_sweep.py   가로세로 비율·해상도 손실 훑기 (val만)
  make_train_lung.py   training/train_aug.py → training/train_lung.py 생성 패치 (train_lung.py 는 이후 load_hulls 한 줄을 직접 고침)
  test_train_lung.py   학습 입력 = 평가 입력 검증
  lung_pipeline.sh / lv_pipeline.sh / lv_resume.sh / submit_lung_wave.sh   제출·평가 자동화 (이어서 하기는 src/resume_lung_evals.sh)
