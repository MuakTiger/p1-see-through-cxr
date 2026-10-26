# 실험 D 최종 test 평가 (evaluate.py · submit_evaluation.py) 리뷰

리뷰: 2026-10-19 17:27 KST. 대상은 `training/evaluate.py`, `submit_evaluation.py`, `reports/test_eval/` 산출물과 `test_report.md`다. 기준은 사전 고정 계획 `evaluation_plan.md`와 가이드 v3 12장이다. AWS 작업과 bootstrap은 다시 돌리지 않았다. 대신 `test_predictions.csv`(aws/, local/)로 지표를 직접 다시 계산했고, 해시·개수·보안 검사도 직접 했다.

계획 준수는 확인됐다. 계획 파일, 분할 파일, D 가중치의 sha256이 계획과 같다. 제외 6장도 계획 목록과 같고, 618/624 세트의 클래스별 수(231/240/147, 234/242/148)와 환자 그룹 수(426/427)도 맞다. Macro-F1, 혼동행렬, 클래스별 precision·recall은 `test_predictions.csv`로 다시 계산해도 metrics.csv와 6자리까지 같다. AWS와 iMac 예측은 618/618 일치한다. bootstrap은 환자 그룹 단위다. 남은 문제는 두 가지다. 첫째, AUROC는 `test_predictions.csv`로 다시 계산하면 보고 값이 나오지 않는다. 둘째, 보고서 3장의 정상 오분류 해석이 실제 확률 분포와 반대다.

Watch for: 정상 AUROC가 보고서 0.937 대 CSV 재계산 0.919로 다르고, Macro AUROC도 0.908 대 0.902다 (confirmed). 정상→폐렴 129장 대부분은 경계 근처가 아니라 강한 확신으로 틀린 사진인데, 보고서는 반대로 읽힌다 (confirmed).

**Verdict**: NEEDS_CHANGES

## High-level view

계획 고정 장치는 촘촘하다. evaluate.py는 계획 해시, 분할 해시, 주 모델 해시, 제외 목록, 세트별 개수를 실행 중에 고정 값과 비교하고, 하나라도 다르면 멈춘다. 다시 계산해 봐도 모두 계획과 같았다. 주 모델 D와 참고 모델 B·C·P1~P4의 가중치는 학습 결과 폴더의 원본과 sha256이 같다. 모델마다 metadata의 preprocess(P0/P1~P4)에 맞춰 `train.build_transform`을 불러 쓰므로 전처리 정의가 학습 때와 하나로 묶여 있다.

분류 지표 재현은 통과했다. AUROC 재현은 실패했다. 원인은 `test_predictions.csv`가 확률을 소수 8자리로 반올림해 저장한다는 점이다. 618장 중 363장의 p_normal이 `0.00000000`으로 같아지고, 이 동점 때문에 CSV로 계산한 정상 AUROC가 0.019 낮게 나온다. 반대로 보면 보고된 정상 AUROC 0.937은 1e-8 아래 확률들의 순서에 기대고 있다는 뜻이다.

보고서는 숫자 대부분이 산출물과 맞고, 한계 절도 충실하다. 문제는 3장 한 문장이다. 정상 사진 상당수가 "argmax 경계를 넘을 만큼 차이가 크지 않다"고 해석했지만, 실제로는 틀린 정상 129장 중 96장의 p_normal이 0.01 미만이다. 이 프로젝트의 확신도 기반 판단 보류(τ)와 직접 이어지는 내용이라 고쳐야 한다.

보안 노출은 없다. 역할 ARN, 계정 번호, 이미지 주소가 산출물·문서·코드·제출 기록 어디에도 없다. 제출 기록에는 해시만 남는다.

<details>
<summary>Issues (6)</summary>

1. **AUROC를 test_predictions.csv로 재현할 수 없음 (blocking)**: 확률을 8자리로 반올림해 저장한 탓에 정상 AUROC가 0.937 대 0.919, Macro AUROC가 0.908 대 0.902로 다르다. 가이드 12장 완료 기준에 어긋난다. 확률을 전체 정밀도(`.17g` 등)로 저장하거나 logit 열을 함께 저장한 뒤 다시 실행해 CSV를 다시 만든다. 재실행 사실과 이유는 test_report.md에 남긴다 (계획 머리말 규칙).
2. **정상 오분류 해석이 숫자와 반대 (blocking)**: 3장의 "그 차이가 argmax 경계를 넘을 만큼 크지 않은 사진이 많다"는 틀렸다. 틀린 정상 129장 중 96장은 p_normal < 0.01이고, 예측 확률 중앙값은 0.98이다. 확신 높은 오류라는 점과 τ 보류로 걸러지지 않는다는 점을 사실대로 고쳐 쓴다. 정상 AUROC가 1e-8 아래 확률들의 순서에 기댄다는 점도 덧붙인다.
3. **"연 영상" 기록이 실제 열기를 기록하지 않음**: `opened`는 `__getitem__`이 실제로 연 경로가 아니라 test_rows를 그대로 복사한 목록이라, `opened_images_all_test` 확인은 항상 참이 된다. 실제 보장은 manifest 필터와 TestDataset 검사에서 온다. docstring과 보고서 문구를 그에 맞게 고치거나, Dataset 안에서 기록한다.
4. **5-2 보조 지표가 결과를 본 뒤 추가된 것임을 표시하지 않음**: "폐렴 사진만의 정답 비율"은 iMac에서 test 결과를 본 뒤 추가됐다. 1장 기록에는 있지만 5-2 본문에는 표시가 없다. 5-2 표 옆에 "계획 밖, 결과를 본 뒤 추가"라고 적는다.
5. **모델 A 가중치 출처를 로컬에서 추적할 수 없음**: `runs/results/a-…/`에는 model/ 폴더가 없어서 `test_eval_models/A/best_model.pt`(sha256 7c3a14d1…)를 학습 산출물과 대조할 수 없었다. metadata의 선택 에폭 10과 val F1 0.8058은 experiment_summary와 같다. 보고서에 A를 가져온 S3 경로를 적는다 (참고 모델이라 비차단).
6. **iMac 실행이 AWS 주 실행보다 먼저 test 점수를 봄**: 계획 6장은 AWS 평가를 먼저 하고 iMac은 재현용으로 둔다. 실제로는 iMac 실행이 먼저 test 점수를 봤다. 모델·규칙·주 지표는 바꾸지 않았고 1장에 기록돼 있어 비차단이다. 7장 한계에 한 줄 더 적는다.

</details>

<details>
<summary>Details</summary>

## 계획 고정 장치와 세트 구성

evaluate.py 맨 위에 계획 해시, 분할 해시, D 가중치 해시, 제외 6장, 세트별 개수를 상수로 박아 두고 실행 중에 비교한다. 다르면 SystemExit로 멈추게 했으므로, 계획과 다른 조건에서는 점수 자체가 나오지 않는다. 독립 재계산 결과는 다음과 같다. 계획 sha256 `ba614c0a…`, 분할 `205286f3…`, D `a8bb8a96…`(`runs/results/d-…/model/best_model.pt` 원본과 동일). sha256 묶음에서 사전순 첫 파일만 남기면 제외 목록이 계획 2장의 6장과 정확히 같다. 남는 세트는 618장(231/240/147, 환자 426)과 624장(234/242/148, 환자 427)이다. 제출 기록의 상태는 Completed, 과금 290초, evaluate.py sha256 `d2939168…`이고 보고서 값과 같다.

## AUROC 재현 실패와 확률 반올림

`test_predictions.csv`만 가지고 sklearn으로 다시 계산한 값이다 (aws/와 local/ 결과가 같음).

```
618장   metrics.csv   CSV 재계산
정상    0.937414      0.918538
세균성  0.967488      0.967488
바이러스 0.818645      0.818630
Macro   0.907849      0.901552
```

Macro-F1 0.692433, 혼동행렬 [[102,26,103],[0,229,11],[0,35,112]], recall 0.4416/0.9542/0.7619, precision 1.000/0.7897/0.4956, 정확도 0.7168은 metrics.csv·confusion_matrix.csv와 1e-6 안에서 같다. 624장도 마찬가지로 분류 지표는 같고 AUROC만 다르다 (정상 0.9380 대 0.9194).

원인은 저장 형식이다 (confirmed). evaluate.py는 확률을 `f"{v:.8f}"`로 쓴다. 618장 중 363장의 p_normal이 `0.00000000`이 되고, 그중 28장이 실제 정상이다. 컨테이너 안의 sklearn 교차 확인은 전체 정밀도 확률로 했기 때문에 차이 0으로 나왔다. 그래서 컨테이너 밖에서 CSV로 다시 계산하면 같은 값이 나오지 않는다. 가이드 12장 완료 기준은 "test_predictions.csv에서 클래스별 지표가 다시 계산"되는 것이므로 지금 상태로는 기준을 채우지 못한다.

이 현상은 해석상의 사실도 하나 드러낸다. 정상 AUROC 0.937과 0.919의 차이 0.019는 전부 확률 5e-9 아래 사진들의 순서에서 나온다. float32 softmax 순서 자체가 틀린 것은 아니다. 하지만 "정상 사진의 폐렴 확률이 대체로 낮다"는 말을 받쳐 주는 근거로는 약하다. 고칠 때는 확률을 `.17g`로 저장하거나 logit 세 열을 함께 남기면 된다. 예측은 결정적이고 모델 선택도 이미 끝났으므로, 다시 실행해도 test가 선택에 쓰일 위험은 없다. iMac 실행은 약 1분이면 끝난다.

## 정상→폐렴 오류의 확신도

보고서 3장은 정상 AUROC가 높게 남아 있다는 점을 근거로 "그 차이가 argmax 경계를 넘을 만큼 크지 않은 사진이 많다"고 썼다. `test_predictions.csv`로 확인하면 사실은 반대다 (confirmed). 정상인데 폐렴으로 예측된 129장의 p_normal 분위수는 10%·25%·50%가 모두 0.000이고, 75%가 0.013, 90%가 0.171이다. 96장은 p_normal < 0.01, 111장은 < 0.1이다. 이 사진들에서 가장 큰 확률의 중앙값은 0.982다.

모델은 test 정상 사진 절반 이상을 확신을 갖고 폐렴으로 판단했다. 기획안의 판단 보류 규칙(τ 미만이면 보류)으로는 이 오류 대부분이 걸러지지 않는다. 그리고 이것은 보호자용 설명 단계에서 "높게(≥0.70)"라는 문구로 그대로 나간다. 보고서에는 사실대로 적고, 7장 한계에도 "확신도 보류가 test 정상 오분류를 막지 못함"을 넣어야 한다. "test 정상 사진의 특성이 다를 수 있다"는 가설은 검증하지 않은 가설로 이미 적혀 있으니 그대로 두면 된다.

## 환자 단위 bootstrap과 전처리 경로

`bootstrap_ci`와 `bootstrap_scalar`는 `np.unique(groups)`로 patient_group을 뽑는다. 그룹 인덱스를 복원 추출한 다음, 각 그룹의 사진 인덱스를 모두 이어 붙인다 (confirmed). 사진 단위 추출이 아니다. seed를 함수마다 새로 고정하므로 같은 세트라면 모든 모델이 같은 재추출 표본으로 계산된다.

전처리는 모델별 metadata의 `preprocess`로 묶은 뒤 `T.build_transform(pre)`를 부른다. P1~P4 metadata는 각각 P1/P2/P3/P4, A~D는 P0이다 (A는 키가 없어서 기본값 P0을 쓰는데, A는 전처리 옵션이 생기기 전에 학습했으므로 맞다). AWS 로그에도 P0 묶음 [A,B,C,D]와 P1~P4가 따로 예측된 기록이 있다. `build_transform`은 train.py가 val에 쓰는 바로 그 함수라서 학습 때 정의와 어긋날 여지가 없다.

test 외 영상을 열지 않는다는 보장은 두 단계로 이뤄진다. `read_manifest`는 split=="test"이면서 `test/`로 시작하는 행만 남기고, TestDataset이 한 번 더 검사한다. train 행에서는 patient_group 문자열만 읽는다. 다만 `opened` 목록은 `__getitem__`에서 기록하지 않고 test_rows 경로를 그대로 복사한다. 그래서 `opened_images_all_test: True`는 그 자체로는 증거가 되지 않는다 (confirmed). 보장 자체는 성립하므로 비차단이다. 하지만 docstring의 "연 경로를 모두 기록"과 보고서의 "test/ 아래 624장만 열었다"는 위의 두 필터에 근거한다고 써야 정확하다.

## AWS와 iMac 재현

aws/와 local/의 `test_predictions.csv`는 행 순서, 라벨, 그룹, 제외 표시가 같다. 주 모델 예측은 618/618, 624/624 일치하고, 확률 최대 차이는 2.14e-5다. metrics.csv 전체 숫자의 최대 차이는 2.2e-5로, 보고서 6장과 reproducibility.csv 값과 같다. 이 차이가 1e-8 아래 확률 순서까지 바꾸지는 않아서 전체 정밀도 AUROC도 두 환경에서 같았다. 다만 이 점은 위의 AUROC 민감도를 보여 주는 대목이기도 하다.

## 보고서 숫자 대조와 시점 기록

5-3 표의 val·test Macro-F1, 구간, 정상 recall, 순위는 all_models_test.csv와 같다. 순위 상관 −0.5도 Spearman으로 다시 계산해 같았다 (p ≈ 0.21, 모델 8개라 불확실하다는 서술과도 맞음). val 쪽 비교 숫자(정상 recall 0.981, 바이러스→세균 80/276, 세균→바이러스 85/488)는 D의 val_predictions.csv와 같다. 5-2의 묶음 구성(317장: 0/192/125, 환자 170 / 301장: 231/48/22, 환자 256)과 비율도 subgroup_overlap.csv와 같다. 5-3의 "P4가 더 낫다고 말할 수 없다"와 "test를 보고 모델을 바꾸지 않는다"는 서술은 계획에 맞고 과장이 없다.

시점 기록에서 두 가지를 보탤 필요가 있다. 첫째, "폐렴 사진만의 정답 비율"은 iMac에서 test 결과를 처음 본 뒤 추가된 지표다. 코드 주석에는 "계획 밖"이라고 적혀 있지만 5-2 본문에는 표시가 없어, 사전 계획 지표처럼 읽힌다. 둘째, 계획 6장은 AWS 평가를 먼저 하고 iMac은 재현용으로 둔다. 실제로는 iMac 실행이 먼저 test 점수를 봤다. 주 결과 숫자와 규칙은 바뀌지 않았고 1장에 사실이 적혀 있으므로, 7장에 한 줄만 보태면 된다.

## 모델 A 출처

B·C·D·P1~P4는 `runs/test_eval_models/<ID>/best_model.pt`와 metadata.json이 `runs/results/<실험>/model/` 원본과 바이트 단위로 같다. A만 로컬 결과 폴더에 model/이 없어 가중치(sha256 `7c3a14d1…`)를 대조하지 못했다. metadata의 선택 에폭 10과 val Macro-F1 0.80577은 A의 experiment_summary.json과 같아서 같은 작업의 산출물일 가능성이 크다 (likely). 참고 모델이므로 비차단이며, 가져온 S3 경로를 보고서에 적어 두면 된다.

## 보안

aws_config.json의 역할 ARN, ARN 안의 12자리 계정 번호, 역할 이름, 학습 이미지 주소를 reports/test_eval/ 전체(29개 파일), evaluate.py, submit_evaluation.py, src/test_report.py, 제출 기록 JSON에서 찾아봤다. 값은 화면에 출력하지 않았고, 발견 0건이다. 제출 기록에는 역할 ARN과 이미지 주소의 sha256만 남는다. aws_config.json과 runs/는 .gitignore에 포함돼 있다.

</details>

<details>
<summary>File map</summary>

- `training/evaluate.py`: 계획 고정 확인, 중복 제외, 모델 8개 예측, 지표·환자 단위 bootstrap, 산출 7종 저장 (신규, 616줄)
- `submit_evaluation.py`: 모델 업로드·크기 확인, SageMaker 평가 작업 제출·대기·결과 풀기, 해시만 남기는 제출 기록 (신규)
- `reports/test_eval/aws/`, `local/`: 두 환경의 산출 7종과 로그
- `reports/test_eval/test_report.md`, `test_report_tables.md`, `test_confusion.png`, `test_vs_val.png`, `reproducibility.csv`, `evaluation_summary.json`: 결과 문서와 통합본
- 전체 변경은 `git status`/`git diff` 기준 (p1 폴더는 상위 저장소 안이며 아직 커밋되지 않음)

</details>
