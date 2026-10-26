# AWS 종료 기록 (읽기 전용 조회)

- 조회 시각: 2026-10-26 10:04 KST (서울 리전 ap-northeast-2)
- 방법: 학생 계정으로 조회 API만 호출했다 (생성·삭제·수정 없음). 학생 ID는 `<학생ID>`, 12자리 계정 번호는 `<ACCOUNT>`로 가렸다.
- 이 계정은 다른 수강생과 함께 쓰는 공용 계정이라, 아래는 이름에 `p1-cxr-<학생ID>`가 들어간 내 자원만 센 것이다.
- 만든 스크립트: `tools/aws_teardown_check.py`

## 1. 비용이 나가는 자원이 남아 있는가

| 자원 | 남은 개수 | 판정 |
|---|---|---|
| SageMaker Endpoint | 0 | 삭제됨 |
| SageMaker Endpoint 구성 | 0 | 삭제됨 |
| SageMaker 모델 | 0 | 삭제됨 |
| SageMaker 노트북 인스턴스 | 0 | 없음 |
| Lambda 함수 | 0 | 삭제됨 |
| EC2 인스턴스 (종료되지 않은 것) | 0 | 전부 종료됨 |
| EBS 볼륨 | 0 | 없음 |
| 보안 그룹 | 0 | 삭제됨 |
| 키 페어 | 0 | 삭제됨 |
| 탄력적 IP | 0 | 없음 |
| 진행 중인 학습 작업 | 0 | 없음 |

EC2 인스턴스 상태 분포(이름 태그 기준): 기록 없음 (종료된 인스턴스는 한 시간쯤 뒤 목록에서 사라진다)

## 2. 남아 있는 것 (비용은 거의 없음)

- IAM 역할 `p1-cxr-<학생ID>-lambda`: 존재함 (인라인 정책만 있음, 비용 없음). 이 계정에는 삭제 권한이 없어 지우지 못했다 → 필요하면 강사에게 삭제를 요청한다.
- IAM 역할 `p1-cxr-<학생ID>-web`: 존재함 (인라인 정책만 있음, 비용 없음). 이 계정에는 삭제 권한이 없어 지우지 못했다 → 필요하면 강사에게 삭제를 요청한다.
- IAM 인스턴스 프로필 `p1-cxr-<학생ID>-web`: 존재함
- S3 `s3://<버킷>/students/<학생ID>/P1/cxr/deployment/b512/model_with_code.tar.gz` (42 MB, 보관 비용은 월 몇 센트 수준) — 배포용 모델 묶음. 다시 배포할 때 쓰려고 남겨 두었다.

## 3. SageMaker 학습 작업과 청구 시간

이름에 `p1-cxr-<학생ID>`가 들어간 학습 작업 43개를 조회했다. 청구 시간은 AWS가 기록한 `BillableTimeInSeconds`다.

| 실험 묶음 | 작업 수 | 상태 | 인스턴스 | 청구 시간(분) |
|---|---|---|---|---|
| smoke | 4 | Completed 4 | ml.g4dn.xlarge 4 | 16 |
| a | 1 | Completed 1 | ml.g4dn.xlarge 1 | 22 |
| envcheck | 1 | Completed 1 | ml.g4dn.xlarge 1 | 4 |
| b | 1 | Completed 1 | ml.g4dn.xlarge 1 | 22 |
| d | 3 | Completed 3 | ml.g4dn.xlarge 1, ml.g4dn.2xlarge 2 | 47 |
| c | 1 | Completed 1 | ml.g4dn.xlarge 1 | 22 |
| p1 | 1 | Completed 1 | ml.g4dn.xlarge 1 | 19 |
| p2 | 1 | Completed 1 | ml.g4dn.xlarge 1 | 20 |
| p3 | 1 | Completed 1 | ml.g4dn.xlarge 1 | 19 |
| p4 | 1 | Completed 1 | ml.g4dn.xlarge 1 | 20 |
| testeval | 2 | Completed 2 | ml.g4dn.xlarge 2 | 10 |
| b448 | 1 | Completed 1 | ml.g4dn.2xlarge 1 | 16 |
| b512 | 3 | Completed 3 | ml.g4dn.2xlarge 3 | 54 |
| shifteval | 1 | Completed 1 | ml.g4dn.2xlarge 1 | 10 |
| r (과적합 줄이기, 취소) | 15 | Stopped 15 | ml.g4dn.2xlarge 15 | 67 |
| l | 3 | Completed 3 | ml.g4dn.2xlarge 3 | 51 |
| lv | 3 | Completed 3 | ml.g4dn.2xlarge 3 | 53 |

인스턴스 종류별 청구 시간 합계:

| 인스턴스 | 청구 시간(시간) |
|---|---|
| ml.g4dn.2xlarge | 4.61 |
| ml.g4dn.xlarge | 3.23 |
| 합계 | 7.83 |

단가는 AWS 요금표를 확인하지 못해 금액으로 바꾸지 않았다. 위 시간에 해당 리전의 시간당 요금을 곱하면 학습 비용이 된다. JupyterLab(Studio) 공간에서 돌린 모델 6개(기준 b 3개, 장비 효과 증강 v 3개)는 학습 작업이 아니라서 위 표에 없다. 학습 기록(`experiment_summary.json`)의 학습 시간 합계가 10,187초(약 2.83시간)이고, 인스턴스는 ml.g4dn.xlarge였다. 학습이 끝난 뒤 공간의 앱은 삭제했다 (공간이 켜져 있던 시간은 학습 시간보다 길 수 있다).

