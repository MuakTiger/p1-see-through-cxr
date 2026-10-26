# split 분할 결과

## 1. 사용한 규칙

- test 624장: 원래 test 그대로 `split=test`. 건드리지 않음.
- 원래 val 16장: `split=check` (화면 점검용). 학습·검증 풀에 넣지 않음.
- 원래 train 5,216장만 환자 그룹 단위 + 클래스(label_id) 층화로 5등분하고 첫 조각을 val 로 사용 (scikit-learn `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, 약 20%). seed 를 바꿔 가며 고르지 않았다.
- 같은 `patient_group` 은 같은 쪽. sha256 이 같은 파일도 patient_group 이 달라도 같은 쪽이 되도록, 풀 안에서 patient_group 과 sha256 중복을 union-find 로 묶어 `split_group`(`G:` + 묶음 안 사전순 첫 patient_group)을 만들고 그 단위로 나눔.
- 그룹 안에 라벨이 섞이면 최빈값을 그 그룹의 대표 라벨로 층화에 사용 (동수이면 label_id 가 작은 쪽).
- 원래 train·test 양쪽에 걸친 그룹: test 는 고정이므로 그 그룹의 train 쪽 사진은 다른 그룹과 똑같이 train/val 풀에 포함 (목록은 4장).
- check·test 행의 `split_group` 은 합치지 않고 `check:`/`test:` + patient_group 으로 표기.

## 2. 분할 결과

| split | 라벨 | 영상 수 | patient_group 수 | split_group 수 | split 내 비율 |
|---|---|---|---|---|---|
| train | NORMAL | 1075 | 969 | 969 | 25.7% |
| train | BACTERIAL_PNEUMONIA | 2042 | 1090 | 1090 | 48.8% |
| train | VIRAL_PNEUMONIA | 1069 | 864 | 864 | 25.5% |
| train | ALL | 4186 | 2277 | 2277 | 100.0% |
| val | NORMAL | 266 | 242 | 242 | 25.8% |
| val | BACTERIAL_PNEUMONIA | 488 | 263 | 263 | 47.4% |
| val | VIRAL_PNEUMONIA | 276 | 226 | 226 | 26.8% |
| val | ALL | 1030 | 569 | 569 | 100.0% |
| check | NORMAL | 8 | 8 | 8 | 50.0% |
| check | BACTERIAL_PNEUMONIA | 8 | 7 | 7 | 50.0% |
| check | VIRAL_PNEUMONIA | 0 | 0 | 0 | 0.0% |
| check | ALL | 16 | 15 | 15 | 100.0% |
| test | NORMAL | 234 | 225 | 225 | 37.5% |
| test | BACTERIAL_PNEUMONIA | 242 | 77 | 77 | 38.8% |
| test | VIRAL_PNEUMONIA | 148 | 126 | 126 | 23.7% |
| test | ALL | 624 | 427 | 427 | 100.0% |

전체: train 4186 / val 1030 / check 16 / test 624 = 5856

### 클래스별 val 비율 (val ÷ (train+val), 영상 기준)

| 라벨 | train | val | val 비율 | 18~22% 범위 |
|---|---|---|---|---|
| NORMAL | 1075 | 266 | 19.8% | 안 |
| BACTERIAL_PNEUMONIA | 2042 | 488 | 19.3% | 안 |
| VIRAL_PNEUMONIA | 1069 | 276 | 20.5% | 안 |

전체 val 비율: 19.7%

### 그룹 내 라벨 혼재

- 풀 안의 patient_group 2846개 중 라벨이 둘 이상인 그룹: 808개 (예: `P:person1000` 에 세균성 1장 + 바이러스성 1장)
- 합친 split_group 2846개 중 라벨이 섞인 것: 808개 (그중 최빈값이 동수인 것 484개)
- 대표 라벨 분포 (split_group 기준): NORMAL 1211, BACTERIAL_PNEUMONIA 1286, VIRAL_PNEUMONIA 349
- 층화에는 대표 라벨을 쓰지만, 위 표의 영상 수·비율은 모두 영상 자체의 실제 라벨로 센 값이다.
- 폐렴 번호 `personN` 하나에 세균성·바이러스성이 함께 있는 경우가 많아, 이 번호가 실제 한 사람인지 세균/바이러스 파일에 따로 매긴 번호인지는 확인할 수 없다 (5장 한계 참고).
- sha256 때문에 서로 다른 patient_group 이 합쳐진 횟수: 0회

## 3. 검증 결과

- [x] train+val 합이 5,216장
- [x] check 16장
- [x] test 624장
- [x] 전체 5,856장
- [x] 빈 split 값 0건
- [x] train·val 양쪽에 있는 patient_group 0개
- [x] train·val 양쪽에 있는 split_group 0개
- [x] 같은 sha256 파일이 서로 다른 split 에 있는 경우 0건
- [x] test 파일 집합이 원래 test 와 정확히 같음
- [x] check 파일 집합이 원래 val 과 정확히 같음
- [x] train+val 파일 집합이 원래 train 과 정확히 같음
- [x] 클래스별 val 비율이 18~22% 범위 안 (정보용)

- 정보: check 16장의 patient_group 15개 중 train/val 풀과 겹치는 것 0개 (train 0개, val 0개). 겹쳐도 실패로 보지 않는다.
- 정보: test 안의 `NORMAL2-IM-0095-0001` 과 `NORMAL2-IM-0096-0001` 은 같은 이미지인데 patient_group 이 다르다. 둘 다 test 이므로 영향 없음.

## 4. 원래 train·test 에 걸친 그룹 (170개)

- train 쪽 사진이 전부 train 으로 배정된 그룹: 141개
- 전부 val 로 배정된 그룹: 29개
- train·val 로 나뉜 그룹: 0개 (sha256 로 묶이지 않는 한 0이어야 함)
- 이 그룹들의 test 쪽 사진은 그대로 test 다. train 쪽 사진은 다른 그룹과 똑같이 취급했다.

| patient_group | train 쪽 장수 | test 쪽 장수 | 배정된 분할 | split_group |
|---|---|---|---|---|
| P:person1 | 2 | 7 | train 2 | G:P:person1 |
| P:person10 | 1 | 1 | train 1 | G:P:person10 |
| P:person100 | 1 | 7 | train 1 | G:P:person100 |
| P:person101 | 2 | 4 | train 2 | G:P:person101 |
| P:person102 | 1 | 1 | train 1 | G:P:person102 |
| P:person103 | 1 | 3 | train 1 | G:P:person103 |
| P:person104 | 1 | 2 | train 1 | G:P:person104 |
| P:person108 | 3 | 4 | val 3 | G:P:person108 |
| P:person109 | 1 | 9 | train 1 | G:P:person109 |
| P:person11 | 1 | 1 | train 1 | G:P:person11 |
| P:person110 | 4 | 1 | train 4 | G:P:person110 |
| P:person111 | 3 | 5 | train 3 | G:P:person111 |
| P:person112 | 1 | 2 | train 1 | G:P:person112 |
| P:person113 | 2 | 4 | train 2 | G:P:person113 |
| P:person114 | 1 | 3 | train 1 | G:P:person114 |
| P:person117 | 1 | 3 | val 1 | G:P:person117 |
| P:person118 | 1 | 2 | train 1 | G:P:person118 |
| P:person119 | 1 | 4 | val 1 | G:P:person119 |
| P:person120 | 1 | 4 | train 1 | G:P:person120 |
| P:person122 | 1 | 6 | val 1 | G:P:person122 |
| P:person123 | 1 | 1 | train 1 | G:P:person123 |
| P:person124 | 16 | 4 | train 16 | G:P:person124 |
| P:person125 | 1 | 2 | train 1 | G:P:person125 |
| P:person126 | 1 | 3 | train 1 | G:P:person126 |
| P:person128 | 1 | 4 | train 1 | G:P:person128 |
| P:person130 | 1 | 5 | train 1 | G:P:person130 |
| P:person131 | 1 | 1 | train 1 | G:P:person131 |
| P:person132 | 1 | 1 | train 1 | G:P:person132 |
| P:person133 | 1 | 5 | train 1 | G:P:person133 |
| P:person134 | 1 | 5 | train 1 | G:P:person134 |
| P:person135 | 2 | 2 | train 2 | G:P:person135 |
| P:person137 | 1 | 1 | train 1 | G:P:person137 |
| P:person138 | 1 | 3 | train 1 | G:P:person138 |
| P:person139 | 1 | 6 | val 1 | G:P:person139 |
| P:person14 | 1 | 1 | val 1 | G:P:person14 |
| P:person140 | 2 | 2 | train 2 | G:P:person140 |
| P:person141 | 1 | 5 | train 1 | G:P:person141 |
| P:person142 | 1 | 3 | train 1 | G:P:person142 |
| P:person143 | 1 | 3 | train 1 | G:P:person143 |
| P:person145 | 2 | 1 | train 2 | G:P:person145 |
| P:person146 | 1 | 3 | train 1 | G:P:person146 |
| P:person147 | 1 | 4 | val 1 | G:P:person147 |
| P:person149 | 1 | 1 | train 1 | G:P:person149 |
| P:person15 | 1 | 1 | train 1 | G:P:person15 |
| P:person151 | 2 | 1 | val 2 | G:P:person151 |
| P:person152 | 1 | 5 | train 1 | G:P:person152 |
| P:person153 | 1 | 2 | train 1 | G:P:person153 |
| P:person154 | 2 | 1 | train 2 | G:P:person154 |
| P:person155 | 1 | 3 | val 1 | G:P:person155 |
| P:person157 | 1 | 3 | train 1 | G:P:person157 |
| P:person158 | 1 | 4 | val 1 | G:P:person158 |
| P:person16 | 3 | 1 | train 3 | G:P:person16 |
| P:person1608 | 1 | 1 | train 1 | G:P:person1608 |
| P:person161 | 1 | 3 | train 1 | G:P:person161 |
| P:person1610 | 1 | 1 | train 1 | G:P:person1610 |
| P:person1613 | 1 | 1 | train 1 | G:P:person1613 |
| P:person1614 | 1 | 1 | train 1 | G:P:person1614 |
| P:person1615 | 1 | 1 | val 1 | G:P:person1615 |
| P:person1616 | 1 | 1 | train 1 | G:P:person1616 |
| P:person1618 | 1 | 1 | train 1 | G:P:person1618 |
| P:person1619 | 8 | 1 | train 8 | G:P:person1619 |
| P:person1620 | 1 | 1 | train 1 | G:P:person1620 |
| P:person1625 | 1 | 1 | train 1 | G:P:person1625 |
| P:person1626 | 1 | 1 | train 1 | G:P:person1626 |
| P:person1628 | 6 | 2 | train 6 | G:P:person1628 |
| P:person1629 | 1 | 1 | val 1 | G:P:person1629 |
| P:person1634 | 3 | 1 | val 3 | G:P:person1634 |
| P:person1635 | 1 | 1 | train 1 | G:P:person1635 |
| P:person1637 | 1 | 1 | train 1 | G:P:person1637 |
| P:person1640 | 1 | 1 | val 1 | G:P:person1640 |
| P:person1641 | 1 | 1 | train 1 | G:P:person1641 |
| P:person1642 | 2 | 1 | train 2 | G:P:person1642 |
| P:person1643 | 1 | 1 | train 1 | G:P:person1643 |
| P:person1644 | 6 | 1 | train 6 | G:P:person1644 |
| P:person1645 | 1 | 1 | train 1 | G:P:person1645 |
| P:person1647 | 1 | 1 | train 1 | G:P:person1647 |
| P:person1649 | 3 | 1 | train 3 | G:P:person1649 |
| P:person1651 | 1 | 1 | train 1 | G:P:person1651 |
| P:person1657 | 3 | 1 | val 3 | G:P:person1657 |
| P:person1660 | 1 | 1 | train 1 | G:P:person1660 |
| P:person1661 | 1 | 2 | val 1 | G:P:person1661 |
| P:person1663 | 2 | 1 | train 2 | G:P:person1663 |
| P:person1665 | 1 | 1 | train 1 | G:P:person1665 |
| P:person1667 | 2 | 1 | train 2 | G:P:person1667 |
| P:person1668 | 2 | 1 | train 2 | G:P:person1668 |
| P:person1669 | 3 | 2 | val 3 | G:P:person1669 |
| P:person1670 | 7 | 1 | train 7 | G:P:person1670 |
| P:person1671 | 1 | 1 | train 1 | G:P:person1671 |
| P:person1673 | 1 | 1 | train 1 | G:P:person1673 |
| P:person1674 | 1 | 1 | train 1 | G:P:person1674 |
| P:person1676 | 1 | 1 | train 1 | G:P:person1676 |
| P:person1678 | 1 | 1 | val 1 | G:P:person1678 |
| P:person1679 | 5 | 1 | train 5 | G:P:person1679 |
| P:person1680 | 1 | 1 | train 1 | G:P:person1680 |
| P:person1682 | 1 | 1 | train 1 | G:P:person1682 |
| P:person1685 | 1 | 1 | train 1 | G:P:person1685 |
| P:person17 | 1 | 1 | train 1 | G:P:person17 |
| P:person18 | 1 | 1 | train 1 | G:P:person18 |
| P:person19 | 6 | 1 | train 6 | G:P:person19 |
| P:person20 | 5 | 1 | train 5 | G:P:person20 |
| P:person21 | 2 | 2 | train 2 | G:P:person21 |
| P:person22 | 3 | 2 | val 3 | G:P:person22 |
| P:person23 | 30 | 1 | train 30 | G:P:person23 |
| P:person24 | 5 | 1 | train 5 | G:P:person24 |
| P:person25 | 9 | 1 | train 9 | G:P:person25 |
| P:person26 | 11 | 1 | train 11 | G:P:person26 |
| P:person28 | 4 | 2 | train 4 | G:P:person28 |
| P:person29 | 1 | 1 | val 1 | G:P:person29 |
| P:person3 | 4 | 3 | train 4 | G:P:person3 |
| P:person30 | 14 | 1 | train 14 | G:P:person30 |
| P:person31 | 5 | 1 | train 5 | G:P:person31 |
| P:person32 | 2 | 1 | train 2 | G:P:person32 |
| P:person33 | 5 | 1 | train 5 | G:P:person33 |
| P:person34 | 1 | 1 | train 1 | G:P:person34 |
| P:person35 | 3 | 1 | train 3 | G:P:person35 |
| P:person36 | 4 | 1 | train 4 | G:P:person36 |
| P:person37 | 4 | 1 | train 4 | G:P:person37 |
| P:person38 | 7 | 2 | train 7 | G:P:person38 |
| P:person39 | 2 | 1 | val 2 | G:P:person39 |
| P:person40 | 4 | 1 | val 4 | G:P:person40 |
| P:person41 | 6 | 1 | train 6 | G:P:person41 |
| P:person43 | 2 | 1 | train 2 | G:P:person43 |
| P:person44 | 2 | 2 | train 2 | G:P:person44 |
| P:person45 | 3 | 1 | train 3 | G:P:person45 |
| P:person46 | 2 | 1 | train 2 | G:P:person46 |
| P:person47 | 1 | 1 | train 1 | G:P:person47 |
| P:person48 | 4 | 1 | train 4 | G:P:person48 |
| P:person49 | 3 | 1 | train 3 | G:P:person49 |
| P:person50 | 1 | 1 | train 1 | G:P:person50 |
| P:person51 | 10 | 1 | train 10 | G:P:person51 |
| P:person52 | 2 | 1 | train 2 | G:P:person52 |
| P:person53 | 4 | 2 | train 4 | G:P:person53 |
| P:person54 | 2 | 1 | train 2 | G:P:person54 |
| P:person55 | 7 | 1 | train 7 | G:P:person55 |
| P:person56 | 3 | 1 | train 3 | G:P:person56 |
| P:person57 | 2 | 1 | train 2 | G:P:person57 |
| P:person59 | 6 | 1 | train 6 | G:P:person59 |
| P:person60 | 3 | 1 | val 3 | G:P:person60 |
| P:person61 | 10 | 1 | train 10 | G:P:person61 |
| P:person62 | 6 | 1 | val 6 | G:P:person62 |
| P:person63 | 1 | 1 | train 1 | G:P:person63 |
| P:person64 | 7 | 1 | train 7 | G:P:person64 |
| P:person65 | 1 | 1 | train 1 | G:P:person65 |
| P:person66 | 4 | 1 | val 4 | G:P:person66 |
| P:person67 | 7 | 1 | train 7 | G:P:person67 |
| P:person69 | 1 | 1 | train 1 | G:P:person69 |
| P:person70 | 6 | 1 | train 6 | G:P:person70 |
| P:person71 | 5 | 2 | train 5 | G:P:person71 |
| P:person72 | 3 | 1 | train 3 | G:P:person72 |
| P:person74 | 3 | 1 | train 3 | G:P:person74 |
| P:person75 | 6 | 1 | val 6 | G:P:person75 |
| P:person76 | 3 | 1 | train 3 | G:P:person76 |
| P:person77 | 4 | 1 | train 4 | G:P:person77 |
| P:person8 | 1 | 2 | train 1 | G:P:person8 |
| P:person80 | 1 | 5 | train 1 | G:P:person80 |
| P:person81 | 2 | 4 | train 2 | G:P:person81 |
| P:person82 | 2 | 4 | train 2 | G:P:person82 |
| P:person83 | 1 | 6 | train 1 | G:P:person83 |
| P:person86 | 1 | 2 | val 1 | G:P:person86 |
| P:person87 | 1 | 2 | train 1 | G:P:person87 |
| P:person88 | 6 | 3 | train 6 | G:P:person88 |
| P:person89 | 1 | 1 | train 1 | G:P:person89 |
| P:person90 | 2 | 2 | val 2 | G:P:person90 |
| P:person92 | 1 | 2 | train 1 | G:P:person92 |
| P:person93 | 1 | 2 | train 1 | G:P:person93 |
| P:person94 | 1 | 3 | train 1 | G:P:person94 |
| P:person95 | 1 | 1 | train 1 | G:P:person95 |
| P:person96 | 2 | 3 | val 2 | G:P:person96 |
| P:person97 | 2 | 1 | train 2 | G:P:person97 |
| P:person99 | 1 | 2 | train 1 | G:P:person99 |

## 5. 한계

- 파일명 번호가 같은 환자를 뜻하는지는 원천 설명으로 확인되지 않았다. 번호가 같으면 같은 사람이라고 가정하고 같은 쪽에 넣었을 뿐이다.
- 원래 train·test 에 걸친 그룹은 같은 환자일 수도, 분할마다 번호를 따로 매긴 것일 수도 있다. 같은 환자라면 train 에서 본 환자가 test 에도 있어 test 성능이 실제보다 좋게 나올 수 있다.
- 정상(N)과 폐렴(P)은 번호 체계가 같다고 가정하지 않아, 같은 사람이 두 라벨에 모두 있어도 찾을 수 없다.
- sha256 이 같지 않은 거의 같은 사진(다시 저장·크기 변경 등)은 찾지 못한다.
- 그룹 단위 분할이라 클래스별 val 비율이 정확히 20%는 아니다.
- 원래 val(check) 16장은 바이러스성이 0장이라 3분류 점검에는 한계가 있다.
