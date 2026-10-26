# manifest 점검 결과

## 1. 원래 분할 × 라벨 영상 수 (가이드 표와 비교)

| 분할 | 라벨 | 실제 | 가이드 | 결과 |
|---|---|---|---|---|
| train | NORMAL | 1341 | 1341 | OK |
| train | BACTERIAL_PNEUMONIA | 2530 | 2530 | OK |
| train | VIRAL_PNEUMONIA | 1345 | 1345 | OK |
| val | NORMAL | 8 | 8 | OK |
| val | BACTERIAL_PNEUMONIA | 8 | 8 | OK |
| val | VIRAL_PNEUMONIA | 0 | 0 | OK |
| test | NORMAL | 234 | 234 | OK |
| test | BACTERIAL_PNEUMONIA | 242 | 242 | OK |
| test | VIRAL_PNEUMONIA | 148 | 148 | OK |

전체 영상 수: 5856 (가이드 5,856)

## 2. 라벨 누락 · 읽기 실패

- 라벨 누락: 0건
- 읽기 실패: 0건
- 영상이 아니어서 건너뛴 파일: 0건

## 3. 영상 형식

- 채널 수 분포: {1: 5573, 3: 283}
- 너비 범위: 384 ~ 2916, 높이 범위: 127 ~ 2713

## 4. 환자 그룹

- 그룹 수: 3118 (정상 N: 1444, 폐렴 P: 1674, 규칙 밖 unknown: 0)
- 원래 분할 여러 곳에 걸친 그룹: 170개 (예: P:person1, P:person10, P:person100, P:person101, P:person102)

규칙:
- 폐렴: 파일명 `personN_(bacteria|virus)_M` (끝에 `_K`가 더 붙어도 됨) → 그룹 `P:personN`
- 정상: 파일명 `IM-XXXX-YYYY` 또는 `NORMAL2-IM-XXXX-YYYY` (끝에 `-ZZZZ`가 더 붙어도 됨) → 그룹 `N:IM-XXXX` / `N:NORMAL2-IM-XXXX`

한계:
- 파일명 번호가 실제 동일 환자를 뜻하는지는 원천 설명으로 확인되지 않았다.
- 정상(N)과 폐렴(P)의 번호 체계가 같다고 가정하지 않으므로, 같은 사람이 두 라벨에 모두 있어도 찾아낼 수 없다.
- 원래 분할(train/test)에 걸친 그룹이 실제로 같은 환자인지, 분할마다 번호를 따로 매긴 것인지는 알 수 없다. test는 고정이므로, 걸치는 그룹의 처리 규칙은 분할 단계에서 정하고 기록한다.

## 5. 같은 파일(sha256) 중복

- 중복 해시 묶음: 30개 (그중 원래 분할을 가로지르는 묶음: 0개)
  - train/NORMAL/NORMAL2-IM-0587-0001-0001.jpeg = train/NORMAL/NORMAL2-IM-0587-0001-0002.jpeg
  - train/PNEUMONIA/person1159_virus_1944.jpeg = train/PNEUMONIA/person1159_virus_1945.jpeg
  - train/PNEUMONIA/person124_virus_246.jpeg = train/PNEUMONIA/person124_virus_247.jpeg
  - train/PNEUMONIA/person1261_virus_2147.jpeg = train/PNEUMONIA/person1261_virus_2148.jpeg
  - train/PNEUMONIA/person1312_bacteria_3317.jpeg = train/PNEUMONIA/person1312_bacteria_3318.jpeg
  - train/PNEUMONIA/person1343_bacteria_3413.jpeg = train/PNEUMONIA/person1343_bacteria_3414.jpeg
  - train/PNEUMONIA/person1349_bacteria_3436.jpeg = train/PNEUMONIA/person1349_bacteria_3437.jpeg
  - train/PNEUMONIA/person1372_bacteria_3501.jpeg = train/PNEUMONIA/person1372_bacteria_3502.jpeg = train/PNEUMONIA/person1372_bacteria_3503.jpeg
  - train/PNEUMONIA/person1430_bacteria_3695.jpeg = train/PNEUMONIA/person1430_bacteria_3696.jpeg
  - train/PNEUMONIA/person1481_bacteria_3862.jpeg = train/PNEUMONIA/person1481_bacteria_3863.jpeg

## 6. 완료 기준

- [x] 라벨 누락 0건
- [x] 읽기 실패 내역 확인 (위 2번)
- [x] 클래스별 수가 가이드 표와 일치
- [x] 환자 그룹 규칙과 한계 문서화 (위 4번)
