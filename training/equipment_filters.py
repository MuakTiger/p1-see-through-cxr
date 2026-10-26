"""X선 장치 설정이 사진에 만드는 효과를 고전 영상처리로 흉내 내는 필터 v2 (임시 강도: 갤러리로 확인한 뒤 확정한다).

v1(shift_filters.py)은 "사진이 이렇게 달라지면 모델이 버티나"를 224 사진에 바로 걸었다.
v2는 "장치 설정을 바꾸면 사진에 무엇이 생기나"를 물리 순서대로 흉내 낸다. 새로 만드는 5종:

    dose    선량(mAs)     선량이 줄면 광자 수가 줄어 양자 노이즈가 늘어난다 (노이즈 ∝ 1/√선량). 밝기에 따라 크기가 다르다.
    motion  조사시간      오래 찍으면 아이가 움직여 세로로 번진다. 횡격막이 있는 아래쪽일수록 더 번진다.
    scatter 산란(격자 없음) 산란선이 필름에 안개처럼 깔려 작은 구조의 대비가 줄어든다 (격자가 이것을 막는다).
    kvp     관전압(kVp)   관전압이 높으면 X선이 더 잘 뚫고 지나가 대비가 줄고, 뼈 같은 밝은 부위가 덜 하얘진다.
    sensor  센서 화소 크기 화소가 커지면(해상도가 낮으면) 미세한 무늬가 뭉개진다.

규칙
- 원본 크기가 384~2916 픽셀로 제각각이라, 먼저 896×896 "작업 해상도"로 맞춘다 (224의 4배, 가슴 폭 약 20cm 가정 시 화소 약 0.22mm).
  노이즈와 번짐을 이 해상도에서 만든 뒤 224로 줄인다. 실제 장비도 센서에서 생긴 노이즈가 줄여져 보이는 것과 같다.
  모델이 실제로 보는 원본 224 사진은 그대로 두고, 필터가 만든 변화량만 더한다 (to_input).
- 사진 값은 0~1(표시 값, 공기=어둡고 뼈=밝음). 표시 값이 노출량과 거의 선형이라고 가정한다 (8비트 JPEG라 실제 톤 곡선은 모른다).
- 결과를 디스크에 저장하지 않는다. 같은 입력과 seed면 같은 출력이다.
- v1의 흐림·경계 강조·대비 보정·JPEG·밝게·어둡게는 그대로 쓴다 (shift_filters.apply_shift, 224 사진에 적용).
"""
import cv2
import numpy as np

INPUT_SIZE = 224
WORK = 896
SCALE = WORK // INPUT_SIZE  # 4: 224 기준 1 픽셀(≈1mm)은 작업 해상도에서 4 픽셀

# 선량 노이즈의 기준: 기준 선량(=원본)에서 224 사진의 노이즈 추정(Immerkaer)이 약 0.9/255가 되도록 맞춘다.
# 작업 해상도에서 서로 독립인 노이즈는 4×4=16 픽셀 평균으로 1/4이 되므로 896 기준 표준편차는 4배.
SIGMA_REF = 4 * 0.9 / 255
E_BAR = 0.5       # 노출량(=1-표시 값)의 대표값. 이 노출량에서 노이즈가 SIGMA_REF.
E_MIN = 0.02      # 완전히 하얀 곳도 노이즈가 0이 되지 않게 하는 아래 한계

SCATTER_SIGMA_MM = 25   # 산란이 퍼지는 폭 (224 기준 mm)
KVP_KNEE = 0.6          # 이 표시 값 위(밝은 부위=뼈)를 눌러 준다

EQUIP = {
    "dose": {"title": "선량 (mAs)", "situation": "선량이 줄면 광자가 줄어 노이즈가 늘어난다",
             "param": "상대 선량", "levels": [1 / 4, 1 / 16, 1 / 64], "labels": ["1/4", "1/16", "1/64"]},
    "motion": {"title": "조사시간 (움직임)", "situation": "오래 찍으면 아이가 움직여 세로로 번진다 (아래쪽일수록 큼)",
               "param": "번짐 길이 mm", "levels": [2, 4, 8], "labels": ["2", "4", "8"]},
    "scatter": {"title": "산란 (격자 없음)", "situation": "산란선이 안개처럼 깔려 작은 구조의 대비가 줄어든다",
                "param": "산란/직접선 비 k", "levels": [0.25, 0.5, 1.0], "labels": ["0.25", "0.5", "1.0"]},
    "kvp": {"title": "관전압 kVp ↑", "situation": "대비가 줄고 밝은 부위(뼈)가 덜 하얘진다",
            "param": "대비 배율 / 밝은 쪽 기울기", "levels": [(0.85, 0.8), (0.70, 0.6), (0.60, 0.5)],
            "labels": ["(0.85, 0.8)", "(0.70, 0.6)", "(0.60, 0.5)"]},
    "sensor": {"title": "센서 화소 크기", "situation": "화소가 커지면 미세한 무늬가 뭉개진다",
               "param": "화소 크기 mm", "levels": [2, 4, 8], "labels": ["0.45", "0.9", "1.8"]},
}


def to_work(image):
    """PIL 영상을 회색조 0~1 실수 896×896으로 맞춘다 (자르기 없음, 모델 입력과 같이 가로세로를 눌러 정사각형으로)."""
    g = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    interp = cv2.INTER_AREA if min(g.shape) >= WORK else cv2.INTER_LINEAR
    return cv2.resize(g, (WORK, WORK), interpolation=interp)


def down(d):
    """작업 해상도 사진(0~1)을 224×224 실수(0~1)로 줄인다 (화소 평균)."""
    return cv2.resize(d, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)


def to_input(work, filtered, base224):
    """필터가 만든 변화량만 모델이 실제로 보는 원본 224 사진(base224, uint8)에 더해 224×224 uint8을 만든다.

    작업 해상도로 맞췄다 줄인 사진을 쓰지 않는 이유: 모델 입력(shift_filters.to_gray_224)과 노이즈 바닥이 달라서
    필터 0단계가 원래 입력과 달라진다. 변화량 = 줄인(필터 후) - 줄인(필터 전)."""
    delta = down(filtered) - down(work)
    return np.clip(np.rint(base224.astype(np.float32) + 255.0 * delta), 0, 255).astype(np.uint8)


def dose(d, rel_dose, seed):
    """선량을 rel_dose배로 줄인다 (1=기준). 노출량 E=1-d에 밝기 비례 포아송형 노이즈를 더한다.

    총 노이즈는 기준 노이즈의 1/√rel_dose배가 되도록, 기준 노이즈에서 모자란 만큼만 더한다."""
    e = 1.0 - d
    sigma = SIGMA_REF * np.sqrt(np.clip(e, E_MIN, 1.0) / E_BAR) * np.sqrt(1.0 / rel_dose - 1.0)
    noise = np.random.default_rng(seed).standard_normal(d.shape).astype(np.float32)
    return np.clip(d - sigma * noise, 0.0, 1.0)


def motion(d, length_mm):
    """세로 방향 움직임 번짐. 위쪽(폐첨)은 25%, 아래쪽(횡격막)은 100% 번진다. 길이는 224 기준 mm."""
    n = int(round(length_mm * SCALE))
    n += (n % 2 == 0)
    kernel = np.ones((n, 1), np.float32) / n
    blurred = cv2.filter2D(d, -1, kernel, borderType=cv2.BORDER_REFLECT)
    w = (0.25 + 0.75 * np.linspace(0.0, 1.0, d.shape[0], dtype=np.float32))[:, None]
    return (1.0 - w) * d + w * blurred


def scatter(d, k):
    """산란을 직접선의 k배 더한다. 산란은 넓게 퍼져 있어 "아주 크게 흐린 사진"으로 흉내 낸다. 평균 밝기는 유지."""
    veil = cv2.GaussianBlur(d, (0, 0), sigmaX=SCATTER_SIGMA_MM * SCALE, borderType=cv2.BORDER_REFLECT)
    return (d + k * veil) / (1.0 + k)


def kvp(d, contrast, slope):
    """관전압을 높인 효과. 평균을 기준으로 대비를 줄이고, 밝은 부위(KVP_KNEE 위)의 기울기를 slope로 눌러 준다.
    마지막에 평균 밝기를 원본과 같게 맞춘다 (창을 다시 늘리면 대비 감소가 지워지므로 늘리지 않는다)."""
    m = float(d.mean())
    x = m + contrast * (d - m)
    x = np.where(x > KVP_KNEE, KVP_KNEE + (x - KVP_KNEE) * slope, x)
    x = x + (m - float(x.mean()))
    return np.clip(x, 0.0, 1.0)


def sensor(d, factor):
    """화소를 factor배 크게 한 센서: 평균으로 줄였다가(화소 안에서 평균) 다시 키운다."""
    n = WORK // factor
    small = cv2.resize(d, (n, n), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (WORK, WORK), interpolation=cv2.INTER_LINEAR)


def apply_equipment(d, name, level, seed=0):
    """작업 해상도 사진 d(0~1)에 장치 필터 name을 강도 level(0~3)로 건다. level 0은 원본. 결과도 작업 해상도 0~1."""
    if name not in EQUIP:
        raise ValueError(f"알 수 없는 필터: {name} (가능: {', '.join(EQUIP)})")
    if level == 0:
        return d.copy()
    v = EQUIP[name]["levels"][level - 1]
    if name == "dose":
        return dose(d, v, seed)
    if name == "motion":
        return motion(d, v)
    if name == "scatter":
        return scatter(d, v)
    if name == "kvp":
        return kvp(d, *v)
    if name == "sensor":
        return sensor(d, v)
    raise AssertionError("도달하면 안 되는 코드")
