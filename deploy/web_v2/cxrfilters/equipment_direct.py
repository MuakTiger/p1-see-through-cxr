"""X선 장치 설정 필터 v2.1: 모델 입력 크기(S)의 사진에 직접 거는 판 (512 등).

equipment_filters.py(v2.0)는 224 모델용이라 896×896 작업 해상도에서 걸고 224로 줄였다. 512 같은 큰 입력에서는
작업 해상도를 따로 두지 않고, 모델이 보는 S×S 회색조(shift_filters.to_gray(im, S))에 바로 건다.
(원본 사진 61%는 짧은 변이 1024 미만이라 1024 작업 해상도로 키우면 사진 선명도에 따라 효과가 달라지고, 속도도 느리다.)
필터의 정의와 강도는 v2.0과 같다. 길이 단위만 정리했다: **224 사진에서 1픽셀 ≈ 1mm** 로 놓고 S 사진에서는 S/224 픽셀 = 1mm.

    dose    선량(mAs)     광자가 줄어 양자 노이즈가 늘어난다 (노이즈 ∝ 1/√선량), 밝기에 따라 크기가 다르다
    motion  조사시간      오래 찍으면 아이가 움직여 세로로 번진다 (아래쪽일수록 큼)
    scatter 산란(격자 없음) 산란선이 안개처럼 깔려 작은 구조의 대비가 줄어든다
    kvp     관전압(kVp)   관전압이 높으면 대비가 줄고 밝은 부위(뼈)가 덜 하얘진다
    sensor  센서 화소 크기 화소가 커지면 미세한 무늬가 뭉개진다 (화소 크기가 입력 1픽셀보다 작으면 효과가 없다)

선량의 기준 노이즈(SIGMA0)는 크기마다 원본 val 사진 200장에서 잰 Immerkaer 노이즈 추정 평균이다 (src/noise_anchor.py).
사진은 0~1 표시 값(공기=어둡고 뼈=밝음), 표시 값이 노출량과 선형이라고 가정한다. 결과를 디스크에 저장하지 않고,
같은 입력과 seed면 같은 출력이다. v1의 흐림·경계 강조·대비 보정·JPEG·밝게·어둡게는 shift_filters.apply_shift 를 그대로 쓴다.
"""
import cv2
import numpy as np

BASE = 224            # 이 크기에서 1픽셀 ≈ 1mm
E_BAR = 0.5           # 노출량(=1-표시 값)의 대표값. 이 노출량에서 노이즈가 기준 노이즈
E_MIN = 0.02          # 완전히 하얀 곳도 노이즈가 0이 되지 않게 하는 아래 한계
SCATTER_SIGMA_MM = 25
KVP_KNEE = 0.6

# 선량 필터의 기준 노이즈 (0~255 기준). src/noise_anchor.py --calibrate 로 정했다 (10/22, val 200장 seed 42, 모델 점수는 쓰지 않음):
# 원본 노이즈 추정 평균(224 0.919, 448 0.735, 512 0.699)에 보정계수(0.963, 0.909, 0.890)를 곱해, 선량 1/16에서
# 노이즈 추정 배율이 원리값 4.0이 되도록 맞췄다 (노이즈 추정기가 크기마다 조금씩 다르게 반응해서 보정이 필요했다).
SIGMA0 = {224: 0.885, 448: 0.668, 512: 0.622}

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
               "param": "화소 크기 mm", "levels": [0.5, 1.0, 2.0], "labels": ["0.5", "1.0", "2.0"]},
}


def px_per_mm(size):
    return size / BASE


def dose(d, rel_dose, seed, sigma0):
    """선량을 rel_dose배로 줄인다 (1=기준). 총 노이즈가 기준 노이즈(sigma0, 0~255)의 1/√rel_dose배가 되도록 모자란 만큼만 더한다."""
    e = 1.0 - d
    sigma = (sigma0 / 255.0) * np.sqrt(np.clip(e, E_MIN, 1.0) / E_BAR) * np.sqrt(1.0 / rel_dose - 1.0)
    noise = np.random.default_rng(seed).standard_normal(d.shape).astype(np.float32)
    return np.clip(d - sigma * noise, 0.0, 1.0)


def motion(d, length_mm):
    """세로 방향 움직임 번짐. 위쪽 25% ~ 아래쪽 100%. 길이는 mm (S 사진에서 S/224 픽셀 = 1mm)."""
    n = int(round(length_mm * px_per_mm(d.shape[0])))
    n += (n % 2 == 0)
    kernel = np.ones((n, 1), np.float32) / n
    blurred = cv2.filter2D(d, -1, kernel, borderType=cv2.BORDER_REFLECT)
    w = (0.25 + 0.75 * np.linspace(0.0, 1.0, d.shape[0], dtype=np.float32))[:, None]
    return (1.0 - w) * d + w * blurred


def scatter(d, k):
    """산란을 직접선의 k배 더한다 (아주 크게 흐린 사진). 평균 밝기는 유지."""
    veil = cv2.GaussianBlur(d, (0, 0), sigmaX=SCATTER_SIGMA_MM * px_per_mm(d.shape[0]), borderType=cv2.BORDER_REFLECT)
    return (d + k * veil) / (1.0 + k)


def kvp(d, contrast, slope):
    """평균 기준 대비 축소 + 밝은 부위(KVP_KNEE 위) 기울기 축소, 평균 밝기는 원본과 같게."""
    m = float(d.mean())
    x = m + contrast * (d - m)
    x = np.where(x > KVP_KNEE, KVP_KNEE + (x - KVP_KNEE) * slope, x)
    x = x + (m - float(x.mean()))
    return np.clip(x, 0.0, 1.0)


def sensor(d, pitch_mm):
    """화소 크기 pitch_mm인 센서: 화소 안에서 평균(줄이기)한 뒤 다시 키운다. 화소가 1픽셀 이하면 그대로."""
    size = d.shape[0]
    p = pitch_mm * px_per_mm(size)
    if p <= 1.0:
        return d.copy()
    n = max(2, int(round(size / p)))
    small = cv2.resize(d, (n, n), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_LINEAR)


def apply_equipment(gray, name, level, seed=0, sigma0=None):
    """S×S uint8 회색조에 장치 필터 name을 단계 level(0~3)로 건다. level 0은 원본. 결과도 S×S uint8."""
    if name not in EQUIP:
        raise ValueError(f"알 수 없는 필터: {name} (가능: {', '.join(EQUIP)})")
    if level == 0:
        return gray.copy()
    d = gray.astype(np.float32) / 255.0
    v = EQUIP[name]["levels"][level - 1]
    if name == "dose":
        out = dose(d, v, seed, SIGMA0[gray.shape[0]] if sigma0 is None else sigma0)
    elif name == "motion":
        out = motion(d, v)
    elif name == "scatter":
        out = scatter(d, v)
    elif name == "kvp":
        out = kvp(d, *v)
    else:
        out = sensor(d, v)
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
