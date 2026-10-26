"""장비 차이를 흉내 내는 고전 필터 모음 (장비 차이 민감도 시험용).

이 데이터는 한 병원(광저우)의 장비로 찍은 사진이다. 다른 병원 장비에서 찍으면
사진의 흐림·선명도·노이즈·밝기·대비·압축 정도가 달라진다. 실제 다른 병원 데이터가
없으므로, 그런 차이를 고전 영상처리 필터로 흉내 내서 모델이 얼마나 버티는지 본다.

규칙
- 모든 필터는 224×224 회색조(uint8) 영상에 적용한다. 원본 크기가 384~2916 픽셀로
  제각각이라, 같은 크기로 줄인 뒤 적용해야 σ(픽셀 단위) 같은 강도가 모든 사진에서
  같은 의미를 가진다.
- 결과를 디스크에 저장하지 않는다. 사진을 읽는 순간 메모리에서만 적용한다.
- 같은 입력에는 항상 같은 출력이 나온다 (노이즈도 사진마다 고정된 seed를 쓴다).
- numpy, cv2(OpenCV), PIL, 표준 라이브러리만 쓴다. 학습 서버(컨테이너)에도
  opencv-python 4.9.0.80, numpy 1.26.4가 있음을 확인했다.

사용 예:
    from shift_filters import SHIFTS, apply_shift
    out = apply_shift(gray_224, "blur", level=2, seed=123)
"""
import io

import cv2
import numpy as np
from PIL import Image

INPUT_SIZE = 224

# 필터 정의. level 0은 항상 "원본 그대로"이고, 1·2·3으로 갈수록 강해진다.
# 강도는 "실제 장비 차이로 있을 법한 범위"를 근거로 정했다 (근거: reports/shift_test/filter_rationale.md).
# 점수를 보고 고르지 않는다. 시험 전에 고정한다 (v2, 2026-10-19).
# 크기 감각: 224 픽셀이 가슴 폭 약 20cm에 해당하므로 1픽셀은 약 1mm다.
SHIFTS = {
    "blur": {
        "title": "흐림 (LPF 가우시안)",
        "situation": "CR 판처럼 해상도가 낮은 검출기, 환자 움직임",
        "param": "sigma",
        "levels": [0.5, 1.0, 2.0],
    },
    "sharpen": {
        "title": "경계 강조 (HPF 언샤프)",
        "situation": "제조사가 자동으로 거는 경계 강조 후처리",
        "param": "alpha (sigma 2)",
        "levels": [0.5, 1.0, 2.0],
    },
    "noise": {
        "title": "노이즈 (가우시안)",
        "situation": "저선량 촬영 (소아는 선량을 낮춰 찍음)",
        "param": "std (0~255 기준)",
        "levels": [4.0, 8.0, 16.0],
    },
    "bright": {
        "title": "밝게 (감마 < 1)",
        "situation": "과다 노출, 밝게 맞춘 표시 설정",
        "param": "gamma",
        "levels": [0.85, 0.7, 0.55],
    },
    "dark": {
        "title": "어둡게 (감마 > 1)",
        "situation": "노출 부족, 어둡게 맞춘 표시 설정",
        "param": "gamma",
        "levels": [1.2, 1.45, 1.8],
    },
    "contrast": {
        "title": "대비 감소",
        "situation": "산란선 차단 격자 없이 촬영, 창 설정 차이로 뿌연 사진",
        "param": "factor (평균 기준 축소)",
        "levels": [0.85, 0.7, 0.55],
    },
    "equalize": {
        "title": "대비 보정 (CLAHE / HE)",
        "situation": "제조사·병원마다 다른 자동 대비 보정",
        "param": "method",
        "levels": ["clahe_1.5", "clahe_3.0", "he"],
    },
    "jpeg": {
        "title": "압축 (JPEG)",
        "situation": "PACS 밖으로 내보내기, 메신저 전송, 재압축",
        "param": "quality",
        "levels": [75, 40, 20],
    },
}


def to_gray(image, size=INPUT_SIZE):
    """PIL 영상을 회색조로 바꾸고 size×size로 줄인다 (자르기 없음)."""
    gray = image.convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(gray, dtype=np.uint8)


def to_gray_224(image):
    """PIL 영상을 회색조로 바꾸고 224×224로 줄인다 (자르기 없음)."""
    return to_gray(image, INPUT_SIZE)


def _clip_uint8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def _gaussian(img, sigma):
    # ksize=(0,0)이면 OpenCV가 sigma에 맞는 커널 크기를 스스로 정한다.
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REFLECT)


def blur(img, sigma):
    """사진을 뿌옇게 만든다. 안경을 벗고 보는 것과 비슷하다."""
    return _gaussian(img, sigma)


def sharpen(img, alpha, sigma=2.0):
    """경계를 강조한다. 원본 + alpha × (원본 − 흐린 원본)."""
    base = img.astype(np.float32)
    detail = base - _gaussian(img, sigma).astype(np.float32)
    return _clip_uint8(base + alpha * detail)


def add_noise(img, std, seed):
    """모래알 같은 노이즈를 더한다. seed가 같으면 노이즈도 같다."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, std, size=img.shape)
    return _clip_uint8(img.astype(np.float32) + noise)


def gamma_curve(img, gamma):
    """밝기 곡선을 바꾼다. 출력 = 255 × (입력/255)^gamma."""
    table = _clip_uint8(255.0 * (np.arange(256) / 255.0) ** gamma)
    return cv2.LUT(img, table)


def reduce_contrast(img, factor):
    """평균 밝기를 중심으로 밝기 차이를 factor배로 줄인다."""
    base = img.astype(np.float32)
    mean = base.mean()
    return _clip_uint8(mean + factor * (base - mean))


def equalize(img, method):
    """히스토그램 평활화. he는 전체 한 번에, clahe는 8×8 구역마다 따로 맞춘다."""
    if method == "he":
        return cv2.equalizeHist(img)
    if method.startswith("clahe_"):
        clip = float(method.split("_")[1])
        return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(img)
    raise ValueError(f"알 수 없는 평활화 방법: {method}")


def jpeg(img, quality):
    """JPEG로 압축했다가 다시 푼다. 품질이 낮을수록 블록 무늬가 생긴다."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L"), dtype=np.uint8)


def apply_shift(img, name, level, seed=0):
    """img(정사각형 uint8 회색조, 기본 224×224)에 필터 name을 강도 level(0~3)로 적용한다.

    level 0은 원본을 그대로 돌려준다. seed는 노이즈에만 쓰인다.
    224가 아닌 크기(예: 448)에서는 "픽셀 단위 길이"인 σ(blur, sharpen)를 크기/224 배로 키워서
    같은 해부학적 크기(같은 mm)에 같은 효과가 나게 한다. 노이즈 std, 감마, 대비, JPEG 품질, CLAHE 구역 수는
    크기와 무관하게 그대로다. 224 에서는 배율이 1.0 이라 예전과 결과가 똑같다.
    """
    if name not in SHIFTS:
        raise ValueError(f"알 수 없는 필터: {name} (가능: {', '.join(SHIFTS)})")
    if level == 0:
        return img.copy()
    value = SHIFTS[name]["levels"][level - 1]
    scale = img.shape[0] / INPUT_SIZE
    if name == "blur":
        return blur(img, value * scale)
    if name == "sharpen":
        return sharpen(img, value, sigma=2.0 * scale)
    if name == "noise":
        return add_noise(img, value, seed)
    if name in ("bright", "dark"):
        return gamma_curve(img, value)
    if name == "contrast":
        return reduce_contrast(img, value)
    if name == "equalize":
        return equalize(img, value)
    if name == "jpeg":
        return jpeg(img, value)
    raise AssertionError("도달하면 안 되는 코드")
