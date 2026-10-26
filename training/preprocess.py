"""입력 전처리 P0~P4 (기획안 v0.4 05장, P 시리즈).

P 시리즈는 "같은 모델(실험 B 설정)에 입력 전처리만 바꿔서 학습하면 결과가 어떻게 달라지나" 를 본다.
전처리는 증강이 아니다. 사진마다 항상 같은 결과가 나오는 결정적 변환이고,
train·val (그리고 나중에 test·서비스) 모두에 똑같이 적용한다.
비유: 증강은 "연습 문제를 매번 조금씩 바꿔 주는 것", 전처리는 "모든 시험지를 같은 안경을 쓰고 보게 하는 것".

정의 (값은 기획안 그대로, 튜닝하지 않는다)
    P0 원본          전처리 없음. train.py 의 기존 경로를 그대로 쓴다 (여기 함수를 거치지 않음).
    P1 HE            cv2.equalizeHist (사진 전체 히스토그램을 한 번에 평평하게)
    P2 CLAHE         cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) (8×8 구역마다 따로, 과한 증폭은 2.0 에서 자름)
    P3 LPF 가우시안   σ 1.5 (cv2.GaussianBlur, ksize (0,0), BORDER_REFLECT) → 미세 무늬(고주파)를 줄임
    P4 HPF 언샤프     σ 2, α 1.0: out = img + α·(img − gaussian(img, σ)), 0~255 로 자름 → 경계를 강조

처리 순서 (P1~P4)
    회색조(L) → PIL Resize (224,224) BILINEAR → 전처리 → 3채널 복제 → ToTensor → Normalize(ImageNet)
    원본 크기가 384~2916 픽셀로 제각각이라, 224 로 줄인 다음에 적용해야 σ(픽셀 단위)와
    CLAHE 타일(8×8 구역)이 모든 사진에서 같은 의미(같은 해부학적 크기)를 가진다.

필터 계산은 shift_filters.py 의 함수(equalize, blur, sharpen)를 그대로 쓴다.
그래서 "장비 차이 필터" 와 "전처리" 의 수식 정의가 한 곳에만 있고 서로 어긋나지 않는다.
"""
from PIL import Image

from shift_filters import blur, equalize, sharpen, to_gray_224

PREPROCESS = {
    "P0": {
        "title": "P0 원본",
        "desc": "P0 원본(전처리 없음)",
        "params": {},
    },
    "P1": {
        "title": "P1 HE",
        "desc": "P1 HE(cv2.equalizeHist)",
        "params": {"method": "cv2.equalizeHist"},
    },
    "P2": {
        "title": "P2 CLAHE",
        "desc": "P2 CLAHE(clip 2.0, 8x8)",
        "params": {"method": "cv2.createCLAHE", "clipLimit": 2.0, "tileGridSize": [8, 8]},
    },
    "P3": {
        "title": "P3 LPF 가우시안",
        "desc": "P3 LPF 가우시안(sigma 1.5)",
        "params": {"method": "cv2.GaussianBlur", "sigma": 1.5, "ksize": [0, 0], "border": "BORDER_REFLECT"},
    },
    "P4": {
        "title": "P4 HPF 언샤프",
        "desc": "P4 HPF 언샤프(sigma 2, alpha 1.0)",
        "params": {"method": "unsharp: out = img + alpha*(img - GaussianBlur(img, sigma)), clip 0~255",
                   "sigma": 2.0, "alpha": 1.0, "border": "BORDER_REFLECT"},
    },
}
PREPROCESS_ORDER = {
    "P0": "RGB 변환(회색은 3채널 복제) → Resize((224,224)) → ToTensor → Normalize(ImageNet mean/std)",
    "P1~P4": "회색조(L) → PIL Resize (224,224) BILINEAR → 전처리(224 해상도에서) → 3채널 복제 → ToTensor "
             "→ Normalize(ImageNet mean/std)",
}


def check_preprocess(name):
    """허용된 전처리 이름인지 확인한다. 아니면 한국어로 알려 주고 멈춘다."""
    if name not in PREPROCESS:
        raise ValueError(
            f"알 수 없는 전처리: {name} (가능: {', '.join(PREPROCESS)}). "
            "P5(3채널 조합)는 이번 범위에서 구현하지 않았습니다."
        )


def apply_preprocess(gray, name):
    """224×224 회색조(uint8) 영상에 전처리 name 을 적용한다. P0 는 그대로 돌려준다."""
    check_preprocess(name)
    if name == "P0":
        return gray.copy()
    if name == "P1":
        return equalize(gray, "he")
    if name == "P2":
        return equalize(gray, "clahe_2.0")  # clipLimit 2.0, tileGridSize (8, 8)
    if name == "P3":
        return blur(gray, 1.5)
    if name == "P4":
        return sharpen(gray, alpha=1.0, sigma=2.0)
    raise AssertionError("도달하면 안 되는 코드")


def describe(name):
    """기록(metadata.json, config.json)에 남길 전처리 정보."""
    check_preprocess(name)
    order = PREPROCESS_ORDER["P0" if name == "P0" else "P1~P4"]
    return {"preprocess": name, "preprocess_desc": PREPROCESS[name]["desc"],
            "preprocess_params": PREPROCESS[name]["params"], "preprocess_order": order}


class GrayPreprocess:
    """PIL 영상 → 224 회색조 → 전처리 → 3채널 PIL 영상 (뒤에 ToTensor, Normalize 가 붙는다).

    P1~P4 전용. 서비스(추론)에서도 이 클래스를 그대로 쓰면 학습과 똑같은 입력이 된다.
    """

    def __init__(self, name):
        check_preprocess(name)
        if name == "P0":
            raise ValueError("P0 는 GrayPreprocess 를 쓰지 않습니다 (train.py 의 기존 경로를 씀).")
        self.name = name

    def __call__(self, image):
        gray = to_gray_224(image)  # convert("L") → resize (224,224) BILINEAR
        out = apply_preprocess(gray, self.name)
        return Image.fromarray(out).convert("RGB")  # 3채널 복제

    def __repr__(self):
        return f"GrayPreprocess({PREPROCESS[self.name]['desc']})"
