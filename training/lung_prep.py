"""폐 영역만 남기는 입력 처리 (폐 마스크 → 볼록 껍질 → 여백 → 폐 밖은 0).

PSPNet(TorchXRayVision)이 입력과 같은 찌그러뜨린 S×S 회색 사진에서 만든 좌·우 폐 마스크를 받아,
  1. 두 폐 마스크의 합의 볼록 껍질을 구한다. 폐렴으로 뿌옇게 된 부위가 마스크에서 빠져도 껍질 안에는 들어온다.
     두 폐 사이(종격동·심장 그림자)도 껍질 안에 들어온다.
  2. 껍질을 DILATE_PX(512 기준 약 3%)만큼 넓혀 흉막 가장자리와 분할 오차를 받아 준다.
  3. 껍질 밖 화소를 0(검정)으로 칠한다. 크기·위치·해상도는 그대로 두고 자르기·확대는 하지 않는다 (한 번에 한 가지만 바꾼다).
마스크가 사진의 MIN_AREA 미만이면 분할 실패로 보고 사진을 그대로 쓴다 (실패 개수는 호출한 쪽이 센다).

필터 시험에서는 장비 효과를 먼저 걸고(필터는 전체 사진 기준의 물리 단위) 그다음 이 처리를 한다. 마스크는 필터를 걸지 않은 사진에서 한 번만 만든 것을 고정해 쓴다.
"""
import numpy as np
import cv2

DILATE_PX = 16
MIN_AREA = 0.05


def hull_mask(mask, dilate_px=DILATE_PX):
    """mask: (S,S) bool 폐 마스크. 반환 (껍질 마스크 bool, ok). ok=False 이면 분할 실패(전체를 남김)."""
    if mask.mean() < MIN_AREA:
        return np.ones(mask.shape, bool), False
    pts = cv2.findNonZero(mask.astype(np.uint8))
    canvas = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(canvas, cv2.convexHull(pts), 1)
    if dilate_px:
        canvas = cv2.dilate(canvas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)))
    return canvas.astype(bool), True


def apply_lung(gray_u8, mask, dilate_px=DILATE_PX):
    """(S,S) uint8 회색 사진과 같은 크기의 폐 마스크 → (폐 밖을 0으로 칠한 사진, ok)."""
    hull, ok = hull_mask(mask, dilate_px)
    return np.where(hull, gray_u8, 0).astype(np.uint8), ok


def load_masks(npz_path, size=512):
    """xrv_mask_split.py 가 저장한 묶음 → {relative_path: (size,size) bool 마스크}."""
    z = np.load(npz_path)
    bits = np.unpackbits(z["masks"], axis=1)[:, :size * size].reshape(len(z["paths"]), size, size).astype(bool)
    return {str(p): bits[i] for i, p in enumerate(z["paths"])}
