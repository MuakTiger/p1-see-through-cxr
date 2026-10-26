"""폐 영역 모델용 장치 설정 필터 시험: equipment_eval_direct.py 와 같은 시험(같은 필터·강도·bootstrap·검증)을 하되,
필터를 건 사진에 폐 껍질 처리(training/lung_prep.py)를 한 뒤 모델에 넣는다.

순서: 원본 → 회색 512 → (필터: 전체 사진 기준의 물리 단위) → 폐 껍질 밖을 0으로 → 모델.
폐 마스크는 필터를 걸지 않은 사진에서 만든 것(xrv_mask_split.py)을 고정해 쓴다. 그래서 이 시험은 "분할이 항상 맞다고 가정했을 때" 분류기의 강건성이고,
필터가 걸린 사진에서 분할 모델이 얼마나 버티는지는 보지 않는다.

equipment_eval_direct.py 는 고정 해시에 들어 있어 수정하지 않고, 이 파일이 그 모듈을 불러와 사진 경로 기록·입력 처리 세 곳만 바꿔 끼운다.
CLI 와 test 보호장치(--plan-file, --plan-sha256)는 원래 스크립트의 main() 그대로다. 추가 옵션: --masks-dir (npz 폴더, val/test 마스크).

사용 예 (p1 폴더에서):
    .venv/bin/python src/equipment_eval_lung.py --masks-dir <npz 폴더> --model-dir runs/results/l-s42-.../model --out reports/equipment_v2/lung/val_l-s42 --device mps --label l-s42
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image as _PILImage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))
import equipment_eval_direct as D  # noqa: E402
import lung_prep as LP  # noqa: E402
from shift_filters import to_gray as _to_gray  # noqa: E402

own = argparse.ArgumentParser(add_help=False)
own.add_argument("--masks-dir", required=True)
own.add_argument("--images-dir", default=os.environ.get("SM_CHANNEL_IMAGES") or str(Path.home() / "data/chest_xray"))
own.add_argument("--out", default=None)
OWN = own.parse_known_args()[0]

MASKS = {}
for _sp in ("val", "test"):
    _p = Path(OWN.masks_dir) / f"{_sp}_lung_masks.npz"
    if _p.is_file():
        MASKS.update(LP.load_masks(_p, 512))
if not MASKS:
    raise SystemExit(f"{OWN.masks_dir} 에 val_lung_masks.npz / test_lung_masks.npz 가 없습니다.")
IMAGES_ROOT = Path(OWN.images_dir)
ALIGNED = []  # 사진을 읽은 순서대로 그 사진의 폐 마스크 (필터 시험 때 같은 순서로 쓴다)
CUR = {"rel": None}


class _ImageProxy:
    """D.Image 대신 쓰는 대리자: open 할 때 어느 사진인지(relative_path)를 기억한다. 나머지는 PIL.Image 그대로."""

    def __getattr__(self, name):
        return getattr(_PILImage, name)

    def open(self, fp, *a, **kw):
        CUR["rel"] = Path(fp).relative_to(IMAGES_ROOT).as_posix()
        return _PILImage.open(fp, *a, **kw)


def _to_gray_rec(im, size):
    g = _to_gray(im, size)
    ALIGNED.append(MASKS[CUR["rel"]])
    return g


_orig_predict_grays = D.predict_grays


def _predict_grays_lung(model, grays, device, chunk=64):
    assert len(grays) == len(ALIGNED), "사진 순서와 마스크 순서가 맞지 않습니다"
    lung = np.stack([LP.apply_lung(g, m)[0] for g, m in zip(grays, ALIGNED)])
    return _orig_predict_grays(model, lung, device, chunk)


_orig_build_transform = D.T.build_transform


def _build_transform(preprocess, size):
    base = _orig_build_transform(preprocess, size)  # PNG 로 저장한 학습 입력과 같은 길: RGB → Resize(무변화) → ToTensor → Normalize

    def tf(img_rgb):
        out, _ = LP.apply_lung(_to_gray(img_rgb, size), MASKS[CUR["rel"]])
        return base(_PILImage.fromarray(out).convert("RGB"))
    return tf


D.Image = _ImageProxy()
D.to_gray = _to_gray_rec
D.predict_grays = _predict_grays_lung
D.T.build_transform = _build_transform

if __name__ == "__main__":
    D.main()
    if OWN.out and (Path(OWN.out) / "shift_report.md").is_file():
        fails = sum(1 for m in ALIGNED if m.mean() < LP.MIN_AREA)
        note = ["", "## 폐 영역 입력 (이 시험의 입력 처리)", "",
                f"- 모델 입력 = 필터를 건 회색 512 사진에서 폐 마스크의 볼록 껍질(여백 {LP.DILATE_PX}px) 밖을 0으로 칠한 것 (`training/lung_prep.py`). 위 본문의 \"증강·전처리 없음\"은 원래 스크립트의 문구이고 이 시험에는 해당하지 않는다.",
                f"- 폐 마스크: TorchXRayVision PSPNet, 필터 없는 사진에서 한 번 만들어 고정 (`{OWN.masks_dir}`). 필터가 걸린 사진에서 분할을 다시 하지 않았다.",
                f"- 분할 실패(마스크 면적 < {LP.MIN_AREA:.0%})로 사진 전체를 남긴 사진: {fails}장 / {len(ALIGNED)}장",
                "- 검증 (a)는 학습 변환 경로와 필터 경로가 같은 입력 처리를 쓰는지의 확인이고, 학습 기록과의 일치 여부는 위 본문의 값으로 본다."]
        with open(Path(OWN.out) / "shift_report.md", "a", encoding="utf-8") as f:
            f.write("\n".join(note) + "\n")
