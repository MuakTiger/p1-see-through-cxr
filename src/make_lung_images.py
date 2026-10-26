"""폐 영역 입력 사진 만들기: 원본 사진 → (평가와 같은 경로) 회색 512×512 → 폐 껍질 밖을 0으로 → PNG 저장.

학습 때 매번 큰 JPEG 를 풀고 줄이는 일을 한 번으로 줄이고, 학습·평가가 같은 함수(shift_filters.to_gray + lung_prep.apply_lung)를 쓰게 한다.
폐 마스크는 xrv_mask_split.py 가 만든 <split>_lung_masks.npz (필터 없는 사진 기준, 라벨 미사용)이다.
새 매니페스트에는 처리한 행의 relative_path 만 .png 로 바꾸고 나머지 열은 그대로 둔다 (train_aug.py 는 train/val 행만 읽는다).

사용 예 (p1 폴더에서):
    .venv/bin/python src/make_lung_images.py --masks-dir <npz 폴더> --out-dir ~/data/lung512 --manifest-out ~/data/lung512/split_manifest.csv --splits train val
"""
import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent / "training", HERE):  # 저장소와 JupyterLab 묶음(같은 폴더) 모두 지원
    sys.path.insert(0, str(cand))
import lung_prep as LP  # noqa: E402
from shift_filters import to_gray  # noqa: E402

SIZE = 512
_G = {}


def _init(masks_dir, splits, images_dir, out_dir):
    masks = {}
    for sp in splits:
        masks.update(LP.load_masks(Path(masks_dir) / f"{sp}_lung_masks.npz", SIZE))
    _G.update(masks=masks, images_dir=Path(images_dir), out_dir=Path(out_dir))


def _work(rel):
    with Image.open(_G["images_dir"] / rel) as im:
        gray = to_gray(im, SIZE)
    out, ok = LP.apply_lung(gray, _G["masks"][rel])
    dst = (_G["out_dir"] / rel).with_suffix(".png")
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(dst, optimize=False)
    return rel, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE.parent / "manifests/split_manifest.csv"))
    ap.add_argument("--images-dir", default=str(Path.home() / "data/chest_xray"))
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest-out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="속도·동작 확인용: 앞의 N장만")
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    todo = [r["relative_path"] for r in rows if r["split"] in args.splits]
    if args.limit:
        todo = todo[:args.limit]
    t0 = time.time()
    fails = []
    with Pool(args.workers, initializer=_init, initargs=(args.masks_dir, args.splits, args.images_dir, args.out_dir)) as pool:
        for i, (rel, ok) in enumerate(pool.imap(_work, todo, chunksize=16), 1):
            if not ok:
                fails.append(rel)
            if i % 500 == 0:
                print(f"  {i}/{len(todo)} ({time.time() - t0:.0f}초)", flush=True)
    done = set(todo)
    for r in rows:
        if r["relative_path"] in done:
            r["relative_path"] = str(Path(r["relative_path"]).with_suffix(".png"))
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"완료: {len(todo)}장 ({time.time() - t0:.0f}초), 폐 분할 실패(면적<{LP.MIN_AREA:.0%})로 전체를 남긴 사진 {len(fails)}장 {fails[:5]}", flush=True)


if __name__ == "__main__":
    main()
