"""폐 영역 모델(l-s42·43·44)을 SageMaker JupyterLab 터미널에서 돌리기 위한 추가 묶음을 만든다.

이미 올려 둔 aug_bundle 폴더(사진 data/, weights/, train_aug.py, shift_filters.py 가 있는 곳) 안에 풀어서 쓴다.
묶음(runs/bundles/lung_bundle.tgz): lung_prep.py, make_lung_images.py, masks/(train·val 폐 마스크), run_lung.sh, README_lung.txt.
계획은 reports/equipment_v2/lung_plan.md. 결과를 S3 로 올리는 주소는 aws_config.json 에서 읽어 run_lung.sh 안에 적는다 (화면에 출력하지 않는다).

사용 예 (p1 폴더에서, runs/lung_masks/ 에 train_lung_masks.npz 와 val_lung_masks.npz 가 있어야 한다):
    .venv/bin/python src/build_lung_bundle.py
"""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "runs/bundles/lung_bundle"
TGZ = ROOT / "runs/bundles/lung_bundle.tgz"
MASKS = ROOT / "runs/lung_masks"

RUN = r'''#!/bin/bash
# 폐 영역 모델 l-s42·43·44 학습. aug_bundle 폴더 안에서 실행한다. 끝난 실험은 건너뛰므로 중간에 꺼져도 다시 실행하면 이어진다.
# 사용: nohup bash run_lung.sh > run_lung.log 2>&1 &   그리고  tail -f run_lung.log
cd "$(dirname "$0")"
python -c "import cv2" 2>/dev/null || python -m pip install --quiet opencv-python-headless==4.9.0.80
python -c "import cv2; print('cv2', cv2.__version__)" || { echo "cv2 설치 실패"; exit 1; }
DATA=./data
LUNG=./lung512
OUT=./out_lung
mkdir -p "$OUT"
if [ ! -d "$DATA/train" ]; then
  echo "[$(date +%H:%M)] 사진 내려받기 (약 1.2GB)"
  aws s3 sync s3://aws-say3-dataset1/pneumonia-xray/chest_xray/ "$DATA" --only-show-errors || { echo "사진 내려받기 실패"; exit 1; }
fi
if [ ! -f "$LUNG/split_manifest.csv" ]; then
  echo "[$(date +%H:%M)] 폐 영역 사진 만들기 (train+val 5,216장)"
  python make_lung_images.py --manifest manifests/split_manifest.csv --images-dir "$DATA" --masks-dir ./masks \
    --out-dir "$LUNG" --manifest-out "$LUNG/split_manifest.csv" --splits train val --workers 4 || { echo "폐 영역 사진 만들기 실패"; exit 1; }
fi
run() {
  local name="$1" seed="$2"; shift 2
  if [ -f "$OUT/$name/output/experiment_summary.json" ]; then echo "[$(date +%H:%M)] 건너뜀(이미 끝남): $name"; return; fi
  echo "[$(date +%H:%M)] 시작: $name"
  python train_aug.py --images-dir "$LUNG" --manifests-dir "$LUNG" --weights-dir ./weights \
    --model-dir "$OUT/$name/model" --output-dir "$OUT/$name/output" --experiment-id "$name" \
    --train-mode layer4 --input-size 512 --epochs 30 --batch-size 16 --lr 0.001 --lr-finetune 0.0001 \
    --num-workers 4 --seed "$seed" --augment none > "$OUT/$name.log" 2>&1
  if [ -f "$OUT/$name/output/experiment_summary.json" ]; then
    echo "[$(date +%H:%M)] 끝: $name  ($(tail -1 "$OUT/$name.log" | cut -c1-90))"
    tar czf "results_$name.tgz" -C "$OUT" "$name" "$name.log"
    aws s3 cp "results_$name.tgz" "__S3_RESULTS__results_$name.tgz" --only-show-errors 2>&1 && echo "  S3 에 올림" || echo "  S3 올리기 실패: results_$name.tgz 를 파일 목록에서 내려받으세요"
  else
    echo "[$(date +%H:%M)] 실패: $name (out_lung/$name.log 마지막 줄: $(tail -1 "$OUT/$name.log" | cut -c1-120))"
  fi
}
run l-s42 42
run l-s43 43
run l-s44 44
echo "[$(date +%H:%M)] 3개 모두 끝"
'''

README = """폐 영역 모델 학습 추가 묶음 (aug_bundle 폴더 안에 푼다)
1) cd ~/aug_bundle && tar xzf ~/lung_bundle.tgz --strip-components=1
2) nohup bash run_lung.sh > run_lung.log 2>&1 &
3) 진행 보기:  tail -f run_lung.log      (Ctrl+C 로 화면만 나가도 학습은 계속된다)
끝난 실험마다 results_l-s4x.tgz 가 생기고 S3 로 올라간다. 다 끝나면 공간을 꼭 끈다.
"""


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    cfg = json.loads((ROOT / "aws_config.json").read_text(encoding="utf-8"))
    results_s3 = f"{cfg['output_s3']}aug_runs/"
    for sp in ("train", "val"):
        if not (MASKS / f"{sp}_lung_masks.npz").is_file():
            raise SystemExit(f"{MASKS}/{sp}_lung_masks.npz 가 없습니다 (xrv_mask_split.py 를 먼저 돌리세요)")
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    (BUNDLE / "masks").mkdir(parents=True)
    shutil.copy2(ROOT / "training/lung_prep.py", BUNDLE / "lung_prep.py")
    shutil.copy2(ROOT / "src/make_lung_images.py", BUNDLE / "make_lung_images.py")
    for sp in ("train", "val"):
        shutil.copy2(MASKS / f"{sp}_lung_masks.npz", BUNDLE / "masks" / f"{sp}_lung_masks.npz")
    (BUNDLE / "run_lung.sh").write_text(RUN.replace("__S3_RESULTS__", results_s3), encoding="utf-8")
    (BUNDLE / "README_lung.txt").write_text(README, encoding="utf-8")
    TGZ.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(TGZ, "w:gz") as tar:
        tar.add(BUNDLE, arcname="lung_bundle")
    print(f"묶음 만듦: {TGZ.relative_to(ROOT)} ({TGZ.stat().st_size / 1e6:.0f} MB, sha256 {sha256(TGZ)[:12]}…)")
    for p in sorted(BUNDLE.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(BUNDLE)}  {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
