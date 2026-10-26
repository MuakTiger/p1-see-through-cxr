"""예방접종 모델(필터 증강)과 기준 모델을 SageMaker JupyterLab(ml.g4dn.xlarge) 터미널에서 돌리기 위한 묶음을 만든다.

묶음(runs/bundles/aug_bundle.tgz): 학습 코드(train_aug.py, preprocess.py, shift_filters.py, equipment_direct.py), split_manifest.csv,
ImageNet 가중치, 점검 스크립트(check.sh), 실행 스크립트(run_all.sh). 계획은 reports/equipment_v2/vaccine_plan.md.
결과를 S3 로 올리는 주소는 aws_config.json 에서 읽어 run_all.sh 안에 적는다 (화면에는 출력하지 않는다).

사용 예 (p1 폴더에서):
    .venv/bin/python src/build_aug_bundle.py
"""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "runs/bundles/aug_bundle"
TGZ = ROOT / "runs/bundles/aug_bundle.tgz"
CODE = ["train_aug.py", "preprocess.py", "shift_filters.py", "equipment_direct.py"]

CHECK = r'''#!/bin/bash
# 환경 점검: GPU, 파이썬 라이브러리, 디스크, 사진 내려받기 권한, 결과 올리기 권한. 아무것도 설치하거나 바꾸지 않는다.
cd "$(dirname "$0")"
echo "== GPU"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -2
echo "== 파이썬 라이브러리"
python - <<'PY'
import importlib
for name in ("torch", "torchvision", "numpy", "PIL", "cv2"):
    try:
        m = importlib.import_module(name)
        print(f"  {name:12s} OK  {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"  {name:12s} 없음 ({type(e).__name__})")
try:
    import torch
    print("  CUDA 사용 가능:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception:
    pass
PY
echo "== 디스크 (여유가 5GB 이상이면 좋다)"; df -h . | tail -1
echo "== 사진 내려받기 권한"
if aws s3 ls s3://aws-say3-dataset1/pneumonia-xray/chest_xray/ 2>&1 | head -3; then :; fi
echo "== 결과 올리기 권한"
echo ok > /tmp/_probe.txt
if aws s3 cp /tmp/_probe.txt "__S3_RESULTS__probe.txt" --only-show-errors 2>&1; then echo "  결과 올리기 OK"; aws s3 rm "__S3_RESULTS__probe.txt" --only-show-errors 2>&1; else echo "  결과 올리기 실패: 학습이 끝나면 results_*.tgz 를 파일 목록에서 직접 내려받으면 됩니다"; fi
echo "== 끝. 위에서 '없음'이나 '실패'가 있으면 그 줄을 알려 주세요."
'''

RUN_ALL = r'''#!/bin/bash
# 기준(증강 없음)과 예방접종(필터 증강)을 seed 42·43·44로 번갈아 학습한다. 끝난 실험은 건너뛰므로 중간에 꺼져도 다시 실행하면 이어진다.
# 사용: nohup bash run_all.sh > run_all.log 2>&1 &   그리고  tail -f run_all.log
cd "$(dirname "$0")"
DATA=./data
OUT=./out
mkdir -p "$OUT"
if [ ! -d "$DATA/train" ]; then
  echo "[$(date +%H:%M)] 사진 내려받기 (약 1.2GB)"
  aws s3 sync s3://aws-say3-dataset1/pneumonia-xray/chest_xray/ "$DATA" --only-show-errors || { echo "사진 내려받기 실패: check.sh 결과를 알려 주세요"; exit 1; }
fi
run() {
  local name="$1" seed="$2"; shift 2
  if [ -f "$OUT/$name/output/experiment_summary.json" ]; then echo "[$(date +%H:%M)] 건너뜀(이미 끝남): $name"; return; fi
  echo "[$(date +%H:%M)] 시작: $name"
  python train_aug.py --images-dir "$DATA" --manifests-dir ./manifests --weights-dir ./weights \
    --model-dir "$OUT/$name/model" --output-dir "$OUT/$name/output" --experiment-id "$name" \
    --train-mode layer4 --input-size 512 --epochs 30 --batch-size 16 --lr 0.001 --lr-finetune 0.0001 \
    --num-workers 4 --seed "$seed" "$@" > "$OUT/$name.log" 2>&1
  if [ -f "$OUT/$name/output/experiment_summary.json" ]; then
    echo "[$(date +%H:%M)] 끝: $name  ($(tail -1 "$OUT/$name.log" | cut -c1-90))"
    tar czf "results_$name.tgz" -C "$OUT" "$name" "$name.log"
    aws s3 cp "results_$name.tgz" "__S3_RESULTS__results_$name.tgz" --only-show-errors 2>&1 && echo "  S3 에 올림" || echo "  S3 올리기 실패: results_$name.tgz 를 파일 목록에서 내려받으세요"
  else
    echo "[$(date +%H:%M)] 실패: $name (out/$name.log 마지막 줄: $(tail -1 "$OUT/$name.log" | cut -c1-120))"
  fi
}
run v-s42 42 --augment equip
run b-s42 42 --augment none
run v-s43 43 --augment equip
run b-s43 43 --augment none
run v-s44 44 --augment equip
run b-s44 44 --augment none
echo "[$(date +%H:%M)] 6개 모두 끝"
'''

README = """예방접종 모델(필터 증강) 학습 묶음
1) 이 폴더에서:  bash check.sh        (환경 점검, 설치·변경 없음)
2) 이상이 없으면:  nohup bash run_all.sh > run_all.log 2>&1 &
3) 진행 보기:      tail -f run_all.log     (Ctrl+C 로 화면만 나가도 학습은 계속된다)
끝난 실험마다 results_<이름>.tgz 가 생기고 S3 로 올라간다. 다 끝나면 공간을 꼭 끈다.
"""


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    cfg = json.loads((ROOT / "aws_config.json").read_text(encoding="utf-8"))
    results_s3 = f"{cfg['output_s3']}aug_runs/"
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    (BUNDLE / "manifests").mkdir(parents=True)
    (BUNDLE / "weights").mkdir()
    for f in CODE:
        shutil.copy2(ROOT / "training" / f, BUNDLE / f)
    shutil.copy2(ROOT / "manifests/split_manifest.csv", BUNDLE / "manifests/split_manifest.csv")
    shutil.copy2(ROOT / "weights/resnet18_imagenet.pt", BUNDLE / "weights/resnet18_imagenet.pt")
    (BUNDLE / "check.sh").write_text(CHECK.replace("__S3_RESULTS__", results_s3), encoding="utf-8")
    (BUNDLE / "run_all.sh").write_text(RUN_ALL.replace("__S3_RESULTS__", results_s3), encoding="utf-8")
    (BUNDLE / "README.txt").write_text(README, encoding="utf-8")
    TGZ.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(TGZ, "w:gz") as tar:
        tar.add(BUNDLE, arcname="aug_bundle")
    print(f"묶음 만듦: {TGZ.relative_to(ROOT)} ({TGZ.stat().st_size / 1e6:.0f} MB, sha256 {sha256(TGZ)[:12]}…)")
    for p in sorted(BUNDLE.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(BUNDLE)}  {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
