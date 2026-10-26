"""ResNet18 ImageNet 가중치를 resnet18_imagenet.pt 로 저장한다 (가이드 6·8장).

SageMaker 학습 인스턴스는 인터넷에서 가중치를 내려받지 않으므로,
여기서 한 번 받아 state_dict 로 저장한 뒤 S3 weights 경로에 올려 둔다.

사용 예:
    python src/save_resnet18_weights.py --out weights

결과:
    weights/resnet18_imagenet.pt    state_dict (학습 코드에서 weights_only=True 로 읽음)
    weights/resnet18_imagenet.json  출처·버전·sha256 기록
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torchvision
from torchvision import models


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="가중치를 저장할 폴더")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / "resnet18_imagenet.pt"

    # torchvision 공식 ImageNet 가중치를 내려받아 state_dict 만 저장한다.
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    torch.save(model.state_dict(), pt_path)

    # 저장한 파일이 가이드의 코드 그대로 읽히는지 확인한다.
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    check = models.resnet18(weights=None)
    check.load_state_dict(state)  # 키나 모양이 다르면 여기서 오류가 난다
    n_params = sum(p.numel() for p in check.parameters())

    meta = {
        "file": pt_path.name,
        "source": "torchvision ResNet18_Weights.IMAGENET1K_V1",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "num_parameters": n_params,
        "fc_out_features": check.fc.out_features,
        "imagenet_mean": list(weights.transforms().mean),
        "imagenet_std": list(weights.transforms().std),
        "sha256": sha256_of(pt_path),
    }
    (out_dir / "resnet18_imagenet.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"저장: {pt_path} ({pt_path.stat().st_size:,} 바이트)")


if __name__ == "__main__":
    main()
