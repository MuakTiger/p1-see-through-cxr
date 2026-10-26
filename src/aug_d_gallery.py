"""D(노이즈 증강)가 적용된 학습 사진을 눈으로 보는 그림 (train 사진만 사용, 학습·평가에는 영향 없음).

그림 1 d_noise_levels.png   클래스별 사진 1장에 노이즈 세기를 바꿔 가며 입힌 모습
                            (0 → D 가 쓰는 범위 3·6·10 → 장비 차이 시험의 가장 센 16, 16 은 학습에 쓰지 않는다)
그림 2 d_training_draws.png 실제 학습 변환(train.build_train_transform, p=0.5, std 0~10)을 같은 사진에 여러 번 적용한 모습과 확대

노이즈 식은 train.py 의 RandomGrayNoise 와 같다: 한 장짜리 가우시안 노이즈(std/255)를 세 채널에 똑같이 더하고 0~1 로 자른다.
측정한 노이즈 세기는 (노이즈 입힌 사진 − 원본)의 표준편차(0~255 기준)이며, 0·255 에 걸려 잘린 픽셀 때문에 목표보다 조금 작다.

사용 예 (p1 폴더에서):
    .venv/bin/python src/aug_d_gallery.py --manifest manifests/split_manifest.csv --images ~/data/chest_xray --out reports/aug_D
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "src"))
import train as T  # noqa: E402
from gradcam import setup_matplotlib  # noqa: E402

SEED = 42
LEVELS = [0, 3, 6, 10, 16]  # 0 = 원본, 10 = D 의 상한, 16 = 장비 차이 시험의 가장 센 단계
CLASSES = [(0, "정상"), (1, "세균성"), (2, "바이러스성")]
CROP = (slice(120, 184), slice(80, 144))  # 확대해서 볼 64×64 영역 (224 입력 기준, 폐 부근)


def load_tensor(path):
    """모델이 보는 것과 같은 크기: RGB → 224×224 → 0~1 (정규화 전)."""
    with Image.open(path) as im:
        return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])(im.convert("RGB"))


def add_noise(x, std255, gen):
    """RandomGrayNoise 와 같은 식. std255 는 0~255 기준."""
    noise = torch.randn(1, x.shape[1], x.shape[2], generator=gen) * std255 / 255.0
    return (x + noise).clamp(0.0, 1.0)


def measured_std(noisy, clean):
    return float(((noisy - clean)[0] * 255.0).std())


def denorm(x):
    """정규화된 (3,224,224) → 0~1 회색 (224,224). 모델이 실제로 본 값."""
    mean = torch.tensor(T.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(T.IMAGENET_STD).view(3, 1, 1)
    return (x * std + mean)[0].clamp(0, 1)


def pick_images(manifest, images_root):
    m = pd.read_csv(manifest)
    train = m[m.split == "train"]
    rng = np.random.default_rng(SEED)
    out = []
    for lab, name in CLASSES:
        rows = train[train.label_id == lab]
        out.append((name, images_root / rows.iloc[int(rng.integers(len(rows)))].relative_path))
    return out


def fig_levels(plt, picks, out):
    fig, axes = plt.subplots(3, len(LEVELS), figsize=(3.1 * len(LEVELS), 9.6))
    print("노이즈 세기별 측정값 (목표 → 측정, 0~255 기준)")
    for r, (name, path) in enumerate(picks):
        x = load_tensor(path)
        row = []
        for c, s in enumerate(LEVELS):
            gen = torch.Generator().manual_seed(SEED)
            noisy = add_noise(x, s, gen) if s else x
            m = measured_std(noisy, x)
            row.append(f"{s}→{m:.1f}")
            ax = axes[r, c]
            ax.imshow(noisy[0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title("원본" if s == 0 else f"노이즈 std {s}" + ("\n(D 의 상한)" if s == 10 else "\n(시험용, 학습에 없음)" if s == 16 else ""),
                             fontsize=11, color="#A3500A" if s == 16 else "black")
            if c == 0:
                ax.set_ylabel(name, fontsize=13)
        print(f"  {name}: " + "  ".join(row))
    fig.suptitle("D 가 입히는 노이즈 (학습 사진 50%, std 0~10 에서 무작위, 224×224 입력)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "d_noise_levels.png", dpi=130)
    plt.close(fig)


def fig_draws(plt, path, out, n_draws=6):
    """실제 학습 변환으로 같은 사진을 여러 번 뽑는다. 사진마다 확률 0.5 로 노이즈가 없거나 std 0~10."""
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    clean = denorm(T.build_transform("P0")(rgb))  # 노이즈 없는 기준 (정규화 → 되돌림이라 float 오차만 있음)
    train_tf = T.build_train_transform("P0", "noise", 10.0, 0.5)
    torch.manual_seed(SEED)
    draws = [denorm(train_tf(rgb)) for _ in range(n_draws)]
    fig, axes = plt.subplots(2, n_draws + 1, figsize=(2.7 * (n_draws + 1), 6.2))
    stds = []
    for c, d in enumerate([clean] + draws):
        none = c == 0 or float((d - clean).abs().max()) < 1e-4
        s = 0.0 if none else float(((d - clean) * 255.0).std())
        if c > 0:
            stds.append(s)
        title = "원본" if c == 0 else ("원본 그대로\n(노이즈 안 걸림)" if none else f"노이즈 std {s:.1f}")
        axes[0, c].imshow(d.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(title, fontsize=10)
        axes[1, c].imshow(d.numpy()[CROP], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        for ax in axes[:, c]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("전체", fontsize=12)
    axes[1, 0].set_ylabel("64×64 확대", fontsize=12)
    fig.suptitle("실제 학습 변환으로 같은 사진을 6번 뽑은 모습 (매번 다르다)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "d_training_draws.png", dpi=130)
    plt.close(fig)
    print("학습 변환 6번 뽑기, 측정 std (0 = 노이즈 안 걸림):", " ".join(f"{s:.1f}" for s in stds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    picks = pick_images(Path(args.manifest).expanduser(), Path(args.images).expanduser())
    for name, p in picks:
        print(f"{name}: {p.relative_to(Path(args.images).expanduser())}")
    plt = setup_matplotlib()
    fig_levels(plt, picks, out)
    fig_draws(plt, picks[0][1], out)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
