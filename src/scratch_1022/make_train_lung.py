"""training/train_aug.py 를 복사해 폐 영역 입력(필터 → 폐 껍질 밖 0)만 더한 training/train_lung.py 를 만든다.
train_aug.py 는 건드리지 않는다. 각 치환은 정확히 한 번만 일어나야 하며, 아니면 멈춘다."""
import hashlib
import sys
from pathlib import Path

ROOT = Path("~/Documents/p1")
src = (ROOT / "training/train_aug.py").read_text(encoding="utf-8")
base_hash = hashlib.sha256(src.encode("utf-8")).hexdigest()
t = src


def sub(old, new):
    global t
    n = t.count(old)
    if n != 1:
        sys.exit(f"치환 대상이 {n}번 나옵니다: {old[:70]!r}")
    t = t.replace(old, new)


# 1. 머리말
sub("[train_aug.py] train.py 에 장치 설정 필터 증강 하나(--augment equip)만 더한 복사본.",
    "[train_lung.py] train_aug.py 에 폐 영역 입력 하나만 더한 복사본 (입력 = 회색 S×S → [학습만 --augment equip 필터] → 폐 껍질 밖을 0으로).\n"
    f"기준 train_aug.py 의 sha256: {base_hash}\n"
    "  폐 마스크의 볼록 껍질은 --lung-masks (기본: weights 폴더의 lung_hull_masks.npz, src/make_hull_masks.py 로 만듦)에서 읽는다.\n"
    "  필터를 먼저 걸고 그다음 폐 처리를 하는 순서다 (평가 src/equipment_eval_lung.py 와 같은 순서, 같은 lung_prep 규칙). --augment none/equip 만 지원한다.\n"
    "  [train_aug.py] train.py 에 장치 설정 필터 증강 하나(--augment equip)만 더한 복사본.")

# 2. import
sub("import equipment_direct as ED  # 같은 폴더(training/)의 equipment_direct.py. 장치 설정 필터 5종",
    "import equipment_direct as ED  # 같은 폴더(training/)의 equipment_direct.py. 장치 설정 필터 5종\n"
    "import lung_prep as LP  # 같은 폴더(training/)의 lung_prep.py. 폐 껍질 밖을 0으로 (규칙은 평가와 같음)\n"
    "from shift_filters import to_gray as _to_gray  # 평가의 필터 경로와 같은 회색 S×S 변환")

# 3. 인자
sub('    p.add_argument("--class-weight", choices=CLASS_WEIGHT_CHOICES, default="none",',
    '    p.add_argument("--lung-masks", default=None, help="폐 껍질 마스크 묶음(npz). 기본: weights 폴더의 lung_hull_masks.npz")\n'
    '    p.add_argument("--class-weight", choices=CLASS_WEIGHT_CHOICES, default="none",')

# 4. 데이터셋
sub('        return self.transform(img), row["label_id"], idx\n',
    '        return self.transform(img), row["label_id"], idx\n\n\n'
    'def load_hulls(path, size):\n'
    '    """lung_hull_masks.npz → {relative_path: 비트로 묶은 껍질 마스크}. 쓸 때 풀어서 쓴다 (메모리 절약)."""\n'
    '    z = np.load(path)\n'
    '    if z["masks"].shape[1] != size * size // 8:\n'
    '        raise SystemExit(f"마스크 크기가 입력 크기 {size} 와 다릅니다: {z[\'masks\'].shape}")\n'
    '    return {str(p): z["masks"][i] for i, p in enumerate(z["paths"])}\n\n\n'
    'class LungDataset(Dataset):\n'
    '    """train/val 행만 받는 Dataset (폐 영역 판). 원본 → 회색 S×S(평가와 같은 to_gray) → [학습만 장치 필터] → 폐 껍질 밖 0 → RGB 3채널 → ToTensor → Normalize."""\n\n'
    '    def __init__(self, rows, images_root, hulls, equip, size):\n'
    '        bad = [r["relative_path"] for r in rows if r["split"] not in ALLOWED_SPLITS]\n'
    '        if bad:\n'
    '            raise SystemExit(f"train/val 이 아닌 행이 Dataset 에 들어왔습니다: {bad[:3]}")\n'
    '        self.rows, self.root, self.hulls, self.equip, self.size = rows, Path(images_root), hulls, equip, size\n'
    '        self.tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])\n\n'
    '    def __len__(self):\n'
    '        return len(self.rows)\n\n'
    '    def __getitem__(self, idx):\n'
    '        row = self.rows[idx]\n'
    '        path = self.root / row["relative_path"]\n'
    '        try:\n'
    '            with Image.open(path) as im:\n'
    '                gray = _to_gray(im, self.size)\n'
    '        except Exception as e:\n'
    '            raise RuntimeError(f"영상을 읽지 못했습니다: {path} ({type(e).__name__}: {e})") from e\n'
    '        img = Image.fromarray(gray).convert("RGB")\n'
    '        if self.equip is not None:\n'
    '            img = self.equip(img)  # 확률 prob 로 5종 중 하나 (회색 복제 → 필터 → 회색 복제)\n'
    '        g = np.asarray(img.convert("L"))\n'
    '        hull = np.unpackbits(self.hulls[row["relative_path"]])[: self.size * self.size].reshape(self.size, self.size).astype(bool)\n'
    '        out = np.where(hull, g, 0).astype(np.uint8)\n'
    '        return self.tf(Image.fromarray(out).convert("RGB")), row["label_id"], idx\n')

# 5. main: 마스크 읽기와 데이터로더
sub('    train_loader = DataLoader(TrainValDataset(train_rows, images_dir, train_transform), shuffle=True, generator=g, **loader_kwargs)\n'
    '    val_loader = DataLoader(TrainValDataset(val_rows, images_dir, transform), shuffle=False, **loader_kwargs)',
    '    if args.augment not in ("none", "equip"):\n'
    '        raise SystemExit("train_lung.py 는 --augment none / equip 만 지원합니다.")\n'
    '    if args.input_size != 512:\n'
    '        raise SystemExit("폐 마스크가 512 기준이라 --input-size 512 만 지원합니다.")\n'
    '    lung_path = Path(args.lung_masks) if args.lung_masks else weights_dir / "lung_hull_masks.npz"\n'
    '    hulls = load_hulls(lung_path, args.input_size)\n'
    '    missing = [r["relative_path"] for r in train_rows + val_rows if r["relative_path"] not in hulls]\n'
    '    if missing:\n'
    '        raise SystemExit(f"폐 마스크가 없는 사진이 {len(missing)}장 있습니다: {missing[:3]}")\n'
    '    full = sum(1 for r in train_rows + val_rows if np.unpackbits(hulls[r["relative_path"]]).all())\n'
    '    print(f"폐 영역 입력: 껍질 마스크 {len(hulls)}장 읽음 ({lung_path.name}), 분할 실패로 전체를 남기는 사진 {full}장, 여백 {LP.DILATE_PX}px", flush=True)\n'
    '    equip = RandomEquipment(args.equip_prob, args.input_size) if args.augment == "equip" else None\n'
    '    train_loader = DataLoader(LungDataset(train_rows, images_dir, hulls, equip, args.input_size), shuffle=True, generator=g, **loader_kwargs)\n'
    '    val_loader = DataLoader(LungDataset(val_rows, images_dir, hulls, None, args.input_size), shuffle=False, **loader_kwargs)')

# 6. 기록
sub('        "equip_prob": args.equip_prob if args.augment == "equip" else None,\n',
    '        "equip_prob": args.equip_prob if args.augment == "equip" else None,\n'
    '        "lung_prep": {"masks_file": lung_path.name, "masks_sha256": sha256_of_file(lung_path), "dilate_px": LP.DILATE_PX, "min_area": LP.MIN_AREA,\n'
    '                      "order": "회색 512 (평가와 같은 to_gray) → (학습만 장치 필터) → 폐 볼록 껍질 밖을 0으로 → RGB → Normalize"},\n')

(ROOT / "training/train_lung.py").write_text(t, encoding="utf-8")
print("train_lung.py 생성, 기준 train_aug.py sha256", base_hash[:12], "…")
