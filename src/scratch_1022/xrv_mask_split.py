"""split(train/val/test/check)의 사진마다 PSPNet 폐 마스크(좌·우 폐의 합, 시그모이드 0.5)를 만든다. 입력은 우리 모델 입력과 같은 찌그러뜨린 512 회색 사진.
사용: xrv_venv/bin/python -I xrv_mask_split.py <출력 폴더> <split>   (노트북 GPU mps, 라벨은 읽지 않는다)"""
import csv, sys, time, numpy as np, torch
from pathlib import Path
from PIL import Image
import torchxrayvision as xrv

ROOT = Path('~/Documents/p1')
OUT, SPLIT = Path(sys.argv[1]), sys.argv[2]
rows = [r for r in csv.DictReader(open(ROOT / 'manifests/split_manifest.csv', encoding='utf-8')) if r['split'] == SPLIT]
model = xrv.baseline_models.chestx_det.PSPNet().eval()
li = [model.targets.index('Left Lung'), model.targets.index('Right Lung')]
dev = torch.device('mps')
model = model.to(dev)
masks = np.zeros((len(rows), 512, 512), dtype=bool)
t0 = time.time()
B = 8
for i in range(0, len(rows), B):
    xs = [xrv.datasets.normalize(np.asarray(Image.open(Path.home() / 'data/chest_xray' / r['relative_path']).convert('L').resize((512, 512), Image.BILINEAR), dtype=np.float32), 255)[None]
          for r in rows[i:i + B]]
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(np.stack(xs)).to(dev)))[:, li].cpu()
    masks[i:i + len(xs)] = ((p[:, 0] > 0.5) | (p[:, 1] > 0.5)).numpy()
    if (i // B) % 32 == 0:
        print(f'{i + len(xs)}/{len(rows)} {time.time() - t0:.0f}초', flush=True)
np.savez_compressed(OUT / f'{SPLIT}_lung_masks.npz', masks=np.packbits(masks.reshape(len(rows), -1), axis=1),
                    paths=np.array([r['relative_path'] for r in rows]))
print('저장 완료', round(time.time() - t0), '초; 폐 면적 평균', round(float(masks.mean()), 3), '; 면적<5% 사진:', int((masks.mean((1, 2)) < 0.05).sum()), flush=True)
