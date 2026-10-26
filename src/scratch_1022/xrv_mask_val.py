import csv, sys, time, numpy as np, torch
from pathlib import Path
from PIL import Image
import torchxrayvision as xrv
ROOT=Path('~/Documents/p1'); OUT=Path(sys.argv[1])
rows=[r for r in csv.DictReader(open(ROOT/'manifests/split_manifest.csv',encoding='utf-8')) if r['split']=='val']
model=xrv.baseline_models.chestx_det.PSPNet().eval(); li=[model.targets.index('Left Lung'),model.targets.index('Right Lung')]
dev=torch.device('mps'); model=model.to(dev)
masks=np.zeros((len(rows),512,512),dtype=bool); t0=time.time(); B=8
for i in range(0,len(rows),B):
    xs=[xrv.datasets.normalize(np.asarray(Image.open(Path.home()/'data/chest_xray'/r['relative_path']).convert('L').resize((512,512),Image.BILINEAR),dtype=np.float32),255)[None] for r in rows[i:i+B]]
    with torch.no_grad(): p=torch.sigmoid(model(torch.from_numpy(np.stack(xs)).to(dev)))[:,li].cpu()
    masks[i:i+len(xs)]=((p[:,0]>0.5)|(p[:,1]>0.5)).numpy()
    if (i//B)%16==0: print(f'{i+len(xs)}/{len(rows)} {time.time()-t0:.0f}초',flush=True)
np.savez_compressed(OUT/'val_lung_masks.npz',masks=np.packbits(masks.reshape(len(rows),-1),axis=1),paths=np.array([r['relative_path'] for r in rows]))
print('저장 완료', round(time.time()-t0),'초; 폐 면적 평균',round(float(masks.mean()),3),'; 폐가 거의 없는 사진(면적<5%):',int((masks.mean((1,2))<0.05).sum()),flush=True)
