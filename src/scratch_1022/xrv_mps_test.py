import csv, sys, time, numpy as np, torch
from pathlib import Path
from PIL import Image
import torchxrayvision as xrv
ROOT=Path('~/Documents/p1'); SC=Path(sys.argv[1])
rows=list(csv.DictReader(open(ROOT/'reports/equipment_v2/gradcam/B512/examples.csv',encoding='utf-8')))
ref=np.load(SC/'xrv_out/masks12.npy')
model=xrv.baseline_models.chestx_det.PSPNet().eval(); li=[model.targets.index('Left Lung'),model.targets.index('Right Lung')]
dev=torch.device('mps'); model=model.to(dev)
X=np.stack([xrv.datasets.normalize(np.asarray(Image.open(Path.home()/'data/chest_xray'/r['image']).convert('L').resize((512,512),Image.BILINEAR),dtype=np.float32),255)[None] for r in rows])
x=torch.from_numpy(X)
with torch.no_grad():
    model(x[:2].to(dev)); torch.mps.synchronize()          # 예열
    t=time.time(); outs=[]
    for i in range(0,len(x),4): outs.append(torch.sigmoid(model(x[i:i+4].to(dev)))[:,li].cpu())
    torch.mps.synchronize(); dt=time.time()-t
p=torch.cat(outs); m=((p[:,0]>0.5)|(p[:,1]>0.5)).numpy()
iou=[(m[i]&ref[i]).sum()/max((m[i]|ref[i]).sum(),1) for i in range(len(m))]
print(f'MPS 장당 {dt/len(x):.3f}초 (배치 4), CPU 결과와 폐 마스크 IoU 최소 {min(iou):.4f} 평균 {np.mean(iou):.4f}')
