import csv, sys, time, numpy as np, torch
from pathlib import Path
from PIL import Image
import torchxrayvision as xrv
ROOT=Path('~/Documents/p1'); OUT=Path(sys.argv[1])
rows=list(csv.DictReader(open(ROOT/'reports/equipment_v2/gradcam/B512/examples.csv',encoding='utf-8')))
t0=time.time(); model=xrv.baseline_models.chestx_det.PSPNet(); model.eval()
print('모델 로드', round(time.time()-t0,1),'초; 출력 대상:', model.targets)
li=[model.targets.index('Left Lung'), model.targets.index('Right Lung')]
masks=[]; ts=[]
for r in rows:
    im=Image.open(Path.home()/'data/chest_xray'/r['image']).convert('L').resize((512,512),Image.BILINEAR)  # 우리 모델 입력과 같은 찌그러뜨린 512 사진
    a=xrv.datasets.normalize(np.asarray(im,dtype=np.float32),255)[None,None]  # [-1024,1024] 범위
    t=time.time()
    with torch.no_grad(): p=torch.sigmoid(model(torch.from_numpy(a)))[0]
    ts.append(time.time()-t)
    m=(p[li[0]]>0.5)|(p[li[1]]>0.5); masks.append(m.numpy())
    print(f"{r['image'].split('/')[-1]:40s} 폐 면적 {m.float().mean():.3f}  좌 {float((p[li[0]]>0.5).float().mean()):.3f} 우 {float((p[li[1]]>0.5).float().mean()):.3f}")
np.save(OUT/'masks12.npy',np.stack(masks)); print('장당 시간(초)', round(float(np.mean(ts[1:])),3))
