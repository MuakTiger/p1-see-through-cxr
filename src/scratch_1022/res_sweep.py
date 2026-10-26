import sys, csv, numpy as np, torch
from pathlib import Path
from PIL import Image
from sklearn.metrics import f1_score, recall_score
ROOT=Path('~/Documents/p1')
sys.path.insert(0,str(ROOT/'training')); sys.path.insert(0,str(ROOT/'src'))
import train as T
from shift_eval import load_model
S=512; dev=torch.device('mps')
rows=[r for r in csv.DictReader(open(ROOT/'manifests/split_manifest.csv',encoding='utf-8')) if r['split']=='val']
y=np.array([int(r['label_id']) for r in rows])
sq=[Image.open(Path.home()/'data/chest_xray'/r['relative_path']).convert('RGB').resize((S,S),Image.BILINEAR) for r in rows]
tf=T.build_transform('P0',S)
def lowres(im,R): return im if R==S else im.resize((R,R),Image.BILINEAR).resize((S,S),Image.BILINEAR)
def predict(model,ims):
    P=[]
    with torch.no_grad():
        for i in range(0,len(ims),32):
            P.append(model(torch.stack([tf(im) for im in ims[i:i+32]]).to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(P)
for name,d in [('B512','runs/results/b512-20261022-111844/model'),('b-s42','runs/results/b-s42-20261022-jlab/model'),('v-s42','runs/results/v-s42-20261022-jlab/model')]:
    model=load_model(ROOT/d).to(dev).eval()
    print(f'\n[{name}] 512 -> R -> 512 (같은 사진을 R 픽셀로 줄였다가 512로 되키움)')
    print(' R     macroF1  recall 정상/세균/바이러스   정상 예측 비율')
    for R in (512,384,256,192,128,96):
        pred=predict(model,[lowres(im,R) for im in sq]); rec=recall_score(y,pred,average=None)
        print(f'{R:4d}   {f1_score(y,pred,average="macro"):.3f}    {rec[0]:.3f} / {rec[1]:.3f} / {rec[2]:.3f}      {(pred==0).mean():.3f}')
