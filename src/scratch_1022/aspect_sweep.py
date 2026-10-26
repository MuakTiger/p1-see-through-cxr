import sys, csv, numpy as np, torch
from pathlib import Path
from PIL import Image
from sklearn.metrics import f1_score, recall_score
ROOT=Path('~/Documents/p1')
sys.path.insert(0,str(ROOT/'training')); sys.path.insert(0,str(ROOT/'src'))
import train as T
from shift_eval import load_model
S=512
dev=torch.device('mps')
rows=[r for r in csv.DictReader(open(ROOT/'manifests/split_manifest.csv',encoding='utf-8')) if r['split']=='val']
y=np.array([int(r['label_id']) for r in rows])
# 학습·평가와 같은 경로: RGB -> Resize((512,512)) 한 회색 3채널 PIL 영상을 한 번만 만들어 둔다
sq=[Image.open(Path.home()/'data/chest_xray'/r['relative_path']).convert('RGB').resize((S,S),Image.BILINEAR) for r in rows]
asp=np.array([ (lambda im:(im.size[0]/im.size[1]))(Image.open(Path.home()/'data/chest_xray'/r['relative_path'])) for r in rows])
print('val 정상 가로/세로 중앙값', np.median(asp[y==0]).round(3))
r0=np.median(asp[y==0])
tf=T.build_transform('P0',S)
def warp(im,f,mode):
    w=max(1,int(round(S*f)))
    a=im.resize((w,S),Image.BILINEAR)
    if f>=1:
        x0=(w-S)//2; return a.crop((x0,0,x0+S,S))
    arr=np.asarray(a); pad=S-w; l=pad//2; r=pad-l
    if mode=='black': out=np.pad(arr,((0,0),(l,r),(0,0)),constant_values=0)
    else: out=np.pad(arr,((0,0),(l,r),(0,0)),mode='edge')
    return Image.fromarray(out)
def predict(model,ims):
    P=[]
    with torch.no_grad():
        for i in range(0,len(ims),32):
            x=torch.stack([tf(im) for im in ims[i:i+32]]).to(dev)
            P.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(P)
for name,d in [('B512','runs/results/b512-20261022-111844/model'),('b-s42','runs/results/b-s42-20261022-jlab/model'),('v-s42','runs/results/v-s42-20261022-jlab/model')]:
    model=load_model(ROOT/d).to(dev).eval()
    print(f'\n[{name}] 효과적 비율 = {r0:.3f}/f  (f<1 은 좌우를 채움, f>1 은 좌우를 자름)')
    print('비율   채움     macroF1  recall 정상/세균/바이러스')
    for rr in (1.0,1.11,r0,1.35,1.53):
        f=r0/rr
        for mode in (('edge','black') if f<1 else ('none',)):
            pred=predict(model,[im if abs(f-1)<1e-9 else warp(im,f,mode) for im in sq])
            rec=recall_score(y,pred,average=None)
            print(f'{rr:5.2f}  {mode:5s}  {f1_score(y,pred,average="macro"):.3f}    {rec[0]:.3f} / {rec[1]:.3f} / {rec[2]:.3f}')
