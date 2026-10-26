import sys, csv, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
ROOT=Path('~/Documents/p1')
sys.path.insert(0,str(ROOT/'training')); sys.path.insert(0,str(ROOT/'src'))
import train as T, gradcam as G
from shift_eval import load_model
S=512; G.SIZE=S; torch.set_num_threads(6)
tf=T.build_transform('P0',S)
rows=[r for r in csv.DictReader(open(ROOT/'manifests/split_manifest.csv',encoding='utf-8')) if r['split']=='val']
rng=np.random.default_rng(42); pick=rng.permutation(len(rows))[:300]
model=load_model(ROOT/'runs/results/b512-20261022-111844/model'); cf=G.GradCAM(model)
X=torch.stack([tf(Image.open(Path.home()/'data/chest_xray'/rows[i]['relative_path']).convert('RGB')) for i in pick])
# (1) 직접 CAM: 마지막 블록 출력 A (N,512,16,16), fc 가중치 w[class] -> relu(sum_c w_c A_c)
feats={}
h=model.layer4[-1].register_forward_hook(lambda m,i,o: feats.__setitem__('a',o.detach()))
with torch.no_grad(): lg=model(X)
h.remove()
pred=lg.argmax(1); w=model.fc.weight.detach()[pred]               # (N,512)
cam_d=F.relu((w[:,:,None,None]*feats['a']).sum(1,keepdim=True))
cam_d=F.interpolate(cam_d,size=(S,S),mode='bilinear',align_corners=False)[:,0]
lo=cam_d.amin((1,2),keepdim=True); sp=cam_d.amax((1,2),keepdim=True)-lo
cam_d=((cam_d-lo)/sp.clamp_min(1e-12)).numpy()
_,cams,_=cf(X,[None]); cam_g=cams[0]
cors=[np.corrcoef(cam_d[i].ravel(),cam_g[i].ravel())[0,1] for i in range(len(X))]
print(f'(1) 직접 CAM 대 gradcam.py 지도: 상관 최소 {np.nanmin(cors):.6f}, 최대 절대차 {np.abs(cam_d-cam_g).max():.2e}  ({len(X)}장)')
# (2) 가리기: 16x16 칸(32px), 40칸(약 16%)을 사진 평균색으로 가림
G=16; cell=S//G
cs=cam_g.reshape(len(X),G,cell,G,cell).mean((2,4)).reshape(len(X),-1)
mean_val=X.mean((2,3),keepdim=True)
def run(order_fn):
    Xm=X.clone()
    for i in range(len(X)):
        idx=order_fn(cs[i])
        for k in idx:
            r,c=divmod(int(k),G); Xm[i,:,r*cell:(r+1)*cell,c*cell:(c+1)*cell]=mean_val[i]
    with torch.no_grad(): lm=model(Xm)
    p0=torch.softmax(lg,1).gather(1,pred[:,None])[:,0]; p1=torch.softmax(lm,1).gather(1,pred[:,None])[:,0]
    return (p1-p0).mean().item(), (lm.argmax(1)!=pred).float().mean().item()
K=40; r2=np.random.default_rng(0)
for nm,fn in [('지도 높은 칸',lambda s:np.argsort(-s)[:K]),('무작위 칸',lambda s:r2.permutation(len(s))[:K]),('지도 낮은 칸',lambda s:np.argsort(s)[:K])]:
    dp,ch=run(fn); print(f'(2) {nm} {K}칸 가림: 예측 확률 변화 {dp:+.3f}, 예측이 바뀐 비율 {ch:.1%}')
# 어디가 높은가: 가장 높은 칸이 테두리(바깥 2칸 띠)에 있는 비율
top=cs.argmax(1); r,c=top//G,top%G; edge=((r<2)|(r>=G-2)|(c<2)|(c>=G-2)).mean()
print(f'    가장 높은 칸이 바깥 2칸 띠에 있는 사진: {edge:.1%} (사진 면적에서 바깥 띠 비율 {1-(12/16)**2:.1%})')
