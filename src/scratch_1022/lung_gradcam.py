import sys, csv, glob, numpy as np, torch, cv2
from pathlib import Path
from PIL import Image
ROOT=Path('~/Documents/p1'); sys.path.insert(0,str(ROOT/'training')); sys.path.insert(0,str(ROOT/'src'))
import lung_prep as LP, gradcam as G, shift_eval as SE
from shift_filters import to_gray
name=sys.argv[1]; S=512; G.SIZE=S; torch.set_num_threads(4)
model_dir=sorted(glob.glob(str(ROOT/f'runs/results/{name}-*')))[0]+'/model'
cf=G.GradCAM(SE.load_model(model_dir))
rows=[r for r in csv.DictReader(open(ROOT/'manifests/split_manifest.csv',encoding='utf-8')) if r['split']=='val']
raw=LP.load_masks(ROOT/'runs/lung_masks/val_lung_masks.npz',S)
y=np.array([int(r['label_id']) for r in rows])
ins=np.zeros(len(rows)); band_f=np.zeros(len(rows)); band_a=np.zeros(len(rows)); lung_f=np.zeros(len(rows)); lung_a=np.zeros(len(rows)); peak=np.zeros(len(rows),bool); pred=np.zeros(len(rows),int)
store={}
want={r['image']:i for i,r in enumerate(csv.DictReader(open(ROOT/'reports/equipment_v2/gradcam/B512/examples.csv',encoding='utf-8')))}
ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(33,33))
for i0 in range(0,len(rows),16):
    chunk=rows[i0:i0+16]; grays=[]; hulls=[]
    for r in chunk:
        g=to_gray(Image.open(Path.home()/'data/chest_xray'/r['relative_path']),S); h,ok=LP.hull_mask(raw[r['relative_path']]); grays.append(np.where(h,g,0).astype(np.uint8)); hulls.append(h)
    X=SE.gray_to_tensor(np.stack(grays)); lg,cams,_=cf(X,[None]); c=cams[0]
    for j,r in enumerate(chunk):
        k=i0+j; h=hulls[j]; m=raw[r['relative_path']]; s=c[j].sum(); pred[k]=int(lg[j].argmax())
        er=cv2.erode(h.astype(np.uint8),ker).astype(bool); band=h&~er
        ins[k]=(c[j]*h).sum()/s; band_f[k]=(c[j]*band).sum()/s; band_a[k]=band.sum()/h.sum()
        lung_f[k]=(c[j]*m).sum()/s; lung_a[k]=m.sum()/h.sum(); peak[k]=m.ravel()[c[j].argmax()]
        if r['relative_path'] in want: store[r['relative_path']]=(grays[j],c[j],m,h,int(lg[j].argmax()),torch.softmax(lg[j],0).numpy())
ok=pred==y
print(f'[{name}] val {len(rows)}장, 원본 val F1 은 학습 기록과 같은 입력')
print(f' 지도 중 껍질 안 비율 {ins.mean():.3f} (껍질 밖은 검정이라 거의 1이어야 함)')
print(f' 지도 중 껍질 윤곽 띠(안쪽 16px) 비율 {band_f.mean():.3f} (그 띠의 면적 비율 {band_a.mean():.3f})')
print(f' 지도 중 폐 마스크 안 비율 {lung_f.mean():.3f} (껍질 안에서 폐 마스크 면적 비율 {lung_a.mean():.3f}), 최고점이 폐 마스크 안 {peak.mean():.1%} (고르게 퍼졌다면 약 {lung_a.mean():.0%})')
for nm,m in [('맞힌',ok),('틀린',~ok),('정상',y==0),('세균성',y==1),('바이러스성',y==2)]:
    print(f'   {nm:6s} {m.sum():4d}장: 폐 마스크 안 {lung_f[m].mean():.3f} (면적 {lung_a[m].mean():.3f}), 최고점이 폐 안 {peak[m].mean():.1%}, 윤곽 띠 {band_f[m].mean():.3f} (면적 {band_a[m].mean():.3f})')
plt=G.setup_matplotlib(); ex=list(csv.DictReader(open(ROOT/'reports/equipment_v2/gradcam/B512/examples.csv',encoding='utf-8')))
fig,ax=plt.subplots(3,4,figsize=(12,10.6)); CL=['정상','세균성','바이러스성']
for i,r in enumerate(ex):
    g,c,m,h,p,pr=store[r['image']]; a=ax[i//4,i%4]; a.imshow(g,cmap='gray',vmin=0,vmax=255); a.imshow(np.ma.masked_where(~h,c),cmap='jet',alpha=0.45,vmin=0,vmax=1)
    a.contour(m.astype(float),levels=[0.5],colors='white',linewidths=1.2)
    t=CL[[n for n in ('정상','세균성','바이러스성')].index(r['true'])]; okp=(CL[p]==r['true'])
    a.set_title(f"정답 {r['true']} → 예측 {CL[p]} ({'맞음' if okp else '틀림'})\n정상 {pr[0]:.2f} · 세균 {pr[1]:.2f} · 바이러스 {pr[2]:.2f}",fontsize=8.5)
    for sp in a.spines.values(): sp.set_edgecolor('#2a7a3b' if okp else '#c0392b'); sp.set_linewidth(2.5)
    a.set_xticks([]); a.set_yticks([])
fig.suptitle(f'{name} (폐 영역 모델) Grad-CAM (val, 같은 12장, 흰 선 = 폐 마스크, 모델 입력은 폐 껍질 밖이 검정)',fontsize=12)
fig.subplots_adjust(left=0.01,right=0.99,bottom=0.01,top=0.93,hspace=0.2,wspace=0.04)
out=ROOT/f'reports/equipment_v2/lung_model/gradcam_{name}.png'; fig.savefig(out,dpi=105); print('저장',out)
