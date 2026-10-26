import sys, csv, numpy as np, torch
from pathlib import Path
from PIL import Image
ROOT=Path('~/Documents/p1'); SC=Path(sys.argv[1])
sys.path.insert(0,str(ROOT/'training')); sys.path.insert(0,str(ROOT/'src'))
import train as T, gradcam as G
from shift_eval import load_model
S=512; G.SIZE=S; torch.set_num_threads(6)
rows=list(csv.DictReader(open(ROOT/'reports/equipment_v2/gradcam/B512/examples.csv',encoding='utf-8')))
masks=np.load(SC/'xrv_out/masks12.npy')
tf=T.build_transform('P0',S); cf=G.GradCAM(load_model(ROOT/'runs/results/b512-20261022-111844/model'))
X=torch.stack([tf(Image.open(Path.home()/'data/chest_xray'/r['image']).convert('RGB')) for r in rows])
lg,cams,valid=cf(X,[None]); cam=cams[0]; gray=G.denorm_gray(X)
plt=G.setup_matplotlib(); out=ROOT/'reports/equipment_v2/gradcam/B512'
for name,mode in [('lung_mask_check.png','mask'),('gradcam_with_lung_contour.png','cam')]:
    fig,ax=plt.subplots(3,4,figsize=(12,10.6))
    for i,r in enumerate(rows):
        a=ax[i//4,i%4]; a.imshow(gray[i],cmap='gray',vmin=0,vmax=255)
        if mode=='mask': a.imshow(np.ma.masked_where(~masks[i],masks[i]),cmap='autumn',alpha=0.40,vmin=0,vmax=1)
        else:
            a.imshow(cam[i],cmap='jet',alpha=0.42,vmin=0,vmax=1); a.contour(masks[i].astype(float),levels=[0.5],colors='white',linewidths=1.6)
        inside=float((cam[i]*masks[i]).sum()/max(cam[i].sum(),1e-9))
        ttl=f"{r['true']}→{r['pred']} ({'맞음' if r['correct']=='True' else '틀림'})"
        a.set_title(ttl+(f"\n지도 중 폐 안 {inside:.0%} (폐 면적 {masks[i].mean():.0%})" if mode=='cam' else f"\n폐 면적 {masks[i].mean():.0%}"),fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle('PSPNet 폐 마스크 확인 (주황 = 폐로 분할된 곳, val 12장)' if mode=='mask' else 'B512 Grad-CAM + PSPNet 폐 윤곽 (흰 선 = 폐, val 12장)',fontsize=12)
    fig.subplots_adjust(left=0.01,right=0.99,bottom=0.01,top=0.92,hspace=0.2,wspace=0.04); fig.savefig(out/name,dpi=105); plt.close(fig)
ins=[float((cam[i]*masks[i]).sum()/cam[i].sum()) for i in range(len(rows))]; ar=[float(m.mean()) for m in masks]
print(f'12장 평균: 지도 중 폐 안 비율 {np.mean(ins):.3f} (폐 면적 비율 {np.mean(ar):.3f}); 맞힌 {np.mean([v for v,r in zip(ins,rows) if r["correct"]=="True"]):.3f} / 틀린 {np.mean([v for v,r in zip(ins,rows) if r["correct"]!="True"]):.3f}')
