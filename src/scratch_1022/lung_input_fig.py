import sys, csv, numpy as np
from pathlib import Path
from PIL import Image
ROOT=Path('~/Documents/p1'); SC=Path(sys.argv[1])
sys.path.insert(0,str(ROOT/'training')); sys.path.insert(0,str(ROOT/'src'))
import lung_prep as LP, gradcam as G
from shift_filters import to_gray
rows=list(csv.DictReader(open(ROOT/'reports/equipment_v2/gradcam/B512/examples.csv',encoding='utf-8')))
masks=np.load(SC/'xrv_out/masks12.npy'); plt=G.setup_matplotlib()
fig,ax=plt.subplots(3,4,figsize=(12,10.6)); frac=[]
for i,r in enumerate(rows):
    g=to_gray(Image.open(Path.home()/'data/chest_xray'/r['image']),512); out,ok=LP.apply_lung(g,masks[i]); frac.append((out>0).mean())
    a=ax[i//4,i%4]; a.imshow(out,cmap='gray',vmin=0,vmax=255); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"{r['true']}→{r['pred']}  남긴 면적 {(LP.hull_mask(masks[i])[0]).mean():.0%} (폐 마스크 {masks[i].mean():.0%})",fontsize=9)
fig.suptitle('폐 영역 모델이 실제로 보게 될 입력 (폐 볼록 껍질 + 여백 16px 밖은 검정, val 12장)',fontsize=12)
fig.subplots_adjust(left=0.01,right=0.99,bottom=0.01,top=0.93,hspace=0.18,wspace=0.04)
out=ROOT/'reports/equipment_v2/lung_model'; out.mkdir(parents=True,exist_ok=True); fig.savefig(out/'lung_input_examples.png',dpi=105)
print('저장', out/'lung_input_examples.png')
