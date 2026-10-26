import sys, csv, numpy as np, cv2
from pathlib import Path
from PIL import Image
ROOT=Path('~/Documents/p1'); sys.path.insert(0,str(ROOT/'training')); sys.path.insert(0,str(ROOT/'src'))
import lung_prep as LP, gradcam as G
from shift_filters import to_gray
rows={r['relative_path']:r for r in csv.DictReader(open(ROOT/'manifests/split_manifest.csv',encoding='utf-8'))}
M={}
for sp in ('train','val','test'):
    for k,v in LP.load_masks((ROOT/f'runs/lung_masks/{sp}_lung_masks.npz') if (ROOT/f'runs/lung_masks/{sp}_lung_masks.npz').is_file() else Path(f'<임시 폴더>'),512).items(): M[k]=v
info=[]
for k,m in M.items():
    a=m.mean(); h,_=LP.hull_mask(m,0) if a>=LP.MIN_AREA else (m,False)
    info.append((k,a,(m.sum()/max(h.sum(),1)) if a>=LP.MIN_AREA else np.nan))
CL={'NORMAL':'정상','BACTERIAL_PNEUMONIA':'세균성','VIRAL_PNEUMONIA':'바이러스성'}
fails=[i for i in info if i[1]<LP.MIN_AREA]
ok=[i for i in info if i[1]>=LP.MIN_AREA]
print(f'전체 {len(info)}장 (train+val+test), 규칙 실패(면적<5%) {len(fails)}장:',[(rows[k]["split"],CL[rows[k]["label_name"]]) for k,_,_ in fails])
lowsol=sorted(ok,key=lambda x:x[2])[:6]; small=sorted(ok,key=lambda x:x[1])[:3]; big=sorted(ok,key=lambda x:-x[1])[:3]
print('솔리디티(마스크 면적/껍질 면적) 하위:',[(rows[k]['split'],CL[rows[k]['label_name']],round(s,2)) for k,_,s in lowsol])
plt=G.setup_matplotlib(); out=ROOT/'reports/equipment_v2/lung_model'; out.mkdir(parents=True,exist_ok=True)
def gray(k): return to_gray(Image.open(Path.home()/'data/chest_xray'/k),512)
# 그림 1: 규칙 실패 (원본 + 마스크)
n=len(fails); fig,ax=plt.subplots(1,n,figsize=(3.4*n,4.1),squeeze=False)
for j,(k,a,_) in enumerate(fails):
    g=gray(k); A=ax[0,j]; A.imshow(g,cmap='gray'); m=M[k]
    if m.any(): A.imshow(np.ma.masked_where(~m,m),cmap='autumn',alpha=0.7,vmin=0,vmax=1)
    A.set_xticks([]); A.set_yticks([]); A.set_title(f"{rows[k]['split']} {CL[rows[k]['label_name']]}\n폐로 잡은 면적 {a:.1%} → 전체 사진 사용",fontsize=9)
fig.suptitle('폐 분할 실패 (마스크 면적 5% 미만, 빨간 곳 = 폐로 잡힌 곳). 규칙에 따라 이 사진들은 폐 밖을 지우지 않고 전체를 씀',fontsize=11)
fig.subplots_adjust(left=0.01,right=0.99,bottom=0.01,top=0.82,wspace=0.04); fig.savefig(out/'seg_failures_rule.png',dpi=105); plt.close(fig)
# 그림 2: 의심 사례 (솔리디티 낮음, 면적 작음, 면적 큼)
sus=[('솔리디티 낮음',x) for x in lowsol]+[('면적 작음',x) for x in small]+[('면적 큼',x) for x in big]
fig,ax=plt.subplots(3,4,figsize=(12.5,10.8))
for j,(why,(k,a,s)) in enumerate(sus):
    g=gray(k); m=M[k]; h,_=LP.hull_mask(m); A=ax[j//4,j%4]; A.imshow(g,cmap='gray')
    A.imshow(np.ma.masked_where(~m,m),cmap='autumn',alpha=0.45,vmin=0,vmax=1); A.contour(h.astype(float),levels=[0.5],colors='cyan',linewidths=1.4)
    A.set_xticks([]); A.set_yticks([]); A.set_title(f"[{why}] {rows[k]['split']} {CL[rows[k]['label_name']]}\n면적 {a:.0%}, 솔리디티 {s:.2f}",fontsize=9)
for j in range(len(sus),12): ax[j//4,j%4].axis('off')
fig.suptitle('폐 분할이 의심스러운 사례 (주황 = 폐 마스크, 청록 선 = 모델이 실제로 남길 껍질 + 여백)',fontsize=12)
fig.subplots_adjust(left=0.01,right=0.99,bottom=0.01,top=0.93,hspace=0.2,wspace=0.04); fig.savefig(out/'seg_suspects.png',dpi=100); plt.close(fig)
print('저장:',out/'seg_failures_rule.png',out/'seg_suspects.png')
