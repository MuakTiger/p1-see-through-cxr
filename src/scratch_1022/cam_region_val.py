import sys, csv, numpy as np, torch
from pathlib import Path
from PIL import Image
ROOT=Path('~/Documents/p1')
sys.path.insert(0,str(ROOT/'training')); sys.path.insert(0,str(ROOT/'src'))
import train as T, gradcam as G
from shift_eval import load_model
SIZE=512; G.SIZE=SIZE
G.MASKS=G.region_masks(SIZE); G.AREA={k:float(m.mean()) for k,m in G.MASKS.items()}
G._YC=((np.arange(SIZE)+0.5)/SIZE)[:,None].repeat(SIZE,1).ravel()
torch.set_num_threads(6)
tf=T.build_transform('P0',SIZE)
rows=[r for r in csv.DictReader(open(ROOT/'manifests/split_manifest.csv',encoding='utf-8')) if r['split']=='val']
def run(model_dir):
    cf=G.GradCAM(load_model(model_dir)); out=[]
    for i in range(0,len(rows),16):
        b=rows[i:i+16]
        x=torch.stack([tf(Image.open(Path.home()/'data/chest_xray'/r['relative_path']).convert('RGB')) for r in b])
        lg,cams,valid=cf(x,[None])
        f=G.region_fractions(cams[0])
        for j,r in enumerate(b):
            out.append((int(r['label_id']),int(lg[j].argmax()),f['center'][j],f['border'][j],f['corner'][j],f['band_bottom'][j],f['band_top'][j],f['active'][j]))
    return np.array(out,dtype=float)
print('기대(고르게 퍼졌을 때):',{k:round(v,3) for k,v in G.AREA.items() if k in('center','border','corner','band_bottom','band_top')})
for name,d in [('B512','runs/results/b512-20261022-111844/model'),('v-s42','runs/results/v-s42-20261022-jlab/model')]:
    a=run(ROOT/d)
    print(f'\n[{name}] val {len(a)}장 평균: 중앙 {np.nanmean(a[:,2]):.3f} 테두리 {np.nanmean(a[:,3]):.3f} 위모서리 {np.nanmean(a[:,4]):.3f} 아래띠 {np.nanmean(a[:,5]):.3f} 위띠 {np.nanmean(a[:,6]):.3f} 활성면적 {np.nanmean(a[:,7]):.3f}')
    ok=a[:,0]==a[:,1]
    for nm,m in [('맞힌',ok),('틀린',~ok)]:
        print(f'   {nm} {m.sum()}장: 중앙 {np.nanmean(a[m,2]):.3f} 테두리 {np.nanmean(a[m,3]):.3f}')
    for t,n in enumerate(['정상','세균성','바이러스성']):
        m=a[:,0]==t; print(f'   정답 {n}: 중앙 {np.nanmean(a[m,2]):.3f} 테두리 {np.nanmean(a[m,3]):.3f}')
