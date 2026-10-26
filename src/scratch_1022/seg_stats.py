import sys, csv, numpy as np
from pathlib import Path
ROOT=Path('~/Documents/p1'); sys.path.insert(0,str(ROOT/'training'))
import lung_prep as LP
rows={r['relative_path']:r for r in csv.DictReader(open(ROOT/'manifests/split_manifest.csv',encoding='utf-8'))}
CL={'NORMAL':'정상','BACTERIAL_PNEUMONIA':'세균성','VIRAL_PNEUMONIA':'바이러스성'}
acc={}
for sp in ('train','val','test'):
    p=ROOT/f'runs/lung_masks/{sp}_lung_masks.npz'; p=p if p.is_file() else Path(f'{sys.argv[1]}/xrv_out/{sp}_lung_masks.npz')
    for k,m in LP.load_masks(p,512).items():
        a=m.mean(); h,_=LP.hull_mask(m,0) if a>=LP.MIN_AREA else (m,False); sol=(m.sum()/max(h.sum(),1)) if a>=LP.MIN_AREA else 0.0
        acc.setdefault(CL[rows[k]['label_name']],[]).append((a,sol,sp))
print(f"{'클래스':8s} {'장수':>5s} {'면적<5%(실패)':>12s} {'면적<15%':>9s} {'솔리디티<0.5':>11s} {'면적 중앙값':>10s}")
for c,v in acc.items():
    v=np.array([(a,s) for a,s,_ in v]); n=len(v)
    print(f"{c:8s} {n:5d} {int((v[:,0]<0.05).sum()):6d} ({(v[:,0]<0.05).mean():5.1%}) {(v[:,0]<0.15).mean():9.1%} {(v[:,1]<0.5).mean():11.1%} {np.median(v[:,0]):10.3f}")
