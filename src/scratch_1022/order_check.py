import sys, csv, numpy as np, cv2
from pathlib import Path
from PIL import Image
ROOT=Path('~/Documents/p1'); sys.path.insert(0,str(ROOT/'training'))
import lung_prep as LP, equipment_direct as ED
from shift_filters import to_gray, apply_shift
M=LP.load_masks(ROOT/'runs/lung_masks/val_lung_masks.npz')
rows=[r for r in csv.DictReader(open(ROOT/'manifests/split_manifest.csv',encoding='utf-8')) if r['split']=='val']
rng=np.random.default_rng(42); pick=rng.permutation(len(rows))[:60]
def make(g,name,lv,seed):
    return ED.apply_equipment(g,name,lv,seed=seed) if name in ED.EQUIP else apply_shift(g,name,lv,seed=seed)
names=['dose','motion','scatter','kvp','sensor','sharpen','equalize','jpeg']
print(f"{'필터(강)':10s} {'B: 밖 평균밝기':>14s} {'A−B 폐 안쪽 차이':>16s} {'A−B 윤곽 근처 차이':>18s}   (60장 평균, 0~255)")
for nm in names:
    out_b=[];d_in=[];d_edge=[]
    for k,i in enumerate(pick):
        r=rows[i]; g=to_gray(Image.open(Path.home()/'data/chest_xray'/r['relative_path']),512); hull,ok=LP.hull_mask(M[r['relative_path']])
        if not ok: continue
        A=np.where(hull,make(g,nm,3,42+k),0).astype(np.float32)        # A: 필터 → 마스크
        Braw=make(np.where(hull,g,0).astype(np.uint8),nm,3,42+k).astype(np.float32)   # B: 마스크 → 필터
        Bm=np.where(hull,Braw,0)
        er=cv2.erode(hull.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(33,33))).astype(bool)   # 윤곽에서 16px 안쪽
        band=hull&~er
        out_b.append(Braw[~hull].mean()); d_in.append(np.abs(A-Bm)[er].mean()); d_edge.append(np.abs(A-Bm)[band].mean())
    print(f"{nm:10s} {np.mean(out_b):14.2f} {np.mean(d_in):16.2f} {np.mean(d_edge):18.2f}")
