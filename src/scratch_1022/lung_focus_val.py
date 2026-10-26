import sys, csv, numpy as np, torch
from pathlib import Path
from PIL import Image
ROOT=Path('~/Documents/p1'); SC=Path(sys.argv[1])
sys.path.insert(0,str(ROOT/'training')); sys.path.insert(0,str(ROOT/'src'))
import train as T, gradcam as G
from shift_eval import load_model
S=512; G.SIZE=S; torch.set_num_threads(6)
rows=[r for r in csv.DictReader(open(ROOT/'manifests/split_manifest.csv',encoding='utf-8')) if r['split']=='val']
z=np.load(SC/'xrv_out/val_lung_masks.npz'); assert list(z['paths'])==[r['relative_path'] for r in rows]
masks=np.unpackbits(z['masks'],axis=1)[:,:S*S].reshape(len(rows),S,S).astype(bool)
y=np.array([int(r['label_id']) for r in rows]); groups=np.array([r['patient_group'] for r in rows])
tf=T.build_transform('P0',S); area=masks.mean((1,2))
MODELS=[('B512 (AWS, seed 42)','runs/results/b512-20261022-111844/model'),('b-s42 (기준)','runs/results/b-s42-20261022-jlab/model'),('v-s42 (예방접종)','runs/results/v-s42-20261022-jlab/model')]
rng=np.random.default_rng(42); ug=np.unique(groups); gi={g:np.where(groups==g)[0] for g in ug}
def boot(v,n=1000):
    ms=[]
    for _ in range(n):
        idx=np.concatenate([gi[g] for g in rng.choice(ug,len(ug))]); ms.append(v[idx].mean())
    return np.percentile(ms,[2.5,97.5])
L=['# Grad-CAM이 폐 안에 얼마나 있나 (val 1,030장, PSPNet 폐 마스크 기준)','',
   '폐 마스크는 TorchXRayVision PSPNet(좌·우 폐의 합, 시그모이드 0.5)이 입력과 같은 찌그러뜨린 512 사진에서 만든 것이다 (모델 학습 없음, 재학습 없음, test 안 씀, 마스크 품질은 사진 12장을 눈으로만 확인함). '
   '"폐 안 비율" = 지도 총합 중 폐 마스크 안에 있는 몫. 지도가 사진 전체에 고르게 퍼져 있으면 폐 면적 비율과 같다. 구간은 환자 묶음 단위 bootstrap 1,000번.','',
   f'폐 면적 비율 평균 {area.mean():.3f} (사진 {len(rows)}장).','']
csvrows=[]
for name,d in MODELS:
    cf=G.GradCAM(load_model(ROOT/d)); inside=np.zeros(len(rows)); peak=np.zeros(len(rows),dtype=bool); pred=np.zeros(len(rows),dtype=int)
    for i in range(0,len(rows),16):
        X=torch.stack([tf(Image.open(Path.home()/'data/chest_xray'/r['relative_path']).convert('RGB')) for r in rows[i:i+16]])
        lg,cams,_=cf(X,[None]); c=cams[0]
        for j in range(len(c)):
            k=i+j; s=c[j].sum(); inside[k]=(c[j]*masks[k]).sum()/s if s>0 else np.nan
            peak[k]=masks[k].ravel()[c[j].argmax()]; pred[k]=int(lg[j].argmax())
    ok=pred==y; diff=inside-area; v=~np.isnan(inside)
    lo,hi=boot(np.where(v,diff,0.0))
    L+=[f'## {name}','',f'- 지도 중 폐 안 비율 평균 **{np.nanmean(inside):.3f}** (폐 면적 비율 {area[v].mean():.3f}, 차이 {np.nanmean(diff):+.3f}, 95% 구간 [{lo:+.3f}, {hi:+.3f}])',
        f'- 지도 최고점이 폐 안에 있는 사진 **{peak.mean():.1%}** (고르게 퍼졌다면 약 {area.mean():.0%})',
        f'- 폐 안 비율이 폐 면적 비율보다 큰 사진(폐에 몰린 사진) {np.nanmean(diff[v]>0):.1%}','',
        '| 묶음 | 장수 | 지도 중 폐 안 비율 | 폐 면적 비율 | 최고점이 폐 안 |','|---|---|---|---|---|']
    for nm,m in [('맞힌 사진',ok),('틀린 사진',~ok),('정답 정상',y==0),('정답 세균성',y==1),('정답 바이러스성',y==2)]:
        L.append(f'| {nm} | {m.sum()} | {np.nanmean(inside[m]):.3f} | {area[m].mean():.3f} | {peak[m].mean():.1%} |')
    L.append('')
    for k in range(len(rows)): csvrows.append([name,rows[k]['relative_path'],int(y[k]),int(pred[k]),f'{inside[k]:.4f}',f'{area[k]:.4f}',int(peak[k])])
    print(L[-12] if False else f'{name}: 폐 안 {np.nanmean(inside):.3f} / 면적 {area.mean():.3f} / 최고점 {peak.mean():.1%}',flush=True)
L+=['## 한계','','- 폐 마스크는 소아 사진에서 12장만 눈으로 확인했고, 폐렴으로 뿌옇게 된 부위를 놓치거나 복부로 번지는 경우가 있다 (마스크 오류가 "폐 안 비율"을 낮출 수 있다).','- 지도는 16×16 칸을 늘린 것이라 거칠고, 폐 윤곽 가까이의 값은 칸 경계에 영향을 받는다.','- "폐 안에 있다"는 것이 병변을 본다는 뜻은 아니고, "폐 밖에 있다"는 것이 지름길이라는 증명도 아니다.','- 모델은 seed 하나씩이다.']
out=ROOT/'reports/equipment_v2/gradcam'
(out/'lung_focus_val.md').write_text('\n'.join(L)+'\n',encoding='utf-8')
import csv as _c
with open(out/'lung_focus_val.csv','w',newline='',encoding='utf-8') as f:
    w=_c.writer(f); w.writerow(['model','image','true','pred','cam_inside_lung','lung_area','peak_in_lung']); w.writerows(csvrows)
print('저장 완료',flush=True)
