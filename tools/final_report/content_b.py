"""결과보고서 본문 2부: 4~8장과 부록."""
import json
import re
from pathlib import Path

import numpy as np

import charts as C
import data as D
from content_a import f3, fig, pm, read_lung_csv, read_vac_csv, sg, tbl, verdict

ROOT = D.ROOT
EQ = D.EQ

LEVEL = {1: "약", 2: "중", 3: "강"}
PARAM_FIX = {"clahe_1.5": "CLAHE 1.5", "clahe_3.0": "CLAHE 3.0", "he": "HE"}


def md_tables(path):
    """마크다운 파일의 표를 [행[칸]] 목록의 목록으로 읽는다 (구분선 행 제외)."""
    tables, cur = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            cur.append(cells)
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    return tables


# ---------------------------------------------------------------------------------------------
# 4장 방법
# ---------------------------------------------------------------------------------------------
def sec4(ctx):
    ex = ctx["exp"]
    eps = " / ".join(str(ex[k]["epoch"]) for k in ("b512", "b512-s43", "b512-s44"))
    secs = " / ".join(f"{ex[k]['sec']:.0f}" for k in ("b512", "b512-s43", "b512-s44"))
    t6 = tbl(["항목", "내용", "항목", "내용"], [
        ["백본", "ResNet18, ImageNet 사전학습 가중치, 출력층 fc 512 → 3", "학습하는 층", "layer4 + fc (나머지는 고정)"],
        ["입력", "512×512. 회색 사진을 3채널로 복제하고 ImageNet 값으로 정규화. 가로세로는 정사각형으로 눌러 맞춤", "증강·전처리", "없음 (기준 모델)"],
        ["손실·최적화", "Cross-Entropy, Adam (fc 학습률 0.001, layer4 학습률 0.0001)", "배치·에폭", "배치 16, 최대 30에폭"],
        ["에폭 선택", "val Macro-F1이 가장 높은 에폭 (선택 규칙은 학습 전에 고정)", "seed", f"42 · 43 · 44 (선택된 에폭 {eps})"],
        ["학습 환경", f"SageMaker 학습 작업, ml.g4dn.2xlarge (T4), PyTorch 2.2 컨테이너. 학습 시간 {secs}초", "코드", "<code>training/train.py</code> (기준)<br><code>train_aug.py</code> (장비 효과 증강)<br><code>train_lung.py</code> (폐 영역)"],
    ], widths=[11, 41, 12, 36], cls="compact")

    filt = [
        ["<b>장비 설정 5종</b><br><span class='dim'>(학습에도 씀)</span>", "<b>관전압 ↑</b> (kVp)", "관전압이 높으면 영상의 대비가 줄고 뼈가 덜 하얘진다", "평균 기준으로 대비를 줄이고 밝은 쪽을 눌러 줌", "(대비 배율, 밝은 쪽 기울기) (0.85, 0.8) · (0.70, 0.6) · (0.60, 0.5)"],
        ["", "<b>관전류↓</b> (선량)", "선량이 줄면 광자가 줄어 양자 노이즈가 늘어난다 (노이즈 ∝ 1/√선량)", "노출량에 비례하는 노이즈를 더함", "상대 선량 1/4 · 1/16 · 1/64 (노이즈 추정 2.1 · 4.0 · 7.9배)"],
        ["", "<b>노출시간 ↑</b>", "오래 찍으면 아이가 움직여 세로로 번진다", "세로 방향 상자 필터 (사진 위 25% → 아래 100%로 갈수록 크게)", "번짐 길이 2 · 4 · 8 mm"],
        ["", "<b>산란</b> (격자 없음)", "산란선이 안개처럼 깔려 작은 구조의 대비가 줄어든다", "(원본 + k × 크게 흐린 사진) ÷ (1 + k)", "k = 0.25 · 0.5 · 1.0 (폭 25 mm)"],
        ["", "<b>센서 해상도</b>", "화소가 커지면 미세한 무늬가 뭉개진다", "화소 안에서 평균을 낸 뒤 다시 키움", "화소 0.5 · 1.0 · 2.0 mm"],
        ["<b>제조사 후처리·전송 3종</b><br><span class='dim'>(학습에는 쓰지 않음)</span>", "<b>경계 강조</b>", "제조사의 선명화 후처리 (대표 기법으로 대신함)", "언샤프 마스킹", "α = 0.5 · 1.0 · 2.0"],
        ["", "<b>대비 보정</b>", "제조사의 대비 보정 (대표 기법으로 대신함)", "CLAHE 또는 히스토그램 평활화", "CLAHE 1.5 · CLAHE 3.0 · HE"],
        ["", "<b>JPEG 압축</b>", "전송·저장 과정의 손실 압축", "JPEG으로 다시 저장", "품질 75 · 40 · 20"],
    ]
    t7 = tbl(["구분", "필터", "장치에서 일어나는 일", "구현", "약 · 중 · 강"], filt, widths=[14, 14, 27, 22, 23], cls="compact", sub=())

    tiles = "".join(
        f'<div class="tile"><img src="assets/{fn}.png" alt="{label}"><b>{label}</b></div>'
        for fn, label in [("original", "원본"), ("kvp", "관전압 ↑"), ("dose", "관전류↓"), ("motion", "노출시간 ↑"), ("scatter", "산란"),
                          ("sensor", "센서 해상도"), ("sharpen", "경계 강조"), ("equalize", "대비 보정"), ("jpeg", "JPEG 압축")])
    tiles += '<div class="tile" style="text-align:left;padding-top:4pt">가장 센 단계 값<br>관전압 (0.60, 0.5) · 선량 1/64 · 번짐 8 mm · 산란 k=1.0 · 센서 2.0 mm · 경계 강조 α=2 · HE · JPEG 20</div>'
    return f"""
<h1 id="sec4"><span class="no">4</span>방법</h1>
<h2>4.1 기준 모델 B512</h2>
<p>'아무것도 더하지 않고 그냥 돌린 기본 모델'을 기준으로 삼았다 (증강·전처리 없이 마지막 잔차 블록 layer4와 출력층만 학습한 ResNet18).</p>
{t6}
<h2>4.2 장비 설정 필터 8종</h2>
<p>흉부 X선 장치에서 설정을 바꾸면 영상에 생기는 효과를 고전 영상처리로 만들었다. <b>장비의 변수를 직접 조절한 것이 아니라 그 변수가 만드는 결과를 흉내 낸 것</b>이다. 필터마다 약·중·강 3단계, 시험 조건은 8종 × 3단계 = 24개다 (부록 A에는 흐림·밝게·어둡게 9개를 더한 33조건).
필터는 모델이 실제로 보는 512×512 회색 사진에 직접 걸었고, 길이는 '모델 입력에서 1 mm = 2.29픽셀'로 통일했다.</p>
{t7}
<figure><div class="tiles">{tiles}</div>
<figcaption><b>그림 1.</b> 같은 사진(train 정상)에 필터를 가장 센 단계로 건 모습. 보기 쉽도록 가운데 정사각형으로 잘랐고(모델 입력은 눌러서 맞춤) 필터 코드와 강도는 시험과 같다. 이 사진 한 장에서 눈에 띄는 정도는 필터마다 다르다.</figcaption></figure>
<ul class="small">
<li>강도는 <b>시험 전에 문서(<code>reports/equipment_v2/strength_fixed_512.md</code>)로 고정</b>했다. 224판 시험의 점수를 본 뒤 단위만 정리했고(입력을 224에서 512로 옮기며 mm 단위 통일), 이후 점수를 보고 바꾸지 않았다.</li>
<li><b>필터끼리 강도가 보정되지 않았다.</b> 예를 들어 관전압 '강'은 가운데 척추·심장 부위가 거의 평평해질 만큼 세다. 그래서 필터 사이의 점수 하락을 직접 비교하지 말고 '어느 필터에서 크게 떨어지는가'의 순서 정도로만 읽는다.</li>
<li>선량 필터의 기준 노이즈는 원본 사진의 노이즈 추정에 맞춘 값이고 모델 점수는 쓰지 않았다. 8비트 JPEG라 실제 톤 곡선을 모르고, 관전압은 뼈와 연부조직의 에너지 의존 대비를 만들지 못해 근사했다.</li>
<li>제외한 필터: 흐림(초점·거리 번짐), 밝게·어둡게(노출·표시 설정)는 장비 설정 분류에 들어가지 않아 본편에서 뺐고 결과는 부록 A에 남겼다.</li>
</ul>
<h2>4.3 폐 영역 모델 (시험 ②)</h2>
<p>공개 폐 분할 모델 PSPNet(TorchXRayVision)이 좌·우 폐를 칠하고, 그 <b>볼록 껍질 + 16픽셀</b> 밖을 검게(0) 지운 사진으로 같은 방식(B512 설정 그대로, seed 3개)으로 학습한 것이 <b>폐 영역 모델(l)</b>이다. 자르거나 확대하지 않고 사진 크기를 그대로 둔다.
필터는 <b>먼저 걸고 같은 고정 마스크를 씌운다</b> (마스크 뒤에 거는 순서는 산란·관전압·대비 보정에서 폐 안쪽까지 다르게 만들어 쓰지 않았다). 폐 면적이 사진의 5% 미만이면(분할 실패) 사진 전체를 쓴다 — 5장(train 4, val 1), 모두 세균성이었다 (폐가 하얗게 찬 사진에서 PSPNet이 공기가 찬 조각만 잡음).</p>
<h2>4.4 장비 효과 증강 (시험 ③)</h2>
<p>학습 사진의 <b>50%</b>에 장비 설정 필터 5종 중 하나를 골라 <b>무작위 세기 u ~ U(0, 1)</b>로 건다 (u = 1이 시험의 '강'과 같고, u = 0이 원본과 같다). 경계 강조·대비 보정·JPEG은 학습에 쓰지 않고 <b>'처음 보는 유형'</b>으로 남겼다.
기준 모델에 적용한 것이 <b>증강 모델(v)</b>, 폐 영역 모델에 적용한 것이 <b>폐 영역 + 증강 모델(lv)</b>이다. 에폭 선택 규칙은 기준과 같고(필터를 안 건 val 점수가 가장 높은 에폭), 증강은 학습에만 쓴다. 증강 코드: <code>training/train_aug.py --augment equip</code>.</p>
<h2>4.5 평가 방법</h2>
<ul>
<li>지표: Macro-F1(주 지표, 세 클래스 F1의 평균), 클래스별 recall, Macro AUROC. 필터 효과는 <b>같은 모델의 원본 점수 대비 변화</b>로 보고, 95% 구간은 <b>환자 묶음 단위 짝지은 bootstrap 1,000회</b>(같은 환자의 사진을 함께 뽑음)로 구한다.</li>
<li>val 1,030장은 개발·선택용이다. test 618장에는 모델마다 <b>평가 계획 문서(<code>reports/equipment_v2/test_plan_*.md</code>)와 sha256 해시를 먼저 고정한 뒤 한 번만</b> 필터 33조건을 걸었다 (평가 코드 <code>src/equipment_eval_direct.py</code>, 필터를 안 건 점수가 학습 기록과 같은지 자동 검증).</li>
<li>묶음 비교: 같은 환경의 모델끼리, 평균 차이가 <b>기준 묶음 표준편차(seed 3개, ddof=1)의 2배 이상</b>이면 '차이 있음', 아니면 '말할 수 없음'. 표준편차가 seed 3개에서 나온 거친 값이라 판정은 엄격하다. 이 규칙은 결과를 보기 전에 정했고 바꾸지 않았다.</li>
<li>학습에 쓴 필터 5종을 건 점수는 <b>'배운 유형'</b>이라 방어력의 증거로 쓰지 않는다. 처음 보는 3종과 필터를 안 건 원본 점수가 방어력을 보는 부분이다.</li>
</ul>
"""


# ---------------------------------------------------------------------------------------------
# 5장 결과
# ---------------------------------------------------------------------------------------------
def sec5(ctx, raw):
    d = ctx["data"]
    A, L, LV, JB, JV = (d[k] for k in ("aws_b", "lung", "lung_aug", "jl_b", "jl_v"))
    lung = read_lung_csv()
    vac = read_vac_csv()
    ex = ctx["exp"]

    # 5.1 기준 모델 표
    rows = []
    for s, key in zip(D.SEEDS, ("b512", "b512-s43", "b512-s44")):
        v, t = A["val"][s], A["test"][s]
        rows.append([f"B512 s{s}", str(ex[key]["epoch"]), f3(v["orig_f1"]), f3(v["auroc"]), f3(v["rec_n"]),
                     f3(t["orig_f1"]), f3(t["auroc"]), f3(t["rec_n"]), f"{t['pneu_missed']}장 ({t['pneu_sens']:.3f})"])
    vm = lambda k: D.agg(A["val"], k)
    tm = lambda k: D.agg(A["test"], k)
    rows.append(["<b>평균±SD</b>", "", pm(*vm("orig_f1")[:2]), pm(*vm("auroc")[:2]), pm(*vm("rec_n")[:2]),
                 pm(*tm("orig_f1")[:2]), pm(*tm("auroc")[:2]), pm(*tm("rec_n")[:2]), ""])
    t9 = tbl(["모델", "선택 에폭", "val F1", "val AUROC", "val 정상 recall", "test F1", "test AUROC", "test 정상 recall", "test 폐렴을 정상으로 놓친 장 (민감도)"],
             rows, align="lnnnnnnnn", sub=(3,), cls="compact", widths=[10, 6, 11, 11, 12, 11, 11, 12, 16])

    cm = [["정상 (231)", "103", "38", "90"], ["세균성 (240)", "1", "233", "6"], ["바이러스성 (147)", "0", "51", "96"]]
    t10 = tbl(["정답 \\ 예측 (test 618장, seed 42)", "정상", "세균성", "바이러스성"], cm, align="lnnn", widths=[46, 18, 18, 18], cls="compact", hl=(0,))

    # 5.2: 그림 2, 3
    fig2 = C.fig_strong_drops(d)
    fig3 = C.fig_scatter(raw)
    pts = D.val_test_changes(raw)
    rho = D.spearman([p[2] for p in pts], [p[3] for p in pts])
    r_p = float(np.corrcoef([p[2] for p in pts], [p[3] for p in pts])[0, 1])
    both = sum(1 for p in pts if p[2] < 0 and p[3] < 0)
    n_sig_val = sum(1 for s in D.MAIN for l in (1, 2, 3) if not (raw["val"][(s, l)]["lo"] <= 0 <= raw["val"][(s, l)]["hi"]))
    n_sig_test = sum(1 for s in D.MAIN for l in (1, 2, 3) if not (raw["test"][(s, l)]["lo"] <= 0 <= raw["test"][(s, l)]["hi"]))
    kv = raw["val"][("kvp", 3)]
    mo = raw["val"][("motion", 3)]
    cl = raw["val"][("clean", 0)]
    sv = lambda sh: D.agg_strong(A["val"], sh, True)[0]
    st = lambda sh: D.agg_strong(A["test"], sh, True)[0]

    # 5.3 표 (기준 대 폐)
    def lrow(split, metric, label, pct=False):
        r = lung[(split, "lung-aws", metric)]
        b, bs, m, df, ratio = float(r["base_mean"]), float(r["base_sd"]), float(r["model_mean"]), float(r["diff"]), float(r["ratio"])
        return [split, label, pm(b, bs), f3(m), sg(df), f"{ratio:+.1f}배".replace("-", "−"), verdict(r["verdict"])]
    lr = []
    for split in ("val", "test"):
        lr += [lrow(split, "원본 F1", "원본 사진 Macro-F1"), lrow(split, "AUROC", "원본 사진 AUROC"), lrow(split, "정상 recall", "원본 사진 정상 recall"),
               lrow(split, "본편 24조건 평균", "필터 24조건 평균 F1"), lrow(split, "본편 24조건 하락", "필터 24조건 평균 하락 (원본 대비)")]
    t11 = tbl(["", "지표", "기준 모델 (AWS 3개)", "폐 영역 모델 (3개)", "차이", "기준 SD의 몇 배", "판정"], lr,
              align="llnnnnl", widths=[6, 28, 17, 13, 9, 11, 16], cls="compact", sub=())
    fig4 = C.fig_base_vs_lung(d)

    # 5.4 표 (증강)
    def vrow(split, metric, label):
        r = vac[(split, metric)]
        return [split, label, pm(float(r["base_mean"]), float(r["base_sd"])), f3(float(r["vac_mean"])), sg(float(r["diff"])), verdict(r["verdict"])]

    def vrow_l(split, metric, label):
        r = lung[(split, "lv-lung", metric)]
        return [split, label, pm(float(r["base_mean"]), float(r["base_sd"])), f3(float(r["model_mean"])), sg(float(r["diff"])), verdict(r["verdict"])]
    a_rows = []
    for split in ("val", "test"):
        a_rows += [vrow(split, "원본 F1", "필터를 안 건 원본 Macro-F1"), vrow(split, "정상 recall", "원본 정상 recall"),
                   vrow(split, "학습에 쓴 5종 (15조건)", "연습에서 본 필터 5종 (15조건 평균 F1)"),
                   vrow(split, "쓰지 않은 3종 (9조건)", "못 본 필터 3종 (9조건 평균 F1)")]
    t12a = tbl(["", "지표", "증강 전 (JupyterLab 기준 3개)", "증강 후 (v 3개)", "차이", "판정"], a_rows, align="llnnnl", widths=[6, 33, 20, 14, 9, 18], cls="compact")
    l_rows = []
    for split in ("val", "test"):
        l_rows += [vrow_l(split, "원본 F1", "필터를 안 건 원본 Macro-F1"), vrow_l(split, "정상 recall", "원본 정상 recall"),
                   vrow_l(split, "배운 5종 평균", "연습에서 본 필터 5종 (15조건 평균 F1)"),
                   vrow_l(split, "안 배운 3종 평균", "못 본 필터 3종 (9조건 평균 F1)")]
    t12b = tbl(["", "지표", "증강 전 (폐 영역 l 3개)", "증강 후 (lv 3개)", "차이", "판정"], l_rows, align="llnnnl", widths=[6, 33, 20, 14, 9, 18], cls="compact")
    fig5 = C.fig_aug(d)

    return f"""
<h1 id="sec5"><span class="no">5</span>결과</h1>
<h2>5.1 기준 모델의 점수</h2>
{t9}
<p>기준 모델의 val 점수는 {pm(*vm('orig_f1')[:2])}, 같은 모델에 test 618장을 한 번씩 넣으면 {pm(*tm('orig_f1')[:2])}다. 가장 크게 달라진 것은 <b>정상 사진</b>이다. val에서 정상을 {vm('rec_n')[0]*100:.1f}% 맞히던 모델이 test에서는 {tm('rec_n')[0]*100:.1f}%만 맞힌다 (정상 231장 중 {231 - round(raw['test'][('clean', 0)]['rec_n'] * 231)}장을 폐렴으로 판정, seed 42).
반대로 폐렴을 정상으로 놓친 사진은 387장 중 1~2장뿐이다. 이 모델은 <b>정상을 폐렴이라고 말하는 쪽으로</b> 틀린다. 같은 설정으로 학습을 다시 해도 test F1이 {min(tm('orig_f1')[2]):.3f}~{max(tm('orig_f1')[2]):.3f}으로 흔들려서, 'test {tm('orig_f1')[0]:.2f}'은 고정된 값이 아니다.</p>
{t10}
<h2>5.2 시험 ①: 필터를 걸면 점수가 얼마나 떨어지는가</h2>
<p>기준 모델 3개(seed 42·43·44)에 필터 8종을 약·중·강으로 걸어 원본 대비 점수 변화를 쟀다. 그림 2는 가장 센 단계만, 전체 조건은 부록 A에 있다.</p>
<ul>
<li><b>장비 설정 5종이 크게 흔들었다.</b> val에서 관전압({sg(sv('kvp'))}), 노출시간({sg(sv('motion'))}), 센서 해상도({sg(sv('sensor'))}), 산란({sg(sv('scatter'))}) 순이었고, test에서는 같은 필터들이 더 크게 떨어졌다({sg(st('kvp'))}, {sg(st('motion'))}, {sg(st('sensor'))}, {sg(st('scatter'))}). 관전류(선량)가 가장 약했다({sg(sv('dose'))} / {sg(st('dose'))}).</li>
<li>후처리·전송 3종 중 경계 강조는 val에서 {sg(sv('sharpen'))}였지만 <b>test에서는 오히려 올랐다</b>({sg(st('sharpen'))}). 약한 단계에서 test 정상 recall이 {raw['test'][('clean', 0)]['rec_n']:.3f}에서 {raw['test'][('sharpen', 1)]['rec_n']:.3f}으로 올랐고, 원인은 확인하지 못했다. JPEG은 두 곳 모두 떨어졌다({sg(sv('jpeg'))} / {sg(st('jpeg'))}).</li>
<li>한 모델(seed 42)로 보면 val의 본편 24조건 중 <b>{n_sig_val}개</b>, test는 {n_sig_test}개의 변화가 95% 구간에서 0이 아니었다 (전체 33조건표는 부록 A).</li>
<li><b>쏠림의 방향:</b> 관전압 '강'을 걸면 val 정상 사진의 recall이 {cl['rec_n']:.3f}에서 {kv['rec_n']:.3f}으로 떨어지고 바이러스성으로 예측하는 비율이 {cl['share_v']*100:.0f}%에서 {kv['share_v']*100:.0f}%로 늘었다. 노출시간 '강'에서는 바이러스성이 {cl['share_v']*100:.0f}% → {mo['share_v']*100:.0f}%로 줄고 세균성 예측이 늘었다. 이 방향은 모델마다 다를 수 있어(224 입력 모델은 노이즈에서 바이러스성 쪽으로 쏠렸다) 일반화하지 않는다.</li>
</ul>
{fig('<div>' + fig2 + '</div>', '<b>그림 2.</b> 필터 8종을 가장 센 단계로 걸었을 때 원본 점수 대비 Macro-F1 변화. 막대는 기준 모델 3개의 평균, 점은 모델 하나다. 위 5개는 학습에도 쓴 장비 설정, 아래 3개는 학습에 쓰지 않은 후처리·전송이다. 필터끼리 강도가 보정되지 않았으므로 막대 길이를 필터 사이에서 그대로 비교하지 않는다.')}
<div class="row2">
<div>{fig('<div>' + fig3 + '</div>', f'<b>그림 3.</b> 같은 조건에서 val의 점수 변화(가로)와 test의 변화(세로), 기준 모델 seed 42, 본편 24조건 (점 하나 = 조건 하나).')}</div>
<div style="padding-top:6pt">
<p><b>val에서 본 것이 test에서도 같은 방향이다.</b> 24조건의 변화량 순위상관이 <b>{rho:.2f}</b>(피어슨 {r_p:.2f}), 둘 다 떨어진 조건이 {both}개다. 예외는 test에서 점수가 오른 경계 강조 약·중과 센서 약 조건이다.</p>
<p class="small">test는 정상 사진 집단이 달라(3.4절) 절대 점수가 낮고 seed마다 크게 흔들린다. 그래서 test는 val에서 본 순서가 반복되는지만 확인하는 용도로 썼다. 세 모델(seed)에서도 관전압·노출시간·센서가 크게 떨어지는 것은 같았고 크기는 seed마다 달랐다 (그림 2의 점).</p>
</div></div>
<h2>5.3 시험 ②: 폐만 남겨도 같은가</h2>
<p>폐 영역 모델은 원본 점수가 기준 모델과 비슷하거나(val) 더 높은(test) 반면, 필터에 약한 정도는 비슷했다.</p>
{t11}
<p class="small">판정의 화살표: ↑ 폐 영역 모델이 더 높음, ↓ 더 낮음 (하락 행은 값이 클수록 덜 떨어졌다는 뜻). 두 묶음의 평균 차이가 기준 묶음 표준편차의 2배 이상일 때만 '차이 있음'.</p>
{fig('<div>' + fig4 + '</div>', '<b>그림 4.</b> 기준 모델 3개와 폐 영역 모델 3개의 Macro-F1. 점은 모델 하나, 막대는 평균. 두 모델 모두 필터를 걸면(오른쪽 열) 원본(왼쪽 열)보다 낮아진다.')}
<ul>
<li><b>원본 점수:</b> val에서는 폐 영역 모델이 낮거나 같고(F1은 말할 수 없음, AUROC·정상 recall은 낮음), test에서는 높다(F1 0.775 대 0.705, 기준 SD의 2.1배 → '차이 있음'의 문턱 근처). 환경이 다른 JupyterLab 기준과 비교하면 1.5배로 규칙에 못 미쳐, <b>'폐 영역 모델의 test 점수가 더 높다'는 결과는 견고하지 않다.</b> 같은 폐 영역 모델이 val에서는 약간 낮은 점은 폐 밖 단서가 val(같은 병원·같은 분포)에서는 도움이 되었을 가능성과 맞지만 증명은 아니다.</li>
<li><b>필터 약점:</b> 24조건 평균 하락이 val {sg(D.agg(A['val'],'drop24')[0])}(기준) 대 {sg(D.agg(L['val'],'drop24')[0])}(폐 영역), test {sg(D.agg(A['test'],'drop24')[0])} 대 {sg(D.agg(L['test'],'drop24')[0])}으로 <b>폐 밖을 지워도 약점이 그대로 남았다.</b> 필터로 점수가 떨어지는 것이 폐 밖 단서를 쓰는 이 모델만의 현상은 아니라는 뜻이다.</li>
<li>이 시험의 한계: 폐 마스크는 필터 없는 사진에서 만든 것이라 분할 모델 자체의 강건성은 들어 있지 않다. 분할은 병이 심한 쪽(폐가 하얗게 찬 세균성 사진)에서 틀리고, 폐 모양이 새로운 단서가 될 수 있다.</li>
</ul>
<h2>5.4 시험 ③: 장비 효과를 섞어서 학습하면</h2>
<p>기준 모델과 폐 영역 모델 각각에 장비 효과 증강을 적용해 각 3개씩 학습했다. 증강 전 모델과 같은 환경끼리 비교한다 (기준: JupyterLab, 폐 영역: AWS 학습 작업).</p>
{t12a}
{t12b}
<p class="small">판정의 화살표: ↑ 증강 후가 더 높음. 연습에서 본 필터 5종은 학습에 섞은 것이라 방어력의 증거가 아니다.</p>
{fig('<div>' + fig5 + '</div>', '<b>그림 5.</b> 증강 전(파랑 원)과 후(주황 마름모)의 가장 센 단계 Macro-F1, 모델 3개 평균. 가로축 범위는 val과 test가 다르다. 구분선 위 5개가 학습에 섞은 필터다.')}
<ul>
<li><b>연습한 필터에서는 올랐다.</b> 기준 모델은 val 0.774 → 0.815, test 0.580 → 0.700으로 둘 다 '차이 있음'이고, 특히 노출시간과 관전압에서 컸다 (test 0.401 → 0.636, 0.444 → 0.697). 폐 영역 모델도 같은 방향이다(test 0.660 → 0.721). 다만 폐 영역 모델은 seed 편차가 커서 '말할 수 없음'이다.</li>
<li><b>그러나 이것은 같은 종류의 필터로 본 시험이다.</b> 학습에 섞은 필터를 시험에도 썼으므로 '실제 장비에 강해졌다'는 증거가 아니다. 방어력을 볼 수 있는 부분은 학습에 쓰지 않은 3종과 원본 점수다.</li>
<li><b>못 본 필터 3종과 원본 점수는 나아졌다고 말할 수 없다.</b> 기준 모델은 못 본 3종이 val 0.811 → 0.802, test 0.694 → 0.706, 원본 점수 val 0.836 → 0.831, test 0.712 → 0.741로 모두 '말할 수 없음'이다. 폐 영역 모델은 못 본 3종이 test에서 0.779 → 0.734로 낮아졌으나 이것도 '말할 수 없음'이다. 증강이 원본 점수를 해치지는 않았다.</li>
<li>따라서 이 실험으로 말할 수 있는 것은 '<b>배운 유형의 장비 효과에는 강해진다</b>'까지이고, '장비가 바뀌어도 괜찮다'는 아니다. 필터는 실제 장비가 아니라는 한계(8장)가 그대로 남는다.</li>
</ul>
"""


# ---------------------------------------------------------------------------------------------
# 6장 점수를 믿을 수 있는가
# ---------------------------------------------------------------------------------------------
def sec6(ctx):
    d = ctx["data"]
    A = d["aws_b"]
    gc = ctx["gradcam"]
    lf = ctx["lungfocus"]
    lb = ctx["lungbox"]
    asp = ctx["aspect"]
    jr = ctx["jpegresave"]
    early = ctx["early"]
    ns = ctx["normal_shift"]
    img = "assets/gradcam_compare_slide.png"
    cause = [
        ["정상 사진의 <b>저장 방식</b> (품질 96 + 편집 기록 대 품질 75)", "val 정상 사진을 품질 75로 다시 저장해 넣음 (B 224, D 224, B448)",
         f"예측이 바뀐 정상 사진 {jr['min']}~{jr['max']}장 (1,030장 중)", "<b>단독 원인은 아님</b>"],
        ["<b>가로세로 비율</b>", "val 사진의 내용을 가로로 늘려 test 정상 사진의 비율처럼 만들어 B512에 넣음",
         f"정상 recall {asp['base']} → {asp['f905']} (비율 0.905), {asp['f122']} (1.22)", "<b>재현되지 않음</b> (이 방식은 폐의 세로 점유를 바꾸지 못함)"],
        ["<b>폐가 사진 세로를 채우는 정도</b>", "test 정상 231장을 폐 상자 세로 점유 3분위로 나눠 B512 seed 42의 정상 recall을 봄",
         f"낮음 {lb['t1']:.3f} / 중간 {lb['t2']:.3f} / 높음 {lb['t3']:.3f}, 상관 {lb['rho']:+.2f}".replace("-", "−"), "<b>연관 있음</b> (인과 아님). 폐 상자로 자르는 재학습은 하지 않음"],
        ["<b>파일 속성의 지름길</b>", "파일 속성(크기·JPEG 품질·메타데이터 등)만으로 정상/폐렴을 맞히는 분류기", f"train 안에서 AUROC ≈ 1.00, test {ns['attr_auroc_test']:.2f}", "데이터에 지름길이 있다는 사실. 모델이 쓰는지는 <b>미확인</b>"],
        ["test 정상 사진 <b>집단 자체</b>의 차이", "나이·촬영 자세·촬영 시기 등", "데이터로 확인할 수 없음", "<b>모름</b>"],
    ]
    t13 = tbl(["원인 후보", "시험", "결과", "판단"], cause, widths=[22, 31, 25, 22], cls="compact")

    gcrow = [
        ["B512 (기준, AWS s42)", f"{lf['b512']:.3f}", f"{lf['area']:.3f}", f"{lf['peak_b512']:.1f}%", f"{lf['normal_peak_b512']:.1f}%"],
        ["b-s42 (기준, JupyterLab)", f"{lf['b_s42']:.3f}", f"{lf['area']:.3f}", f"{lf['peak_b']:.1f}%", f"{lf['normal_peak_b']:.1f}%"],
        ["v-s42 (장비 효과 증강)", f"{lf['v_s42']:.3f}", f"{lf['area']:.3f}", f"{lf['peak_v']:.1f}%", f"{lf['normal_peak_v']:.1f}%"],
    ]
    t14 = tbl(["모델 (val 1,030장)", "열지도 중 폐 안 비율", "폐 면적 비율", "최고점이 폐 안인 사진", "정답 정상 사진 중 최고점이 폐 안"], gcrow, align="lnnnn", cls="compact", widths=[32, 16, 14, 18, 20])
    e_rows = [[r["name"], f3(r["val"]), f3(r["test"]), sg(r["test"] - r["val"])] for r in sorted(early, key=lambda r: -r["val"])]
    return f"""
<h1 id="sec6"><span class="no">6</span>점수를 믿을 수 있는가: 진단</h1>
<h2>6.1 val 점수는 test를 예측하지 못했다</h2>
<p>정상 사진의 recall이 val {D.agg(A['val'],'rec_n')[0]:.3f}에서 test {D.agg(A['test'],'rec_n')[0]:.3f}으로 떨어지는 현상은 기준 모델만의 것이 아니다. 초기 224 모델 8개도 모두 val에서 정상 recall 0.955~0.989였다가 test에서 0.433~0.632로 떨어졌고(부록 B), 어떤 변화(클래스 가중치·노이즈 증강·전처리·해상도)를 줘도 이 낙폭은 사라지지 않았다.
모델 8개의 val 순위와 test 순위의 순위상관은 −0.50이라 <b>val 점수가 test 점수를 예측하지 못했다.</b> val은 train에서 떼어 낸 같은 분포라 이 차이를 잡을 수 없다 (3.4절).
틀린 정상 사진은 확신도 높게 틀린다 (224 모델 D: 정상을 폐렴으로 본 129장 중 113장이 최대 확률 0.70 이상이라 '판단 보류' 규칙에 걸리지 않음).</p>
<h2>6.2 원인 후보와 확인한 것</h2>
{t13}
<p><b>결론: test 점수가 낮은 원인은 확정하지 못했다.</b> 시험한 요인 하나만으로는 낙폭이 재현되지 않았고, 구도(폐가 사진 세로를 채우는 정도)가 가장 유력한 연관이다. 강사 의견(정상과 환자 폐의 비율 때문이고 폐 영역 상자로 크롭하면 잘 나온다)과 방향이 맞는 정황이지만 크롭 재학습은 하지 않았다.</p>
<h2>6.3 모델은 어디를 보는가 (Grad-CAM)</h2>
<figure><img src="{img}" alt="Grad-CAM 비교" style="max-width:74%"><figcaption><b>그림 6.</b> 예시 3장(val, 두 모델이 모두 맞힌 사진)의 Grad-CAM. 흰 선은 PSPNet이 찾은 폐 윤곽이다. 위: 기준 모델, 아래: 폐 영역 모델. 한 장씩의 예시이므로 일반화하지 않고 아래 평균 수치로 판단한다.</figcaption></figure>
{t14}
<p class="small">val 1,030장, 폐 마스크는 PSPNet(12장만 눈으로 확인). 지도가 고르게 퍼졌다면 폐 안 비율은 폐 면적 비율과 같고 최고점이 폐 안일 확률은 약 36%다.</p>
<ul>
<li><b>구현은 맞다.</b> 훅 없이 layer4 출력과 fc 가중치로 직접 만든 지도와 상관 {gc['corr']}, 최대 절대 차이 {gc['maxdiff']} (val 300장).</li>
<li><b>열지도는 폐에 몰리지 않는다.</b> 기준 모델 열지도 중 폐 안 비율은 {lf['b512']:.2f}로 폐 면적 비율({lf['area']:.2f})과 같거나 낮고, 최고점은 폐 밖(가장자리, 어깨·목, 횡격막·복부, 글자 R)에 더 많다. 정상 정답 사진은 최고점이 폐 안인 비율이 {lf['normal_peak_b512']:.1f}%뿐이다. 맞힌 사진과 틀린 사진의 폐 안 비율은 비슷했다({lf['right']:.3f} 대 {lf['wrong']:.3f}).</li>
<li><b>가리기 시험:</b> 지도 점수가 높은 칸 40개(전체의 16%)를 가리면 예측 확률이 {sg(gc['hi'])}, 무작위 칸을 가려도 {sg(gc['rand'])}, 낮은 칸을 가리면 {sg(gc['lo'])}만 변했다 (가리는 값을 두 가지로 바꿔도 같은 경향). <b>지도가 '영향이 없는 곳'은 가려냈지만, 높은 칸이 무작위 칸보다 더 중요하다는 근거는 없다.</b> 모델이 사진 전체의 질감·통계 같은 넓은 단서에 기대고 있을 가능성과 맞지만, 이 시험으로 알 수는 없다.</li>
<li><b>그래서 '모델이 폐를 본다'고 말할 수 없다.</b> Grad-CAM은 '어디'만 알려 주고 그곳에서 '무엇'에 반응했는지(글자 표식, 어깨, 촬영 방식, 폐 소견 등)는 모른다. 시험 ①·②의 결론도 '이 기본 모델 계열'로 범위를 좁혀 읽어야 한다. 필터가 폐 소견이 아니라 가장자리·질감 단서를 건드렸을 가능성을 배제하지 못한다.</li>
</ul>
"""


# ---------------------------------------------------------------------------------------------
# 7장 AWS
# ---------------------------------------------------------------------------------------------
def sec7(ctx):
    v = ctx["verify"]
    aws = ctx["aws"]
    cfg = tbl(["구성 요소", "AWS 서비스", "설정", "상태"], [
        ["데이터·결과 저장", "Amazon S3", "원천 5,856장은 읽기 전용(<code>s3://aws-say3-dataset1/pneumonia-xray/chest_xray/</code>), 학생 경로 <code>s3://aws-say3-student-dataset/students/&lt;학생ID&gt;/P1/cxr/</code> 아래에 manifests·weights·code·experiments·lung512·deployment", "<b>유지</b> (모델 포함)"],
        ["모델 학습", "SageMaker 학습 작업", "PyTorch 2.2 GPU 컨테이너(py310). 초기 실험(A~D·P1~P4)은 ml.g4dn.xlarge, 512 이후는 ml.g4dn.2xlarge(vCPU 8, JPEG 디코딩이 병목), 최대 실행 5,400초", "작업 완료"],
        ["학습 (일부)", "SageMaker JupyterLab", "ml.g4dn.xlarge(T4), PyTorch 2.8. 기준 b 3개·증강 v 3개 (합 약 2.8시간)", "앱 삭제됨"],
        ["모델 추론", "SageMaker Endpoint", "ml.m5.large 1대(CPU), pytorch-inference 2.6 컨테이너(py312), 모델: 기준 B512 seed 42, 입력 512 (<code>inference.py</code>와 배포용 <code>metadata.json</code>은 따로 작성)", "<b>삭제됨</b>"],
        ["중계", "AWS Lambda", "Python 3.12, 256 MB, 제한 85초, 환경변수로 Endpoint 이름", "<b>삭제됨</b>"],
        ["화면", "EC2 t3.small + Flask", "15 GiB gp3. 포트 8000(사진 올리기), 8001(필터 체험: 필터 8종 × 강도를 골라 원본과 필터본의 확률 비교)", "<b>삭제됨</b>"],
        ["권한·보안", "IAM, 보안 그룹", "인라인 최소 권한 정책(Lambda: 본인 Endpoint 호출, 웹: 본인 함수 호출), 보안 그룹은 <b>내 공인 IP /32만</b> 허용(0.0.0.0/0 아님), 업로드 2 MiB 제한과 JPEG·PNG 형식 검증. 로그인이 없는 서비스라 보안 그룹이 유일한 접근 통제", "보안 그룹 삭제됨, IAM 역할 2개·인스턴스 프로필 1개 남음 (7.5절)"],
    ], widths=[13, 15, 56, 16], cls="compact")

    chk = tbl(["비교", "사진", "결과"], [
        ["배포 가중치 대 학습 결과 <code>best_model.pt</code>", "파일", f"sha256 동일 (<code>{v['sha'][:16]}…</code>) → 배포 모델 = B512 seed 42 (선택 에폭 18)"],
        ["Endpoint 대 내 컴퓨터의 같은 모델 <b>(기획 Q4)</b>", "val 20장", "<b>예측 20/20 일치, 확률 최대 차이 7.5e-6</b> (기준 1e-4)"],
        ["Endpoint 호출 (추가 확인)", "val 29장", "29/29 일치, 최대 차이 7.4e-6, 왕복 평균 0.30초"],
        ["배포용 추론 코드를 로컬에서 실행 (추가 확인)", "val 59장", "59/59 일치, 최대 차이 7.1e-6"],
        ["필터 체험 화면의 필터 경로 대 로컬 계산 (추가 확인)", "필터 4종 × 약·강 × 예시", "확률 최대 차이 2.7e-6"],
        ["화면 → Lambda → Endpoint 전체 경로 <b>(기획 Q5)</b>", "val 1장", "<b>1.1초</b>, 세균성 0.9998 (내 컴퓨터의 같은 모델도 0.9998)"],
    ], widths=[44, 18, 38], cls="compact")

    jr = [[g, n, i.replace("ml.g4dn.", ""), m] for g, n, i, m in aws["job_rows"]]
    half = (len(jr) + 1) // 2
    left, right = jr[:half], jr[half:] + [["", "", "", ""]] * (half - len(jr[half:]))
    pairs = [[f"{a[0]} ({a[1]}개)", a[2], a[3], f"{b[0]} ({b[1]}개)" if b[0] else "", b[2], b[3]] for a, b in zip(left, right)]
    jobs = tbl(["실험 묶음", "인스턴스", "청구(분)", "실험 묶음", "인스턴스", "청구(분)"], pairs, align="llnlln", widths=[22, 20, 8, 22, 20, 8], cls="compact")
    stat = tbl(["자원", "남은 개수", "자원", "남은 개수"], aws["stat_rows"], align="lnln", widths=[32, 18, 32, 18], cls="compact")
    repo_line = (f"GitHub 저장소(<code>{ctx['repo_url'].replace('https://', '')}</code>)" if ctx.get("repo_url") else "저장소")
    return f"""
<h1 id="sec7"><span class="no">7</span>AWS 구성, 검산, 종료 기록</h1>
<h2>7.1 구성</h2>
{fig('<div>' + C.fig_arch() + '</div>', '<b>그림 7.</b> AWS 구성도. 서비스 경로는 10/23에 만들어 발표 시연에 쓰고 발표 직후 삭제했다. 학습 경로의 S3 모델과 기록은 남아 있다.')}
{cfg}
<h2>7.2 예측 확률 검산</h2>
<p>기획안의 재현성 목표(Q4)를 확인했다. 배포한 모델이 학습 결과 파일과 같은 파일이고, Endpoint가 돌려준 확률이 같은 모델을 내 컴퓨터에서 돌린 확률과 같은지 보았다.</p>
{chk}
<p class="small">검산 스크립트는 <code>src/endpoint_verify.py</code>, <code>deploy/verify_local.py</code>다. 실행 로그는 파일로 저장하지 않고 작업 기록에 숫자만 남겼고, Endpoint를 삭제해서 지금은 다시 돌릴 수 없다 (<code>deploy/endpoint_ctl.py</code> 등으로 다시 만들면 재현할 수 있다).</p>
<h2>7.3 학습 작업과 청구 시간</h2>
<p>{aws['total_jobs']}개의 학습 작업을 조회했고 청구 시간 합계는 <b>{aws['total_hours']:.2f}시간</b>이다 (ml.g4dn.2xlarge {aws['h_2xl']:.2f}시간, ml.g4dn.xlarge {aws['h_xl']:.2f}시간). 이와 별도로 JupyterLab에서 모델 6개를 약 2.83시간 학습했다. 시간당 단가는 확인하지 못해 금액으로 바꾸지 않았다.
과적합을 줄이는 실험(r, 15개)은 설계 방향이 test 결과에 영향받을 수 있다는 지적으로 <b>결과를 보지 않고 취소</b>했고, 약 1시간치가 청구되었다.</p>
{jobs}
<h2 id="sec75">7.4 배포 때 알게 된 점</h2>
<ul class="small">
<li>강의 예시 <code>inference.py</code>는 입력 크기 224와 metadata.json의 <code>preprocess.resize / mean / std</code> 형식을 요구해서, 우리 모델(512)에 맞춰 두 곳을 고치고 배포용 metadata.json을 따로 만들었다 (학습 결과 파일은 건드리지 않음).</li>
<li>Endpoint는 확률 3개만 돌려준다. Grad-CAM과 설명문은 배포 경로에 없다 (기획안의 Endpoint 설계와 다름).</li>
</ul>
<h2>7.5 종료 기록</h2>
<p>발표 직후(10/23) 삭제를 실행하고, 읽기 전용 조회로 <b>10/23 18:06과 10/26 {aws['time_now']}에 두 번</b> 남은 자원을 확인했다. 아래는 최신 조회 결과다 (스크립트 <code>tools/aws_teardown_check.py</code>, 전문 <code>reports/aws_teardown_check_20261023.md</code>, <code>…20261026.md</code>). 가이드의 '콘솔 스크린샷' 대신 조회 기록을 남겼다.</p>
{stat}
<ul>
<li><b>남은 것 (비용은 거의 없음):</b> IAM 역할 <code>p1-cxr-&lt;학생ID&gt;-lambda</code>·<code>-web</code>과 인스턴스 프로필 <code>-web</code>(인라인 정책만 있음)은 이 계정에 삭제 권한이 없어 지우지 못했다 → 필요하면 강사에게 삭제를 요청한다. S3의 배포용 모델 묶음(<code>deployment/b512/model_with_code.tar.gz</code>, 42 MB)은 다시 배포할 때 쓰려고 남겼다.</li>
<li>Endpoint와 EC2는 켜 두는 동안 과금되어 <b>배포 후 바로 지우는 계획</b>이었고 그대로 실행했다. 이 때문에 발표 때 쓴 접속 주소는 더 이상 열리지 않는다. 대신 코드·결과·보고서를 {repo_line}에 올려 제출한다.</li>
</ul>
"""


# ---------------------------------------------------------------------------------------------
# 8장 한계와 다음 단계
# ---------------------------------------------------------------------------------------------
def sec8():
    return """
<h1 id="sec8"><span class="no">8</span>한계와 다음 단계</h1>
<h2>8.1 한계</h2>
<ol>
<li><b>필터는 실제 장비가 아니다.</b> 장비 변수를 조절한 것이 아니라 결과를 흉내 낸 것이고, 강도는 임의로 정해서 필터끼리 점수 하락을 직접 비교할 수 없다. 제조사 후처리는 실제 알고리즘이 아닌 대표 기법이다. 8비트 JPEG라 톤 곡선을 모르고 관전압의 에너지 의존 대비를 만들지 못했다. 다른 병원의 실제 사진으로는 확인하지 못했다.</li>
<li><b>데이터가 한 병원, 1~5세 소아 사진</b>이다. 성인·다른 병원에는 적용할 수 없고, 세균성/바이러스성 라벨의 확진 근거가 공개되지 않아 라벨 오류 가능성이 있다.</li>
<li><b>반복 수가 적다.</b> 모델마다 seed 3개이고 표준편차는 거친 추정이다. val 점수가 가장 높은 에폭을 고르는 규칙이 흔들림을 키운다 (선택 에폭이 1인 모델이 있었다). 같은 설정에서도 test F1이 0.67~0.74로 흔들렸다.</li>
<li><b>test는 여러 번 열렸다.</b> 224 모델 8개에 한 번씩, B512 계열은 모델마다 한 번씩 평가했고, 설계 때 B512의 test 결과를 알고 있었다. 사전 계획 문서와 해시, 결과를 보고 설계를 바꾸지 않은 점으로 완화했지만 '한 번도 안 본 test'는 아니다. test 정상 사진은 집단이 달라 절대 점수보다 변화량 중심으로 읽었다.</li>
<li><b>test 점수가 낮은 원인은 모른다.</b> 저장 화질·가로 늘이기는 단독 원인이 아니었고, 구도(폐가 사진 세로를 채우는 정도)가 연관되지만 크롭 재학습으로 확인하지 못했다.</li>
<li><b>폐 영역 모델은 두 단계 시스템</b>이다. 분할 오류가 그대로 전달되고(실패는 전부 세균성), 폐 마스크는 사진 12장만 눈으로 확인했으며, 폐 윤곽의 모양이 새 단서가 될 수 있다. 필터 시험의 마스크는 필터 없는 사진 기준이다.</li>
<li><b>Grad-CAM은 '어디'만 알려 준다.</b> 가리기 시험에서 높은 칸과 무작위 칸의 영향이 비슷했다. 모델이 폐를 본다고 말할 수 없다.</li>
<li><b>기획안의 일부를 하지 못했다.</b> 판단 보류 곡선, 보호자용 문장 틀과 이해도 평가, Grad-CAM·설명문이 들어간 배포, 시연 영상 파일, 콘솔 스크린샷.</li>
<li><b>발표 때 쓴 접속 주소로는 확인할 수 없다.</b> 배포는 발표 때만 운영하고 지웠다. 코드·결과·보고서는 GitHub 저장소로 제출하고, 같은 화면은 코드(<code>deploy/</code>)로 다시 만들 수 있다 (예상 비용: Endpoint와 EC2를 켜 둔 시간만큼).</li>
</ol>
<h2>8.2 다음 단계</h2>
<ol>
<li>다른 병원·다른 장비로 찍은 사진으로 확인한다 (이 보고서의 필터 시험이 실제 장비 차이를 얼마나 닮았는지 보는 것이 핵심).</li>
<li>폐 상자로 자르거나 패딩 리사이즈로 구도를 맞춰 다시 학습한다 (강사 제안). 원인이 구도인지 확인하는 가장 직접적인 실험이다.</li>
<li>증강의 평가는 학습에 쓰지 않은 종류의 변화로 해서 방어력을 따로 본다.</li>
<li>판단 보류와 쉬운 설명문을 최종 모델에 연결하고 사람이 읽고 맞는지 평가한다.</li>
</ol>
"""


# ---------------------------------------------------------------------------------------------
# 부록
# ---------------------------------------------------------------------------------------------
def appendix(ctx, raw):
    ex = ctx["exp"]
    # A. 33조건
    rows = []
    cv, ct = raw["val"][("clean", 0)], raw["test"][("clean", 0)]
    rows.append(["<b>원본 (필터 없음)</b>", "-", "-", f3(cv["f1"]), "-", "-", f3(ct["f1"]), "-", "-"])
    order = D.DEVICE + D.POST + D.APPENDIX
    for sh in order:
        for lv in (1, 2, 3):
            v, t = raw["val"][(sh, lv)], raw["test"][(sh, lv)]
            p = PARAM_FIX.get(v["param"], v["param"])
            ci = lambda r: f"[{r['lo']:+.3f}, {r['hi']:+.3f}]".replace("-", "−")
            star = lambda r: "" if r["lo"] <= 0 <= r["hi"] else " *"
            rows.append([f"{D.NAME_KO[sh]}" if lv == 1 else "", LEVEL[lv], p, f3(v["f1"]), sg(v["delta"]) + star(v), ci(v), f3(t["f1"]), sg(t["delta"]) + star(t), ci(t)])
    tA = tbl(["필터", "단계", "값", "val F1", "val 변화", "95% 구간", "test F1", "test 변화", "95% 구간"], rows, align="llcnnnnnn", cls="compact",
             widths=[15, 6, 11, 8, 11, 17, 8, 11, 13], sub=(0,), keep=False)

    # B. 초기 224 실험
    t = md_tables(ROOT / "reports/model_comparison.md")
    t1, t2 = t[0], t[1]
    sc = {r[0]: r for r in t1[1:]}
    rc = {r[0]: r for r in t2[1:]}
    names = [("A 몸통 고정", "A"), ("B", "B"), ("C 가중치", "C"), ("D", "D"), ("P1 HE", "P1"), ("P2 CLAHE", "P2"), ("P3 블러", "P3"), ("P4 샤프닝", "P4")]
    desc = {"A": "layer4도 얼림, fc만 학습", "B": "기준: layer4 미세조정, 증강·전처리 없음", "C": "B + 클래스 가중치 손실", "D": "B + 노이즈 증강 (학습 사진 50%, std 0~10)",
            "P1": "B + 히스토그램 평활화 전처리", "P2": "B + CLAHE 전처리", "P3": "B + 가우시안 블러(σ1.5) 전처리", "P4": "B + 언샤프 마스킹 전처리"}
    e_rows = []
    for key, short in names:
        r1, r2 = sc[key], rc[key]
        e_rows.append([f"<b>{short}</b>", desc[short], r1[2], r1[4], r2[1], r2[2]])
    tB = tbl(["실험", "바꾼 것 (입력 224, seed 42)", "val F1", "test F1 [95% 구간]", "정상 recall (val / test)", "바이러스성 recall (val / test)"], e_rows,
             align="llnnnn", cls="compact", widths=[7, 30, 8, 22, 16, 17])
    figH = C.fig_early(ctx["early"])

    # D. 모델 S3 주소
    m_rows = []
    for key, label, how in ctx["models"]:
        e = ex[key]
        m_rows.append([f"<b>{label}</b>", how, f"<code>{e['addr']}</code>", str(e["epoch"]), f3(e["valf1"])])
    tD = tbl(["모델", "학습", "모델 주소 (S3, 학생 경로 <code>s3://aws-say3-student-dataset/students/&lt;학생ID&gt;/P1/cxr/</code> 아래)", "선택 에폭", "val F1"], m_rows,
             align="llcnn", cls="compact", widths=[13, 11, 52, 9, 15])

    deliver = tbl(["가이드 필수 제출", "이 프로젝트의 산출물", "패키지 안 위치"], [
        ["분할", "manifest.csv · split_manifest.csv · split_summary.csv, 분할 규칙·점검 보고서", "<code>manifests/</code>"],
        ["학습 이력", "실험별 history.csv, config.json, experiment_summary.json (모델 26개)", "<code>results/&lt;실험&gt;/</code>"],
        ["평가 CSV", "val 예측 val_predictions.csv, 필터 시험 shift_results.csv(모델 15개 × val·test), 224 모델 test_predictions.csv·metrics.csv", "<code>results/</code>, <code>reports/equipment_v2/</code>, <code>reports/test_eval/aws/</code>"],
        ["S3 모델 주소", "부록 D의 표 + 작업 제출 기록(작업 이름·입력·출력·해시)", "<code>aws/submissions/</code>"],
        ["AWS 구성", "7.1절, 배포 스크립트와 웹 화면 코드", "<code>deploy/</code>"],
        ["시연", "발표 중 실시간 시연 (화면 캡처는 발표자료 12쪽). 시연 영상 파일 없음", "발표자료"],
        ["(추가) 소스코드·결과물 링크", "GitHub 저장소 (첫 화면에 결과 그림과 보고서 링크)", "저장소 첫 화면(README)"],
        ["종료 기록", "조회 기록 2회(10/23, 10/26) + 스크립트 (콘솔 스크린샷 없음)", "<code>aws/aws_teardown_check_*.md</code>"],
        ["(추가) 전처리 비교", "P0~P4 갤러리·통계·비교표", "<code>reports/preprocess/</code>"],
        ["(추가) 보호자 설명", "규칙 기반 설명문 코드와 예시 (보호자용 4단 문장 틀·이해도 평가는 미완)", "<code>src/explain.py</code>, <code>reports/explain/</code>"],
    ], widths=[16, 54, 30], cls="compact")

    files = tbl(["위치", "내용"], [
        ["<code>README.md</code>", "무엇이 어디에 있는지, 재현 방법, 제외한 것"],
        ["<code>manifests/</code>", "목록표와 최종 분할 (5,856장, sha256, 환자 그룹, split)"],
        ["<code>training/</code>", "학습·평가 코드: train.py(기준), train_aug.py(장비 효과 증강), train_lung.py(폐 영역), equipment_direct.py(필터 5종), shift_filters.py, evaluate.py 등"],
        ["<code>src/</code>", "분석·평가·그림 스크립트 (필터 시험 equipment_eval_direct.py, 요약 lung_summary.py, 점검 gradcam_b512_checks.py 등)"],
        ["<code>submit_*.py</code>", "SageMaker 작업 제출 스크립트"],
        ["<code>deploy/</code>", "Endpoint·Lambda·EC2 생성·삭제 스크립트와 웹 화면(Flask) 코드. 모델 가중치와 개인키는 제외"],
        ["<code>app/</code>", "224 모델 D용 Streamlit 시연 화면 (초기 버전)"],
        ["<code>results/</code>", "실험별 학습 기록 (모델 가중치 제외)"],
        ["<code>reports/</code>", "실험 보고서·표·그림, 평가 계획 문서와 해시 파일"],
        ["<code>aws/</code>", "작업 제출 기록, 종료 기록"],
        ["<code>tools/</code>", "이 보고서를 만드는 코드 (<code>tools/final_report/</code>), 종료 기록 스크립트"],
    ], widths=[22, 78], cls="compact")

    glossary = tbl(["용어", "뜻"], [
        ["Macro-F1", "정상·세균성·바이러스성 세 클래스의 F1 점수를 단순 평균한 값. 클래스 수가 달라도 고르게 맞히는지 본다."],
        ["recall (민감도)", "그 병인 사진 중 맞힌 비율. 정상 recall = 정상 사진을 정상이라고 맞힌 비율."],
        ["AUROC", "확률의 순서만으로 두 클래스를 얼마나 잘 가르는지 (0.5는 무작위, 1.0은 완벽)."],
        ["val / test", "val은 모델을 고르고 개발하는 데 쓴 사진(1,030장), test는 마지막에 확인하는 사진(618장)."],
        ["seed", "학습 때의 난수를 정하는 숫자. 데이터 분할은 그대로 두고 초기화·순서·증강의 우연만 바꾼다. 3개 평균 ± 표준편차로 흔들림을 본다."],
        ["필터 약·중·강", "같은 장비 효과를 3단계 세기로 건 시험 조건. '강'이 가장 센 단계."],
        ["연습에서 본 필터 / 못 본 필터", "장비 효과 증강 때 학습에 섞은 필터 5종(관전압·관전류·노출시간·산란·센서)과, 학습에 쓰지 않은 3종(경계 강조·대비 보정·JPEG)."],
        ["장비 효과 증강", "학습 사진의 50%에 장비 필터 5종 중 하나를 무작위 세기로 거는 데이터 증강."],
        ["'차이 있음' 판정", "두 묶음의 평균 차이가 기준 묶음 표준편차(seed 3개)의 2배 이상일 때. 아니면 '말할 수 없음'."],
        ["Grad-CAM", "모델이 판단할 때 사진의 어느 부분에 반응했는지를 색으로 보여 주는 지도. 어디만 알려 주고 무엇에 반응했는지는 모른다."],
    ], widths=[22, 78], cls="compact")

    repro = """<pre style="font-size:8pt;line-height:1.5;background:#f6f5f1;padding:6pt 9pt;border-radius:4px;white-space:pre-wrap;word-break:break-all">
# 1) 환경 (Python 3.11): pip install -r requirements.txt   # AWS 접근은 aws_config.json(제출물에 없음)에 프로필·역할·버킷을 채워서 쓴다
# 2) 학습 (SageMaker 학습 작업): 기준 모델 B512, seed 42
python submit_training.py --experiment-id b512 --extra train-mode=layer4 --extra input-size=512 --num-workers 8 --instance-type ml.g4dn.2xlarge
# 3) 필터 시험 (모델 가중치는 제출물에 없다: S3의 model.tar.gz(부록 D)를 풀어 results/<실험>/model/ 에 둔다. test는 계획 문서 해시가 맞을 때만 열림)
python src/equipment_eval_direct.py --model-dir results/b512-20261022-111844/model --split val --device cpu
# 4) 요약표와 판정
python src/lung_summary.py ;  python src/vaccine_summary.py
# 5) 이 보고서 다시 만들기
python tools/final_report/build.py
</pre>"""

    return f"""
<h1 id="secA" class="pb"><span class="no">부록 A</span>필터 33조건 전체 결과 (기준 모델 B512, seed 42)</h1>
<p class="small">val 1,030장, test 618장. 변화 = 같은 모델의 원본 점수 대비. *는 95% 구간(환자 묶음 bootstrap)이 0을 포함하지 않는 조건. 위쪽 8종이 본편 24조건, 아래 흐림·밝게·어둡게가 부록 9조건이다. 필터를 안 건 원본 점수는 val 0.843, test 0.670.</p>
{tA}
<h1 id="secB" class="pb"><span class="no">부록 B</span>초기 실험 (입력 224)</h1>
<p>10/16~10/19에 입력 224로 학습한 8개 모델이다 (seed 42, ml.g4dn.xlarge). 실험 D가 평가 계획이 지정한 최종 모델이었고, 이후 기준을 B512로 옮겼다 (2.4절).</p>
{tB}
{fig('<div>' + figH + '</div>', '<b>그림 8.</b> 초기 실험 8개의 val 점수와 test 점수. 모든 모델이 test에서 낮고(−0.058~−0.147) val 순위가 test 순위와 달랐다 (순위상관 −0.50). 전처리·증강·클래스 가중치 중 어느 것도 이 낙폭을 없애지 못했다.')}
<p class="small">B와 D의 test 점수는 구간이 겹쳐(B 0.719 [0.677, 0.757], D 0.692 [0.652, 0.732]) 차이를 말할 수 없다. D를 seed 3개로 다시 학습한 원본 val 점수는 0.834 / 0.818 / 0.818로, 노이즈 증강의 원본 점수 개선은 seed 흔들림 안쪽이었고 노이즈 약점 보완(std 16에서 F1 0.363 → 0.747~0.774)만 3개 seed에서 재현되었다. 입력 448 모델(B448)은 흐림에 더 약했다(흐림 3단계 평균 F1 0.585, B는 0.738).
전처리 4종(P1~P4)은 원본 점수가 흔들림 수준이었고 필터마다 강한 곳·약한 곳이 달랐다: CLAHE는 바이러스성 recall이 가장 높았지만 노이즈에 가장 약했고(노이즈 16에서 0.148), 블러는 노이즈·흐림에 강했다.</p>
<h1 id="secC"><span class="no">부록 C</span>가이드 필수 제출물 대응표</h1>
{deliver}
<h1 id="secD"><span class="no">부록 D</span>모델 S3 주소</h1>
<p class="small">AWS 학습 작업 모델은 <code>experiments/&lt;작업 이름&gt;/output/model.tar.gz</code>이고, 작업 이름은 <code>p1-cxr-&lt;학생ID&gt;-</code> 뒤에 표의 이름과 시각이 붙는다. JupyterLab 모델은 결과 묶음 <code>experiments/aug_runs/results_&lt;이름&gt;.tgz</code>다 (S3에서 존재를 확인함).</p>
{tD}
<h1 id="secE"><span class="no">부록 E</span>제출 패키지 구성과 재현 방법</h1>
{files}
{repro}
<p class="small">제외한 것: AWS 접근 정보(aws_config.json 등), 개인키, 가상환경, 모델 가중치(.pt, 용량이 크고 S3에 있음), 원본 영상(공개 데이터, 약 1.2 GB, 출처와 목록표 sha256으로 대신), 다른 프로젝트 폴더. 학생 ID는 <code>&lt;학생ID&gt;</code>로, 12자리 계정 번호는 <code>&lt;ACCOUNT&gt;</code>로 가렸다.</p>
<h1 id="secF"><span class="no">부록 F</span>용어</h1>
{glossary}
<p class="foot">참고문헌: [1] Kermany et al., 2018, Cell 172(5). [2] Zech et al., 2018, PLOS Medicine 15(11). [3] Selvaraju et al., 2017, Grad-CAM, ICCV. [4] He et al., 2016, ResNet, CVPR. [5] Cohen et al., 2022, TorchXRayVision, MIDL. [6] Zuiderveld, 1994, CLAHE, Graphics Gems IV.</p>
"""
