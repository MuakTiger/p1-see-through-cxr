"""결과보고서 본문 1부: 공통 도우미, 표지·요약, 1~3장."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import data as D

ROOT = D.ROOT
EQ = D.EQ


# ---------------------------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------------------------
def f3(x):
    return f"{x:.3f}".replace("-", "−")


def sg(x, d=3):
    return ("+" if x >= 0 else "−") + f"{abs(x):.{d}f}"


def pm(mean, sd, d=3):
    return f"{mean:.{d}f} ± {sd:.{d}f}".replace("-", "−")


def tbl(headers, rows, align=None, widths=None, cls="", sub=(), hl=(), keep=None):
    """headers: 문자열 목록, rows: 칸이 HTML 문자열인 목록의 목록. align: 'lnnc' 처럼 칸마다 l(왼쪽)/n(오른쪽 숫자)/c(가운데)."""
    align = align or "l" * len(headers)
    cm = {"l": "", "n": ' class="n"', "c": ' class="c"'}
    keep = (len(rows) <= 14) if keep is None else keep
    out = [f'<table class="t {cls}{" keep" if keep else ""}">']
    if widths:
        out.append("<colgroup>" + "".join(f'<col style="width:{w}%">' for w in widths) + "</colgroup>")
    out.append("<thead><tr>" + "".join(f"<th{cm[a]}>{h}</th>" for h, a in zip(headers, align)) + "</tr></thead><tbody>")
    for i, r in enumerate(rows):
        rc = ' class="sub"' if i in sub else (' class="hl"' if i in hl else "")
        out.append(f"<tr{rc}>" + "".join(f"<td{cm[a]}>{c}</td>" for c, a in zip(r, align)) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def verdict(v):
    """요약 CSV의 판정 문구를 짧게 줄인다."""
    if v.startswith("차이를 말할 수 없음"):
        return '<span class="dim">말할 수 없음</span>'
    up = ("앞 모델이 높음" in v) or ("예방접종이 높음" in v)
    dn = ("앞 모델이 낮음" in v) or ("예방접종이 낮음" in v)
    arrow = "↑" if up else ("↓" if dn else "")
    tail = " (배운 유형)" if "배운 유형" in v else ""
    return f"<b>차이 있음 {arrow}</b>{tail}"


def read_lung_csv():
    rows = {}
    with open(EQ / "lung/lung_summary.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["split"], r["comparison"], r["metric"])] = r
    return rows


def read_vac_csv():
    rows = {}
    with open(EQ / "vaccine_summary.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["split"], r["metric"])] = r
    return rows


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fig(svg_or_html, caption, label=None):
    return f'<figure>{svg_or_html}<figcaption>{caption}</figcaption></figure>'


# ---------------------------------------------------------------------------------------------
# 표지와 요약
# ---------------------------------------------------------------------------------------------
def cover(ctx, toc_html):
    A, L = ctx["data"]["aws_b"], ctx["data"]["lung"]
    JB, JV = ctx["data"]["jl_b"], ctx["data"]["jl_v"]
    v0, v0sd, _ = D.agg(A["val"], "orig_f1")
    t0, t0sd, _ = D.agg(A["test"], "orig_f1")
    kv_val, _ = D.agg_strong(A["val"], "kvp", True)
    kv_test, _ = D.agg_strong(A["test"], "kvp", True)
    dv_a, dt_a = D.agg(A["val"], "drop24")[0], D.agg(A["test"], "drop24")[0]
    dv_l, dt_l = D.agg(L["val"], "drop24")[0], D.agg(L["test"], "drop24")[0]
    b5, _, _ = D.agg(JB["test"], "learned5")
    v5, _, _ = D.agg(JV["test"], "learned5")
    repo = ctx.get("repo_url", "")
    repo_row = (f'<tr><td>소스코드·데이터</td><td>GitHub 저장소 <b>{repo.replace("https://", "")}</b> (코드, 목록표·분할, 실험별 학습 기록, 보고서·발표자료, 종료 기록)</td></tr>\n' if repo else "")
    return f"""
<p class="kicker">2026 성균관대학교 AI CAMPUS 바이오헬스 AI 교육과정 1기 · [P1] CXR 흉부 질환 다중 분류 프로젝트 · 결과보고서</p>
<h1 class="title" style="border:0;margin:0 0 6pt;padding:0">See-Through CXR</h1>
<p class="subtitle">장비 설정이 만드는 영상 차이에 흉부 X선 폐렴 3분류 모델은 얼마나 흔들리는가</p>
<table class="meta">
<tr><td>작성자</td><td>오형준 (개인 프로젝트) · 보고서 작성 2026.10.26</td></tr>
<tr><td>수행 기간</td><td>2026.10.13 (화) ~ 10.23 (금) · 최종 발표 10.23</td></tr>
<tr><td>데이터</td><td>Chest X-Ray Images (Pneumonia), Kermany et al. 2018 · 5,856장 · 정상 / 세균성 폐렴 / 바이러스성 폐렴</td></tr>
<tr><td>모델 · 환경</td><td>ResNet18 (ImageNet 사전학습) 전이학습, 입력 512×512 · AWS 서울 리전 (S3, SageMaker 학습 작업·Endpoint, Lambda, EC2)</td></tr>
<tr><td>기준 문서</td><td>프로젝트 기획안 v0.4 (2026.10.14 제출) · 과정 AWS 프로젝트 가이드 v3</td></tr>
{repo_row}</table>
<p class="lead"><b>한 줄 결론.</b> 원본 사진으로 잰 점수(val {v0:.2f}, test {t0:.2f})만으로는 장비가 바뀔 때의 성능을 알 수 없다.
장비 설정을 흉내 낸 필터를 걸자 점수가 크게 떨어졌고, 폐만 남긴 모델에서도 같은 약점이 남았다. 기획안과 달라진 점이 많아서 <b>2장</b>에 바뀐 내용과 이유, 기획 목표 대비 실측값을 먼저 정리했다.</p>
<div class="cards">
<div class="card"><div class="big">{v0:.3f} <span class="sm">/ {t0:.3f}</span></div><div class="lab">기준 모델의 원본 사진 점수, val / test (Macro-F1, 모델 3개 평균)</div></div>
<div class="card hl"><div class="big">{sg(kv_val)}</div><div class="lab">관전압 필터(가장 센 단계)에서 val 점수 변화. test는 {sg(kv_test)} (모델 3개 평균)</div></div>
<div class="card"><div class="big">{sg(dv_a)} <span class="sm">/ {sg(dv_l)}</span></div><div class="lab">필터 24조건 평균 점수 변화(val), 기준 모델 / 폐만 남긴 모델. test는 {sg(dt_a)} / {sg(dt_l)}</div></div>
<div class="card"><div class="big">{b5:.3f} <span class="sm">→ {v5:.3f}</span></div><div class="lab">장비 효과 증강 뒤 test 점수 (학습에 섞은 필터 5종, 기준 모델). 같은 종류 필터라 실제 장비에 강해졌다는 증거는 아님</div></div>
</div>
<h2 style="margin-top:8pt">목차</h2>
{toc_html}
"""


# ---------------------------------------------------------------------------------------------
# 1장 개요
# ---------------------------------------------------------------------------------------------
def sec1():
    return """
<h1 id="sec1" class="pb" style="margin-top:0"><span class="no">1</span>프로젝트 개요</h1>
<h2>1.1 배경과 목적</h2>
<p>흉부 X선 사진은 병원과 촬영 조건에 따라 밝기·대비·노이즈·선명도가 달라진다. 관전압(kVp), 관전류와 노출시간, 산란선을 줄이는 격자, 검출기 화소 크기, 제조사의 영상 후처리가 모두 영상에 흔적을 남기기 때문이다.
기획안은 한 병원 데이터로 학습한 폐렴 모델이 병원별 장비의 특성이나 영상 위 글자·마커 같은 '지름길'을 배워서 다른 병원에서 성능이 떨어진 사례(Zech et al., 2018)를 문제의식으로 삼았다.</p>
<p>우리가 가진 데이터는 광저우의 한 병원 사진뿐이라 다른 병원 사진으로 직접 확인할 수 없다. 그래서 <b>같은 사진에 장비 설정이 만드는 효과를 고전 영상처리로 흉내 내어 걸어 보고, 모델의 점수가 얼마나 변하는지 재는 방법</b>을 택했다.
기획안의 '정확도 숫자만으로는 맞는 이유로 맞혔는지 알 수 없다'는 문제의식을 그대로 이어받아, 설명을 보여 주는 것에서 한 걸음 더 나아가 <b>모델을 흔들어 보고 확인하는 것</b>으로 무게를 옮겼다.</p>
<h2>1.2 연구 질문</h2>
<table class="t">
<thead><tr><th style="width:15%">구분</th><th>질문</th><th style="width:20%">이 보고서의 위치</th></tr></thead><tbody>
<tr><td><b>시험 ①</b></td><td>원본 사진으로 학습한 기준 모델은 장비 효과를 건 사진에서 점수가 얼마나 떨어지는가?</td><td>5.2절</td></tr>
<tr><td><b>시험 ②</b></td><td>점수가 떨어지는 이유가 모델이 폐 밖의 단서를 쓰기 때문인가? 폐 영역만 남겨도 같은가?</td><td>5.3절</td></tr>
<tr><td><b>시험 ③</b></td><td>장비 효과를 섞어서 학습하면 약점이 줄어드는가?</td><td>5.4절</td></tr>
<tr><td><b>점검</b></td><td>test 점수(특히 정상 사진의 recall)를 그대로 믿어도 되는가? 모델은 어디를 보는가?</td><td>6장</td></tr>
</tbody></table>
<h2>1.3 이 보고서의 숫자를 읽는 법</h2>
<ul>
<li><b>val은 개발용, test는 마지막 확인용</b>이다. 필터·강도·모델 설계의 근거는 val 결과와 장비에 대한 사전 지식뿐이고(알고 있던 test 결과는 근거로 쓰지 않았다, 2.4절), test에는 같은 필터를 한 번 걸어 같은 방향인지 확인했다.</li>
<li><b>모델은 설정마다 seed 3개(42·43·44)</b>로 학습했다. seed는 난수만 바꾸고 데이터 분할은 같다. 점수는 3개의 평균 ± 표준편차로 적고, 두 묶음의 평균 차이가 <b>기준 묶음 표준편차의 2배 이상</b>일 때만 '차이 있음'이라 쓴다. 그보다 작으면 '말할 수 없음'이다.</li>
<li>학습 환경이 둘이다 (AWS 학습 작업 torch 2.2, SageMaker JupyterLab torch 2.8). 같은 설정이어도 환경이 다르면 점수가 달라질 수 있어서 <b>비교는 같은 환경의 모델끼리만</b> 했다.</li>
<li>용어는 부록 F에 있다.</li>
</ul>
"""


# ---------------------------------------------------------------------------------------------
# 2장 기획안 대비 변경 사항
# ---------------------------------------------------------------------------------------------
def sec2(ctx):
    A, L = ctx["data"]["aws_b"], ctx["data"]["lung"]
    t_f1, t_sd, t_vals = D.agg(A["test"], "orig_f1")
    l_f1, l_sd, l_vals = D.agg(L["test"], "orig_f1")
    sens = [A["test"][s]["pneu_sens"] for s in D.SEEDS]
    sens_l = [L["test"][s]["pneu_sens"] for s in D.SEEDS]
    lf = ctx["lungfocus"]
    ver = ctx["verify"]

    changes = [
        ["<b>중심 주제</b>",
         "블랙박스 없이: Grad-CAM, 폐 영역 집중도(LFR), 확신도 기반 판단 보류, 보호자용 쉬운 설명, 고전 전처리 비교",
         "<b>장비 설정 차이에 대한 점수 민감도</b>: 필터 8종 시험, 폐 영역 모델, 장비 효과 증강. Grad-CAM은 점수가 낮은 이유를 진단하는 도구로 유지",
         "10/19 첫 test에서 정상 recall이 val 0.98 → test 0.44로 떨어지고 모델 8개가 모두 같았다. 점수를 믿을 수 있는지가 먼저 답할 질문이 되었고, 기획안의 동기(장비 지름길)를 직접 시험하는 쪽으로 옮겼다"],
        ["<b>입력 크기</b>", "224×224 (가이드의 시작 제안)", "<b>512×512</b>",
         "강사 의견: 224로 줄이면 필터 효과가 뭉개진다. 224 시험에서 센서 0.9 mm 필터의 점수 변화가 0이었다"],
        ["<b>기준 모델</b>",
         "검증 점수로 고른 최종 모델 (실제로는 실험 D: 노이즈 증강, 224)",
         "<b>B512</b>: 증강·전처리 없이 layer4+fc만 학습, 512, seed 3개",
         "연구 질문이 '기본 모델의 민감도'라서 증강 모델 D를 기준에서 내렸다. D는 노이즈 시험과 같은 종류의 증강이라 '배운 유형으로 본 시험'이 되기도 했다 (<a href=\"#sec24\">2.4</a> 공개 사항 참고)"],
        ["<b>전처리 비교 (P 시리즈)</b>", "HE · CLAHE · LPF · HPF 전처리 4종을 224에서 비교하고 검증 점수로 최종 전처리 선정",
         "P1~P4 학습·평가는 <b>완료</b>(224, 부록 B). 결과: 원본 점수 차이는 흔들림 수준이라 전처리는 원본(P0) 유지. 이 고전 영상처리 아이디어를 <b>장비 설정을 흉내 내는 필터 8종</b>으로 확장",
         "전처리마다 장비 차이에 강한 곳·약한 곳이 달랐다 (예: CLAHE는 노이즈에 가장 약했다). 필터를 '학습 전 정리'가 아니라 '시험 조건'으로 쓰는 편이 질문에 맞았다"],
        ["<b>설명 가능성</b>", "Grad-CAM + LFR + 판단 보류 + 근거 문장 + 보호자용 문장 틀, Endpoint가 Grad-CAM·LFR까지 계산",
         f"Grad-CAM 구현·점검 완료, 폐 마스크로 '폐 안 비율' 측정({lf['b512']:.2f}, 폐 면적 비율 {lf['area']:.2f}). 규칙 기반 설명문은 224 모델 D로 구현했으나 최종 흐름에는 연결하지 않음. <b>판단 보류 곡선·보호자용 4단 문장 틀·이해도 평가는 하지 않음</b>",
         "우선순위를 '필수 먼저'로 정리하면서 모델·주제가 바뀌었고 일정(실제 작업일 약 7일)이 부족했다. 배포 모델(Endpoint)도 확률 3개만 돌려준다"],
        ["<b>폐 마스킹 실험 (E)</b>", "선택(Could) 항목", "<b>수행</b>: PSPNet 폐 마스크로 폐 밖을 지운 모델 3개 (+ 장비 효과 증강 3개)",
         "Grad-CAM이 폐 밖에 반응하는 것을 보고, 필터 때문에 점수가 떨어지는 것이 '폐 밖 단서' 탓인지 확인하려고 승격했다"],
        ["<b>증강</b>", "실험 D: 증강 조건 1개 (회전 또는 밝기)",
         "초기: 노이즈 증강 D (224). 최종: <b>장비 효과 증강</b> — 학습 사진의 50%에 장비 필터 5종 중 하나를 무작위 세기로",
         "필터 시험에서 노이즈에 크게 무너진 약점(노이즈 std 16에서 F1 0.363)이 먼저 보여 노이즈 증강을 해 보았고, 이를 장비 설정 5종 전체로 넓혔다"],
        ["<b>반복·통계</b>", "단일 seed 42 (시간이 되면 최종 후보 2개를 seed 3개로)", "<b>모든 모델을 seed 3개</b>로 학습, '기준 표준편차의 2배' 판정 규칙, 환자 묶음 bootstrap 구간",
         "D를 seed 3개로 다시 학습하자 원본 점수 개선(0.834)이 seed 흔들림(0.818~0.834) 안쪽으로 드러났다. 한 번의 점수로 모델을 비교하면 안 된다는 것을 확인했다"],
        ["<b>test 사용</b>", "최종 후보에 대해 1회", "224 모델 8개에 각 1회 (평가 계획 해시 고정), 이후 B512 계열은 <b>모델마다 사전 계획 문서와 해시를 고정한 뒤 1회</b>. test 점수를 보고 모델·필터·강도를 바꾸지 않음",
         "test가 이미 열렸으므로 '이미 열린 test에서의 사전 지정 추가 분석'이라고 계획 문서 1장에 밝혔다"],
        ["<b>AWS 서비스</b>", "S3 → 학습 → Endpoint(예측+Grad-CAM+LFR) → EC2 Flask API + Streamlit 화면",
         "S3 → 학습 작업 → <b>Endpoint(확률 3개) → Lambda → EC2 Flask 화면 2종</b>(사진 올리기, 필터 체험). 10/23 배포·검산·발표 시연, <b>발표 직후 모두 삭제</b>",
         "강의 실습서 v4의 구조를 따랐다. Grad-CAM·설명문은 로컬 Streamlit 화면(224 모델 D)에서만 구현했다"],
        ["<b>제출물</b>", "시연 영상(MP4), 종료 체크리스트와 콘솔 스크린샷",
         "시연은 <b>발표 중 실시간</b>(화면 캡처는 발표자료 12쪽). 시연 영상 파일은 없음. 종료 기록은 <b>콘솔 스크린샷 대신 CLI 조회 기록</b>(7.5절)",
         "배포를 발표 당일 오전에 만들었고 발표 직후 지웠다. 영상 파일은 이 제출물에 포함하지 않았다"],
    ]
    t1 = tbl(["항목", "기획안 (10/14)", "최종 결과물", "바뀐 이유"], changes, widths=[10, 24, 34, 32], cls="compact", keep=False)

    timeline = [
        ["10/13~14", "기획안 작성·제출(v0.4), 목록표·환자 단위 분할, 데이터 점검", "train 4,186 / val 1,030 / 점검용 16 / test 624로 확정"],
        ["10/16~19", "입력 224로 실험 A(몸통 고정)·B(layer4)·C(클래스 가중치)·D(노이즈 증강)와 전처리 P1~P4 학습. val에 필터 8종×3단계를 건 '장비 차이 시험' 첫 시도",
         "B가 노이즈에 크게 무너짐(std 16에서 F1 0.363, 예측이 바이러스성으로 쏠림) → 노이즈 증강 D"],
        ["10/19", "test 1회 평가 (224 모델 8개)", "D의 test F1 0.692, <b>정상 recall 0.442</b>. 8개 모델 모두 val보다 0.06~0.15 낮고 val 순위가 test 순위를 예측하지 못함"],
        ["10/20", "원인 분석(파일 저장 방식·가로세로 비율·Grad-CAM), 입력 448 실험, 강사 의견 반영", "저장 화질은 원인이 아님을 확인. 224에서는 필터 효과가 뭉개진다는 의견 → 512로 이동"],
        ["10/21", "D를 seed 3개로 반복, 분할 점검, Streamlit 시연 화면(D, 로컬)", "D의 원본 점수 개선은 seed 흔들림 안쪽. 노이즈 약점 보완만 3개 seed에서 재현"],
        ["10/22", "<b>주제 전환</b>: 장비 설정 필터 v2(원리 기반 5종)·512·기준 모델 B512 확정, 필터 강도를 문서로 고정 → val 시험 → test 1회. 폐 영역 모델·장비 효과 증강을 seed 3개씩 학습·평가",
         "필터 시험 ①②③의 결과 확보 (5장)"],
        ["10/23", "AWS 배포(Endpoint → Lambda → EC2), 검산, 최종 발표, 자원 삭제", "배포 모델 = 로컬 모델 (예측 20/20 일치). 삭제 후 0개 확인 (7.5절)"],
    ]
    t2 = tbl(["날짜", "한 일", "그 결과 · 바뀐 것"], timeline, widths=[10, 52, 38], cls="compact")

    # 기획 성공 기준 대비 실측
    missed = [A["test"][s]["pneu_missed"] for s in D.SEEDS]
    crit = [
        ["<b>Q1</b> 테스트 Macro-F1 ≥ 0.75", f"기준 모델 <b>{t_f1:.3f}</b> ± {t_sd:.3f} ({' / '.join(f'{v:.3f}' for v in t_vals)}). 폐 영역 모델 {l_f1:.3f} ± {l_sd:.3f} ({' / '.join(f'{v:.3f}' for v in l_vals)})", '<span class="j-no">○ 미달</span> (기준) · 폐 모델은 평균만 달성'],
        ["<b>Q1</b> 폐렴 민감도 ≥ 0.95 (세균성+바이러스성 묶음)", f"기준 모델 {min(sens):.3f}~{max(sens):.3f} (387장 중 정상으로 놓친 폐렴 {min(missed)}~{max(missed)}장), 폐 영역 모델 {min(sens_l):.3f}~{max(sens_l):.3f}", '<span class="j-ok">● 달성</span>'],
        ["<b>Q2</b> 세균↔바이러스 혼동 유형 3개 이상 도출", "혼동행렬로 방향은 확인(test, seed 42: 바이러스성 147장 중 51장이 세균성으로, 세균성 240장 중 6장이 바이러스성으로). 오분류 사진을 보고 유형으로 나누는 작업은 하지 못함", '<span class="j-part">◐ 부분</span>'],
        ["<b>Q3</b> 정답 예측의 평균 LFR ≥ 0.70", f"열지도 중 폐 안 비율 정답 사진 평균 <b>{lf['right']:.3f}</b> (폐 면적 비율 {lf['area']:.3f}), val 1,030장. 열지도가 폐에 몰리지 않음", '<span class="j-no">○ 미달</span>'],
        ["<b>Q4</b> Endpoint 예측 = 로컬 예측 (라벨 100%, 확률 차이 < 1e-4)", "val 20장 예측 20/20 일치, 확률 최대 차이 7.5e-6 (비교 대상은 SageMaker Notebook이 아닌 내 컴퓨터의 같은 모델)", '<span class="j-ok">● 달성</span>'],
        ["<b>Q5</b> 사진 1장 응답 ≤ 3초 (Grad-CAM 포함)", "화면 → Lambda → Endpoint 전체 경로 1.1초 (확률 3개만, Grad-CAM 없음)", '<span class="j-part">◐ 부분</span>'],
        ["<b>Q6</b> 보호자 설명: 문장 오류 0건, 이해도 ≥ 4.0", "규칙 기반 설명문을 구현해 결과 8종류 모두 오류 없이 실행(224 모델 D 기준). 이해도 평가(10건)와 보호자용 4단 문장 틀은 하지 않음", '<span class="j-no">○ 미실시</span>'],
        ["<b>Q7</b> 전처리별 비교표 완성, 검증 기준으로 선정", "P0~P4의 val Macro-F1·클래스별 recall·장비 차이 시험·test 점수 비교표 완성(224, 부록 B). LFR 비교는 하지 않음. 원본(P0) 유지", '<span class="j-part">◐ 부분</span>'],
    ]
    t3 = tbl(["기획안의 성공 기준 (계획값)", "실측", "판정"], crit, widths=[27, 55, 18], cls="compact")

    return f"""
<h1 id="sec2"><span class="no">2</span>기획안 대비 변경 사항</h1>
<p>기획안(10/14)은 <b>설명 가능한 폐렴 3분류 서비스</b>를 목표로 했다. 실제로 한 일은 <b>'점수를 믿을 수 있는가, 장비가 바뀌면 점수가 얼마나 변하는가'를 시험하는 쪽</b>으로 옮겨 갔다.
옮겨 간 이유는 첫 test에서 정상 recall이 0.44로 떨어진 일과 강사 의견이고, 방향은 기획안 1.1절의 문제의식(장비 지름길, 숫자만으로는 맞는 이유로 맞혔는지 모른다)과 이어진다.
기획안 1.3절에서 약속한 대로 계획값 옆에 실측값을 나란히 적는다(2.3절).</p>
<h2>2.1 무엇이 어떻게 바뀌었나</h2>
{t1}
<p class="small">바뀌지 않은 것: 데이터(Kermany 5,856장)와 3분류, ResNet18 전이학습, Adam·배치 16·최대 30에폭·검증 Macro-F1 최고 에폭 선택, 환자 그룹 단위 분할, test 624장 고정, 기획서의 범위 밖 항목(Bedrock·LLM 설명, 진단 목적 사용 금지).</p>
<h2 id="sec22">2.2 바뀐 순서</h2>
{t2}
<h2>2.3 기획안의 성공 기준 대비 실측</h2>
<p>기획안 2.1절의 계획값(성공 기준) 7개를 그대로 가져와 실측값을 옆에 적었다. ●는 달성, ◐는 부분 달성, ○는 미달 또는 미실시다.</p>
{t3}
<h2 id="sec24">2.4 알려 둘 사항 (사전 계획과 달라진 점)</h2>
<div class="note warn">
<p><b>기준 모델을 test를 본 뒤에 바꿨다.</b> 10/14 기획안과 10/19 평가 계획(해시 고정)이 지정한 최종 모델은 실험 D(노이즈 증강, 224)였다. 그런데 연구 질문이 '기본 모델의 민감도'여서, test를 본 뒤 기준을 B 설정(증강 없음)으로 바꿨다.
두 모델의 test 점수는 B 0.719 [0.677, 0.757], D 0.692 [0.652, 0.732]로 구간이 겹쳐 어느 쪽이 낫다고 말할 수 없었고, 둘 다 보고한다 (부록 B).</p>
<p><b>test는 여러 번 열렸다.</b> 224 모델 8개에 한 번씩, B512 계열은 모델마다 한 번씩 평가했다. B512 계열은 평가 계획 문서와 sha256 해시를 먼저 고정하고 실행했으며, <b>결과를 보고 모델·필터·강도를 바꾸지 않았다.</b> 다만 장비 효과 증강과 폐 영역 모델을 설계할 때 B512의 test 결과를 이미 알고 있었다는 점을 각 계획 문서 1장에 밝혔다.
가이드는 test 점수를 보고 설정을 반복해서 조정하지 말라고 한다. 이 프로젝트는 알고 있던 test 결과로 설계를 바꾸지 않았고(설계 근거는 val 결과와 장비 물리 지식뿐), 다만 이미 열린 데이터라는 한계는 남는다.</p>
</div>
"""


# ---------------------------------------------------------------------------------------------
# 3장 데이터와 분할
# ---------------------------------------------------------------------------------------------
def sec3(ctx):
    ns = ctx["normal_shift"]
    lb = ctx["lungbox"]
    split_rows = []
    with open(ROOT / "manifests/split_summary.csv", encoding="utf-8") as f:
        ss = list(csv.DictReader(f))

    def get(sp, label):
        for r in ss:
            if r["split"] == sp and r["label_name"] == label:
                return int(r["images"]), int(r["patient_groups"])
        raise KeyError((sp, label))

    spec = [("train", "학습"), ("val", "에폭 선택·개발·필터 시험"), ("check", "화면·입출력 점검 (원래 val, 점수 계산 안 함)"), ("test", "최종 확인")]
    rows = []
    for sp, use in spec:
        a, b, c, t = get(sp, "NORMAL"), get(sp, "BACTERIAL_PNEUMONIA"), get(sp, "VIRAL_PNEUMONIA"), get(sp, "ALL")
        tot = f"{t[0]:,}" + (" (평가 618)" if sp == "test" else "")
        grp = f"{t[1]:,}" + (" (426)" if sp == "test" else "")
        rows.append([f"<b>{sp}</b>", f"{a[0]:,}", f"{b[0]:,}", f"{c[0]:,}", tot, grp, use])
    t4 = tbl(["구분", "정상", "세균성", "바이러스성", "합계", "환자 그룹", "용도"], rows, align="lnnnnnl", widths=[9, 9, 9, 11, 16, 11, 35], cls="compact")

    tbl5 = tbl(
        ["특성", "train·val 정상", "폐렴 (train)", "test 정상"],
        [["JPEG 저장 품질", f"약 {ns['jpeg_trn']:.0f}", f"약 {ns['jpeg_pn']:.0f}", f"<b>약 {ns['jpeg_tn']:.0f}</b>"],
         ["포토샵 처리 기록이 있는 사진", f"{ns['ps_trn']*100:.0f}%", f"{ns['ps_pn_b']*100:.0f}~{ns['ps_pn_v']*100:.0f}%", f"<b>{ns['ps_tn']*100:.0f}%</b>"],
         ["원본 가로세로 비율 (중앙값)", f"{ns['asp_trn']:.2f}", f"{ns['asp_b']:.2f}~{ns['asp_v']:.2f}", f"<b>{ns['asp_tn']:.2f}</b>"],
         ["폐 상자가 사진 세로를 채우는 비율 (중앙값)", f"{lb['v_norm_tv']:.3f}", f"{lb['v_pneu_tv']:.3f}", f"<b>{lb['v_norm_test']:.3f}</b>"]],
        widths=[40, 20, 20, 20], align="lccc", cls="compact")

    return f"""
<h1 id="sec3"><span class="no">3</span>데이터와 분할</h1>
<h2>3.1 데이터</h2>
<p>Kermany 등(2018)이 공개한 광저우 여성아동의료센터 1~5세 소아 흉부 X선(전후면) 5,856장이다. 폐렴 폴더의 파일명(bacteria / virus)으로 세균성·바이러스성 라벨을 만들었고, 파일명은 라벨을 만드는 데만 쓰고 모델 입력에는 넣지 않았다.
원래 분할은 train 5,216장(정상 1,341 / 세균성 2,530 / 바이러스성 1,345), val 16장(8 / 8 / 0), test 624장(234 / 242 / 148)이다.</p>
<h2>3.2 분할</h2>
<p>원래 val은 16장뿐이고 <b>바이러스성이 0장</b>이라 검증셋으로 쓸 수 없었다. 그래서 원래 train 5,216장에서 <b>환자 그룹 단위·클래스 층화</b>(StratifiedGroupKFold, seed 42, 5등분의 한 조각)로 학습과 검증을 다시 나눴다. test 624장은 원래대로 두었다.
환자 그룹은 파일명의 환자 번호(personN)로 묶었고 같은 번호는 한 덩어리로 다뤄 같은 환자가 학습과 검증에 섞이지 않게 했다. 같은 파일(sha256 동일)도 같은 쪽에 두었다.</p>
{t4}
<p class="small">분할 규칙과 근거: <code>manifests/split_report.md</code>, 점검 결과: <code>reports/seed_check_and_split_balance.md</code>. 분할 파일의 sha256은 <code>{ctx['split_sha'][:16]}…</code> (모든 실험이 같은 파일을 썼다).</p>
<h2>3.3 점검한 것</h2>
<ul>
<li><b>train과 val의 분포는 같다.</b> 클래스별 val 비율 19.3~20.5%, 그룹 수·그룹당 사진 수·저장 품질·가로세로 비율이 거의 같다.</li>
<li><b>test 안에 같은 사진 6쌍</b>(sha256 동일)이 있어 평가에서 사본 6장을 빼고 <b>618장</b>(정상 231 / 세균성 240 / 바이러스성 147, 환자 그룹 426)으로 평가했다.</li>
<li>원래 train과 test에 같은 번호의 환자 그룹 170개(457장)가 있으나 모두 병이 서로 달라(같은 병이 겹치는 것 0) 제거하지 않았고, test에서 누수의 증거는 확인하지 못했다.</li>
<li>세균성 사진이 바이러스성의 약 1.9배다(환자 수로는 약 1.2배: 세균성 환자 1명당 사진이 많다). 클래스 가중치(실험 C)는 효과가 작아 최종 설정에 넣지 않았다.</li>
</ul>
<h2>3.4 test의 정상 사진은 train·val의 정상 사진과 다르다</h2>
<p>val은 train에서 떼어 낸 것이라 train과 같은 분포인데, test의 정상 사진은 다른 묶음이다. 아래는 사진 파일 자체의 특성이다.</p>
{tbl5}
<p>train·val의 정상 사진은 모두 품질 약 96으로 저장되고 포토샵 처리 기록이 있는 반면, 폐렴 사진과 test 정상 사진은 품질 약 75에 기록이 없다. 파일 속성만으로 정상/폐렴을 맞히는 분류기가 train 안에서는 거의 완벽(AUROC ≈ 1.00)한데 test에서는 {ns['attr_auroc_test']:.2f}로 내려간다 (<code>reports/test_eval/normal_shift/</code>).
이 차이가 <b>val 점수가 test 점수를 예측하지 못하는 이유의 후보</b>이고(6장), 확인된 원인은 아니다. 이 때문에 test 정상 사진의 절대 점수보다 <b>같은 모델의 필터 전후 변화량과 방향</b>을 중심으로 해석했다.</p>
"""
