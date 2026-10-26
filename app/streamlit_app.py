"""시연 화면 (Streamlit): 사진 한 장 → 확률 + Grad-CAM 열지도 + 규칙 기반 설명문.

모델은 최종 모델 D (seed 42). 확률·열지도·문장은 모두 src/explain.py 의 함수를 그대로 쓴다
(LLM 없음, 같은 입력이면 같은 결과). 이 화면은 맥에서 직접 돌리는 시연용이고
SageMaker Endpoint / EC2 에 올린 것이 아니다.

실행 (p1 폴더에서):
    .venv/bin/streamlit run app/streamlit_app.py --server.address localhost
(--server.address localhost 가 없으면 같은 네트워크의 다른 컴퓨터도 접속할 수 있다.)
사진 폴더가 ~/data/chest_xray 가 아니면 환경변수 CXR_IMAGES 로 알려 준다.
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import explain as E  # noqa: E402  (모델 D 경로, 입력 전처리, 설명문 규칙)
from gradcam import GROUP_KO, GROUPS, GradCAM, denorm_gray, overlay, setup_matplotlib  # noqa: E402
from shift_eval import load_model  # noqa: E402

IMAGES = Path(os.environ.get("CXR_IMAGES", "~/data/chest_xray")).expanduser()
PREDICTIONS = ROOT / "reports/test_eval/aws/test_predictions.csv"
HOLD = "판단 보류 (확률 70% 미만)"
N_PER_GROUP = 10  # 예시 목록에 보여 줄 사진 수 (무작위, seed 42)


@st.cache_resource
def get_cam_model():
    return GradCAM(load_model(E.D_DIR))


@st.cache_data
def load_examples():
    """test 결과 그룹마다 무작위(seed 42) N_PER_GROUP 장. 평가에서 제외한 사본(excluded)은 뺀다."""
    with open(PREDICTIONS, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["excluded"] == "False"]
    probs = lambda r: max(float(r["p_normal"]), float(r["p_bacterial"]), float(r["p_viral"]))  # noqa: E731
    rng = np.random.default_rng(42)
    groups = {}
    for key, t, p, ko in GROUPS:
        pool = [r for r in rows if int(r["true_label"]) == t and int(r["pred_label"]) == p and probs(r) >= E.TAU]
        if pool:
            idx = rng.choice(len(pool), size=min(N_PER_GROUP, len(pool)), replace=False)
            groups[GROUP_KO[key]] = [pool[i] for i in idx]
    hold = [r for r in rows if probs(r) < E.TAU]
    idx = rng.choice(len(hold), size=min(N_PER_GROUP, len(hold)), replace=False)
    groups[HOLD] = [hold[i] for i in idx]
    return groups


def pick_source():
    """사이드바에서 사진을 고른다. (파일 경로 또는 업로드 파일, 정답 이름 또는 None) 을 돌려준다."""
    mode = st.sidebar.radio("사진 고르기", ["test 예시", "직접 올리기"])
    if mode == "직접 올리기":
        up = st.sidebar.file_uploader("흉부 X선 사진 (jpg, png)", type=["jpg", "jpeg", "png"])
        return up, None
    if not PREDICTIONS.exists() or not IMAGES.exists():
        st.sidebar.error(f"예시를 쓰려면 {PREDICTIONS.name} 과 사진 폴더({IMAGES})가 필요합니다.")
        return None, None
    groups = load_examples()
    group = st.sidebar.selectbox("결과 종류", list(groups))
    rows = groups[group]
    names = [r["relative_path"].split("/", 1)[1] for r in rows]
    name = st.sidebar.selectbox("사진", names)
    row = rows[names.index(name)]
    truth = E.CLASS_KO[int(row["true_label"])]
    return IMAGES / row["relative_path"], truth


def show_lines(lines):
    for ln in lines:
        tag, _, rest = ln.partition("] ")
        tag = tag[1:]
        if tag == "경고":
            st.warning(rest)
        else:
            st.markdown(f"**{tag}** {rest}")


st.set_page_config(page_title="블랙박스 없이 · 흉부 X선 폐렴 3분류", layout="wide")
st.title("블랙박스 없이")
st.caption("흉부 X선 폐렴 3분류 (정상 · 세균성 · 바이러스성) · 모델 D (ResNet18, 노이즈 증강) · 연구용이며 진단이 아닙니다.")

source, truth = pick_source()
if source is None:
    st.info("왼쪽에서 사진을 고르거나 올려 주세요.")
    st.stop()

try:
    x = E.load_input(source)[None]  # (1,3,224,224). 학습·평가와 똑같은 전처리
except Exception as e:  # 이미지가 아닌 파일 등
    st.error(f"사진을 읽지 못했습니다: {type(e).__name__}")
    st.stop()

(p, cam, lines, tier), = E.explain_batch(get_cam_model(), x)
gray = denorm_gray(x)[0]
top = int(np.argmax(p))

left, right = st.columns([1.1, 1])
with left:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    overlay(axes[0], gray, None, "모델이 본 사진 (224×224)")
    overlay(axes[1], gray, cam, "열지도 (붉을수록 많이 참고)")
    E.draw_zone_lines(axes[1])
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("점선은 흉부 상자 근사와 6구역. 폐 영역을 따로 찾은 것이 아니고, 열지도는 참고한 곳일 뿐 병변의 증거가 아닙니다.")

with right:
    if tier == "보류":
        st.warning(f"판단 보류: 가장 높은 확률이 {E.CLASS_KO[top]} {p[top] * 100:.1f}% ({E.TAU * 100:.0f}% 미만)")
    else:
        st.subheader(f"{E.CLASS_KO[top]} {p[top] * 100:.1f}% (확신도: {tier})")
    if truth is not None:
        st.caption(f"정답: {truth}" + (" · 맞힘" if truth == E.CLASS_KO[top] else " · 틀림") if tier != "보류"
                   else f"정답: {truth}")
    for name, prob in zip(E.CLASS_KO, p):
        st.progress(float(prob), text=f"{name} {prob * 100:.1f}%")
    show_lines(lines)
