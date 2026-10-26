"""결과보고서의 그림 (인라인 SVG). 모든 값은 data.py 가 저장된 결과 파일에서 읽은 것이다.

색과 마크 규격은 dataviz 규칙을 따른다: 범주 색 두 가지(파랑 #2a78d6, 주황 #eb6834)는 항상 이 순서로 쓰고,
색만으로 구분하지 않도록 점 모양(원/마름모)과 범례를 함께 쓴다. 글자는 데이터 색이 아니라 글자 색을 쓴다.
막대는 24px 이하, 데이터 쪽 끝만 4px 둥글게, 점은 지름 8px 이상에 흰 고리 2px.
"""
import numpy as np

import data as D

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt(v, d=3, sign=True):
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace("-", "−")


class Svg:
    def __init__(self, w, h, label):
        self.w, self.h, self.label = w, h, label
        self.parts = []

    def add(self, s):
        self.parts.append(s)

    def line(self, x1, y1, x2, y2, color=GRID, width=1):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"/>')

    def rect(self, x, y, w, h, fill, rx=0, stroke=None):
        st = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{st}/>')

    def text(self, x, y, s, size=11, anchor="start", fill=INK2, weight=400):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{esc(s)}</text>')

    def dot(self, x, y, color, shape="circle", r=4.5):
        """점: 흰 고리 2px. shape = circle | diamond."""
        if shape == "circle":
            self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" stroke="{SURF}" stroke-width="2"/>')
        else:
            d = r * 1.35
            self.add(f'<polygon points="{x:.1f},{y-d:.1f} {x+d:.1f},{y:.1f} {x:.1f},{y+d:.1f} {x-d:.1f},{y:.1f}" fill="{color}" stroke="{SURF}" stroke-width="2" stroke-linejoin="round"/>')

    def bar_h(self, x_zero, x_end, y, h, color, r=4):
        """가로 막대: 기준선(x_zero) 쪽은 직각, 데이터 쪽 끝만 둥글게."""
        if abs(x_end - x_zero) < 1.0:
            self.rect(min(x_zero, x_end), y, max(abs(x_end - x_zero), 1.5), h, color)
            return
        rr = min(r, abs(x_end - x_zero) / 2, h / 2)
        if x_end < x_zero:
            d = f"M{x_zero:.1f},{y:.1f} H{x_end + rr:.1f} Q{x_end:.1f},{y:.1f} {x_end:.1f},{y + rr:.1f} V{y + h - rr:.1f} Q{x_end:.1f},{y + h:.1f} {x_end + rr:.1f},{y + h:.1f} H{x_zero:.1f} Z"
        else:
            d = f"M{x_zero:.1f},{y:.1f} H{x_end - rr:.1f} Q{x_end:.1f},{y:.1f} {x_end:.1f},{y + rr:.1f} V{y + h - rr:.1f} Q{x_end:.1f},{y + h:.1f} {x_end - rr:.1f},{y + h:.1f} H{x_zero:.1f} Z"
        self.add(f'<path d="{d}" fill="{color}"/>')

    def render(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="100%" role="img" '
                f'aria-label="{esc(self.label)}" style="display:block;max-width:{self.w}px">' + "".join(self.parts) + "</svg>")


def tw(s, size):
    """글자 폭 어림 (한글은 글자 크기, 그 밖은 0.56배)."""
    return sum(size * (1.0 if ord(c) > 0x2E80 else 0.56) for c in s)


# ---------------------------------------------------------------------------------------------
def fig_strong_drops(data):
    """그림: 가장 센 단계에서 원본 대비 Macro-F1 변화, 필터 8종, val·test, 기준 모델 3개 평균(점 = 모델 하나)."""
    A = data["aws_b"]
    order = lambda shifts: sorted(shifts, key=lambda s: D.agg_strong(A["val"], s, True)[0])
    groups = [("장비 설정 5종 (학습에도 쓴 필터)", order(D.DEVICE)), ("제조사 후처리·전송 3종 (학습에는 쓰지 않은 필터)", order(D.POST))]
    W, left, x0, x1 = 680, 136, 150, 520          # 막대 영역 x0..x1, 오른쪽 숫자 열은 x1 뒤
    col_val, col_test = 590, 662                  # 숫자 열(오른쪽 끝 기준)
    lo, hi = -0.30, 0.04
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    row_h, grp_h, top, bottom = 42, 28, 44, 46
    n_rows = sum(len(g[1]) for g in groups)
    H = top + len(groups) * grp_h + n_rows * row_h + bottom
    s = Svg(W, H, "필터 8종의 가장 센 단계에서 원본 대비 Macro-F1 변화. val과 test 모두 기준 모델 3개의 평균과 모델별 점.")
    # 범례
    s.rect(4, 12, 12, 12, BLUE, rx=2); s.text(22, 22.5, "val 1,030장", 11.5, fill=INK)
    s.rect(112, 12, 12, 12, ORANGE, rx=2); s.text(130, 22.5, "test 618장", 11.5, fill=INK)
    s.dot(236, 18, INK2, r=3.8); s.text(246, 22.5, "= 모델 하나 (seed 42·43·44)", 11, fill=INK2)
    # 오른쪽 숫자 열 머리
    s.text(col_val, 22.5, "val 평균", 11, "end", MUTED)
    s.text(col_test, 22.5, "test 평균", 11, "end", MUTED)
    # 격자
    y_plot0 = top - 6
    y_plot1 = H - bottom + 4
    for v in (-0.30, -0.20, -0.10, 0.0):
        s.line(sx(v), y_plot0, sx(v), y_plot1, AXIS if v == 0 else GRID)
        s.text(sx(v), H - bottom + 20, fmt(v, 2, sign=False) if v == 0 else fmt(v, 2), 11, "middle", MUTED)
    s.text((x0 + x1) / 2, H - 8, "원본 대비 Macro-F1 변화 (왼쪽일수록 많이 떨어짐)", 11, "middle", MUTED)
    y = top
    for title, shifts in groups:
        s.text(0, y + 14, title, 11.5, fill=INK, weight=600)
        s.line(0, y + 21, W, y + 21, GRID)
        y += grp_h
        for sh in shifts:
            s.text(left, y + 26, D.NAME_KO[sh], 13, "end", INK)
            for k, (split, color, shape) in enumerate((("val", BLUE, "circle"), ("test", ORANGE, "diamond"))):
                mean, vals = D.agg_strong(A[split], sh, True)
                by = y + 7 + k * 15
                s.bar_h(sx(0), sx(mean), by, 11, color)
                for v in vals:
                    s.dot(sx(v), by + 5.5, INK2, "circle", r=3.8)
            m_val, _ = D.agg_strong(A["val"], sh, True)
            m_test, _ = D.agg_strong(A["test"], sh, True)
            s.text(col_val, y + 26, fmt(m_val), 12, "end", INK)
            s.text(col_test, y + 26, fmt(m_test), 12, "end", INK)
            y += row_h
    return s.render()


# ---------------------------------------------------------------------------------------------
def fig_scatter(raw):
    """그림: 같은 조건에서 val 변화(가로)와 test 변화(세로), 기준 모델 seed 42, 본편 24조건."""
    pts = D.val_test_changes(raw)
    W, H = 430, 400
    L, R, T, B = 58, 22, 16, 56
    lo, hi = -0.30, 0.06
    sx = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    sy = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    s = Svg(W, H, "본편 24개 조건에서 val의 점수 변화와 test의 점수 변화. 대부분 같은 방향으로 움직인다.")
    for v in (-0.30, -0.20, -0.10, 0.0):
        s.line(sx(v), T, sx(v), H - B, AXIS if v == 0 else GRID)
        s.line(L, sy(v), W - R, sy(v), AXIS if v == 0 else GRID)
        s.text(sx(v), H - B + 18, fmt(v, 2, sign=False) if v == 0 else fmt(v, 2), 10.5, "middle", MUTED)
        s.text(L - 8, sy(v) + 4, fmt(v, 2, sign=False) if v == 0 else fmt(v, 2), 10.5, "end", MUTED)
    s.line(sx(lo), sy(lo), sx(hi), sy(hi), "#b9b8b0", 1.2)       # y = x (val과 test가 같은 변화)
    s.text((L + W - R) / 2, H - 10, "val에서의 점수 변화 (원본 대비)", 11, "middle", MUTED)
    s.add(f'<text transform="translate(14,{(T + H - B) / 2:.0f}) rotate(-90)" font-size="11" text-anchor="middle" fill="{MUTED}">test에서의 점수 변화 (원본 대비)</text>')
    names = {}
    for sh, lv, dv, dt in pts:
        device = sh in D.DEVICE
        s.dot(sx(dv), sy(dt), BLUE if device else ORANGE, "circle" if device else "diamond", r=4.2)
        names[(sh, lv)] = (dv, dt)
    notes = [(("kvp", 3), "관전압 강", 9, 4, "start"), (("motion", 3), "노출시간 강", -9, 4, "end"),
             (("sensor", 3), "센서 강", -6, 17, "end"), (("sharpen", 1), "경계 강조 약", -9, 0, "end")]
    for key, label, dx, dy, anchor in notes:
        dv, dt = names[key]
        s.text(sx(dv) + dx, sy(dt) + dy, label, 10.5, anchor, INK)
    # 범례
    s.dot(L + 12, T + 14, BLUE, "circle", r=4); s.text(L + 22, T + 18, "장비 설정 5종", 10.5, fill=INK)
    s.dot(L + 112, T + 14, ORANGE, "diamond", r=4); s.text(L + 124, T + 18, "후처리·전송 3종", 10.5, fill=INK)
    s.line(L + 8, T + 34, L + 20, T + 34, "#b9b8b0", 1.2); s.text(L + 26, T + 38, "점이 이 선 위: val과 test의 변화가 같음", 10.5, fill=INK2)
    return s.render()


# ---------------------------------------------------------------------------------------------
def fig_base_vs_lung(data):
    """그림: 기준 모델 대 폐 영역 모델, 원본 점수와 필터 24조건 평균 점수 (val·test), 점 = 모델 하나."""
    A, Lg = data["aws_b"], data["lung"]
    W, H = 680, 306
    L, R, T, B = 52, 14, 46, 38
    lo, hi = 0.55, 0.90
    sy = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    clusters = [("val", "orig_f1", "val · 원본"), ("val", "main24", "val · 필터 24조건 평균"),
                ("test", "orig_f1", "test · 원본"), ("test", "main24", "test · 필터 24조건 평균")]
    cw = (W - L - R) / 4
    s = Svg(W, H, "기준 모델 3개와 폐 영역 모델 3개의 Macro-F1. val과 test, 원본 사진과 필터 24조건 평균.")
    for v in (0.6, 0.7, 0.8, 0.9):
        s.line(L, sy(v), W - R, sy(v), GRID)
        s.text(L - 8, sy(v) + 4, fmt(v, 1, sign=False), 11, "end", MUTED)
    s.line(L, H - B, W - R, H - B, AXIS)
    # 범례
    s.dot(L + 10, 18, BLUE, "circle"); s.text(L + 22, 22, "기준 모델 (AWS 학습 3개)", 11.5, fill=INK)
    s.dot(L + 204, 18, ORANGE, "diamond"); s.text(L + 218, 22, "폐 영역 모델 (3개)", 11.5, fill=INK)
    s.line(L + 342, 18, L + 358, 18, INK2, 2.5); s.text(L + 364, 22, "= 평균", 11, fill=INK2)
    for i, (split, key, title) in enumerate(clusters):
        cx = L + cw * i + cw / 2
        if i % 2 == 0 and i > 0:
            s.line(L + cw * i, T - 6, L + cw * i, H - B + 6, GRID)
        for j, (fam, color, shape) in enumerate((("aws_b", BLUE, "circle"), ("lung", ORANGE, "diamond"))):
            mean, sd, vals = D.agg(data[fam][split], key)
            x = cx + (-1 if j == 0 else 1) * cw * 0.2
            for k, v in enumerate(vals):
                s.dot(x + (k - 1) * 7, sy(v), color, shape, r=4.2)
            s.line(x - 16, sy(mean), x + 16, sy(mean), INK2, 2.5)
            s.text(x, sy(max(vals)) - 12, fmt(mean, 3, sign=False), 11.5, "middle", INK, weight=600)
        s.text(cx, H - B + 20, title, 11.5, "middle", INK)
    return s.render()


# ---------------------------------------------------------------------------------------------
def fig_aug(data):
    """그림: 장비 효과 증강 전 → 후, 가장 센 단계의 Macro-F1. 위: 기준 모델, 아래: 폐 영역 모델. 왼쪽 val, 오른쪽 test.
    구분선 위 5개는 학습에 섞은 필터, 아래 3개는 학습에 쓰지 않은 필터다."""
    pairs = [("기준 모델 → 장비 효과 증강 모델", "jl_b", "jl_v"), ("폐 영역 모델 → 폐 영역 + 장비 효과 증강 모델", "lung", "lung_aug")]
    shifts = D.DEVICE + D.POST
    W = 680
    panels = {"val": (124, 350, 0.56, 0.86, (0.6, 0.7, 0.8)), "test": (428, 654, 0.36, 0.80, (0.4, 0.5, 0.6, 0.7))}
    row_h, gap, top, ph = 20, 7, 38, 238
    s = Svg(W, top + 2 * ph, "장비 효과 증강 전과 후의 Macro-F1, 필터 8종의 가장 센 단계. 기준 모델과 폐 영역 모델, val과 test.")
    s.dot(4, 14, BLUE, "circle"); s.text(16, 18, "증강 전 (모델 3개 평균)", 11.5, fill=INK)
    s.dot(182, 14, ORANGE, "diamond"); s.text(196, 18, "증강 후 (모델 3개 평균)", 11.5, fill=INK)
    s.text(W, 18, "구분선 위 5개 = 학습에 섞은 필터", 11, "end", MUTED)
    for r, (title, fb, fa) in enumerate(pairs):
        y0 = top + r * ph
        s.text(0, y0 + 14, title, 12, fill=INK, weight=600)
        for split, (xl, xr, lo, hi, ticks) in panels.items():
            sx = lambda v, xl=xl, xr=xr, lo=lo, hi=hi: xl + (v - lo) / (hi - lo) * (xr - xl)
            s.text((xl + xr) / 2, y0 + 36, split, 11.5, "middle", INK, weight=600)
            ytop = y0 + 46
            ybot = ytop + 8 * row_h + gap
            for v in ticks:
                s.line(sx(v), ytop - 4, sx(v), ybot, GRID)
                s.text(sx(v), ybot + 16, fmt(v, 1, sign=False), 10.5, "middle", MUTED)
            for i, sh in enumerate(shifts):
                yy = ytop + i * row_h + (gap if i >= 5 else 0)
                if split == "val":
                    s.text(xl - 10, yy + 11, D.NAME_KO[sh], 11.5, "end", INK)
                mb, _ = D.agg_strong(data[fb][split], sh)
                ma, _ = D.agg_strong(data[fa][split], sh)
                s.line(sx(mb), yy + 7, sx(ma), yy + 7, "#b9b8b0", 2)
                s.dot(sx(mb), yy + 7, BLUE, "circle", r=4)
                s.dot(sx(ma), yy + 7, ORANGE, "diamond", r=4)
            ysep = ytop + 5 * row_h + gap / 2 - 1
            s.line(0 if split == "val" else xl - 8, ysep, xr + 4, ysep, AXIS)
    return s.render()


# ---------------------------------------------------------------------------------------------
def fig_early(early):
    """그림: 초기 실험(입력 224) 8개의 val 점수와 test 점수. 모두 test가 낮고 순위도 같지 않다."""
    rows = sorted(early, key=lambda r: -r["val"])
    W, left, x0, x1 = 680, 188, 202, 560
    lo, hi = 0.64, 0.86
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    row_h, top, bottom = 30, 34, 44
    H = top + len(rows) * row_h + bottom
    s = Svg(W, H, "초기 실험 8개의 val 점수와 test 점수. 모든 모델이 test에서 낮고 val 순위가 test 순위와 다르다.")
    s.dot(left + 8, 14, BLUE, "circle"); s.text(left + 20, 18, "val (1,030장)", 11.5, fill=INK)
    s.dot(left + 130, 14, ORANGE, "diamond"); s.text(left + 144, 18, "test (618장)", 11.5, fill=INK)
    s.text(W - 8, 18, "test − val", 11, "end", MUTED)
    for v in (0.65, 0.70, 0.75, 0.80, 0.85):
        s.line(sx(v), top - 6, sx(v), top + len(rows) * row_h, GRID)
        s.text(sx(v), top + len(rows) * row_h + 18, fmt(v, 2, sign=False), 10.5, "middle", MUTED)
    s.text((x0 + x1) / 2, H - 6, "Macro-F1", 11, "middle", MUTED)
    for i, r in enumerate(rows):
        y = top + i * row_h + 12
        s.text(left, y + 4, r["name"], 12, "end", INK)
        s.line(sx(r["val"]), y, sx(r["test"]), y, "#b9b8b0", 2)
        s.dot(sx(r["val"]), y, BLUE, "circle", r=4.2)
        s.dot(sx(r["test"]), y, ORANGE, "diamond", r=4.2)
        s.text(W - 8, y + 4, fmt(r["test"] - r["val"], 3), 11.5, "end", INK)
    return s.render()


# ---------------------------------------------------------------------------------------------
def fig_arch():
    """그림: AWS 구성. 위: 학습 경로, 아래: 서비스 경로 (발표 뒤에 모두 삭제)."""
    W, H = 680, 262
    s = Svg(W, H, "AWS 구성도. 학습 경로는 S3에서 SageMaker 학습 작업을 거쳐 S3에 모델을 저장한다. 서비스 경로는 브라우저에서 EC2, Lambda, SageMaker Endpoint 순서로 호출한다.")

    def box(x, y, w, h, title, sub, accent=False):
        s.rect(x, y, w, h, "#fcfcfb", rx=6, stroke=ORANGE if accent else AXIS)
        s.text(x + w / 2, y + 22, title, 12.5, "middle", INK, weight=600)
        for i, line in enumerate(sub):
            s.text(x + w / 2, y + 40 + i * 15, line, 10.5, "middle", INK2)

    def arrow(x1, x2, y):
        s.line(x1, y, x2 - 5, y, MUTED, 1.5)
        s.add(f'<polygon points="{x2:.1f},{y:.1f} {x2 - 7:.1f},{y - 4:.1f} {x2 - 7:.1f},{y + 4:.1f}" fill="{MUTED}"/>')

    s.text(0, 14, "학습 경로 (iMac에서 제출)", 11.5, fill=MUTED, weight=600)
    bw, bh, gap = 144, 70, 34
    xs = [0 + i * (bw + gap) for i in range(4)]
    box(xs[0], 24, bw, bh, "iMac", ["코드·분할 CSV", "작업 제출 (SDK)"])
    box(xs[1], 24, bw, bh, "Amazon S3", ["원천 5,856장", "매니페스트·가중치"])
    box(xs[2], 24, bw, bh, "SageMaker 학습 작업", ["ml.g4dn.xlarge / 2xlarge", "PyTorch 2.2 컨테이너"])
    box(xs[3], 24, bw + 2, bh, "Amazon S3", ["model.tar.gz", "history.csv·평가 CSV"])
    for i in range(3):
        arrow(xs[i] + bw, xs[i + 1], 59)
    s.text(0, 126, "서비스 경로 (10/23 배포, 발표 직후 삭제)", 11.5, fill=MUTED, weight=600)
    bw2, gap2 = 112, 26
    xs2 = [i * (bw2 + gap2) for i in range(4)]
    box(xs2[0], 136, bw2, 74, "브라우저", ["JPEG·PNG 한 장", "(2 MiB 이하)"])
    box(xs2[1], 136, bw2, 74, "EC2 · Flask", ["t3.small", "포트 8000·8001"])
    box(xs2[2], 136, bw2, 74, "Lambda", ["Python 3.12", "화면 ↔ 모델 중계"])
    box(xs2[3], 136, bw2 + 10, 74, "SageMaker Endpoint", ["ml.m5.large (CPU)", "기준 모델 B512"], accent=True)
    arrow(xs2[3] + bw2 + 10, xs2[3] + bw2 + 10 + gap2, 173)
    box(xs2[3] + bw2 + 10 + gap2, 136, 100, 74, "출력", ["확률 3개", "(합 100%)"])
    for i in range(3):
        arrow(xs2[i] + bw2, xs2[i + 1], 173)
    s.text(0, 232, "보안 그룹은 내 공인 IP(/32)만 허용 · 역할은 인라인 최소 권한 (Lambda: 본인 Endpoint 호출, 웹: 본인 함수 호출)", 10.5, fill=INK2)
    s.text(0, 250, "Endpoint가 돌려주는 것은 확률 3개뿐이다 (Grad-CAM·설명문은 배포 경로에 없음).", 10.5, fill=INK2)
    return s.render()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    d, raw = D.load_all()
    D.check(d)
    early = [
        {"name": "A 몸통 고정 (fc만)", "val": 0.806, "test": 0.748}, {"name": "B 기준 (layer4)", "val": 0.818, "test": 0.719},
        {"name": "C 클래스 가중치", "val": 0.821, "test": 0.690}, {"name": "D 노이즈 증강", "val": 0.834, "test": 0.692},
        {"name": "P1 히스토그램 평활화", "val": 0.822, "test": 0.700}, {"name": "P2 CLAHE", "val": 0.831, "test": 0.684},
        {"name": "P3 가우시안 블러", "val": 0.818, "test": 0.733}, {"name": "P4 샤프닝", "val": 0.829, "test": 0.757},
    ]
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/charts_test.html")
    html = ["<!doctype html><meta charset='utf-8'><body style=\"font-family:'Apple SD Gothic Neo',sans-serif;width:680px;margin:20px auto\">"]
    for name, svg in [("strong", fig_strong_drops(d)), ("scatter", fig_scatter(raw)), ("base_vs_lung", fig_base_vs_lung(d)),
                      ("aug", fig_aug(d)), ("early", fig_early(early)), ("arch", fig_arch())]:
        html.append(f"<h4>{name}</h4>{svg}<hr>")
    out.write_text("".join(html), encoding="utf-8")
    print(out)
