"""저장소 첫 화면(README)에 넣을 그림을 PNG로 만든다.

보고서와 같은 SVG 차트(charts.py)를 Chrome(headless)으로 2배 해상도로 캡처한다. 숫자는 저장된 결과 파일에서 읽는다.
결과: docs/figures/*.png

사용 예 (p1 폴더에서):
    .venv/bin/python tools/final_report/export_figures.py
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import charts as C  # noqa: E402
import data as D  # noqa: E402

ROOT = D.ROOT
OUT = ROOT / "docs/figures"
TMP = OUT / "_tmp"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FONT = "'Apple SD Gothic Neo', 'Noto Sans KR', 'Nanum Gothic', sans-serif"


def shot(body_html, name, width, height):
    TMP.mkdir(parents=True, exist_ok=True)
    page = TMP / f"{name}.html"
    page.write_text(f"<!doctype html><meta charset='utf-8'><body style=\"margin:0;padding:12px;background:#fff;font-family:{FONT};width:{width}px\">{body_html}</body>",
                    encoding="utf-8")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files", f"--window-size={width + 24},{height + 24}",
                    "--force-device-scale-factor=2", f"--screenshot={OUT / (name + '.png')}", f"file://{page}"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def size_of(svg):
    m = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    return int(m.group(1)), int(m.group(2))


def tiles_html():
    tdir = ROOT / "reports/equipment_v2/tiles_slide"
    items = [("original", "원본"), ("kvp", "관전압 ↑"), ("dose", "관전류↓ (선량)"), ("motion", "노출시간 ↑"), ("scatter", "산란"),
             ("sensor", "센서 해상도"), ("sharpen", "경계 강조"), ("equalize", "대비 보정"), ("jpeg", "JPEG 압축")]
    cells = "".join(
        f'<div style="text-align:center;font-size:15px;color:#0b0b0b"><img src="file://{tdir / (fn + ".png")}" style="width:100%;border:1px solid #e1e0d9;display:block;margin-bottom:5px"><b>{label}</b></div>'
        for fn, label in items)
    cells += '<div style="font-size:13px;color:#52514e;line-height:1.5;padding-top:6px">가장 센 단계<br>관전압 (0.60, 0.5) · 선량 1/64 · 번짐 8 mm · 산란 k=1.0 · 센서 2.0 mm · 경계 강조 α=2 · HE · JPEG 20</div>'
    return f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px">{cells}</div>'


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    d, raw = D.load_all()
    D.check(d)
    jobs = [("fig1_filters", None), ("fig2_strong_drops", C.fig_strong_drops(d)), ("fig3_base_vs_lung", C.fig_base_vs_lung(d)),
            ("fig4_augmentation", C.fig_aug(d)), ("fig5_aws", C.fig_arch())]
    for name, svg in jobs:
        if svg is None:
            shot(tiles_html(), name, 900, 420)
        else:
            w, h = size_of(svg)
            shot(svg, name, w, h)
        print("저장:", OUT / f"{name}.png")
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
