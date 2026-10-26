"""결과보고서(HTML → PDF)를 만든다. 숫자는 저장된 결과 파일에서 읽는다.

사용 예 (p1 폴더에서):
    .venv/bin/python tools/final_report/build.py            # docs/final_report/report.html 과 PDF 를 만든다
    python tools/final_report/pdf_post.py                     # (선택, pymupdf 필요) 목차 쪽 번호와 PDF 책갈피를 넣는다. 쪽 번호가 바뀌면 build.py 를 한 번 더 돌린다.
PDF 는 Chrome(headless)으로 인쇄한다.
"""
import csv
import glob
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import charts as C  # noqa: E402
import content_a as A  # noqa: E402
import content_b as B  # noqa: E402
import data as D  # noqa: E402

ROOT = D.ROOT
EQ = D.EQ
OUT = ROOT / "docs/final_report"
PDF_NAME = "P1_형준팀_프로젝트 결과보고서.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

TOC = [
    ("sec1", "1", "프로젝트 개요"), ("sec2", "2", "기획안 대비 변경 사항"), ("sec3", "3", "데이터와 분할"), ("sec4", "4", "방법"),
    ("sec5", "5", "결과"), ("sec6", "6", "점수를 믿을 수 있는가: 진단"), ("sec7", "7", "AWS 구성, 검산, 종료 기록"), ("sec8", "8", "한계와 다음 단계"),
    ("secA", "부록 A", "필터 33조건 전체 결과"), ("secB", "부록 B", "초기 실험 (입력 224)"), ("secC", "부록 C", "가이드 필수 제출물 대응표"),
    ("secD", "부록 D", "모델 S3 주소"), ("secE", "부록 E", "제출 패키지 구성과 재현 방법"), ("secF", "부록 F", "용어"),
]


def sha(path):
    return A.sha(path)


def load_exp():
    """실험 결과 폴더에서 선택 에폭·val F1·학습 시간을 읽는다."""
    res = ROOT / "runs/results" if (ROOT / "runs/results").exists() else ROOT / "results"   # 제출 묶음에서는 results/
    ex = {}
    keys = ["b512", "b512-s43", "b512-s44", "l-s42", "l-s43", "l-s44", "lv-s42", "lv-s43", "lv-s44",
            "b-s42", "b-s43", "b-s44", "v-s42", "v-s43", "v-s44", "a", "b", "c", "d", "d-s43", "d-s44", "p1", "p2", "p3", "p4", "b448"]
    for k in keys:
        cands = [p for p in glob.glob(str(res / f"{k}-2026*")) if Path(p).name.split("-2026")[0] == k]
        assert len(cands) == 1, (k, cands)
        folder = Path(cands[0])
        s = json.load(open(folder / "experiment_summary.json"))
        name = folder.name
        if name.endswith("-jlab"):
            addr = f"aug_runs/results_{k}.tgz"
        else:
            addr = f"p1-cxr-&lt;학생ID&gt;-{name}/output/model.tar.gz"
        ex[k] = {"epoch": s["selected_epoch"], "valf1": s["best_val_macro_f1"], "sec": s["total_training_seconds"], "addr": addr, "dir": folder}
    return ex


def load_lungfocus():
    rows = list(csv.DictReader(open(EQ / "gradcam/lung_focus_val.csv", encoding="utf-8")))
    models = sorted({r["model"] for r in rows})

    def pick(prefix):
        m = [x for x in models if x.startswith(prefix)]
        assert len(m) == 1, (prefix, models)
        return [r for r in rows if r["model"] == m[0]]

    def mean(rs, key):
        return sum(float(r[key]) for r in rs) / len(rs)

    out = {}
    for tag, prefix in (("b512", "B512"), ("b_s42", "b-s42"), ("v_s42", "v-s42")):
        rs = pick(prefix)
        out[tag] = mean(rs, "cam_inside_lung")
        pk = 100 * mean(rs, "peak_in_lung")
        npk = 100 * mean([r for r in rs if r["true"] == "0"], "peak_in_lung")
        key = {"b512": "b512", "b_s42": "b", "v_s42": "v"}[tag]
        out[f"peak_{key}"] = pk
        out[f"normal_peak_{key}"] = npk
        if tag == "b512":
            ok = [r for r in rs if r["true"] == r["pred"]]
            bad = [r for r in rs if r["true"] != r["pred"]]
            out["right"], out["wrong"] = mean(ok, "cam_inside_lung"), mean(bad, "cam_inside_lung")
            out["area"] = mean(rs, "lung_area")
    return out


def load_gradcam():
    t = Path(EQ / "gradcam/B512/gradcam_checks_mean.md").read_text(encoding="utf-8")
    corr = re.search(r"상관: 평균 ([0-9.]+)", t).group(1)
    maxd = re.search(r"최대 절대 차이 ([0-9.e+-]+)", t).group(1)
    vals = re.findall(r"\| 지도 점수 (?:높은|무작위|낮은) 칸 \| ([+\-0-9.]+) \|", t)
    assert len(vals) == 3, vals
    hi, rd, lo = (float(v.replace("−", "-")) for v in vals)
    return {"corr": corr, "maxdiff": maxd, "hi": hi, "rand": rd, "lo": lo}


def load_aspect():
    tabs = B.md_tables(EQ / "aspect_check/aspect_report.md")
    rows = {r[0]: r for r in tabs[0][1:]}
    g = lambda k: rows[k][2].split()[0]
    return {"base": g("원본"), "f905": g("가로 0.905배"), "f122": g("가로 1.22배")}


def load_jpegresave():
    rows = B.md_tables(ROOT / "reports/shift_test_B448/jpeg_resave_check.md")[0][1:]
    v = [int(r[5]) for r in rows if "재저장" in r[1]]
    return {"min": min(v), "max": max(v)}


def load_early():
    t1 = B.md_tables(ROOT / "reports/model_comparison.md")[0][1:]
    sc = {r[0]: r for r in t1}
    names = [("A 몸통 고정", "A 몸통 고정 (fc만)"), ("B", "B 기준 (layer4)"), ("C 가중치", "C 클래스 가중치"), ("D", "D 노이즈 증강"),
             ("P1 HE", "P1 히스토그램 평활화"), ("P2 CLAHE", "P2 CLAHE"), ("P3 블러", "P3 가우시안 블러"), ("P4 샤프닝", "P4 샤프닝")]
    out = []
    for key, label in names:
        r = sc[key]
        out.append({"name": label, "val": float(r[2]), "test": float(r[4].split()[0])})
    return out


def load_normal_shift():
    d = json.load(open(ROOT / "reports/test_eval/normal_shift/normal_shift_summary.json"))
    lk = d["lookalike"]
    return {
        "jpeg_trn": lk["jpeg_quality"]["train 정상"], "jpeg_pn": lk["jpeg_quality"]["train 세균성(표본)"], "jpeg_tn": lk["jpeg_quality"]["test 정상"],
        "ps_trn": lk["has_photoshop"]["train 정상"], "ps_pn_b": lk["has_photoshop"]["train 세균성(표본)"], "ps_pn_v": lk["has_photoshop"]["train 바이러스성(표본)"], "ps_tn": lk["has_photoshop"]["test 정상"],
        "asp_trn": lk["aspect"]["train 정상"], "asp_b": lk["aspect"]["train 세균성(표본)"], "asp_v": lk["aspect"]["train 바이러스성(표본)"], "asp_tn": lk["aspect"]["test 정상"],
        "attr_auroc_test": d["label_from_attributes"]["a_file"]["test"],
    }


def load_lungbox():
    t = Path(EQ / "lung_box_composition.md").read_text(encoding="utf-8")
    rows = {r[0]: r for r in B.md_tables(EQ / "lung_box_composition.md")[0][1:]}
    ter = {r[0]: r for r in B.md_tables(EQ / "lung_box_composition.md")[1][1:]}
    rho = float(re.search(r"세로 점유 ([+\-−0-9.]+) \(p=", t).group(1).replace("−", "-"))
    return {"v_norm_tv": float(rows["train·val 정상"][2]), "v_pneu_tv": float(rows["train·val 폐렴"][2]), "v_norm_test": float(rows["test 정상"][2]),
            "t1": float(ter["낮음"][3]), "t2": float(ter["중간"][3]), "t3": float(ter["높음"][3]), "rho": rho}


def load_aws(path):
    text = Path(path).read_text(encoding="utf-8")
    tabs = B.md_tables(path)
    res, jobs, hours = tabs[0][1:], tabs[1][1:], tabs[2][1:]
    stat = []
    for i in range(0, len(res), 2):
        a = res[i]
        b = res[i + 1] if i + 1 < len(res) else ["", "", ""]
        stat.append([a[0], a[1], b[0], b[1]])
    job_rows = [[r[0], r[1], r[3], r[4]] for r in jobs]
    h = {r[0]: float(r[1]) for r in hours}
    total = sum(int(r[1]) for r in jobs)
    t = re.search(r"조회 시각: (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})", text)
    return {"stat_rows": stat, "job_rows": job_rows, "total_jobs": total, "total_hours": h["합계"], "h_2xl": h["ml.g4dn.2xlarge"], "h_xl": h["ml.g4dn.xlarge"], "time_now": t.group(2)}


def build_ctx():
    data, raw = D.load_all()
    D.check(data)
    ex = load_exp()
    dep, loc = ROOT / "deploy/b512/best_model.pt", ex["b512"]["dir"] / "model/best_model.pt"
    sha_file = ROOT / "reports/deploy_weights_sha256.txt"
    if dep.exists() and loc.exists():
        dep_sha, loc_sha = sha(dep), sha(loc)
        assert dep_sha == loc_sha, "배포 가중치와 학습 결과 가중치가 다르다"
        sha_file.write_text(f"{dep_sha}  best_model.pt\n# 배포한 가중치(deploy/b512/best_model.pt)와 학습 결과(b512 seed 42, 선택 에폭 18)의 가중치가 같은 파일임을 확인한 sha256.\n# 가중치 파일은 용량 때문에 제출물에 넣지 않았다 (S3 주소는 결과보고서 부록 D).\n", encoding="utf-8")
    else:   # 제출 묶음에는 가중치가 없다: 미리 기록해 둔 해시를 읽는다
        dep_sha = re.search(r"([0-9a-f]{64})", sha_file.read_text(encoding="utf-8")).group(1)
    models = [("b512", "기준 B512 (seed 42)", "학습 작업"), ("b512-s43", "기준 B512 (seed 43)", "학습 작업"), ("b512-s44", "기준 B512 (seed 44)", "학습 작업"),
              ("l-s42", "폐 영역 l (seed 42)", "학습 작업"), ("l-s43", "폐 영역 l (seed 43)", "학습 작업"), ("l-s44", "폐 영역 l (seed 44)", "학습 작업"),
              ("lv-s42", "폐 영역 + 증강 lv (seed 42)", "학습 작업"), ("lv-s43", "폐 영역 + 증강 lv (seed 43)", "학습 작업"), ("lv-s44", "폐 영역 + 증강 lv (seed 44)", "학습 작업"),
              ("b-s42", "기준 b (seed 42)", "JupyterLab"), ("b-s43", "기준 b (seed 43)", "JupyterLab"), ("b-s44", "기준 b (seed 44)", "JupyterLab"),
              ("v-s42", "증강 v (seed 42)", "JupyterLab"), ("v-s43", "증강 v (seed 43)", "JupyterLab"), ("v-s44", "증강 v (seed 44)", "JupyterLab"),
              ("a", "초기 A (224, seed 42)", "학습 작업"), ("b", "초기 B (224, seed 42)", "학습 작업"), ("c", "초기 C (224, seed 42)", "학습 작업"),
              ("d", "초기 D (224, seed 42)", "학습 작업"), ("d-s43", "초기 D (224, seed 43)", "학습 작업"), ("d-s44", "초기 D (224, seed 44)", "학습 작업"),
              ("p1", "초기 P1 HE (224)", "학습 작업"), ("p2", "초기 P2 CLAHE (224)", "학습 작업"), ("p3", "초기 P3 블러 (224)", "학습 작업"),
              ("p4", "초기 P4 샤프닝 (224)", "학습 작업"), ("b448", "B448 (입력 448)", "학습 작업")]
    aws_file = sorted((ROOT / "reports").glob("aws_teardown_check_2026*.md"))[-1]
    return {
        "data": data, "exp": ex, "models": models, "verify": {"sha": dep_sha}, "lungfocus": load_lungfocus(), "gradcam": load_gradcam(),
        "aspect": load_aspect(), "jpegresave": load_jpegresave(), "early": load_early(), "normal_shift": load_normal_shift(), "lungbox": load_lungbox(),
        "split_sha": sha(ROOT / "manifests/split_manifest.csv"), "aws": load_aws(aws_file),
        "repo_url": (HERE / "repo_url.txt").read_text(encoding="utf-8").strip() if (HERE / "repo_url.txt").exists() else "",
    }, raw


def toc_html(pages):
    items = []
    for sid, no, title in TOC:
        pg = pages.get(sid, "")
        items.append(f'<div><a href="#{sid}">{no}&nbsp;&nbsp;{title}</a><span class="pg">{pg}</span></div>')
    return '<div class="toc">' + "".join(items) + "</div>"


def build_html(ctx, raw, pages):
    body = A.cover(ctx, toc_html(pages)) + A.sec1() + A.sec2(ctx) + A.sec3(ctx) + B.sec4(ctx) + B.sec5(ctx, raw) + B.sec6(ctx) + B.sec7(ctx) + B.sec8() + B.appendix(ctx, raw)
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>P1 See-Through CXR 결과보고서</title>
<meta name="author" content="오형준">
<link rel="stylesheet" href="style.css"></head><body>{body}</body></html>"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "assets").mkdir(exist_ok=True)
    shutil.copy(HERE / "style.css", OUT / "style.css")
    for fn in ("original", "kvp", "dose", "motion", "scatter", "sensor", "sharpen", "equalize", "jpeg"):
        shutil.copy(EQ / f"tiles_slide/{fn}.png", OUT / f"assets/{fn}.png")
    shutil.copy(EQ / "lung_model/gradcam_compare_slide.png", OUT / "assets/gradcam_compare_slide.png")
    pages_file = OUT / "toc_pages.json"
    pages = json.load(open(pages_file)) if pages_file.exists() else {}
    ctx, raw = build_ctx()
    html = build_html(ctx, raw, pages)
    (OUT / "report.html").write_text(html, encoding="utf-8")
    pdf = OUT / PDF_NAME
    cmd = [CHROME, "--headless=new", "--disable-gpu", "--allow-file-access-from-files", "--no-pdf-header-footer",
           "--virtual-time-budget=20000", f"--print-to-pdf={pdf}", f"file://{OUT / 'report.html'}"]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print("HTML:", OUT / "report.html")
    print("PDF :", pdf, f"({pdf.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
