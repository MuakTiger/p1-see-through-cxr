"""과적합 줄이기 실험(r0~r4 × seed 3개 = 15개)의 뒷일을 자동으로 이어 간다: 끝난 학습 결과 내려받기 → 노트북 GPU 로 val 필터 시험.

60초마다 작업 상태를 보고, 끝난 모델부터 src/equipment_eval_direct.py 로 평가한다 (동시에 2개, --device mps).
평가 결과는 reports/equipment_v2/reg/<실험ID-시각>/ 에 저장한다. 15개가 모두 평가되거나 실패가 나면 끝난다.
test 는 열지 않는다 (val 만).

사용 예 (p1 폴더에서):
    .venv/bin/python src/reg_pipeline.py
"""
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fetch_results import fetch  # noqa: E402

PATTERN = r"^r[0-4]-"
EXPECTED = 15
WORKERS = 2
OUT = ROOT / "reports/equipment_v2/reg"


def result_dirs():
    return sorted(p for p in (ROOT / "runs/results").iterdir() if re.match(r"r[0-4]-", p.name) and (p / "model/best_model.pt").is_file())


def start_eval(d):
    out = OUT / d.name
    log = open(OUT / f"{d.name}.log", "w", encoding="utf-8")
    cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "src/equipment_eval_direct.py"), "--model-dir", str(d / "model"),
           "--out", str(out), "--label", d.name, "--device", "mps", "--plot", "0"]
    return subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), d.name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    running, done, failed = [], set(), []
    while True:
        status = fetch(PATTERN, quiet=True)
        bad = {k: v for k, v in status.items() if v in ("Failed", "Stopped")}
        for k, v in bad.items():
            if k not in failed:
                failed.append(k)
                print(f"[{time.strftime('%H:%M:%S')}] 학습 실패: {k} ({v})", flush=True)
        for p, name in list(running):
            if p.poll() is not None:
                running.remove((p, name))
                (done if p.returncode == 0 else set()).add(name)
                if p.returncode != 0:
                    failed.append(name)
                print(f"[{time.strftime('%H:%M:%S')}] 평가 {'끝' if p.returncode == 0 else '실패'}: {name}", flush=True)
        queued = {n for _, n in running} | done | set(failed)
        for d in result_dirs():
            if len(running) >= WORKERS:
                break
            if d.name not in queued and not (OUT / d.name / "shift_report.md").is_file():
                running.append(start_eval(d))
                queued.add(d.name)
                print(f"[{time.strftime('%H:%M:%S')}] 평가 시작: {d.name}", flush=True)
        n_have = sum(v in ("have", "downloaded") for v in status.values())
        n_eval = len(list(OUT.glob("*/shift_report.md")))
        print(f"[{time.strftime('%H:%M:%S')}] 학습 끝 {n_have}/{EXPECTED}, 평가 끝 {n_eval}/{EXPECTED}, 평가 중 {len(running)}, 실패 {len(failed)}", flush=True)
        if n_eval >= EXPECTED or (failed and not running and n_have + len(failed) >= EXPECTED and n_eval + len(failed) >= EXPECTED):
            break
        time.sleep(60)
    print("끝", flush=True)


if __name__ == "__main__":
    main()
