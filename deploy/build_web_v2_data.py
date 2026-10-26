"""웹 앱(deploy/web_v2)이 보여 줄 '이 필터를 걸었을 때 B512 점수 변화' 데이터를 만든다.

저장된 시험 결과 파일(reports/equipment_v2/val_B512, test_B512 의 shift_results.csv)을 읽기만 하고 모델은 돌리지 않는다.
출력: deploy/web_v2/data/effects.json   {split: {필터: {단계(0~3): {...}}}}
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy/web_v2/data/effects.json"
SRC = {"val": ROOT / "reports/equipment_v2/val_B512/shift_results.csv",
       "test": ROOT / "reports/equipment_v2/test_B512/shift_results.csv"}
N = {"val": 1030, "test": 618}


def main():
    out = {}
    for split, path in SRC.items():
        d = pd.read_csv(path)
        res = {}
        for r in d.itertuples():
            name = "clean" if r.shift == "clean" else r.shift
            lv = int(r.level)
            sig = bool(lv and (r.delta_f1_ci_high < 0 or r.delta_f1_ci_low > 0))
            res.setdefault(name, {})[str(lv)] = {
                "macro_f1": round(float(r.macro_f1), 3), "delta_f1": round(float(r.delta_f1), 3),
                "ci": [round(float(r.delta_f1_ci_low), 3), round(float(r.delta_f1_ci_high), 3)], "significant": sig,
                "recall": [round(float(r.recall_normal), 3), round(float(r.recall_bacterial), 3), round(float(r.recall_viral), 3)],
                "share": [round(float(r.pred_share_normal), 3), round(float(r.pred_share_bacterial), 3), round(float(r.pred_share_viral), 3)],
                "changed_pct": round(float(r.changed_pred_pct), 1)}
        out[split] = {"n": N[split], "rows": res}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장:", OUT, {k: len(v["rows"]) for k, v in out.items()})


if __name__ == "__main__":
    main()
