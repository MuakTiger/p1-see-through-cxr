"""split_manifest.csv 생성 스크립트 (분할 단계).

manifest.csv 의 split 열을 채워서 학습/검증/점검/시험 용도를 정한다.
열: manifest.csv 의 모든 열 + split_group (마지막 열)

분할 규칙:
    test  624장  원래 test 그대로 (절대 건드리지 않음)
    check  16장  원래 val 그대로 (화면 점검용, 학습/검증에 쓰지 않음)
    train/val    원래 train 5,216장만 환자 그룹 단위 + 클래스 층화 80:20 (seed 42)

사용 예 (iMac 터미널, p1 폴더에서):
    .venv/bin/python src/split_manifest.py --manifest manifests/manifest.csv --out manifests --seed 42

결과:
    split_manifest.csv   split 열이 채워진 목록표 (행 순서는 manifest.csv 와 같음)
    split_summary.csv    split × 라벨별 영상 수·그룹 수·비율
    split_report.md      규칙, 결과 표, 검증 체크리스트, 걸친 그룹 목록, 한계
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

LABEL_NAMES = {0: "NORMAL", 1: "BACTERIAL_PNEUMONIA", 2: "VIRAL_PNEUMONIA"}
SPLIT_ORDER = ("train", "val", "check", "test")
N_SPLITS = 5  # 5등분 중 첫 번째 조각을 val 로 쓴다 (약 20%)
EXPECTED_SOURCE = {"train": 5216, "val": 16, "test": 624}
EXPECTED_TOTAL = 5856
RATIO_LOW, RATIO_HIGH = 0.18, 0.22  # 클래스별 val 비율 허용 범위 (점검용)


class UnionFind:
    """서로 연결된 것들을 하나의 묶음으로 합치는 도구 (같은 묶음 = 같은 대표값)."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # 사전순으로 앞선 쪽을 대표로 삼아서 실행할 때마다 결과가 같게 한다
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    need = {"relative_path", "label_id", "label_name", "source_split", "split", "patient_group", "sha256"}
    missing = need - set(fieldnames or [])
    if missing:
        raise SystemExit(f"manifest.csv 에 필요한 열이 없습니다: {sorted(missing)}")
    return fieldnames, rows


def check_input(rows):
    """입력 목록표가 예상한 모양인지 먼저 확인한다."""
    if len(rows) != EXPECTED_TOTAL:
        raise SystemExit(f"manifest.csv 행 수가 {len(rows)}입니다. 가이드 기준은 {EXPECTED_TOTAL}입니다.")
    counts = Counter(r["source_split"] for r in rows)
    if dict(counts) != EXPECTED_SOURCE:
        raise SystemExit(f"원래 분할별 장수가 다릅니다: {dict(counts)} (기대값 {EXPECTED_SOURCE})")
    if any(r["split"] for r in rows):
        raise SystemExit("manifest.csv 의 split 열이 이미 채워져 있습니다. 비어 있어야 합니다.")
    if any(r["patient_group"].startswith("unknown:") for r in rows):
        raise SystemExit("규칙 밖(unknown) 환자 그룹이 있습니다. make_manifest.py 결과를 먼저 확인하세요.")


def build_split_groups(pool_rows):
    """풀 안에서 patient_group 과 sha256 중복을 함께 묶어 split_group 을 만든다.

    비유: 같은 사람 표시(patient_group)가 있거나 똑같은 사진(sha256)이 있으면
    한 상자에 같이 담는다. 상자 단위로 나누면 같은 사람·같은 사진이 갈라지지 않는다.
    """
    uf = UnionFind()
    by_hash = defaultdict(list)
    for r in pool_rows:
        uf.find(r["patient_group"])
        by_hash[r["sha256"]].append(r["patient_group"])
    merged_by_hash = 0  # sha256 때문에 서로 다른 patient_group 이 합쳐진 횟수
    for groups in by_hash.values():
        for g in groups[1:]:
            if uf.find(groups[0]) != uf.find(g):
                merged_by_hash += 1
            uf.union(groups[0], g)
    return {r["patient_group"]: "G:" + uf.find(r["patient_group"]) for r in pool_rows}, merged_by_hash


def mode_label(counter):
    """가장 많은 라벨. 동수이면 라벨 번호가 작은 쪽 (항상 같은 결과가 나오게)."""
    return min(counter, key=lambda k: (-counter[k], k))


def split_pool(pool_rows, group_of, seed):
    """풀(원래 train)을 그룹 단위 + 클래스 층화로 5등분하고 첫 조각을 val 로 한다."""
    sg_ids = [group_of[r["patient_group"]] for r in pool_rows]

    # 그룹 안에 라벨이 섞이면 최빈값을 그 그룹의 대표 라벨로 쓴다
    label_counts = defaultdict(Counter)
    for sg, r in zip(sg_ids, pool_rows):
        label_counts[sg][int(r["label_id"])] += 1
    rep_label = {sg: mode_label(c) for sg, c in label_counts.items()}

    y = np.array([rep_label[sg] for sg in sg_ids])
    groups = np.array(sg_ids)
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    _, val_idx = next(splitter.split(np.zeros(len(y)), y, groups))
    val_set = set(val_idx.tolist())
    return ["val" if i in val_set else "train" for i in range(len(pool_rows))], label_counts, rep_label


def assign(rows, seed):
    """모든 행에 split 과 split_group 을 채우고 부가 정보를 돌려준다."""
    pool_idx = [i for i, r in enumerate(rows) if r["source_split"] == "train"]
    pool_rows = [rows[i] for i in pool_idx]
    group_of, merged_by_hash = build_split_groups(pool_rows)
    pool_split, label_counts, rep_label = split_pool(pool_rows, group_of, seed)

    for i, s in zip(pool_idx, pool_split):
        rows[i]["split"] = s
        rows[i]["split_group"] = group_of[rows[i]["patient_group"]]
    for r in rows:
        if r["source_split"] == "val":
            r["split"] = "check"
            r["split_group"] = "check:" + r["patient_group"]
        elif r["source_split"] == "test":
            r["split"] = "test"
            r["split_group"] = "test:" + r["patient_group"]
    return {"merged_by_hash": merged_by_hash, "label_counts": label_counts, "rep_label": rep_label,
            "group_of": group_of}


def verify(rows, info):
    """자동 검증. 실패하면 한국어 메시지와 함께 중단하고, 통과 항목은 (설명, 결과) 로 돌려준다."""
    checks = []

    def record(text, ok, detail=""):
        checks.append((text, ok, detail))
        if not ok:
            raise SystemExit(f"[검증 실패] {text} {detail}")

    c = Counter(r["split"] for r in rows)
    record("train+val 합이 5,216장", c["train"] + c["val"] == 5216, f"(실제 {c['train'] + c['val']})")
    record("check 16장", c["check"] == 16, f"(실제 {c['check']})")
    record("test 624장", c["test"] == 624, f"(실제 {c['test']})")
    record("전체 5,856장", len(rows) == EXPECTED_TOTAL, f"(실제 {len(rows)})")
    record("빈 split 값 0건", all(r["split"] in SPLIT_ORDER for r in rows))

    by_split_pg = defaultdict(set)
    by_split_sg = defaultdict(set)
    for r in rows:
        by_split_pg[r["split"]].add(r["patient_group"])
        by_split_sg[r["split"]].add(r["split_group"])
    both_pg = by_split_pg["train"] & by_split_pg["val"]
    both_sg = by_split_sg["train"] & by_split_sg["val"]
    record("train·val 양쪽에 있는 patient_group 0개", not both_pg, f"(실제 {len(both_pg)}개)")
    record("train·val 양쪽에 있는 split_group 0개", not both_sg, f"(실제 {len(both_sg)}개)")

    hash_splits = defaultdict(set)
    for r in rows:
        hash_splits[r["sha256"]].add(r["split"])
    bad_hash = [h for h, s in hash_splits.items() if len(s) > 1]
    record("같은 sha256 파일이 서로 다른 split 에 있는 경우 0건", not bad_hash, f"(실제 {len(bad_hash)}건)")

    test_now = {r["relative_path"] for r in rows if r["split"] == "test"}
    test_orig = {r["relative_path"] for r in rows if r["source_split"] == "test"}
    record("test 파일 집합이 원래 test 와 정확히 같음", test_now == test_orig)
    check_now = {r["relative_path"] for r in rows if r["split"] == "check"}
    check_orig = {r["relative_path"] for r in rows if r["source_split"] == "val"}
    record("check 파일 집합이 원래 val 과 정확히 같음", check_now == check_orig)
    pool_now = {r["relative_path"] for r in rows if r["split"] in ("train", "val")}
    pool_orig = {r["relative_path"] for r in rows if r["source_split"] == "train"}
    record("train+val 파일 집합이 원래 train 과 정확히 같음", pool_now == pool_orig)
    return checks


def ratio_table(rows):
    """split × 라벨별 영상 수, 그룹 수, split 내 비율."""
    out = []
    for split in SPLIT_ORDER:
        sub = [r for r in rows if r["split"] == split]
        total = len(sub)
        for label_id, label_name in list(LABEL_NAMES.items()) + [(None, "ALL")]:
            part = sub if label_id is None else [r for r in sub if int(r["label_id"]) == label_id]
            groups = {r["patient_group"] for r in part}
            sgs = {r["split_group"] for r in part}
            out.append({
                "split": split,
                "label_name": label_name,
                "images": len(part),
                "patient_groups": len(groups),
                "split_groups": len(sgs),
                "ratio_in_split": f"{(len(part) / total if total else 0):.4f}",
            })
    return out


def crossing_groups(rows, info):
    """원래 train 과 test 양쪽에 걸친 patient_group 목록과 배정 결과."""
    by_group = defaultdict(lambda: {"train": [], "test": []})
    for r in rows:
        if r["source_split"] in ("train", "test"):
            by_group[r["patient_group"]][r["source_split"]].append(r)
    out = []
    for g in sorted(by_group):
        d = by_group[g]
        if d["train"] and d["test"]:
            assigned = Counter(r["split"] for r in d["train"])
            out.append({
                "group": g,
                "train_n": len(d["train"]),
                "test_n": len(d["test"]),
                "assigned": "+".join(f"{k} {assigned[k]}" for k in ("train", "val") if assigned[k]),
                "split_group": d["train"][0]["split_group"],
            })
    return out


def build_report(rows, checks, info, table, crossing, seed, fieldnames_in):
    c = Counter(r["split"] for r in rows)
    lines = ["# split 분할 결과", ""]

    # 1) 규칙
    lines += [
        "## 1. 사용한 규칙", "",
        "- test 624장: 원래 test 그대로 `split=test`. 건드리지 않음.",
        "- 원래 val 16장: `split=check` (화면 점검용). 학습·검증 풀에 넣지 않음.",
        f"- 원래 train 5,216장만 환자 그룹 단위 + 클래스(label_id) 층화로 {N_SPLITS}등분하고 첫 조각을 val 로 사용 "
        f"(scikit-learn `StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True, random_state={seed})`, 약 20%). "
        "seed 를 바꿔 가며 고르지 않았다.",
        "- 같은 `patient_group` 은 같은 쪽. sha256 이 같은 파일도 patient_group 이 달라도 같은 쪽이 되도록, "
        "풀 안에서 patient_group 과 sha256 중복을 union-find 로 묶어 `split_group`(`G:` + 묶음 안 사전순 첫 patient_group)을 만들고 그 단위로 나눔.",
        "- 그룹 안에 라벨이 섞이면 최빈값을 그 그룹의 대표 라벨로 층화에 사용 (동수이면 label_id 가 작은 쪽).",
        "- 원래 train·test 양쪽에 걸친 그룹: test 는 고정이므로 그 그룹의 train 쪽 사진은 다른 그룹과 똑같이 train/val 풀에 포함 (목록은 4장).",
        "- check·test 행의 `split_group` 은 합치지 않고 `check:`/`test:` + patient_group 으로 표기.",
        "",
    ]

    # 2) 결과 표
    lines += ["## 2. 분할 결과", "",
              "| split | 라벨 | 영상 수 | patient_group 수 | split_group 수 | split 내 비율 |",
              "|---|---|---|---|---|---|"]
    for t in table:
        lines.append(f"| {t['split']} | {t['label_name']} | {t['images']} | {t['patient_groups']} | "
                     f"{t['split_groups']} | {float(t['ratio_in_split']) * 100:.1f}% |")
    lines += ["", f"전체: train {c['train']} / val {c['val']} / check {c['check']} / test {c['test']} = {len(rows)}", ""]

    pool = Counter((r["split"], int(r["label_id"])) for r in rows if r["split"] in ("train", "val"))
    lines += ["### 클래스별 val 비율 (val ÷ (train+val), 영상 기준)", "",
              f"| 라벨 | train | val | val 비율 | {int(RATIO_LOW * 100)}~{int(RATIO_HIGH * 100)}% 범위 |", "|---|---|---|---|---|"]
    all_in_range = True
    ratio_lines = []
    for k, name in LABEL_NAMES.items():
        tr, va = pool[("train", k)], pool[("val", k)]
        ratio = va / (tr + va)
        ok = RATIO_LOW <= ratio <= RATIO_HIGH
        all_in_range &= ok
        ratio_lines.append(f"| {name} | {tr} | {va} | {ratio * 100:.1f}% | {'안' if ok else '밖'} |")
    lines += ratio_lines
    total_val_ratio = c["val"] / (c["train"] + c["val"])
    lines += ["", f"전체 val 비율: {total_val_ratio * 100:.1f}%", ""]

    # 그룹 내 라벨 혼재
    lc = info["label_counts"]
    mixed = {sg: cnt for sg, cnt in lc.items() if len(cnt) > 1}
    rep = Counter(info["rep_label"].values())
    rows_pool = [r for r in rows if r["split"] in ("train", "val")]
    pg_labels = defaultdict(set)
    for r in rows_pool:
        pg_labels[r["patient_group"]].add(int(r["label_id"]))
    mixed_pg = sum(len(v) > 1 for v in pg_labels.values())
    tie = sum(1 for sg in mixed if sorted(lc[sg].values(), reverse=True)[0] == sorted(lc[sg].values(), reverse=True)[1])
    lines += ["### 그룹 내 라벨 혼재", "",
              f"- 풀 안의 patient_group {len(pg_labels)}개 중 라벨이 둘 이상인 그룹: {mixed_pg}개 "
              "(예: `P:person1000` 에 세균성 1장 + 바이러스성 1장)",
              f"- 합친 split_group {len(lc)}개 중 라벨이 섞인 것: {len(mixed)}개 (그중 최빈값이 동수인 것 {tie}개)",
              f"- 대표 라벨 분포 (split_group 기준): " + ", ".join(f"{LABEL_NAMES[k]} {rep[k]}" for k in LABEL_NAMES),
              "- 층화에는 대표 라벨을 쓰지만, 위 표의 영상 수·비율은 모두 영상 자체의 실제 라벨로 센 값이다.",
              "- 폐렴 번호 `personN` 하나에 세균성·바이러스성이 함께 있는 경우가 많아, 이 번호가 실제 한 사람인지 "
              "세균/바이러스 파일에 따로 매긴 번호인지는 확인할 수 없다 (5장 한계 참고).",
              f"- sha256 때문에 서로 다른 patient_group 이 합쳐진 횟수: {info['merged_by_hash']}회", ""]

    # 3) 검증
    check_pool_pg = {r["patient_group"] for r in rows if r["split"] in ("train", "val")}
    check_rows = [r for r in rows if r["split"] == "check"]
    check_pg = {r["patient_group"] for r in check_rows}
    overlap = check_pg & check_pool_pg
    overlap_train = check_pg & {r["patient_group"] for r in rows if r["split"] == "train"}
    overlap_val = check_pg & {r["patient_group"] for r in rows if r["split"] == "val"}
    lines += ["## 3. 검증 결과", ""]
    for text, ok, detail in checks:
        lines.append(f"- [{'x' if ok else ' '}] {text}")
    lines.append(f"- [{'x' if all_in_range else ' '}] 클래스별 val 비율이 {int(RATIO_LOW * 100)}~{int(RATIO_HIGH * 100)}% 범위 안 (정보용)")
    lines += ["",
              f"- 정보: check 16장의 patient_group {len(check_pg)}개 중 train/val 풀과 겹치는 것 {len(overlap)}개 "
              f"(train {len(overlap_train)}개, val {len(overlap_val)}개). 겹쳐도 실패로 보지 않는다.",
              "- 정보: test 안의 `NORMAL2-IM-0095-0001` 과 `NORMAL2-IM-0096-0001` 은 같은 이미지인데 patient_group 이 다르다. "
              "둘 다 test 이므로 영향 없음.",
              ""]
    if not all_in_range:
        lines += ["> 일부 클래스의 val 비율이 범위를 벗어났다. 그룹 단위로 나누면 한 그룹이 여러 장이라 "
                  "정확히 20%로 맞출 수 없다. seed 나 fold 를 바꿔 고르지 않고 이대로 사용한다.", ""]

    # 4) 걸친 그룹
    n_val = sum(1 for x in crossing if x["assigned"].startswith("val"))
    n_train = sum(1 for x in crossing if "val" not in x["assigned"])
    n_both = len(crossing) - n_val - n_train
    lines += [f"## 4. 원래 train·test 에 걸친 그룹 ({len(crossing)}개)", "",
              f"- train 쪽 사진이 전부 train 으로 배정된 그룹: {n_train}개",
              f"- 전부 val 로 배정된 그룹: {n_val}개",
              f"- train·val 로 나뉜 그룹: {n_both}개 (sha256 로 묶이지 않는 한 0이어야 함)",
              "- 이 그룹들의 test 쪽 사진은 그대로 test 다. train 쪽 사진은 다른 그룹과 똑같이 취급했다.", "",
              "| patient_group | train 쪽 장수 | test 쪽 장수 | 배정된 분할 | split_group |",
              "|---|---|---|---|---|"]
    for x in crossing:
        lines.append(f"| {x['group']} | {x['train_n']} | {x['test_n']} | {x['assigned']} | {x['split_group']} |")
    lines.append("")

    # 5) 한계
    lines += ["## 5. 한계", "",
              "- 파일명 번호가 같은 환자를 뜻하는지는 원천 설명으로 확인되지 않았다. 번호가 같으면 같은 사람이라고 가정하고 같은 쪽에 넣었을 뿐이다.",
              "- 원래 train·test 에 걸친 그룹은 같은 환자일 수도, 분할마다 번호를 따로 매긴 것일 수도 있다. "
              "같은 환자라면 train 에서 본 환자가 test 에도 있어 test 성능이 실제보다 좋게 나올 수 있다.",
              "- 정상(N)과 폐렴(P)은 번호 체계가 같다고 가정하지 않아, 같은 사람이 두 라벨에 모두 있어도 찾을 수 없다.",
              "- sha256 이 같지 않은 거의 같은 사진(다시 저장·크기 변경 등)은 찾지 못한다.",
              "- 그룹 단위 분할이라 클래스별 val 비율이 정확히 20%는 아니다.",
              "- 원래 val(check) 16장은 바이러스성이 0장이라 3분류 점검에는 한계가 있다.",
              ""]
    return "\n".join(lines)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest.csv 경로")
    ap.add_argument("--out", required=True, help="결과를 저장할 폴더")
    ap.add_argument("--seed", type=int, default=42, help="분할 seed (기본 42)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames_in, rows = load_manifest(manifest_path)
    check_input(rows)
    info = assign(rows, args.seed)
    checks = verify(rows, info)

    table = ratio_table(rows)
    crossing = crossing_groups(rows, info)

    write_csv(out_dir / "split_manifest.csv", fieldnames_in + ["split_group"], rows)
    write_csv(out_dir / "split_summary.csv",
              ["split", "label_name", "images", "patient_groups", "split_groups", "ratio_in_split"], table)
    report = build_report(rows, checks, info, table, crossing, args.seed, fieldnames_in)
    (out_dir / "split_report.md").write_text(report, encoding="utf-8")

    c = Counter(r["split"] for r in rows)
    print(f"분할 완료: train {c['train']} / val {c['val']} / check {c['check']} / test {c['test']}")
    print(f"저장: {out_dir / 'split_manifest.csv'}")
    print(f"저장: {out_dir / 'split_summary.csv'}")
    print(f"저장: {out_dir / 'split_report.md'}")


if __name__ == "__main__":
    main()
