"""manifest.csv 생성 스크립트 (가이드 3장).

영상 5,856장을 한 줄씩 기록한 목록표를 만든다.
열: relative_path, label_id, label_name, source_split, split, patient_group,
    width, height, channels, sha256, readable

사용 예 (SageMaker Studio 터미널):
    python make_manifest.py --root ~/data/chest_xray --out ~/p1/manifests

결과:
    manifest.csv         목록표 (split 열은 다음 단계에서 채움)
    manifest_report.md   점검 결과와 환자 그룹 규칙·한계 기록
"""
import argparse
import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

LABEL_NAMES = {0: "NORMAL", 1: "BACTERIAL_PNEUMONIA", 2: "VIRAL_PNEUMONIA"}
IMAGE_EXTS = {".jpeg", ".jpg", ".png"}
SOURCE_SPLITS = ("train", "val", "test")

# 가이드 1장의 실제 집계값 (원래 분할 × 라벨)
EXPECTED = {
    ("train", 0): 1341, ("train", 1): 2530, ("train", 2): 1345,
    ("val", 0): 8, ("val", 1): 8, ("val", 2): 0,
    ("test", 0): 234, ("test", 1): 242, ("test", 2): 148,
}

# 환자 그룹 규칙. 정상과 폐렴의 번호 체계가 같다고 가정하지 않도록 접두어로 구분한다.
# 파일명 끝에 번호가 한 마디 더 붙은 변형(IM-0011-0001-0001, person447_virus_921_1)도 허용한다.
PNEUMONIA_RE = re.compile(r"^(person\d+)_(bacteria|virus)_\d+(?:_\d+)?$", re.IGNORECASE)
NORMAL_RE = re.compile(r"^((?:NORMAL2-)?IM-\d+)(?:-\d+)+$", re.IGNORECASE)


def get_label(image_path):
    """가이드의 라벨 규칙. 파일명은 정답 생성과 추적에만 쓴다."""
    folder = image_path.parent.name
    name = image_path.name.lower()
    if folder == "NORMAL":
        return 0
    if folder == "PNEUMONIA" and "bacteria" in name:
        return 1
    if folder == "PNEUMONIA" and "virus" in name:
        return 2
    raise ValueError("정답 라벨을 확인할 수 없습니다")


def get_patient_group(image_path, label_id):
    stem = image_path.stem
    if label_id == 0:
        m = NORMAL_RE.match(stem)
        return f"N:{m.group(1).upper()}" if m else f"unknown:{stem}"
    m = PNEUMONIA_RE.match(stem)
    return f"P:{m.group(1).lower()}" if m else f"unknown:{stem}"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_image_info(path):
    """실제로 디코딩까지 해 보고 크기와 채널 수를 돌려준다."""
    try:
        with Image.open(path) as im:
            im.load()
            return im.width, im.height, len(im.getbands()), True, ""
    except Exception as e:  # 깨진 파일은 기록만 하고 계속 진행
        return "", "", "", False, type(e).__name__


def collect_files(root):
    files, skipped = [], []
    for split in SOURCE_SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            raise SystemExit(f"{split_dir} 폴더가 없습니다. --root 는 train/val/test 가 바로 들어 있는 폴더여야 합니다.")
        for p in sorted(split_dir.rglob("*")):
            if not p.is_file():
                continue
            hidden = p.name.startswith(".") or "__MACOSX" in p.parts
            if hidden or p.suffix.lower() not in IMAGE_EXTS:
                skipped.append(p)
                continue
            files.append((split, p))
    return files, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="train/val/test 가 들어 있는 데이터 루트")
    ap.add_argument("--out", required=True, help="manifest.csv 를 저장할 폴더")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    files, skipped = collect_files(root)
    print(f"영상 파일 {len(files)}개 처리 시작 (건너뛴 파일 {len(skipped)}개)", flush=True)

    rows, label_errors, read_errors = [], [], []
    for i, (source_split, path) in enumerate(files, 1):
        rel = path.relative_to(root).as_posix()
        try:
            label_id = get_label(path)
        except ValueError:
            label_errors.append(rel)
            label_id = -1
        width, height, channels, readable, err = read_image_info(path)
        if not readable:
            read_errors.append(f"{rel} ({err})")
        rows.append({
            "relative_path": rel,
            "label_id": label_id,
            "label_name": LABEL_NAMES.get(label_id, "UNKNOWN"),
            "source_split": source_split,
            "split": "",  # 다음 단계(split_manifest)에서 확정
            "patient_group": get_patient_group(path, label_id) if label_id >= 0 else f"unknown:{path.stem}",
            "width": width,
            "height": height,
            "channels": channels,
            "sha256": sha256_of(path),
            "readable": readable,
        })
        if i % 500 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = build_report(rows, label_errors, read_errors, skipped, root)
    (out_dir / "manifest_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"저장: {manifest_path}")
    print(f"저장: {out_dir / 'manifest_report.md'}")


def build_report(rows, label_errors, read_errors, skipped, root):
    lines = ["# manifest 점검 결과", ""]

    # 1) 클래스별 수 비교
    counts = Counter((r["source_split"], r["label_id"]) for r in rows if r["label_id"] >= 0)
    lines += ["## 1. 원래 분할 × 라벨 영상 수 (가이드 표와 비교)", "",
              "| 분할 | 라벨 | 실제 | 가이드 | 결과 |", "|---|---|---|---|---|"]
    all_match = True
    for (split, label), expected in EXPECTED.items():
        actual = counts.get((split, label), 0)
        ok = actual == expected
        all_match &= ok
        lines.append(f"| {split} | {LABEL_NAMES[label]} | {actual} | {expected} | {'OK' if ok else '불일치'} |")
    lines += ["", f"전체 영상 수: {len(rows)} (가이드 5,856)", ""]

    # 2) 라벨 누락, 읽기 실패, 건너뛴 파일
    lines += ["## 2. 라벨 누락 · 읽기 실패", "",
              f"- 라벨 누락: {len(label_errors)}건",
              f"- 읽기 실패: {len(read_errors)}건",
              f"- 영상이 아니어서 건너뛴 파일: {len(skipped)}건"]
    for rel in label_errors[:20]:
        lines.append(f"  - 라벨 누락: {rel}")
    for item in read_errors[:20]:
        lines.append(f"  - 읽기 실패: {item}")
    for p in skipped[:10]:
        lines.append(f"  - 건너뜀: {p.relative_to(root).as_posix()}")
    lines.append("")

    # 3) 영상 형식
    channel_counts = Counter(r["channels"] for r in rows if r["readable"])
    widths = [r["width"] for r in rows if r["readable"]]
    heights = [r["height"] for r in rows if r["readable"]]
    lines += ["## 3. 영상 형식", "",
              f"- 채널 수 분포: {dict(channel_counts)}",
              f"- 너비 범위: {min(widths)} ~ {max(widths)}, 높이 범위: {min(heights)} ~ {max(heights)}" if widths else "- 읽은 영상 없음",
              ""]

    # 4) 환자 그룹
    groups = defaultdict(set)
    for r in rows:
        groups[r["patient_group"]].add(r["source_split"])
    unknown = [g for g in groups if g.startswith("unknown:")]
    crossing = sorted(g for g, s in groups.items() if len(s) > 1 and not g.startswith("unknown:"))
    lines += ["## 4. 환자 그룹", "",
              f"- 그룹 수: {len(groups)} (정상 N: {sum(g.startswith('N:') for g in groups)}, "
              f"폐렴 P: {sum(g.startswith('P:') for g in groups)}, 규칙 밖 unknown: {len(unknown)})",
              f"- 원래 분할 여러 곳에 걸친 그룹: {len(crossing)}개 (예: {', '.join(crossing[:5]) or '없음'})",
              "",
              "규칙:",
              "- 폐렴: 파일명 `personN_(bacteria|virus)_M` (끝에 `_K`가 더 붙어도 됨) → 그룹 `P:personN`",
              "- 정상: 파일명 `IM-XXXX-YYYY` 또는 `NORMAL2-IM-XXXX-YYYY` (끝에 `-ZZZZ`가 더 붙어도 됨) → 그룹 `N:IM-XXXX` / `N:NORMAL2-IM-XXXX`",
              "",
              "한계:",
              "- 파일명 번호가 실제 동일 환자를 뜻하는지는 원천 설명으로 확인되지 않았다.",
              "- 정상(N)과 폐렴(P)의 번호 체계가 같다고 가정하지 않으므로, 같은 사람이 두 라벨에 모두 있어도 찾아낼 수 없다.",
              "- 원래 분할(train/test)에 걸친 그룹이 실제로 같은 환자인지, 분할마다 번호를 따로 매긴 것인지는 알 수 없다. "
              "test는 고정이므로, 걸치는 그룹의 처리 규칙은 분할 단계에서 정하고 기록한다.",
              ""]

    # 5) 해시 중복
    by_hash = defaultdict(list)
    for r in rows:
        by_hash[r["sha256"]].append(r)
    dup = [v for v in by_hash.values() if len(v) > 1]
    dup_cross = [v for v in dup if len({r["source_split"] for r in v}) > 1]
    lines += ["## 5. 같은 파일(sha256) 중복", "",
              f"- 중복 해시 묶음: {len(dup)}개 (그중 원래 분할을 가로지르는 묶음: {len(dup_cross)}개)"]
    for v in dup[:10]:
        names = [r["relative_path"] for r in v]
        lines.append("  - " + " = ".join(names[:5]) + (f" 외 {len(names) - 5}개" if len(names) > 5 else ""))
    lines += ["", "## 6. 완료 기준", "",
              f"- [{'x' if not label_errors else ' '}] 라벨 누락 0건",
              "- [x] 읽기 실패 내역 확인 (위 2번)",
              f"- [{'x' if all_match else ' '}] 클래스별 수가 가이드 표와 일치",
              "- [x] 환자 그룹 규칙과 한계 문서화 (위 4번)",
              ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
