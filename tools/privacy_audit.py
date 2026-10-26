"""저장소를 공개하기 전에 개인정보·비밀 정보가 남아 있는지 훑어 본다 (읽기만 한다, 아무것도 고치지 않는다).

찾는 것: 접근 키·토큰·개인키, 계정 번호·ARN·AWS 자원 ID, IP, 이메일, 전화번호, 내 컴퓨터 경로와 사용자 이름,
수강생 ID(say3-숫자 등), 학교·과정 식별 문자열, 사람 이름 후보. PDF는 글자를 뽑아서(pdftotext) 같이 본다. 이미지는 글자를 읽을 수 없으니 눈으로 본다.
결과는 규칙별로 개수와 앞부분 몇 개(값은 일부 가림)만 출력한다.

사용 예:
    python3 tools/privacy_audit.py <저장소 폴더>
"""
import collections
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).expanduser()
TEXT_SUFFIX = {".py", ".md", ".csv", ".json", ".txt", ".sh", ".html", ".css", ".yml", ".yaml", ".sha256", ".log", ".ndjson", ".cfg", ".toml", ".gitignore", ""}

RULES = [
    ("AWS 접근 키", r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA|AIPA)[0-9A-Z]{16}\b"),
    ("비밀 값 대입(secret/token/password/api key = ...)", r"(?i)\b(?:secret|token|passwd|password|api[_-]?key|access[_-]?key)\b['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9/+=_\-]{12,}"),
    ("GitHub 토큰", r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    ("개인키", r"-----BEGIN [A-Z ]*PRIVATE KEY"),
    ("12자리 숫자(계정 번호 후보)", r"(?<![0-9A-Za-z])\d{12}(?![0-9A-Za-z])"),
    ("ARN", r"\barn:aws[a-z\-]*:[a-z0-9\-]*:[a-z0-9\-]*:[0-9<>A-Za-z_\-]*:[^\s\"'`)]+"),
    ("AWS 자원 ID(sg-, subnet-, vpc-, vol-, i-, ami- ...)", r"\b(?:sg|subnet|vpc|vol|eni|ami|igw|rtb|acl)-[0-9a-f]{8,17}\b|\bi-[0-9a-f]{8,17}\b"),
    ("AWS 호스트 이름", r"[A-Za-z0-9.\-]*\.(?:compute|compute-1)\.amazonaws\.com|ec2-\d+-\d+-\d+-\d+"),
    ("IPv4 주소", r"(?<![\d.A-Za-z])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
    ("이메일", r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+\.[A-Za-z0-9.\-]+"),
    ("전화번호", r"(?<!\d)01[016789][-\s]?\d{3,4}[-\s]?\d{4}(?!\d)"),
    ("주민번호 모양", r"(?<!\d)\d{6}-[1-4]\d{6}(?!\d)"),
    ("내 컴퓨터 경로(/Users, /home, /Volumes, /private/tmp)", r"/Users/[^/\s\"'`)]+|/home/[^/\s\"'`)]+|/Volumes/[^\s\"'`)]+|/private/tmp/[^\s\"'`)]+|C:\\Users\\[^\\\s]+"),
    ("사용자 이름 조각(skku, aws3, 08)", r"(?i)skku[_\-]?aws3|aws3[_\-]\d+|skku_"),
    ("수강생 ID 모양(say3-숫자)", r"say3[-_]\d+"),
    ("학교·과정 식별(SKKU, 성균관, AI CAMPUS)", r"SKKU|성균관|AI ?CAMPUS|AICAM"),
    ("사람 이름 후보(오형준, 유진, 교수, 강사님)", r"오형준|유진|교수님|강사님|[가-힣]{2,4} (?:교수|강사|박사)"),
    ("컴퓨터 이름(iMac, MacBook, .local)", r"\biMac\b|MacBook|\b[\w\-]+\.local\b"),
    ("GitHub 사용자 이름", r"MuakTiger"),
]


def files():
    for p in sorted(REPO.rglob("*")):
        if p.is_file() and ".git" not in p.relative_to(REPO).parts:
            yield p


def text_of(p: Path):
    suf = p.suffix.lower()
    if suf == ".pdf":
        try:
            return subprocess.run(["pdftotext", "-q", str(p), "-"], capture_output=True, text=True, check=True).stdout + subprocess.run(["pdfinfo", str(p)], capture_output=True, text=True, check=True).stdout
        except Exception:
            return None
    if suf in TEXT_SUFFIX:
        try:
            return p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return None
    return None


def mask(s):
    s = s.strip()
    return s if len(s) <= 6 else s[:3] + "…" + s[-2:]


def main():
    hits = collections.defaultdict(lambda: collections.defaultdict(list))   # 규칙 -> 파일 -> [값]
    n_text = n_pdf = n_bin = 0
    for p in files():
        t = text_of(p)
        if t is None:
            n_bin += 1
            continue
        n_pdf += p.suffix.lower() == ".pdf"
        n_text += p.suffix.lower() != ".pdf"
        rel = str(p.relative_to(REPO))
        for name, pat in RULES:
            for m in re.finditer(pat, t):
                hits[name][rel].append(m.group(0))
    print(f"검사: 글자 파일 {n_text}개, PDF {n_pdf}개, 그림 등 {n_bin}개(이미지는 별도로 눈으로 확인)\n")
    for name, _ in RULES:
        per_file = hits.get(name, {})
        total = sum(len(v) for v in per_file.values())
        print(f"■ {name}: {total}건, 파일 {len(per_file)}개")
        shown = 0
        uniq = collections.Counter(v for vs in per_file.values() for v in vs)
        for val, c in uniq.most_common(8):
            files_with = [f for f, vs in per_file.items() if val in vs][:3]
            print(f"    '{mask(val) if name in ('이메일', '12자리 숫자(계정 번호 후보)', 'IPv4 주소', '전화번호') else val[:90]}' ×{c}  예: {', '.join(files_with)}")
            shown += 1
        if shown == 0:
            print("    (없음)")


if __name__ == "__main__":
    main()
