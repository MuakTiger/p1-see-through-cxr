"""PDF 후처리 (pymupdf 필요: pip install pymupdf).

1) 표지의 목차 링크가 가리키는 쪽을 읽어 docs/final_report/toc_pages.json 에 쓴다. 번호가 바뀌었으면 build.py 를 다시 돌려야 한다.
2) --finalize: PDF 책갈피(목차)와 문서 정보(제목·작성자)를 넣는다.

사용 예 (p1 폴더에서):
    python tools/final_report/build.py && python tools/final_report/pdf_post.py     # 쪽 번호 읽기
    python tools/final_report/build.py && python tools/final_report/pdf_post.py --finalize
"""
import json
import sys
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE.parents[1] / "docs/final_report"
PDF = OUT / "P1_형준팀_프로젝트 결과보고서.pdf"
TOC = [  # build.py 의 TOC 와 같아야 한다
    ("sec1", "1", "프로젝트 개요"), ("sec2", "2", "기획안 대비 변경 사항"), ("sec3", "3", "데이터와 분할"), ("sec4", "4", "방법"),
    ("sec5", "5", "결과"), ("sec6", "6", "점수를 믿을 수 있는가: 진단"), ("sec7", "7", "AWS 구성, 검산, 종료 기록"), ("sec8", "8", "한계와 다음 단계"),
    ("secA", "부록 A", "필터 33조건 전체 결과"), ("secB", "부록 B", "초기 실험 (입력 224)"), ("secC", "부록 C", "가이드 필수 제출물 대응표"),
    ("secD", "부록 D", "모델 S3 주소"), ("secE", "부록 E", "제출 패키지 구성과 재현 방법"), ("secF", "부록 F", "용어"),
]


def read_pages(doc):
    """표지(0쪽)의 목차 링크는 이름 붙은 위치(sec1…)로 가는 링크다. 그 위치가 있는 쪽 번호를 읽는다."""
    dest = {l["nameddest"]: l["page"] + 1 for l in doc[0].get_links() if l.get("nameddest")}
    missing = [sid for sid, _, _ in TOC if sid not in dest]
    assert not missing, missing
    return {sid: str(dest[sid]) for sid, _, _ in TOC}


def main():
    finalize = "--finalize" in sys.argv
    doc = fitz.open(PDF)
    pages = read_pages(doc)
    f = OUT / "toc_pages.json"
    old = json.load(open(f)) if f.exists() else {}
    json.dump(pages, open(f, "w"), ensure_ascii=False, indent=1)
    print("쪽수", doc.page_count, "| 목차 쪽:", pages)
    if old != pages:
        print("→ 목차 쪽 번호가 바뀌었다. build.py 를 다시 돌린 뒤 다시 확인한다.")
        if finalize:
            sys.exit(1)
        return
    if finalize:
        toc = [[1, f"{no}  {title}", int(pages[sid])] for sid, no, title in TOC]
        doc.set_toc(toc)
        doc.set_metadata({"title": "P1 See-Through CXR 결과보고서", "author": "오형준", "subject": "성균관대 AI CAMPUS 바이오헬스 AI 1기 P1 프로젝트 결과보고서",
                          "keywords": "흉부 X선, 폐렴 3분류, ResNet18, 장비 설정 필터, AWS SageMaker"})
        tmp = PDF.with_suffix(".tmp.pdf")
        doc.save(tmp, garbage=3, deflate=True)
        doc.close()
        tmp.replace(PDF)
        print("책갈피·문서 정보 저장:", PDF)


if __name__ == "__main__":
    main()
