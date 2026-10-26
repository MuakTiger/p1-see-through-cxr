import sys
from docx import Document
d = Document(sys.argv[1])
body = d.element.body
from docx.table import Table
from docx.text.paragraph import Paragraph
for el in body.iterchildren():
    tag = el.tag.split('}')[1]
    if tag == 'p':
        t = Paragraph(el, d).text
        if t.strip(): print(t)
    elif tag == 'tbl':
        for row in Table(el, d).rows:
            print(' | '.join(c.text.replace('\n', ' / ') for c in row.cells))
for s in d.sections:
    print('HEADER:', ' '.join(p.text for p in s.header.paragraphs))
