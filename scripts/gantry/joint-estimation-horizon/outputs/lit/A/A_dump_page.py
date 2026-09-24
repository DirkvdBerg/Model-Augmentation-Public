import sys
from pypdf import PdfReader

fname = sys.argv[1]
pg_idx = int(sys.argv[2])  # 0-based
r = PdfReader(fname)
t = r.pages[pg_idx].extract_text() or ""
print(t)
