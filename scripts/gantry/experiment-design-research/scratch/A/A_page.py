import sys
from pypdf import PdfReader
r=PdfReader(sys.argv[1])
for p in sys.argv[2:]:
    print(f"##### page {p}"); print(r.pages[int(p)-1].extract_text())
