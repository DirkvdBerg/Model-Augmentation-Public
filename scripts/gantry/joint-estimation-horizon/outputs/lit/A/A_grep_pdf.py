import sys, re
from pypdf import PdfReader

fname = sys.argv[1]
terms = sys.argv[2:]
r = PdfReader(fname)
print(f"FILE={fname} PAGES={len(r.pages)}")
for i, pg in enumerate(r.pages):
    t = pg.extract_text() or ""
    tl = t.lower()
    for p in [x.lower() for x in terms]:
        for m in re.finditer(re.escape(p), tl):
            a = max(0, m.start() - 500)
            b = min(len(t), m.end() + 500)
            print(f"=== p{i+1} [{p}]")
            print(" ".join(t[a:b].split()))
            print()
