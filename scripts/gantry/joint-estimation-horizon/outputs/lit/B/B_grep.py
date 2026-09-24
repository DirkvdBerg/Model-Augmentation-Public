import sys, re
from pypdf import PdfReader

def grep(fn, terms, ctx=500):
    r = PdfReader(fn)
    for i, pg in enumerate(r.pages):
        t = pg.extract_text() or ""
        tl = t.lower()
        for p in terms:
            for m in re.finditer(re.escape(p.lower()), tl):
                a = max(0, m.start()-ctx); b = min(len(t), m.end()+ctx)
                print(f"=== p{i+1} [{p}] ===")
                print(" ".join(t[a:b].split()))
                print()

fn = sys.argv[1]
terms = sys.argv[2:]
grep(fn, terms)
