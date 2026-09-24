import sys, re
from pypdf import PdfReader

fname = sys.argv[1]
terms = sys.argv[2:]
r = PdfReader(fname)
print(f"FILE={fname} PAGES={len(r.pages)}")
for i, pg in enumerate(r.pages):
    t = pg.extract_text() or ""
    # ligature placeholder cleanup: remove stray slashes used for fi/ti/ffi glyphs, collapse ws
    tclean = t.replace("/", "")
    tclean = re.sub(r"\s+", " ", tclean)
    tl = tclean.lower()
    for p in [x.lower() for x in terms]:
        for m in re.finditer(re.escape(p), tl):
            a = max(0, m.start() - 400)
            b = min(len(tclean), m.end() + 400)
            print(f"=== p{i+1} [{p}]")
            print(tclean[a:b])
            print()
