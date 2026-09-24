import sys, re
from pypdf import PdfReader

fname = sys.argv[1]
terms = sys.argv[2:]
r = PdfReader(fname)
print(f"FILE={fname} PAGES={len(r.pages)}")
for i, pg in enumerate(r.pages):
    t = pg.extract_text() or ""
    # aggressive: strip ligature-slash markers and ALL whitespace, so word matches survive
    # font-run space insertion and missing-glyph slashes.
    tclean = t.replace("/", "")
    tclean_nospace = re.sub(r"\s+", "", tclean).lower()
    for p in [x.lower().replace(" ", "") for x in terms]:
        if p in tclean_nospace:
            idx = tclean_nospace.find(p)
            print(f"=== p{i+1} [{p}] found at char {idx} (page has {len(tclean_nospace)} chars)")
