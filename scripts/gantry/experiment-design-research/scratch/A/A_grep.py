import sys,re
from pypdf import PdfReader
r=PdfReader(sys.argv[1])
ctx=int(sys.argv[2])
pats=[x.lower() for x in sys.argv[3:]]
print("PAGES",len(r.pages))
for i,pg in enumerate(r.pages):
    t=pg.extract_text() or ""; tl=t.lower()
    for p in pats:
        for m in list(re.finditer(re.escape(p),tl))[:3]:
            a=max(0,m.start()-ctx); b=min(len(t),m.end()+ctx)
            print(f"=== p{i+1} [{p}]"); print(" ".join(t[a:b].split()))
