import sys,re
from pypdf import PdfReader
r=PdfReader(sys.argv[1])
W=int(sys.argv[2])
for i,pg in enumerate(r.pages):
    t=pg.extract_text() or ""; tl=t.lower()
    for p in [x.lower() for x in sys.argv[3:]]:
        for m in re.finditer(re.escape(p),tl):
            a=max(0,m.start()-W); b=min(len(t),m.end()+W)
            print(f"=== p{i+1} [{p}]"); print(" ".join(t[a:b].split()))
