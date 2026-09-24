import sys,re,glob
from pypdf import PdfReader
pats=[p.lower() for p in sys.argv[2:]]
for f in glob.glob(sys.argv[1]):
    try: r=PdfReader(f)
    except Exception as e: continue
    for i,pg in enumerate(r.pages):
        try: t=pg.extract_text() or ""
        except Exception: continue
        tl=t.lower()
        for p in pats:
            for m in list(re.finditer(re.escape(p),tl))[:1]:
                a=max(0,m.start()-300); b=min(len(t),m.end()+300)
                print(f"=== {f.split('/')[-1][:60]} p{i+1} [{p}]: "+" ".join(t[a:b].split()))
