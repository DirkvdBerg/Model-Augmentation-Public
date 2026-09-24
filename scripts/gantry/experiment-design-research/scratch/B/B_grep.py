import sys,re,os,pickle,hashlib
from pypdf import PdfReader
# usage: B_grep.py <pdf> <ctx> <pat1> [pat2...]; caches text
pdf=sys.argv[1]; ctx=int(sys.argv[2])
h=hashlib.md5(pdf.encode()).hexdigest()[:10]
cache=os.path.join(os.path.dirname(os.path.abspath(__file__)),f"cache_{h}.pkl")
if os.path.exists(cache):
    pages=pickle.load(open(cache,'rb'))
else:
    r=PdfReader(pdf); pages=[(pg.extract_text() or "") for pg in r.pages]
    pickle.dump(pages,open(cache,'wb'))
print("pages",len(pages))
maxhits=int(os.environ.get("MAXHITS","6"))
for p in sys.argv[3:]:
    n=0
    for i,t in enumerate(pages):
        tl=t.lower()
        for m in re.finditer(p.lower(),tl):
            a=max(0,m.start()-ctx); b=min(len(t),m.end()+ctx)
            print(f"=== p{i+1} [{p}]"); print(" ".join(t[a:b].split()))
            n+=1
            if n>=maxhits: break
        if n>=maxhits: break
    print(f"--- {p}: shown {n}")
