import sys,os,pickle,hashlib
from pypdf import PdfReader
pdf=sys.argv[1]; a=int(sys.argv[2]); b=int(sys.argv[3])
h=hashlib.md5(pdf.encode()).hexdigest()[:10]
cache=os.path.join(os.path.dirname(os.path.abspath(__file__)),f"cache_{h}.pkl")
if os.path.exists(cache): pages=pickle.load(open(cache,'rb'))
else:
    r=PdfReader(pdf); pages=[(pg.extract_text() or "") for pg in r.pages]; pickle.dump(pages,open(cache,'wb'))
for i in range(a-1,min(b,len(pages))):
    print(f"######## page {i+1}"); print(pages[i])
