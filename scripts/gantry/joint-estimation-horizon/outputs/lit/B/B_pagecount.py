from pypdf import PdfReader
import glob
for fn in ["B_konda2004.pdf","B_newman2021varpro.pdf","B_vonstosch2014.pdf","B_bradley2022.pdf"]:
    try:
        r = PdfReader(fn)
        print(fn, len(r.pages), "pages")
    except Exception as e:
        print(fn, "ERROR", e)
