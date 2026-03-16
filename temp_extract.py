import PyPDF2
import sys
pdf = PyPDF2.PdfReader(sys.argv[1])
text = ''
for page in pdf.pages:
    text += page.extract_text()
print(text[:5000])