from pypdf import PdfReader
r = PdfReader("B_gaskin2023_cam.pdf")
print(len(r.pages), "pages")
