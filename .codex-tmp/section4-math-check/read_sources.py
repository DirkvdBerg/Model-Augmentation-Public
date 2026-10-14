from pathlib import Path
import fitz
root=Path('.codex-tmp/section4-math-check')
for tag,path,pages in [('g26','literature/Orthogonality/gyorok2026_orthogonal-by-construction-augmentation_IFAC-JSC_arXiv2511.01321.pdf',[2,3,4,5,9,10]),('g25','literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf',[4,5,6,7,8]),('kes','literature/augmentation/kessels2025_ai-control.pdf',[193,210])]:
 doc=fitz.open(path)
 for p in pages:
  page=doc[p-1]
  page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(str(root/f'{tag}-{p}.png'))
  print(f'===== {tag} PDF PAGE {p} =====\n{page.get_text()}')
