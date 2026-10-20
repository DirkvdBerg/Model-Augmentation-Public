from pathlib import Path
import pymupdf
p=Path('Thesis-writeup/Documentation/REFERENCE-CHECK-2026-10-20.md')
s=p.read_text(encoding='utf-8')
old='**Nine selected versions are already in `literature/thesis-referenced/`; twelve are absent or represented by a different version.** Two of the twelve are conditional: Forssell appears only in deleted text, and Venkatraman only in a todo. The final lists identify exact versions and official acquisition links.'
new='**Nine selected versions are already in `literature/thesis-referenced/`. Györök L4DC 2025 is also local, in `literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf` and `citation-audit/2026-10-18/external-sources/gyorok2025l4dc-published.pdf`; it does not need downloading.** Twelve selected versions are absent from the dedicated thesis folder or represented there by a different version, but only eleven remain on the download list. Two of those eleven are conditional: Forssell appears only in deleted text, and Venkatraman only in a todo. Other download statuses retain the original audit scope and are not claims that every alternate repository location has been exhaustively searched.'
assert old in s
s=s.replace(old,new)
old='NO: no key PDF in thesis-referenced. Official proceedings PDF was inspected online only.'
new='YES, already local outside thesis-referenced: `literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf`; another published copy is `citation-audit/2026-10-18/external-sources/gyorok2025l4dc-published.pdf`. Both are 13-page PMLR 283 proceedings PDFs; front matter inspected. No download needed. The dedicated `<key>.pdf` is absent.'
assert old in s
s=s.replace(old,new)
lines=s.splitlines(keepends=True)
removed=[line for line in lines if line.startswith('- `gyorok2025l4dc`: published L4DC 2025 / PMLR 283 proceedings PDF')]
assert len(removed)==1
s=''.join(line for line in lines if line not in removed)
needle='## Version discrepancies and remaining checks\n\n'
note='- **Correction after local-copy check:** Györök L4DC 2025 is already held in the published proceedings version at the two paths above. Both local PDFs carry the PMLR 283 proceedings front matter, title, authors and 13-page extent. Absence from `literature/thesis-referenced/` was incorrectly turned into a download requirement. No files were moved or copied.\n'
assert needle in s
s=s.replace(needle,needle+note)
for filename in ['literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf','citation-audit/2026-10-18/external-sources/gyorok2025l4dc-published.pdf']:
    d=pymupdf.open(filename)
    first=d[0].get_text()
    assert len(d)==13 and 'Proceedings of Machine Learning Research vol 283' in first and 'Orthogonal projection-based regularization' in first,filename
assert '\u2014' not in s and '--' not in s
p.write_text(s,encoding='utf-8')
print('Report corrected; both local copies have published PMLR front matter; 11 download entries remain.')
