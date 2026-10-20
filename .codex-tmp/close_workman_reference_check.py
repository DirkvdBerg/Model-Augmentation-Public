from pathlib import Path
import pymupdf,re
pdf=Path('literature/thesis-reference-new/Workman-Die_to_Wafer_Hybrid_Bonding_and_Fine_Pitch_Considerations.pdf')
d=pymupdf.open(pdf)
assert len(d)==7
first=d[0].get_text(); last=d[-1].get_text()
assert '10.1109/ECTC32696.2021.00326' in first and '2071' in first and '2077' in last
report=Path('Thesis-writeup/Documentation/REFERENCE-CHECK-2026-10-20.md')
s=report.read_text(encoding='utf-8')
old=next(line for line in s.splitlines() if line.startswith('**22 of the 23'))
new='**All 23 selected source versions are now local across the checked folders.** Ten are already in `literature/thesis-referenced/`; thirteen are available in `literature/thesis-reference-new/` (Györök 2025 also had earlier published local copies). The new folder contains 14 PDFs representing 13 distinct papers, with Gautier downloaded twice as byte-identical files. Workman is now verified as the published IEEE ECTC 2021 article. No downloads remain for this 23-key citation snapshot. The four arXiv sources remain the selected preprints because no published version was found in the official searches. No PDF was renamed, moved, copied, replaced or deleted during these checks.'
s=s.replace(old,new)
old=next(line for line in s.splitlines() if line.startswith('| `workman2021diewafer`'))
cols=old.split(' | ')
cols[1]=cols[1].replace('local manuscript p. 1 verifies authors/title.','downloaded published PDF p. 1 verifies authors, title, conference, year, DOI and first page 2071; PDF p. 7 verifies final printed page 2077.')
cols[3]='YES: [Published IEEE PDF](<../../literature/thesis-reference-new/Workman-Die_to_Wafer_Hybrid_Bonding_and_Fine_Pitch_Considerations.pdf>), 7 PDF pages, printed pp. 2071–2077. Conference header, DOI 10.1109/ECTC32696.2021.00326 and ©2021 IEEE copyright establish the publisher version. The old manuscript remains untouched in thesis-referenced.'
cols[5]='No required bibliographic correction; the current entry already has all ten authors, pages, year and DOI. |'
s=s.replace(old,' | '.join(cols))
old='The Workman PDF already in thesis-referenced is a manuscript with template copyright metadata; the published IEEE version is the only remaining acquisition item in this refreshed citation set.'
new='The older Workman PDF in thesis-referenced has template copyright metadata, but the newly added Workman file in thesis-reference-new is the published IEEE version. Its first and final pages confirm the journal-independent conference pagination 2071–2077, matching DOI and 2021 IEEE copyright. No acquisition item remains.'
assert old in s
s=s.replace(old,new)
start=s.index('## To download')
end=s.index('## Bib corrections',start)
s=s[:start]+'''## To download

**None for the 23 cited keys in the refreshed extraction.** Workman's newly added IEEE ECTC 2021 publisher PDF resolves the last acquisition item. The four works with no published version found remain the selected arXiv preprints: `drenth2025lpvlfr`, `hoekstra2026lfr`, `hoekstra2026encoder`, and `raissi2017physicsinformedpart1`.

Correct new copies remain in `literature/thesis-reference-new/`; they have not replaced the older files under the key filenames in `literature/thesis-referenced/`. The table links directly to them. Bibliography corrections below are still proposals and have not been applied to refs.bib.

'''+s[end:]
assert '\u2014' not in s and '--' not in s
report.write_text(s,encoding='utf-8')
print('Published Workman verified: ECTC 2021, 7 pages, 2071–2077, matching DOI. Report updated: all 23 selected versions local, no downloads remain.')
