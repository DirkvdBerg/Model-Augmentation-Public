# Handoff: check every reference the thesis cites, decide the version to cite, and list what Dirk still has to download (with official links)
**From**: session of 2026-10-20 | **For**: Codex | **Branch**: Augmentation | **Effort suggested**: medium (mechanical,
but every bibliographic claim must be read from the source, not from memory)

## 1. Task
For every citation key that appears in the thesis text (`Thesis-writeup/Writing/sections/*.tex`, `main.tex`,
`frontmatter/`), produce one row in a report table that states: (1) the bibliographic record as it should be cited
(authors, title, venue, volume, pages or article number, year, DOI), checked against the publisher's or the
proceedings' own page; (2) which version to cite, under Dirk's rule below; (3) whether that exact version is present
as a PDF in `literature/thesis-referenced/`; (4) if not, the official download link for that version; (5) every
pinpoint the text uses for that key (`\cite[Sec.~5.2]{...}`, `[Eq.~(22)]`, `[Remark~5.6]`, ...), with its location
`file:line`, and whether it exists with that number in the version to be cited; (6) the corrections the `refs.bib`
entry needs. Write the report to `Thesis-writeup/Documentation/REFERENCE-CHECK-2026-10-20.md`. End the report with
two lists: **to download** (key, version, official link) and **bib corrections** (key, field, current value, correct
value, source of the correct value).

**Dirk's version rule (2026-10-20):** cite the peer-reviewed published version wherever it exists. Cite an arXiv
version only when no published version exists; then cite it as a preprint, with "submitted to <venue>" if the venue is
known from the paper or the authors' page. The PDF on disk must be the version that is cited, because pinpoint numbers
(sections, equations, remarks) can differ between versions.

## 2. Out of scope
- The thesis text: do not edit any `.tex` file. Pinpoints that do not match go in the report.
- `refs.bib`: do not edit it; Dirk applies the corrections from the report.
- Whether a cited source supports the sentence it is attached to (the claim check): done per section later, against
  the final wording. Report only bibliographic facts and pinpoint existence.
- Downloading files: list links; Dirk downloads (publisher access goes through his TU/e account).
- `docs/` and `tasks/`: read only (cross-agent ownership); proposals go in the report.
- `kamtin-fp-model/` and `kamtin-data/Data Telica/`: never read or write.

## 3. Where things stand
- Branch `Augmentation`, last commit `e588e63` (2026-10-20); the tree is dirty, Dirk commits himself. No git write
  operations.
- Bibliography: `Thesis-writeup/Writing/refs.bib` (IEEEtran style). Key-to-PDF map with notes:
  `docs/references.md`. PDFs of cited sources: `literature/thesis-referenced/<key>.pdf`; other copies of some sources
  under `literature/` (for example `literature/lpv-lfr/`, `literature/augmentation/`, `literature/books/`,
  `literature/gantry/`).
- Citation markers in the text: `\cite{}` for checked sources, `\ucite{}` for sources Dirk has not checked yet (shown
  in orange in the editor build, left out of the review build). Some citations sit inside `\todo{}` or `\revnote{}`;
  report them in a separate group (they may be deleted).

## 4. Established and verified
Keys cited in the text on 2026-10-20 (extracted from `\cite`, `\ucite` outside comments; `pdf` = file present in
`literature/thesis-referenced/`; bib type and DOI as in `refs.bib`):

| Key | bib type | Marker | Cited in | PDF | DOI in bib | Note |
|-|-|-|-|-|-|-|
| beintema2023subnet | article | cite | 03, 05, 99 | yes | none | Automatica 2023 |
| downey2017psrnn | inproceedings | cite | 03 | yes | none | NeurIPS; no year in bib |
| drenth2025lpvlfr | misc | cite | 02, 03, 99 | yes | none | bib = arXiv 2605.12203 (2026); a local IFAC version exists at `literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf`; its equation numbers differ from the arXiv version (Eq. (9) vs (5)) |
| forssell1999revisited | techreport | cite | 03 | yes | none | key says 1999, bib year 1998; the citing sentence was struck in Section IV-B, check whether it still appears outside `\del{}` |
| garcia2013model | article | cite | 01, 02, 05, 99 | yes | none | Control Engineering Practice 2013 |
| gautier1990minimum | article | ucite | 02 | no | 10.1109/70.56655 | unchecked source |
| gyorok2025l4dc | inproceedings | cite | 04 | no | none | L4DC 2025, PMLR vol. 283 |
| gyorok2026obc | article | cite | 01, 03, 04 | yes | none | IFAC Journal of Systems and Control 2026 |
| hefny2015supervised | inproceedings | cite | 03 | yes | none | NeurIPS; no year in bib |
| hoekstra2026encoder | misc | cite | 03, 99 | yes | none | arXiv 2602.13108; submitted to an IFAC venue per `docs/references.md` |
| hoekstra2026lfr | misc | cite | 01, 02, 03 | yes | none | arXiv 2602.17297; "submitted to Automatica" |
| karniadakis2021physicsinformed | article | cite | 01 | yes | 10.1038/s42254-021-00314-5 | |
| kessels2025ai | phdthesis | cite | 01, 03, 04 | yes | none | TU/e PhD thesis 2025 |
| noel2018greybox | article | cite | 01 | yes | none | International Journal of Control 2018 |
| ovchinnikov2021computing | article | cite | 02 | no | 10.1016/j.sysconle.2021.105030 | |
| raissi2017physicsinformedpart1 | article | cite | 01 | yes | none | arXiv 1711.10561 |
| retzler2024 | article | cite | 01 | yes | none | Data-Centric Engineering 2024 |
| teo2007dynamic | article | cite | 01 | yes | none | Mechatronics 2007 |
| toth2010modeling | book | cite | 02 | yes | none | Springer LNCIS 403, 2010 |
| venkatraman2017psd | inproceedings | ucite | 03 | no | none | inside a `\todo{}`; unchecked |
| willard2020integrating | article | cite | 01 | yes | 10.1145/3514228 | key says 2020, bib year 2022 |

- PDFs in `literature/thesis-referenced/` whose key is no longer cited: `drenth2025thesis`, `hwang2024semiconductor`,
  `irds2022`, `lambrechts2005trajectory`, `workman2021diewafer`. Report them as "no longer cited" (Dirk decides).
- Section files: 01 introduction, 02 identification problem and physics baseline, 03 augmentation, 04 orthogonal
  construction, 05 experiment design, 06 results and discussion, 99 appendix.

## 5. Assumed but not verified
- That the table above is complete. Re-extract the keys yourself from the current files (Dirk edits the text in
  parallel); add any new key and mark it "new since 2026-10-20".
- That a PDF named `<key>.pdf` is the version the bib entry describes (arXiv versus published). Settle from the PDF's
  first page (journal header, DOI, arXiv stamp).
- Which arXiv-only entries now have a published version (`drenth2025lpvlfr`, `hoekstra2026lfr`, `hoekstra2026encoder`,
  `raissi2017physicsinformedpart1`). Settle from the publisher, the proceedings (IFAC-PapersOnLine, Automatica), or
  the authors' pages; for `raissi2017` note that the later journal paper is a different, combined work, so it is not
  simply the published version of Part I.
- Years and pages for the NeurIPS entries and the L4DC entry; settle from the proceedings pages.

## 6. Tried and failed
- None for this task. Known pitfall: equation and section numbers differ between versions of the same paper (Drenth:
  Eq. (9) in the IFAC version, (5) in the public arXiv preprint, (2.4) in his thesis). Never carry a pinpoint from one
  version to another without opening the other version.

## 7. Achieved
- Nothing yet for this task. `docs/references.md` holds earlier per-key notes (location, what was verified); use them as
  leads, not as proof.

## 8. The open question
Nothing blocked. For each arXiv entry, the report states one of: "published version exists (cite it, link)", or "no
published version found (cite as preprint, submitted to <venue>)".

## 9. Next action
Re-extract every citation key and pinpoint from the current `.tex` files (outside comments; flag those inside `\del{}`,
`\todo{}` or `\revnote{}`), then work through the keys in the order of the table above, filling one report row per key
from the source's own page and PDF.

## 10. Acceptance criterion
- Every key cited in the text has a row with all six items of section 1 filled, each bibliographic field with its
  source (publisher page, DOI record, proceedings page, or the PDF's first page).
- Every pinpoint used in the text is marked "exists in the cited version" or "differs / missing", with the version
  checked.
- The "to download" list has, for each item, the version and an official link (DOI resolver `https://doi.org/<doi>`,
  the publisher's article page, the proceedings page such as `proceedings.mlr.press` or `papers.nips.cc`, or the
  university repository for theses). No links to unofficial mirrors.

## 11. Read these first
1. `Thesis-writeup/Writing/refs.bib`: the current entries.
2. `docs/references.md`: key-to-file map and earlier verification notes.
3. `Thesis-writeup/Writing/util/format.tex`: what `\cite`, `\ucite`, `\del`, `\mv`, `\todo`, `\revnote` do, so marked
   or struck citations are classified correctly.

## 12. Do not
- Do not edit `.tex` files or `refs.bib`; do not move or delete PDFs.
- Do not take a bibliographic field from memory or from a search snippet; take it from the source's page or PDF.
- Do not link unofficial copies (sci-hub, personal reuploads); use the official page, or note "no official free copy,
  TU/e access needed".
- No git write operations.
- In the report, use no em-dashes (the Unicode character or a double hyphen).

## 13. Operational
- Python: `conda run -n GraduationProject python ...` (PyMuPDF 1.27 for reading PDF first pages).
- Output: `Thesis-writeup/Documentation/REFERENCE-CHECK-2026-10-20.md`, one table plus the two lists of section 1.
- Web access is needed for the publisher and proceedings pages; where it is unavailable, mark the row "not checked
  online" rather than filling the field.

## 14. Delegation
None: a list of about 21 keys, worked one by one.
