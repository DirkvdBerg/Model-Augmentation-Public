from pathlib import Path
from collections import defaultdict,Counter
import json,re,hashlib,html
base=Path('.codex-tmp/reference-check-20261020')
citations=json.loads((base/'citations.json').read_text())
order=['beintema2023subnet','downey2017psrnn','drenth2025lpvlfr','forssell1999revisited','garcia2013model','gautier1990minimum','gyorok2025l4dc','gyorok2026obc','hefny2015supervised','hoekstra2026encoder','hoekstra2026lfr','karniadakis2021physicsinformed','kessels2025ai','noel2018greybox','ovchinnikov2021computing','raissi2017physicsinformedpart1','retzler2024','teo2007dynamic','toth2010modeling','venkatraman2017psd','willard2020integrating']
doi_by_key={
'beintema2023subnet':'10.1016/j.automatica.2023.111210','forssell1999revisited':'10.1016/S0005-1098(99)00022-9','garcia2013model':'10.1016/j.conengprac.2012.10.010','gautier1990minimum':'10.1109/70.56655','gyorok2026obc':'10.1016/j.ifacsc.2026.100376','karniadakis2021physicsinformed':'10.1038/s42254-021-00314-5','noel2018greybox':'10.1080/00207179.2017.1308557','ovchinnikov2021computing':'10.1016/j.sysconle.2021.105030','retzler2024':'10.1017/dce.2024.8','teo2007dynamic':'10.1016/j.mechatronics.2007.04.004','willard2020integrating':'10.1145/3514228'}
def link(label,url):return f'[{label}]({url})'
def cr(doi):return link('publisher-deposited DOI record','https://api.crossref.org/works/'+doi.replace('/','%2F'))
records={}; sources={}; official={}; decision={}; pdf={}; change={}; detail={}
for key,doi in doi_by_key.items():
 m=json.loads((base/('crossref-'+doi.replace('/','_')+'.json')).read_text(encoding='utf-8'))
 authors='; '.join((a.get('given','')+' '+a['family']).strip() for a in m['author'])
 year=m.get('published-print',m['published'])['date-parts'][0][0]
 venue=html.unescape(m['container-title'][0]);vol=m['volume'];issue='('+m['issue']+')' if m.get('issue') else ''
 pages=m.get('article-number') or m.get('page');pages=pages.replace('-','–')
 records[key]=f'{authors}. “{m["title"][0]}.” {venue} {vol}{issue}, {pages} ({year}). DOI: {doi}.'
 sources[key]=cr(doi)+' (authors, title, journal, volume, issue if assigned, pagination/article number, year and DOI).'
 official[key]='https://doi.org/'+doi
 decision[key]='Cite the published journal version.'
 change[key]='No bibliographic correction required.'
 detail[key]=''
hashes={'downey2017psrnn':('2017','2bb0502c80b7432eee4c5847a5fd077b'),'hefny2015supervised':('2015','9a3d458322d70046f63dfd8b0153ece4'),'venkatraman2017psd':('2017','61b4a64be663682e8cb037d9719ad8cd')}
for key,(year,h) in hashes.items():
 official[key]=f'https://proceedings.neurips.cc/paper_files/paper/{year}/hash/{h}-Abstract.html'
 sources[key]=link('proceedings page',official[key])+' (title, volume, year); '+link('official metadata',f'https://proceedings.neurips.cc/paper_files/paper/{year}/file/{h}-Metadata.json')+' (page range); '+link('published paper',f'https://proceedings.neurips.cc/paper_files/paper/{year}/file/{h}-Paper.pdf')+' p. 1 (author order and names).'
 decision[key]='Cite the published NeurIPS proceedings paper. No DOI is supplied in the official proceedings record inspected.'
records['downey2017psrnn']='Carlton Downey; Ahmed Hefny; Boyue Li; Byron Boots; Geoff Gordon. “Predictive State Recurrent Neural Networks.” Advances in Neural Information Processing Systems 30, 6053–6064 (2017). DOI: none listed.'
records['hefny2015supervised']='Ahmed Hefny; Carlton Downey; Geoffrey J. Gordon. “Supervised Learning for Dynamical System Learning.” Advances in Neural Information Processing Systems 28, 1963–1971 (2015). DOI: none listed.'
records['venkatraman2017psd']='Arun Venkatraman; Nicholas Rhinehart; Wen Sun; Lerrel Pinto; Martial Hebert; Byron Boots; Kris M. Kitani; J. Andrew Bagnell. “Predictive-State Decoders: Encoding the Future into Recurrent Networks.” Advances in Neural Information Processing Systems 30, 1172–1183 (2017). DOI: none listed.'
change['downey2017psrnn']='Add pages = 6053–6064. Author-order conflict: retain the PDF order currently in refs.bib; the proceedings landing page and metadata move Li to last. Do not silently reorder from that metadata. The paper prints Geoff Gordon; the landing page expands this to Geoffrey J. Gordon.'
change['hefny2015supervised']='Add pages = 1963–1971. Year 2015 and volume 28 are already present and correct.'
change['venkatraman2017psd']='Add pages = 1172–1183 and an official proceedings URL if retained. Author order in the current entry matches the paper.'
records['gyorok2025l4dc']='Bendegúz M. Györök; Jan H. Hoekstra; Johan Kon; Tamás Péni; Maarten Schoukens; Roland Tóth. “Orthogonal projection-based regularization for efficient model augmentation.” Proceedings of the 7th Annual Learning for Dynamics & Control Conference, Proceedings of Machine Learning Research 283, 166–178 (2025). DOI: none listed.'
official['gyorok2025l4dc']='https://proceedings.mlr.press/v283/gyorok25a.html'
sources['gyorok2025l4dc']=link('PMLR paper and citation record',official['gyorok2025l4dc'])+' (all publication fields); '+link('official proceedings PDF','https://raw.githubusercontent.com/mlresearch/v283/main/assets/gyorok25a/gyorok25a.pdf')+' p. 1 (accented author names).'
decision['gyorok2025l4dc']='Cite the published L4DC 2025 paper. The PDF uses internal pages 1–13; PMLR assigns proceedings pages 166–178.'
change['gyorok2025l4dc']='Set booktitle to the exact PMLR proceedings title; pages, volume, year and editors are already correct. Optional: add publisher = PMLR and the official URL.'
for key,arxiv,title,authors,venue in [
('drenth2025lpvlfr','2605.12203','Efficient Learning of Affine and Rational Dependency LPV Models With Linear Fractional Representation','Roel Drenth; Jan H. Hoekstra; Maarten Schoukens; Roland Tóth','Accepted for IFAC World Congress 2026'),
('hoekstra2026encoder','2602.13108','Encoder initialisation methods in the model augmentation setting','J. H. Hoekstra; B. Györök; R. Tóth; M. Schoukens','Submitted to IFAC World Congress 2026'),
('hoekstra2026lfr','2602.17297','Learning-based augmentation of first-principle models: A linear fractional representation-based approach','Jan H. Hoekstra; Bendegúz M. Györök; Roland Tóth; Maarten Schoukens','Submitted to Automatica'),
('raissi2017physicsinformedpart1','1711.10561','Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations','Maziar Raissi; Paris Perdikaris; George Em Karniadakis','Submission venue not established')]:
 year='2017' if arxiv.startswith('17') else '2026'
 records[key]=f'{authors}. “{title}.” arXiv:{arxiv}v1, preprint ({year}). {venue}. DOI: 10.48550/arXiv.{arxiv}. No journal volume or pagination assigned.'
 official[key]='https://arxiv.org/pdf/'+arxiv+'v1'
 sources[key]=link('arXiv record','https://arxiv.org/abs/'+arxiv)+' (title, authors, date, DOI and submission comments); PDF p. 1 (author typography).'
 decision[key]='No published version found in the official/author sources searched on 2026-10-20. Cite the arXiv v1 preprint, with the stated venue status.'
 change[key]='No bibliographic correction required.'
sources['hoekstra2026encoder']+=' '+link('author-maintained repository','https://github.com/JanHHoekstra/Model-Augmentation-Public')+' also lists this as submitted to IFAC WC 2026.'
sources['hoekstra2026lfr']+=' '+link('author-maintained repository','https://github.com/JanHHoekstra/Model-Augmentation-Public')+' also lists this as submitted to Automatica.'
change['drenth2025lpvlfr']='Add note = Preprint; accepted for IFAC World Congress 2026. Keep year 2026 and arXiv DOI. Do not invent an IFAC-PapersOnLine volume, pages or DOI.'
change['hoekstra2026encoder']='Add note = Preprint; submitted to IFAC World Congress 2026. Existing author accents match the PDF; arXiv metadata misspells Tóth as Töth.'
change['raissi2017physicsinformedpart1']='Recommended preprint encoding: change type article to misc, replace journal with eprint = 1711.10561 and archivePrefix = arXiv, add note = Preprint. Retain 2017 and the arXiv DOI. No submission venue verified.'
decision['raissi2017physicsinformedpart1']='No standalone published Part I version found. Cite this preprint. The later combined journal work has a different title and scope; it is not substituted under this key.'
records['kessels2025ai']='Bas Mats Kessels. AI-based Model Updating for Nonlinear Dynamical Systems. PhD thesis, Eindhoven University of Technology, Eindhoven (2025), 246 numbered pages. ISBN 978-90-386-6218-3. DOI: none listed in the university record.'
official['kessels2025ai']='https://research.tue.nl/en/publications/ai-based-model-updating-for-nonlinear-dynamical-systems/'
sources['kessels2025ai']=link('TU/e thesis record',official['kessels2025ai'])+' (author, title, university, date, extent and ISBN); local PDF repository cover and title pages.'
decision['kessels2025ai']='Cite the published thesis itself, which contains the thesis-specific remark and equation numbering. A related journal article does not replace the whole thesis.'
change['kessels2025ai']='No required correction. Optional: add ISBN and the university repository URL.'
records['toth2010modeling']='Roland Tóth. Modeling and Identification of Linear Parameter-Varying Systems. Lecture Notes in Control and Information Sciences 403, first edition, Springer Berlin, Heidelberg (2010), 325 pages. DOI: 10.1007/978-3-642-13812-6.'
official['toth2010modeling']='https://link.springer.com/book/10.1007/978-3-642-13812-6'
sources['toth2010modeling']=link('Springer bibliographic record',official['toth2010modeling'])+' (author, title, edition, publisher, year, extent, DOI); local PDF title/copyright pages 2–4 (series volume 403).'
decision['toth2010modeling']='Cite the published 2010 book.'
change['toth2010modeling']='No bibliographic correction required.'
change['forssell1999revisited']='If retained: replace the 1998 technical report entry with the 1999 Automatica article. Change type, title, year and month; replace institution/report number with journal, volume, issue, pages and DOI. Only a deleted citation currently uses this key.'
change['willard2020integrating']='For the final numbered issue use year = 2023 and month = apr. Crossref records online publication as 2022-11-21 and issue publication as 2023-04-30; the current 2022/nov is the online date, not a fabricated date. Article number 66 is in the current bib but could not be independently verified from the accessible official record; keep it flagged for the final PDF.'
records['willard2020integrating']+=' Issue date: April 2023; first online 21 November 2022. Current bib article number 66 remains unverified.'
decision['willard2020integrating']='Cite the published ACM journal article with the final issue date (2023), rather than the local 2022 arXiv manuscript.'
for key in order:
 detail.setdefault(key,'')
present={p.stem for p in Path('literature/thesis-referenced').glob('*.pdf')}
matching={'beintema2023subnet','hoekstra2026encoder','hoekstra2026lfr','karniadakis2021physicsinformed','kessels2025ai','raissi2017physicsinformedpart1','retzler2024','teo2007dynamic','toth2010modeling'}
pdf.update({
'beintema2023subnet':'YES: publisher PDF with TU/e cover, 14 PDF pages; Automatica header and DOI on PDF p. 2.',
'downey2017psrnn':'NO: local PDF is arXiv:1705.09353v2, 14 pages; published proceedings PDF has 12 pages.',
'drenth2025lpvlfr':'NO: local 7-page unversioned IFAC-layout manuscript has no publisher volume, DOI or arXiv stamp; official arXiv v1 has 6 pages. Both local Drenth article copies have the same SHA-256.',
'forssell1999revisited':'NO: local PDF is the 1998 LiTH-ISY-R-2021 updated technical report, 55 pages. Published article is 1999.',
'garcia2013model':'NO: local 29-page SAM author manuscript, not the 10-page journal layout. The alternative gantry copy is also this manuscript.',
'gautier1990minimum':'NO: no key PDF in thesis-referenced.',
'gyorok2025l4dc':'NO: no key PDF in thesis-referenced. Official proceedings PDF was inspected online only.',
'gyorok2026obc':'NO: local 10-page arXiv:2511.01321v3. Published PDF was inspected through TU/e and has 11 PDF pages including its repository cover. The alternative Orthogonality copy is also the preprint.',
'hefny2015supervised':'NO: local 20-page arXiv:1505.05310v2. The published proceedings PDF has 9 pages. The additional local copy inspected is also the preprint.',
'hoekstra2026encoder':'YES: arXiv:2602.13108v1, stamp on PDF p. 1, 7 pages.',
'hoekstra2026lfr':'YES: arXiv:2602.17297v1, stamp on PDF p. 1, 16 pages.',
'karniadakis2021physicsinformed':'YES: 19-page Nature Reviews Physics typeset article with matching DOI and journal pagination.',
'kessels2025ai':'YES: TU/e published thesis, 275 PDF pages including cover and unnumbered material; repository cover identifies the final published version.',
'noel2018greybox':'NO: local 30-page arXiv:1804.10758v1 manuscript, with placeholder journal volume 00 and pages 1–30.',
'ovchinnikov2021computing':'NO: no key PDF in thesis-referenced.',
'raissi2017physicsinformedpart1':'YES: arXiv:1711.10561v1, 22 pages, stamp on PDF p. 1.',
'retzler2024':'YES: publisher PDF with TU/e cover, 18 pages, journal DOI and e12 page numbering.',
'teo2007dynamic':'YES: publisher PDF, 7 pages, Mechatronics 17 (2007) 361–367 and matching DOI.',
'toth2010modeling':'YES: published Springer book, 337 PDF pages including front matter; title and copyright pages match.',
'venkatraman2017psd':'NO: no key PDF in thesis-referenced. Official proceedings PDF inspected online only.',
'willard2020integrating':'NO: local 35-page arXiv:2003.04919v6, dated 14 March 2022; first page still contains a template DOI and 2020 reference-format placeholders.',
})
detail['downey2017psrnn']='Published PDF Sec. 4.2 “Learning PSRNNs” exists on PDF p. 5. Author order is taken from the published PDF because the proceedings web metadata disagrees.'
detail['drenth2025lpvlfr']='Official arXiv v1: Eq. (2), p. 2; Definition 1, p. 2; Theorem 6, p. 3; Sec. 4.1 “Parameter Estimation”, p. 3. All requested numbers exist. Version difference: the local manuscript calls Sec. 4.1 “Well-posedness conditions”; that material is in Sec. 3 of arXiv v1. The eliminated-variable equation is local (9), arXiv (5). Do not transfer the local section meaning when changing PDF.'
detail['beintema2023subnet']='Published PDF: Sec. 3.1 at PDF p. 4; Sec. 3.2 and Eq. (9a–c) at p. 5; Condition 1 at p. 7; Eq. (43) at p. 10. Bare Eq. (9) denotes the displayed group.'
detail['gyorok2025l4dc']='Published PMLR PDF p. 7 contains Sec. 4, Eqs. (15), (16), (18), and Remark 2. All requested numbers exist.'
detail['gyorok2026obc']='Published PDF (includes TU/e cover): Eqs. (1), (2), (5), p. 3; Sec. 3.1, Eqs. (6), (8)–(11), (13), Assumption 1 and Lemma 3, p. 4; Eqs. (14), (15), (21), Condition 4, Remark 5 and Assumption 6, p. 5; Sec. 4.3, p. 6; Eq. (39), p. 7. All requested numbers exist in the journal version.'
detail['hefny2015supervised']='Published proceedings PDF: Sec. 2 starts on p. 2 and continues on pp. 3–4; Fig. 2 exists on p. 2. All requested numbers exist.'
detail['hoekstra2026encoder']='arXiv v1: footnote 1 on PDF p. 2; Eq. (8) and Sec. 3.1 on p. 3; Sec. 3.2 and Eqs. (33), (35) on p. 5. All requested numbers exist.'
detail['hoekstra2026lfr']='arXiv v1: Sec. 2 on PDF p. 2; Table 1 on p. 3; Conditions 6, 8 and Theorem 9 on p. 7; Eq. (22a–e) and Sec. 5.2 on p. 8; Secs. 5.3, 5.4 and Eqs. (29)–(31) on p. 9; Sec. 5.5 on p. 10. All requested numbers exist; Eq. (22e) is part of (22).'
detail['kessels2025ai']='Published thesis: Remark 5.1 at PDF p. 179; Eq. (5.12) at p. 183; Eq. (5.13a–d) spans pp. 183–184; Remarks 5.3 and 5.4 at p. 184; Sec. 5.3.1.3 at p. 191. All requested numbers exist.'
detail['venkatraman2017psd']='Published proceedings PDF Sec. 3 “Predictive-State Decoders” exists on PDF p. 5. Citation occurs only inside a todo.'
detail['garcia2013model']='PENDING: not checked in the published full text because publisher PDF access returned 403. Sec. 2.2 and Table I exist in the local author manuscript, but this does not prove the journal numbering. Five pinpoint occurrences require verification after the journal PDF is obtained.'
detail['ovchinnikov2021computing']='PENDING: Definition 7 not checked in the journal version. Publisher PDF returned 403 and full-text API returned 401. Publication metadata was verified; a preprint definition number is not accepted as proof of the journal number.'

def location(r):return f"`{r['file']}:{r['line']}`"
def verdict(key,pin):
 if not pin:return 'No pinpoint requested.'
 if key in ('garcia2013model','ovchinnikov2021computing'):return '**PENDING: cited-version full text inaccessible.**'
 if key=='drenth2025lpvlfr' and '4.1' in pin:return '**Exists in arXiv v1; section title/content differs from local manuscript.**'
 return 'Exists in the selected cited version.'
def citegroups(key,ancillary=False):
 rr=[r for r in citations if r['key']==key and bool(r['flags'])==ancillary]
 groups=defaultdict(list)
 for r in rr:groups[(r['pinpoint'],r['marker'],','.join(r['flags']))].append(r)
 items=[]
 for (pin,marker,flags),rows in groups.items():
  where=', '.join(location(r) for r in rows)
  label=f'`{marker}`'+(f' [{flags}]' if flags else '')
  if pin:items.append(f'{label} `[{pin}]`: {where}. {verdict(key,pin)}')
  else:items.append(f'{label}, no pinpoint: {where}.')
 return '<br>'.join(items) if items else 'None in main text; see the marked-text group below.'

intro='''# Thesis reference check, 2026-10-20

Report-only audit of `Thesis-writeup/Writing/sections/*.tex`, `main.tex`, and `frontmatter/*.tex`. No thesis text, bibliography, PDFs, `docs/`, or `tasks/` were edited. No git write operations were used. Official PDF bytes were inspected online without saving PDF files.

## Result and limits

The current extraction finds **21 keys, 141 key occurrences, and 80 citation occurrences with pinpoints**. These are the same 21 keys named in the handoff; no new key was found. Of the 141 occurrences, 119 are in main text, 7 are inside todo, 12 are in deleted or moved-away text, and 3 are inside wip. No citation was found inside revnote, main.tex, or frontmatter. Comments are excluded. Multi-key citations count once per key.

**74 of the 80 pinpoint occurrences were checked in the selected version and their numbers exist. Six occurrences remain pending:** five for García and one for Ovchinnikov. The published full texts for those two papers were inaccessible through the available publisher routes. These are not claimed as either verified matches or missing numbers. Consequently the handoff's full pinpoint acceptance criterion remains open for these two sources. Bibliographic records for all 21 keys were inspected in primary records, DOI metadata deposited by the publishers, or source PDFs.

**Nine selected versions are already in `literature/thesis-referenced/`; twelve are absent or represented by a different version.** Two of the twelve are conditional: Forssell appears only in deleted text, and Venkatraman only in a todo. The final lists identify exact versions and official acquisition links.

The version rule used here is the published journal/proceedings version when available, otherwise the identified arXiv preprint. For Drenth, arXiv states acceptance for IFAC WC 2026, but no published proceedings record was found. Acceptance and an IFAC-style layout do not establish a version of record. The provisional citation is therefore the public arXiv v1, explicitly labelled preprint and accepted. Both Hoekstra works remain preprints in the sources inspected. A negative publication search means “no published version found”, not proof that publication cannot exist.

PDF page numbers below are one-based viewer page indices, including university repository cover sheets. They are distinct from printed journal, proceedings, or thesis page numbers. “Exists” checks the numbered object only; it does not judge whether that object supports the citing sentence.

Extraction respects balanced macro arguments. `del`, `delmv`, and the old argument of `chg` are marked as removed; `mv` retains its moved-in text. Todo citations are grouped separately because they may be deleted. Wip citations remain part of the editor source but are excluded in the review build. Unchecked `ucite` citations are orange in the editor and suppressed in the review build, as defined in `util/format.tex`.

The handoff table was stale in several fields: the current bib already contains the relevant journal DOIs and NeurIPS years. Corrections below compare the actual current entries, rather than proposing those additions again. Existing keys containing earlier years are retained as identifiers.

## Per-key audit

The DOI-record source attached to a record verifies all bibliographic fields listed in that record, except where another source or an explicit limitation is named. A DOI record can verify metadata even when the publisher's web page blocks full-text access. No field was inferred from a search snippet. Journal issue numbers are omitted when the inspected record assigns none. For preprints, journal volume/pages are not applicable. For the book and thesis, extent replaces article pagination.

All local PDF names in the table refer to `literature/thesis-referenced/<key>.pdf`. Main-text citation locations below are relative to `Thesis-writeup/Writing/`. Citations in notes, deleted text, and wip are listed separately after the table.

| Key | Checked bibliographic record and evidence | Version to cite | Exact cited PDF present; official link if absent | Main-text citations and pinpoint existence | refs.bib action |
|-|-|-|-|-|-|
'''
lines=[intro]
for k in order:
 pin=citegroups(k)
 if detail[k]:pin+='<br>**Version checked:** '+detail[k]
 acquisition='' if k in matching else '<br>'+link('Official version / download route',official[k])
 row=[f'`{k}`',records[k]+'<br>'+sources[k],decision[k],pdf[k]+acquisition,pin,change[k]]
 lines.append('| '+' | '.join(x.replace('|','&#124;').replace('\n',' ') for x in row)+' |\n')
lines.append('\n## Citations inside notes or removed text\n\nThese occurrences are kept separate from the main-text audit. Verdicts use the same selected versions as the table.\n')
for category,title in [('todo','Todo notes'),('removed','Deleted or moved-away text'),('wip','Wip text')]:
 lines.append('\n### '+title+'\n\n')
 for r in citations:
  selected= ('todo' in r['flags']) if category=='todo' else (any(f in ('del','delmv','chg') for f in r['flags']) if category=='removed' else 'wip' in r['flags'])
  if selected:
   pin=' `['+r['pinpoint']+']`' if r['pinpoint'] else ', no pinpoint'
   lines.append(f"- `{r['key']}`: {location(r)}, `{r['marker']}`, {', '.join(r['flags'])}{pin}. {verdict(r['key'],r['pinpoint'])}\n")
lines.append('''
## Version discrepancies and remaining checks

- **Drenth:** the two local 7-page article files are identical to each other. Neither is the public 6-page arXiv v1, and neither has bibliographic evidence establishing a publisher version. All cited numbers exist in arXiv v1, including Sec. 4.1, but the section title changed. The local well-posedness discussion is in Sec. 3 of arXiv v1. The cited Sec. 4.1 occurrence at `sections/03_augmentation.tex:887` needs Dirk's review when swapping versions. This records the numbering difference without making a claim-support judgement.
- **Downey:** the published PDF lists Downey, Hefny, Li, Boots, Gordon; its proceedings landing page and metadata list Downey, Hefny, Boots, Gordon, Li. The current author order agrees with the paper. Retain it and keep this discrepancy visible; there is no justified automatic author-order correction.
- **Willard:** the publisher-deposited DOI record explicitly distinguishes 2022-11-21 online publication from 2023-04-30 publication of volume 55 issue 4. This report chooses the final issue date, 2023/April, consistently with the cited volume and issue. The present 2022/November describes the real online date. The article number 66 could not be independently checked because the ACM page/PDF and citation export returned 403; it is not promoted to a verified field. Authors, title, journal, volume, issue, page range and both dates were read from the DOI record.
- **García:** publication metadata is checked, but Sec. 2.2 and Table I remain unverified in the final journal version. The existing local manuscript contains both labels. Do not certify the journal labels from that manuscript. The affected main-text locations are `sections/02_system_baseline.tex:305`, `sections/02_system_baseline.tex:362`, `sections/05_setup.tex:146`, `sections/99_appendix.tex:22`; the fifth occurrence is the todo at `sections/02_system_baseline.tex:350`.
- **Ovchinnikov:** verify Definition 7 in the published article after obtaining its PDF. The prior arXiv check recorded elsewhere in this repository is not evidence for the journal numbering. The affected location is `sections/02_system_baseline.tex:553`.
- **Preprint publication searches:** Drenth and encoder exact-title searches on publisher/proceedings sources and Crossref returned no matching published record. Their arXiv records were read, including the IFAC acceptance/submission comments. Hoekstra's author-maintained repository still identifies the LFR work as submitted to Automatica and the encoder work as submitted to IFAC WC 2026. Recheck these publication states when the thesis is finalized.
- **Raissi Part I:** no standalone published version was found; keep its preprint identity. Do not replace it with the later, differently titled combined forward/inverse-problems article under the same key. No “submitted to” venue was verified for Part I.
- Publisher browsing/full-text routes for some journal papers were unavailable. DOI metadata was recovered directly from Crossref for all cited journal articles. Published Györök OBC was read from the university-hosted publisher PDF; L4DC and the three NeurIPS papers were read from official proceedings PDFs. The inaccessible García/Ovchinnikov full texts and Willard article number remain explicitly unresolved.

## PDFs no longer cited

These filenames occur in `literature/thesis-referenced/` but their keys do not occur in the extracted citation set. No removal is proposed automatically.

- `drenth2025thesis.pdf`
- `hwang2024semiconductor.pdf`
- `irds2022.pdf`
- `lambrechts2005trajectory.pdf`
- `workman2021diewafer.pdf`

Forssell is a separate case: it is still cited in the source, but only inside `del`. Venkatraman remains in `ucite` inside todo. These two keys are not counted as no longer cited.

## To download

Acquire these exact versions into `literature/thesis-referenced/` under their citation keys. No PDFs were downloaded to disk, replaced, moved or removed during this audit. An official landing page is supplied where institutional access or a publisher download button is needed. Availability to the auditing tools does not determine availability through TU/e.

''')
for k in order:
 if k in matching:continue
 version={
 'downey2017psrnn':'NeurIPS 2017 published proceedings PDF, 12 pages',
 'drenth2025lpvlfr':'arXiv:2605.12203v1, 6-page preprint, accepted for IFAC WC 2026; replace the unversioned 7-page local manuscript unless an official published version is located first',
 'forssell1999revisited':'Automatica 35(7), 1215–1241 (1999), version of record; conditional on retaining the currently deleted citation',
 'garcia2013model':'Control Engineering Practice 21(3), 298–307 (2013), version of record; publisher access through TU/e may be needed',
 'gautier1990minimum':'IEEE Transactions on Robotics and Automation 6(3), 368–373 (1990), version of record; publisher access through TU/e may be needed',
 'gyorok2025l4dc':'published L4DC 2025 / PMLR 283 proceedings PDF, internally numbered 1–13, assigned proceedings pages 166–178',
 'gyorok2026obc':'IFAC Journal of Systems and Control 35, 100376 (2026), version of record; free university-hosted publisher PDF',
 'hefny2015supervised':'NeurIPS 2015 published proceedings PDF, 9 pages',
 'noel2018greybox':'International Journal of Control 91(5), 1118–1139 (2018), version of record; publisher access through TU/e may be needed',
 'ovchinnikov2021computing':'Systems & Control Letters 157, 105030 (2021), version of record; publisher marks it open access, although the audit tool could not open its PDF',
 'venkatraman2017psd':'NeurIPS 2017 published proceedings PDF, 12 pages; conditional on retaining the todo citation',
 'willard2020integrating':'ACM Computing Surveys 55(4), 1–37, final numbered issue (April 2023), DOI 10.1145/3514228; confirm article number 66 on the final PDF',
 }[k]
 url=official[k]
 if k in hashes:
  y,h=hashes[k];url=f'https://proceedings.neurips.cc/paper_files/paper/{y}/file/{h}-Paper.pdf'
 if k=='gyorok2026obc':url='https://pure.tue.nl/ws/files/381189611/1-s2.0-S2468601826000167-main.pdf'
 lines.append(f'- `{k}`: {version}. '+link('Official link',url)+'.\n')
lines.append('''
## Bib corrections

Required bibliographic changes and optional record improvements are distinguished. Values shown as page ranges use an en dash in this report; encode ranges in BibTeX using its normal two-hyphen range convention. Existing DOI and year fields that are already correct should not be added again. Citation keys remain unchanged.

''')
corrections=[
('downey2017psrnn','pages','absent','6053–6064',sources['downey2017psrnn']),
('hefny2015supervised','pages','absent','1963–1971',sources['hefny2015supervised']),
('venkatraman2017psd','pages','absent','1172–1183, if the todo citation is retained',sources['venkatraman2017psd']),
('gyorok2025l4dc','booktitle','Proceedings of the 7th Annual Conference on Learning for Dynamics and Control',r'Proceedings of the 7th Annual Learning for Dynamics \& Control Conference',link('PMLR official citation',official['gyorok2025l4dc'])),
('drenth2025lpvlfr','note','absent','Preprint; accepted for IFAC World Congress 2026',link('arXiv comments','https://arxiv.org/abs/2605.12203')),
('hoekstra2026encoder','note','absent','Preprint; submitted to IFAC World Congress 2026',link('arXiv comments','https://arxiv.org/abs/2602.13108')),
('willard2020integrating','year','2022','2023, when citing the final numbered issue',cr('10.1145/3514228')+'; published-print = 2023-04-30, published-online = 2022-11-21'),
('willard2020integrating','month','nov','apr, paired with final issue year 2023',cr('10.1145/3514228')),
('forssell1999revisited','entry type','techreport','article, conditional on retaining the deleted citation',cr('10.1016/S0005-1098(99)00022-9')),
('forssell1999revisited','title','Closed-loop Identification Revisited: Updated Version','Closed-loop identification revisited',cr('10.1016/S0005-1098(99)00022-9')),
('forssell1999revisited','year','1998','1999',cr('10.1016/S0005-1098(99)00022-9')),
('forssell1999revisited','month','apr','jul, or omit month',cr('10.1016/S0005-1098(99)00022-9')),
('forssell1999revisited','journal','absent','Automatica',cr('10.1016/S0005-1098(99)00022-9')),
('forssell1999revisited','volume','absent','35',cr('10.1016/S0005-1098(99)00022-9')),
('forssell1999revisited','number','LiTH-ISY-R-2021','7',cr('10.1016/S0005-1098(99)00022-9')),
('forssell1999revisited','pages','absent','1215–1241',cr('10.1016/S0005-1098(99)00022-9')),
('forssell1999revisited','doi','absent','10.1016/S0005-1098(99)00022-9',cr('10.1016/S0005-1098(99)00022-9')),
('forssell1999revisited','institution',r'Department of Electrical Engineering, Link{\"o}ping University','remove from the journal article entry',cr('10.1016/S0005-1098(99)00022-9')),
('raissi2017physicsinformedpart1','entry type','article','misc, recommended to represent the preprint explicitly',link('arXiv record','https://arxiv.org/abs/1711.10561')),
('raissi2017physicsinformedpart1','journal','arXiv preprint arXiv:1711.10561','remove if switching to misc; replace with eprint = 1711.10561 and archivePrefix = arXiv',link('arXiv record','https://arxiv.org/abs/1711.10561')),
('raissi2017physicsinformedpart1','note','absent','Preprint; do not supply an unverified submission venue',link('arXiv record','https://arxiv.org/abs/1711.10561')),
('kessels2025ai','isbn','absent','978-90-386-6218-3 (optional)',link('TU/e thesis record',official['kessels2025ai'])),
('kessels2025ai','url','absent',official['kessels2025ai']+' (optional)',link('TU/e thesis record',official['kessels2025ai'])),
('gyorok2025l4dc','publisher','absent','PMLR (optional)',link('PMLR official citation',official['gyorok2025l4dc'])),
]
for key,field,current,correct,source in corrections:
 lines.append(f'- `{key}` | `{field}` | current: `{current}` | correct: `{correct}` | source: {source}.\n')
lines.append('''
- `willard2020integrating` | `articleno` | current: `66` | correct: **unresolved; no replacement value asserted** | source needed: final ACM article or accessible publisher citation export. This is a verification item, not an instruction to delete 66.
- `downey2017psrnn` | `author` | current order: `Downey; Hefny; Li; Boots; Gordon` | correct action: retain this published-PDF order; official web metadata conflicts. The current expanded `Geoffrey J.` differs from the PDF's `Geoff`, while the proceedings page itself uses `Geoffrey J.`; no mandatory name expansion change is asserted.
- All other entries: no required bibliographic field correction identified. In particular, the current DOI fields for Beintema, García, Györök OBC, Noël, Retzler, Teo and Tóth, and the current NeurIPS years, already match the inspected primary records.
''')
report=''.join(lines)
assert '\u2014' not in report and '--' not in report
assert set(order)=={r['key'] for r in citations}
target=Path('Thesis-writeup/Documentation/REFERENCE-CHECK-2026-10-20.md')
target.write_text(report,encoding='utf-8')
print(str(target),len(report),'characters')
print('table rows',sum(line.startswith('| `') for line in report.splitlines()))
print('download rows',len(set(order)-matching))
print('pinpoint occurrences',sum(bool(r['pinpoint']) for r in citations),'pending',sum(bool(r['pinpoint']) and r['key'] in ('garcia2013model','ovchinnikov2021computing') for r in citations))
old=json.loads((base/'hashes.json').read_text())
print('source files changed since extraction',[f for f,h in old.items() if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h])
print('bib sha256',hashlib.sha256(Path('Thesis-writeup/Writing/refs.bib').read_bytes()).hexdigest())
