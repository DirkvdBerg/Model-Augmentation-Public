from pathlib import Path
from collections import defaultdict
import json,re,hashlib
root=Path.cwd()
evidence=Path('.codex-tmp/reference-check-20261020')
report=Path('Thesis-writeup/Documentation/REFERENCE-CHECK-2026-10-20.md')
s=report.read_text(encoding='utf-8')
c=json.loads((evidence/'citations.json').read_text())
manifest=json.loads((evidence/'new-local/manifest.json').read_text(encoding='utf-8'))
prefixes={'ovchinnikov2021computing':'Computing','gautier1990minimum':'Direct_','downey2017psrnn':'Downey','garcia2013model':'Garcia','gyorok2025l4dc':'gyorok25a','hefny2015supervised':'Hefny','noel2018greybox':'JP-Noël','venkatraman2017psd':'NIPS-2017','gyorok2026obc':'Orthogonal-by-construction','drenth2025lpvlfr':'Roel','forssell1999revisited':'Urban','willard2020integrating':'Williard'}
local={k:next(m for m in manifest if Path(m['file']).name.startswith(prefix)) for k,prefix in prefixes.items()}
def loc(r):return f"`{r['file']}:{r['line']}`"
def verdict(k,pin):
 if not pin:return 'No pinpoint requested.'
 if k=='drenth2025lpvlfr' and '4.1' in pin:return '**Exists in arXiv v1; section title/content differs from the old local manuscript.**'
 return 'Exists in the selected cited version.'
def citegroups(k):
 groups=defaultdict(list)
 for r in c:
  if r['key']==k and not r['flags']:groups[(r['pinpoint'],r['marker'])].append(r)
 parts=[]
 for (pin,marker),rr in groups.items():
  label=f'`{marker}`'
  if pin:parts.append(f'{label} `[{pin}]`: '+', '.join(loc(r) for r in rr)+'. '+verdict(k,pin))
  else:parts.append(f'{label}, no pinpoint: '+', '.join(loc(r) for r in rr)+'.')
 return '<br>'.join(parts) if parts else 'None in main text; see the marked-text group below.'
def pdf_link(m):return f"[Downloaded PDF](<../../{m['file']}>)"
details={
'garcia2013model':'Downloaded publisher PDF, 10 pages, matching journal DOI: Sec. 2.2 “Dynamic modeling” exists on PDF p. 2 (printed p. 299); Table I exists on PDF p. 4 (printed p. 301). All five cited pinpoint occurrences are now verified, including the todo. The body refers to “Table 1”, but the table caption itself is “Table I”.',
'ovchinnikov2021computing':'Downloaded publisher PDF, 13 pages, matching journal DOI: Definition 7 “Identifiability” exists on PDF p. 3 (printed p. 3). The formerly pending pinpoint is verified.',
'gyorok2026obc':'Downloaded publisher PDF without a repository cover, 10 pages: Eqs. (1), (2), (5), p. 2; Sec. 3.1, Eqs. (6), (8)–(11), (13), Assumption 1 and Lemma 3, p. 3; Eqs. (14), (15), (21), Condition 4, Remark 5 and Assumption 6, p. 4; Sec. 4.3, p. 5; Eq. (39), p. 6. All requested numbers exist in this journal version.',
}
for line in s.splitlines():
 if not line.startswith('| `'):continue
 cols=line.split(' | ');k=cols[0].split('`')[1]
 old_detail=cols[4].split('<br>**Version checked:** ',1)
 detail=details.get(k,old_detail[1] if len(old_detail)>1 else '')
 cols[4]=citegroups(k)+('<br>**Version checked:** '+detail if detail else '')
 if k in local:
  m=local[k];kind='arXiv:2605.12203v1 preprint' if k=='drenth2025lpvlfr' else 'published proceedings PDF' if k in ('downey2017psrnn','hefny2015supervised','venkatraman2017psd','gyorok2025l4dc') else 'publisher journal PDF'
  cols[3]=f"YES, now local in `literature/thesis-reference-new/`: {pdf_link(m)}, {m['pages']} PDF pages, {kind}. The old thesis-referenced copy or absence is superseded for local availability; files have not been moved."
 if k=='willard2020integrating':
  cols[1]=cols[1].replace('1–37 (2023)','1–37 (2022)').replace('Issue date: April 2023; first online 21 November 2022. Current bib article number 66 remains unverified.','Article 66. The published PDF explicitly cites November 2022; retain that date. Crossref separately records an April 2023 issue date.')
  cols[1]+='<br>Downloaded published PDF p. 1 verifies Article 66 and its own November 2022 citation; printed footers repeat that publication date.'
  cols[2]='Cite the published ACM article using its own reference format: volume 55, issue 4, Article 66, November 2022, 37 pages.'
  cols[5]='No correction required to year = 2022, month = nov or articleno = 66. The earlier recommendation to change the date to 2023/April is withdrawn based on the published PDF. |'
 s=s.replace(line,' | '.join(cols))

# New keys appeared while the user revised the introduction.
workman=json.loads((evidence/'crossref-10.1109_ECTC32696.2021.00326.json').read_text(encoding='utf-8'))
authors='; '.join(a['given']+' '+a['family'] for a in workman['author'])
newrows=[
['irds2022','IEEE International Roadmap for Devices and Systems. “Executive Summary 2022.” Institute of Electrical and Electronics Engineers (2022). DOI: 10.60627/C13Z-V363. No journal volume or article pagination; local PDF has 86 viewer pages.', '[IEEE 2022 edition citation](https://irds.ieee.org/irds2022/) supplies corporate author, title, publisher, year and DOI. The local published report cover identifies the 2022 Executive Summary.','Cite the official published 2022 roadmap report. New since the original 21-key extraction.','YES: `literature/thesis-referenced/irds2022.pdf`, official 2022 Executive Summary; 86 PDF pages.','Add the official DOI. The current longer title identifies the same report; the exact IEEE recommended short title is Executive Summary 2022.'],
['workman2021diewafer',authors+'. “'+workman['title'][0]+'.” '+workman['container-title'][0]+', 2071–2077 (2021). DOI: 10.1109/ECTC32696.2021.00326.', '[Publisher-deposited DOI record](https://api.crossref.org/works/10.1109%2FECTC32696.2021.00326) supplies all publication fields; local manuscript p. 1 verifies authors/title.','Cite the published IEEE ECTC 2021 proceedings article. New since the original 21-key extraction.','NO: `literature/thesis-referenced/workman2021diewafer.pdf` is a 7-page manuscript whose first page still prints the template copyright XXX-X-XXXX-XXXX-X/XX/$XX.00 and ©20XX IEEE, without final DOI or proceedings pagination. It is not verified as the publisher version. [Official published-version route](https://doi.org/10.1109/ECTC32696.2021.00326).','No required bibliographic correction; the current entry already has all ten authors, pages, year and DOI. Obtain the published proceedings PDF.'],
]
rows=[]
for k,record,source,decision,pdf,action in newrows:
 rows.append('| '+' | '.join([f'`{k}`',record+'<br>'+source,decision,pdf,citegroups(k),action])+' |\n')
needle='\n## Citations inside notes or removed text'
assert needle in s
s=s.replace(needle,'\n'+''.join(rows)+needle,1)

# Refresh note/deletion groups from the same extraction as the table.
start=s.index('## Citations inside notes or removed text')
end=s.index('## Version discrepancies and remaining checks',start)
group='## Citations inside notes or removed text\n\nThese occurrences are kept separate from main text. All numbered pinpoints below were checked in the selected cited versions.\n'
for category,title in [('todo','Todo notes'),('removed','Deleted or moved-away text'),('wip','Wip text')]:
 group+='\n### '+title+'\n\n'
 for r in c:
  selected='todo' in r['flags'] if category=='todo' else any(f in ('del','delmv','chg') for f in r['flags']) if category=='removed' else 'wip' in r['flags']
  if selected:
   pin=' `['+r['pinpoint']+']`' if r['pinpoint'] else ', no pinpoint'
   group+=f"- `{r['key']}`: {loc(r)}, `{r['marker']}`, {', '.join(r['flags'])}{pin}. {verdict(r['key'],r['pinpoint'])}\n"
s=s[:start]+group+'\n'+s[end:]

start=s.index('## Version discrepancies and remaining checks')
end=s.index('## PDFs no longer cited',start)
section='''## Version discrepancies and remaining checks

- **New downloads checked:** 13 PDF files in `literature/thesis-reference-new/` represent 12 distinct papers. Both Gautier filenames have the same SHA-256. The downloaded PMLR and journal versions carry the expected proceedings/journal front matter; Drenth is the selected arXiv v1. Gautier is scanned: its first page was inspected visually and shows the correct title/authors, IEEE Transactions on Robotics and Automation 6(3), June 1990, printed p. 368.
- **All 80 pinpoint occurrences now checked:** the downloaded García publisher PDF contains Sec. 2.2 on PDF p. 2 and Table I on PDF p. 4. The downloaded Ovchinnikov publisher PDF contains Definition 7 on PDF p. 3. These resolve the six formerly pending occurrences. No cited numbered object was found missing; Drenth's section-title difference remains recorded below.
- **Willard date correction:** the downloaded published article gives its own reference format as ACM Computing Surveys 55(4), Article 66, November 2022, 37 pages. Its footers also state publication date November 2022. Retain the existing 2022/nov/66 fields. The earlier recommendation to use the April 2023 issue date is withdrawn; Crossref's later issue date does not override the article's own explicit citation format.
- **Drenth:** the new 6-page PDF is arXiv:2605.12203v1, matching the selected preprint version. The old 7-page local manuscript remains a different version. All cited numbers exist in v1, but its Sec. 4.1 is “Parameter Estimation”; the old manuscript's Sec. 4.1 “Well-posedness conditions” corresponds to material in v1 Sec. 3. The eliminated-variable equation is old (9), arXiv (5). Review the citation at `sections/03_augmentation.tex:887` when using the new version. No published proceedings record was found in the earlier official searches; the preprint's acceptance notice alone does not establish a version of record.
- **Downey:** the downloaded published PDF lists Downey, Hefny, Li, Boots, Gordon. The proceedings landing page/metadata instead put Li last. Retain the PDF author order currently in refs.bib. The PDF prints Geoff Gordon; the landing page expands it to Geoffrey J. Gordon.
- **Györök 2025:** the downloaded PMLR file is an additional published copy. Published copies were already held under `literature/Orthogonality/` and `citation-audit/2026-10-18/external-sources/`. No download remains necessary.
- **Current thesis refresh:** the introduction changed during the download follow-up. Citation extraction was repeated, yielding 23 keys and 144 occurrences. `irds2022` and `workman2021diewafer` are newly cited relative to the initial 21-key extraction; both are now included in the table. Neither new citation has a pinpoint. The Workman PDF already in thesis-referenced is a manuscript with template copyright metadata; the published IEEE version is the only remaining acquisition item in this refreshed citation set.
- **Preprint decisions:** the Hoekstra LFR and encoder works remain preprints in the official/author records inspected earlier, respectively submitted to Automatica and IFAC WC 2026. No standalone published version of Raissi Part I was found. Do not substitute the later, differently titled combined journal work under the Part I key. No submission venue was verified for Part I.

'''
s=s[:start]+section+s[end:]

# Replace the result paragraphs, leaving methodology and the other checked records intact.
start=s.index('The current extraction finds')
end=s.index('The version rule used here',start)
summary='''The refreshed extraction finds **23 keys, 144 key occurrences, and 80 citation occurrences with pinpoints**. Two keys, `irds2022` and `workman2021diewafer`, were added to the introduction after the original 21-key extraction and are marked new in the table. Of the 144 occurrences, 116 are in main text, 7 inside todo, 18 in deleted or moved-away text, and 3 inside wip. No citation was found inside revnote, main.tex, or frontmatter. Comments are excluded; a multi-key citation counts once per key.

**All 80 pinpoint occurrences have now been checked in the selected cited versions.** García's five and Ovchinnikov's one formerly pending occurrences were resolved using the downloaded publisher PDFs. All numbered objects exist. Drenth Sec. 4.1 has a different title/content from the old local manuscript, as recorded in the table.

**22 of the 23 selected versions are local across the checked folders.** Ten are already in `literature/thesis-referenced/`; twelve are available in `literature/thesis-reference-new/` (Györök 2025 also had earlier local published copies). Only Workman's published IEEE ECTC article remains unverified locally following its reintroduction into the text. The new folder contains 13 files because Gautier was downloaded twice as byte-identical PDFs. No PDF was renamed, moved, copied, replaced or deleted during this follow-up.

'''
s=s[:start]+summary+s[end:]
s=s.replace('- `irds2022.pdf`\n','').replace('- `workman2021diewafer.pdf`\n','')

start=s.index('## To download')
end=s.index('## Bib corrections',start)
downloads='''## To download

The twelve distinct papers in the new folder satisfy the earlier acquisition list. A refreshed citation extraction adds one acquisition item because Workman is cited again:

- `workman2021diewafer`: published IEEE ECTC 2021 proceedings PDF, pp. 2071–2077, DOI 10.1109/ECTC32696.2021.00326. The existing 7-page local manuscript still has a template copyright string and ©20XX IEEE; obtain the final publisher PDF. [Official publisher link](https://doi.org/10.1109/ECTC32696.2021.00326). TU/e access may be needed.

The existing copies in `literature/thesis-reference-new/` have not yet been placed under the old thesis-referenced key filenames. The table links directly to the correct newly downloaded versions, so another download is unnecessary for those sources.

'''
s=s[:start]+downloads+s[end:]

# Withdraw date changes that the published ACM PDF disproves and resolve article number 66.
s='\n'.join(line for line in s.split('\n') if not (line.startswith('- `willard2020integrating` | `year`') or line.startswith('- `willard2020integrating` | `month`') or line.startswith('- `willard2020integrating` | `articleno`')))
marker='- All other entries: no required bibliographic field correction identified.'
s=s.replace(marker,'- `willard2020integrating` | `year`, `month`, `articleno` | current: `2022`, `nov`, `66` | correct: retain all three | source: downloaded publisher PDF p. 1, ACM Reference format; printed footers also give November 2022. The prior date-change recommendation is withdrawn.\n- `irds2022` | `doi` | current: absent | correct: `10.60627/C13Z-V363` | source: [IEEE 2022 edition citation](https://irds.ieee.org/irds2022/).\n- `irds2022` | `title` | current: `International Roadmap for Devices and Systems: 2022 Edition Executive Summary` | exact publisher-recommended short form: `Executive Summary 2022` (optional normalization) | source: [IEEE 2022 edition citation](https://irds.ieee.org/irds2022/).\n'+marker)

# Ensure there is no stale pending verdict or obsolete local-availability instruction.
assert '**PENDING:' not in s
assert 'Six occurrences remain pending' not in s
assert '\u2014' not in s and '--' not in s
keys={r['key'] for r in c}
tablekeys=set(re.findall(r'^\| `([^`]+)`',s,re.M))
assert tablekeys==keys,(tablekeys^keys)
assert len(tablekeys)==23
report.write_text(s,encoding='utf-8')
print('Updated report: 23 keys, 144 occurrences, 80 pinpoint occurrences checked, 1 acquisition item.')
print('Downloads: 13 files, 12 unique SHA-256 values.')
for m in manifest:
 assert hashlib.sha256(Path(m['file']).read_bytes()).hexdigest()==m['sha256']
print('All downloaded PDFs unchanged.')
