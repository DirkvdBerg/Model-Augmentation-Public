"""Assemble reviewed evidence; does not compile LaTeX or run research code."""
from pathlib import Path
import json,csv,re,html,hashlib,collections,datetime
O=Path(__file__).resolve().parent
BASE=O.parents[1]
R=O/'snapshot/Writing'
def readjson(n):return json.loads((O/n).read_text(encoding='utf-8'))
def writejson(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def tsv(n):return list(csv.DictReader((O/n).open(encoding='utf-8'),delimiter='\t'))
def esc(v):return html.escape(str(v))
def csvout(n,rows):
 if not rows:return
 with (O/n).open('w',encoding='utf-8-sig',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
rows=readjson('occurrences.json');ass={x['id']:x for x in tsv('assessments.tsv')};bib=readjson('bibliography.json');sources=tsv('source-notes.tsv');scope=readjson('scope.json')
assert set(ass)=={x['id'] for x in rows if x['main_visible']}
for r in rows:
 if r['id'] in ass:r.update(ass[r['id']])
 else:r.update(verdict='EXCLUDED: UNUSED TEMPLATE' if not r['reachable'] else 'EXCLUDED: COMMENTED' if r['commented'] else 'EXCLUDED: FALSE CONDITIONAL',evidence='',note='Retained in inventory; does not execute in either thesis build. Not counted as an unresolved thesis citation.')
 r.pop('status',None)
writejson('occurrence-assessments.json',rows);csvout('occurrence-assessments.csv',rows)
refs=[e for e in bib if e['file']=='refs.bib'];loaded=[e for e in bib if e['file'] in ('refs.bib','IEEEabrv.bib')]
unused=[e for e in refs if not e['cited_main']];csvout('unused-bibliography.csv',unused);csvout('all-bibliography.csv',bib)
mainkeys={k for r in rows if r['main_visible'] for k in r['keys']}
reviewkeys={k for r in rows if r['review_visible'] for k in r['keys']}
assert mainkeys=={s['key'] for s in sources}
source_map={s['key']:s for s in sources}
for s in sources:
 e=next(e for e in refs if e['key']==s['key']);s['bibliography']=e;s['main_occurrences']=[r['id'] for r in rows if r['main_visible'] and s['key'] in r['keys']]
 s['local_pdf']=f'literature/thesis-referenced/{s["key"]}.pdf' if (BASE/f'literature/thesis-referenced/{s["key"]}.pdf').exists() else {'gyorok2025l4dc':'literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf','venkatraman2017psd':'literature/added-states-training/1709.08520.pdf','ovchinnikov2021computing':'citation-audit/2026-10-18/external-sources/ovchinnikov2021computing-v3.pdf','gautier1990minimum':'citation-audit/2026-10-18/external-sources/gautier1990minimum.pdf'}[s['key']]
writejson('source-register.json',sources);csvout('source-register.csv',sources)
# Compare actual executed citations with source occurrences. Moving captions can execute twice.
aux=(R/'build/main.aux').read_text(encoding='utf-8')
auxcalls=re.findall(r'\\citation\{([^}]+)\}',aux)
expected=collections.Counter(k for r in rows if r['main_visible'] for k in r['keys'])
actual=collections.Counter(k for c in auxcalls for k in c.split(','))
dups={}
for field in ('key','doi','title'):
 groups=collections.defaultdict(list)
 for e in loaded:
  v=e['key'] if field=='key' else e['fields'].get(field,'')
  v=re.sub(r'[^a-z0-9]','',v.lower())
  if v:groups[v].append(e['key'])
 dups[field]=[v for v in groups.values() if len(v)>1]
inline=[]
for p in R.rglob('*.tex'):
 s=p.read_text(encoding='utf-8');f=p.relative_to(R).as_posix()
 for m in re.finditer(r'\\bibitem(?:\[[^]]*\])?\{([^}]+)\}',s):inline.append({'file':f,'line':s.count('\n',0,m.start())+1,'key':m[1],'included':f in scope['included_files']})
writejson('inline-bibliography.json',inline)
checks={'source_calls_main':sum(r['main_visible'] for r in rows),'source_key_uses_main':sum(len(r['keys']) for r in rows if r['main_visible']),'source_calls_review':sum(r['review_visible'] for r in rows),'source_key_uses_review':sum(len(r['keys']) for r in rows if r['review_visible']),'distinct_keys_main':len(mainkeys),'distinct_keys_review':len(reviewkeys),'aux_calls':len(auxcalls),'aux_extra_key_uses':dict(actual-expected),'source_not_in_aux':dict(expected-actual),'unresolved_keys_main':sorted(mainkeys-{e['key'] for e in loaded}),'duplicate_loaded_entries':dups,'refs_entries':len(refs),'unused_refs_entries':len(unused),'unused_template_bib_entries':len(bib)-len(loaded),'inline_template_entries':len(inline),'review_suppressed_source_keys':sorted(mainkeys-reviewkeys),'no_nocite':not any(r['command']=='nocite' for r in rows),'all_main_calls_individually_assessed':True,'final_journal_locator_verification_gaps':['gyorok2026obc','ovchinnikov2021computing']}
writejson('reconciliation.json',checks)
# Numbered evidence pages make source-line links stable, including formfeed boundaries.
(O/'numbered-sources').mkdir(exist_ok=True);(O/'numbered-thesis').mkdir(exist_ok=True)
for p in (O/'source-text').glob('*.txt'):
 page=1; lines=[]
 for i,l in enumerate(p.read_text(encoding='utf-8').splitlines(keepends=True),1):
  lines.append(f'<div id="L{i}"><a href="#L{i}">{i:05d}</a> [PDF p.{page}] {esc(l.rstrip())}</div>');page+=l.count('\f')
 (O/'numbered-sources'/f'{p.stem}.html').write_text('<meta charset="utf-8"><title>'+esc(p.stem)+'</title><pre>'+'\n'.join(lines)+'</pre>',encoding='utf-8')
for p in R.rglob('*.tex'):
 f=p.relative_to(R).as_posix();target=O/'numbered-thesis'/(f.replace('/','__')+'.html')
 target.write_text('<meta charset="utf-8"><title>'+esc(f)+'</title><pre>'+'\n'.join(f'<div id="L{i}"><a href="#L{i}">{i:04d}</a> {esc(l)}</div>' for i,l in enumerate(p.read_text(encoding='utf-8').splitlines(),1))+'</pre>',encoding='utf-8')
def evidlinks(e):
 a=[]
 for item in e.split(';'):
  if not item:continue
  n,rng=item.split(':');st=int(rng.split('-')[0]);assert (O/'source-text'/f'{n}.txt').exists(),n
  a.append(f'<a href="numbered-sources/{esc(n)}.html#L{st}">{esc(n)} lines {esc(rng)}</a>')
 return '<br>'.join(a)
# Source mentions without commands include prose names, outline shorthand and editorial pointers.
mentions=[]
names={'garcia2013model':r'Garc|garcia2013model','teo2007dynamic':r'\bTeo\b|teo2007dynamic','willard2020integrating':r'Willard|willard2020integrating','hoekstra2026lfr':r'Hoekstra|hoekstra2026lfr','hoekstra2026encoder':r'encoder paper|hoekstra2026encoder','drenth2025lpvlfr':r'Drenth|drenth2025lpvlfr','gyorok2026obc':r'Gy.{0,10}r.{0,10}k|gyorok2026obc','kessels2025ai':r'Kessels|kessels2025ai','hefny2015supervised':r'Hefny|hefny2015supervised','downey2017psrnn':r'Downey|downey2017psrnn','beintema2023subnet':r'Beintema|beintema2023subnet','venkatraman2017psd':r'Venkatraman|venkatraman2017psd','ovchinnikov2021computing':r'Ovchinnikov|ovchinnikov2021computing','gautier1990minimum':r'Gautier|gautier1990minimum','toth2010modeling':r'toth2010modeling','retzler2024':r'Retzler|retzler2024'}
for f in scope['included_files']:
 if not f.endswith('.tex'):continue
 for i,l in enumerate((R/f).read_text(encoding='utf-8').splitlines(),1):
  keys=[k for k,pat in names.items() if re.search(pat,l,re.I)]
  if not keys or re.search(r'\\(?:cite|ucite|fbcite)\b',l):continue
  nearby=[x['id'] for x in rows if x['file']==f and any(k in x['keys'] for k in keys) and abs(x['line']-i)<45]
  mentions.append({'file':f,'line':i,'classification':'COMMENTED SOURCE/EDITORIAL POINTER' if l.lstrip().startswith('%') else 'TEXTUAL SOURCE MENTION OR DRAFT POINTER','candidate_keys':keys,'related_occurrence_ids':nearby,'text':l,'assessment':'Supplementary source mention; no additional citation-command execution. Resolve meaning using complete paragraph and linked occurrence assessments, not this line alone.'})
csvout('textual-source-mentions.csv',mentions);writejson('textual-source-mentions.json',mentions)
findings=[
('F01','Confirmed bibliography error','kessels2025ai author is Bas Mats Kessels, not Bart M.','Use Bas Mats Kessels or B. M. Kessels; title page and university record agree.'),
('F02','Confirmed local/bibliographic version mismatch','drenth2025lpvlfr local PDF is not arXiv:2605.12203v1 named by refs.bib.','Identify the manuscript actually used or use arXiv v1 numbering. Parameter estimation is 4.1 in v1, not 5.1 (C091). Nominal LFR is Eq. (2) in v1.'),
('F03','Confirmed time-domain attribution issue','C052 introduces a continuous-time form while citing Drenth Eq. (6), which is discrete time.','Describe the CT expression as an adaptation, or cite the CT form in Drenth’s thesis. C145 can retain the partition analogy if explicitly described as adapted.'),
('F04','Confirmed locator error','C094 cites encoder Sec. 4.4 for the data-based method and Eq. (35); the method is in Sec. 3.2.','Replace the method section locator with 3.2; Eq. (35) itself is correct.'),
('F05','Scope/wording issue','C130 says Györök et al. update the linearization point every iteration; source presents this as an available option.','Use “discuss updating” unless an implementation passage establishes that it was actually done.'),
('F06','Overstatement risk','C036’s negative claim about position-dependent baselines is stronger than what one paper establishes.','Limit to the demonstrated examples and this gantry setting. The source formulation permits nonlinear/state-dependent baselines.'),
('F07','Conditional result','C038 and C063 need the source’s linear-in-parameter IO scope and the thesis’s own local state-space extension.','Do not read source uniqueness as a guarantee of a unique nonlinear closed-loop minimum. C134–C136 correctly state the fixed-tuple/first-order limitation.'),
('F08','Source is conceptual support','C054’s Definition 7 does not prove exactly ten identifiable gantry combinations.','Keep the count as the thesis’s derivation and cite Definition 7 only for the concept; preserve genericity/input/output assumptions.'),
('F09','Outdated draft note','C143 claims loss weights are only inverse output variances; current Eq. aug_loss includes Wr servo-error weighting.','Update or remove the TODO when revising. It is not retained review prose.'),
('F10','Sufficient theorem, not necessity','C046 suggests C2 smoothness as a necessary reason to omit Coulomb friction.','Condition 8 is part of a sufficient well-posedness theorem. Do not infer necessity or instability of all nonsmooth systems.'),
('F11','Version-check gaps','Györök OBC local v3 and Ovchinnikov v3 locators are verified, but final journal-numbering equivalence is not.','Final PDFs were unavailable (403/timeout). Explicitly cite the checked arXiv versions, or compare the final papers before submission.'),
('F12','Traceability improvements','Some entries lack stable URLs/version suffixes; Willard uses online year with later issue volume.','Keep date conventions consistent. Distinguish accepted manuscript, arXiv version and version of record. Missing DOI is not an error when none is established. Official NeurIPS BibTeX records themselves have empty page fields, so these page omissions are not established errors.'),
('F13','Mathematical caveat in a cited source','C149 calls Drenth Theorem 6 a sufficient condition; the spectral-radius product inequality used in the source does not hold for general matrices under its stated assumptions.','For D=0.5[[1,1],[-1,-1]] and Delta=diag(1,-1), D squared is zero (spectral radius zero) and Delta is allowed, yet I-D*Delta=0.5[[1,1],[1,1]] is singular. Use a justified stronger condition, or label it as a source statement needing qualification. The thesis’s direct mass-matrix proof is separate.'),
]
csvout('findings.csv',[dict(id=a,severity=b,finding=c,recommended_action=d) for a,b,c,d in findings])
head='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Thesis citation audit — 18 October 2026</title><style>body{font:16px/1.5 system-ui,sans-serif;max-width:1500px;margin:40px auto;padding:0 24px;color:#172532}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #c8d2dc;padding:9px;text-align:left;vertical-align:top}th{background:#e9f0f5}pre{white-space:pre-wrap;font-size:12px;max-height:360px;overflow:auto}summary{cursor:pointer}a{color:#005d91}h2{margin-top:2em}.notice{padding:14px;background:#fff3d8;border-left:5px solid #ba8200}input{font:inherit;width:95%;padding:10px}small{color:#4b5f70}</style><body><h1>Evidence-based thesis citation audit</h1>'''
parts=[head,'<p>Audit date: 18 October 2026 (Europe/Amsterdam). All findings refer to the frozen <a href="snapshot/Writing/main.tex">source snapshot</a>, not later live edits.</p>',
'<div class="notice"><b>Outcome: corrections and version qualifications required; this is not a clean citation certificate.</b> Every formal citation use in the snapshot editor build has a source-based assessment. Final-journal locator equivalence remains open for two sources. No thesis, bibliography, research code or data was edited; no LaTeX build, experiment, job, commit or result regeneration was performed.</div>',
'<p>The user message ended at “Prefer”; no additional requirements after that word were supplied. This audit applies the supplied inventory, identity, bibliographic and per-use source-check requirements, and also records claim scope and adaptation boundaries.</p>',
'<h2>Scope and coverage</h2>',
f'<p>Main root: <code>main.tex</code>; alternate wrapper: <code>review.tex</code>. Main includes the cover, abstract, Sections I–VIII, references and appendices. {len(scope["included_files"])} reachable TeX files include four TikZ fragments and eight table fragments. The appendix is included after the bibliography. All {len(rows)} literal citation calls across the Writing tree are inventoried: {checks["source_calls_main"]} editor-visible calls ({checks["source_key_uses_main"]} source-key uses, six grouped calls), 12 commented/false-conditional calls in included files, and 10 calls in an unused IEEE template. There are {len(mainkeys)} editor-build sources.</p>',
'<p>Commands in use: <code>cite</code>, <code>ucite</code>, <code>fbcite</code>. The latter two are local wrappers/aliases. No executed nocite was found. Literal macro parameters such as #2 are definitions, not unresolved source keys. No additional external citations were found in extracted generated-figure text. Captions, table rows, notes and appendix uses are included in the ledger; no citation-bearing footnotes were found.</p>',
f'<p>Review build: {checks["source_calls_review"]} calls, {checks["source_key_uses_review"]} source-key uses and {len(reviewkeys)} sources predicted from the macros. Review removes TODO/note, wip, deleted and old change text, suppresses ucite entirely, and strips every cite locator. It still shows revnote and outline content. Sources suppressed entirely: {esc(", ".join(checks["review_suppressed_source_keys"]))}. The review count is static analysis, not a newly compiled review PDF.</p>',
f'<p>Existing main.aux has {len(auxcalls)} executed citation writes. Its only excess over source-level key counts is one García and one Hefny write, consistent with the two citation-bearing moving captions. No source use is missing from aux. The existing build log reports an undefined internal <code>sec</code> reference, not an unresolved bibliographic key. No citation-key warning was found.</p>',
f'<p>Bibliography: <code>IEEEabrv.bib,refs.bib</code>. The abbreviation file contains string definitions, not article entries. <code>refs.bib</code> has {len(refs)} entries: {len(mainkeys)} cited in the editor build and {len(unused)} unused. An unused example bibliography adds {checks["unused_template_bib_entries"]} records; the unused conference template contains {len(inline)} inline bibitems. No duplicate keys, duplicate normalized titles or repeated DOIs among loaded entries; no unresolved active keys. Historical years in keys (Drenth 2025 / actual year 2026; Forssell 1999 / report 1998) are not automatically errors.</p>',
'<p>Files changed during discovery: live <code>03_augmentation.tex</code>. A new snapshot was frozen after this was detected. IDs remained stable; Section III line positions moved by eight lines. Initial inventory and manifest are retained separately. Do not treat subsequent edits as audited.</p>',
'<p><a href="scope.json">Include graph and unused files</a> · <a href="reconciliation.json">Counts/build reconciliation</a> · <a href="source-register.csv">Source register CSV</a> · <a href="occurrence-assessments.csv">Occurrence ledger CSV</a> · <a href="all-bibliography.csv">All bibliography records</a> · <a href="unused-bibliography.csv">Unused entries</a> · <a href="textual-source-mentions.csv">Supplementary prose/editorial source mentions</a> · <a href="figure-content.json">Generated figure text</a></p>',
'<h2>Findings requiring attention</h2><table><tr><th>ID</th><th>Type</th><th>Evidence-based finding</th><th>Recommended action (not applied)</th></tr>']
for a,b,c,d in findings:parts.append(f'<tr><td>{a}</td><td>{esc(b)}</td><td>{esc(c)}</td><td>{esc(d)}</td></tr>')
parts.append('</table><p>Visual evidence: <a href="garcia-figure2.png">García source Fig. 2 (physical PDF p. 8)</a> · <a href="thesis-figure1.png">Thesis adapted Fig. 1 (existing snapshot PDF p. 3)</a> · <a href="hefny-page2.png">Hefny source Fig. 2 (PDF p. 2)</a>. The two captions acknowledge adaptation. The thesis PS figure was checked in its included TikZ source and caption.</p><h2>Every cited source: identity, bibliography and exact version</h2><table><tr><th>Source</th><th>Identity/version read</th><th>Bibliographic result</th><th>Locator/version result</th></tr>')
for s in sources:
 url=s['primary_record'];link=f'<a href="{esc(url)}">Primary record</a>' if url.startswith('http') else esc(url)
 parts.append(f'<tr><td><b>{esc(s["key"])}</b><br>{link}<br><small>{esc(s["local_pdf"])}</small><details><summary>Bibliography fields</summary><pre>{esc(json.dumps(s["bibliography"]["fields"],ensure_ascii=False,indent=2))}</pre></details></td><td>{esc(s["identity_result"])}<br>{esc(s["version_read"])}</td><td>{esc(s["bibliographic_result"])}</td><td>{esc(s["locator_result"])}</td></tr>')
parts.append('</table><h2>Per-occurrence evidence ledger</h2><p>SUPPORTED means the cited source supports the specific statement with its stated conditions; it does not certify the thesis’s implementation, experiments, mathematical derivations or global novelty. Notes distinguish cited results from thesis adaptations. Numbered-source links give exact extracted lines and physical PDF page counters, including repository cover pages. For journal-cited preprints, “supported” refers to the explicitly identified consulted version; the source-register version caveat still applies.</p><p><input id="filter" placeholder="Filter by key, ID, file, verdict or wording…" oninput="document.querySelectorAll(\'.occ\').forEach(x=>x.hidden=!x.textContent.toLowerCase().includes(this.value.toLowerCase()))"></p><table><tr><th>Occurrence / visibility</th><th>Source and locator</th><th>Verdict and reasoning</th><th>Primary-source evidence</th></tr>')
for r in rows:
 visibility='unused file' if not r['reachable'] else 'commented' if r['commented'] else 'editor + review' if r['review_visible'] else 'editor only' if r['main_visible'] else 'false conditional'
 loc=f'numbered-thesis/{r["file"].replace("/","__")}.html#L{r["line"]}'
 parts.append(f'<tr class="occ" id="{r["id"]}"><td><b>{r["id"]}</b> · {esc(visibility)}<br><a href="{esc(loc)}">{esc(r["file"])}:{r["line"]}</a><br><small>{esc(", ".join(r["tags"]))}</small></td><td>{esc(", ".join(r["keys"]))}<br>{esc(r["locator"])}<br><code>{esc(r["command"])}</code><details><summary>Complete paragraph context</summary><pre>{esc(r["context"])}</pre></details></td><td><b>{esc(r["verdict"])}</b><br>{esc(r["note"])}</td><td>{evidlinks(r["evidence"])}</td></tr>')
parts.append('</table><h2>Unused bibliography entries</h2><p>These are inventory items, not citation errors. Metadata and claim support were not audited for uncited entries.</p><p>'+esc(', '.join(e['key'] for e in unused))+'</p><h2>Method and practical limits</h2><p>The include graph and macro definitions were read before deciding scope. Citation keys were parsed as balanced arguments with comment masking; conditional, draft, wip and deletion spans were recorded. Source-level occurrences were reconciled against existing aux output. Existing generated figures were text-extracted without rebuilding them. Individual claim paragraphs were compared with source sections/equations; relevant figure and tabular adaptations were checked separately. Metadata used primary PDFs, publisher proceedings, university records, arXiv and saved DOI-registry records. Previous project verification comments were treated as leads, never as proof.</p><p>Supplementary textual-source-mentions inventory includes author names, draft-outline shorthand and editorial source pointers without citation commands. These are not added to the formal citation-call count; the full source and related assessed formal occurrences are linked. Instructions such as “check source” are not treated as substantive literature claims. This is a citation audit, not a plagiarism-similarity scan, a reproduction of research results, or an independent proof of the source papers’ theorems. Statistical consistency, global uniqueness and absence-of-prior-work claims must not be inferred from matching a citation.</p><p>Known access gaps: final Györök journal PDF returned 403 through ScienceDirect and two university URLs; final Ovchinnikov NSF PDF timed out. Local v3 texts were fully available for passage checks. Willard’s final ACM article number was not independently read from publisher content. These remain explicitly open; no guessed verification is recorded. Exact versions should be pinned before submission. Missing pages/URLs on some conference entries are traceability omissions rather than invented references.</p><p>Artifacts: <a href="input-manifest.json">frozen input hashes</a>, <a href="initial-input-manifest.json">initial live-input hashes</a>, <a href="retrieval-log.json">retrieval log</a>, <a href="supplemental-retrieval-log.json">supplemental retrieval log</a>. Source text is a reading aid; consult the PDF when glyphs, equations or scan OCR are ambiguous. All 129 editor-visible calls and all 135 grouped-key uses are assessed, with the open version qualifications above.</p></body></html>')
(O/'report.html').write_text('\n'.join(parts),encoding='utf-8')
# Audit-only manifest. Includes full evidence/provenance without recursively hashing itself.
manifest=[]
for p in sorted(O.rglob('*')):
 if p.is_file() and p.name!='artifact-manifest.json':manifest.append({'path':p.relative_to(O).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
writejson('artifact-manifest.json',manifest)
print(json.dumps(checks,indent=2));print('REPORT',O/'report.html')
