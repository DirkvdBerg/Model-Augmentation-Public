---
name: deep-research
description: Literature search and retrieval across control engineering (LPV, LFR, system identification, model augmentation) and machine learning (NeurIPS/ICML/ICLR/L4DC/CoRL, physics-informed and learning-based modelling). Use for any request to find, survey, or fetch academic papers - "what's the state of the art on X", "find papers about Y", "get me that paper", "who cites Z", literature reviews, related-work sweeps. ALSO use whenever research is driven by a DOCUMENT or a described problem rather than a topic - "read docs/<file>.md and research it", "here is our problem statement, find relevant work", "what does the literature say about this", "is this novel", "has anyone solved this", or any pasted problem description. In that case step 0 (FRAME) is mandatory and runs before any query: it extracts sub-questions, seed DOIs already cited, entry points, a disqualification filter and an anti-scope, instead of compressing the document into a keyword query, which is the documented failure mode. Runs a frame-seed-expand-filter-access procedure over dblp/OpenAlex/Crossref/arXiv/PMLR/TU-e Pure instead of keyword web search, and reports a diagnostic Research Log so the procedure itself can be improved.
---

# Deep Research - control + ML literature

Every rule here is measured, not assumed. Numbers in brackets are from probe runs
(v1 2026-07-25, v2 2026-07-25 after a 40-query capability probe, v3 2026-07-25 after a
7-agent fan-out on `docs/drift-problem-statement.md`, ~330 queries; report in
`docs/drift-literature-sweep-2026-07-25.md`).

v3 added: the fan-out hazards (shared IP, shared scratchpad, OpenAlex daily cap), the
`search_arxiv` replacement, the author-homepage and HAL access routes, grep-don't-page for
long PDFs, and two rules that exist because a frame error cost real findings (local-holdings
must cover `scripts/**/research/`; dblp cannot match a property).

## Windows preamble (required)

Prefix every snippet below with `export PYTHONIOENCODING=utf-8`. Without it the curl+python
snippets die with `UnicodeEncodeError: cp1252` the moment a title contains a non-ASCII
character, which for Tóth, Györök and Péni is constant. Also: `conda run python -c` rejects
multi-line snippets; write to a scratchpad file and run the file.

**Never use `/tmp` for intermediate files.** `curl -o /tmp/x.html` writes a real file that
`ls` can see, but Windows python then raises `FileNotFoundError` on the same path
(`/tmp` resolves to `...\anaconda3\Library\tmp\`). Use the absolute scratchpad directory
from CLAUDE.md instead. Costs a turn every time.

**In a fan-out, `mkdir` a per-agent subdirectory and work only in it.** The scratchpad root
is shared between sibling agents, and a collision there fails *silently and wrongly*, which
is worse than a crash. Measured 2026-07-25 in a 7-agent fan-out, two distinct failures:
a sibling's `enum.py` in the root shadowed the stdlib and broke every `python` call with a
circular-import error from `json`; and a sibling overwrote `cr.py` between two calls, so a
batch of six Crossref lookups returned confident, correctly-formatted metadata for **entirely
unrelated papers** (a Synfacts note, a JHEP orientifold paper) which was nearly reported as
findings. Name helper scripts `<agent>/<name>.py`, never `enum.py`/`json.py`/`re.py`.

## Why this skill exists

Keyword search fails on niche control topics. Measured: OpenAlex `search=` for "orthogonal
projection regularization model augmentation" returned brain tumour classification, Quantum
ESPRESSO and climate projections. The paper that mattered (Györök et al. 2026) was found by
**author-ID enumeration**, with no topical keyword at all. Authors also rename concepts between
papers ("projection-based regularization" 2025 -> "orthogonal-by-construction" 2026), so a
query written from the old vocabulary structurally cannot match the new paper.
**Enumeration beats matching.**

## 0. FRAME - mandatory whenever the input is a document or a described problem

**Trigger:** the request points at a file, pastes a problem statement, or describes a
situation rather than naming a topic. Examples: "read `docs/X.md` and research it",
"here is our problem, find relevant work", "what does the literature say about this".

**Do NOT compress the document into a keyword query.** That is the documented failure
mode of this entire skill. Measured: `orthogonal projection model augmentation` returned
Quantum ESPRESSO and climate projections;
`preventing neural network from compensating physical model parameters regularization`
returned graph neural networks for transportation and medical imaging, 0 on-target. A
distilled query is the worst possible use of a rich document.

Read the document in full, then extract these six things **before running any query**:

| # | Extract | Why |
|---|---|---|
| 1 | **Sub-questions** | One per genuinely open question. These become the subagent fan-out. Documents often state them outright (a "what is not known" section). |
| 2 | **Seed DOIs / arXiv IDs already cited** | Worth more than any keyword. Feed straight into citation traversal (step 2). A document arguing with a specific paper hands you the best seed you will get. |
| 3 | **Entry points per sub-question** | Venue + year, author IDs, OpenAlex topic. Never a keyword string. |
| 4 | **Disqualification filter** | Constraints that make a result irrelevant no matter how well it matches. Report these as "found but disqualified by constraint N", not as hits. |
| 5 | **Anti-scope** | Explicitly closed questions. Do not return work on them. |
| 6 | **Vocabularies** | See below. Mandatory. |

### Multi-vocabulary requirement (mandatory, from `tasks/lessons.md` rule 117)

**Before any novelty or gap claim, search the same idea in at least two other fields'
words, and state which vocabularies you searched next to the claim.** Fields worth
translating into: control, machine learning, thermodynamics, navigation and estimation,
statistics, econometrics, applied maths.

This is not theoretical. Three "unreported" claims were refuted in one session by
re-searching in another field's vocabulary: a soft one-sided power penalty was already
published twice, once in control words (DiLaR-Soft) and once as the Macauley bracket in
thermodynamics-informed ML; and a separation problem thought novel is standard practice in
inertial navigation under "bias observability" and "modulation".

Also from that rule: **an abstract can invert the verdict.** One paper's abstract says
"constraint on the weights", reading as a hard constraint, while the body implements a soft
penalty. An abstract-only read produces false negatives, so downgrade, never conclude.

### Check local holdings first

Before searching, grep `literature/` and `docs/references.md`. A located-but-unread local
paper outranks any new search result. Report anything the document cites that is already
on disk.

**The check must also cover `scripts/**/research/`, `scripts/**/results/` and any
`thread-*.md`, not just `literature/` and `docs/`.** Measured 2026-07-25: two agents
independently "found" Zhuang et al. (AdamW-as-prox, TMLR 2022) and reported it as the
headline result, while the project already held it in
`scripts/gantry/drift-fix-trials/research/thread-AB-optimizer-mechanics.md` items A3/A4
with the venue and the key equation. The frame pointed them at `literature/` only, so both
burned their best find on a rediscovery. Grep the whole repo for an author surname and the
arXiv ID before claiming any paper is new to the project.

**Where a local note records its own verification level, carry it into the frame.** The same
thread had flagged the algebra `PARTIAL-FETCH` with an explicit "re-read Section 3 before
citing". That is a *research task the document is handing you*, and it is worth more than a
new query: closing it converted an unciteable claim into a verified one.

### Output of step 0

State the frame explicitly before querying, so the user can correct it while it is cheap:

```
Sub-questions: <n, listed>
Seed DOIs from the document: <list>
Entry points: <venue/author/topic per sub-question>
Disqualifies: <constraints>
Out of scope: <closed items>
Vocabularies to search: <>=3 fields>
Already held locally: <from literature/>
```

Only then proceed to step 1.

## 1. SEED

### dblp - best precision for both control and ML venues

**dblp AND-matches every term.** This is the single most common way to waste a turn.
Measured: `model augmentation` = 1668 hits, `+identification` = 23, `+neural` = 1.
**Cap queries at 3 terms.** A long descriptive query silently returns 0 and looks like
"no such work exists".

**dblp AND-matches TITLE words, so it cannot find a PROPERTY.** Variance, bias, conditioning,
stability, identifiability: these are what a paper is *about*, not what its title says.
Measured 2026-07-25 across a 7-agent fan-out, 9 dblp queries returned 0 useful results:
`truncated backpropagation variance` = 0, `rollout curriculum horizon` = 0,
`identifiability marginally stable` = 0, `marginal stability neural` = 0,
`proximal dissipativity` = 0, all genuine zeros with valid non-empty bodies. Spend the budget
only on **named artefacts** (a method name, a system name, `venue:CDC:+<artefact>`) and never
on an analytical property. With a small per-run budget, venue-scoped queries are the highest
value use, because they are the *only* route to recent CDC/ECC/ACC.

```bash
curl -s "https://dblp.org/search/publ/api?q=<3+TERMS>&format=json&h=15" \
| python -c "
import sys,json
d=json.load(sys.stdin)
for h in d['result']['hits'].get('hit',[]):
    i=h['info']
    print(f\"{i.get('year','?')} [{str(i.get('venue'))[:26]:26s}] {i.get('title','')[:70]}\")
    print(f'      {i.get(\"ee\")}')
"
```

**Venue-scoped search** via the undocumented `venue:X:` prefix. This is the only working route
to recent CDC/ECC/ACC/IROS proceedings, which OpenAlex cannot reach at all:

```bash
q=venue:CDC:+physics      # measured 34 hits incl. ten 2024-2025 CDC papers
q=venue:NeurIPS:+<term>   # 131    q=venue:ICML:+<term>   # 74
q=venue:ICLR:+<term>      # 77
```

**dblp IP-blocks after roughly 10 queries. Budget ~8 per run.** Two distinct failures:
early on you see empty bodies on HTTP 200; past the threshold you get **HTTP 000, 0 bytes**
(connection refused) on every query including ones that worked minutes before. Recovery took
~10 minutes; 3 retries at 3 s backoff all failed, so back off in **minutes**, not seconds.

Neither failure is distinguishable from a genuine zero-result miss, and misreading it costs
real conclusions: a probe run briefly concluded `venue:L4DC:` was an invalid venue key when
it was simply blocked. It is valid and returns 18 hits once unblocked.

**dblp does NOT index IFAC-PapersOnLine.** For IFAC symposia use OpenAlex source
`S2898405271` (29,400 works) or Crossref.

### OpenAlex `search=` - narrow use only

**Works only for broad, established terminology. Fails completely on niche multi-concept
queries.** Measured side by side, same session:

| Query | Top-5 |
|---|---|
| `LPV system identification` | 5/5 on-target (basis functions, subspace ID, MIMO LPV) |
| `orthogonal projection model augmentation` | 0/5: Quantum ESPRESSO, climate projections, brain tumours |

Worse, `search=` given a paper's **exact title** returned brain-tumour classification at
rank 1 and never returned the paper at all. If you know the title, use
`filter=title.search:<TITLE>` which returned `COUNT: 1`, exact.

Rules: `search=` for a broad field term. `filter=title.search:` for a known title.
**Neither for a concept you are trying to name.** For that, enumerate (step 2).
Never `filter=title_and_abstract.search:` on an acronym: "LPV" also matches the drug
lopinavir and returns oncology papers.

## 2. EXPAND - enumerate, don't match

**OpenAlex has a shared per-IP daily spend cap. Budget it like dblp, and guard every parse.**
Measured 2026-07-25: a 7-agent fan-out exhausted it mid-run
(`{"error":"Rate limit exceeded", ..., "dailyRemainingUsd":0}`, `retryAfter` 36240 s, resets
midnight UTC). Two agents then lost whole query batches, and one ran **no citation-graph
queries at all**. Worse, the 429 body carries no `results` key, so the snippets below print
nothing and it reads as a genuine miss: three "no such work exists" conclusions in that run
were rate-limit refusals. Always parse with `d=json.load(sys.stdin); assert 'error' not in d, d`.
**Crossref is the drop-in substitute for metadata** (it resolved 9 of 9 with DOI, venue and
pages while OpenAlex was 429ing on 7 of 7).

**Author corpus.** The highest-recall move in this skill:

```bash
curl -s "https://api.openalex.org/authors?search=<NAME>&per-page=3"
curl -s "https://api.openalex.org/works?filter=author.id:<AID>,from_publication_date:2025-01-01&sort=publication_date:desc&per-page=25"
```

**A single author's corpus is a neighbourhood, not a field.** Enumerating only a supervisor
returns their orbit.

**Resolve authors from the question, not from a stored list.** Take the authors off a seed
paper you actually found in step 1, resolve them with `authors?search=`, then enumerate. This
works identically in ML and control and carries no prior about who matters.

**Author IDs fragment. Enumerate EVERY returned ID and union by DOI.** `authors?search=`
routinely returns several IDs for one person, and picking "the right one" silently halves
recall. Measured: `Bendeguz Gyorok` returns `A5107471352` (5 works), `A5141295088` (1),
`A5133717479` (1); enumerating only the largest loses two 2026 papers.

> **Project-specific seeds - control only. Do NOT use for a machine-learning question.**
> Relevant only when the question is about *this thesis's area* (LPV/LFR identification,
> model augmentation, precision motion). For an ML question these names will actively
> misdirect: skip this box and use breadth mechanisms 2 and 3 below.
>
> | Author | OpenAlex ID | Works | Note |
> |---|---|---|---|
> | Roland Tóth | `A5088619613` | 394 | two other "Roland Tóth" exist; this is the control one |
> | Maarten Schoukens | `A5056957450` | 175 | TU/e, supervisor |
> | Johan Schoukens | `A5076996530` | 1006 | nonlinear sysid |
> | Paul M.J. Van den Hof | `A5048625929` | 446 | TU/e, identification for control |
> | Mircea Lazar | `A5045512282` | 444 | physics-guided / learning MPC |
> | Tom Oomen | `A5050892930` | 494 | TU/e, precision motion |

### Pick the entry point from the question type

| Question is about... | Use | Do not use |
|---|---|---|
| This thesis's area (LPV, augmentation, LFR) | the seed box above, then mechanisms 1-3 | - |
| Control generally | mechanism 1 (venue-year) + 3 (topic `T11236`) | the seed box |
| **Machine learning** | **mechanism 2 (PMLR/NeurIPS) + 3 (topic `T11206`)** | **the seed box** |
| A named paper's neighbourhood | citation graph from that paper's own authors | any stored roster |

For breadth use the three mechanisms below.

### Breadth mechanism 1: venue-year enumeration (complete, unbiased)

Enumerate an entire proceedings rather than searching it. No keyword, no ranking bias:

```bash
# IFAC-PapersOnLine 2025: measured 2,709 works
curl -s "https://api.openalex.org/works?filter=primary_location.source.id:S2898405271,publication_year:2025&per-page=200&cursor=*"
```

Page with `cursor=*` then the returned `meta.next_cursor`. For CDC/ECC/ACC/IROS, OpenAlex has
no source records 2023-2026, so use dblp `venue:CDC:` instead (see step 1).

**Crossref beats OpenAlex for venue-year completeness.** Same journal-year: Crossref
returned 2,957 IFAC-PapersOnLine 2025 works against OpenAlex's 2,709, i.e. 248 records
OpenAlex does not hold. Use it as the completeness check:

```bash
curl -s "https://api.crossref.org/journals/2405-8963/works?filter=from-pub-date:2025-01-01,until-pub-date:2025-12-31&rows=0"
```

**Filter regexes need a negative list.** `augment` matches "Augmented Reality" and "data
augmentation": 8 of 22 IFAC hits and most of 138 NeurIPS hits were that noise. Exclude
`augmented reality`, `data augmentation`, `image augmentation` before counting.

**Enumerate then filter locally. Never add `search=` inside a venue filter.** Measured on
IFAC-PapersOnLine 2025: `filter=source,year` + `search=model augmentation physics` returned
**1** result from a 2,709-work pool. Pulling 800 titles by cursor and regex-filtering them
locally returned **16** on-topic papers from seven unrelated groups. `search=` inside a
narrow filter over-constrains and silently looks like "this venue has nothing".

Same pattern for PMLR: fetch the volume page, extract all titles, filter locally.
Measured on L4DC v283, 120 papers enumerated, 10 on-topic.

Bonus finding: every IFAC-PapersOnLine hit was `diamond` OA. IFAC proceedings are fully
open, so venue enumeration there has no access cost at all.

### Breadth mechanism 2: PMLR volume enumeration (the ML side)

PMLR is fully open and enumerable per volume. Verified: **L4DC 2025 = v283** (120 PDFs),
**L4DC 2024 = v242** (139 PDFs); volumes currently run past v336. Also ICML, AISTATS, CoRL.

```bash
# titles + the REAL pdf hrefs (do not construct the URL yourself)
curl -s "https://proceedings.mlr.press/v283/" | grep -oE 'href="[^"]*\.pdf"'
```

**Never fabricate the PDF path.** `https://proceedings.mlr.press/v<VOL>/<id>/<id>.pdf`
**404s** (returns 9,379 B of HTML, which pypdf rejects as `invalid pdf header: b'<!DOC'`).
The real pattern, verified 200 / 2,214,319 B:

```
https://raw.githubusercontent.com/mlresearch/v<VOL>/main/assets/<id>/<id>.pdf
```

Always take the href from the volume index rather than building it. A run that constructs
the mlr.press path silently loses the entire PMLR route.

**NeurIPS parser** (the plausible `<li class="conference">` shape does not exist):

```bash
curl -s "https://papers.nips.cc/paper_files/paper/2025" \
| grep -oE 'href="/paper_files/paper/2025/hash/[^"]+">([^<]+)</a>'   # 5,823 titles
```

**Do not use OpenAlex for ML venue enumeration.** Source records exist (NeurIPS
`S4306420609`, ICML `S4306419644`) but are badly undercounted: NeurIPS shows 4,134 works
total when recent single years exceed that. Same failure class as the missing CDC/ECC
records. For ML venues use PMLR directly, `papers.nips.cc`, or dblp `venue:NeurIPS:`.

### Breadth mechanism 3: topic enumeration (field-level, no keyword)

```bash
curl -s "https://api.openalex.org/topics?search=<FIELD%20WORDS>&per-page=5"
curl -s "https://api.openalex.org/works?filter=primary_topic.id:<TID>,from_publication_date:2025-01-01&sort=publication_date:desc&per-page=50"
```

Resolved: **`T11236` Control Systems and Identification** (59,135 works),
**`T11206` Model Reduction and Neural Networks** (71,281 works).

**Both sort orders fail on a 59k-work topic, in opposite ways.** `sort=cited_by_count:desc`
on T11236 for 2025+ returned a top-10 entirely from one prolific hierarchical-identification
group, 8 of 10 closed, none relevant. `sort=publication_date:desc` returned MDPI *Mathematics*
papers, Zenodo self-deposits and a Leicester dataset record: recent low-quality deposits, not
the frontier.

**Treat topic enumeration as the weakest of the three mechanisms.** Prefer venue-year
(mechanism 1) and PMLR (mechanism 2), which are bounded and curated. Use topics only to
discover venues and author names you did not know, then switch back to enumeration.

**Citation graph.** The direction labels are counter-intuitive; these are verified empirically
(seed with `cited_by_count=1236` and 54 references returned 1232 and 44 respectively):

```bash
# works that CITE this one (FORWARD)
curl -s "https://api.openalex.org/works?filter=cites:<WID>&per-page=25"
# works this one CITES, i.e. its references (BACKWARD)
curl -s "https://api.openalex.org/works?filter=cited_by:<WID>&per-page=25"
```

**Resolve the PUBLISHED DOI's work ID before traversing. This is not optional.**
OpenAlex records of `type: preprint` carry `referenced_works: 0`, so both directions return
nothing. Measured on the same paper: the arXiv record `W4406318306` gave **0** references,
the published record `W4416435552` gave **19**, including the single most important ancestor
(Kon et al. 2022, the origin of the projection method) that no other query in a 45-query
sweep found. A 0 here means "wrong record", not "no references".

**But first check that a published record EXISTS, or you will hunt a work ID that does not.**
The rule above assumes a published version is out there. Measured 2026-07-25 on
`arXiv:2604.18277`: both OpenAlex records were `type: preprint` with `referenced_works: 0`,
and Crossref had no version at all because the paper says "accepted for IFAC publication" and
the proceedings are not yet deposited. So: query Crossref for the title; if it has nothing and
every OpenAlex record is a preprint, the citation graph is **genuinely unreachable**. Report
that as a coverage gap and move to the paper's *stated ancestors* instead, traversing their
graphs (that route worked: following the seed's named predecessor gave 13 forward citers when
the seed itself gave 0).

**Forward traversal is dead at the frontier.** 2025-2026 control papers show
`cited_by_count=0`. Indexing lag, not a tooling fault.

Backward traversal is also lossy where it works: 19 of ~26 references on a checked paper,
dropping books (Bohlin 2006 confirmed absent). `referenced_works:` is an undocumented alias
for `cites:`.

## 3. FILTER - the two fields

**Control:** IFAC-PapersOnLine (OpenAlex `S2898405271`), Automatica, IEEE TAC/TCST/TIE, CDC,
ECC, ACC, IROS, Systems & Control Letters, IFAC J. Systems and Control, arXiv eess.SY / math.OC.

**Machine learning:** NeurIPS, ICML, ICLR, L4DC, CoRL, AISTATS, TMLR, JMLR,
arXiv cs.LG / stat.ML. ML literature is far more open than control (see step 4).

**Skip entirely - biomedical, zero yield:** PubMed, PMC, bioRxiv, medRxiv, chemRxiv,
Europe PMC, DOAJ.

## 4. ACCESS

**Every full-text success in the capability probe came through arXiv, PMLR or TU/e Pure.
Zero came through a publisher.** Build the plan around that.

### Never fetch a publisher URL

`oa_status` does **not** predict machine reachability. A `gold` OA Elsevier item returned a
2,703-byte JS shell; IEEE returned HTTP 202 with zero bytes. The response is indistinguishable
between free and paywalled content. `gold`/`hybrid`/`diamond` mean free **to a human browser**,
not fetchable. Go straight to `locations[]`.

### Query `locations[]`, and use `works/doi:`

```bash
curl -s "https://api.openalex.org/works/doi:<DOI>" \
| python -c "
import sys,json
d=json.load(sys.stdin)
print(d.get('title'),'|',d.get('open_access',{}).get('oa_status'))
for L in d.get('locations',[]):
    s=(L.get('source') or {})
    print(f\"  [{'OA' if L.get('is_oa') else '  '}] {str(s.get('display_name'))[:32]:32s} pdf={L.get('pdf_url')}\")
"
```

`works/doi:` returns the rich record. **Do not judge accessibility from a `title.search:`
result row** - the same DOI can appear twice, and the first row may carry a single location
with `pdf=None`, which reads as inaccessible when five locations exist.

### Resolution order

Ordered by measured hit rate. Stop at the first route that returns a PDF.

1. **arXiv** `https://arxiv.org/pdf/<ID>` - 100% success on every attempt.
2. **PMLR** `https://proceedings.mlr.press/v<VOL>/` - fully open, all PDFs listed.
   L4DC 2025 = **v283** (120 PDFs). Also hosts ICML, AISTATS, CoRL.
   NeurIPS: `papers.nips.cc`.
3. **TU/e Pure** - serves the Version of Record. The HTML search at `research.tue.nl` is
   **403 bot-blocked**; the OAI-PMH endpoint is the only programmatic door:
   ```bash
   curl -s "https://pure.tue.nl/ws/oai?verb=ListRecords&metadataPrefix=oai_dc&set=publications:year2026:withFiles"
   ```
   (`publications:year2026:withFiles` = 1,244 records with attached full text.)
   Direct PDFs live at `pure.tue.nl/ws/files/<id>/<name>.pdf`.
3b. **Author homepage - the route for closed economics, econometrics and statistics.**
   OpenAlex reports `oa_status: closed`, zero OA locations and `pdf=None` for
   Econometrica / Econometric Theory / Biometrika / J. Econometrics items even when the author
   hosts a free copy. Measured 2026-07-25: 3 of 4 closed econometrics classics resolved this
   way, 0 via OpenAlex or Unpaywall.
   **Scrape the publications index; never guess the filename.** Filenames are idiosyncratic
   (`I0I1discP.pdf`, `beyondLTU2.pdf`, `wp-content/uploads/2026/06/lower_risk_bounds.pdf`) and
   5 guessed URLs 404'd across 4 wasted turns. This is the **same rule as "never fabricate the
   PMLR PDF path"**.
   ```bash
   curl -sL "<homepage>" | grep -oiE 'href="[^"]*\.pdf"'
   ```
   Then walk any `?page_id=` subpages the same way. **Do not use WebFetch for the index**: it
   returned navigation chrome and no link list on 3 of 3 attempts on the same page where
   `curl | grep` returned 21 PDF hrefs, because the summarising model drops long link lists.
   Also try central-bank working-paper series
   (`federalreserve.gov/pubs/ifdp/<year>/<n>/ifdp<n>.pdf` worked first try).
   **Caveat to report:** these are often pre-publication versions, so quotes cannot carry
   journal page numbers. Say which version was read.

3c. **HAL - the route for the French control corpus (Bombois, Gevers, Colin, Scorletti).**
   HAL bot-walls `curl` on `/document` and `/file/*.pdf` (12.5 kB HTML titled "Making sure
   you're not a bot!"). Check a file exists, then read it through the MCP wrapper:
   ```bash
   curl -s "https://api.archives-ouvertes.fr/search/?q=halId_s:<ID>&fl=fileMain_s,files_s"
   ```
   then `mcp__paper-search__read_hal_paper`. The API check correctly predicted which records
   had no attached file, saving the fetch. HAL is the primary free route for a corpus that is
   otherwise closed *Automatica*.

4. **Unpaywall** - reliable OA *oracle*, weak *locator* (`url_for_pdf=None` even for OA
   repository entries). Pair it with OpenAlex `locations[]`.
   ```bash
   curl -s "https://api.unpaywall.org/v2/<DOI>?email=$UNPAYWALL_EMAIL"
   ```
5. **TU/e authenticated browser session** (Claude-in-Chrome) for anything still closed.
   Uses the user's own institutional entitlement. Covers Elsevier, IEEE, Springer and
   Wiley, so it reaches most remaining closed items including 2025-2026 conference papers.
   Boundary measured 2026-07-25: AIAA ARC returns "No Access" even with eduVPN, so TU/e
   holds no AIAA licence. Continue to step 6 when this route fails.

   **Preflight, required before relying on this route.** Two independent layers must both
   hold, and they fail differently:

   ```
   Skill(claude-in-chrome)                              # load first
   mcp__claude-in-chrome__tabs_context_mcp{createIfEmpty:true}
   ```

   | Outcome | Meaning | Action |
   |---|---|---|
   | Returns tab list | bridge up, go on to layer 2 | navigate to a known-closed DOI |
   | `Browser extension is not connected` | **layer 1 down**: Claude cannot reach Chrome at all | user installs/enables the extension at `claude.ai/chrome`, same account as Claude Code, may need a Chrome restart |
   | Bridge up but page shows a paywall/login prompt | **layer 2 down**: Chrome is not authenticated to TU/e | user signs in via TU/e library proxy or campus network |
   | Bridge up, page loads, but tool is blocked | site permission not granted in the extension | user grants the site in the extension UI |

   Layer 1 being down makes layer 2 untestable. Do not report "no institutional access"
   when the bridge is what failed; they are different problems with different fixes.

   **Reporting is mandatory.** If this route is unavailable for any reason, say so
   explicitly in the results, name which layer failed, and mark every item that would
   have needed it as `unreachable - browser route unavailable` rather than
   `unreachable`. Silently skipping step 5 makes a reachable paper look permanently
   closed.

   **VERIFIED end to end 2026-07-25** on `10.1109/IROS60139.2025.11247377` (IEEE Xplore,
   closed, no preprint anywhere): full text of all five sections retrieved via
   `navigate` + `get_page_text`, including LaTeX-rendered equations. Route works.

   Sequence that worked, and the three traps on the way:
   1. `Skill(claude-in-chrome)`, then `list_connected_browsers`.
      **Trap A:** `[]` means either *not installed* OR *installed but not signed in to
      the extension*. Indistinguishable, and the error text points only at installation.
      Signing in to Chrome is NOT signing in to the extension.
      **Trap B:** the bridge is startup-scoped, so an extension connected mid-session
      stays invisible until Claude Code restarts (`claude --continue`).
   2. `select_browser{deviceId}` after asking the user which browser.
   3. `navigate` to the DOI, then `get_page_text`.
      **Trap C:** anonymous IEEE returns abstract + Section I then
      `"Sign in to Continue Reading"`. That string is the layer-2 tell. With eduVPN or
      the library proxy active, the same URL returns every section.
   Re-navigate after the user enables VPN; the paywalled render is cached in the tab.
6. **MCP DOI retrieval** - if a DOI is available and every route above has failed, call
   `download_scihub`; if that fails, call `download_with_fallback`, preserving its upstream
   fallback behavior. If no DOI is known, resolve one from OpenAlex/Crossref metadata first.

Why this order and not the reverse: routes 1-3 are the only ones that produced full text in the
capability probe, arXiv at 100%. Placing a low-hit-rate route first costs a failed request on
every paper that was freely available anyway, which is the large majority (~74% of one measured
corpus was legally free, and 58 of 60 titles the probe surfaced were reachable).

### What is structurally unreachable

- All publisher sites (Elsevier JS shell, IEEE 202), including gold OA.
- IEEE conference papers with no preprint (CDC/ECC/ACC/IROS are frequently closed).
- Elsevier articles from groups that do not self-archive. TU/e and SZTAKI papers are
  near-always on Pure or arXiv; other groups' often are not.
- Recent CDC/ECC as a browsable OpenAlex venue (no source records 2023-2026; use dblp `venue:`).
- Forward citations for 2025-2026 work.
- Books in the citation graph.

## 5. READ AND CITE

**The deliverable is a citable finding.** A paper read in full through the browser and
correctly cited is already a complete success, so do not burn turns chasing a file you do
not need.

**But if the text cannot be read, downloading is how you get it. Work the resolution order
all the way down, including step 6.** Never abandon a paper because reading failed;
a downloaded PDF is read locally with `Read`. The two are alternative means to the same
end, and whichever works is the right one.

Note for either path: IEEE returns HTTP 502 to `curl` even *with* valid institutional
entitlement (a bot wall, separate from the paywall), and the extension blocks in-page URL
extraction, so a publisher-direct download attempt failing says nothing about whether other
routes will work. Do not treat a failed download as a failed retrieval, and never report a
paper as unreachable when its text was read.

- Prefer the TU/e Pure Version of Record over a preprint when the published version matters.
- `Read` handles PDFs natively (`pages` for long ones).

### Grep long documents, do not page through them

**For any PDF over ~30 pages, extract-and-grep with context instead of reading pages.**
Measured 2026-07-25 on a 126-page monograph: two `pypdf` greps for `"spectral radius"` and
`"runcated"` with +-700 characters of context located the load-bearing assumption, the scope
disclaimer and the convergence theorem in two turns. Paging it would have cost 7+ turns at the
20-page `Read` limit and would plausibly have missed the decisive sentence entirely.

```python
import sys,re
from pypdf import PdfReader
r=PdfReader(sys.argv[1])
for i,pg in enumerate(r.pages):
    t=pg.extract_text() or ""; tl=t.lower()
    for p in [x.lower() for x in sys.argv[2:]]:
        for m in re.finditer(re.escape(p),tl):
            a=max(0,m.start()-700); b=min(len(t),m.end()+700)
            print(f"=== p{i+1} [{p}]"); print(" ".join(t[a:b].split()))
```

**Grep the ASSUMPTION vocabulary, not the topic**: `we assume`, `beyond the scope`,
`future work`, `finite variance`, `does not hold`, `spectral radius`, `if the system is`.
A paper's scope disclaimers are where "nobody has done this" is actually written, and they
never appear in the title, the abstract, or any index.

**Do not trust `file` for page count.** It reported 3 and 6 pages for arXiv PDFs that pypdf
read as 18 and 36, because it parses only the linearised first chunk. Two correct downloads
were nearly discarded as truncated. Check with pypdf; never re-download on a low `file` count.
- Route 5 (browser) reliably yields **text**, not **files**. `get_page_text` on IEEE returns
  every section including rendered equations. That is enough to read, quote and cite.
- **Do not conclude from an abstract.** State what you verified in full text and what you did not.

### Citation metadata is mandatory for every paper named

Pull it from Crossref rather than transcribing from a page; publisher pages routinely omit
page numbers and the venue's full name.

```bash
# ready-to-paste BibTeX
curl -sL -H "Accept: application/x-bibtex" "https://doi.org/<DOI>"

# structured fields (authors, venue, pages, date, type)
curl -s "https://api.crossref.org/works/<DOI>"
```

Verified 2026-07-25 on `10.1109/IROS60139.2025.11247377`: returned a complete
`@inproceedings` entry including `pages={18320-18326}`, which the IEEE Xplore page does
not display anywhere.

Every paper in the output needs: **authors, title, venue, year, DOI, and where the free
copy is.** If a field is genuinely unavailable, say so rather than omitting it silently.

## 6. REPORT

Findings first, then the Research Log. See the format at the end of this file.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `UnicodeEncodeError: cp1252` | Windows console | `export PYTHONIOENCODING=utf-8` |
| dblp returns 0 on a descriptive query | AND semantics | Cap at 3 terms |
| dblp returns an empty body | rate transient | Retry with backoff; check for empty |
| OpenAlex acronym search returns medicine | "LPV" = lopinavir | Use `search=`, not `title_and_abstract.search` |
| Publisher URL gives JS shell / 202 | bot wall, applies to gold OA too | Never fetch publisher; use `locations[]` |
| `pdf=None`, looks paywalled | reading a `title.search` row | Use `works/doi:`, read all `locations[]` |
| `cites:` returns 0 | it is FORWARD; frontier papers have no citers | Use `cited_by:` for references |
| Semantic Scholar returns `[]` | 429, and the MCP wrapper hides it silently | Needs an API key; treat `[]` as suspect |
| `research.tue.nl` 403 | bot-blocked HTML | Use `pure.tue.nl/ws/oai` |
| `conda run python -c` newline error | harness limit | Write a scratchpad file |
| **OpenAlex returns no results, on a query that should hit** | **HTTP 429 daily spend cap. The 429 body is valid JSON with an `error` key and NO `results` key, so every parser in this file renders it as zero hits** | **Guard every snippet: `assert 'error' not in d, d`. Add `-w "\nHTTP %{http_code}"` to the curl. Fall back to Crossref** |
| dblp / arXiv API returns an empty body on HTTP 200 | rate transient, not a zero | Sleep 3+ s between arXiv calls; back off in **minutes** for dblp. Never record it as a zero |
| `file` reports 3 pages for an 18-page PDF | it parses only the linearised first chunk | Check page count with pypdf; do not re-download |
| Crossref metadata is confidently wrong for the DOI you asked about | a sibling agent overwrote your helper script in the shared scratchpad root | Work in a per-agent subdirectory (see Windows preamble) |

## MCP tools

`mcp__paper-search__*` wraps the same APIs.

**`search_arxiv` is unusable for multi-concept queries. Use the arXiv API directly.**
Condemned independently by four agents in one 2026-07-25 fan-out: it matched the token "Deep"
in author *names* (returning papers by "Deep Pandey", "Deep Ray", "Indra Deep Mastan", 15/15
off-target, twice); it returned CMS/ATLAS particle physics for an **exact title**, with a
101 kB spill file; and it returned air-pollution LSTMs, fake-news detection and symplectic
4-manifold topology for a marginal-stability query, 0/15. Same failure class as OpenAlex
`search=`. The raw API with field prefixes and boolean AND was high-precision on nearly every
attempt (`ti:"proximal" AND ti:"Adam"` returned exactly 1 hit and it was the right paper;
`abs:"antisymmetric" AND abs:"recurrent neural network"` returned 2, both bullseyes):

```bash
curl -s -G "http://export.arxiv.org/api/query" \
  --data-urlencode 'search_query=abs:"<CONCEPT A>" AND abs:"<CONCEPT B>"' \
  --data-urlencode "max_results=15"
```

Two title terms is usually the right size; the API AND-matches like dblp. Use `ti:` for a
known title, `abs:` for a concept, `cat:` to scope. **Sleep 3+ seconds between calls**, or
three rapid queries return a silent empty body indistinguishable from a zero-result miss.

**Read `opensearch:totalResults`, because the count is itself the deliverable.** A total of
0 to 2 over a whole-of-arXiv abstract search is the strongest novelty evidence this skill can
produce, stronger than any ranked list. Measured: only **2** arXiv abstracts contain both
"proximal operator" and "physics-informed"; `"one-sided" AND "penalty" AND "dissipative"` = 0;
`abs:"port-Hamiltonian" AND abs:"lossless" AND abs:"neural"` = 0. Those zeros carried more
weight in the final report than most of the hits.

**`search_google_scholar` is a MANDATORY cross-check after enumeration, not an optional
extra.** In a 45-query probe it was the *only* route that found OrthoReg (Richter &
Kilbertus, TU Munich / Helmholtz Munich, `arXiv:2606.19145`), a paper directly on the
physics/neural overlap problem, from a group with no link to TU/e or SZTAKI. No OpenAlex
query, no dblp query, no venue enumeration and no citation traversal surfaced it.
Enumeration and keyword search are **complementary**; neither is strictly better. Run one
Google Scholar query per research question after enumerating, and treat a hit it alone
found as a signal that the enumeration frame was too narrow.

Confirmed at scale on 2026-07-25: it was the sole source of the highest-value item in **four
of seven** sub-questions, each time from a community unconnected to the seeds (computational
neuroscience, probabilistic numerics, computer graphics, robot dynamic-parameter
identification). Two distinct mechanisms, and the second is the new one:

1. **Recall breadth**, as in the OrthoReg case above.
2. **It indexes full text, so it is the only route to an in-body SCOPE DISCLAIMER** - the
   sentence where authors say a case is out of scope, which is where "nobody has done this" is
   actually written. Measured: a 14-result query snippet-matched "allowing the spectral radius
   of the sequence to tend to 1 over time, but this is beyond the scope of the present work"
   on p24 of a 126-page arXiv monograph. No OpenAlex, dblp, arXiv or Crossref query in the
   same run could reach it.

So when the question is "has anyone treated case X", **write the Scholar query as the sentence
you expect the paper to contain**, not as keywords. And note the vocabulary traps: Scholar's
"proximal" space is owned by Proximal Policy Optimization (13 of 15 hits were RL).

**`search_core` is a usable locator**, contrary to earlier text: it returned a working
direct PDF link (14 pp, 2.38 MB) for a paper it found. Precision is poor (4 of 5 off-target),
so use it to *locate* a known paper, not to discover.

**`search_semantic` returns `[]` masking an HTTP 429.** Confirmed against the raw API. A `[]`
means "unknown", never "nothing". Dead weight until a key is set.

## Subagent fan-out

One agent per independent seed or sub-question. Do not spawn for a single lookup. Give each
agent the sub-question, known author IDs / seed DOIs, the field filter, and an instruction to
return findings plus its Research Log. Merge logs, deduplicate by DOI.

**Every agent shares one IP and one scratchpad, so tell each of them, explicitly, in the
prompt:**
- a **dblp budget** (2 queries at 7 agents; the IP blocks after ~10 for ~10 minutes),
- an **OpenAlex budget** and the `assert 'error' not in d` guard (the daily spend cap is
  shared and a 7-way fan-out exhausts it),
- to `mkdir` and work in a **per-agent scratchpad subdirectory**,
- **not** to run the TU/e browser preflight. One preflight belongs in the parent; seven
  agents each asking the user which browser to use is a bad experience. Have them mark items
  `needs-browser-route` and chase the ranked list from the parent afterwards.

**Set the frame before fanning out, and show it to the user.** The agents inherit whatever
scoping error it contains, multiplied by N. Measured 2026-07-25: a frame that named
`literature/` as the local-holdings location caused two of seven agents to spend their best
find rediscovering a paper the repo already held.

**Ask each agent to grade its own negative claims.** A "nobody has done this" from an agent
whose OpenAlex calls were 429ing, or whose venue sweep reached 30% of one year, is provisional
and must be reported as such. Merge those caveats into the final report rather than dropping
them; a gap claim is only as strong as the coverage behind it.

## Required output format

```markdown
## Findings
<answer to the question. For EACH paper: authors, title, venue, year, DOI, and where the
 free copy is. State its actual finding, not just that it exists. Include BibTeX for
 anything worth citing.>

## Access status (MANDATORY - every run, even when nothing was paywalled)
TU/e browser access: AVAILABLE | UNAVAILABLE (<which layer failed>)
<If UNAVAILABLE, state it plainly in the visible answer, not only here. Any item that
 would have needed it is marked "unreachable - browser route unavailable", never a bare
 "unreachable". If AVAILABLE and used, say which papers came through it.>

## Evidence quality
<read in full vs abstract-only vs metadata-only - be explicit>

## Research Log
- Queries run: <source + query + n results + on-target?>
- What worked: <the query shape that produced the find>
- What failed: <query/source that returned noise or errors, and why>
- Dead ends: <looked relevant, was not>
- Coverage gaps: <what this sweep could NOT reach>
- Suggested skill fix: <concrete change to this SKILL.md, or "none">
```

Search, do not recall. Every paper named must come from a query run in the session; if
something came from repo files instead, say so.

**Always report the TU/e browser access status**, on every literature run, not only on
probes and not only when something turned out to be paywalled. The user needs to know
whether the run had institutional access available, because a sweep done without it has a
systematically different reachability profile: closed Elsevier and IEEE items silently drop
out and the result looks complete when it is not. State it even when every paper was open.
