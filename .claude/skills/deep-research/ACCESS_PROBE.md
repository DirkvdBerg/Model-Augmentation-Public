# Access probe - what can we reach, and where does the procedure break?

Not a recall exam. Measures coverage, breadth and access, and returns ranked improvements.
No answer key, so nothing here can leak into the run.

Run in a **fresh session** (`claude --continue` keeps context; a genuinely new session tests
the skill text harder). Iteration history at the bottom.

**Preconditions before starting.** The run cannot test route 5 without these:
- eduVPN connected (or TU/e campus network / library proxy)
- Chrome open, signed in to the Claude **extension** (not just to claude.ai)
- The session started AFTER the extension was connected; the bridge is startup-scoped

---

## Paste this

```
Read .claude/skills/deep-research/ACCESS_PROBE.md and run it.

Capability probe of our literature-research setup, not a search for one paper.
Run all eight probes. For EACH: source used, exact query shape, results returned,
how many on-target, turns spent.

P1  BREADTH - CONTROL
    Without using any author name, map recent control work on model augmentation /
    physics-guided identification. Use venue and topic enumeration.
    -> Does enumeration reach groups outside one lab? Name the distinct groups found.

P2  BREADTH - MACHINE LEARNING
    Same question, ML side: recent work on combining physics models with learned
    components. Do NOT use the control author roster in SKILL.md.
    -> Does the skill route you to the right sources, or does it drag you back to control?

P3  RENAMED CONCEPT
    Find work on making a learned component non-overlapping / orthogonal to a physics
    model, WITHOUT using the phrase "orthogonal-by-construction" in any query.
    -> Tests enumeration vs keyword matching.

P4  VENUE DEPTH
    Enumerate one full venue-year (your choice: IFAC-PapersOnLine, L4DC, NeurIPS, CDC).
    Report total papers, how many on-topic, and their access states.

P5  DOWNLOAD PIPELINE
    Fetch full text for FOUR papers from four DIFFERENT routes in the resolution order,
    and one of them MUST be route 5 (TU/e authenticated browser session) on a closed
    IEEE or Elsevier paper. Run the step-5 preflight first and report its outcome.
    For each: which route, did it work, file size / page count / section coverage.
    -> Verifies retrieval end to end, not just that a URL responds.

P6  CITATION GRAPH
    Pick the most relevant paper found. Traverse both directions.
    -> Does it add anything the enumeration missed?

P7  THE BOUNDARY
    Find something you CANNOT reach EVEN WITH institutional access active. Walk the
    full resolution order and report where each step failed.
    -> The boundary moved: with eduVPN up, most closed IEEE/Elsevier items are now
       reachable, so this probe is harder than it was. Candidates: paywalled content
       outside TU/e's subscriptions, books, non-deposited theses, industrial reports.
       If you cannot find anything unreachable, say so - that is a real result.

P8  SKILL DEFECTS
    Report every place SKILL.md was wrong, stale, ambiguous, or cost you a wasted turn.
    Quote the line. This is the most valuable probe - be specific and unsparing.

Then produce:

## Access map
| Source | Worked for | Failed for | Verdict |
Cover: dblp, OpenAlex, Crossref, Semantic Scholar, arXiv, PMLR, NeurIPS, Unpaywall,
TU/e Pure, and the mcp__paper-search__* tools.

## Breadth assessment
How many DISTINCT research groups did you reach? List them. If most results trace to
one lab, the breadth mechanisms are not working.

## Full-text reachability
Of everything surfaced: full text / abstract-only / metadata-only / unreachable, and
which route each resolved by.

## What is structurally out of reach
Specific publishers, venues, years.

## Ranked optimizations
Highest value first. API keys worth getting, query shapes that outperformed, sources
worth dropping, concrete SKILL.md edits. Cost and benefit for each.

## Research Log
Per the skill's required format.

Ground rules:
- Search, do not recall. Every paper named must come from a query run in this session.
  If something came from repo files instead, say so.
- Report failures. A probe reporting everything worked is not useful.
- Do not stop at titles. P5 requires actual downloaded files.
```

---

## Reading the result

Working if it returns things we did not know: an underperforming source, a failed route, a
query shape that beat the documented one. A clean sweep means the probes were too easy.

Feed **Ranked optimizations** and **Suggested skill fix** into SKILL.md, then re-run this
file unchanged so versions stay comparable.

## Iteration history

**v1 probe (2026-07-25, 40 queries).** Found 4 defects: `cites:`/`cited_by:` labels swapped;
`oa_status=gold` wrongly claimed machine-fetchable; dblp AND-semantics undocumented (long
queries silently return 0); Windows cp1252 crash in the skill's own snippets. Net new
capability found: dblp `venue:X:` prefix, the only route to recent CDC/ECC. All applied.

**Between v1 and v2, untested by any fresh run:** breadth mechanisms 1-3, the quarantined
author roster + question-type routing table, ML coverage (PMLR/NeurIPS/topic IDs), rewritten
resolution order, enumerate-then-filter-locally rule, MCP DOI retrieval at position 6.
P1, P2, P4 and P8 target this surface.
