# Handoff: literature sweep on why a fixed band-drawn basis beats an identified pole
**From**: session of 2026-08-25 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

Run the `deep-research` skill on four sub-questions arising from `DISCUSSION-POINTS.md` sections
I5, N and O. The frame is written in section 4 below and has already been through step 0; do not
re-derive it, but do correct it if the holdings sweep contradicts it. Write the result into
`scripts/gantry/augmented-states/DISCUSSION-POINTS.md` as `### I6.`, following the precedent of
`### I4. Literature amendments (deep-research run, 2026-08-24)`. Do not create a new document and
do not open a new lettered section.

## 2. Out of scope

- **No code, no training, no diagnostics.** This is a literature run.
- **Do not re-open which cause is which.** Section O is the consolidated map and is current.
- **Do not repeat the queries in section 6.** They are recorded true zeros.
- **Do not run the browser preflight without asking.** One preflight belongs in the parent
  session, and it needs the user to name a browser.

## 3. Where things stand

Branch `Augmentation`. `DISCUSSION-POINTS.md` is at sections A-O, 2661 lines. Section I5 (lines
1544-1646) and section M (2067+) were written this session; N and O by a sibling session. No run
in flight. Weekly usage was near its limit at handoff time.

## 4. The frame, already built

**The problem.** A fixed, band-drawn, input-driven, long-memory linear feature bank plus a
trainable nonlinear readout reaches `3.790189e-07 m` (`F5`, `arm 2`). An exactly identified single
pole, `157.9045 Hz`, agreeing with an independent ARX estimate to five digits, with matched drive,
budget, optimizer, routing and seed, plateaus at `1.231200848e-06 m`. Frozen poles tie trainable
poles. Zero encoder augmented rows beat learned ones and random ones are `2.1x` worse. **So
identification accuracy is measurably not the variable.** The gap is also qualitative: `F = 0.88`
for the winning arm against `F = 0.186` and `0.266` for the BLA arms.

**Sub-questions.**

1. **Fixed-basis design.** How is the pole set of an *untrained* recurrent feature bank chosen:
   band, radius, count, spread? ORBF / Laguerre / Kautz, Kolmogorov n-width, reservoir pole design.
   Deliverable: a rule derivable from data, or evidence that none exists.
2. **Encoder and latent initial state for unmeasured ADDED states.** What does SUBNET
   (Beintema, Toth, Schoukens) and Kessels actually require of the encoder for added states, does
   anyone report encoder-initialised latent states being *harmful*, and is a model-based observer
   the alternative to a learned encoder?
3. **Attribution in hybrid models.** Measuring and enforcing that added states carry the correction
   rather than the direct physical rows absorbing it; parameter-error versus missing-dynamics
   non-identifiability (O5.7).
4. **Control-relevant weighting.** We identify `dP` open loop but fit under an `S_b`-weighted replay
   loss with `sigma_max(S_o) = 1.81` in band. Does identification-for-control supply a
   basis-selection or weighting rule for that mismatch?

**Disqualifies.** Work assuming a large residual, open-loop training, or a trainable recurrence.
All three are measurably not this regime.

**Vocabularies to cross.** Control, machine learning, reservoir computing, approximation theory.
The reservoir vocabulary is the one that produced this session's earlier finding (Jere et al.) and
the project had never searched it.

## 5. Established and verified this session

**Holdings hit, and it partly answers sub-question 2.** `scripts/gantry/ann-blackbox/BLA-LITERATURE.md`
section 4.4 already contains the clause, never used for this question:

> "Nobody initialises the augmented-state block: every method derives the state from the baseline's
> reconstructability and the baseline has no augmented states."

Same section records Ramkannan 2023 and Hoekstra 2026 as "the entire literature" on encoder
initialisation, and lists Hoekstra 2026's three offerings: innovation-form encoder weights using
`A - KC` and `B - KD`; local linearisation with the bias carrying the linearisation point; and a
data-based least-squares or pre-trained-ANN encoder. Item 3 is the citable form of the project's
own `encoder_map_ridge`. **Start sub-question 2 from this clause, not from a query.**

**The two canonical pole-design papers are located, and both are genuinely closed.** I4 records
them as "unread behind paywalls"; that is now verified rather than assumed:

| paper | DOI | cites | access |
|-|-|-:|-|
| Ozturk, Xu, Principe, "Analysis and Design of Echo State Networks", *Neural Computation* 19(1):111-138, 2007 | `10.1162/neco.2007.19.1.111` | 314 | Unpaywall `is_oa: False`, zero OA locations |
| Wyffels, Schrauwen, Verstraeten, Stroobandt, "Band-pass Reservoir Computing", *IJCNN* 2008 | `10.1109/ijcnn.2008.4634252` | 33 | Unpaywall `is_oa: False`, zero OA locations |
| Wyffels, Schrauwen, Stroobandt, "Stable Output Feedback in Reservoir Computing Using Ridge Regression", *ICANN* 2008 | `10.1007/978-3-540-87536-9_83` | 72 | OpenAlex `closed`, `pdf=None`; Unpaywall not queried |

The third was surfaced by the same title search and is **not** in I4's list. It is more cited than
the band-pass paper and concerns output-feedback stability in reservoirs with a ridge-regressed
readout, which is structurally what this project runs. Worth retrieving alongside the other two.

`biblio.ugent.be` returns HTTP 500 on its search API, so the Ghent route for Wyffels is dead. These
need route 5 (TU/e browser) or route 6 (MCP download). **Wyffels is the more on-target of the two**:
band-pass reservoirs are literally the band-and-radius question that `F4a`, `F4b` and `F5` pose.

**Three open leads, found and not yet read:**

| arXiv | relevance |
|-|-|
| `2607.17909` "Beyond the Edge of Chaos: Stability-Expressivity Transfer in Reservoir Forecasting", 2026 | sub-question 1, spectral radius design |
| `2203.09382` "Euler State Networks: Non-dissipative Reservoir Computing", 2022 | sub-question 1, structured non-dissipative reservoirs |
| `2512.06315` "Control-Oriented System Identification: Classical, Learning, and Physics-Informed Approaches", 2025 | sub-question 4, appears to be a survey and is the obvious entry point |

## 6. Do not repeat these queries

Recorded true whole-of-arXiv zeros in I4, with valid non-empty bodies:

- `"echo state network" AND "lightly damped"`
- `"reservoir computing" AND "modal parameter"`
- `"random features" AND "resonant mode" AND "identification"`
- `"reservoir computing" AND "model augmentation"`

Also recorded zero in `BLA-LITERATURE.md` 4.5: OpenAlex title-and-abstract for "linear bypass" with
neural network and dynamical; "warm start" with recurrent neural network and linear model
identification; arXiv "warm start" AND "system identification" returned 3, all off-target. The
conclusion drawn there was that the concept lives entirely in control words, and that should be
carried into this frame rather than retested.

I4 also names two gaps this run should close: **no IFAC, CDC, ECC or ACC venue sweep has ever been
run** on sub-question 1, and Amendment G is outstanding.

### 6b. Queries already run this session, with their totals

Do not re-run these. The totals are findings in their own right, per the skill's rule that
`opensearch:totalResults` over a whole-of-arXiv abstract search is stronger novelty evidence than
any ranked list.

| source | query | total | on target |
|-|-|-:|-|
| arXiv | `abs:"reservoir" AND abs:"spectral radius" AND abs:"design"` | **10** | 2 leads, listed above |
| arXiv | `abs:"band-pass" AND abs:"reservoir computing"` | **4** | **none.** El Nino forecasting, evolutionary network control, memristor charge transport, aqueous memristors |
| arXiv | `abs:"control-relevant" AND abs:"identification"` | **5** | 1 lead, `2512.06315` |
| OpenAlex | `title.search:"Analysis and Design of Echo State Networks"` | 3 | Ozturk located |
| OpenAlex | `title.search:"Band-pass Reservoir Computing"` | 2 | Wyffels located |
| OpenAlex | `title.search:"Stable Output Feedback in Reservoir Computing Using Ridge Regression"` | 1 | third paper located |
| Unpaywall | both closed DOIs | - | `is_oa: False`, zero OA locations, both |
| `biblio.ugent.be` | search API | - | **HTTP 500**, Ghent route dead for Wyffels |

**The `band-pass` total of 4 is the load-bearing one.** Four arXiv abstracts in existence pair
band-pass with reservoir computing, and none of them is about identification or a physical
residual. Combined with I4's four recorded zeros, sub-question 1 is looking thin in the ML
vocabulary specifically. That is what makes the **unrun IFAC/CDC/ECC/ACC venue sweep** the highest
value remaining move on that sub-question, and it is also why the ORBF and Kolmogorov n-width
route (control and approximation-theory words, not reservoir words) should be tried before
concluding anything.

**Not yet run at all:** any query on sub-questions 2 and 3, the venue sweep, any dblp query, any
Google Scholar query. The Scholar cross-check is mandatory per the skill and has not been done.

## 7. Achieved

The frame above, the holdings hit in 4.4, verified access status on both closed papers, and three
open leads. Roughly six queries spent; the arXiv, OpenAlex and dblp budgets are effectively fresh.

## 8. The open question

Whether sub-question 1 has an answer at all. The winning band is `149.90` to `164.06 Hz` with a
radius range, and nothing in the project derives either. If the sweep returns no rule, that is a
reportable result: it would mean the band is a project heuristic that must be declared as one, and
`A5`'s inventory of unsourced items grows rather than shrinks.

## 9. Next action

Read `BLA-LITERATURE.md` sections 4 and 4.4 in full, then grep the reference lists of the PDFs in
`literature/stability-training/lazy-rich/` for `concurrent work`, `companion paper` and
`see also`. Only then query. The skill's own record is that this repo has twice burned its best
find on rediscovering a held paper.

## 10. Acceptance criterion

For each of the four sub-questions, one of: a citable rule with authors, title, venue, year, DOI
and where the free copy is; or a graded negative with the query counts behind it and the
vocabularies searched, per the multi-vocabulary rule. A bare "nothing found" without the coverage
behind it fails this criterion. Ozturk and Wyffels must end the run either read or explicitly
marked `unreachable - browser route unavailable`, never a bare `unreachable`.

## 11. Read these first

1. `scripts/gantry/augmented-states/DISCUSSION-POINTS.md` section O, the consolidated failure map.
2. Same file, section I5, the four eliminations and the working conjunction.
3. Same file, section I4, the previous deep-research run and its recorded zeros.
4. `scripts/gantry/ann-blackbox/BLA-LITERATURE.md` sections 4, 4.4 and 4.5.
5. `.claude/skills/deep-research/SKILL.md`, which the skill invocation loads anyway.

## 12. Do not

- Do not fetch publisher URLs. `oa_status` does not predict machine reachability.
- Do not conclude from an abstract; the skill records a case where the abstract inverts the verdict.
- Do not report a Google Scholar `[]` from a multi-quoted-phrase query as a zero; re-run unquoted.
- Do not treat a 0-byte arXiv body as a zero result.
- Do not write findings anywhere except `### I6.` in `DISCUSSION-POINTS.md`.

## 13. Operational

Invoke with the `Skill` tool, `deep-research`, passing the frame in section 4 as the argument.
`conda run -n GraduationProject python ...`; `export PYTHONIOENCODING=utf-8` on every snippet;
scratchpad only, never `/tmp`.

Budgets, from the skill's measured limits: dblp about 8 queries then a multi-minute IP block;
arXiv about a dozen at 3+ s spacing; OpenAlex has a shared daily cap and its 429 body parses as
zero results, so guard every parse with `assert 'error' not in d, d`.

## 14. Delegation

Inline, one sub-question at a time. Do not fan out: four agents share one IP and would exhaust the
dblp and arXiv budgets against each other, and this run needs the venue sweep that budget is for.
