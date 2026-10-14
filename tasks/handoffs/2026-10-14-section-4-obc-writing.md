# Handoff: write thesis Section IV (orthogonal-by-construction correction) with Dirk, maths checked against quoted papers
**From**: session of 2026-10-14 (Section III-B) | **Branch**: Augmentation | **Effort suggested**: xhigh (the thesis's own contribution; equation-level reading of two Gyorok papers, Kessels and Hoekstra, plus code tracing)

## 1. Task
Dirk's words: "write a prompt for another session that will make sure to properly write the orthogonality section
... we have learned a lot in this session ... and add potentially proper revnotes."
Work with Dirk on Section IV, "Interpretability by Orthogonal Construction" (`Thesis-writeup/Writing/sections/04_obc.tex`),
using the thesis-section skill (`.claude/skills/thesis-section/SKILL.md`) from its step 1. The goal is a Section IV
that (a) states every construction mathematically correctly and as in the cited paper, (b) shows for each element
whether it is adopted, adapted or this work's own, with a positive reason from a paper, a decision or the code,
(c) describes the method of the FINAL runs, with a `\revnote` wherever the current runs differ, and (d) reads concise
and in Dirk's voice. Dirk decides content and writes the sentences; you supply reading, maths, structure, checks and
proposals. This is interactive: present, wait for Dirk, then act.

## 2. Out of scope
- Sections I to III and V to VIII, except a label or reference that Section IV changes (list it, ask first).
- Code changes and reruns. Code differences become `\revnote`s plus a FINAL-RUN-FIXES item plus a decision entry
  (see section 4, "method"), not edits.
- Compiling `main.tex` (Dirk's editor rebuilds on save; test displays in a scratch copy).
- Git write operations.
- Re-opening Section III decisions (D-249, D-250, D-251 and its amendment; Jan's order and notation in III-B).

## 3. Where things stand
Branch `Augmentation`, last commit `0246725`; uncommitted edits from 2026-10-11 to 10-14 in `03_augmentation.tex`,
`99_appendix.tex`, `08_conclusion.tex`, `docs/decisions.md` (D-249 to D-251), `FINAL-RUN-FIXES.md` (items 7 to 10).
Section IV status (`04_obc.tex`, read 2026-10-14):
- Header: page budget 1.55 pages, WRITING GUIDE, MUST ESTABLISH IV-A 1 to 5, IV-B 6, IV-C 7, 8 (agreed 2026-10-09),
  draft notes (notation, source conflicts, left-out list, moved items, stale lines, code confirmed in cycle 17).
- IV-A `sec:obc_map` (line 83): negation, departure from the research plan's penalty, sensitivity `eq:obc_sens`
  (Jacobian in beta at beta_lin, no offset column), reference tuples, coefficient `eq:obc_coef`, correction
  `eq:obc_corr`, differentiated coefficient, per-epoch refresh.
- IV-B `sec:obc_ident` (line 247): `eq:obc_problem`, per-phase behaviour (PS S1, S2, Adam, validation, polish),
  M_OBC named.
- IV-C `sec:obc_scope` (line 317 on): COMMENTED OUT, including `eq:obc_eps` and the recovery proposition, but still
  referenced by `04_obc.tex` lines 77 and 138, `01_introduction.tex` line 146 and `06_results.tex` line 95.
  Ask Dirk first whether IV-C is out on purpose; until then treat those references as broken.
- Open `\todo`s: offset-column supervisor confirmation (line 159); data tuples vs closed-loop simulated baseline
  states (line 172); planned projection figure (233); principal-angle measurement and the reason against a fixed
  linearisation point (244); model notation (314).
- No `\revnote` in Section IV yet.

## 4. Established and verified
**Method that worked in III-B (Dirk's preferences, 2026-10-11 to 10-14; apply them here):**
- Read the paper passage in the session from the rendered PDF page before stating "X says"; quote or transcribe it in
  your answer. Citing from memory produced wrong section numbers twice.
- Organise the section as adoption and adaptation of the cited method, element by element: what the paper does
  (equation), what we do, adopted / adapted / own, and the reason. Supervisors want exactly this.
- Give a POSITIVE reason for each choice, from a paper where possible ("Gyorok et al. require X, so ..."), not an
  argument against an alternative. Dirk rejected a paragraph that argued "we do not use the other method".
- Follow the cited author unless a deviation has a reason Dirk can defend; if there is none, either follow the paper
  or log the deviation with its reason (D-249 shows the pattern: deviation kept because it generalises the paper).
- No experiment data or measured numbers in a method section (README "Paper structure and ownership"). Evidence
  for a choice goes to Results, the appendix, or only to the decision log.
- The text describes the FINAL method. Where the current runs differ, add a `\revnote` ("The current results ...
  The final runs will ... as described here."), a FINAL-RUN-FIXES item and a `docs/decisions.md` entry before any
  code change. A `\todo` marks an open decision for Dirk, not a code difference.
- Check the cited papers for internal inconsistencies (the encoder paper stacks oldest-first in Sec. 2 and
  newest-first in Eq. 10) and say which equation a thesis line follows.
- Check every symbol's coordinate frame (physical, scaled, scaled-and-centred); a frame error survived review once.
- Verify every derived formula numerically in a small scratchpad script before it goes into the text (the III-B map
  was checked to 6e-17); measure changed displays against the 252 pt column in a scratch compile.
- Use the established terms of Section III: physical state x_tilde, additional states x_bar (latent), training
  window, beta (training coordinates, `eq:mdelta_map`), theta^0_base, window set calligraphic W in `eq:aug_loss`.
- Prose: Dirk writes; your fixes are at most five words, or a proposal he asks for. Plain wording, one fact per
  sentence, "we" with a verb; he flags AI-sounding phrasing ("no physical counterpart" was rejected).
- An independent critique session with a "quote every source" rule caught real errors (Beintema includes y_t;
  scope of Sec. 4.4 claims; window size 29 vs 30). Offer one when a subsection's maths is final.
**Section III facts Section IV builds on (verified this session):**
- Normalisation centred (D-249): x^n = T_x(x - mu_x). IV-A's second reason for omitting the offset column relies on
  this centring; it still holds.
- Encoder: window tau-1 .. tau-n-1 (D-250); physical rows initialised data-based from a closed-loop baseline
  simulation `eq:aug_enc_sim` (D-251 with option-A amendment). That closed-loop baseline simulation exists in the
  method now, which bears on the open IV-A `\todo` at line 172 (data tuples vs simulated states): Gyorok simulates the
  baseline when states are not measured (2025 Sec. 3, per the current text), and the open-loop drift reason no
  longer excludes a closed-loop simulation. Present this to Dirk as the open question; do not decide it.
- `eq:aug_loss` uses calligraphic W; `eq:aug_init` starts the physical encoder rows from theta^{b,0}_enc.
- FINAL-RUN-FIXES items that may change what Section IV states about the current runs: 1 (W^b from true parameters),
  2 (polish acceptance uses combo_err, an oracle), 6 (g_aug live), 7 to 10 (encoder). Check each against IV and add a
  `\revnote` where IV describes the final behaviour.

## 5. Assumed but not verified
- That the IV-A text matches Gyorok 2026 Eqs. (2), (8), (9), (11), (13), Lemma 3, Assumption 1 and Gyorok 2025
  Sec. 3, Sec. 4, Eqs. (13), (15), (18), Remark 1, Remark 2 as cited (README source map says verified 2026-10-09; the
  citation log is `Thesis-writeup/Documentation/CITATION-LOG.md`; re-read before relying on it).
- That the code still does what the header's "CODE CONFIRMED (cycle 17)" list says after the Section III changes:
  `GD/obc_gantry.py` (`build_reference_set`, `make_basis_builder`, `GantryOBCCorrection`), `FS/obc.py` (`OBCBasis`,
  `OBCLifecycle`), `FS/interconnect.py` (`SSE_Interconnect_Composed.loss`, `_sync_prediction_state_for_validation`),
  `FS/blocks.py::transition_from_free`.
- Whether the reference tuples' velocities (fourth-order differences of NOISY positions) satisfy what IV-C claims
  (Gyorok's Assumption 6 is realistic only for noise-free data, README source map row IV-D).
- Whether the OBC runs of the campaign (THESIS-RESULTS step 5, D-242, D-243) used settings that IV must flag with a
  `\revnote` (detuned start; polish acceptance; zero g_aug).

## 6. Tried and failed
- Arguing against an alternative instead of for the choice -> Dirk rejected it -> lead with the paper's reason.
- Experimental numbers in a method paragraph -> moved out, then the supporting appendix removed as unneeded -> keep
  evidence in the decision log unless Dirk asks for it in Results.
- Editing while Dirk's buffer was unsaved -> save conflict -> ask "saved?" before every edit, then tell him to run
  "File: Revert File".
- Attributing a convention to the wrong paper (past-only window to Beintema) -> check the exact equation.
- Bash in this environment lost `cat`, `awk`, `tr`, `conda` mid-session -> use PowerShell and small Python scripts
  (keep CRLF line endings; OneDrive can briefly lock a file, retry with the Edit tool).

## 7. Achieved
Section III-B rewritten and reviewed on 2026-10-14 (data-based encoder, four `\revnote`s, decisions D-249 to D-251,
FINAL-RUN-FIXES items 7 to 10). Section IV has a full draft of IV-A and IV-B from 2026-10-09 with the open `\todo`s
listed in section 3; nothing in IV was changed this session.

## 8. The open question
Whether IV-C (scope and true-recovery condition) is out on purpose. It holds the section's formal contribution
(MUST ESTABLISH item 7, the proposition) and is referenced from four places. Ask Dirk first; his answer decides
whether IV is two or three subsections.

## 9. Next action
Start the thesis-section skill at step 1 for Section IV: read the README (ownership, Math standard, source-map rows
IV-A to IV-D), `03_augmentation.tex` in full (what IV builds on), `04_obc.tex` in full including the commented IV-C,
the papers from their rendered pages (Gyorok 2026: abstract, Secs. 2 to 4 including Assumption 1, Lemma 3, Thm. 7,
Remark 5, Assumption 6; Gyorok 2025: Secs. 3 and 4, Eqs. 13 to 19, Remarks 1, 2; Kessels Sec. 5.3.1.3), and the code
of section 5. Then give Dirk, in chat:
1. a table of every IV construction: Gyorok's version (quoted or transcribed, page and number), ours, adopted /
   adapted / own, the reason and its source, and whether the code matches (or which `\revnote` it needs);
2. the list of current-run differences that need a `\revnote` in IV, each with its FINAL-RUN-FIXES item;
3. the IV-C question of section 8.
Then continue with step 2a (what IV must establish), starting from the existing MUST ESTABLISH block and showing only
what you would change.

## 10. Acceptance criterion
Every equation and claim in Section IV traces to a passage quoted in the session, a decision entry, or the code;
every element is marked adopted, adapted or own with a positive reason; every difference between the text and the
current runs is covered by a `\revnote` with a FINAL-RUN-FIXES item; no `\todo` remains that Dirk has not chosen to
keep; no `\ref` points to a removed label; no display overflows the 252 pt column; and Dirk accepts each paragraph as
his own.

## 11. Read these first
1. `Thesis-writeup/Writing/sections/04_obc.tex` (all, including the header and the commented IV-C).
2. `literature/Orthogonality/gyorok2026_orthogonal-by-construction-augmentation_IFAC-JSC_arXiv2511.01321.pdf`.
3. `literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf`
   (Gyorok et al. 2025, L4DC; cite key `gyorok2025l4dc`).
4. `Thesis-writeup/Writing/README.md` (ownership, Math standard, source-map rows IV-A to IV-D, open conflicts).
5. `.claude/skills/thesis-section/SKILL.md`.

## 12. Do not
- Rewrite Dirk's sentences unasked; edit without his request and a "saved?" check.
- State "Gyorok says" or "Jan says" without the passage read in the session.
- Put measured numbers or experiment outcomes in Section IV's method text.
- Decide the open `\todo`s (offset column, data tuples vs simulated states, IV-C) for Dirk; present options with one
  recommendation.
- Touch code, compile `main.tex`, or run git write operations.

## 13. Operational
PDF pages: Read tool with `pages`. Text search: PyMuPDF in a scratchpad script run with
`conda run -n GraduationProject python <file>` from PowerShell. Display widths: a scratch `IEEEtran` conference file
with `\settowidth` (column 252 pt), as in session 3be8b0ae's `scratchpad/eqtest/w3.tex`. Multi-line `.tex` edits:
the Edit tool, or a Python script that reads and writes UTF-8 with CRLF. Kessels' thesis: `literature/augmentation/
kessels2025_ai-control.pdf` (PDF page = thesis page + 26).

## 14. Delegation
None for the writing. One read-only critique session per finished subsection is useful (Dirk runs it); write its
prompt on request, with the "quote every source" rule.
