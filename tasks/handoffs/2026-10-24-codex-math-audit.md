# Handoff (Codex): read-only audit of every thesis formula against the math, the cited papers and the main pipeline
**From**: Claude session (prompt for Codex) of 2026-10-24 | **Branch**: Augmentation | **Effort suggested**: xhigh (hand derivations, index and dimension checks, paper and code reading; no runs)

## 1. Task
Audit, read-only, every displayed equation, inline formula and mathematical claim in the thesis (`Thesis-writeup/Writing/sections/00_abstract.tex` to `08_conclusion.tex` and `99_appendix.tex`, plus any formula in `tables/*.tex` and `figures/tex/*.tex` that the text relies on). For each formula decide **correct**, **wrong**, or **conditional** (correct only under a condition the text does not state), with three checks wherever they apply: (a) **math**: derive it by hand from the equations it depends on; check dimensions, indices, signs, transposes, sums and limits, and that every symbol is defined before use with one meaning across the thesis; (b) **citation**: where the text cites a source for it, compare it with the cited passage (equation number, variables, context, and whether an adaptation is stated); (c) **code**: compare it with what the main pipeline computes (Section 4.3). Run the audit with 6 independent sub-agents, one per slice in Section 4.2, then merge their reports into one file, `Thesis-writeup/Documentation/MATH-AUDIT-2026-10-24-codex.md`, in the format of Section 4.4. Edit nothing else.

A Claude session runs the same audit independently and writes `MATH-AUDIT-2026-10-24-claude.md`. Do not read that file, and do not coordinate with it: the independence is the point.

## 2. Out of scope
- Any edit to thesis, code, config or docs. The only file you write is the audit report (plus scratch files in a temp folder outside the repository, for example `%TEMP%/math-audit-codex/`).
- LaTeX builds of any kind, numerical runs, git write commands. If a claim can only be settled numerically, say which quantity would settle it and stop there.
- The `Clean/` folder. It is the later port; this audit certifies the main pipeline first. Do not compare against `Clean/` or suggest changes to it.
- Wording and style, unless the wording makes a statement mathematically wrong or ambiguous.
- The `\ifdraft` draft bullets (never mention them).
- `kamtin-data/`, `kamtin-fp-model/` (read-only reference for the FP model is allowed only if a Section II formula claims to match it).
- Re-arguing method decisions. Report whether the math of the chosen method is right, not whether another method would be better.

## 3. Where things stand
Branch `Augmentation`, last commit `e9d3e21` "Update section 5". Dirty tree, including `sections/04_obc.tex` and `util/include.tex`; the author may be editing. Read each `.tex` file once and audit that snapshot; record the file's modification time in the report header. Change marks (`util/format.tex`): read the final reading. `\chg{old}{new}` = new, `\del` = gone, `\mv` = present, `\delmv` = gone, `\kw`/`\kwtwo` = plain text, `\todo`/`\revnote`/`\ifdraft` = not printed. Superseded equation versions kept as `%` comments are not part of the text.

## 4. Established and verified

### 4.1 Thesis structure
`main.tex` inputs `00` to `08`, then `99_appendix.tex`. Section labels: `sec:problem`, `sec:physics`, `sec:baseline`, `sec:lfr`, `sec:params` (02); `sec:augmentation`, `sec:aug_structure`, `sec:aug_loop`, `sec:aug_closed_loop`, `sec:aug_encoder`, `sec:aug_ps2` (03, content starts at line 197); `sec:obc`, `sec:obc_construction`, `sec:obc_map`, `sec:obc_ident`, `sec:obc_measures` (04); `sec:setup_system`, `sec:setup_data`, `sec:setup_models`, `sec:setup_metrics` (05); `sec:res_*`, `sec:discussion` (06). Appendix: `app:params`, `app:lfr-details`, `app:lfr-wellposed`, `app:combos`, `app:ps_stop`, `app:frf`, `app:splits`, `app:choices`, `app:setup_system`, `app:setup_records`, `app:setup_checks`, `app:setup_bb`, `app:hyper`, and the unlabelled "OBC and experiment details". Notation macros: `util/format.tex`.

### 4.2 Agent slices (one sub-agent each, all launched in one message)
| Agent | Thesis scope | Main code to compare (start here, follow imports) |
|-|-|-|
| A | `sec:problem`, `sec:physics`, `sec:baseline` (02, lines 1 to 533); `app:params`, `app:frf` | `model_augmentation/systems/gantry_ss.py`, `gantry_linearization.py`; `blocks.py` (`Gantry_State_Block`, `Reduced_Gantry_State_Block`) |
| B | `sec:lfr`, `sec:params` (02, line 534 to end); `app:lfr-details`, `app:lfr-wellposed`, `app:combos` | `gantry_ss.py`, `blocks.py`, `utils/utils.py` (`normalize_linear_ss_matrices`) |
| C | `sec:augmentation`, `sec:aug_structure`, `sec:aug_loop` (03, lines 197 to 1005); `app:setup_checks` residual-form paragraph | `fit_systems/blocks.py`, `closed_loop.py`, `interconnect.py`; `gantry_dynamic/controller.py`, `data.py` (normalisation, finite differences) |
| D | `sec:aug_closed_loop`, `sec:aug_encoder`, `sec:aug_ps2` (03, line 1006 to end); `app:ps_stop` | `interconnect.py`, `pre_encoder.py`, `ps2_opening.py`, `lbfgs_polish.py`, `utils/torch_nets.py`; `gantry_dynamic/training.py`, `model.py` |
| E | `04_obc.tex` (all); appendix "OBC and experiment details" | `fit_systems/obc.py`; `gantry_dynamic/obc_gantry.py`, `obc_diagnostics.py`, `model.py` |
| F | `05_setup.tex`, `06_results.tex`, every formula or quantitative claim in `00`, `01`, `07`, `08`; `app:splits`, `app:choices`, `app:setup_system`, `app:setup_records`, rest of `app:setup_checks`, `app:setup_bb`, `app:hyper`; `tables/*.tex` | `gantry_dynamic/evaluation.py`, `baselines.py`, `data.py`, `config.py`, `diagnostics.py`; data generation in MATLAB (locate the generator from the Section V text and `docs/`) |

Each agent reads, in addition to its slice, whatever earlier equations its slice depends on (cross-references by label), and `util/format.tex`. Each agent writes its report to that temp folder as `audit_<A..F>.md` in the format of Section 4.4 and returns only the path and its counts (wrong / conditional / correct).

After the agents return, you (the lead) do two things yourself: (1) a cross-slice notation pass: every symbol in `format.tex` and every symbol reused across sections keeps one meaning, dimension and index convention; (2) merge the six reports into the final file, ordered most severe first, removing duplicates where two slices found the same issue (keep both pointers).

### 4.3 What "the main pipeline" is
The code that `scripts/gantry/gantry_interconnect_dynamic.py` executes: `scripts/gantry/gantry_dynamic/*.py` and the `model_augmentation/` modules they import (`fit_systems/blocks.py`, `closed_loop.py`, `interconnect.py`, `obc.py`, `pre_encoder.py`, `ps2_opening.py`, `lbfgs_polish.py`; `systems/gantry_ss.py`, `gantry_linearization.py`; `utils/torch_nets.py`, `utils/utils.py`). `fit_systems/` also contains modules the main pipeline does not import (`obc_projection.py`, `orth_projection.py`, `gyorok_orth_projection.py`, `augmented_dynamics.py`, `continuous_augmentation.py`, `multiple_shooting.py`, `white_box_models.py`): confirm by import trace before using any of them, and never treat one as the implementation unless the main path reaches it. The configuration is selected by `RunConfig` (`gantry_dynamic/config.py`); the thesis runs saved theirs as `config.json` in their output folders (written by `config_json_dict`). Compare against the branch those configs select; where a formula depends on a config flag, name the flag and the value that matches the text.

### 4.4 Report format
Header: date, files audited with modification times, agents run. Then one table, every non-correct finding, most severe first:

| # | Severity (wrong / conditional / notation) | Where (label, first words, file:line) | Check (math / citation / code) | Finding | Evidence (derivation step, paper and page or equation, code file:line) | Smallest fix (exact replace-with LaTeX, or code note) |
|-|-|-|-|-|-|-|
| 1 | wrong | `eq:aug_encoder`, "chi_tau stacks", 03_augmentation.tex:1210 | math | stacks n+1 samples but stated in R^{6n} | indices tau-n-1 ... tau-1 count n+1; `pre_encoder.py:88` builds 6(n+1) | `\in\mathbb{R}^{6(n+1)}` |

(The row above shows the shape only; it is not a finding.) Then a section "Correct", one line per formula or claim with its evidence pointer, so coverage is visible. Then "Not checkable", each with the reason (no code counterpart, paper not in `References/`, needs a numerical run and which quantity). Where text and code disagree, report both and do not decide which is right.

### 4.5 Sources
Papers: `Thesis-writeup/References/*.pdf`; `References/index.csv` maps cite keys to PDFs; `Writing/refs.bib` holds the keys the text cites. A citation check compares against the PDF in `References/`, not memory. Earlier per-section checks exist and may be used as leads, not as verdicts: `Thesis-writeup/Documentation/CITATION-CHECK-III-C-III-E.md`, `CITATION-LOG.md`, `tasks/handoffs/2026-10-23-codex-verify-section4-math.md` (claims list for Section IV; note it compared against `Clean/`, which this audit does not).

## 5. Assumed but not verified
- That the main pipeline at HEAD is the code that produced the thesis results. If a run's `config.json` or git provenance shows otherwise, report it.
- That the slice line ranges in 4.2 still match the files; follow the labels if lines moved.

## 6. Tried and failed
None for this audit.

## 7. Achieved
Not applicable: no audit has covered the whole thesis against the main pipeline.

## 8. The open question
Nothing blocked.

## 9. Next action
Launch agents A to F in one message (prompt for each: Section 1 task restricted to its slice, its row of 4.2, Sections 2, 4.3, 4.4, 4.5, 12), wait for all six, then do the notation pass and write `Thesis-writeup/Documentation/MATH-AUDIT-2026-10-24-codex.md`.

## 10. Acceptance criterion
Every displayed equation and every inline formula in the audited files appears in the report, either as a finding or in "Correct" or "Not checkable"; every finding has evidence and a smallest fix; every citation check names the PDF page or equation; every code check names `file:line` on the main path.

## 11. Read these first
1. `Thesis-writeup/Writing/util/format.tex` (notation and change marks).
2. `Thesis-writeup/Writing/sections/02_system_baseline.tex` (symbols every later section uses).
3. `scripts/gantry/gantry_interconnect_dynamic.py` (the main path and its imports).
4. `Thesis-writeup/References/index.csv`.
5. `tasks/handoffs/2026-10-23-codex-verify-section4-math.md` Section 4.1 (worked claim list for Section IV).

## 12. Do not
- Edit, build, run, or write git. Write only the audit report and scratch files.
- Read or wait for the Claude audit file.
- Print PDF text to the console (cp1252); extract to a UTF-8 file in the temp folder.
- Quote code with a line number unless `conda run -n GraduationProject python scripts/verify_quote.py <file> <start> <end> <quotefile>` printed MATCH OK for it (quote file in the temp folder).
- Read anything under `kamtin-data/Data Telica/` or `kamtin-data/Telica.mat`; write anything under `kamtin-fp-model/`.
- Use em-dashes (Unicode, `---` or double hyphen) anywhere in the report.

## 13. Operational
PDF reading if needed: a PyMuPDF script in the temp folder, `conda run -n GraduationProject python -I <script>`, output to a UTF-8 file in the temp folder. No other runs. Project rules live in `CLAUDE.md`; the ones that bind this task are restated in Sections 2 and 12.

## 14. Delegation
6 Codex sub-agents (A to F), authorised by the user on 2026-10-24. If your harness cannot spawn sub-agents, run the six slices one after another, each starting from this prompt only and writing its own `audit_<A..F>.md` before the next starts, so that one slice's conclusions do not steer another. No further sub-agents inside them. The lead does the notation pass and the merge itself.
