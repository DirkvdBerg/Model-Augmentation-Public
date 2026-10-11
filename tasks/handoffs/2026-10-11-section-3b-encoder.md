# Handoff: go through thesis Section III-B (encoder and initial state) with Dirk, paragraph by paragraph
**From**: session of 2026-10-11 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task
Help Dirk finish Section III-B, "Encoder and initial state" (`Thesis-writeup/Writing/sections/03_augmentation.tex`,
label `sec:aug_encoder`, about lines 393 to 530), the same way III-A was done on 2026-10-10/11: he reads a
paragraph, asks what it should establish or whether a statement holds, you answer briefly with the reason
and the source, and you propose changes in chat; he decides and either writes the sentence himself or asks
you to put a shown proposal in. The goal is a III-B that Dirk understands and can defend: every claim checked
against the encoder paper or the code, every choice with its reason, in his own wording.

## 2. Out of scope
- III-A (done), III-C, III-D, III-E, Sections IV to VIII: separate sessions. Exception: a label or reference
  that III-B changes elsewhere, after asking.
- Code changes and reruns (`Thesis-writeup/Documentation/FINAL-RUN-FIXES.md` is for a later session).
- Compiling `main.tex`; git write operations.

## 3. Where things stand
- Branch `Augmentation`; Section III last committed as `724f79a`, later edits from this session are
  uncommitted (III-A wording, the non-unique-split paragraph moved to III-D, `04_obc.tex:70` reference now
  `sec:aug_closed_loop`, a Section V bullet on n_x̄ = 0).
- A parallel handoff for the intro, todos, appendix and Sections V to VIII exists:
  `tasks/handoffs/2026-10-11-thesis-feedback-draft.md`. Do not do that work here.

## 4. Established and verified
- **Encoder code** `model_augmentation/fit_systems/pre_encoder.py::linear_encoder_init_aug` (lines 306 to 500):
  W^b is the noiseless reconstructability map, encoder paper Eqs. 16 to 17 (`A_n @ O_inv`,
  `-A_n @ O_inv @ Gamma_n + gamma_n`); stacking is oldest sample first, matching χ_τ; W^a is one Xavier-uniform
  draw, gain 1 (`wa_encoder_init='xavier_uniform_gain1'`, Hoekstra 2026 Eq. 31); ψ̃ has a zero output layer.
- **Window** includes sample τ: `GD/model.py::get_encoder_dims`, `na_right = nb_right = 1`, n = 29 (thesis runs).
- **Linearisation** `GD/model.py:207` calls `gantry_linearize_and_discretize(dt=TS_NEW)` with the NOMINAL
  parameters (= the simulated truth) -> FINAL-RUN-FIXES item 1, `\todo` at the end of the W^b paragraph.
- **Encoder paper** (`literature/augmentation/Encoder initialisation methods in the model augmentation setting.pdf`):
  Sec. 2 assumes a stable baseline; footnote 1 puts co-estimation out of scope; Sec. 3.1 Eqs. 10 to 17 noiseless
  map, and after Eq. 17: with white Gaussian measurement noise Eq. 15 is a maximum-likelihood estimator;
  Eqs. 18 to 29 process-noise version (needs gain K); Eqs. 30 to 32 linearisation at an equilibrium;
  Sec. 3.2 data-based alternative; Sec. 4.2 linearises "around the 0 point"; Sec. 4.4: model-based ~1.5 ms vs
  data-based ~4.8 s, better starting values, comparable final accuracy.
- **Hoekstra 2026** (`literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`): Eq. 22e encoder
  reads up to τ-1; Eq. 30 fits the baseline encoder to states simulated with the baseline (open loop, recorded
  input); Eq. 31 Xavier for ψ_aug.
- **Noise**: thesis data `Coulomb-tanh-and-MSD-lowpass-noise` (D-238), encoder (measurement) noise about 1e-8 m
  rms per axis, low-order coloured model (`Thesis-writeup/Documentation/NOISE-INJECTION.md` Sec. 10).
- **Agreed reasons in the text**: Y = 0 because K1 is designed there (D-221) and frozen-Y baseline is LTI;
  measurement noise and no known K, so Eqs. 15 to 17; coloured noise, so W^b is a starting value, not optimal.

## 5. Assumed but not verified
- **The D5 reason (model-based over data-based)** in the paragraph after the W^b `\todo`: the text says Jan's
  simulation drifts on X and Y. True only for the OPEN-loop simulation of 2026 Eq. 30; a closed-loop baseline
  simulation would not drift (III-C). A `\todo{Revise: ...}` is in the text with the candidate "the closed-loop
  alternative would need an extra simulation, which the model-based map avoids". Dirk has not chosen yet.
  Note: the draft bullet near line 401 repeats the drift claim; it prints only in the draft build.
- **"It leaves co-estimation ... out of scope"** (about line 512): "It" follows a sentence about our noise, so it
  reads as if our map leaves co-estimation out. Probably meant "The paper". Not yet raised with Dirk.
- Whether `eq:aug_lin` is still wanted at its current size (three-term block): agreed earlier as the gantry's own
  math, not re-reviewed by Dirk.

## 6. Tried and failed
- Rewriting Dirk's sentences for clarity -> rejected as AI-sounding -> replaced his wording -> rule in memory
  `feedback_keep_dirk_wording.md`.
- Editing the .tex while his buffer was unsaved -> save conflict -> ask "saved?" before every edit.
- Citing Jan from memory -> wrong section numbers twice (2025 EJC vs 2026; Sec. 2 vs 3) -> read the PDF passage
  in this session before stating "Jan says".
- Arguing "simulation drifts" without the loop -> Dirk caught that the model runs in closed loop -> state for
  every stability or drift claim whether it is open or closed loop.

## 7. Achieved
- III-B content as agreed on 2026-10-09: encoder window display, linear-plus-correction display, linearisation
  display `eq:aug_lin`, reconstruction display `eq:aug_wb` with its where-clause, noise paragraph, adaptations.
- W^a init and ψ̃ zero output moved to the initialisation display `eq:aug_init` in III-D.

## 8. The open question
None blocking. Dirk drives the order; the open todos in III-B are: D5 drift revision, E1 W^b fix note,
figure history labels, VERIFY bib for `hoekstra2026encoder`.

## 9. Next action
First read, in full, before answering Dirk on any III-B content (Dirk's explicit request: read the relevant
papers and read and understand the code):
- **Papers** (read the PDF pages, not summaries): the encoder paper, Sec. 2 to 4 (pp. 2 to 6); Hoekstra 2026
  Sec. 5.3 and 5.4 (pp. 9 to 10: normalisation, Eqs. 29 to 31 initialisation, encoder initialisation); Beintema
  2023 SUBNET Sec. 2 to 3 (`docs/references.md` gives the path for `beintema2023subnet`).
- **Code**, traced end to end for the thesis run (`THESIS_*` block of `scripts/gantry/gantry_interconnect_dynamic.py`):
  `scripts/gantry/gantry_dynamic/model.py::get_encoder_dims` and `build_model` (encoder injection, lines 39 to 240);
  `model_augmentation/systems/gantry_linearization.py::gantry_linearize_and_discretize` (what is linearised, at
  which Y and parameters); `normalize_linear_ss_matrices` (where it is defined, called in `build_model`);
  `model_augmentation/fit_systems/pre_encoder.py::linear_encoder_init_aug` (map construction and `forward`, the
  D-055 offsets); `scripts/gantry/gantry_dynamic/data.py::compute_normalization` (the scales W^b is built in).
Then give Dirk, in chat, a short summary (bullets) of how the encoder is built in the code and which paper
equation each step implements, and list any place where the III-B text and the code or paper disagree.
After that, ask which III-B paragraph he wants to start with. When he pastes a paragraph, answer what it
establishes, whether each claim is correct against what you read, and give fixes of at most five words or one
line-format proposal.

## 10. Acceptance criterion
Before content work: the code summary of section 9 names, for each encoder step, the file and function and the
paper equation it implements. Then: every sentence in III-B is one Dirk accepts as his and can defend: each claim traced to section 4 or to a PDF
passage read in the session, no open `\todo` that Dirk has not explicitly chosen to keep, and no line that
overflows the column (test standalone in the scratchpad if a display changes).

## 11. Read these first
1. `Thesis-writeup/Writing/sections/03_augmentation.tex`, III-B (from `\subsection{Encoder and initial state}`).
2. The papers and code listed in section 9 (mandatory, before the first content answer).
3. `Thesis-writeup/Writing/README.md`, Math standard and authorial-voice rules.
4. `Thesis-writeup/Documentation/AUGMENTATION-INITIALIZATION.md` Sec. 4 (encoder initialisation notes; check
   against the code, the code wins).
5. `Thesis-writeup/Documentation/FINAL-RUN-FIXES.md` item 1.

## 12. Do not
- Rewrite Dirk's sentences; edit without his request; edit without asking "saved?" first (memory rules).
- Re-introduce the "drift" reason without the open-loop qualifier.
- Touch the W^b code or rerun anything.

## 13. Operational
`conda run -n GraduationProject python <scratch file>` for PDF text (PyMuPDF `fitz`); Read tool for PDF pages with
equations. After any edit tell Dirk to run "File: Revert File" in VS Code.

## 14. Delegation
None.
