# Handoff: decide from the literature which results the thesis shows in the main text and which in the appendix
**From**: session of 2026-10-15/16 (Sections V and VI) | **Branch**: Augmentation | **Effort suggested**: high (reading
four papers' experiment sections and mapping them onto existing results; no derivation, no runs)

## 1. Task
Independently determine what the Results section (VI) of Dirk's thesis should show in the main text and what in the
appendix, taking the structure of the experiment and results sections of Jan Hoekstra's papers as the primary model,
with Gyorok's two papers and Drenth et al. 2025 as secondary references. Read the papers yourself (rendered pages,
Read tool with `pages`). Produce, in chat, for Dirk: (a) for each paper, how its results are organised (which tables
and figures, which metric, what each one demonstrates), with page and table/figure numbers; (b) your own proposal for
Section VI: per research aspect, the claim, the table or figure that carries it, the metric, and the models compared;
(c) the list of appendix results; (d) only AFTER (a) to (c) are written, a comparison with the current session's
proposal in section 7 below: where you agree, where you differ and why. Interactive: present, wait for Dirk.

## 2. Out of scope
- Editing any file under `Thesis-writeup/` (sections, figures, tables, documentation) or `docs/`: this is advice only.
- Training or evaluation runs, new scripts. Computing new numbers (an "oracle model" row was proposed; it is not computed).
- Section V (Experiment design) text, Sections II to IV, notation sweep (Jasper's point 1): other sessions.
- Real Telica data (out of the main results, Dirk 2026-09-24).
- Reading `kamtin-data/Data Telica/` (CLAUDE.md data policy).

## 3. Where things stand
Branch `Augmentation`, many uncommitted thesis edits (Sections III to VI, appendix, docs). No run in flight. All grey-box
runs (R1 to R7) and the black box (R8) of 2026-10-11 exist; all will be rerun with the fixes of
`Thesis-writeup/Documentation/FINAL-RUN-FIXES.md` (items 1 to 18), so the current numbers are indicative, not final.

## 4. Established and verified
- Research question and four aspects: `Thesis-writeup/Documentation/THESIS-RESULTS.md` Sec. 1 (D-241). Aspect 1 baseline
  form (LPV vs LTI), 2 augmentation structure (added states, PS initialisation), 3 interpretability (OBC vs unconstrained),
  4 generalisation (held-out positions and motions, against a black box).
- Data: one simulated dataset `Thesis-writeup/Data/Coulomb-tanh-and-MSD-lowpass-noise` (truth = LPV baseline + payload
  absorber 150 Hz anti-resonance / 212 Hz coupled mode + tanh rail friction, closed loop with controller K1, noise at a
  representative level); noise-free twin `Coulomb-tanh-and-MSD-noise-free` gives the noise floor. Splits 18 training /
  6 validation (selection only) / 11 test: I1, I2 interpolation (multisine), I3 to I6 operational motion (no multisine),
  E1 to E4, E6 extrapolation (E1 with multisine). E5 (changed controller) is out (Dirk 2026-10-15). Section V draft:
  `Thesis-writeup/Writing/sections/05_setup.tex`.
- Run map: `Thesis-writeup/Code/Evaluation/eval_common.py` (`RUNS`): R1 OBC PS n_a 2 detuned (final model), R2 OBC PS
  correct start, R3 U plain n_a 2 correct, R4 U PS n_a 2 detuned, R5 U PS n_a 2 correct, R6 U plain n_a 0 correct, R7 U PS
  n_a 8 correct, R8 black box. Run plan and per-step intent: `THESIS-RESULTS.md` Sec. 3 and 4.
- Primary metric (D-256, `docs/decisions.md`, last entry): NRMS_e,j = RMS(y_hat_j - y_j) / RMS(y_j - r_j) per test record
  and axis, beside the RMS in m and the noise floor; records with and without multisine reported separately. Defined in
  `05_setup.tex` (eq:setup_rms). Reason: the reference is given, so the servo error is what a closed-loop model must
  predict; RMS(y) is about 0.17 m against RMS(e) of 1e-5 to 6e-5 m.
- Current-run numbers (2026-10-11 runs, scratch computation, indicative): final model 3 to 8 % NRMS_e on every test record
  and axis; baseline (true parameters) on Y 102 / 132 / 131 % on I1 / I2 / E1 and 2 to 4 % on records without multisine;
  final model removes 93 to 95 % of the baseline error on Y on the multisine records and is equal or up to about 1.5
  points worse on the records without multisine; black box about 100 % on every record without multisine. Mean over
  I1 to I6 [X1, X2, Y] %: baseline 9.3 / 10.0 / 41.2, plain n_a 2 6.4 / 6.0 / 34.3, PS n_a 2 5.9 / 5.7 / 4.6, final
  6.2 / 6.1 / 5.0, black box 85.2 / 84.9 / 79.1. Full record in `06_results.tex` header comment.
- Existing result artefacts (`Thesis-writeup/Results/`, figures in `Writing/figures/`): res_lpv_lti, res_structure (table
  in m, 11 rows), res_ps_spectrum, res_time_trace (I1), res_obc_combinations, res_obc_overlap, res_generalisation,
  res_frf (open-loop Bode F_Y to Y, final model vs system vs baseline, mean over Y = -0.39 to 0.39 m), res_frf_full (3 x 3).
  FRF finding: the final model reproduces the absorber anti-resonance and coupled mode (212.0 to 212.2 Hz, zeta about 0.06,
  system 211.9 Hz, zeta 0.043); the X to Y cross terms (about 37 dB below F_Y to Y) show a small spurious resonance near
  212 Hz that the system lacks. OBC result: parameters drift in both arms, OBC not less (`06_results.tex` VI-C outline).
- Supervisor feedback, Jasper (mail 2026-10-15, translated): (1) variables often not introduced, col() operator tricky;
  (2) a very complicated set of experiments, hard to explain in detail; be pragmatic, leave out what the story does not
  need; e.g. does table 1 (the 11-row results table) really need all those models to explain the contributions?;
  (3) explain some experiment-design items by plotting them in an appendix instead of text; (4) the I1 figure is very
  impressive but table 1 does not show it; relate the error to a realistically achievable RMS; (5) show that parameter
  recovery works for the model without added states. Annotated PDF:
  `C:\Users\20203253\Downloads\Feedback-Jasper-1-Model Augmentation for a Dual-Gantry High-Precision Motion Systemv2.pdf`
  (pages 9 and 10 cover Sections V and VI).

## 5. Assumed but not verified
- That NRMS (Gyorok 2025) and BFR (Drenth Eq. 22) are normalised by the output variation: only axis labels and the equation
  number were seen; read the definitions.
- That a detuned n_a 0 run (U and OBC) would show recovery (Jasper point 5): R6 starts at the true values, so no current
  run tests it; whether to add it is Dirk's decision.
- The cause of the spurious FRF cross terms (routing of all rows, D-103) is read from the model structure, not measured.

## 6. Tried and failed
- Reporting one mean over I1 to I6 in metres -> Jasper read the table as unimpressive while the I1 figure is 15x better
  than the baseline -> the mean mixes multisine records (large gain) with motion-only records (no gain) and metres have no
  reference -> D-256.
- Ratio of means as "error removed" (1 - 5.0/41.2 = 88 %) -> misleading, dominated by I1 and I2 -> report per record group.

## 7. The current session's proposal (read only after writing your own (a) to (c))
Main text, one result per aspect, NRMS_e with floor: A LPV vs LTI figure; B table baseline / n_a 0 / plain n_a 2 / PS n_a 2,
I1 time trace, FRF F_Y to Y, "error removed" per record group; C table in Gyorok 2026 Table 1 style: U vs OBC from the
detuned start with NRMS_e, parameter error and the physical part alone (Gyorok 2025 Table 1), plus the n_a 0 recovery if
run; D NRMS_e per I/E pair for baseline, final model, black box. Proposed but not computed: an "oracle" row (baseline
plus the true absorber, true parameters, no friction) as the realistically achievable reference. Appendix: full structure
table (all runs), error spectrum, ten combinations per parameter, network overlap, black-box search table, full 3 x 3 FRF,
setup justification figures (multisine band, coverage).

## 8. The open question
Which results carry the four claims with the fewest tables and figures, as Hoekstra does. Candidates and the evidence that
chooses: Hoekstra's own section structure (primary), Gyorok's interpretability evidence (parameter error and physical
part alone), Drenth's train / val / test table.

## 9. Next action
Read the experiment and results sections of the four papers below, then write (a) to (d) of section 1 in chat.

## 10. Acceptance criterion
Every proposed main-text item maps to exactly one research aspect and one existing artefact (or is marked "needs a new
run/computation"); every paper-based statement carries the page and table/figure number it comes from; the comparison
(d) names each disagreement with the current proposal and its reason.

## 11. Read these first
1. `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`, Sec. 6 and 7 (pages 10 to 14): primary model.
2. `literature/augmentation/hoekstra2025_lfr-augmentation-ejc.pdf`, Sec. 4 (pages 4 to 7).
3. `literature/Orthogonality/gyorok2026_orthogonal-by-construction-augmentation_IFAC-JSC_arXiv2511.01321.pdf`, Sec. 5
   (pages 6 to 9), and `literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf`
   (Gyorok et al. 2025 L4DC), Sec. 5 (pages 8 to 11).
4. `literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf`, Sec. 6 (pages 5 and 6).
5. `Thesis-writeup/Documentation/THESIS-RESULTS.md` Sec. 1 to 3 and `Thesis-writeup/Writing/sections/06_results.tex`.

## 12. Do not
- Edit files; start runs; compute numbers.
- Treat the current numbers as final (they are 2026-10-11 runs, to be rerun).
- Propose Telica real-data results or the E5 controller-transfer case.

## 13. Operational
Read-only. PDFs via the Read tool with `pages`. Python only if a PDF text extraction is needed:
`conda run --no-capture-output -n GraduationProject python -I <script>` (cp1252 console: write UTF-8 to a file).

## 14. Delegation
None. The papers must be read by this session itself (the comparison depends on it).
