# Handoff: make the training code match the final method of the thesis (FINAL-RUN-FIXES + revnotes), cleanly, so the final runs can be launched
**From**: session of 2026-10-16 (Section V rewrite) | **Branch**: Augmentation | **Effort suggested**: xhigh (wide code
work across two packages, two independent analyses to reconcile, then implementation)

## 1. Task
Dirk's words: "get a clean code implementation and fix the issues where our model currently doesn't match how we want
the final run". The thesis text (Sections III to V) already describes the FINAL method. Every place where the code of the
2026-10-11 runs differs is listed in `Thesis-writeup/Documentation/FINAL-RUN-FIXES.md` (the most important document; items
1 to 19 plus "Checks before launch") and in the `\revnote{}`s of the thesis sections. Do this in four steps:
1. **Two independent analyses.** Start two separate sessions with the identical brief of section 14. Each produces, on
   its own, (a) a change list: every FINAL-RUN-FIXES item and every revnote mapped to the code location that must change
   (`file:line`), what changes, and its decision entry; (b) a proposal for a minimal, clean implementation. They must not
   see each other's output, so that their results can be cross-referenced.
2. **Meanwhile, in this session:** find out why the OBC runs of the previous campaign finished in about one day although
   the run plan expected 48 h or more (section 8).
3. **Cross-reference** the two results into one table (item, location, agree / differ, the recommended change) and
   present it to Dirk with the OBC-duration finding. One approval from Dirk covers the implementation.
4. **Implement** the approved changes, make a short local smoke run of each thesis tier pass, and prepare the exact list
   of server jobs for the final runs (Dirk submits them himself).
The deliverable is code that runs, not a plan: steps 1 and 3 exist to make step 4 correct.

## 2. Out of scope
- Regenerating any dataset. Dirk decided 2026-10-16 to keep the current data
  (`Thesis-writeup/Data/Coulomb-tanh-and-MSD-lowpass-noise` and `-noise-free`, plus `Multisine-test-twins`): no issue in
  the data was found (section 4).
- Editing thesis text (`Thesis-writeup/Writing/**`). Revnotes are removed by Dirk once the reruns exist.
- Submitting server jobs, and git writes of any kind. Dirk submits and commits.
- New methods or new experiments beyond the listed fixes (for example the 99 % band edge, the T3 low-frequency check,
  new runs for the n_x̄ = 0 detuned pair). Mention a found issue in one line; do not start it.
- `kamtin-fp-model/` (read only) and `kamtin-data/Data Telica/`.

## 3. Where things stand
Branch `Augmentation`, large dirty tree (thesis and docs edits of 2026-10-15/16, not committed). No run in flight.
Previous campaign (runs.tsv rows 201 to 207, D-238): `scripts/gantry/thesis-results/runs.tsv`, launched by
`scripts/gantry/thesis-results/submit.sh` with `run_thesis.sbatch`, which runs
`scripts/gantry/gantry_interconnect_dynamic.py` (hyperparameters in its THESIS block). Logs and results:
`Thesis-writeup/Results/11-10-2026/thesis-results-2/*.out` (R1 = 201, R2 = 202, R3 to R7 = 203 to 207, black-box search
`bb_search_88757_*.out`) and `Thesis-writeup/Results/11-10-2026/thesis_taf_lowpass_noise_linear_map/<job>/` (configs,
reports, checkpoints). Run-to-R mapping: `Thesis-writeup/Code/Evaluation/eval_common.py:27`.
Abbreviations in FINAL-RUN-FIXES: `GD/` = `scripts/gantry/gantry_dynamic/`, `FS/` = `model_augmentation/fit_systems/`,
`SYS/` = `model_augmentation/systems/`.

## 4. Established and verified
- Run configs of R1 to R7: data `Coulomb-tanh-and-MSD-lowpass-noise`, window 0.1 s, seed 1, `G_INIT=zero`, OBC on
  ran Phi only (no Gamma); R1 and R4 share the detuning factors [0.9,1.1,1.1,1.1,0.9,0.9,1.1,1.1,0.9,0.9]
  (`gantry_interconnect_dynamic.py:328-330`, seed 1). Read 2026-10-16 from the config.json files.
- No config records a git SHA (`GIT: unavailable`). The final runs must record one.
- R1 (201) ran on `blade5`, partition `lanai`, log ends Fri Oct 9 13:04 CEST; R2 (202) on `blade6`, partition
  `molokai`, ends Fri Oct 9 08:59 CEST (headers and last lines of the two `.out` files). `submit.sh:44` gives OBC
  72 h on `hawaii,lanai,molokai`; `submit.sh:12` says oahu is slow for OBC (`PS2-METHOD.md` 7.5).
- Final-method decisions already logged: D-248 (live g_aug), D-250 (encoder window ends at tau-1), D-251 (data-based
  physical encoder rows), D-252 (OBC tuples with successor), D-253 (polish vs the error after the basis rebuild),
  D-255 (polish acceptance on the validation error), D-256 (NRMS_e as loss weight, selection and polish rule),
  D-257 (OBC with the offset column Gamma), D-259 (m_Delta scaled by the mean actuator mass; single seed).
- Revnotes to cover: 10 in `Writing/sections/03_augmentation.tex`, 4 in `04_obc.tex`, 3 in `05_setup.tex`. The three in
  Section V are: NRMS_e is not yet produced by `Thesis-writeup/Code/Evaluation` (only RMS in m, `make_results.py`);
  the predicted first-order bias of eq:obc_recovery is not computed (no code; `eval_overlap.py` computes the overlaps
  only); the black box (R8) had its best validation error at the 20000-update limit, and the 80000-update retrain has
  not run (`scripts/gantry/blackbox-cl-trainable/runners/thesis_r8_grid.conf`, selected index 3: n_x 16, width 8,
  rate 3e-3).
- The combination-error table code now scales m_Delta by (m1+m2)/2 (`Thesis-writeup/Code/Figures/res_obc_combinations.py`,
  D-259); the training meter already did (`GD/training.py:425-432`).
- `Thesis-writeup/Code/Setup/` macros and the setup figures are computed on the current dataset; they need no rerun as
  long as the data stay as they are.
- Data checks (2026-10-16, reports `justification-check-data.md` / `-train.md` in that session's scratchpad, summarised
  here): band, amplitudes (scale 1.00), friction regime, noise level, test pairs and twins all hold on the current data.

## 5. Assumed but not verified
- "The OBC runs finished in about one day" (Dirk). Only the end times are known (section 4); start times, epochs run and
  whether training stopped early are not checked. Settled by section 8.
- The cause of the 10.6 % median change of the feedback force when K1 is re-discretised at 4 kHz
  (`Thesis-writeup/Code/Setup/controller_rate.py`, appendix B-C todo). Candidate: a half-sample offset between the
  block-mean force and the controller output in the reduction step (2 sin(pi f 0.1 ms) gives 9 to 19 % over the band;
  only multisine records show it, motion-only 0.6 %). Candidate two: Tustin warping at 4 kHz (D-141). If the first
  holds, the fix belongs in the preprocessing before the final runs. Include it as an item in both analyses.
- That the evaluation scripts can read `Thesis-writeup/Data/Multisine-test-twins/` (they could not on 2026-10-16;
  D-258). Needed for Section VI, not for training.
- Whether the design-choice runs in the appendix are compared on the validation records (recommended, not confirmed
  by Dirk); affects which logs the evaluation must keep (FINAL-RUN-FIXES item 13).

## 6. Tried and failed
- Monkey-patching or wrapping the framework at run time -> rejected by Dirk earlier (memory: copied scripts, not
  monkey patching) -> changes go into the code with the `model_augmentation/` markers of CLAUDE.md.
- Selecting checkpoints by RMS in metres -> biased toward the axes and records with large servo error -> D-256 moves
  loss weight, selection and polish acceptance to NRMS_e (FINAL-RUN-FIXES 13 to 18).
- Polish acceptance with the truth-based meter gate -> uses truth information -> dropped (D-255, item 2).

## 7. Achieved
- Section V and Appendix B of the thesis describe the final experiment (2026-10-16); the code side is what remains.
- FINAL-RUN-FIXES items 3 and 4 ("Checks before launch") are done or harmless (see the file).

## 8. The open question
Why did the OBC runs end after about a day when 48 h or more was expected? Candidates, each with what settles it:
(a) a faster node or GPU (blade5/lanai, blade6/molokai; compare the "GPU info" block and per-epoch timings in the two
logs with a U run and with the PS2-METHOD 7.5 timings); (b) fewer epochs or updates than planned (epoch and update counts
in the logs and config.json against the THESIS block); (c) early stop, polish skipped, or a failed stage that still
printed "done" (read the log tail and the run report); (d) the OBC basis refreshed less often than per epoch, which would
also be a method mismatch (D-254 "per-epoch refresh"; check where the basis is rebuilt). Report which holds, with the log
lines. If (b), (c) or (d), it is a fix for the final runs, not only an explanation.

## 9. Next action
Start the two analysis sessions with the brief of section 14 (in parallel, same text, fresh context), then read the two
OBC logs and their `thesis_taf_lowpass_noise_linear_map/88755*` config and report for section 8 while they run.

## 10. Acceptance criterion
- Every FINAL-RUN-FIXES item (1 to 19 and the launch checks) and every revnote of Sections III to V appears in the
  cross-reference table with one of: implemented (file:line), already correct (evidence), or not needed for the final
  run (reason, Dirk's approval). No item is missing.
- Each implemented fix has its `docs/decisions.md` entry (existing D-number or a new one) before it is implemented.
- A short local smoke run (CPU, a few updates, the existing `smoke_all.sbatch` settings as reference) passes for the
  thesis-obc, thesis-u and black-box tiers, and its config records the git SHA and the NRMS_e selection.
- A job list for the final runs (ids, arms, starts, partitions, wall times) is ready for Dirk, with the run-table rows of
  the run-discipline rule written before launch.

## 11. Read these first
1. `Thesis-writeup/Documentation/FINAL-RUN-FIXES.md`: the list of code changes; the primary document.
2. The `\revnote{}`s in `Thesis-writeup/Writing/sections/03_augmentation.tex`, `04_obc.tex`, `05_setup.tex`: what the
   text promises that the code does not yet do.
3. `Thesis-writeup/Documentation/THESIS-RESULTS.md`, block "Presentation agreed 2026-10-16" and section 4 (run table):
   which runs the final campaign needs.
4. `scripts/gantry/gantry_interconnect_dynamic.py` (THESIS block) and `scripts/gantry/thesis-results/submit.sh`: how a
   run is configured and launched.
5. `docs/decisions.md` D-248 to D-259.

## 12. Do not
- Do not regenerate data or change the data generator (section 2).
- Do not edit the thesis text or remove revnotes.
- Do not submit jobs or run git writes.
- Do not run multi-hour local training; local runs are smoke tests only.
- Do not let the two analysis sessions share files or results before both have finished.

## 13. Operational
- Python: `conda run -n GraduationProject python ...`; long runs with the live-output convention of CLAUDE.md.
- Smoke reference: `scripts/gantry/thesis-results/smoke_all.sbatch`, `test_ps2_pipeline.py`, `test_live_comparison.py`.
- Server: partitions and wall times in `submit.sh` (OBC tier 72 h on hawaii, lanai, molokai; U tier on hawaii, oahu).
- Black box: `scripts/gantry/blackbox-cl-trainable/` (`train_bb_clmap.py`, `runners/run_bb_search.sh`,
  `runners/thesis_r8_grid.conf`); FINAL-RUN-FIXES item 18 moves it to the same loss and selection.

## 14. Delegation
Exactly two analysis sessions, read-only, started together with this brief (no Explore subagents beyond these):

> Read-only analysis; edit nothing, run nothing longer than a few seconds, no git writes. Goal: list everything in the
> training and evaluation code that must change so that the final thesis runs implement the method the thesis
> describes, and propose a minimal clean implementation. Sources, in order of precedence:
> `Thesis-writeup/Documentation/FINAL-RUN-FIXES.md` (items 1 to 19 and "Checks before launch"), then the `\revnote{}`s
> in `Thesis-writeup/Writing/sections/03_augmentation.tex`, `04_obc.tex`, `05_setup.tex`, then `docs/decisions.md`
> D-248 to D-259. Also include: the git SHA in every run config; the 10.6 % controller-rate change of
> `Thesis-writeup/Code/Setup/controller_rate.py` (candidate cause: a half-sample offset in the reduction step; say
> whether it is a preprocessing fix); NRMS_e and the predicted first-order bias in `Thesis-writeup/Code/Evaluation`;
> evaluation access to `Thesis-writeup/Data/Multisine-test-twins/`. Abbreviations: `GD/` = `scripts/gantry/gantry_dynamic/`,
> `FS/` = `model_augmentation/fit_systems/`, `SYS/` = `model_augmentation/systems/`; entry point
> `scripts/gantry/gantry_interconnect_dynamic.py` (THESIS block); black box in `scripts/gantry/blackbox-cl-trainable/`.
> Output, as your final message, two parts. Part A, a table with one row per item: source item, code location
> (`file:line` you read yourself), current behaviour, required behaviour, change, decision entry, risk. Report every
> item, including those you find already correct (with the evidence). Part B, a minimal clean implementation: the
> smallest set of edits, in order, with the `model_augmentation/` markers of CLAUDE.md, and the smoke test that proves
> each. Example row (format only; the line number is illustrative): "FIX-14 | GD/training.py:NNN | selects best checkpoint by mean RMS in m | by mean NRMS_e over
> validation records and axes (D-256) | divide each record-axis RMS by its servo-error std before the mean | D-256 |
> low".

The main session reconciles the two outputs; it does not forward one session's output to the other.
