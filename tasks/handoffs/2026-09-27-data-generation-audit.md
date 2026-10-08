# Handoff: read-only audit of the thesis data generator and the T_AF data (with and without noise) against the original design documents
**From**: session of 2026-09-27 | **Branch**: Augmentation | **Effort suggested**: xhigh (code, reports and 76 data files checked against five long design documents, with independent recomputation in Python)

## 0. How to read this prompt (read first)
- **The goal, in the user's words:** "audit your implementation based on the original files in thesis-writeup/documentation/. I only want an audit not that the session will touch the files."
- **Deliverable: an audit report in this chat, nothing else.** No file in the repository is created, edited, moved, renamed or deleted: not the reports under audit, not the design documents, not the logs, not `tasks/`. Reading files and computing in Python from your own scratchpad are allowed; they are how you get evidence.
- **Stance:** you are an independent examiner in system identification, precision motion control and experiment design, and you distrust the implementation and its own reports. The implementing session wrote this prompt: its statements below are pointers to evidence, not evidence. A check counts when you have read its output or recomputed its number, never because a report says PASS.
- **The specification** is the ORIGINAL design documents, `Thesis-writeup/Documentation/` as committed at `d641059` (read with `git show d641059:Thesis-writeup/Documentation/<file>`), together with the generation instructions `tasks/handoffs/2026-09-26-data-generation.md` (the working-tree version; the committed version is an older draft with a detuned plant that the user replaced before generation). The user decisions of section 4 change that specification where they say so, and only there.

## 1. Task
Audit whether the generator in `Thesis-writeup/Code/Data/`, the data in `Thesis-writeup/Data/Coulomb-and-MSD/` and `Thesis-writeup/Data/Coulomb-and-MSD-noise-free/`, and the verification material in `scripts/gantry/thesis-data-verification/` do what the original design documents specify, as changed only by the user decisions of section 4. Answer each question with evidence:
1. **Traceability.** Is every record of the T_AF block of `DATA-DESIGN.md` 7.9 in `tdg_manifest.m` with the design's settings (class, set, Y range, levels, profile, controller, noise), or removed by a named user decision? Is every number the generator uses traced to a source that actually says it: band, line grid, line count, amplitude per channel, velocity and acceleration levels, Y ranges, jerk times, absorber, friction, `V_BRK`, noise target, roll-off, controller, record length, sample rate?
2. **Code correctness.** Does the code do what `SPEC.md` says, and does `SPEC.md` say what the documents decide? In particular: the noise path (formula and loop of `NOISE-INJECTION.md` and D-218, injection point, stage versus logical coordinates, which controller the calibration uses, also for E5 with K2-g+); the multisine (D-219); the post-simulation limit check on the truth (`DATA-DESIGN.md` section 10); the controller designed at Y = 0 (5.11); the E6 kinematics (D-210); the seeds (the noise seed separate, so a noise-free record shares every other seed with its noisy one).
3. **Data.** Do the saved files contain what the manifest says? Recompute from the files, not from the config (section 7).
4. **Honesty of the checks.** Does each check C1 to C15 test what the reports claim it proves? Two criteria were revised after a first failure (C12 from 1e-9 N to 1e-6 N; C13 from a single draw to a 40-draw spread test): judge whether each revision corrects a wrong criterion or moves a goalpost.
5. **Completeness** against the acceptance criteria of section 10 of the generation handoff, one by one.
6. **Deviations.** Every difference between the original documents and what was built, classified as: (a) a user decision of section 4 (cite it); (b) a conflict resolved by the documents' own precedence rule and recorded in `SPEC.md` sections 1 and 6; (c) an implementation choice (`GENERATION-REPORT.md` M1 to M27) that leaves the design unchanged; (d) undocumented, unjustified, or changing data without a user decision. Class (d) is the finding that matters most.
7. **The edits of 2026-09-27 to `DATA-DESIGN.md` and `RESULTS-DESIGN.md`** (`git diff d641059 Thesis-writeup/Documentation/`), made on the user's instruction for U6 and U7 although the generation handoff forbade edits there. Do they state exactly U6 and U7 and nothing more, and do they leave inconsistencies elsewhere in those two documents or in the other three?
8. **Stale statements.** Any statement in code comments, `SPEC.md`, `IMPLEMENTATION.md`, `GENERATION-REPORT.md`, `VENDORED.md`, `REFERENCE.md` or `REVIEW.md` that contradicts the current code, data or decisions (record counts, paths, realisations, noise-only twins).

## 2. Out of scope
- Any change to any file (section 0). Proposed fixes go in the report as text.
- Running MATLAB or Simulink, including the existing checks: they write result files, logs and Simulink cache files, and the disk is nearly full (section 13).
- Re-opening a user decision of section 4. You may show that one of them makes a result undecidable or a report claim wrong, and say which, but do not propose reversing it.
- Redesigning the data set, the band or the noise model. A design flaw found on the way is a note for the user, marked (?).
- Generating the 10 missing FRF records; training; results; real Telica data.

## 3. Where things stand
- Branch `Augmentation`, last commit `d641059` (2026-09-26 20:34). The tree is dirty with other sessions' work. The objects under audit are untracked (`Thesis-writeup/Code/Data/`, `Thesis-writeup/Data/`, `scripts/gantry/thesis-data-verification/`), so git holds no history for them.
- No MATLAB job is running.
- `Thesis-writeup/Code/Data/` (36 files):
  - driver `generate_thesis_data.m`, helpers `tdg_*.m`, modified copies `gtd_*.m` (every change marked `THESIS-DATA`);
  - `vendor/` (verbatim runtime copies plus the source `.slx`), `model/gantry_tdg_2025a.slx`, `noise/` (`noise_target.mat`, `export_noise_target.py`);
  - reports `SPEC.md`, `IMPLEMENTATION.md`, `GENERATION-REPORT.md` (user decisions U1 to U8, implementation decisions M1 to M27, findings F1 to F9, proposed decision entry D-220), `VENDORED.md`, `REFERENCE.md`.
- Data: 38 noisy and 38 noise-free records, a `MANIFEST.csv` per version. Training 18, Validation 6, Test 14 (I1 to I6, E1 to E6, TF-1_r1, TF-3_r1). 12 s at 20 kHz, double precision, MAT v7, 40 to 45 MB per file.
- `scripts/gantry/thesis-data-verification/`:
  - `checks/`: check scripts, result `.mat` files, the noise calibration figure;
  - `logs/`: every run's `<run>/attempt*.log`, `generation_log.csv`, `generation_failures.csv`, `superseded/`, and `after_move.log` (checks rerun after the move);
  - `REVIEW.md`: the independent phase D review and how each finding was handled;
  - `vendor/original/`: untouched source copies, the reference for diffs against `Matlab-scripts/`;
  - `noise-twins/`: 4 noise-only twins, not part of the data set; `tools/`: launchers and watchdog.
- `Thesis-writeup/Data/Detuned-with-Coulomb-and-MSD/` is empty by instruction (the plant is never detuned).

## 4. Fixed inputs: user decisions (audit their implementation, not the choice)
From the generation handoff (2026-09-26):
- One truth T_AF: baseline + Coulomb rail friction + payload absorber; never the absorber alone; never detuned.
- Two versions differing only in the noise: noisy (force noise of `NOISE-INJECTION.md`, D-218, own realisation per record) and noise-free (d = 0, everything else identical).
- Noise is a force on the motor input. One controller K1 designed at Y = 0. Maximum acceleration 30 / 50 m/s² (X / Y). Own phase realisation per record.

Given by the user during generation (`GENERATION-REPORT.md` section 4, `SPEC.md` section 7):
- U1: folders `Coulomb-and-MSD/` and `Coulomb-and-MSD-noise-free/`.
- U2: the readings listed in `SPEC.md` section 7 (for example TF records as 12 s standstill records, the I4 Y range, the I6 / E4 dwell, the VA-T1 / VA-T2 settings).
- U3: the noise roll-off above 300 Hz is a raised cosine to zero at 330 Hz.
- U4: keep the Karnopp friction truth, although the noise shifts stick-slip breakaway (F1).
- U5: generation in stages with a 2 GB disk reserve.
- U6: one realisation of training and validation instead of the design's three.
- U7: the noise floor of a record is its noisy minus its noise-free version; the 4 noise-only twins leave the data set.
- U8: the verification material moved from `Thesis-writeup/Code/Data/` to `scripts/gantry/thesis-data-verification/`, so the checks and `REVIEW.md` that the handoff expected in `Code/Data` are there by decision.

Everything else in `GENERATION-REPORT.md` (M1 to M27, F1 to F9) is the implementing session's own work and is under audit.

## 5. Declared open by the implementation (judge each; do not adopt the report's framing)
- 10 of the 12 FRF references (TF-1 r2 to r4, TF-2 r1 to r4, TF-3 r2 to r4) are not generated (disk).
- Noise floors of the multisine records are 2 to 3 times the calibration target; those of the multisine-free records I4 and E5 are micrometres (F1). The calibration check C2 was done with friction off, at standstill.
- Y-loop phase margin 28.5° with K1 and 25.0° with K2-g+ (E5).
- The code hashes stored in the saved records predate later code edits (M27).
- Jerk time 30 ms against 28.4 ms, unconfirmed (?).
- `EXCITATION-VALIDATION.md` section 19 items (2) and (3) are not done.
- The `.mat` files are not git-ignored.
- The U6 line of `GENERATION-REPORT.md` section 4 still says "52 noisy + 48 noise-free records"; the record table now has 48.

## 6. Tool pitfalls met in this project
- `scipy.io.loadmat` returns the storage type of a field (for example an integer type for an integer-valued double) unless `mat_dtype=True`: always pass it.
- `conda run python -c` cannot take a multi-line script: write the script to your scratchpad and run the file.
- On 2026-09-27 a shell pipe failed with "No space left on device" (C: at 0 to 0.3 GB free, recurring, page file): keep outputs small and never copy data files.

## 7. Acceptance criterion (the audit is done when)
- Every question of section 1 is answered with evidence.
- Every record of the original `DATA-DESIGN.md` 7.9 T_AF block is accounted for: generated, removed by a named user decision, or missing with the reason.
- These are recomputed from the data files, with the numbers in the report:
  - for every noisy / noise-free pair: identical in every field outside the noise path (references, multisine, seeds, truth parameters), and d = 0 in the noise-free file;
  - the delivered multisine of at least one record per class: line frequencies, line count, band, rms per logical channel against A_prod;
  - the noise: the force d of a standstill record against the target in `noise/noise_target.mat` (and that file traced to `scripts/gantry/closed-loop-noise/outputs/g1_spectra/g1.npz`); noisy minus noise-free error levels of the standstill records against the calibration target 3.370 / 3.696 / 1.086 nm;
  - peak velocity, acceleration and force of every record against the limits used, on the simulated truth;
  - record length, sample rate, precision, and the stored truth parameters against the design values.
- No finding rests only on a report's own statement.

## 8. The open question
Nothing blocked.

## 9. Next action
Read the five original documents at `d641059` and the generation handoff; then `SPEC.md`; then audit in the order of section 1, computing the data checks of section 7 in Python from your scratchpad; report in chat in the format of section 10 below.

## 10. Output format
Start with a verdict paragraph: is the data set, as generated, a correct implementation of the original design plus the user decisions, and fit to feed results R1 to R8; which findings stand in the way. Then every finding you have, of every severity, most severe first, numbered, each in this shape (hypothetical example, format only):

> **A4. Major. The multisine rms of channel Y is 7 % below A_prod in the validation records.**
> Where: `Thesis-writeup/Code/Data/gtd_make_multisine.m` (scaling line); `DATA-DESIGN.md` section 0.
> Evidence: recomputed from `Validation/VA-S1_r1.mat`: 167 N against 180 N; training records 180 N.
> Class: (d) data change without a user decision.
> Fix (proposed, not applied): scale per record as in training and regenerate the 6 validation records in both versions.

Severity: Blocker (data wrong or unusable for a result), Major (a report claim or a deviation is wrong or undocumented), Minor (stale or misleading text), Note (design question for the user, marked (?)). Then two tables: the handoff's section 10 criteria (met / partly / not met, with the reason), and the record accounting of section 7. Line-number quotes follow the code-quote verification rule.

## 11. Read these first
1. `git show d641059:Thesis-writeup/Documentation/DATA-DESIGN.md`: sections 0, 2, 5, 7 (7.9 manifest), 9 (Q2), 10, 13. Then `NOISE-INJECTION.md` (all), `EXCITATION-VALIDATION.md` (sections 0, 16, 19), `EXCITATION-REVIEW.md`, `RESULTS-DESIGN.md`, all at `d641059`.
2. `tasks/handoffs/2026-09-26-data-generation.md`, sections 4, 5, 10, 12.
3. `Thesis-writeup/Code/Data/SPEC.md`, then `GENERATION-REPORT.md` sections 2 to 4.
4. `docs/decisions.md`: D-209, D-210, D-218, D-219 (D-219 supersedes D-216); grep `### \[D-2`.
5. `scripts/gantry/thesis-data-verification/REVIEW.md` and `logs/after_move.log`.

## 12. Do not
- Create, edit, move, rename or delete any file in the repository, or run git commands that change state (add, commit, stash, checkout, restore, reset, clean).
- Run MATLAB or Simulink.
- Write anything into the repository from Python; scripts and small results go to your scratchpad only.
- Read `kamtin-data/Data Telica/` with any tool; touch `kamtin-data/Telica.mat`.
- Propose reversing a user decision of section 4.

## 13. Operational
- Python: `conda run -n GraduationProject python <scratchpad script>` (numpy, scipy). Load one record at a time; a record loads in seconds.
- Disk: check `(Get-PSDrive C).Free` in PowerShell before a longer computation. If a tool fails for lack of space, tell the user in one sentence and continue with reading.
- Original documents: `git show d641059:<path>`; the edits since: `git diff d641059 Thesis-writeup/Documentation/`.

## 14. Delegation
None. The audit is the reading; do it inline. No Workflow tool.
