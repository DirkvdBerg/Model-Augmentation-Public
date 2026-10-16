# Handoff: plan (then generate) multisine twins of the motion-only test records, matched to the existing thesis data
**From**: session of 2026-10-16 (Sections V and VI) | **Branch**: Augmentation | **Effort suggested**: high (reading the
MATLAB generator end to end and planning a data extension that must match the existing records exactly)

## 1. Task
Plan, and after Dirk approves the plan, generate additional TEST records for the thesis dataset: a multisine twin of
each motion-only test record that forms an interpolation/extrapolation pair, so that every I/E pair also tests the
learned augmentation (whose gain is in the absorber band, 106 to 297 Hz). Records to twin: I3, I5, I6, E2, E3, E4, E6
(same motion profile, same Y_op, same controller K1, same noise model; only `excitation` becomes the multisine of the
existing records). Every generation setting must match how the existing multisine records were made: band and 0.5 Hz
line grid, A_prod amplitudes, best-of-30 crest-factor selection, yaw-budget sizing of the anti channel, record length,
noise model (lowpass, D-238) with its own new noise seed, noise-free twin, post-simulation limit check and the
scale-down rule (DATA-DESIGN section 10). First read the generator and present the plan in chat: what changes where,
the record names and seed keys, where the files go, how the loader and MANIFEST see them, the limit risk per record
(multisine on top of fast moves), runtime and memory. Wait for Dirk's approval before generating anything.

## 2. Out of scope
- Changing or regenerating any EXISTING record, training or validation data, or the noise model.
- Training, evaluation, figures, thesis text.
- `kamtin-fp-model/` (read only), `kamtin-data/Data Telica/` (never read).
- E5 (changed controller): out of the thesis results; do not twin it.

## 3. Where things stand
Branch `Augmentation`, uncommitted edits elsewhere. No run in flight. Dataset complete: 48 records per version (18
training, 6 validation, 12 test, 12 FRF references), `Thesis-writeup/Data/Coulomb-tanh-and-MSD-lowpass-noise` (noisy,
D-238) and `Thesis-writeup/Data/Coulomb-tanh-and-MSD-noise-free` (shared noise-free twin). Final training reruns are
planned on this dataset (`Thesis-writeup/Documentation/FINAL-RUN-FIXES.md`); the new records are evaluation-only.

## 4. Established and verified
- Generator: `Thesis-writeup/Code/Data/generate_thesis_data.m` (driver; one pass per record, noisy then noise-free with
  the same multisine scale; existing files are never overwritten; `opts.simulator='replica'` steps the loop in plain
  MATLAB with about 1 GB RAM, D-231; `opts.out_dir` writes elsewhere), `tdg_config.m` (all constants: band
  `f_low 106, f_high 297, f_step 0.5`, `A_sym 240, A_Y 180`, `yaw_budget 2e-3`, `n_ms_candidates 30`, noise_model
  'lowpass', output folder names), `tdg_manifest.m` (record list, seeds from keys `TAF|set|rec|rN|ntK|ms/sp/noise`
  through `tdg_seed.m`, collision assert), `gtd_make_multisine.m` (best-of-30 crest factor; anti channel scored on the
  closed-loop yaw), `tdg_build_inputs.m`, `tdg_post_check.m`, `tdg_write_manifest.m`.
- Existing test records I3 to I6 and E2 to E6 have `excitation 'none'`; I1, I2, E1 and all standstill/sweep/Lissajous/
  S-curve training records use the multisine at scale 1.00 (MANIFEST.csv of the lowpass folder: every `scale` is 1.00).
- Training-side loader: `scripts/gantry/gantry_dynamic/data.py::record_files` builds the splits from MANIFEST.csv
  `block` (training / validation / test-I / test-E; `frf` is not a rollout record). New records must not enter training
  or validation, and the evaluation scripts (`Thesis-writeup/Code/Evaluation/`) must be able to find them.
- Coverage of the existing references per record: `Thesis-writeup/Results/setup_coverage.csv` (E6 reaches 131 % of the
  X acceleration limit; E3 100 %; E4 98 % velocity).

## 5. Assumed but not verified
- That the multisine on top of the fast moves (E3 at 100 % acceleration, E4 at 98 % velocity, E6 at 131 % X
  acceleration) passes the force and stroke post-check at scale 1.00; if not, the generator scales the multisine down in
  0.1 steps, and the realised scale must be reported (it changes the excitation level against the I partner).
- That adding records with new `rec` names to the existing version folders and MANIFEST keeps every existing record and
  seed unchanged (check `tdg_write_manifest.m` and the collision assert), versus writing them to a new folder. Dirk's
  standing rule: a new dataset gets a new folder; whether a test extension counts as new is his decision (ask).
- Runtime per record (about tens of seconds to minutes with the replica; check `GENERATION-REPORT.md`).

## 6. Tried and failed
None for this task. General: MATLAB batch pitfalls (exit hang, 260-char Simulink path, page file) are in
`Thesis-writeup/Code/Data/IMPLEMENTATION.md` and the user's memory; one MATLAB job at a time.

## 7. Achieved
None for this task.

## 8. The open question
Where the twins go and how they are named: (a) same version folders, new blocks (for example `test-Ims` / `test-Ems`) in
MANIFEST, rec names like `I3M`; (b) a new pair of version folders holding only the twins. Evidence that chooses: whether
(a) can be done without touching any existing file other than MANIFEST.csv, and how the evaluation scripts read records.

## 9. Next action
Read the generator files of section 4 and the loader, then present the generation plan in chat (record list with seed
keys, folder and MANIFEST handling, the exact settings reused, limit risk per record, runtime, the command), and wait.

## 10. Acceptance criterion
Plan stage: every setting of the new records is shown to equal the existing multisine records' setting by pointing to
the config line or code path; no existing file is modified by the plan. Generation stage (after approval): each new
record has a noisy and a noise-free version, passes the post-check, reports its realised multisine scale, and its
crest factor per channel lies within the range of the existing multisine test records (I1, I2, E1, from their `meta`).

## 11. Read these first
1. `Thesis-writeup/Code/Data/tdg_config.m` and `tdg_manifest.m` (settings and record definitions).
2. `Thesis-writeup/Code/Data/generate_thesis_data.m` (driver, versions, scale rule).
3. `Thesis-writeup/Code/Data/gtd_make_multisine.m` (crest-factor selection, amplitudes).
4. `Thesis-writeup/Documentation/DATA-DESIGN.md` sections 7.3, 7.4, 10 (test design, limits, scale rule).
5. `scripts/gantry/gantry_dynamic/data.py` (`record_files`, `THESIS_DATASETS`).

## 12. Do not
- Generate before Dirk approves the plan.
- Overwrite, regenerate or move any existing record or MANIFEST row.
- Run two MATLAB jobs at once; run MATLAB next to another session's MATLAB.

## 13. Operational
MATLAB R2025a, from `Thesis-writeup/Code/Data`: `tdg_init_paths; generate_thesis_data(sel, struct('simulator','replica'))`;
long runs launched detached with a log (user's memory: background shells die with the session). Python checks:
`conda run --no-capture-output -n GraduationProject python <script>`.

## 14. Delegation
None.
