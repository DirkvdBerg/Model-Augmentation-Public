# Handoff: generate the thesis data sets (Coulomb + absorber truth, nominal and 10 % detuned) from the documented design, with a generator that is proven correct
**From**: session of 2026-09-26 | **Branch**: Augmentation | **Effort suggested**: xhigh (a generator change set checked line by line against a long design, number rechecks, and hours of MATLAB generation)

## 0. How to read this prompt (read first)
- **The goal, in the user's words (spelling cleaned):** "implement the actual data generation discussed in these files, for the Coulomb + MSD, with and without the 10 % detune; the 10 % detune is as in `gantry_interconnect_dynamic.py`, of the identifiable combinations. Copy the Coulomb + MSD script in Matlab-scripts to `Thesis-writeup/Code/Data`, save to `Thesis-writeup/Data/`. Be critical and make sure the data generation script is correct. Read the current md files and see the total picture. The main goal is the data generation. Recheck that the numbers are correct, for example for the noise that we derived."
- **This task is computation: the data are the deliverable.** Reading, checking and reporting serve the generation; they do not replace it. Do not end with a plan or a gate report where data should be.
- **The design documents are the specification; your job is to implement them correctly, not to redesign.** Where documents disagree, apply their own source precedence (`DATA-DESIGN.md` opening lines: user decisions, then session 1, then the rest; the later dated decision wins, e.g. D-219 over D-216) and record what you applied. Stop only for a conflict that precedence cannot decide and that changes data; then generate everything it does not affect and leave the affected records for the user. Never deviate silently.
- **Recheck, do not trust.** Every number the generator uses (noise, band, levels, limits, detuning, seeds) is recomputed or traced to its source before generation. "The noise that we derived" is one example of a number to recheck, not the only one. Judge a check by opening its output, never by a file or a heading existing.
- **Why the data exist** (the total picture, from `RESULTS-DESIGN.md`): the training, validation and test sets feed results R1 to R8 (position dependence, added states, plant-level BLA check, interpretability with and without projection, generalisation against black box and LTI, controller transfer, training effort). A record that no result uses is a design question to report, not a reason to skip it.

## 1. Task
Build a self-contained data generator in `Thesis-writeup/Code/Data/`, copied from the Coulomb + absorber generator in `Matlab-scripts/`, and apply every generator change the design requires. Prove each change correct with explicit checks. Recheck every design number. Then generate the data sets of `Thesis-writeup/Documentation/DATA-DESIGN.md` for two truths and write them to `Thesis-writeup/Data/`:
- `Thesis-writeup/Data/Coulomb-and-MSD/{Training,Validation,Test}`: truth T_AF, the nominal baseline + Coulomb rail friction + payload absorber.
- `Thesis-writeup/Data/Detuned-with-Coulomb-and-MSD/{Training,Validation,Test}`: the same truth with the ten identifiable parameter combinations detuned (section 4).

## 2. Out of scope
- Redesigning the data (records, sets, band, levels): the documents decide; you report problems.
- Control truths T_0, T_F and the conditional T_OA (`DATA-DESIGN.md` 7.7), and the deferred records (7.5), unless the user adds them.
- Training, the noise-training questions, results.
- Editing anything outside `Thesis-writeup/Code/Data/` and `Thesis-writeup/Data/`. In particular: `Matlab-scripts/` (copy from it, never edit), `Thesis-writeup/Writing/`, the documents in `Thesis-writeup/Documentation/` (report proposed corrections in your own report instead), `kamtin-fp-model/`, existing datasets under `data/`.

## 3. Where things stand
- The design: `Thesis-writeup/Documentation/DATA-DESIGN.md` (sets, records, manifest 7.9, implementation list in section 10, changes of 2026-09-26 in section 13), `EXCITATION-VALIDATION.md` (the multisine band), `EXCITATION-REVIEW.md` (independent review of the band evidence), `NOISE-INJECTION.md` (the noise), `RESULTS-DESIGN.md` (which result needs which data). Decisions D-210 to D-219 in `docs/decisions.md`.
- Supporting work: `scripts/gantry/data-set-design/` (`DATA-AUDIT.md`, `REVIEW.md`), the excitation session's folder named in `EXCITATION-VALIDATION.md`, `scripts/gantry/closed-loop-noise/` (measured spectra, `outputs/g1_spectra/g1.npz`, a shaping-filter model), `scripts/gantry/noise-training/`.
- The generator to copy: `Matlab-scripts/Augmentation-coulomb/` (`generate_trajectory_data_coulomb.m`, `gantrySystemExtendedCoulomb.m`, `make_coulomb_model.m`, `coulomb_v_brk.m`, the `check_*.m` gates), the Telica-profile generator `Matlab-scripts/Augmentation-telica-profiles-coulomb/generate_trajectory_data_telica_coulomb.m` with `Matlab-scripts/Augmentation-telica-profiles/`, and the shared functions in `Matlab-scripts/Augmentation/data/` (`gtd_*.m`). Copy every file the generator needs so that `Thesis-writeup/Code/Data/` runs on its own; record source path and commit per file in a `VENDORED.md`. Trust `gantrySystemExtendedCoulomb.m` and D-209 over `Matlab-scripts/Augmentation-coulomb/README.md`, whose band section is stale.
- Target folders exist and are empty: `Thesis-writeup/Code/Data/`, `Thesis-writeup/Data/Coulomb-and-MSD/{Training,Validation,Test}`, `Thesis-writeup/Data/Detuned-with-Coulomb-and-MSD/{Training,Validation,Test}`.

## 4. User decisions (fixed inputs) and one reading still to confirm
- **Two truths, one design.** This overrides `DATA-DESIGN.md` Q2 ("one data set; the detuning is a start value of the model"): the user now wants both data sets. Name this conflict in your report.
- **Reading to confirm with the user before generating the detuned set** (it is this prompt's interpretation, not yet the user's word): the detuning is applied to the TRUE plant, so the nominal baseline model starts 10 % off it, and the controller K1 is the same in both sets. Generate the nominal set first; ask this question once, briefly, before the detuned set; if no answer is possible (unattended), generate the detuned set under this reading and label it as such in `GENERATION-REPORT.md`.
- **The detuning** is the production vector of `scripts/gantry/gantry_interconnect_dynamic.py`, `combo_init_detune = [1.10, 1.10, 0.90, 1.10, 0.90, 0.90, 1.10, 1.10, 0.90, 1.10]` in the order `[kb_sum, cg1, cg2, cy, cb_sum, mh, m_total, m_diff, J_eff, d]` (D-191), applied to the identifiable combinations of the truth's physical parameters. The generator works on raw parameters (`gtd_config.m`), so the combinations must be mapped back to raw values: state the mapping, keep every raw value physical (positive masses and inertias, `M(Y)` invertible over the whole Y range), and verify that the combinations computed back from the raw values equal the intended ones. `m_diff` is negative, so its factor is applied to a negative number: check the sign is what the vector intends.
- **The controller** K1 is designed once, at Y = 0, on the nominal baseline (`DATA-DESIGN.md` 5.11); under the reading above it is the same in both data sets, because the engineer's controller does not know the true parameters.
- The benchmark truth is baseline + Coulomb + absorber, never the absorber alone. Maximum acceleration is the datasheet's 30 / 50 m/s² (X / Y). Noise is a force on the motor input (D-218), not on the encoder. Each record gets its own noise and phase realisation.

## 5. What must be rechecked before generation (non-exhaustive; the kind is "every number the generator uses")
- **Noise** (`NOISE-INJECTION.md`): its own status line says the force spectrum has not been recomputed after the calibration changed to the measured spectrum below 300 Hz. Recompute `Phi_d = (S_sim G_sim)^-1 Phi_e (S_sim G_sim)^-H` from `g1.npz`, 19.5 to 300 Hz, and pass its check 2 (friction off, standstill: 3.4 / 3.7 / 1.1 nm within 2 %). It assumes the controller changes with each record's Y_op, while `DATA-DESIGN.md` now fixes one K1: resolve which loop the calibration uses. The superseded white-force numbers (0.312 / 0.342 / 0.099 N) must not be used. The roll-off shape at 300 Hz is still open `(?)`.
- **Multisine** (`DATA-DESIGN.md` section 0, D-219): 106 to 297 Hz on a 0.5 Hz grid (383 lines), A_prod per channel, own phases per channel and record, periodic in 2 s within the 12 s record. D-216 says 140 to 230 Hz: D-219 supersedes it; confirm, and check the generated spectrum, not the config.
- **Levels and limits**: velocity and acceleration levels recomputed from the datasheet maximum; the binding post-simulation limit check on the simulated truth (section 10 item 2), not only the rigid-baseline pre-check.
- **The detuned truth**: the controller gate (stability and margins of K1 on the detuned truth over the full Y range), the absorber parameters unchanged, the mapping of section 4.
- **Absorber and friction parameters**: `ma = 0.50 mh`, damping 0.03 through `ZETA_A_OVERRIDE`, pole 212.13 Hz, anti-resonance 150 Hz; Coulomb and `V_BRK` as in `gantrySystemExtendedCoulomb.m` and D-209.
- **Seeds**: collision-free keys, every seed stored in the manifest.

## 6. Tried and failed (do not repeat)
- A dataset whose multisine amplitude was applied to one channel only (D-207): every channel's delivered level is checked from the data.
- Overwriting figures by overriding `cfg.out_dir` without `cfg.fig_dir` (D-206 note): override both.
- Two MATLAB jobs at once on this 15.8 GB machine, and headless sessions that ended their turn while a job ran (2026-09-24): one job at a time, never end a turn while a job runs.

## 7. Achieved
The design (section 3), the band evidence, the measured noise spectrum, the generator this copies. No generator implements the design yet; no data set exists.

## 8. Open questions (report, mark `(?)`, the user decides)
- Any design element found wrong or inconsistent during implementation.
- The noise roll-off shape at 300 Hz; which loop the noise calibration uses under a fixed K1.
- Disk: about 100 records per truth (`DATA-DESIGN.md` 7.9) at roughly 18 MB each is about 1.8 GB per truth, into a OneDrive folder, on a disk that has run nearly full; report the measured size of the pilot and the free space before the full run.

## 9. Next action: phases, in this order
**Phase A, read and reconcile.** Read the five documents, D-210 to D-219, and the supporting folders of section 3. Write `Thesis-writeup/Code/Data/SPEC.md`: every record of both truths with its settings (the manifest), the full list of generator changes (starting from `DATA-DESIGN.md` section 10), every number with its source, and every conflict or stale statement you found, each resolved by the precedence rule of section 0 (say which rule decided it) or, if precedence cannot decide it and it changes data, marked `(?)` with the affected records held back while everything else proceeds.

**Phase B, implement and prove.** Copy the generator into `Thesis-writeup/Code/Data/` and implement the changes. For each change a check that prints PASS or FAIL, stated before it runs, at least: noise off reproduces the copied generator bitwise; the noise calibration check of section 5; the generated multisine spectrum (band, lines, level per channel); the post-simulation limit check; the controller gate on both truths; the detuning round trip (raw values back to the ten combinations); seeds unique across the manifest; the absorber and friction parameters in the saved records equal the design values.

Keep `IMPLEMENTATION.md` up to date as each change lands, not only at the end.

**Phase C, pilot.** One record per record class and truth. Check every saved field, units and precision (double), lengths, the delivered excitation, the added noise level against the intended level, and the file size. Report; then generate the full campaign one job at a time, in stages, checking free disk before each stage.

**Phase D, independent review.** One subagent compares the generator and the pilot with the design, as a reviewer who distrusts both: does the code do what `SPEC.md` says, does `SPEC.md` say what the documents decide, are the numbers right. Save as `REVIEW.md`, fix, and list how each point was handled.

## 10. Acceptance criterion
Done when:
- `Thesis-writeup/Code/Data/` runs on its own and holds `SPEC.md`, `VENDORED.md`, the checks with their PASS output, `REVIEW.md`, and `GENERATION-REPORT.md` (what was generated, every check result, every conflict and how it was resolved or left open `(?)`, the measured sizes and runtimes).
- `Thesis-writeup/Code/Data/IMPLEMENTATION.md`, an overview of what was implemented, readable without the code: the folder layout and what each file does; per generator change, what changed, in which file and function, why (the design section or decision it implements), and the check that proves it with its result; what differs from the copied `Matlab-scripts/` generator; how to regenerate one record and a whole data set; and the parts of the design not implemented, with the reason.
- Both data sets are complete in `Thesis-writeup/Data/` per the manifest, each record with its seeds, settings and truth parameters stored in the file, and a `MANIFEST.csv` per data set.
- Every number in `SPEC.md` is traced to a source or recomputed, and every recheck of section 5 is reported with its value.
- No record was generated from a design element with an unresolved conflict.
Format of the reports: headers plus short one-line bullets, open points marked `(?)`.

## 11. Read these first
1. `Thesis-writeup/Documentation/DATA-DESIGN.md`, sections 0, 2, 7, 10, 13.
2. `Thesis-writeup/Documentation/NOISE-INJECTION.md`, all.
3. `Thesis-writeup/Documentation/EXCITATION-VALIDATION.md`, sections 0 and 16; `EXCITATION-REVIEW.md`.
4. `Matlab-scripts/Augmentation-coulomb/generate_trajectory_data_coulomb.m` and `Matlab-scripts/Augmentation/data/gtd_config.m`.
5. `docs/decisions.md` D-210 to D-219.

## 12. Do not
- Edit `Matlab-scripts/`, `Thesis-writeup/Documentation/`, `Thesis-writeup/Writing/`, `kamtin-fp-model/`, or any existing dataset.
- Generate records from a design element with an unresolved conflict, or use a number you have not traced or recomputed.
- Run two MATLAB jobs at once, or MATLAB beside another session's MATLAB.
- Read `kamtin-data/Data Telica/` with file tools; touch `kamtin-data/Telica.mat`.

## 13. Operational
- MATLAB R2025a (`matlab -batch`); Python via `conda run -n GraduationProject`; live-output convention.
- **Resources (user instruction, repeated to three sessions: "ignore the limiter"):** do not wait for free RAM and do not add launch gates. Keep only an emergency watchdog kill at 1.0 GB available RAM so the PC itself does not freeze (copy `telica-real/tools/watchdog.ps1` and set that level). One MATLAB job at a time, never beside another session's MATLAB.
- **Disk is the real risk:** on 2026-09-24 C: filled to a few MB and freed space was taken back within minutes (page file or shadow copies), so deleting is not a reliable fix. Check free space before every generation stage; if a stage would leave less than 2 GB, stop that stage, report, and continue with what fits.
- About 1.4 min per record (`DATA-DESIGN.md` 7.9), so several hours for both truths: generate in stages and report progress in `GENERATION-REPORT.md`.
- If run unattended: run every subagent in the foreground and never end a turn while a job or agent is running; poll long MATLAB jobs with short foreground checks. Two headless sessions died by ending a turn while waiting.

## 14. Delegation
One subagent, for the phase D review. No Workflow tool.
