# Handoff: audit the regenerated Coulomb dataset `..._coulomb_a6all` before anything is built on it
**From**: session of 2026-09-22 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task
A defect in the Coulomb data generator (D-207) was fixed and the full 22-record friction dataset
was regenerated into
`data/gantry/matlab/trajectory/augmentation_ma50_b140-230_a6_z03_coulomb_a6all/`. Establish
whether that dataset is physically valid at its new excitation level, and report the verdict per
gate with numbers. The excitation on two of three logical channels is now 6x larger than in every
previous Coulomb run, and three separately documented quantities were measured at the OLD, smaller
amplitudes and are therefore stale: the solver's stick-band margin `V_EPS_MARGIN = 9`
(`gantrySystemExtendedCoulomb.m:178`), the Leine collocation verdict, and the stick fractions and
force peaks recorded in `Matlab-scripts/Augmentation-coulomb/README.md`. Re-measure them on the
new dataset, decide whether each still passes its D-204 criterion, and write the outcome into
D-207 in `docs/decisions.md` plus the affected README rows. The dataset's own internal
consistency as a controlled pair is already established (section 4) and does not need redoing.

## 2. Out of scope
- **Do not regenerate any dataset.** The 22-record run is complete and passed its pair checks.
  Regeneration is a 1 to 2 hour, 540 MB operation and there is no reason to repeat it.
- **Do not delete or modify `augmentation_ma50_b140-230_a6_z03_coulomb`** (the defective folder)
  or `augmentation_ma50_z03_b140-230_a6_telica_coulomb` (which inherits the same defect). They
  stay on disk so earlier quoted numbers do not change underneath. Deleting them is the user's
  call, not this task's.
- **Do not repoint any training config** at the new folder. `scripts/gantry/gantry_dynamic/data.py`
  and `gantry_interconnect_dynamic.py` reference the old dataset names; changing them is a
  separate decision once this audit returns a verdict.
- **Do not train anything.** No augmentation runs, no parameter recovery.
- **Do not redesign the figures.** `Matlab-scripts/Augmentation-coulomb/plot_baseline_error_friction.m`
  is in its final agreed form (Telica only, 3x2). Replotting the multisine cases against the new
  pair is a follow-on the user has not yet asked for.
- **Do not touch `kamtin-fp-model/`** or anything under `Matlab-scripts/Augmentation/` (the
  frictionless generator is correct and shared).

## 3. Where things stand
Branch `Augmentation`, last commit `ccd2fee`. Tree is dirty in many directories from prior work;
the files this task changed are
`Matlab-scripts/Augmentation-coulomb/generate_trajectory_data_coulomb.m`,
`Matlab-scripts/Augmentation-coulomb/plot_baseline_error_friction.m` and `docs/decisions.md`.
Nothing is committed.

No run is in flight. The generation run finished with exit code 0; its log is at
`C:\Users\20203253\AppData\Local\Temp\claude\C--Users-20203253-OneDrive---TU-Eindhoven-Graduation-Project-Baseline-FP-model-Baseline-LPV-Augmentation\9fd5c89f-df33-4f0f-99d1-0dec9b55c780\tasks\bskc1462e.output`
(session-scoped temp; it may be gone. Every number it holds that matters is reproduced below).

`generate_trajectory_data_coulomb.m` is currently left configured for that run: `SELECT = {}`
(line 86) and `cfg.out_dir` pointing at `..._coulomb_a6all` (line 136-137). The user has not yet
said whether to revert those.

Disk is tight: `C:` at 98%, about 12 GB free. The new dataset is 540 MB.

## 4. Established and verified
Measured this session. Build on these without redoing them.

- **The defect.** `generate_trajectory_data_coulomb.m` applied `AMP_SCALE = 6` to `cfg.A_sym`
  only. `cfg.A_anti` is DERIVED at config time (`Matlab-scripts/Augmentation/data/gtd_config.m:115`,
  `cfg.A_anti = 0.5 * cfg.A_sym * cfg.Lb`), so scaling the primitive afterwards left it and
  `cfg.A_Y` at their defaults. Generated `amp_rms` was `(240, 14.5, 30)` against the frictionless
  `(240, 87, 180)`. The correct three-line form has existed all along at
  `Matlab-scripts/Augmentation/data/generate_trajectory_data.m:170-180`, with a comment warning
  that scaling `A_sym` alone "would silently break that relation". Logged as D-207.
- **The fix is live.** New run's log line 2: `Amplitude override x6: A_sym=240.0 N  A_Y=180.0 N
  A_anti=87.0 N*m`.
- **The new pair is controlled.** All 22 records, `augmentation_ma50_b140-230_a6_z03_coulomb_a6all`
  against `augmentation_ma50_b140-230_a6_z03`: `amp_rms`, `r_sim`, `f_sim` and `seed` are
  **bit-identical** (`numpy.array_equal`, not a tolerance). `f_sim` is the stored multisine force,
  so the excitation is provably the same on both sides and the Coulomb term is the only
  difference.
- **The limiter never fired.** Zero `scaled multisine to X%` lines across all 22 records. So
  `AMP_SCALE = 6` has force headroom at a genuine 6x with friction in the loop. This was the main
  risk going in and it is closed.
- **Friction is active.** T3 output difference between the pair: `y` RMS `[1.62e-06, 1.56e-06,
  1.57e-06]` m. Not two identical runs.
- **All 22 records exist**: T1-T14 train, V1-V4 val, E1-E4 test, 540 MB, log ends
  `Done: 22 records in ...coulomb_a6all`.
- **Telica datasets are unaffected** by the defect: `augmentation_ma50_z03_telica{,_coulomb}`
  carry no multisine (`excitation = 'none'` on every record, `gtd_build_records_telica.m`), so
  `f_sim` is identically zero on both sides.
- **Closed-loop friction magnitudes on Telica TP1** (measured, for context on what friction does
  here): Coulomb levels 16.80 / 18.35 / 11.60 N against peak actuation 1270 / 1353 / 531 N, i.e.
  1.3 / 1.4 / 2.2 %. The controller supplies 14.4 / 15.9 / 9.8 N RMS of extra force, about 86 % of
  the Coulomb level, so friction lands in the INPUT and the output trajectory changes by at most
  1.16e-05 m on strokes of 0.08 and 0.24 m.

## 5. Assumed but not verified
Each of these is load-bearing for using the new dataset and none has been measured at the new
amplitudes.

- **`V_EPS_MARGIN = 9` still sizes the stick band correctly.** `gantrySystemExtendedCoulomb.m:152`
  says it is "MEASURED on the PRODUCTION excitation" and line 168 says "RE-MEASURE IT if
  AMP_SCALE or the excitation band changes again". The nominal `AMP_SCALE` is unchanged at 6, but
  the EFFECTIVE amplitude on X_anti and Y rose 6x, which is exactly the condition that comment
  guards. Settled by `check_leine_collocation`.
- **The Leine collocation criterion still passes.** README records "Production T3: 2 of 2020
  (0.10%), worst 1.097, gain 1.348 against frictionless 1.332, ratio 1.012" at the defective
  amplitudes. Settled by rerunning gate 3 on the new T3.
- **Coulomb's law still holds at the states actually visited** (gate 2). Higher excitation visits
  more breakaway events. Settled by `check_friction_invariants`.
- **The absorber is still recoverable from the output** at the new amplitudes. The D-204 follow-up
  measured the absorber's output contribution at the old ones. Settled by
  `measure_absorber_contribution`.
- **Force peaks stay inside hardware limits.** `gtd_config.m:105` sets `lim.force_rms =
  [916, 916, 656]`. Logged PEAK forces on the new run run 800 to 1180 N per record (T3:
  `[942 1043 428]`, T14: `[989 1178 449]`, V1: `[1013 931 458]`). Peak against an RMS limit is not
  a like-for-like comparison, and the generator's own pre-check (which runs on the FRICTIONLESS
  linear plant, `generate_trajectory_data.m:75`) passed. Whether the realised friction response
  exceeds the RMS limit has not been computed. Settled by taking `std` of `u_total` per record
  against `lim.force_rms`.
- **Every record is physically sane.** Absence of an error is not evidence of no divergence. Not
  checked beyond the per-record diagnostics the generator itself prints.

## 6. Tried and failed
- **Acceptance test on `u_total - u_fb`** -> failed on 21 of 22 records with a max difference of
  1.22e-04 N -> the records store forces in single precision, so reconstructing the feedforward by
  subtracting two stored floats leaves round-off at 1000 N * 1.2e-07. The quantity is not the
  stored feedforward. -> Use `f_sim`, which is stored directly and is bit-identical. Do not
  reintroduce the subtraction form.
- **Plotting the friction/frictionless comparison on the OLD multisine pair** -> the friction
  residual came out SMALLER than the frictionless one (Y ratio 0.14) -> because the amplitude
  ratio was 30/180 = 0.167; the plot was measuring the excitation difference, not friction ->
  this is what exposed D-207. Any figure on the old `_coulomb` folders reproduces it.
- **`ytickformat('%.1e')` alone for scientific-notation axes** -> tick labels read 1e5 too large
  -> MATLAB keeps its own axis multiplier alongside the tick format, so the corner shows `x10^-5`
  while a tick shows `1.5 x 10^0` -> set `ax.YAxis.Exponent = 0`. Implemented as the local `sci`
  helper in `plot_baseline_error_friction.m`.
- **Writing the regenerated dataset to the session scratchpad** -> rejected before running -> 540 MB
  on a disk at 98 % would need a second copy to move it into place afterwards, and the property
  actually wanted (old dataset untouched) comes from a new folder NAME, not from a temp location.
  `/data/` is gitignored, so a new folder in the normal location costs nothing in git.

## 7. Achieved
- **Implemented and validated**: the generator fix
  (`generate_trajectory_data_coulomb.m`, the `AMP_SCALE` block), proven by the log line in
  section 4 and by all 22 records matching the frictionless `amp_rms`.
- **Implemented and validated**: the regenerated dataset
  `data/gantry/matlab/trajectory/augmentation_ma50_b140-230_a6_z03_coulomb_a6all/`, 22 records,
  540 MB, pair checks in section 4.
- **Implemented and validated**: `Matlab-scripts/Augmentation-coulomb/plot_baseline_error_friction.m`,
  now Telica-only, one 3x2 figure (trajectory | FP baseline error, rows X1/X2/Y), output
  `scripts/gantry/meeting/meeting-21-09-2026/figures/telica_trajectory_and_baseline_error.png`.
  Measured baseline error, full record, 0.1 s closed-loop windows: X1 1.82e-06 -> 3.48e-06 m
  (1.9x), X2 1.78e-06 -> 3.58e-06 m (2.0x), Y 9.15e-07 -> 3.28e-06 m (3.6x). The script now
  asserts `r_sim` and feedforward equality at load, so it cannot silently plot a mismatched pair.
- **Documented**: D-207 in `docs/decisions.md`. Its "Constrains" section still says the datasets
  need regenerating; that needs an outcome line once this audit lands.

## 8. The open question
Does `V_EPS_MARGIN = 9` still size the Karnopp stick band correctly now that the X_anti and Y
excitation is 6x larger than when it was measured?

- **Candidate A: it still passes.** The margin scales with friction deceleration and step size,
  neither of which changed; only the acceleration the rails see changed.
- **Candidate B: it no longer passes.** More excitation means faster traverses of the band and
  more collocation points landing inside it, which is the conditioning failure `leine1998` names.
  The stick fractions already moved a long way on T3 (README, old: X1 3.71 %, X2 5.01 %, Y 19.53 %;
  new run: X1 7.3 %, X2 7.9 %, Y 5.2 %), so the friction regime is materially different.

Chosen between by `check_leine_collocation` on the new T3: the rate of collocation points inside
the band, the worst excursion, and the perturbation gain against the frictionless control.

## 9. Next action
Run gate 3 on the regenerated T3:

```
matlab -batch "addpath('Matlab-scripts/Augmentation-coulomb'); check_leine_collocation('augmentation_ma50_b140-230_a6_z03_coulomb_a6all','T3_standstill_Y000.mat',0.50)"
```

T3 first because it is the standstill record, the most stick-prone case and therefore the worst
case for `V_EPS`, and because it is the record every earlier D-204 gate number was measured on, so
the comparison is like-for-like. If it passes, run gates 2 and the absorber-contribution
measurement on the same record, then the force-RMS sweep over all 22.

## 10. Acceptance criterion
From D-204's corrected gate-3 criterion (the binary "zero collocation points outside the band"
form is unsatisfiable by construction; see `Matlab-scripts/Augmentation-coulomb/README.md` row 3):

- worst excursion <= 1.25
- rate of collocation points inside the band <= 0.5 %
- perturbation gain within 10x of the frictionless control

These are the thresholds already established against `leine1998` and against a frictionless
control run, not against the model's own output. Reference values at the old amplitudes: rate
0.10 %, worst 1.097, gain 1.348 against frictionless 1.332 (ratio 1.012).

For the force check, the threshold is `gtd_config.m:105`, `lim.force_rms = [916, 916, 656]` N,
compared against `std(u_total)` per record, per channel.

## 11. Read these first
1. `docs/decisions.md`, D-207 (top of file). The defect, the evidence, and what it invalidates.
2. `Matlab-scripts/Augmentation-coulomb/README.md`, rows 3 and 5. The D-204 gate criteria and the
   numbers that are now stale.
3. `Matlab-scripts/Augmentation-coulomb/gantrySystemExtendedCoulomb.m:145-185`. The `V_EPS_MARGIN`
   derivation and its own re-measure instruction.
4. `Matlab-scripts/Augmentation-coulomb/check_leine_collocation.m` header. The criterion and why
   it was corrected.
5. `docs/decisions.md`, D-204. The five-knob port and the three gates this dataset is judged by.

## 12. Do not
- Do not reconstruct feedforward as `u_total - u_fb` for any equality test (section 6).
- Do not plot or quote anything from `augmentation_ma50_b140-230_a6_z03_coulomb` or
  `augmentation_ma50_z03_b140-230_a6_telica_coulomb`. Both carry the defective amplitudes.
- Do not change `AMP_SCALE`, `BAND_OVERRIDE` or `ZETA_A_OVERRIDE` to make a gate pass. If a gate
  fails, report it; the knob values are matched to `augmentation_ma50_b140-230_a6_z03` by design
  and changing one breaks the pair.
- Do not edit `Matlab-scripts/Augmentation/data/gtd_config.m` or
  `Matlab-scripts/Augmentation/data/generate_trajectory_data.m`. Both are correct and shared by
  every dataset on disk.

## 13. Operational
- MATLAB: `"/c/Program Files/MATLAB/R2025a/bin/matlab" -batch "<cmd>"` from the repo root. R2025a.
- Python checks: `conda run -n GraduationProject python <script>`, scripts in the scratchpad.
- Anything over a few seconds goes in the background with streaming output, per the live-output
  convention. The gate scripts are single-record and take minutes, not hours.
- Data consumed: `data/gantry/matlab/trajectory/augmentation_ma50_b140-230_a6_z03_coulomb_a6all/`
  (new) and `augmentation_ma50_b140-230_a6_z03/` (frictionless control).
- `check_leine_collocation(dataset, record, ma_frac)`, `check_friction_invariants(dataset, record,
  ma_frac)` and `measure_absorber_contribution(dataset, record, ma_frac, use_friction)` all take
  the dataset FOLDER NAME, not a path. `ma_frac` is 0.50 for this dataset.
- Disk at 98 %, 12 GB free. Do not generate new datasets.

## 14. Delegation
None. Every file involved is named in this handoff and the work is three scripted measurements
plus two document updates. An Explore subagent would only re-find what sections 4 and 11 already
point at.
