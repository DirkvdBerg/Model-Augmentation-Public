# Augmentation-kxy: X/Y stiffness variant of the data generator

Serves **T4** of `scripts/gantry/drift-isolation/PLAN.md`: regenerate the trajectory data with a
weak spring on the X and Y axes, retrain, and see whether the drift disappears.

**Nothing under `Matlab-scripts/Augmentation/` is modified.** This folder holds only new files.
The existing pipeline stays exactly as it is, and the existing dataset stays exactly where it is.

T4 is a **diagnostic that changes the plant, not a proposed fix.** The real machine has no X or Y
spring. The result is "the drift is, or is not, explained by the marginal poles"; if it is, the
fix on the real system still has to come from somewhere else.

## Why an X/Y spring at all

The gantry stiffness matrix has structural zeros on the X and Y diagonal (only Theta has
`kb1 + kb2`). Those two continuous poles sit at s = 0 and map to z = 1 exactly, so a state error
on those axes never decays.

Measured, not assumed (`scripts/gantry/drift-isolation/t4_xy_stiffness/derive_k_small.py`):

| | K = 0 (current) | **K = 10 (adopted)** | K = 1000 (rejected) |
|---|---|---|---|
| max pole radius abs(z) | 1.000000000000 | 0.999919146671 | 0.999919138931 |
| stability margin 1 - abs(z) | 0.000e+00 | 8.085e-05 | 8.086e-05 |
| decay constant tau | infinite | 3.09 s | 3.09 s |
| added resonance f_X, f_Y | none | 0.069, 0.158 Hz | 0.686, 1.584 Hz |
| station-keeping force at Y = -0.30 m | 0 N | **3.0 N (10% of A_Y)** | **300 N (1000% of A_Y)** |

The last two rows are the whole story. Above `k = c^2/(4m)` (5.63 N/m for X, 2.48 for Y)
both modes are underdamped, so the decay rate is `-c/(2m)` and does **not depend on k**.
k = 1000 therefore bought nothing over k = 10 and cost 100x the bias force.

T5 already showed the consequence directly on the real model: in a zero-ANN free run the X and Y
errors settle to a **constant offset** while Theta returns to zero. That is the marginal-pole
signature. T4 asks whether removing the marginality removes the drift.

`k_xy = 10 N/m` is adopted (**lowered from 1000 on 2026-07-29**, see the table above): poles
strictly inside with a 3.09 s decay constant, so an x0 error decays about 50x within a 12 s
record, and the added resonance three orders below the 130 Hz band edge.

**Why 1000 was retired.** It satisfied the original two criteria (P1 poles inside, P2 resonance
below the band) and was adopted on that basis, but neither criterion asks what the spring does
STATICALLY across the travel range. The records park at |Y| = 0.30 m, so an absolute spring
costs `k*0.30` N just to hold station: 300 N at k = 1000, against a Y multisine of 30 N RMS.
The spring was ten times the excitation, and the first generation batch showed visibly
suppressed motion. `gtd_enforce_limits` cannot catch this, because its linear pre-check runs on
the frozen-`cfg.K` (unsprung) plant and never sees the spring force. Criterion **P3** (station-
keeping force at most 10% of the multisine amplitude) was added to `derive_k_small.py` and the
admissible range collapses from 1.33 to 6.68e4 N/m down to **1.33 to 9.77 N/m**.

## Files

| File | Role |
|---|---|
| `gantrySystemExtendedKxy.m` | Copy of `Augmentation/gantrySystemExtended.m`; the ONLY change is that the two structural zeros in `K4` become `k_xy`. |
| `check_kxy_noop.m` | Class A gate. `k_xy = 0` must reproduce the original derivative bit-identically; `k_xy > 0` must change it. Runs the functions directly, outside Simulink. |
| `check_kxy_reaches_plant.m` | Frozen-controller gate. At `k_xy = 0` the kxy model must be bit-identical to the original (measured 0.000000e+00); at 1000 it differs by 1.55e-4, at 1e6 by 1.11e-1. |
| `make_kxy_model.m` | Reproducible builder for the `.slx` copy. |
| `gantry_additional_state_kxy_2025a.slx` | Model copy whose "Extended ODE" chart calls `gantrySystemExtendedKxy`. Route (b) below. |
| `generate_trajectory_data_kxy.m` | The production generator. `K_XY = 10`, controller frozen, writes to `data/gantry/matlab/trajectory/augmentation_kxy/`. |

The Python half is DONE and is NOT in this folder and NOT in `gantry_ss.py`: it is
`scripts/gantry/drift-isolation/t4_xy_stiffness/` (D-131), reached as `--k_xy 10`.

## The trap in "just compare the output data"

Comparing generated data with and without the spring is the right instinct, and it is the
decisive test, **but a plain difference test is ambiguous.** `cfg.K` feeds two different
consumers:

| Path | Consumer | Effect on the data |
|---|---|---|
| `cfg.K` -> `gtd_build_plant` | linear plant `G` and the `ruleOfThumb` controller `Cfb` | changes the **controller**, so the applied forces change, so the data changes |
| base workspace -> Simulink chart | the ODE that actually integrates | changes the **plant**, which is what T4 needs |

Confirmed by reading the `.slx`: the ODE chart receives stiffness as the scalars `kb1, kb2, ka`
and calls `gantrySystemExtended` by name. It never receives the `K` matrix. So changing `cfg.K`
alone would redesign the controller while leaving the plant unsprung, and **the data would still
differ**. A naive difference test would report success on a dataset that is physically incoherent:
a controller designed for a sprung plant driving an unsprung one.

Two ways to make the comparison decisive. Run both.

**1. Freeze the controller.** Leave `cfg.K` at its original value so `gtd_build_plant` produces
the identical `Cfb` and `G`, and change only the ODE's `k_xy`. Then the controller, the reference
and the limit scaling are all bit-identical between the two runs, and **any** difference in the
output is the plant. This is the change-one-thing version.

**2. Signature test.** Use a deliberately large `k_xy`, for example 1e6 N/m, which puts the Y
resonance near 50 Hz, and look for that resonance in the output spectrum. A controller redesign
cannot manufacture a plant resonance at a frequency set by `sqrt(k/m)`. This one is unambiguous
even if something unexpected is wired up.

Only after both pass should the production T4 dataset be generated with `k_xy = 10` and `cfg.K`
consistent, so that model and truth agree.

## Settled: how the Simulink model reaches the new function

**Route (b) was taken.** `gantry_additional_state_kxy_2025a.slx` is a copy whose chart calls
`gantrySystemExtendedKxy`, and `generate_trajectory_data_kxy.m` points `cfg.mdl` at it. Nothing
is shadowed and nothing under `Augmentation/` is touched. The two routes considered:

- **(a) Path shadowing.** Rename this folder's function to `gantrySystemExtended.m` and put the
  folder earlier on the path. No file and no `.slx` is touched. Cheap, but implicit: any other
  script running while that folder is on the path silently gets the modified plant. If this route
  is taken, the path manipulation must be scoped and reverted in a `cleanup` object, never left
  in a startup file.
- **(b) Copy the model.** Copy the `.slx` into this folder, edit the copied chart once to call
  `gantrySystemExtendedKxy` with the extra argument, and point `cfg.mdl` at the copy. Explicit and
  safe, but requires one GUI edit and duplicates a binary artifact.

Related gotcha, worth knowing either way: `Matlab-scripts/Augmentation/gantrySystemExtendedMFile.m`
is a **dead copy**. The model carries its own embedded version of that wrapper, with a different
argument order. Editing the on-disk file changes nothing about what the model runs. Only
`gantrySystemExtended.m` is live, because it is called by name.

## Order of work

Steps 1 to 4 are DONE and passing. Step 5 is the only thing left before T4 can run.

1. ~~Run `check_kxy_noop`~~ PASS: max abs diff 0.000e+00 over 200 random states at `k_xy = 0`.
2. ~~Decide route (a) or (b)~~ Route (b), the model copy.
3. ~~Write the `cfg` wrapper~~ `generate_trajectory_data_kxy.m`.
4. ~~Frozen-controller check~~ PASS: bit-identical at `k_xy = 0`, 1.55e-4 at 1000, 1.11e-1 at 1e6.
5. **Generate the production dataset at `k_xy = 10`.** 22 records, 12 s each, two Simulink runs
   per record. This is the remaining blocker.
   ```
   matlab -batch "addpath(genpath('Matlab-scripts/Augmentation')); addpath('Matlab-scripts/Augmentation-kxy'); generate_trajectory_data_kxy"
   ```
6. Copy `data/gantry/matlab/trajectory/augmentation_kxy/` to the cluster, then
   `sbatch scripts/gantry/drift-isolation/runners/run_t4.sh`. No data-root override is needed:
   `gantry_dynamic/data.py:70` builds the path from `cfg.mode`, and the drift-isolation CLI now
   accepts `--mode augmentation_kxy`. The model-side spring is `--k_xy 10`, and
   `run_training.py` refuses to start with only one of the two set.
