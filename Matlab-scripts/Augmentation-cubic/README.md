# Augmentation-cubic: cubic-spring variant of the BASELINE data generator

Serves **D-135** and `scripts/gantry/discrepancy-ladder/PLAN.md`: regenerate the trajectory data
with the hidden absorber removed and a cubic hardening spring added on X and Y, then ask whether
the augmentation learns it.

**Nothing under `Matlab-scripts/Augmentation/` is modified, and `kamtin-fp-model/` is read only
and only ever copied from.** This folder holds only new files. Existing pipelines and existing
datasets stay exactly as they are.

## Why a cubic spring at all

The augmentation is net destructive and every completed run in the drift-isolation programme
selected epoch 0. The reason is not that the ANN cannot fit things: the current discrepancy, a
hidden 150 Hz absorber, is about `1e-9` of the sim-RMS metric while the untrained baseline already
scores `1.66e-4 m`, so the ANN's best possible gain is near zero and its damage potential on the
`K = 0` axes is unbounded. Training has a strictly negative expected value.

A cubic spring replaces that with a discrepancy that is **static** (no hidden states, fully visible
inside a 400-sample training window) and **large in the metric** (it deflects the rigid-body
trajectory, which is essentially all of the score). Supervisor's criterion, 2026-07-30: "iets
waardoor het systeem net iets anders gedraagt", something that makes the system behave slightly
differently.

Crucially, the ANN does **not** have to learn any dynamics here. The baseline supplies the
integrators exactly, so none of the theory that sank the standalone black box (D-134, Ribeiro's
`O(N)` loss sensitivity at the unit circle, resolving `|lambda-1| < 2e-8`) applies to this target.

## Truth only, and why that is the whole point

| | Where | Gets the spring |
|-|-|-|
| truth | `gantrySystemCubic.m`, inside the copied `gantry_2025a_cubic.slx`, logged to `q1` | **yes** |
| baseline model | `model_augmentation/systems/gantry_ss.py`, `Gantry_State_Block` | **no** |

The Python baseline is deliberately wrong by exactly `k3*q^3`, and closing that gap is the ANN's
entire job. **This is the one place the design differs from T4 on purpose.** T4 put its spring in
both truth and model so they agreed, because T4 asked a question about poles. Putting it in both
here would leave nothing to learn.

## Why the absorber is removed rather than kept

`USE_MSD = false` selects the 6-state baseline plant. That makes the discrepancy purely static, and
it buys a null control that no other dataset in this project has: **at `k3 = 0` the Python baseline
equals the truth exactly**, so the augmented model must return epoch 0 and the pipeline is proven
or falsified in the same batch.

Two things worth recording, because an earlier draft of D-135 got both wrong:

- `ma = 0` in `gantrySystemExtended.m` genuinely does make its 4x4 mass matrix singular (row 4
  becomes identically zero), so that route to a static-only truth is closed. It is not needed.
- The claim that the no-MSD path is a Simscape plant with no editable ODE was **wrong**, and was
  inferred from a code comment rather than checked. `gantry_2025a.slx` runs three plants in
  parallel and logs them separately: a Simscape Multibody `Single H-gantry` to `q`, a MATLAB
  Function chart `gantrySystemMFile` calling `gantrySystem` to `q1`, and
  `gantrySystemCoriolisCentripetalMFile` to `q2`. Block connectivity in `system_root.xml` traces
  `q1` back through Gain4, Selector1 and the integrator to `MATLAB Function1`. Since
  `gtd_run_simulation.m` reads `q1` on the non-MSD branch, the baseline truth already **is** the
  m-file `gantrySystem.m`.

## Files

| File | Role |
|-|-|
| `gantrySystemCubic.m` | Copy of `kamtin-fp-model/03 Simulink gantry/functions/gantrySystem.m`; the ONLY change is an added generalised force `f_nl = [-k3*X^3; 0; -k3*Y^3]`. |
| `check_cubic_noop.m` | Class A gate. `k3 = 0` must reproduce the original derivative bit-identically; `k3 > 0` must change it; the force must vanish at `X = Y = 0`; the sign must be restoring. Runs the functions directly, outside Simulink. |
| `make_cubic_model.m` | Reproducible builder for the `.slx` copy, entirely through the Stateflow API. No GUI edit. |
| `check_cubic_reaches_plant.m` | Class B gate. `k3 = 0` bit-identical to the original model through the real generator path, `k3 > 0` different, and the difference scales linearly in `k3`. |
| `derive_k3.m` | Sizes `k3` from the free-run degradation it induces, prints the per-record exposure table, and checks the workspace limit. Run before generating anything. |
| `generate_trajectory_data_cubic.m` | The production generator. Writes to `data/gantry/matlab/trajectory/augmentation_cubic/` (or `..._cubic_k0` for the null arm). |
| `gantry_2025a_cubic.slx` | Model copy whose `MATLAB Function1` chart calls `gantrySystemCubic`. Built by `make_cubic_model`, not committed by hand. |

## Why it is not a stiffness-matrix entry

T4 could put its linear `k_xy` straight into `K`. This file computes `dxdt = A*x + B*u` with
`A = [0 I; -M\K, -M\C]`, and a cubic term is not linear in the state, so it cannot be represented
there. It enters as an added generalised force after `A` and `B` are built, using `pinv` to match
how `B` is already constructed. **The Theta spring `kb1 + kb2` is untouched**; it is real physics
on the actual machine. The cubic occupies the X and Y slots, which are structural zeros, i.e. the
same physical slot T4 filled with `k_xy`.

Consequence worth carrying: the local stiffness is `3*k3*q^2`, which is **exactly zero at the
origin**. The cubic spring softens the marginal poles at large amplitude and not at all at
standstill, so it is not a fix for the `K = 0` problem and must never be presented as one.

## Sizing: the metric sets `k3`, not the excitation

An earlier draft proposed T4's P3 force-fraction rule and landed on 30 to 110 N/m^3. That is wrong
by one to two orders of magnitude. T4's spring went into both truth and model, so no mismatch force
ever existed and excitation was the only thing at risk. Here the entire spring force is a mismatch
force on a `K = 0` integrating axis.

Derived from the plant: `mh = 10.1 kg`, `cy = 10 Ns/m`, so a missing force `dF` gives terminal
velocity `dF/cy` with `tau = mh/cy = 1.01 s` (T5 measured about 1 s), the position ramps linearly,
and the RMS of a ramp is its endpoint over `sqrt(3)`:

> **gain = `(T - mh/cy) / (cy*sqrt(3))` = about 0.64 m of sim-RMS per newton.**

At `k3 = 2.5 N/m^3` a record at `|Y| = 0.1 m` degrades to about `1.6e-3 m` (10x baseline) and one at
`|Y| = 0.30 m` to about `4.3e-2 m` (258x), both well inside the `0.4 m` travel limit. Candidate
range **1 to 5 N/m^3**. `derive_k3.m` does this properly against the real record set.

Because the force goes as `|Y_op|^3` while the metric gain is linear, **the exposure must be
tabulated per record**: a validation set concentrated near `Y = 0` would leave no headroom at any
`k3`. That is also why the cubic is on X as well as Y, and why `derive_k3` prints the reference
trajectory ranges: a coordinate parked near zero contributes nothing.

## Order of work

1. **`check_cubic_noop`** — **PASS**, 2026-07-30. `A1` max abs diff `0.000e+00` at `k3 = 0`;
   `A2` min abs diff `4.234e-05` at `k3 = 1000`; `A3` force exactly zero at `X = Y = 0`;
   `A4` restoring sign. 200 random states.
2. **`derive_k3`** — **PASS**, 2026-07-30. **`k3 = 2.616 N/m^3`**, inside the predicted 1 to 5
   range. All four validation records carry DC exposure (10x, 142x, 38x, 58x the untrained
   baseline); 20 of 22 records carry both DC and AC, the two exceptions being `T3_standstill_Y000`
   and `E1_resonance_sweep`, both centred exactly at the origin, which is correct for a spring to
   ground. Worst free-run excursion `0.078 m` against a `0.4 m` limit, and the spring is 0.235% of
   the multisine, so P3 is satisfied by 40x and is not binding.

   Two things the first version of this script got wrong, fixed and worth not repeating: it sized
   from `rec.Y_op`, which is the SCHEDULING operating point used to freeze `M(Y)` and is `0` for
   every sweeping record even though those reach `|Y| = 0.3 m`; and it treated peak force as the
   driver when only the **DC** part of a mismatch force ramps. The corrected script reads the
   reference trajectory and separates DC from AC per axis. The frame is confirmed, not assumed:
   `gtd_make_reference` returns stage `[X1; X2; Y]` and `gtd_logical_to_stage.m` documents
   `q_stage = P' * q_logical`, so logical `X = (X1+X2)/2`.
3. **`make_cubic_model`** — **PASS**, 2026-07-30. Chart `gantry_2025a_cubic/MATLAB Function1`
   located and rewritten, `k3` created with scope Input and demoted to Parameter, inventory
   2 input / 1 output / 16 parameter (15 original plus `k3`), and the read-only source model
   verified untouched by size and timestamp.

   Note carried from the build log, **pre-existing and not caused by this folder**: loading the
   model prints "Model was exported from R2025b to R2025a. To find blocks that were removed during
   the export operation...". That notice belongs to the source file itself, which our copy
   inherits byte for byte, so it applies equally to the existing baseline dataset. Class B gate
   B1 is what would catch it if the re-save had changed behaviour.
4. **`check_cubic_reaches_plant`** — needs step 3. Four Simulink runs of 12 s, so allow time.
5. **Generate the null arm** (`K3 = 0`) and then the production arm. The null arm goes first: it is
   the one configuration where the baseline model equals the truth exactly.
6. Copy the datasets to the cluster and add the Python-side mode. See PLAN.md section 7 for the
   `nx_ann = 0` and `nx` changes, and for the reading A versus reading B question on the black-box
   arm.

## Traps, all previously paid for

- **Override `cfg.fig_dir` as well as `cfg.out_dir`.** `gtd_config` bakes
  `fig_dir = fullfile(out_dir, 'figures')` at config time, so overriding `out_dir` alone left the
  T4 wrapper writing PNGs into the baseline folder under identical filenames and it overwrote two
  baseline figures. The `.mat` files were unaffected, which made the symptom misleadingly mild.
  Handled here, plus a hard refusal to write into a folder that already holds `.mat` records.
- **`gtd_enforce_limits` cannot see the spring**, because its pre-check runs on the frozen-`cfg.K`
  plant. The per-record force-peak print is the only place a sizing error surfaces early.
- **Two charts match a naive substring search.** `gantrySystemCoriolisCentripetalMFile` also
  contains `gantrySystem`, so `make_cubic_model` matches on `gantrySystemMFile` and asserts exactly
  one hit.
- **`Matlab-scripts/Augmentation/gantrySystemExtendedMFile.m` is a dead copy.** The model carries
  its own embedded wrapper. Only functions called by name from a chart are live.
- **Preflight must not share a production tag or output directory.** A 1-epoch 2-record preflight
  checkpoint landed in a production directory and destroyed T1's nf=6400 rung.
