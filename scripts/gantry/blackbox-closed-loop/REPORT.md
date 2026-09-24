# blackbox-closed-loop: report

```
G1 Adapter: PASS. Zero-controller = open loop to 0.0 (exact); residual identity exact (0.0), units
  1.5e-07; grey-box number reproduced 1.4768032959e-05 vs 1.4768032959e-05 m (own path = this
  scorer = original repo packages, rel 0.0).
G3 Init: random UNSTABLE (6 of 6 records diverge, NaN loss and gradients); N4SID BLA not
  constructible at 4 kHz (every SS_f rejected); FRF BLA stable 6/6, val 3.532e-03 m, grad norm
  1.7e+06, finite -> server arm frf.
G4 Cost: 0.18 s/update (estimate) at nf 400, batch 512 -> 33,400 updates = 1.6 h + 26 validations
  0.65 h on one A100 (estimate), plus CUDA-graph recording and polish. BUT one Adam step at Jan's
  lr 1e-3 destabilised the loop: measured margin delta* = 1e-6, server lr set to 1e-7 (BB-013).
Resources: peak RAM 1.99 GB (tree), 0 watchdog kills, 2 training updates.
```

Every threshold was pre-registered in `DECISIONS.md` (BB-001 to BB-013) before the number it
governs was computed; every run is in `RUNS.md` with its hypothesis written before launch.

## What this means, in one paragraph
The black box can be put in the grey box's closed-loop objective with the grey box's own code
(G1 exact), but only one of three starts gives a stable loop (G3), and that loop is extremely
fragile: the controller, folded into normalised coordinates, has a feedthrough of about 5.9e3, so
a random change of 1e-5 in the linear weights of `f` and `h` destabilises 76 to 89 % of draws and
one Adam step at 1e-3 did destabilise it (G4). The grey box does not have this problem because
its physics fixes the plant the controller acts on and its ANN starts at zero. So the fair
comparison is buildable and costs about 3 h on the server, but at the stable step size (1e-7)
whether the black box learns anything in 33,400 updates is an open question the server run
answers; "similar results to the grey box" should not be expected from this parameterisation
(start 3.53e-03 m against the grey box's 1.48e-05 m at epoch 0, a factor 239).

## G0 Infrastructure: PASS (attempt 3)
Criteria BB-002. Run g0_infra.
- 30 vendored-prefix modules loaded, all from `vendor/`; `closed_loop.py`, `controller.py`,
  `plant.py`, `common.oracle` resolve inside the folder.
- `CFG` is the object of the repo's `scripts/gantry/gantry_interconnect_dynamic.py`, read at run
  time (mode `augmentation_ma50_z03_b140-230_a6all_telica_coulomb`, fs 4000, nf 400, na_nb 29,
  stride 10, burn_in 0, batch 512, epochs 200, its_per_val 1300, seed 42); 29 of 29 records present.
- 46 vendored files compared to their sources: 7 carry marked `VENDOR-PATCH` hunks (paths only),
  0 unmarked differences.
- Attempts 1 and 2 failed on real defects (my `common.py` shadowing the grey box's `common`
  package; the vendored namespace package merging with the repo copy); fixed and logged in BB-002.

## G1 Adapter: PASS
Criteria BB-006. Runs g1c_original, g1_adapter.
| Check | Result | Tolerance |
|-|-|-|
| (a) zero controller vs open loop, black box, 64 windows | max diff 0.0 | 1e-6 rel |
| (b) y_model = y_data gives u_plant = u_data | max diff 0.0 | 1e-7 rel |
| (b) units self-check, 12 controllers | max rel 1.48e-07 | 1e-5 |
| (c) grey box own selection path (vendored) | 1.4768032959e-05 m | reference |
| (c) this folder's scorer | 1.4768032959e-05 m (rel 0.0) | 1e-6 |
| (c) original repo packages, separate process | 1.4768032959e-05 m (rel 0.0) | 1e-6 |
The grey-box number is the untrained CFG model (FP baseline, zero-output ANN, `linear_map`
encoder), i.e. the grey box's own `Loss_val[0]`; no trained grey box exists on this dataset.
Per record (m): V1 2.019e-05, V2 2.030e-05, V3 2.058e-05, V4 2.059e-05, VP1 3.415e-06, VP2 3.531e-06.

## G2 Transfer measurement: PASS (four numbers reported)
Criteria BB-007. Runs g2_random, g2_n4sid, g2_frf, g2_oracle, g2_loop_check. Their own data (old
`augmentation`, 800 Hz), Tustin controller at 800 Hz, their own normalisation, `k0 = 17`.
| Checkpoint | Open loop V2, deepSI (metrics JSON) | Open loop V2 / T10, harness | Closed loop V2 / T10 |
|-|-|-|-|
| random | 8.632e-02 (8.632e-02) | 8.632e-02 / 8.189e-02, stable | NaN / NaN, UNSTABLE |
| N4SID BLA | 9.511e-03 (9.511e-03) | 9.510e-03 / 3.679e-03, stable | NaN / NaN, UNSTABLE |
| FRF BLA | 2.194e-02 (2.194e-02) | 2.194e-02 / 6.608e-03, stable | NaN / NaN, UNSTABLE |
| oracle init | 3.053e-02 (3.053e-02) | 3.053e-02 / 2.408e-02, stable | NaN / NaN, UNSTABLE |
Mechanism, checked rather than assumed: the 800 Hz controller stabilises both the FP plant and
the old truth at 800 Hz (loop max|z| 0.912 to 0.964, g2_loop_check), so the divergence belongs to
the models. An open-loop fit good to 1e-2 m says nothing about whether the known controller
stabilises the model; none of the four does. This is a transfer measurement on their data, not a
result on the production data.

## G3 Initialisation on the production data: PASS (frf)
Criteria BB-008. Runs g3_random, g3_n4sid, g3_frf.
| Arm | Closed loop 6 val records | Initial val sim-RMS (m) | Open loop (m) | Loss, batch of 512 | Grad norm |
|-|-|-|-|-|-|
| random | 0/6 stable | NaN | 1.060e-01 | NaN x3 | NaN |
| N4SID BLA | not built: deepSI N4SID rejected SS_f 20 to 80 at 4 kHz ("x exploded") | | | | |
| FRF BLA | 6/6 stable | 3.532e-03 | 2.080e-01 | 3.59 / 3.94 / 3.91 | 1.68e6 / 1.84e6 / 1.86e6, finite |
FRF per record (m): V1 3.76e-03, V2 4.69e-03, V3 1.76e-03, V4 9.22e-03, VP1 1.52e-03, VP2 2.43e-04.
Server arm by the BB-008 rule: **frf** (only passing arm); random runs as the control.
Notes: the controller reduces the FRF model's free-run error 59x against open loop (2.08e-01 to
3.53e-03 m), the same effect `cl_headroom.py` shows for the FP baseline. The window loss (3.6 in
normalised units) is dominated by the random encoder's initial state; the free run averages it out.
The FRF model was identified on a separate experiment (`joint_lowf_ma50_a5`, standstill multisine,
no friction), not on the production records.

## G4 Smoke and cost: PASS (runs end to end; cost written), with the lr finding
Criteria BB-009, BB-011, BB-012, BB-013. Runs g4_smoke, g3_greybox, g4_lr_margin, g4_loop_poles,
g4_cost.
- **Smoke** (the server script itself, 4 training records, CPU, eager, polish off): built, trained
  through the grey box's `fit()`, scored val and test, wrote `result.json`. Update 1 applied; the
  loss of update 2 was NaN and `fit()` stopped and restored the initial checkpoint. Update wall
  time 0.96 and 1.07 s, validation 21 s, peak 982 MB. Session budget: 2 of 2 updates used.
- **Why it diverged** (no optimizer step): FRF-init loop max|z| 0.9777 on all 12 controllers
  (physical FP loop 0.977), folded controller feedthrough about 5.9e3. Random-sign perturbations
  of all parameters: 1e-3 to 3e-6 diverge in 3 of 3 draws; 1e-6 stays stable in 3 of 3
  (val 4.7e-03 to 3.7e-02 m, already degraded). Linear loop: unstable fraction 100 % at 1e-4,
  76 to 89 % at 1e-5, 2 to 5 % at 1e-6, 0 % at 1e-7. Hence delta* = 1e-6 and **server lr 1e-7**
  by the pre-registered rule (delta*/10).
- **Cost** (BB-009): 85,644 windows, 167 updates/epoch, 200 epochs = 33,400 updates, 26
  validations. Black box / grey box on this CPU: forward+backward 0.88 / 6.14 s (ratio 0.144),
  validation 21.5 / 147.6 s (ratio 0.146). Server estimate from the grey box's measured
  1.236 s/update and 619 s/validation: 0.177 s/update and 90 s/validation, so 1.65 h + 0.65 h =
  **2.3 h**, plus the one-off CUDA-graph recording (grey box 55 to 61 min; smaller graph here) and
  the L-BFGS polish. Local CPU upper bound 9.6 h. Fits the 24 h wall of the runner with margin.
  Not exercised locally: `torch.compile` of `BlackBoxHF` and the L-BFGS polish.

## G5 Server preparation and the comparison table: PASS (job ready, baseline rows measured; trained grey-box cell pending)
Criteria BB-010. Runs g5_baseline, g5_print_config.
- Job: `runners/run_bb_arm.sh` (same SLURM header, environment and cache handling as the grey
  box's `run_obc_arm.sh`) calling `runners/train_bb.py`. Submit, from the same checkout as the
  grey box's run: `BB_ARM=frf sbatch scripts/gantry/blackbox-closed-loop/runners/run_bb_arm.sh`,
  then `BB_ARM=random ...` (control; expected to stop at update 1 on a NaN loss). NOT submitted.
- Matched to the grey box by construction: data folder, 18/6/5 split, 4 kHz, nf 400, stride 10,
  burn-in 0, batch 512, 200 epochs, its_per_val 1300, closed-loop sim-RMS selection, the same
  `train_model` Adam call and the same `run_lbfgs_polish` spec. The configuration snapshot lists
  exactly two CFG differences: `lr 1e-07` (BB-013) and `adam_eps 1e-08` (Jan / deepSI default).
- Black-box settings and sources: BB-004 (nx 8, f and h 2 x 8, encoder 2 x 16, na 29 with
  na_right 1 so the windows equal the grey box's), BB-013 (lr).

**Comparison table** (closed-loop free-run RMS in metres, same harness, same records):
| Model | Val (6 records, selection) | Test (5 records) | Source |
|-|-|-|-|
| FP baseline = grey box at epoch 0 | 1.4768e-05 | 1.1458e-05 | g1_adapter, g5_baseline |
| Grey box trained (CFG, this dataset) | pending: no run exists on this dataset | pending | grey-box entry file, server |
| Black box, FRF init (epoch 0) | 3.5323e-03 | 2.4927e-03 | g3_frf, g4_smoke |
| Black box, random init (epoch 0) | diverges (NaN) | not scored | g3_random |
| Black box trained, frf arm | pending server | pending server | `run_bb_arm.sh` |
| Black box trained, random control | expected NaN at update 1 | | `run_bb_arm.sh` |
| Oracle floor | not computable in this harness (BB-010) | | |
Test per record, FP baseline: E1 1.437e-05, E2 1.610e-05, E3 2.049e-05, E4 2.762e-06, EP1 3.571e-06.
Test per record, FRF init: E1 4.41e-04, E2 2.90e-03, E3 8.03e-03, E4 4.41e-04, EP1 6.53e-04.
**Pre-registered comparison rule** (BB-010): R = black box / trained grey box on the validation
set (selection) and the test set; R <= 1.25 similar, 1.25 to 2 same order and worse, > 2 clearly
worse, < 0.8 black box better (HEURISTIC bands); replaced by seed-range overlap if both are run
with two or more seeds. Both are also reported against the FP baseline row; a model that does not
beat it is reported as such regardless of R.

## Next action
Submit the frf arm (`BB_ARM=frf`), then the random control: about 3 h, and it answers the one
question this session could not, whether the black box learns at the only step size its closed
loop tolerates. Follow-up only, not done here: a black-box parameterisation whose loop gain is
not amplified by the normalisation (or SUBNET_LPV variants), if the frf arm stalls.
