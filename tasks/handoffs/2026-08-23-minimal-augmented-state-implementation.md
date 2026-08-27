# Handoff: reach the `e-7` decade with the configuration most faithful to Jan's framework
**From**: session of 2026-08-23 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

**Reach a pooled closed-loop free-run RMS in the `e-7` decade using the configuration that deviates
least from Jan's framework**, and show that the augmented states are what got you there. The `e-7`
result already exists: arm 2 reached `3.795974e-07` on 2026-08-21 with `nx_aug = 8` and a `5.21x`
ablation, so this is not an open research question about whether the target is reachable. It is a
question about **what is actually required to reach it**, and the current results in section 4
already narrow that to four ingredients, three of which have published citations and one of which
(pole accuracy) is measurably unnecessary.

Read section 4 first and let it decide your configuration, rather than assuming any element is
needed. The binding constraint is defensibility: **every element you keep must trace to a published
equation or to a measurement in this repository**, because the user cannot defend project-invented
machinery in the thesis. Anything that is ours and unsupported is a liability even if it works. The
deliverable is a configuration that reaches `e-7`, a demonstration that the augmented states carry
it, and the survivors landed in `model_augmentation/` with the markers `CLAUDE.md` requires and a
citation on each.

## 2. Out of scope

- **BLA initialization and everything in `Augmentation-with-BLA/independent-init-a/artefacts/`.**
  Section 6 records why it cannot pay off. Do not run, repair or extend it.
- **The 57 attribution runs of the BLA factorial** (`runners/init_a_runs.sh` with `--array=0-80`).
  They inherit the three configuration defects of section 6 and would cost about 200 CPU-hours to
  confirm an explained null.
- **The data-derived band recipe** (`lru_band_from_artifact`). It is a project invention, it is
  exactly the kind of thing the user cannot back up, and C7 shows a uniform draw may match it. It
  stays as a comparison arm only if the campaign says the uniform draw fails.
- **The orthogonal-projection contribution, Telica, repairing the physical baseline, joint parameter
  estimation.** Different threads.
- **`kamtin-fp-model/`**: read-only, never modify.
- Do not modify `scripts/gantry/BLA-Augmentation/` or `independent-init-b/`.

## 3. Where things stand

Branch `Augmentation`, HEAD `a8e45a0` ("Route to all (0,1,2,3,4,5,6,7) always", user's own commit,
which changed `gantry_interconnect_dynamic.py` only). Tree is dirty in three places:

| path | state |
|-|-|
| `model_augmentation/fit_systems/closed_loop.py` | modified, +102/-2, additive: `return_error` on `closed_loop_free_run_rms` plus `window_starts`, `make_window_tensors`, `closed_loop_window_rms` |
| `scripts/gantry/Augmentation-with-BLA/` | untracked, the whole init-a probe |
| `scripts/gantry/augmented-states/` | untracked, `README.md` only |

**No runs are in flight.** The BLA campaign (SLURM `78307`) finished; its logs are in
`scripts/gantry/Augmentation-with-BLA/server-results/init-a/`.

Note for later cleanup, not a task: `gantry_dynamic/config.py:63` still carries
`ann_route_ix: tuple = (1, 4, 6, 7)` as the dataclass default. `cl_train.py:236` and
`gantry_interconnect_dynamic.py` both override it, but the stale default is why several sessions
have repeated "we route to rows 1 and 4".

## 4. Established and verified

**The four ingredients, and which have citations.** This is the core of the task.

| ingredient | needed? | evidence | citation if kept |
|-|-|-|-|
| **excitation**, a live linear input path `B_u` at init | **yes** | plain Jan gives `x_a = 0` for every `k`; measured ablation `1.0002x`, `F = 0.0007` | Hoekstra 2026 p10, matrices not pinned are `U(-1,1)` |
| **memory**, `A_aa` with `\|lambda\|` near 1 | **yes** for a lightly damped mode | without it `x_a` is a one-sample delay of `u`, already in `z` | Orvieto ICML 2023 Sec. 3.3, `lambda = exp(-exp(nu) + i theta)` |
| **span**, four or more DISTINCT poles | **yes, this is the 3.63x** | arm 1 `1.02x` against arm 2 `5.21x` | Orvieto Lemma 3.2 full-disk draw |
| accurate pole placement | **no** | C7 below | nothing needed; this is what gets dropped |

Everything in that table except the band recipe is Orvieto or Hoekstra. **The minimal implementation
is therefore plausibly "Jan's framework plus the Orvieto LRU block", with no project invention in
it**, and that is the hypothesis the campaign in section 9 tests.

**From `tasks/overnight-2026-08-21-verdicts.md`**, all on the same metric
(`closed_loop_free_run_rms` over complete V1-V4, quadratic mean, metres):

- arm 2, `nx_aug = 8`, band-drawn poles, 520 updates: **`3.795974e-07`**, ablation **`5.21x`**.
- arm 1, `nx_aug = 2`, one band-drawn pole, 520 updates: **`1.379891e-06`**, ablation **`1.02x`**.
- width-matched control, 828 ANN params at `nx_aug = 2`: **`1.384274e-06`**, i.e. more parameters
  than arm 2 and no better than arm 1. **The 3.63x is capacity, not network size.**
- C7: band draw median `1-R^2` **`0.98819`**, true planted mode **`0.98842`**, full circle
  **`0.99889`**. Pole placement is saturated; the band and the truth are indistinguishable.
- C6: with the true mode planted, `dL/d(nu_log) < 0` on 7 of 8 batches, `nu_log` monotone over 150
  steps, `r` `0.986980 -> 0.986967`. **The objective damps a correct pole.**
- T3: the batch-consistent damping term is strictly positive under EVERY non-negative weighting, so
  no residual weighting can flip that sign.
- Poles move **under `0.15 Hz` over 520 updates** in both arms; nothing migrated, nothing decayed.
  The block is a fixed basis, not an adaptive resonator.
- C8: the encoder is NOT the binding constraint, gap `0.997x` against a `2.0x` boundary.
- C1: D-072 baseline equality holds bit-identically **only at `na_nb = 17`** (`0.000e+00` there,
  `1.336e-04` at 32). Do not change the encoder lag.

**From `scripts/gantry/BLA-Augmentation/RESULTS.md:298`**: arm A0, which is plain Jan at commit
`4cdb7c1` (zero-init ANN, kaiming `W^a`, `nx_aug = 2`), ablation **`1.0002x`**, `F = 0.0007`. Its
whole improvement came through the physical ANN rows; **the augmented states were dead.**

**Plain Jan has no input path to the augmented states.** `gantry_dynamic/model.py` `build_model`
adds exactly three blocks: the physical block, `Linear_Output_Block` whose augmented columns are
zero, and `Static_ANN_Block(net=zero_init_feed_forward_nn)`. Nothing else writes rows `6..13`, and
the ANN output is zero at init, so `x_a = 0` for every `k`.

**Jan's ECC MSD has the same structure** (`scripts/ecc_2025/msd_ndof_interconnect_dynamic.py:86`):
`Static_ANN_Block(nz=nxd+nu, nw=nxd, net=zero_init_feed_forward_nn)`, `nw = nxd` so it writes ALL
rows, `nf = 200`, `na = nb = nxd*2+1`. The augmented states there are also excited only by the
encoder's `W^a` at each window start. **The dead-state problem is not inherent to his design**, which
is why "add `B_u`" is a hypothesis and not a conclusion.

**Measured this session** (`Augmentation-with-BLA/independent-init-a/runs/preflight_routeall/`
against `runs/preflight/`, both `76 checks, 0 failed`, seed 42, identical except routing):

| arm | gate gradient, routing `(1,4)`+aug | gate gradient, routing `0..N` |
|-|-|-|
| `INIT_A.2` | `1.54e-08` | `2.10e-06` |
| `RANDOM_LRU.8` | `4.60e-09` | `4.09e-06` |
| `INIT_A_OFF_TARGET.2` | `4.80e-09` | `3.00e-06` |

Two orders on every arm including the random and off-target controls, so it is a property of the
routing and not of the pole. Physical encoder-row gradients went DOWN, `7.05e-07 -> 5.28e-07`, so
routing to the `K = 0` rows does not blow up at `lr = 1e-5`.

**The closed-loop harness in `augtrain/` reproduces the reference**: untrained pooled
`2.1866133e-06` against the historical `2.1866011034177349e-06` recorded in `RESULTS.md:248`, seven
significant figures, from an independently assembled pipeline.

## 5. Assumed but not verified

- **That a full-disk draw reproduces arm 2.** Untested. C7's proxy is the reason to doubt it:
  full-circle scored `0.99889` against the band's `0.98819` and the two sets did not overlap. The
  section 9 campaign settles this and everything else in this handoff depends on it.
- **That plain Jan is dead at `nx_aug = 8` too.** A0 measured it only at `nx_aug = 2`. "Eight dead
  states instead of two" is a mechanism argument. A `nx_aug = 8` run with the LRU block removed
  entirely would settle it, and it is cheap.
- **That the `nx_aug = 2` capacity control transfers across pipelines.** Its expected `~1.4e-06`
  comes from arm 1, run by `cl_train.py`, not by `augtrain`. The seven-digit untrained agreement
  argues it transfers. If the control does not land near `1.4e-06`, distrust the cross-pipeline
  comparison before distrusting the `nx_aug = 8` result.
- **That a larger step-zero gate gradient predicts a better trained outcome.** The two-order
  measurement above is a step-zero quantity. It makes the BLA null unsurprising; it does not prove
  the routing fix changes the result.
- **The provenance of the `1.215e-06` target** used throughout the verdicts file. I did not trace
  where it comes from. Section 10 avoids depending on it.
- **`ENC_WA_ZERO`.** T1 proves `W^a = 0` is the correct value at zero readout gain and C3 measures
  the choice as worth `1.3 %` on the free run. Since the encoder is the ONLY excitation Jan relies
  on, a deliberately larger `W^a` is an untested and cheap probe, and it is the one idea in this
  handoff with no measurement behind it at all.

## 6. Tried and failed

- **BLA initialization, 21 server runs, SLURM `78307`** -> no arm moved more than `0.12 %` from
  untrained; best `2.184e-06` against a `1e-07` target -> pole accuracy was already saturated (C7),
  the objective damps a correct pole (C6), and the campaign additionally differed from arm 2 in
  three ways: routing `(1,4)` against `0..13`, stride 100 against 10, and 108 updates against 520 ->
  `scripts/gantry/augmented-states/README.md` sections 2, 4 and 5.
- **`EXACT_REPLICATED` at `nx_aug = 8`** -> four identical copies of one pole at `136.8975 Hz`, so
  the span stayed one-dimensional while arm 2's four DISTINCT poles spanned `151.9` to `162.9 Hz` ->
  replication adds parameters, not basis directions -> README section 4.1. The pole was also `21 Hz`
  from the true absorber at `157.8937 Hz`, because the artefact came from a synthetic fixture.
- **Scoring on an open-loop free run** -> pooled `1.224e-04` against the closed-loop `2.187e-06`, a
  factor of 56 -> `apply_experiment` cannot carry `y_data` and so cannot form the residual the loop
  is closed on -> `interconnect.py` `cal_validation_error` docstring; `cl_validation.py` header calls
  such a run "invalid rather than a negative result".
- **Arm 3, the objective fix by residual weighting** -> not runnable -> T3 proved the damping term is
  strictly positive under every non-negative weighting, and the only lever with leverage is a time
  mask needing `K_burn = 520` against `nf = 400` -> verdicts, arm 3 row. **Do not attempt a
  weighting arm.**
- **`na_nb` other than 17** -> D-072 baseline equality fails, rel dev `1.336e-04` at 32 and
  `4.028e-04` at 103, monotone in `n` -> float32 conditioning of the observability inverse -> C1.
- **`use_f64 = True`** -> `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and
  Double` inside `pre_encoder.py`, even though encoder and `hfn` parameters are both float64 -> a
  float32 input reaches the encoder net from the simulation loop; root cause not diagnosed, in
  shared code.

## 7. Achieved

**Implemented and validated:**

- `Augmentation-with-BLA/independent-init-a/augtrain/`, a closed-loop training probe: one model path
  for `BASELINE`, `CURRENT_INCUMBENT`, `RANDOM_LRU`, `INIT_A`, `INIT_A_OFF_TARGET`, all four
  ablations, manifest/preflight/run commands, restart-safe RNG and data order.
  **46 unit tests pass**; **preflight `76 checks, 0 failed`** in both routings.
- The closed loop is attached through the declared `fit_sys.simulator` seam, so training, validation
  and checkpoint selection call one `closed_loop_rollout` and cannot disagree.
- `runners/capacity_runs.sh`: the section 9 campaign, syntax-checked, array indices verified.

**Implemented, not yet validated:** nothing outstanding.

**Documented:** `scripts/gantry/augmented-states/README.md`, 419 lines, the full analysis with every
number traced to an artefact.

## 8. The open question

**Does a full-disk uniform pole draw reproduce arm 2's `3.795974e-07` and `5.21x` ablation, or does
the data-derived band recipe do real work?**

| answer | consequence | evidence that chooses it |
|-|-|-|
| full disk matches arm 2 | the minimal implementation is Jan's framework plus the Orvieto LRU block, with NO project invention anywhere. This is the outcome the user wants and it is fully citable | `RANDOM_LRU.8` reaches `~4e-07` with ablation `>= 2.0x` |
| full disk fails, band works | the band recipe is load-bearing and must be defended, or replaced by something citable | `RANDOM_LRU.8` lands near `1.4e-06` like `nx_aug = 2` |
| both fail at 522 updates | the 520-update figure does not transfer to this harness; re-examine before concluding anything | even the `nx_aug = 2` control misses `~1.4e-06` |

A better task may exist and the decision is the user's: if the full-disk arm succeeds, the natural
follow-up is removing the LRU block entirely at `nx_aug = 8` to test whether the encoder's `W^a`
alone suffices, which would make the addition to Jan's framework smaller still.

## 9. Next action

**Refresh the server copy, then launch the capacity campaign.** It is built, preflighted, and it
decides section 8, on which everything else depends.

```bash
# 1. refresh: augtrain/{runner,build,preflight}.py and runners/capacity_runs.sh all changed
R=/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation
rsync -av --relative ./scripts/gantry/Augmentation-with-BLA/independent-init-a dirk@server:$R/

# 2. launch, 9 runs, one wave
cd $R/scripts/gantry/Augmentation-with-BLA/independent-init-a
sbatch runners/capacity_runs.sh
```

Nine runs, three arms at seeds 42/43/44, each with `--stride 10 --epochs 2 --route-all`, giving
**522 updates** against arm 2's 520 and routing `0..13`:

| idx | arm | role |
|-|-|-|
| 0-2 | `RANDOM_LRU.2.RANDOM_INDEPENDENT.AB_TRAIN.GATE_ZERO.WA_XAVIER` | capacity control, arm 1 analogue |
| 3-5 | `RANDOM_LRU.8.RANDOM_INDEPENDENT.AB_TRAIN.GATE_ZERO.WA_XAVIER` | the question |
| 6-8 | `RANDOM_LRU.8.RANDOM_INDEPENDENT.AB_FIXED.GATE_ZERO.WA_XAVIER` | frozen basis, never run anywhere |

`RANDOM_LRU` draws `r = sqrt(u1)`, `theta = 2 pi u2` on the full stable disk and `draw_poles` returns
before it reads the artefact, so these runs are independent of BLA entirely.

## 10. Acceptance criterion

**Primary: pooled closed-loop free-run RMS `< 1.0e-06` on the complete V1-V4 set, i.e. inside the
`e-7` decade.** The reference values on this exact metric, all measured, none oracle-based:

| | pooled RMS | what it is |
|-|-|-|
| untrained | `2.1866011e-06` | the D-072 baseline |
| arm 1, `nx_aug = 2` | `1.379891e-06` | one pole, decoration |
| **`1.0e-06`** | | **the acceptance threshold, the `e-7` boundary** |
| arm 2, `nx_aug = 8` | `3.795974e-07` | the target to match |
| planted true physics, C2 | `4.176627e-07` | what handing the model the physics achieves |
| implied genuine floor, C9 | `~1e-08` | consistent with the data-derived `2.81e-08` |

Reaching `e-7` alone is not sufficient, because the physical ANN rows can move the metre RMS on
their own: arm 1 improved `36.89 %` against untrained with an ablation of `1.02x`, i.e. entirely
through the physical rows. So the second number is required.

**Secondary, and it is what makes the result about augmented states: ablation ratio `>= 2.0x`**,
measured post-hoc on the best checkpoint by zeroing the ANN input columns holding `x_a`. Plain Jan's
dead case is `1.0002x`, arm 1's decoration case is `1.02x`, arm 2's load-bearing case is `5.21x` and
the planted-physics ceiling is `6.010x`. A ratio of `2.0` sits unambiguously outside the decoration
cluster.

Report both per seed. If the seed spread exceeds the gap between arms, say so and do not rank: the
BLA campaign's `INIT_A.8` spanned `-0.99 %` to `-8.63 %` in X1 across three seeds, and that spread
swallowed every between-arm difference.

**Also report the per-channel split.** `Y` carries about 96 % of the pooled squared error, so a
pooled number can hide an `8.63 %` X improvement or be carried entirely by `Y`. The run summary
serialises `rms_per_channel_m` for every epoch, record and checkpoint.

## 11. Read these first

1. `tasks/overnight-2026-08-21-verdicts.md` VERDICT block and THE SYNTHESIS section. The mechanism,
   the pole tables and the two arms this whole task is calibrated against.
2. `scripts/gantry/augmented-states/README.md`. This session's analysis; sections 4.2, 5 and 6.0 are
   the ones that change what you would otherwise do.
3. `Augmentation-with-BLA/independent-init-a/augtrain/build.py`, `draw_poles` and
   `attach_closed_loop`. The two functions the next change touches.
4. `scripts/gantry/BLA-Augmentation/aug_block.py` docstring. The citation map for every element of
   the LRU block, which is what makes the minimal implementation defensible.
5. `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py:69-101`. Jan's own augmented-state wiring, and
   the reference point for "minimal addition to Jan's framework".

## 12. Do not

- Do not run a residual-weighting arm; T3 proved it cannot work (section 6).
- Do not change `na_nb` from 17; D-072 fails at every other value (C1).
- Do not use `apply_experiment` for scoring; it cannot form the closed-loop residual (section 6).
- Do not reintroduce `EXACT_REPLICATED` or any construction that repeats one pole; it adds no span.
- Do not edit `kamtin-fp-model/`, `independent-init-b/`, or `BLA-Augmentation/`.
- Do not promote anything into `model_augmentation/` before a run has exercised it, and never
  without the `@added` / `__project_origin__` / `# CHANGED` marker and its citation.
- Do not launch `runners/init_a_runs.sh --array=0-80`.

## 13. Operational

- Env: `conda run -n GraduationProject python ...`; on the server,
  `/dataB1/dirk_van_den_berg/conda-envs/GraduationProject`.
- Runtime: about 3 h per run, 9 runs in one wave at `%9`, so roughly 3 h wall clock. `-t 08:00:00`.
- Output: `runs/capacity/<ARM>/<SEED>/run_summary.json` plus `pred_*.npz`; SLURM logs at
  `/home/dirk_van_den_berg/logs/augmentation/init-a/capacity_%A_%a.{out,err}`.
- Runs are idempotent and restart-safe: a completed run is skipped, a killed one resumes from
  `ckpt_last` with RNG state and epoch count restored, so resubmitting the same file fills gaps.
- Validation is about 96 % of wall clock; one four-record closed-loop pass is roughly 13 min.
- Data needed on the server: `data/gantry/matlab/trajectory/augmentation/[TVE]*.mat`, 424 MB, 22
  files. The `_cache/`, `baseline/` and `figures/` subdirectories are not read.
- Local verification, about 2 min each:
  `conda run -n GraduationProject python -u -m unittest tests.test_augtrain` and
  `python -u run_augtrain.py preflight --out runs/preflight_routeall --seed 42 --route-all --stride 10`.

## 14. Delegation

**None.** The next action is a launch plus an analysis of nine known output paths. The reading list
in section 11 is five named files; an Explore subagent would add cost with nothing to search for.
