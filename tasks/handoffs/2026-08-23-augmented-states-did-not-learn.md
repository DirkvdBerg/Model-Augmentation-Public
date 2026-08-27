# Handoff: build the meeting material diagnosing why the augmented states never learned
**From**: session of 2026-08-23 | **Branch**: Augmentation | **Effort suggested**: high

**This file replaces an earlier version written the same day.** That version built its argument on a
run (`arm 2`) produced by machinery this project has decided it cannot defend, and quoted its
result as a target. Section 2 now forbids that explicitly. If you have read the earlier version,
discard it.

## 1. Task

Build the supervisor meeting material in `scripts/gantry/augmented-states/meeting-23-08/`, in two
parts, in order.

**Part 1 diagnoses the failure and proposes nothing.** Establish, with figures and numbers, that the
augmented states carried essentially none of this project's headline `-36.89 %` improvement over
baseline; that this was structural rather than a training accident; that it was not a matter of
network size; what the same pathway is capable of when it is handed the answer; and how the
training-window horizon relates to the evaluation horizon. Part 1 ends by stating the open question,
not by answering it.

**Part 2 explains the mechanism a fix must supply**: why the augmented poles cannot be learned, why
placement therefore decides the outcome, and what the `B` matrix and the ANN routing do. Again with
figures doing the work.

Both parts are written as **one markdown file per slide** so they can be assembled into a deck.
Deliver figures as image files next to them.

## 2. Out of scope, and the first item is the reason this file was rewritten

- **Do not use `arm 2` in any form.** Not its `3.795974e-07`, not its `5.21x` ablation, not its
  `F = 0.88`, not as a target, a reference, a "capacity result", or an aside. It was produced by the
  data-derived band recipe (`lru_band_from_artifact`), which is project-invented, unpublished, and
  the specific thing the user has decided is indefensible in the thesis. Quoting it puts that
  machinery in front of the supervisors and invites the one question with no answer. It exists in
  `tasks/overnight-2026-08-21-verdicts.md` as internal history and it stays there.
- **Do not claim any solution result.** No `e-7` number, no "and our new implementation achieves".
  Part 2 explains a mechanism; the number arrives later from SLURM `78465` and is not yours to
  report.
- **Do not modify the implementation**: `model_augmentation/fit_systems/augmented_dynamics.py`,
  `scripts/gantry/gantry_dynamic/{config,model,pole_init}.py`,
  `scripts/gantry/augmented-states/run_augmented.py`. One exception in section 9.
- **Do not re-run training.** `78465` is in flight and its results change nothing in this material.
- **Do not open the orthogonal-projection thread.** Different document.
- `kamtin-fp-model/` read-only. Do not modify `scripts/gantry/BLA-Augmentation/` or
  `Augmentation-with-BLA/independent-init-b/`.

## 3. Where things stand

Branch `Augmentation`, HEAD `a8e45a0`. Dirty in `model_augmentation/fit_systems/`
(`closed_loop.py` modified, `augmented_dynamics.py` new), `scripts/gantry/gantry_dynamic/`
(`config.py`, `model.py` modified, `pole_init.py` new), and `scripts/gantry/augmented-states/`
(untracked).

**In flight, ignore it:** SLURM `78465`, array `0-8`, launched 16:27 CEST 2026-08-23. Logs
`/home/dirk_van_den_berg/logs/augmentation/aug_states_78465_<idx>.out`. Not an input to this task.

**Checkpoint availability, which shapes section 9.** The arm 1 checkpoint
`SSE_Interconnect_MultipleShooting_jBLNYQ_best.pth` is **not on this machine** (searched). Only
`scripts/gantry/closed-loop-controller/server-results/deep-SI-checkpoints/FitSys_ClosedLoop_Go1qTA_best.pth`
exists locally and its provenance is unknown. `78465` saves no checkpoint. So the one deliverable
that strictly needs a trained model is conditional; everything else is not.

## 4. Established and verified

Every number below is on one metric unless stated: `closed_loop_free_run_rms` over the complete
V1-V4 records, quadratic mean, metres. Untrained reference `2.1866011034177349e-06`.

**The failure, which is Part 1's spine:**

| configuration | trained RMS | vs untrained | ablation, blind | ablation, zeroed | `F` (D-157) | evidence |
|-|-|-|-|-|-|-|
| **A0, plain Jan** (`4cdb7c1`, zero-init ANN, kaiming `W^a`, `nx_aug = 2`) | `1.9050e-06` | `+24.8 %` | `1.0002x` | `1.0002x` | **`0.0007`** | `BLA-Augmentation/RESULTS.md:294-302` |
| **arm 1** (`nx_aug = 2`, 520 updates) | `1.379891240402659e-06` | **`-36.89 %`** | `1.405174e-06` = `1.0183x` | `1.405157e-06` = `1.0183x` | **`0.03`** | verdicts:128, `runs/arm_ablation_arm1_520upd.json` |
| width-matched control (828 ANN params, `nx_aug = 2`) | `1.384274e-06` | | | | | verdicts, THE SYNTHESIS |
| **planted true physics (C2)**, an ORACLE probe | `4.176627e-07` | `-80.90 %` | `2.509986e-06` | `2.509978e-06`, both `6.010x` | **`0.93`** | verdicts:120, `runs/representation_ceiling.json` |

Four independent facts, and Part 1 needs all four:

1. **Both ablation surfaces agree to four digits** on the dead cases (`1.0002x`/`1.0002x`,
   `1.0183x`/`1.0183x`). Agreement is what a route carrying nothing looks like; disagreement would
   mean the measurement is broken.
2. **`F = 0.03`**: 3 % of arm 1's own improvement is undone by removing `x_a`. The other 97 % went
   through the physical ANN rows, i.e. the static augmentation. The `-36.89 %` also reproduces the
   previously recorded `-36.3 %` static figure, so it is the same result by the same route.
3. **The cause is structural.** `gantry_dynamic/model.py` `build_model` adds exactly three blocks:
   the physical block, `Linear_Output_Block(C=Cd_norm, D=Dd_np)` whose augmented columns are zero,
   and `Static_ANN_Block(net=zero_init_feed_forward_nn)` whose output is zero at init. Nothing else
   writes rows `6..13`, so `x_a = 0` for every `k`; the ANN's read weights on those columns get zero
   gradient and its write path changes nothing downstream.
4. **It is not network size.** The width-matched control carries 828 ANN parameters and is `0.32 %`
   WORSE than arm 1's 600.

**Why the poles cannot be learned, which is Part 2's spine:**

- **C6**: with the true absorber mode planted (`r = 0.986982` at `157.8937 Hz`, computed from the
  plant), `dL/d(nu_log) < 0` on 7 of 8 disjoint batches, `nu_log` monotone on 100 % of recorded
  steps, and over 150 steps `r` falls `0.986980 -> 0.986967` while `f` drifts
  `157.9120 -> 157.8178 Hz`. Artefact
  `closed-loop-controller/transient-investigation/runs/objective_sign_probe.json`.
- **T3**: the batch-consistent damping term is strictly positive under EVERY non-negative weighting,
  so no per-row, per-frequency or combined residual weighting flips that sign.
- **Measured pole motion**: arm 1's pole `154.52 Hz` at init to `0.992038 @ 154.543 Hz` after 520
  updates, i.e. `0.02 Hz`. The block is a fixed basis, not an adaptive resonator.
- Arm 1's single pole sat `3.35 Hz` from the mode, **inside** the half-power width
  (`2 zeta f_n = 16.7 Hz`), and still scored `F = 0.03`.

**Context measured this session:**

- **Unmodelled fraction**, `RMS(baseline error)/RMS(y)`: ECC MSD (3-DOF cubic truth against its
  linear BLA) **`20.5 %`**; gantry open loop **`0.121 %`**; gantry closed loop, the scoring surface,
  **`0.00217 %`**. So this problem is `170x` harder like-for-like and `9450x` harder on the metric
  than the benchmark the zero-init design was demonstrated on.
- **The closed loop AMPLIFIES the band being learned.** `closed-loop-controller/loop_sensitivity.py`:
  `smax(So)` is `3.53e-04` at 1 Hz, `2.14e-02` at 10 Hz, **`1.81` at 157.89 Hz**, stable across
  `Y_op`. So the metric is not diluting the absorber, it is emphasising it.
- **Floors.** C9's data-derived floor `~2.81e-08`; the harness's measured float32 cross-dimension
  floor `2.980e-08` (printed by `augmented-states/tests`). Two independent estimates agreeing at
  `~3e-08`.

## 5. Assumed but not verified

- **That `window_starts`, `make_window_tensors` and `closed_loop_window_rms` are correct.** They are
  new, in the uncommitted `model_augmentation/fit_systems/closed_loop.py` diff, and **have no
  tests**. `window_starts` claims to reproduce deepSI's `kmid` from `system_data.py:322-329`; a
  silent off-by-one moves every number in the window figure. Section 9 makes testing them the first
  step, because a supervisor-facing number rests on them.
- **That the static augmentation is exhausted at `~1.38e-06`.** Three configurations land in
  `[1.379, 1.385]e-06`, which is suggestive with a sample of three. No run has ever had "find the
  static ceiling" as its purpose. **Do not write it as a measured plateau**; write it as three
  coincident points.
- **What `FitSys_ClosedLoop_Go1qTA_best.pth` is.** Unknown provenance. Do not use it for a
  supervisor figure unless you can establish which run produced it.

## 6. Tried and failed

- **Building the argument on arm 2** -> rejected by the user -> its number came from the
  project-invented band recipe, so quoting it imports indefensible machinery into the meeting ->
  this is why section 2 exists and it is the highest-value line in this file.
- **BLA initialisation, 21 runs, SLURM `78307`** -> nothing moved more than `0.12 %` -> pole accuracy
  was already saturated, the objective damps a correct pole, and the campaign differed from its
  comparators in four ways (routing `(1,4)`, stride 100, 108 updates, four identical copies of one
  pole) -> `augmented-states/README.md` sections 4 and 5. Cite that file; do not re-derive it.
- **Residual weighting as an objective fix** -> not runnable -> T3 proves the sign cannot flip ->
  verdicts, arm 3 row.
- **`na_nb` other than 17** -> D-072 baseline equality fails, `1.336e-04` at 32, `4.028e-04` at 103,
  monotone in `n` -> float32 conditioning of the observability inverse -> C1.

## 7. Achieved

Nothing for this task yet; `meeting-23-08/` does not exist.

Available and validated for you to build on: `augmented-states/README.md` (419 lines, the BLA
campaign analysis, every number traced), `augmented-states/CHANGES-TO-JANS-FRAMEWORK.md` (the three
steps with quote-verified citations), D-160 and D-161 in `docs/decisions.md`, and 41 passing tests
under `augmented-states/tests/`.

## 8. The open question

**Can Part 1 include the trained model's nf-window RMS at all?** It is the specific thing the
supervisor asked for and it needs the arm 1 checkpoint, which is not local.

| answer | what to do |
|-|-|
| `jBLNYQ` appears in the tree | compute it, and run the ablation on BOTH the window surface and the free-run surface. If `x_a` is decoration on the free run but load-bearing at `nf = 400`, the diagnosis changes from "dead" to "does not survive the horizon", and that is a finding worth its own slide |
| it does not | build the horizon curve on the UNTRAINED model, which still shows how strongly the metric depends on horizon, and mark the trained panel "pending checkpoint" rather than omitting it |

Do not block on this. Everything else in Part 1 needs no trained model.

## 9. Next action

**Create `scripts/gantry/augmented-states/meeting-23-08/` and build Part 1**, in this order:

1. Write a test for `window_starts` asserting its grid against the real training arrays (section 5).
   This is the one sanctioned edit under `model_augmentation/`, and it adds a test only.
2. Produce the figures below.
3. Write the slides.

**Figures for Part 1.** Load the `dataviz` skill before writing any plotting code.

| file | content | why it earns a slide |
|-|-|-|
| `fig01_xa_timeseries` | `x_a(t)` at initialisation, plain Jan against the block installed | plain Jan is a flat line at exactly zero. The whole argument in one image |
| `fig02_F_ladder` | `F` for A0 `0.0007`, arm 1 `0.03`, planted physics `0.93` | the gap between what we got and what the pathway can carry |
| `fig03_ablation_traces` | error time series, intact against `x_a` blind, for arm 1 | the ablation question is about trajectories, not scalars |
| `fig04_width_control` | RMS against ANN parameter count | more parameters, no better: not network size |
| `fig05_horizon` | RMS against rollout length, `nf = 400` to 48000 | the window issue, quantified |
| `fig06_ladder` | untrained `2.19e-06`, static points `~1.38e-06`, planted `4.18e-07`, floor `~3e-08` | what is achievable, and that headroom remains |

**Slide format.** One file per slide, `NN-short-slug.md`, and **the title is the claim, not the
topic**. Example, matching the tone of `README.md`:

```markdown
# The augmented states carried 3 % of the 36 % improvement

![](fig02_F_ladder.png)

Arm 1 reached `1.379891e-06`, a `-36.89 %` improvement on the untrained `2.1866011e-06`.
Removing `x_a` costs `1.0183x` on both ablation surfaces, so `F = 0.03`: three per cent of the
arm's own gain. The other 97 % came through the physical ANN rows.

Source: `tasks/overnight-2026-08-21-verdicts.md:128`, `runs/arm_ablation_arm1_520upd.json`.
```

**Part 1's last slide poses Part 2's question** and does not answer it: the states are dead, and the
objective cannot repair that, because with the true pole planted `dL/d(nu_log) < 0` on 7 of 8
batches and the poles move `0.02 Hz` in 520 updates.

**Then Part 2**, same folder, same format: why the poles must be placed rather than learned (C6, T3,
the pole-motion measurement), why a single pole inside the half-power band still missed, and what
`B_u` and the ANN routing do. Figures that carry it: the unit disk with drawn poles against the true
mode; resonator gain curves over the residual spectrum; decay time per mode against the `na_nb = 17`
encoder window; `x_a(t)` with `B_u` live against `B_u` zeroed.

## 10. Acceptance criterion

Part 1 is done when a reader who has not followed this thread can answer from it alone:

1. **What fraction of the `-36.89 %` came from the augmented states?** `3 %`, from `F = 0.03`, with
   both ablation surfaces quoted.
2. **Why?** Nothing writes rows `6..13` at init, so `x_a = 0` for every `k`.
3. **Was it network size?** No: 828 parameters is `0.32 %` worse than 600.
4. **What could the pathway carry?** `F = 0.93`, from the planted-physics oracle probe.

Part 2 is done when the same reader can say why the poles cannot be learned and cite `7 of 8
batches` and `0.02 Hz` for it.

No threshold here is oracle-based: `F` is defined on each arm's own untrained and trained values per
D-157, and the plant's `157.8937 Hz` appears only as a distance, never as an input to a method.

## 11. Read these first

1. `tasks/overnight-2026-08-21-verdicts.md`, rows C2 (line 120) and arm 1 (line 128), and the
   THE SYNTHESIS section. Every Part 1 number except A0. **Ignore the arm 2 row.**
2. `scripts/gantry/BLA-Augmentation/RESULTS.md:294-302`. A0, i.e. plain Jan measured.
3. `scripts/gantry/augmented-states/README.md`. The prior analysis and its house style; sections 4.2
   and 6.0 are the ones that change what you would otherwise write.
4. `model_augmentation/fit_systems/closed_loop.py:547-620`. The three window functions the horizon
   figure depends on, and which you are testing first.
5. `docs/decisions.md` D-157. Why `F` and not an ablation ratio.

## 12. Do not

- Do not mention arm 2 (section 2). This is the one that matters.
- Do not present the planted-physics probe as a ceiling: it is an oracle capability probe, and it
  was regressed on the one-step correction while the training objective is a rollout, so the two
  have different minima.
- Do not describe the static plateau as measured (section 5).
- Do not use `Go1qTA` without establishing its provenance.
- Do not propose or evaluate a fix in Part 1.
- Do not modify the implementation files in section 2; the only sanctioned edit is adding a test.

## 13. Operational

- Env: `conda run -n GraduationProject python ...`. Long runs per the live-output rule.
- Test suite, about 90 s: `cd scripts/gantry/augmented-states && conda run -n GraduationProject
  python -u -m unittest discover -s tests -t .`
- The horizon figure and any window number need `closed_loop_free_run_rms(..., return_error=True)`
  and `closed_loop_window_rms`, both in the uncommitted `closed_loop.py`.
- Data: `data/gantry/matlab/trajectory/augmentation/[TVE]*.mat`, already present.
- One full four-record closed-loop validation pass costs roughly 11 minutes; a single 12 s rollout
  is `212 s` at `nx_aug = 2` and `281 s` at `nx_aug = 8`. Budget the horizon sweep accordingly and
  reuse one rollout across horizons rather than re-running per length.

## 14. Delegation

**None.** Every number is in one of the five files in section 11 and the work is measurement plus
writing. An Explore subagent would add cost with nothing to search for.
