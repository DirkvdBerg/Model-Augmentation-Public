# Handoff: locate, overnight, which rung of the chain stops the augmented states from learning

**From**: session of 2026-08-20 | **Branch**: Augmentation | **Effort suggested**: xhigh

## 1. Task

Run unattended overnight and produce, by morning, three things: a single located fault (which rung
of the representation / estimation / objective / capacity chain stops the augmented states from
contributing, with evidence), the derivation of what the method *should* do at that rung, and
**that fix implemented, env-gated, D-072-checked, and tested if time allowed**. Write the five
missing probes and the C5 code change first (block W), walk the decision tree with gates C1-C9, run the two
pre-registered arms, then run the **conditional arm 3 selected by the leaf from the table in
section 9** so the remaining hours are not idle. Six literature-and-derivation agents T1-T6 run
concurrently from launch and feed arm 3 through the cutoff rule. Every unit writes a verdict to
the overview file whether it succeeds, fails or aborts. The deliverable is verdicts plus a
runnable fix, not an improved number: a night in which every arm fails but the fault is located
and its fix is implemented is a success, and a night with an unexplained improvement is not.

## 2. Out of scope

* **Do not commit, push, or stage anything.** The tree is dirty by design and the user does not
  yet trust the uncommitted `gantry_interconnect_dynamic.py` / `model.py` changes around `W^a` and
  the `rho` application. Everything runs from env gates on a throwaway tree.
* **Do not modify `model_augmentation/`**, `kamtin-fp-model/`, or read `kamtin-data/Data Telica/`.
* **`gantry_dynamic/{config,evaluation,orth_penalty}.py`**: the block is cleared, but they are
  still not part of this task. Do not modify them.
* **Deferred, do not start**: the `xc = 0` accuracy cost (needs two more training arms, §4.2 /
  §7 row 3) and per-row **loss** weighting for the nine-decade state rows (§4.5 / §7 row 11).
  Both get a "deferred" row in the overview with the experiment written out, nothing more.
* **Do not invent an arm.** Arm 3 exists, but only as one of the four pre-registered rows in the
  section 9 decision table, selected by the leaf. Anything outside that table is invention and is
  forbidden. If the table's row is not runnable, use only the fallback stated in that row (32 for
  B1); if the row has no fallback, write "not runnable, <reason>" as the verdict and stop rather
  than substituting something else.
* Do not overwrite `runs/cl_residual_spectrum.json`, `runs/cl_aug_spectrum.json`, or any existing
  artefact. New outputs get new names.

## 3. Where things stand

Branch `Augmentation`, last commit `4cdb7c1`. Tree dirty in `model_augmentation/fit_systems/`
(`closed_loop.py` +104, `pre_encoder.py` +47), `scripts/gantry/gantry_dynamic/model.py`, and the
untracked `transient-investigation/` and `ANN-learning-issue/` folders. Nothing in flight.

`docs/augmentation-training-status.md` (2026-08-20) is the current conclusions file and supersedes
the session records; its §6 lists nine corrections to earlier claims and is the reference for what
does **not** count as evidence here.

## 4. Established and verified

Build on these without re-deriving. Each names its artefact.

| Fact | Value | Source |
|-|-|-|
| Free-run validation RMS, untrained | `2.1866011034177349e-06 m` | D-072 gate, `bootstrap_probe.json`; bit-identical with `AUG_LRU` on and off |
| Free-run validation RMS, trained plateau | `1.3933793e-06 m`, -36.3 %, best at epoch 3 | run table §12 |
| Acceptance target | `1.215e-06 m` | run table §12 |
| Data-derived floor | `2.81e-08 m` | run table §12 |
| **Ceiling for this lever** | `9.1327e-07 m` = free-run error with the in-band `[140,175] Hz` component removed entirely | `error_budget.json`, §1.1 |
| Band share of free-run error POWER | `0.826`, per-record 0.718 / 0.809 / 0.837 | `error_budget.json` |
| `W^a` planted-model **window** RMS | random `1.2068e-06` \| zeroed `7.6156e-07` \| true latent `x0` `7.1603e-07` | `model.py:462-464`, `cl_capability.py` 2026-08-19 |
| Pole gradient coherence, no injection | `0.086` (isotropic reference `0.354`), sign agreement 50 % | `consistency_probe.json` |
| Pole gradient coherence, `AUG_LRU_B=0.377` | `0.827` at `na_nb=17`, `0.994` at 32 | `consistency_probe.json`, `na_sweep_probe.json` |
| `dL/d(nu_log) < 0` on all 8 batches | descent damps the mode | `consistency_probe.json` |
| Encoder input-noise amplification | `1919.8x`, constant over four decades of sigma | `encoder_conditioning.json` |
| Encoder velocity-row amplification vs `n` | `1.000 / 0.396 / 0.137 / 0.043` at `n = 17/32/64/128`, i.e. `n^-1.57` | `encoder_conditioning.json` |
| Stabilized-PEM cancellation | physical residual pinned at float32 floor across four decades | `alpha_cancellation.json` |
| Windowed (`xc=0`, `nf=400`) cancellation | saturates at `2.0x`, residual linear in sigma | §4.2 |

Structural facts, read in code this session:

* `AugLRUBypass` at `model.py:35`; augmented update `x_a,k+1 = A_aa x_a,k + gamma*NL[6:8]`,
  `lambda = exp(-exp(nu_log) + i exp(theta_log))`, `gamma = sqrt(1-r^2)` (`model.py:90-108`).
* `AUG_LRU_B` adds `g = g + X @ B_a.T` (`model.py:100-103`); `B_a`'s augmented columns are zeroed
  (`model.py:337-339`) so `x_a` cannot feed back and move the pole.
* `ENC_WA_ZERO=1` zeroes `Wa_psi_y` / `Wa_psi_u` (`model.py:469-474`). Default OFF.
* `Cd_norm` has **zero columns on the augmented states** (`model.py:242-244`): `x_a` reaches `y`
  only through ANN output rows 0-5. This is why C2 is the first gate.
* `cl_train.py:241-244` reads `CL_NOISE_SIGMA` and perturbs `sd.y` only.

## 5. Assumed but not verified

* That the augmented route can carry useful information at all. **C2 settles it.** Everything
  else in the plan is conditional on it.
* That `W^a` zeroing transfers from the planted-model **window** metric to the **free-run** metric.
  The `model.py:463` numbers are window RMS and are not comparable to `1.3933793e-06`. **C3
  settles it.**
* That P3 (objective damps the mode) is a run-level property and not a single-snapshot artefact of
  8 batches. **C6 settles it.**
* That `nx_aug = 2` is the binding constraint (§2 P6). Untested. T4 predicts it analytically; arm 2
  tests it if it survives the drop order, and leaf B2b's arm 3 (`nx_aug = 14`) tests it otherwise.
* That D-150's "survives Telica-level noise" means anything. It rests on the `sd.y`-only gate,
  which is non-physical (§4.4). **C5 settles it.**
* Wall clock per arm. The 2026-08-19 handoff says 4 s/iteration and 260 iterations/epoch
  (~17 min/epoch); the 2026-08-20 handoff says ~5.5 min/epoch for Arm F; the user says four epochs
  takes a couple of hours. **Do not trust any of these.** Measure epoch 1, then apply the timebox
  rule in section 13.

## 6. Tried and failed

Four mechanisms, each verified to do what it was designed to do, none moved the free-run number.
Do not re-run any of them as a fix.

* **D-150 live `A_aa`** -> `rho` 0 to 0.9920 -> free run **-0.665 %** -> the pole existing is not
  the same as the pole being used; `rho`'s gradient was exactly zero (§6.5).
* **Burn-in `K = 100`** -> discrimination 1.56x to 11.56x -> never trained successfully. The
  Kessels citation for it is **false** (§6.1): `tau = n_o` in Eq. 5.14 is the encoder history
  length, no transient argument exists. Burn-in is `# HEURISTIC:`.
* **Multiple-shooting defect** -> `||grad W^a||` 0 to 1.83e-01 -> epoch 1 validation **2.8x
  worse**. It is **degenerate**, not mis-weighted: with `x_a` undriven, `d_a,j` is minimised by
  `enc_a == 0`; measured `<grad_{W^a} L_defect, W^a> = +1.983`, both `L_defect` and `RMS(enc_a)`
  fall ~2.3 % per 15 updates. **Delete it, never retune it.** It is in
  `transient-investigation/train_combined_arm.py`.
* **`na_nb` coherence sweep** -> inconclusive and the "one period" reading is unsupported: 128 at
  4.94 periods collapses to `0.289`, below the isotropic reference. Cause is the probe (each
  `na_nb` builds a different model with its own 40-step warm-up), not the encoder. **Do not use
  gradient coherence as an acceptance metric anywhere tonight.**

## 7. Achieved

The **static** augmentation works and is validated: `2.1866011e-06` to `1.3933793e-06`, -36.3 %,
entirely through the ANN's correction on physical rows as a function of `(x,u)`. Nothing dynamic
contributed. `closed_loop_rollout` is validated as Sugie & Maruta 2020 stabilized PEM with exact
noise cancellation (`alpha_cancellation.json`). D-072 baseline equality holds bit-identically.

## 8. The open question

**Which rung of the chain is broken?** Candidates, and the gate that chooses between them:

* **Representation** (`x_a` cannot reach `y` usefully; `Cd_norm` has zero augmented columns) -> C2.
* **Estimation** (the encoder cannot initialise `x_a`; `W^a` unsourced, `na_nb` a rank rule) -> C8,
  with C3 sizing the `W^a` part.
* **Objective** (the loss damps a correct mode; nothing weights the 82.6 % of error power in a
  35 Hz slice) -> C6.
* **Capacity** (one resonator, one random direction, sampling a band the recipe assumes is
  covered) -> T4 predicts analytically; arm 2, or leaf B2b's arm 3, tests.

They are ordered causally: a failure at representation makes the rest untestable, which is why C2
runs first and why an `nx_aug` sweep was **not** made the opening move.

## 9. Next action

**Start with block W, then walk the tree.**

**Block W (write, do not run; ~1 h, no timebox, no GPU).** C2, C3, C6, C8 and C9 do not exist as
scripts, and C5 needs its env-gated code change. Write all six first, in one block, from the
`probe_*.py` pattern, before any timed run starts. Do not fold script-writing into a gate's
timebox: a 25-minute box that has to contain both authoring and running a novel experiment will
time out, and C2 and C8 are the two gates the whole tree depends on. Each probe must print its own
D-072 gate line and write a JSON artefact under `transient-investigation/runs/` with a new name.

Then the gates, in this order. **Expected run time is 5-10 min each** (the prior diagnostics ran in
tens of seconds); 25 min is a **cap**, not a budget. On timeout or crash write the verdict row and
move to the next.

**C1. D-072 bit-identity matrix.** Untrained free-run RMS must equal `2.1866011034177349e-06`
exactly for `na_nb` in {17, 32, 64} and `nx_aug` in {2, 8, 14}. Any arm whose configuration fails
here is cancelled, not run. Answers §7 row 4. Pattern: `probe_input_injection.py:78` holds the gate
constant.

**C2. Representation ceiling (planting).** Plant the true absorber mode into `A_aa`, `B_a` and the
routing, and measure **free-run** RMS on V1-V4. The planted model exists via `cl_capability.py`;
this re-points it at the free-run metric instead of the window metric.
*Permitted here because it is a ceiling MEASUREMENT. Planted weights must never be used as a
training init, and no oracle constant (159 Hz, 0.9856) may enter any arm.*

**Pre-registered boundary.** Available headroom from the untrained model to the ceiling is
`2.1866011e-06 - 9.1327e-07 = 1.2733e-06`. Compare the planted free-run RMS against it:
 - **>= `2.1720e-06`** (no better than the failed D-150 live-`A_aa` attempt, -0.665 %) ->
   **LEAF A**, representation is blocked, unambiguously.
 - **<= `1.869e-06`** (recovers at least 25 % of headroom # HEURISTIC: no data-derived boundary
   exists for a ceiling test; 25 % is a stated engineering choice, and the measured value goes in
   the verdict row so the boundary can be revisited) -> representation confirmed, continue.
 - **between** -> inconclusive. Continue down the tree, and flag in the verdict row that every
   downstream leaf inherits this uncertainty.

On **LEAF A**: skip C6, C7, C8 and arms 1 and 2 (each still gets a verdict row reading "skipped,
LEAF A"), still run C3, C4, C5 and C9 since they are independent of the tree, and go to the LEAF A
row of the decision table: implement the readout path (`model.py:242-244`, the removed trainable
`C_aug`) env-gated, prove D-072 still holds bit-identically with it present and zero-initialised,
re-run C2 through it, and if that clears the 25 % boundary, train it as arm 3. **This leaf gets the
most work, not the least**: it would explain all four failures in section 6 at once, and the whole
night's remaining budget goes to it.

**C3. `W^a` three-way on free-run.** random vs `ENC_WA_ZERO=1` vs true latent `x0`. Converts
`model.py:463` into the headline currency and fixes the `ENC_WA_ZERO` setting used by arm 1.

**C4. Plateau error budget.** Re-run `probe_error_budget.py` on the plateau checkpoint, not the
untrained model. Answers §7 row 10: how much in-band energy is left for the augmented states.

**C5. Physical noise gate.** Add an env-gated path (`CL_NOISE_CONSISTENT=1`) that perturbs `y -> y+v`
**and** `u -> u - C_fb(v)` together, leaving the existing `CL_NOISE_SIGMA` behaviour untouched by
default. Re-run the D-150 noise claim through it. Answers §7 row 12.

**C6. Objective sign test.** With a **correct** mode planted, run a few hundred steps of the real
training loss and record `nu_log`, its gradient, and the free-run RMS.
**Pre-registered boundary**: P3 is confirmed (leaf **B2a**) if `dL/d(nu_log) < 0` on at least 7 of
8 batches **and** `nu_log` increases monotonically over the recorded steps, i.e. the mode is being
damped. Either condition failing means the objective is not the wall and the tree goes to B2b.

**C7. Band draw vs Jan-faithful draw.** `AUG_LRU_BAND` / `AUG_LRU_RHO` set to a full-circle phase
and wide annulus, against the artefact-derived band, compared at init on planted-fit quality. This
is the §8.0 question and needs no new code.

**C8. Encoder isolation.** Fit the augmented pathway alone against the measured in-band residual,
once with the true `x_a` handed in and once with the encoder's. **The gap is the encoder's
contribution**, with the confound the coherence sweep could not remove.
**Pre-registered boundary**: the gap is "large" (leaf **B1**) if the encoder-initialised fit's
residual exceeds the true-`x_a` fit's residual by more than **2x**
# HEURISTIC: no literature boundary exists for this comparison; 2x is a stated engineering choice
and the measured ratio goes in the verdict row. Below 2x the encoder is not the binding
constraint, and the tree goes to C6's outcome.

**C9. Out-of-band decomposition.** The `9.1327e-07` that remains once the augmented mode is fully
accounted for is uncharacterised, and it is the whole of the distance between this project's
ceiling and the e-8 the user wants. Decompose it by source: baseline physical-parameter error,
static ANN capacity, and encoder startup transient (§4.3, the `K = 0` velocity rows at `7.76e+03`
relative amplification). This does not feed tonight's tree; it gives the next phase a target
instead of a blank.

Then the arms, both with the ablation test of §6.7 as the primary criterion:

**Arm 1**: `AUG_LRU=1`, `AUG_LRU_B=0.377`, `ENC_WA_ZERO` set by C3, `nx_aug=2`, no defect,
`lr_enc = lr_ann = 1e-5`, Adam `eps=1e-16`. Tests "do the augmented states help at all".
**Arm 2**: as arm 1 but `nx_aug=8` (which moves `na_nb` to 29 by Jan's rule; record the confound).
Tests capacity, and checks T4's prediction.

**Arm 3 is conditional and pre-registered.** Select exactly one row by the leaf the tree reached.
This is what stops the session idling from 04:00, and it is the only route by which a derivation
made tonight gets tested tonight.

| Leaf | Reached when | Arm 3 |
|-|-|-|
| **A** representation | C2 does not move the free-run number | env-gated readout path from `x_a` to `y` (restored `C_aug`, zero-initialised so D-072 holds), trained |
| **B1** estimation | C8 gap large | `na_nb` set to the value T2 derives (fall back to 32 if T2 did not cross-check), `nx_aug=2`, `ENC_WA_ZERO` per C3. **This is also the clean answer to §7 row 2 that arm 2 cannot give**, because it moves `na_nb` with `nx_aug` held fixed |
| **B2a** objective | C6 damps a correct planted mode | the weighting T3 derives, applied to the training loss, `nx_aug=2` |
| **B2b** capacity | C6 keeps the mode and C8's gap is small | `nx_aug=14` (Kessels' settled value, §2 P6) |

**Precedence when more than one row matches.** C6 runs before C8 in the gate order, but the leaves
are decided only after both have a verdict, and the chain is causal: **A > B1 > B2a > B2b**. If C8
shows a large gap *and* C6 shows damping, the leaf is **B1**, because estimation is upstream of the
objective: a loss cannot be shown to mis-weight a mode the encoder never initialises correctly.
Record the other matching row in the verdict as a secondary finding with its number.

**T-item cutoff: 01:00.** A derivation is eligible to configure arm 3 only if it has landed by then
**and** its cross-check against the named existing artefact passed. If B1 or B2a is the leaf and its
derivation missed the cutoff or failed its cross-check, run the stated fallback (32 for B1) or, for
B2a, skip arm 3 and write "objective fix derived but untested, T3 cross-check failed" as the
verdict. Never run an underived weighting.

**Implement the leaf's fix regardless of whether arm 3 gets to run.** By morning the change must
exist in the tree, env-gated and OFF by default, with its D-072 bit-identity line printed and
recorded. A diagnosis with nothing runnable attached is half a result.

Concurrently from launch, six agents, one per gap, each using the `deep-research` skill per D-121:

* **T1 `W^a`.** Hoekstra Eq. 8 defines it and never initialises it; Eqs. 16-17, 28-35 derive `W^b`
  from the **baseline's** reconstructability, which has no augmented states, and his experiment is
  static (`nx_aug=0`). Derive the construction for the augmented block, including what to do when
  it is unobservable from `y` at init (it is, by D-072). Must explain the measured ordering
  `1.2068e-06` (random) > `7.6156e-07` (zeroed) > `7.1603e-07` (true latent `x0`).
* **T2 lag rule.** Beintema §3.2 and §5.6 plus Darouach & Zasadzinski 1997. Produce a variance-vs-`n`
  expression for the `K=0` double-integrator rows and a stated rule for `n`. Must reproduce `n^-3/2`.
* **T3 objective.** Derive the sign of `dL/d(nu_log)` for an uncorrelated mode in the closed-loop
  rollout residual, then **derive a weighting that fixes it**. Entry points: `landau2002duality`
  Eq. 32, `zang1995iterative` Eq. 27, Kessels Eq. 5.15, frequency-weighted PEM. Must match the 8
  negative batches.
* **T4 band coverage.** Orvieto Lemma 3.2 assumes coverage; `n_pairs=1` samples. Derive how many
  pairs cover `[149.90, 164.06] Hz` to within the residual mode's half-power width. **Emit the
  prediction before arm 2 runs.**
* **T5 gauge.** `x_a` is defined only up to invertible transformation (§6.6), which is why
  `RMS(x_a)` and `rho` are not evidence. Derive a canonicalisation (modal or balanced) that does
  not break D-072 or the LRU stability guarantee, and state what it buys the orthogonal-projection
  contribution.
* **T6 windowed closed loop.** What the literature prescribes for controller-state handling in
  windowed closed-loop PEM (Sugie & Maruta 2020, Kessels, dual Youla), and whether an encoder for
  `xc`, an overlap warm start, or whole-record training is the principled replacement for `xc=0`.
  This replaces the deferred §7 row 3 with a derived answer.

Each T-agent returns: literature at source with page or equation, the derivation, the proposed
implementation change with its `# THEORY:` label, the cross-check against a named existing
artefact, and a falsifiable prediction. Then one refutation pass per T-item.

## 10. Acceptance criterion

**The night is done when every unit W, C1-C9, arm 1, arm 2, arm 3 and T1-T6 has a verdict row, and
the leaf's fix is implemented, env-gated and D-072-checked in the tree.** Not when a number
improves. A unit that was deliberately skipped (LEAF A) or dropped for time still needs its row,
reading "skipped, LEAF A" or "dropped for time, <what ran instead>". **An absent row is the only
real failure mode**, because it is indistinguishable from a unit that was forgotten.

Numeric criteria, fixed now and not to be redefined after seeing a result (per the run-discipline
rule; write each row into `docs/gantry-augmentation-problem-log.md` §12 **before** its launch):

* D-072 gate: exactly `2.1866011034177349e-06`. Not "close".
* Arm primary: **ablation**. Zero the readout's augmented columns in the trained model, re-run the
  free run. No degradation means the augmented states are decoration, and the arm is a negative
  regardless of its RMS.
* Arm secondary: free-run sim-RMS on V1-V4 against `1.3933793e-06` (plateau) and `1.215e-06`
  (target).
* Arm stop condition: epoch 1 worse than `2.1866011e-06` stops the arm.
* Ceiling: `9.1327e-07`. Nothing in this plan can go below it, and no unit may claim otherwise.
* A derivation is accepted only if it reproduces a number already measured on this machine.
* A citation is accepted only if the PDF was read at source with page or equation named. §6.1 (the
  false Kessels burn-in citation) is why.

**Evidence that does not count** (all from §6): `RMS(x_a)` or any gauge-dependent quantity;
parameter-movement counts (under Adam any non-zero gradient moves everything by about `lr`);
`rho` reported without its gradient; `rho(A_aa) > 0.5`; gradient coherence; oracle- or
model-derived thresholds. A unit reporting only these is logged as inconclusive.

## 11. Read these first

1. `docs/augmentation-training-status.md` - the conclusions file; §1.1, §6 and §7 are the spine of
   this task.
2. `docs/aug-lru-implementation.md` - exact env contract for `AUG_LRU*`, band recipe, checkpoint
   rules (a gated checkpoint loads only into a gated build).
3. `scripts/gantry/gantry_dynamic/model.py:35-145, 240-352, 440-475` - `AugLRUBypass`, the band
   draw, `B_a`, `Cd_norm`'s zero augmented columns, `ENC_WA_ZERO`.
4. `docs/decisions.md` D-072, D-142, D-150, D-151.
5. `scripts/gantry/closed-loop-controller/transient-investigation/probe_*.py` - the probe pattern
   to build C2/C3/C6/C8/C9 from; `probe_input_injection.py:78` holds the D-072 gate.

## 12. Do not

* Do not commit, push, stage, or weaken D-072.
* Do not re-run the multiple-shooting defect, burn-in as a fix, or a gradient-coherence sweep.
* Do not use planted weights as a training init, or any oracle constant (159 Hz, 0.9856).
* Do not use gradient coherence, `rho`, `RMS(x_a)`, or parameter-movement counts as acceptance.
* Do not retry a unit that produced a bad number. One retry only, and only for an infrastructure
  failure (crash, OOM, dead kernel).
* Do not run an arm outside the section 9 decision table, and do not start the two deferred
  experiments in section 2.
* Do not run a weighting, an `na_nb`, or any other derived quantity whose T-item missed the 01:00
  cutoff or failed its artefact cross-check.
* Do not pipe a running job through `grep`: the pipe buffers and nothing appears until it exits.
  Read the `.output` file.

## 13. Operational

Env `GraduationProject`. Every run longer than a few seconds uses the live-output convention.
Training launch pattern:

```
cd scripts/gantry/closed-loop-controller
AUG_LRU=1 AUG_LRU_B=0.377 ENC_WA_ZERO=<from C3> \
CL_EPOCHS=<n> CL_LR=1e-5 CL_ADAM_EPS=1e-16 CL_STRIDE=10 CL_ITS_PER_VAL=epoch \
CL_PROBE=1 CL_FLOOR=0 CL_BURNIN=0 CL_CONS_FRAC=0 CL_TAG=<tag> \
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
  -n GraduationProject python -u cl_train.py
```

Results land in `runs/cl_train_<tag>.json`. Any `AUG_LRU=1` build needs
`runs/cl_residual_spectrum.json` (present).

**Budget in updates, not epochs.** The plateau was reached at ~1250 updates at stride 10. Time
epoch 1, compute the wall clock for ~1600 updates, and if it exceeds **2 h 45 m** reduce epochs so
the arm fits, recording in the run row that the arm was truncated and at how many updates.
Raising `CL_STRIDE` to 100 gives ~10x fewer updates per epoch, so if it is used the epoch count
must rise ~10x to hold updates constant. Do not compare arms at different update counts.

**Timeboxes.** Block W is untimed (writing only, ~1 h, no GPU). Cheap gates 25 min of run time
each. Arms 2 h 45 m each. On timeout: kill, write the verdict row with what was measured up to
that point, move on. Nothing is left running at the end.

**The budget does not close, and that is planned for.** Block W ~1 h, plus nine gates at an
expected 5-10 min each (~1.5 h, or up to 3 h 45 m if every one hits its cap), plus three arms at
2 h 45 m is roughly 11-13 h against a night of about 9-10. **Two arms is the realistic number.**

**Drop order under time pressure**, applied without asking:
1. Drop **arm 2** first. It tests a hypothesis T4 will have already predicted analytically, while
   arm 3 tests the fault the night actually located. Record the drop and the reason.
2. If arm 3 still does not fit, truncate it in **updates** and record the count, rather than not
   running it.
3. **Arm 1 is never dropped**: it is the "do the augmented states help at all" test that §7 row 1
   needs, and without it the night has no trained result at all.
4. Never drop a gate to make room for an arm. The gates are the diagnosis; the arms only test it.

**The overview file is `tasks/overnight-2026-08-21-verdicts.md`, appended after each unit**, never
composed at the end, so it is correct if the session dies at 03:00. The user asked for this file
explicitly, so it is authorised despite the no-new-files rule; do not ask before creating it and do
not create any other new document. Top of file:

```
VERDICT:   <leaf>. The augmented states fail at <rung>.
EVIDENCE:  <gate>, <number> vs <threshold>, artefact <path>
FIX:       <the one change>, derived in <T#>, predicted effect <number>
IMPLEMENTED: <env gate name>, D-072 line <number>, OFF by default
TESTED?    <arm 3 result, or "implemented but untested, reason">
CEILING:   9.13e-07 (in-band removal, unchanged); out-of-band breakdown in C9
```

Then one row per unit (hypothesis, what ran, artefact path, number, verdict, what it eliminated),
the twelve §7 rows updated in place, the two deferrals, and the six derivations each marked pass or
fail against the artefact it had to reproduce. End with one recommended next action, not a menu.

**Honest coverage note to carry into the overview.** §7 row 2 ("is `na_nb = 17` wrong") is
answered **cleanly only if the leaf is B1**, because arm 3 then moves `na_nb` with `nx_aug` held
fixed. On any other leaf, arm 2 moves `na_nb` to 29 only as a side effect of `nx_aug = 8`, so the
row is answered **jointly and confounded**. Record it that way; do not report a confounded arm as
having settled the lag question.

## 14. Delegation

Six T-agents, one per gap, plus one refutation pass per T-item. That is the ceiling: no Explore
subagents for the compute track, which is targeted work in one context, and no agent may launch a
training run. The T-agents use the `deep-research` skill per D-121 and never ad-hoc web search.
Refutation attacks the derivation, not the writing.

T1-T6 launch at the start of block W so they run through the whole night alongside the compute
track. **T2 and T3 are on the critical path** for arm 3's B1 and B2a rows and must clear the 01:00
cutoff, including refutation, to be eligible; give them their agents first. T4 must return its
prediction before arm 2 starts, since arm 2's value is checking that prediction.
