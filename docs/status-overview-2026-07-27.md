# Gantry augmentation: status overview, 2026-07-27

**Purpose.** One document a new person can act on: what is established and at what grade, what
is open, what is void, and what to do next. No training was run for this. Every number below
was either read out of a stored artifact by me or is explicitly marked as unsourced.

**Relationship to the other documents.** `docs/diagnostic-overview.md` remains the
authoritative claim-by-claim record; its section 10 is a dated addendum that changes several
conclusions above it, and must be read last. This document does not restate it. It adds the
2026-07-27 verification pass, two findings that pass produced, and a single recommendation.

**Ownership.** Another session is actively writing `scripts/gantry/dc-accumulation/*`,
`diagnostic-overview.md` section 10, `results-log-2026-07-26.md` sections 11 to 13, and the
last six rows of `gantry-augmentation-problem-log.md` section 12. Nothing here edits those.
The corrections in section 4 and the action in section 7 are proposals to that session.

**Two conventions that are not optional here.**
1. Every error number carries its horizon. The same ANN-off model reads `7.86e-05` at 2 s and
   `1.66e-04` at 12 s, and at least two wrong conclusions in this repository came from
   comparing across horizons. (The 2 s value itself is unsourced: see section 4.)
2. Every result carries its seed count. The project floor is 3 seeds. Almost everything in the
   recent campaign is 1 seed and is therefore below the floor.

## 1. The failure

A learned block is trained inside the LPV baseline by minimising simulation error over
**0.1 s** windows. The deliverable is a **12 s** free run. The block acquires a constant
output offset, and on axes with poles at `z = 1` a constant force integrates as `f t^2 / 2`,
which is amplified roughly 14400x between the objective and the metric that judges it.

Verified by me this pass, directly from `71167/gantry_results_71167.npz` and
`gantry_ckpt_71167.npz` (not from any document):

| quantity | value |
|-|-|
| val sim-RMS, **12 s** free run, epoch 0 | `1.661376e-04 m` |
| same, epoch 1 | `2.108657e-02 m` (**126.9x**) |
| same, epoch 20 | `1.950576e-02 m` |
| `argmin` over 21 epochs | **0** |
| stored `bestfit` | `1.66137600899674e-04`, i.e. the epoch-0 value |

Over the same run the **validation** windowed (0.1 s) error falls 14 percent. The windowed
view improves while the deployed view collapses, on the same model and the same record. That
is the whole problem in one line, and it is the best provenanced claim here (three artifacts
from one run agree). Grade SINGLE, 1 seed.

Corollary verified in the same pass, and still under-reported: the augmentation never learned
the absorber. `gantry_state_recovery_71167.npz` gives `r2_lin` on the six physical states of
`0.9998 / 0.99996 / 0.941 / 0.9999 / 0.99998 / 0.941` but `r2_aug_lin = 0.344 / 0.350` and
`r2_aug_raw = -1.210 / -1.354` on the two augmented latent states. The same figure is stored
independently in `gantry_ckpt_71167.npz` as `diag_r2_linmap`.

## 2. What is settled, and at what grade

Do not re-litigate these. Grades follow the project convention: ROBUST = 3 seeds, 2 protocols;
SOLID = 3 seeds, 1 protocol; SINGLE = 1 seed or 1 record.

| claim | grade | artifact | horizon |
|-|-|-|-|
| Training is net destructive; best checkpoint is epoch 0 for all 20 epochs | SINGLE | `71167/gantry_results_71167.npz` | 12 s vs 0.1 s |
| The DC **is** the failure, not a correlate. On three ARTBP checkpoints the per-row mean alone reproduces **99.0 / 98.7 / 99.5 percent** of the error, and removing it collapses the model to **1.264 / 0.868 / 1.029x** its own ANN-off floor | SINGLE (1 seed, 1 record) | `dc-accumulation/results/step0_dc_sufficiency.json`, control passed both legs at 0.064 and 0.133 percent | 12 s, V1 |
| Same decomposition on the production checkpoint: MEAN-ONLY 112.8 percent of FULL, MEAN-REMOVED still 37.2 percent and 22x floor | SINGLE, cross-checked by a different script 11 days earlier (`drift-visual/data/f07.npz`) | `pysynth-data/results/MS12_ann_dc_force.json` | 12 s, V1 |
| The growth is polynomial and sub-linear, not exponential: no diverging mode | SINGLE, but three methods on two checkpoints | `MS11_orrell_two_run.json`, `ARTBP/data/growth_aug_vs_base_*.npz`, `pole_horizon_diag.npz` | 12 s |
| ARTBP collapses the dY DC and the variance ordering `poly6 < poly4 < geom` holds | **SOLID, 5 seeds**, the best replication in the repository | `ARTBP/data/train_{mode}_seed{0..4}.npz` | 0.1 s windowed, DC endpoint |
| ARTBP is the benchmark, not a fix: at `H_max = 1600` it cuts the first-epoch collapse about 10x and the drift 4 to 6x at about 2 percent windowed cost, and still fails G1 everywhere | SINGLE per arm | `dc-accumulation/results/step3_artbp_recompute.json` | 12 s val sim-RMS, 0.5 s tail of a 2 s run for drift |
| A minimal reproduction of the failure shape exists with no encoder, no LPV, no MIMO, no absorber, no latent states | SINGLE at publication; see section 3 for what 3 seeds did to it | `pysynth-data/results/K_sweep_minimal.json` | 4 s free run |
| The failure needs the integrator: harm ratio `1.713` at `K = 0` goes to `0.998` at `wn = 10 Hz` | SINGLE | same | 4 s |

**ARTBP is NOT ruled out.** Three documents rule it out on a theoretical argument
(Beatson and Adams at `|lambda| = 1`) that pre-dates none of the evidence: six converged runs
dated 2026-07-23 sat unread on disk while the ruling propagated into two research briefs and
two literature sweeps in a single day. The measured statement is "a 5 to 10x mitigation with a
hard `H_max` ceiling", not "ruled out". The anti-scope in
`dc-accumulation-research-brief-2026-07-26.md` section 3 has been corrected in place.

**ARTBP's own `dc` metric watches one row of eight.** `train_artbp.py:226` records the dY row
of the ANN output averaged over the probe set, and the pre-registered band at
`ARTBP/README.md:136` is a dY-only band (both confirmed by reading those lines). On the three
ARTBP checkpoints the dY row carries **0.77 / 0.95 / 1.76 percent** of the DC norm. So C-6's
"the DC was removed and the drift was not" was never supported: only the dY component was ever
measured. `dc_endpoint` is not a measure of "the DC" and no arm should be scored on it alone.

**Six mechanisms for why the DC is acquired are measured false**: the loss is blind to it; the
loss rewards it; it is paired with a compensator; it is exposure bias; it is entangled with the
useful fit; it is a transient more steps would remove. See `diagnostic-overview.md` C-8, C-12,
C-13, C-14, C-2. Why it is acquired is currently unexplained. Note the qualification in
section 4: the artifact behind the first of those six is missing from disk.

## 3. What changed in the last 24 hours

### 3.1 C-20 is WITHDRAWN. There is still no model that beats its own initialisation.

C-20 briefly read as the first trained model ever to beat its epoch-0 init on the deployment
metric, at `0.868x`. Step 0b showed that constant had been fitted on the evaluation record.
Recomputed with the constant estimated from the **14 training records only** (240 windows,
`nf = 400`), frozen, and subtracted at deployment, scored on the production selector
(V1 to V4 pooled, **12 s**) against the stored epoch-0 reference `1.846057e-04 m`:

| arm | FULL | TRAIN-MEAN-REMOVED | ratio to epoch 0 |
|-|-|-|-|
| `poly6 h1600 ep8` BEST (ep 7) | `2.042176e-03` | `5.947608e-04` | **3.222x** |
| `poly6 h1600 ep8` LAST (ep 8) | `3.766600e-03` | `7.334566e-04` | **3.973x** |
| `poly6 h3200 ep10` BEST (ep 6) | `2.931958e-03` | `6.459543e-04` | **3.499x** |

`beats_epoch0` is `false` on all three. Artifact:
`dc-accumulation/results/step0b_train_constant.json`, verdict reading (b). Grade SINGLE.

**The by-product matters more than the withdrawal.** The train-estimated and free-run constants
agree to `0.9275 / 0.9273 / 0.9237` in norm, i.e. about 7 to 8 percent, and the control passed
on every arm. That 7 percent costs, on V1 at 12 s, `1.038e-04` becoming `5.261e-04` on the LAST
arm and the same pattern on the others: a penalty of **4.3x to 6.1x**. Any "estimate the
constant and subtract it" fix needs the constant far better than 7 percent, which is a much
harder estimation problem than it looked like yesterday.

### 3.2 The minimal testbed is not a valid screen at 3 seeds.

It ports faithfully: `dc-accumulation/step1_testbed.py --configs C0 --seeds 0` reproduces
C-15's published `1.713x` free-run harm and `0.943` windowed ratio at 0.0 percent deviation.
But run at 3 seeds it does not hold still:

| arm | harm ratio per seed (**4 s** free run) | spread |
|-|-|-|
| C0 (published C-15 config) | `1.713 / 3.403 / 1.214` | 2.8x |
| C0b (start index shifted to `NA = 17`) | `1.050 / 3.710 / 6.232` | 5.9x |
| C1 (C0b + 3x steps) | `2.690 / 0.627 / 5.507` | 8.8x |
| C3 (absorber + fitted encoder) | `2.789 / 1.124 / 1.415` | 2.5x |

**C-15's published `1.713x` is one seed of a quantity with 2.8x seed spread**, and one C1 seed
lands at `0.627`, i.e. the failure is absent in that seed. Seed scatter exceeds the effect
sizes the testbed would be used to screen. The stored verdict is "gap does not close
(best 3.66x vs 58.9x), usable for relative comparisons only"; the 3-seed scatter makes even
that weaker than it sounds. Artifact: `dc-accumulation/results/step1_testbed.json`.

**The C2 arm is confounded and must not be read as published.** Adding the coupled absorber
raised the ANN-off free-run floor from about `2e-05` to about `6.5e-02`, roughly 3000x, so the
residual magnitude changed at the same time as its state-dependence. Its harm ratios
(`0.033 / 0.015 / 0.046`) are a scaling artefact. Re-run scaled before reading.

### 3.3 ARTBP training barely moves the encoder.

Reported by the session running the campaign: across all three checkpoints the total encoder
delta is `5.002781e-06`, identical between checkpoints, most tensors bit-identical to init and
the rest at about `1e-7` relative. I could not find an artifact holding that number, so it is
**ASSERTED** here. It is consistent with what is on disk: `gantry_grad_norms_71167.npz` records
a whole-encoder gradient norm of `2.277e-03` against `7.712e-03` for the ANN, with
`encoder.2` through `encoder.8` at exactly zero (C-1b).

## 4. Void, unsupported, or misreported

### 4.1 New this pass: the two-rig curvature upgrade has no artifact on either rig.

C-8's addendum upgrades "the windowed loss is not flat in the DC direction" to "confirmed on
two independent rigs", citing production-path curvatures X `7.064e+04`, Y `3.387e+04` from
`pysynth-data/results/A6_dc_resistance.json`, agreeing with the null rig's `7.084e+04` to
0.3 percent.

**That file does not exist.** `a6_dc_resistance_per_axis.py` writes it at its line 166, the
`results/` folder contains every other A-series artifact (A1, A3A4, A5, A7, A8, A9) and not
this one, no JSON anywhere in the tree contains `7.064e`, and the folder is untracked so git
cannot recover it. The second rig, `baseline-null/curvature_sensitivity.py`, contains no
`json.dump`, no `savez` and no `savefig`: it is print-only. So **both** rigs behind the
"two independent rigs" upgrade are unreadable, and every A6 number in
`diagnostic-overview.md` C-8, section 4b and the run table row for A6 is ASSERTED.

What survives: D1 (`drift-diagnostics/data/D1_zeroinit_2d_seed{0,1,2}.json`) is on disk and
does show a positive definite Hessian with eigenvalues `200.12` and `3153.66`. But D1 is a
6-state perfect-match null with routing `(3,4,5)`, and its three seeds are bit-identical by
construction (the evaluation point is deterministic), so it is n = 1. **"The loss is not blind
to the DC" therefore rests on one artifact, on one rig, on the physical velocity rows only.**
This is exactly the failure mode `diagnostic-overview.md` section 8 names as the project's
worst habit, and it recurred after that section was written. Rerunning A6 costs minutes and
should write its file this time.

### 4.2 `scripts/gantry/msd_transfer_diagnostics/` needs a correction before anything is built on it.

Verified against `data/diag_msd_summary.json`:

* The `adam_1e-05` and `sgd_1e-07` arms have **exactly one entry in every array**, equal to the
  value all five arms share (`train 3.807907e-05`, `val 2.756605e-05`, `sim-NRMS 16.276630`),
  which is the pre-training probe. **Those two arms completed zero epochs.**
* A summary circulated on 2026-07-27 attributed `train nf-RMS 3.80e-05 -> 3.43e-05` and
  `sim-NRMS 16.27 -> 284.60` to `adam_1e-05`. Neither number is in the file. The string `284`
  appears nowhere in the folder.
* Every arm that did complete an epoch moved the free-run metric the **other** way:
  `16.2766 -> 15.58` (adam 1e-7), `-> 12.07` (adam 1e-6), `-> 16.272` (sgd 1e-5), and
  `-> 7.24` in the companion proximal file.
* The companion `diag_proximal_summary.json` has `prox_lr` of `0.0 / 1e-4 / 1e-3` producing
  **bit-identical values in every field**. The penalty did nothing at any strength. Treat that
  file as a null result about the harness, not about proximal penalties.

Config issues in `diag_msd_failure_cause.py` that limit what the arms can support: it builds
its config with `dataclasses.replace(RunConfig(), ...)` without touching `up_sample`, so it
runs at `2` (`config.py:66`) against the entry file's `1`
(`gantry_interconnect_dynamic.py:72`) and against every checkpoint; it trims training to 3 of
14 records (lines 72 and 73); and line 98 catches every training exception, so a crashed arm
still writes a row that looks complete. **One correction to the brief I was given:**
`compute_normalization` is called on the full 14-record dataset before the trim, and the
resulting `norm` is passed into `build_model` with `auto_fit_norm=False`, so the
normalisation-changes-the-encoder trap that voided MS4 does **not** apply to this script. The
trim still changes the training distribution.

### 4.3 Carried forward, unchanged

* **Void framings, do not re-adopt:** `narrowband-objective-problem-2026-07-26.md` section 5;
  `flat-direction-problem-2026-07-26.md` sections 2 and 3.
* **Void runs** (own control failed, zero readable rows): MS3, MS4, MS8, MS9, MS10, D7, D9,
  `diag_xy_routing_blowup`, the `GANTRY_KX_ART` pole perturbation, T0.1, the R2 SGD sweep, the
  v6b Lipschitz sweep, v7. Full list with reasons in `diagnostic-overview.md` sections 5
  and 5b. Their numbers must not be quoted.
* **Unsourced numbers that are still in circulation:** the `7.86e-05` at 2 s ANN-off figure
  (appears exactly once in the repository, in `results-log-2026-07-26.md` line 26, and is the
  illustration of the horizon trap); MS2's 2 s arm; MS7's entire result (its JSON is 184 bytes
  and truncates mid-value); the R2/R4 tradeoff and the R3 pole gate (both print-only); the
  whole Route B evidence base.
* **Documentation lag:** the run table rows for step 1 and step 0b both still read
  `OUTCOME: pending` while both artifacts are on disk with verdicts. For the owning session.

## 5. What is open

Ranked by how much rests on the answer.

1. **Nothing has ever beaten the epoch-0 initialisation on the 12 s deliverable.** The best is
   ARTBP `geom` at 9.5x worse on the best-epoch basis. Step 0b removed the one apparent
   counterexample. This is the actual deliverable question and it has never been answered
   affirmatively.
2. **Why the DC is acquired is unexplained**, after six candidate mechanisms were falsified.
   See section 7: part of the paradox may be an artefact of which rows were probed.
3. **The orthogonal-projection route, which is the thesis's stated scientific contribution, has
   three failed gates that nobody followed up.** Verified by me in
   `simulations/gantry_subnet/diagnostics/orth_projection/`: `step7b_result.json` has
   `pass_a false`, `pass_b false`, `pass_all false` with a maximum principal angle of
   **56.656 degrees against a 5 degree tolerance** (the penalty basis is not stable to a
   10 percent parameter detune); `step8b_result.json` has `pass_b3 false`, `pass_all false` at
   **38.185 degrees**; `harness_check.json` has `pass_h3 false` (`h3_hat_over_nominal` `0.0959`
   on val and `1.1268` on test); `step7c_result.json` returns verdict `MIXED` at 11.07 degrees.
   Steps 0 to 6 and step 8 pass cleanly. This appears in no framing document, in no run-table
   row, and in `decisions.md` nowhere. **If the deliverable is the projection, this is the first
   thing that needs an answer.**
4. **Is the MSD-augmented system observable? The source file is STALE, so this must be
   recomputed before it is used.** `simulations/gantry_subnet/diagnostics/system_dynamics.json`
   records `baseline.obs_rank = 6` of 6 and `msd.obs_rank = 3` of 8. That file is dated
   **2026-06-16**, predates the D-126 data rebuild, and describes an absorber at **421.6 Hz**.
   The data actually in use has its absorber near **160 Hz**: the stored ground truth
   `val_x_aug` on V1 peaks at `159.75 Hz`, consistent with the generator's `fa = 150 Hz`
   tuning and the coupled resonance `150*sqrt(1 + ma/mh_rigid) = 158.1 Hz` at
   `ma_frac = 0.10` (corrected 2026-07-27, user-caught). Observability rank is generically
   insensitive to the absorber tuning, so the finding may well survive, but as it stands the
   number describes a system that is not the one generating our data. Recompute it on the
   current parameters, with a scaled or staircase algorithm rather than a plain rank call
   (the system still spans two poles at exactly 0 up to the absorber mode), before drawing any
   structural conclusion. Treat every other number in that file as suspect until re-derived.
5. Merely unfinished, design exists: D7 rerun with the indexing bug fixed; D9 with the
   optimiser-creation bug fixed and routing corrected to `(0..7)`; ARTBP gate 2 at 3 seeds;
   the injected-friction sim (`datasilent-friction-sim` steps 3a and 3b), which multiple
   documents correctly call a prerequisite for validating anything in the DC-handling family,
   because on the current testbed the correct DC is genuinely zero; MS7 rerun.

## 6. Evidence quality

* **Below the 3-seed floor:** the entire dc-accumulation campaign except step 1
  (step 0, step 0b, step 3 are all seed 0), every MS result (MS1 to MS12, mostly 1 record too),
  every `baseline-null` result, every ARTBP gate-2 arm, and all of `diagnostic-overview.md`
  sections 2b, 4b, 5b and 6b.
* **At or above the floor:** ARTBP Phase D (5 seeds, C-5), the frozen-rig campaign D1 to D6 and
  T0.1b to T1b (3 seeds), and step 1's testbed sweep (3 seeds, which is what killed it as a
  screen).
* **Two nominal 3-seed results are n = 1:** D1's three seeds are bit-identical because the
  evaluation point is deterministic, and the three `v3x0sgd_encoder_seed*` files are
  numerically identical in every array, so "2000x less DC under SGD" is one seed.
* Pre-registration is real here and is working: four runs on 2026-07-26 and three more since
  were voided by their own pre-declared controls, and the C-20 withdrawal came from a control
  the author declared before launch. That is the mechanism functioning, not a failure rate. It
  only works because the control is declared in advance.

## 7. The single next action

**Run a row-masked MEAN-ONLY / MEAN-REMOVED decomposition on the checkpoints step 0 already
loads: which of the eight constants actually carries the 12 s error?**

No training. Same rig, same record, same horizon, same passing control as step 0, with a row
mask over the constant vector. Roughly one 12 s free run per mask.

**Why this and not something else.** The gate answered on 2026-07-26 night established that the
constant vector as a whole is the failure (99 percent of the error on three checkpoints). What
nobody has ever measured is which rows of it matter, and the row composition is not what the
framing assumes. Computed by me from the stored per-row means:

| model | `\|\|traj mean\|\|` | aug0 and aug1 share | dTheta share | dX and dY share |
|-|-|-|-|-|
| `poly6 h1600 ep8` BEST | `7.5914e-05` | **97.6%** | 21.8% | 0.79% |
| `poly6 h1600 ep8` LAST | `7.9774e-05` | **96.9%** | 24.7% | 0.99% |
| `poly6 h3200 ep10` BEST | `2.7691e-05` | **98.7%** | 8.6% | 2.64% |
| `gantry_drift_71167_last` | `1.9172e-04` | 66.0% | **74.9%** | 0.66% |

(Shares are in normalised state units and do not add to 100 percent because the rows are not
orthogonal contributors to a single quantity. They are not a dynamic attribution, which is
precisely the point: a constant on a position row, a velocity row and a latent row have
completely different consequences over 12 s, so this must be settled by a free run and not by
more norm arithmetic.)

Three things follow, and together they make this the highest-value cheap measurement available:

1. **The famous "17 mN on X and 23 mN on Y" sits on rows carrying under 1 percent of the DC
   norm.** That framing may still be right dynamically, since those rows feed pure double
   integrators, but it has never been tested against the alternative, and the alternative is
   where 97 percent of the magnitude is.
2. **Every DC-resistance measurement ever made covers the wrong rows.** D1 probed routing
   `(3,4,5)`; `curvature_sensitivity.py` iterates 6 states; A6 reported X, Y, dX, dY and
   returned Theta and dTheta as UNRESOLVED. **No rig has ever measured the loss's curvature on
   `aug0` or `aug1`.** So falsified-mechanism 1, "the loss is blind to the DC", was established
   on rows that carry 1 percent of it, using an artifact that is no longer on disk (section
   4.1). If the loss turns out to be soft on the rows that do carry it, the section 6b paradox
   dissolves with no new theory required.
3. **It connects the three orphaned findings.** The latent states never learned the absorber
   (`r2_aug_lin` about 0.35), most of the encoder receives exactly zero gradient, and
   `msd.obs_rank` reads 3 of 8. If the constants that matter live on `aug0` and `aug1`, those
   three stop being separate curiosities and become one statement: the augmentation is carrying
   a free constant in a direction the objective cannot see and the plant may not expose.

**Pre-declare before launching** (required by the campaign's own rule 4, and the reason the last
four voids were caught):

* Masks: `{dX, dY}`, `{aug0, aug1}`, `{dTheta}`, `{Theta, X, Y}`, and full 8-row as the
  reference.
* Control: the full 8-row mask must reproduce step 0's MEAN-ONLY numbers (`99.0 / 98.7 / 99.5`
  percent of FULL) to within 1 percent on the same checkpoints, and the zero-mask arm must
  reproduce FULL exactly. If either fails, no row is readable.
* Reading (a): if `{dX, dY}` alone reproduces most of FULL, the existing force framing is
  confirmed, DC-targeting candidates stay aimed where they are, and the next step is A6 rerun
  on those rows with its artifact written.
* Reading (b): if `{aug0, aug1}` alone reproduces most of FULL, the failure is a free constant
  on the augmented latent rows, the DC-targeting candidate list must be re-aimed, and the
  observability check in section 5 item 4 becomes the immediate follow-up rather than a
  curiosity.
* Reading (c): if no single mask reproduces it and only the full vector does, the constants
  interact and the fix must act on the vector, not on selected rows.
* Grade will be SINGLE (1 seed, 1 record). Say so in the row.

**Ranking, stated explicitly rather than offered as a menu.** The orthogonal-projection gates
(section 5 item 3) are the more important problem for the thesis, since the projection is the
stated scientific contribution and a 56.66 degree basis-stability failure against a 5 degree
tolerance is not a detail. They are second here only because they are a design question needing
a session of their own, while this is an afternoon on a rig that already works with a control
that already passes, and it is a precondition for aiming step 4 of the campaign. If this lands
quickly, the projection gates should be the next session's opening item.

**Do not** re-propose, in this order of certainty: longer training windows; the
multiple-shooting continuity term; adjoint re-weighting of a long-horizon position functional
(weight ratio DC to 150 Hz is `2.27e12`); optimiser swap or `lr` tuning as the deliverable; the
GAM split as an entanglement fix; hard model-class restrictions; zero-mean or window-mean priors
on the residual; oracle states at deployment.

## 8. Traps that have already invalidated runs

* `RunConfig` defaults `up_sample = 2`; the entry file and every checkpoint use `1`.
* Trimming `TRAIN_FILES` changes `compute_normalization`, hence the encoder, hence every
  downstream number. A 4-record trim moved epoch-0 from `1.66e-04` to `1.13e-01`. Validation
  trims are safe; training trims are not. Check whether the script computes `norm` before or
  after the trim.
* `gantry_ckpt_*.pt` is the **best** checkpoint, and since best is epoch 0, that file **is** the
  initialisation. Use `*_last.pth` for a trained model.
* `.pth` files pickle `gantry_dynamic` as a top-level module, so put `scripts/gantry` on
  `sys.path` before `torch.load`. They carry their own `norm`; take weights and `norm` together,
  never mixed.
* The ARTBP checkpoints carry **no** `norm`. Rebuild it with
  `drift-demo/demo_common.build_pipeline(dataclasses.replace(CFG, seed=0))`, the rig that
  trained them. `CFG.seed` is 42 but `train_artbp.py` overrides it to the arm's seed.
* Three ARTBP arms store no epoch-0 row, and the borrowed reference is not arm-invariant: the
  `ep8` arm ran `val_n = 2` against the others' 4, a 21 percent swing on every G1 ratio. Step
  0's control resolved which applies (`1.846057e-04` for `val_n = 4`, `2.242571e-04` for 2).
* In the ARTBP `.npz` files `best_val_sim` and `drift_ratio` describe **different models** in all
  six arms. Fix the scoring basis (best-epoch or final-model) before a run, not after; the two
  bases disagree by 3.4x on `geom`.
* Filtering non-finite values out of a metric series makes divergence look like a flat pass.
* Y in the record names is in units of **10 mm**: `V1_standstill_Yp10` sits at `+100.000 mm`,
  and the training Y envelope is `[-300.018, +300.017] mm`.
* Background jobs get OOM-killed on this machine. Make every run resumable and checkpoint per
  epoch. Piping a run through `grep` or `tail` block-buffers stdout, so the log stays empty
  until exit.
* Every script that produces a number a document will cite must **write that number, with its
  horizon and its units, to a file**. Section 4.1 is what happens otherwise.

## 9. Provenance of this pass

Artifacts I opened and checked directly, 2026-07-27:
`dc-accumulation/results/{step0_dc_sufficiency,step0b_train_constant,step1_testbed,step3_artbp_recompute,step1_control_check}.json`;
`71167/{gantry_results,gantry_ckpt,gantry_state_recovery,gantry_grad_norms}_71167.npz`;
`pysynth-data/results/MS12_ann_dc_force.json` and a listing of that whole results folder;
`msd_transfer_diagnostics/data/{diag_msd_summary,diag_proximal_summary}.json` and
`diag_msd_failure_cause.py`;
`orth_projection/{step7b,step7c,step8,step8b,harness_check}/*.json`;
`diagnostics/system_dynamics.json`; `ARTBP/train_artbp.py` line 226 and `ARTBP/README.md`
line 136; `gantry_dynamic/config.py` line 66 and `gantry_interconnect_dynamic.py` line 72;
`baseline-null/curvature_sensitivity.py` and `pysynth-data/a6_dc_resistance_per_axis.py` for
their output paths.

Documents read in full: `dc-accumulation/README.md`, `diagnostic-overview.md` (section 10 last),
`results-log-2026-07-26.md` sections 11 to 13, and the last six rows of
`gantry-augmentation-problem-log.md` section 12.

Read-only. No training was run, and no file outside this one was modified.
