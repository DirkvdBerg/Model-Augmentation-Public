# Ablation 2026-08-22: which mechanism earned its place, and does it survive noise

Executing `tasks/handoffs/2026-08-21-which-change-made-the-augmented-states-train.md` unattended.
Appended after every unit of work, never composed at the end.

**Reference numbers, fixed before any run** (all `closed_loop_free_run_rms` on V1-V4, 4 x 12.000 s,
`k0 = 17`, closed-loop rollout):

| | value |
|-|-|
| untrained / D-072 gate | `2.1866011034177349e-06` |
| previous plateau | `1.3933793e-06` |
| arm 1 (`nx_aug=2`) | `1.379891e-06`, ablation `1.0183x` |
| **arm 2 (`nx_aug=8`), seed 0** | **`3.795974e-07`**, ablation **`5.2081x`** |
| width-matched control | `1.384274e-06` |
| planted (oracle) reference | `4.176627e-07`, ablation `6.010x` |

Class A boundary (steps 1-2, noiseless): necessary if removal pushes free-run RMS above
`6.0e-07` OR drops the ablation ratio below `2.0x`. `# HEURISTIC:` both (handoff section 10).
Class B boundary (step 3, under noise): scored against the step 2b noise baseline, necessary if
removal moves free-run RMS more than `1.5x` above that baseline OR ablation below `2.0x`.
`# HEURISTIC: 1.5x`.

---

## DEVIATIONS

Logged at the moment each is made, per the handoff's execution discipline.

### D1. `CL_SEED` added to `cl_train.py` (2026-08-22, before any launch)

* **Spec says**: step 0, "Replicate arm 2 at `cfg.seed` 1 and 2, unchanged otherwise."
* **What I did**: added a `CL_SEED` env gate to `cl_train.py`, default `0`.
* **Why**: `cfg.seed` was hard-coded (`_over = dict(seed=0, ...)`, `cl_train.py`). Step 0 is not
  runnable without this. Default `0` reproduces every earlier run bit-identically, and the gate
  prints a line stating that a different seed is a different POLE DRAW, not merely a different
  weight init. `cl_train.py` is an editable file per handoff section 2.
* **Not a substitution**: no pre-registered run is changed by this.

### D2. `AUG_LRU_FREEZE` and `ENC_WA_FREEZE` added to `gantry_dynamic/model.py`

Pre-authorised by the handoff (sections 9 step 2 and step 3) and by the user. Env-gated, OFF by
default. D-072 no-op verification below before any use.

### D3. `PROBE_PERPAIR=1` dropped from all ablations except step 0 seed 1 (logged 11:25Z, BEFORE the first ablation is launched)

* **Spec says**: the section 13 ablation template carries `PROBE_PERPAIR=1`. Section 12 forbids it
  only for F1's ablation, where there is no `A_aa` and no pair structure.
* **What I will do**: pass `PROBE_PERPAIR=1` on the step 0 seed 1 ablation only. Every other
  ablation runs without it.
* **Why**: it is a speed change and is therefore logged before launch, not after. At `nx_aug = 8`
  the per-pair block adds 5 free-run passes to a 3-pass probe, measured at **~35 min extra per
  ablation**, which is 40 % of a whole training run. It is kept exactly where it answers the live
  question: step 0 asks whether the result depends on the DRAW, and "which of the four drawn poles
  carries the ablation" is that question in its sharpest form. Everywhere else the deliverable
  needs only the worst-surface ratio, which the 3-pass core already gives.
* **Cost if wrong**: the per-pair split for the later arms is recoverable at any time by re-running
  `probe_arm_ablation.py` against the saved checkpoint; no training is repeated.

### D3-REVISED (12:30Z). `PROBE_PERPAIR` moved to the server rather than dropped

D3 above dropped `PROBE_PERPAIR` everywhere except step 0 seed 1. Superseded once the work moved to
the cluster: the local seed-1 ablation runs WITHOUT it (25 min instead of 60, and the step 0 verdict
is what the local slot is for), and wave 2 turns it back ON by default for every arm that has an
`A_aa` to split, where the extra five free-run passes cost ~8 min rather than ~35. Net effect: MORE
per-pair data than the handoff asked for, at no wall-clock cost. Nothing is lost either way, since
the split is recomputable from a saved checkpoint without repeating training.

### D5 (12:20Z, BEFORE wave 1 is submitted). The tree is flattened into one parallel wave

* **Spec says**: handoff section 9, "The factors are NESTED, not parallel ... this is a decision
  tree with F1 at the root, and running it as a flat five-row sweep would waste three runs
  answering questions F1 has already closed."
* **What I did**: converted the tree into a flat 15-arm SLURM array
  (`runners/run_ablation_wave1.sh`), with the F1 branch applied at ANALYSIS time.
* **Why**: the user has 14 concurrent server slots. The nesting argument is explicitly about
  WASTED RUNS, i.e. compute, and the arms are scientifically independent; parallel capacity removes
  the only stated objection. Serial-local was measured at 97 min per arm, so the tree costs 11-19 h;
  the flat wave costs about 30 min of wall clock.
* **What is preserved unchanged**: every arm's configuration, all four pre-registered thresholds,
  and the branch semantics. **If F1 passes, F2/F4a/F4b/F5 are reported `moot`, not
  `necessary`/`not necessary`**, exactly as the tree would have. A sub-feature of `AUG_LRU` cannot
  earn a place in the clean `model_augmentation/` implementation if `AUG_LRU` itself is not in it,
  no matter how good its measured number is. Having the number anyway is strictly more information;
  letting it drive an inclusion decision would be scope drift.
* **The one thing the flattening had to solve**: step 2b's noise baseline is defined as "the winner
  of steps 0-2, re-run under noise", and the winner is unknown until F1 returns. Both candidates
  are therefore run (arm-2 config, task 11; F1 config, task 14) and one is discarded at analysis
  time. Zero wall-clock cost.

### D6 (12:22Z). `cl_train.py` now records the checkpoint path it wrote

Not a spec deviation but a defect the parallel restructure exposed and would otherwise have hidden.
deepSI names checkpoints `<ClassName>_<random6>_best.pth` in one shared per-user directory. With 15
arms training concurrently, "the newest `_best`" is whichever arm validated LAST, so the obvious
`ls -t | head -1` in an ablation runner would ablate the wrong arm's model and report a wrong
RESULT rather than crash. `cl_train.py` now writes `checkpoint.best` into its result JSON and
`run_ablation_wave2.sh` reads it back. The local seed-1 run predates this edit, so its checkpoint
was resolved by hand (`SSE_Interconnect_MultipleShooting_Z37aYA_best.pth`, unambiguous because it
was the only run in flight).

### D7 (13:13Z). INCIDENT: I overwrote the source-of-truth band artefact, and restored it

Recorded in full because it nearly invalidated every arm in the factorial and left no trace while
doing so.

* **What happened**: I ran `cl_residual_spectrum.py` with no arguments as a regression check on my
  own edits. Its default record set is `VAL_FILES` (4 records); the artefact in the repo was
  generated with `CL_RS_FILES=all` (18 records). The clean path writes to the fixed name
  `runs/cl_residual_spectrum.json`, so the 4-record result **replaced** the 18-record one.
* **Why it matters**: that file is the SOURCE OF TRUTH for the `AUG_LRU` initialisation band.
  `lru_band_from_artifact` reads it at every gated build. The band changed from
  `[149.90234375, 164.06250] Hz` to `[153.80859, 164.06250] Hz` and `rho` from
  `[0.9794118, 0.9955750]` to `[0.9834960, 0.9933980]`. **Every subsequent `AUG_LRU` arm would have
  been initialised from a different band than the arms it was being compared against, including the
  15 server arms, and nothing in any log would have said so.** The factorial would have been
  measuring the band change as if it were the factor under test.
* **How it was caught**: `READING 3`, which I had added minutes earlier, prints the resulting band.
  It read `12 dominant peaks` where the record says 54, which is 4 records x 3 channels rather than
  18 x 3. The diagnostic caught the mistake the same edit made possible.
* **Recovery**: restored from a backup taken before the run, then verified by calling
  `lru_band_from_artifact` on the restored file: `149.90234375,164.0625` and
  `0.97941181741523697,0.99557503786296653`, byte-identical to the values used by seeds 0 and 1.
* **Re-verification of the edits themselves**: re-run with `CL_RS_FILES=all` to a scratch path.
  **54 dominant peaks, band `[149.90234, 164.06250]` Hz, `rho [0.979412, 0.995575]`,
  `over_floor_db` min 65.4 / median 76.4** -- every one of these matches the handoff's recorded
  values exactly, so the clean path is unchanged by this session's edits.
* **Fix, so it cannot recur**: added `CL_RS_OUT` to `cl_residual_spectrum.py`, same role and same
  rationale as `PROBE_OUT` in `probe_d072_matrix.py`. A verification run now sends its artefact
  somewhere else instead of relying on the operator remembering which file is load-bearing.
* **The general lesson, which is not "be careful"**: the script had a fixed output path for its
  most important product and a default record set that differed from the one that produced it.
  Either alone is survivable; together they make an accidental overwrite the DEFAULT behaviour of
  running the script the obvious way. The user's standing never-overwrite-datasets rule exists for
  exactly this and I did not apply it to an artefact I was only "checking".

### D8 (13:45Z, BEFORE wave 1 is submitted). F4c added: the published-default initialisation

* **Spec says**: step 2 lists four factor arms, F2/F4a/F4b/F5. F4a widens the frequency band, F4b
  widens `rho`, and the handoff explicitly rejects a combined F4 because it "cannot say which one
  carried it".
* **What I did**: kept F4a and F4b unchanged and ADDED a sixteenth arm, F4c, running Orvieto
  Lemma 3.2 exactly as published: `r` uniform on the annulus `[0.05, 0.99]`, `theta` uniform over
  the full representable circle (`1` to `2000 Hz` is `theta in (0, pi]` at `ts = 2.5e-4`), with no
  data input at all.
* **Why**: the handoff's F4a/F4b answer "which HALF of our recipe carries it". They cannot answer
  "should our recipe exist", because neither is the literature default. Our residual-spectrum band
  recipe has **no literature source** (step 4 audit), so without a published-default baseline the
  claim "the band recipe helps" is unfalsifiable. Under the user's direction of 2026-08-22 --
  literature grounding over engineering tuning -- this is the highest-value arm in the wave: **if
  F4c matches the artefact-band arms, the recipe is not earning its place and the initialisation
  falls back to a citable default.** That is a better outcome for the thesis than a recipe that
  works, because it removes the one component a reviewer can press on.
* **Cost**: one extra array slot. At 14 concurrent the array runs `1-16%14`, so wall clock rises by
  one arm, about 25 min.
* **Interpretability**: F4b and F4c share the same `rho` range `[0.05, 0.99]` and differ ONLY in
  the frequency arc, so the pair isolates the frequency half cleanly, and F4a/F4c isolate the
  damping half. The three together are a proper 2x2 minus the redundant corner.

### D9 (13:45Z). Arm PRIORITY reordered, though no arm is dropped

The handoff's drop order is F3c, then F5, then F4b, i.e. it ranks the band arms as least valuable.
Under the user's literature-grounding criterion the right question per arm is **"if it passes,
which UNSOURCED component does it delete?"**, which nearly inverts that:

| rank | arm | deletes, if it passes |
|-|-|-|
| 1 | F1 | `AUG_LRU` entirely: band recipe, `0.377`, phase-arc heuristic, both Telica blockers |
| 2 | F4c, then F4a / F4b | the band recipe, or one half of it. Our least defensible component |
| 3 | F5 | the claim that we IDENTIFY the mode. Frozen poles put this in random-feature / reservoir territory, which has its own literature and is a STRONGER grounding than an unsourced recipe |
| 4 | F2 | nothing: Orvieto Sec. 3.3 prescribes an input path, so F2 tests a deviation FROM the literature rather than a heuristic |
| 5 | F3a/b/c | decides Hoekstra Eq. (31) against our Eq. (7) refutation. Citable either way |

The four seed arms also stop being bookkeeping under this criterion: `6.0e-07` and `2.0x` are
themselves heuristics, and the seed spread is what replaces them with a data-derived interval.
Nothing is dropped; if the queue binds, submit `--array=5,16,11,14` first.

### D4. `CL_CONCURRENT=0` on every run (not a deviation, recorded so its absence is never inferred)

`cl_train.py:130` reads `CONCURRENT = bool(int(os.environ.get('CL_CONCURRENT', 0 if PROBE else 1)))`,
so `CL_PROBE=0` silently ENABLES concurrent subprocess validation, which scores a stale model. Every
launch in this file sets `CL_CONCURRENT=0` explicitly. Verified in the step 0 run A output: the
pre-fit validation returned `2.1866011034e-06`, i.e. the recorded untrained scalar, and the run is
on the serial path.

---

## Timing log

Wall clock of each unit, so the schedule can be re-planned from measurement rather than from the
handoff's estimates.

| unit | started | wall clock | note |
|-|-|-|-|
| D-072 no-op check, both freeze gates unset | 2026-08-22 | **460 s** | `probe_d072_matrix.py` at `na_nb=17 x nx_aug=2` |
| step 0 run A (`CL_SEED=1`) | 10:51:34Z | **97 min** | first training run; the schedule was re-planned from this measurement |
| step 0 run A ablation, no `PROBE_PERPAIR` | 12:30:59Z | **30 min** | 3 free-run passes; add ~8 min/pass for `PROBE_PERPAIR` at `nx_aug = 8` |

### REVISED SCHEDULE, from measurement (recorded 11:21Z, mid-run, as the discipline requires)

Measured on step 0 run A, not assumed:

| component | measured | handoff assumed |
|-|-|-|
| one full 4-record closed-loop free run at `nx_aug = 8` | **~7.5 min** (build + first validation took 8 min from launch) | not stated |
| one training update at `nf = 400`, batch 256 | **6.2-7.2 s/it**, mean ~6.5 s | not stated |
| 520 updates of fit alone | **~56 min** | |
| **one training run end to end** (build + pre-fit val + 520 updates + 2 epoch validations + final val) | **~85 min** | "~50 min" |
| **one ablation WITHOUT `PROBE_PERPAIR`** (3 free-run passes: intact, A, B) | **~25 min** | "~25 min", correct |
| **one ablation WITH `PROBE_PERPAIR=1` at `nx_aug = 8`** (3 + 4 pairs + 1 `W^a` = 8 passes) | **~60 min** | hidden inside the same "~25 min" |

**The handoff underestimates a training run by 1.7x and a per-pair ablation by 2.4x.** Revised
totals, serial, which is the discipline:

* step 0: 2 runs + 2 ablations = `2*85 + 60 + 25` = **4.2 h**
* step 1 (F1): 1 run + 1 ablation = **1.8 h**
* step 2b + step 3 = **5.5 h** (see the saving below)
* **best case (F1 passes): about 11.5 h**
* worst case (F1 fails, + F2, F4a, F4b, F5): **about 19 h**, against the handoff's "about 14 h"

**One saving, and it is structural rather than a corner cut.** Step 2b re-runs the winner with
`CL_NOISE_CONSISTENT=1` and `ENC_WA_ZERO=1`. Step 3's arm F3b is defined as `W^a` zero and
trainable, under noise. **Those are the same configuration**, so step 2b's run IS F3b and step 3
costs two further runs (F3a, F3c), not three. Stated explicitly rather than silently reused, so
that the F3b row in the deliverable table can be traced to the step 2b artefact.

If the budget binds, the handoff's documented drop order stands: F3c first, then F5, then F4b.

Calibration already available from the D-072 unit: **460 s for one build + four full free runs at
`nx_aug = 2`**. A `cl_train.py` run does that same four-record free run once before fit and once
after, plus one per validation. So the fixed overhead of a training run is roughly `2-3 x 115 s`
per free-run pass plus the fit itself, and `probe_arm_ablation.py` is `1 + 2` full free-run passes
(intact, surface A, surface B), plus `n_pairs + 1` more when `PROBE_PERPAIR=1`, i.e. **5 passes at
`nx_aug = 8`** rather than 3. That is the number the handoff's flat "~25 min per ablation" hides,
and it is why the per-pair split is priced separately below.

---

## Results

### Gate: D-072 no-op with `AUG_LRU_FREEZE` and `ENC_WA_FREEZE` unset

```
na_nb    nx_aug   free-run RMS [m]           rel dev        gate
17       2        2.186601103417735e-06      0.000e+00      PASS
```

`runs/d072_noop_check_freezegates.json`, 460.4 s. Both new gates are certified no-ops when unset,
which is the precondition the handoff sets on using either. The build ran with `AUG_LRU=1
AUG_LRU_B=0.377`, i.e. the configuration every arm below runs in.

### Step 0. Does the result exist?

Arm 2's line verbatim plus `CL_SEED=<n>`. Criterion: free-run RMS below `6.0e-07`.

| seed | drawn poles (at init) | free-run RMS | ablation (worst surface) | verdict |
|-|-|-|-|-|
| 0 (recorded) | 159.350 / 162.854 / 151.995 / 153.475 Hz | `3.795974e-07` | `5.2081x` | reference |
| **1** | `r 0.9882 @ 157.90` / `0.9815 @ 162.65` / `0.9849 @ 152.15` / `0.9801 @ 159.98` Hz | **`4.8867311476e-07`** | **`4.5807x`** | **REPRODUCES**, below `6.0e-07` and far above `2.0x` |
| 2, 3, 4, 5 | | | | queued to the server (wave 1 tasks 1-4) |

**Seed 1, full record.** Launched 10:51:34Z, finished 12:29:02Z, **97 min wall clock**.

```
untrained 2.1866011034e-06 m   trained 4.8867311476e-07 m   +77.6515 %
val series (3 points): 2.1866e-06  5.9946e-07  4.8867e-07
  V1_standstill_Yp10.mat   4.353631e-07 m
  V2_aprbs_Ylow.mat        5.982026e-07 m
  V3_ysweep_Yp10.mat       4.292376e-07 m
  V4_lissajous_Ym10.mat    4.728355e-07 m
best at validation 2 of 3   selection picked a trained checkpoint
```

Checkpoint `SSE_Interconnect_MultipleShooting_Z37aYA_best.pth`. Result JSON `runs/cl_train_s0_seed1.json`.

**Three things this settles, and one it does not.**

1. **The e-7 result is not a property of draw 0.** `4.887e-07` at seed 1 against `3.796e-07` at
   seed 0, both far below the `6.0e-07` boundary and both far below arm 1's `1.38e-06`. The
   handoff's "neither reproduces -> STOP" branch is closed, and so is the "exactly one reproduces
   -> the method is draw-dependent" branch, at least at n = 2.
2. **D-072 holds under a new draw.** The pre-fit validation returned `2.1866011034e-06`, the
   recorded untrained scalar, so the pole draw does not disturb baseline equality. Expected from
   the structure (the readout of the augmented rows is exactly zero at init regardless of where the
   poles sit) but now measured rather than argued.
3. **The two draws are not equivalent, and the DIFFERENCE is informative.** Seed 1's draw contains
   a pole at `157.90 Hz`, i.e. within `0.01 Hz` of the true absorber mode at `157.8937 Hz` --
   closer than seed 0's nearest pole (`159.350 Hz`, `1.46 Hz` off) -- and yet seed 1 lands 29 %
   WORSE (`4.887e-07` vs `3.796e-07`). **So "how close is the nearest drawn pole to the true mode"
   does not predict the outcome.** That is direct evidence against the single-resonator reading of
   the mechanism and for the handoff's phase-coherence hypothesis: what matters is that the four
   poles SPAN the mode and can be combined, not that one of them sits on it. Recorded because it
   is the first evidence either way on section 5's "Why arm 2 works", which was explicitly
   unmeasured.

**What it does not settle**: whether the SPREAD `3.80e-07` to `4.89e-07` is draw variance or noise
in the 520-update truncation. Seeds 2-5 answer that.

**Timing**: 97 min, against the handoff's "~50 min" and my own mid-run estimate of 85. The last
two validations cost more than the projection because `its_per_val=epoch` puts a full 4-record
free run inside the progress bar's final tick.

#### Seed 1's ablation, the primary criterion. Launched 12:30:59Z, finished 13:00:37Z, **30 min**.

```
trained, intact                 : 4.886731e-07 m   (-77.65 % vs untrained, -64.93 % vs plateau)
A  ANN blind to x_a             : 2.238442e-06 m   ratio to intact 4.5807x
   in-band 2.013355e-06 (4.921x intact)   out-of-band 9.781839e-07 (3.659x)   band share 0.8090
B  x_a driven to zero           : 2.238425e-06 m   ratio to intact 4.5806x
   in-band 2.013315e-06 (4.921x intact)   out-of-band 9.782303e-07 (3.659x)   band share 0.8090
VERDICT: the augmented states ARE load-bearing: removing them costs 4.5807x
```

`transient-investigation/runs/arm_ablation_s0_seed1.json`, 3 free-run passes, no `PROBE_PERPAIR`
(D3-REVISED).

**STEP 0 IS PASSED, at n = 2.** Both seeds land below `6.0e-07` and both are load-bearing far above
`2.0x` (`5.2081x` at seed 0, `4.5807x` at seed 1). The handoff's stop branch and its
draw-dependence branch are both closed. **The factorial is measuring something real, and step 1 is
authorised to proceed.**

Four further readings from this artefact, none of which the handoff asked for but all of which bear
on the clean implementation:

* **The two ablation surfaces agree to 4 significant figures** (`4.5807x` vs `4.5806x`). Surface A
  blinds the ANN to `x_a`; surface B drives `x_a` to zero. That they cost the SAME says the
  augmented state's only route to the output is through the ANN reading it, with no separate
  contribution from the state's own dynamics. Structurally expected (`Cd_norm` has zero columns on
  the augmented rows, which is why the obvious readout ablation is a no-op) but worth having
  measured: it means **one ablation surface would have sufficed**, and wave 2 could be a third
  cheaper if this holds across arms.
* **Ablating costs more IN BAND than out**: `4.921x` against `3.659x`, and the ablated residual is
  `80.9 %` in-band power. The augmented states are preferentially cancelling the mode rather than
  acting as a generic extra-capacity term. This is the same in-band preference the handoff records
  for seed 0 (`6.241x` in band against `3.409x` out), reproduced at a second draw.
* **Blinding the model to `x_a` returns it to `2.238e-06`, i.e. WORSE than untrained**
  (`2.187e-06`). The trained ANN rows 0-5 have adapted to the presence of `x_a`; take it away and
  the remaining correction is actively harmful. That is the strongest form of "load-bearing" on
  offer, and it is not what a decoration arm looks like (arm 1's `1.0183x` left it essentially
  where it started).
* **Seed 1 is worse on RMS (`4.89e-07` vs `3.80e-07`) AND worse on ablation (`4.58x` vs `5.21x`),
  and the two move together.** So the seed-to-seed spread is not a scoring artefact; a draw that
  fits the mode less well also leans on the augmented states less. Consistent, and it means the
  seeds 2-5 spread will estimate the variance of BOTH numbers at once.


---

## Step 4. Provenance, checked at the PDF

Every check below was done by opening the page, not by reading a previous comment. Four of the six
came out as stated; **two are corrections, and one of those is a correction to a citation currently
in the code.**

### Confirmed, as `# THEORY:`

| mechanism | code | source, checked | verdict |
|-|-|-|-|
| annulus radius draw | `r_init = sqrt(u*(rho_hi^2 - rho_lo^2) + rho_lo^2)` (`model.py`) | Orvieto et al., ICML 2023, **Lemma 3.2, p.7**: `nu = -(1/2) log(u1 (r_max^2 - r_min^2) + r_min^2)`, `theta = 2 pi u2`, and `exp(-nu + i theta)` is uniform on the ring between radii `r_min`, `r_max`. `exp(-nu)` is our expression exactly. | **CONFIRMED**, formula, variable and context all match |
| stable exponential parameterisation of the magnitude | `r = exp(-exp(nu_log))`, `nu_log = log(-log r)` | Orvieto, **Sec. 3.3, p.8**: `lambda_j := exp(-exp(nu_j^log) + i theta_j)`, `nu^log := log(nu)` at init, and `\|lambda\| <= 1` by construction since `nu > 0`. | **CONFIRMED** |
| input normalisation `gamma` | `gamma = sqrt(1 - r*r)`, recomputed every forward pass, not a parameter | Orvieto, **Eq. (7), Sec. 3.4, p.10**: `x_k = Lambda x_{k-1} + exp(gamma^log) (B u_k)` with `gamma^log <- log(sqrt(1 - \|lambda_i\|^2))`. Orvieto makes `gamma^log` TRAINABLE; **footnote 9 on the same page** states they also tried setting `gamma_i` to `sqrt(1 - \|lambda_i\|^2)` at every training iteration and "found it to work similarly in practice". Ours is that footnote-9 variant. | **CONFIRMED**, with the variant named rather than glossed |
| the phase-arc restriction is NOT Lemma 3.2 | `theta_init = theta_lo + v*(theta_hi - theta_lo)` over the data-derived band | Lemma 3.2 draws `theta = 2 pi u2`, i.e. the FULL circle. Restricting the arc is **Sec. 3.4, p.10, "Reducing Eigenvalue Phase at Initialization"**, which is argued empirically (Fig. 4, Fig. 5) and for a different reason (long-sequence reasoning), not proved. | **CONFIRMED AS A HEURITIC**, exactly as the handoff pre-registered. Label it `# HEURISTIC:`, never Lemma 3.2 |

### Correction 1, and it is in `gantry_dynamic/model.py`, not in `model_augmentation/`

`model.py`'s comment reads

```
# THEORY: Orvieto et al. ICML 2023 Sec. 3.3 -- nu_log = log(-log r), theta_log = log th.
```

The `nu_log` half is right. **The `theta_log` half is not in Orvieto.** Sec. 3.3 p.8 gives
`lambda_j = exp(-exp(nu_j^log) + i theta_j)`: the magnitude is exponentiated, the PHASE is learned
directly as `theta_j`. Our `th = exp(theta_log)` puts a second exponential on the phase that the
source does not have. It is not wrong as engineering, and it has a real justification (it keeps the
frequency strictly positive, so a pole cannot cross onto the real axis and split the conjugate
pair during training), but that justification is ours, so the label must split:

```
# THEORY: Orvieto et al. ICML 2023 Sec. 3.3 -- nu_log = log(-log r) is theirs, verbatim.
# HEURISTIC: theta_log = log(theta) is NOT. Orvieto learns the phase directly. The extra
# exponential is ours, to keep the phase strictly positive so a conjugate pair cannot
# collapse onto the real axis mid-training.
```

**Not applied in this session**, because it is in `model.py` and the ranking runs are in flight
against that exact file; applying it now would put a different source file behind the later arms.
It goes into the clean `model_augmentation/` implementation, and it is recorded in `docs/decisions.md`.

### Correction 2, RECORDED NOT APPLIED, per the user's instruction

`model_augmentation/fit_systems/pre_encoder.py:422` currently reads

```
# HEURISTIC, with no literature source: kaiming_uniform_ on both blocks.
```

**That is false, and verified false at the PDF.** Hoekstra, Gyorok, Verhoek, Toth, Schoukens,
arXiv:2602.17297 (`literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`),
**p.9, Sec. 5.4.2, Eq. (31)** gives the augmented-state encoder block
`x_{a,k\|k} = psi_a(theta_aug, y^{k-1}_{k-na}, u^{k-1}_{k-nb})` and states directly underneath:
*"where the weights and biases of psi_aug are initialised by the Xavier approach."*
**p.10, Sec. 5.4.3** adds: *"All matrices not required to set the baseline behaviour at
initialisation (29) have all elements m of the matrix initialised randomly, according to [32],
i.e., m ~ U(-1,1)."* So a random `W^a` **is** Hoekstra's stated convention, and the comment's
central claim is wrong.

**A second error the handoff did not catch, found by reading the page**: the source says **Xavier**
(Glorot). Our code calls `nn.init.kaiming_uniform_`. Those are different initialisers with
different scales. So the comment is wrong twice over: it denies a source that exists, and the code
does not implement the convention that source specifies.

**Why we still depart from it**, and the argument is by refutation, not by absence of a source:
Hoekstra's own encoder paper (arXiv:2602.13108,
`literature/augmentation/Encoder initialisation methods in the model augmentation setting.pdf`),
**p.3, Eq. (7)**, defines the encoder as approximating
`x_bar_k = E_e[x_k | u^{k-1}_{k-n}, y^{k-1}_{k-n}]` and calls it "an unbiased estimator of `x_k`".
Under D-072 the augmented readout is **exactly zero**, so at initialisation `x_a` has no effect on
`y` whatsoever and the window `(u_hist, y_hist)` carries no information about it. Eq. (7)'s
conditional expectation therefore collapses to the unconditional one, and the only value consistent
with baseline equality is `0`. **A random `W^a` is not an unbiased estimator of anything at
initialisation; it is an arbitrary O(1) functional of the window injected into a state the output
cannot see.**

The required comment text, wherever this ships:

```
# THEORY, then REFUTED: Hoekstra, Gyorok, Verhoek, Toth, Schoukens, arXiv:2602.17297 p.9
# Sec. 5.4.2 Eq. (31) initialises psi_aug by the Xavier approach (p.10 Sec. 5.4.3: m ~ U(-1,1)),
# so a random W^a IS his stated convention. We depart from it deliberately: his own encoder
# paper, arXiv:2602.13108 p.3 Eq. (7), defines the encoder as E[x_k | u_hist, y_hist], and under
# D-072 the augmented readout is exactly zero, so the window carries no information about x_a
# and that conditional mean is zero.
```

Never "no literature source". `model.py`'s `ENC_WA_ZERO` comment block repeats the same wrong claim
and needs the same correction. `docs/references.md` line 47 also asserts "Our `W^a` init is
therefore an assumption with no literature source"; that sentence is wrong for the same reason and
is corrected in `docs/decisions.md` rather than edited here, since the correction belongs with the
decision.

`model_augmentation/` is left unmodified, as instructed.

### Correction 3, milder: the 2x2 real Jordan form is cited to the wrong paper

Resolved by one `deep-research` subagent (handoff section 14), because the PDF is not in
`literature/` and the key is not in `docs/references.md`.

**The bibliographic record CHECKS OUT.** Forgione, Mejari, Piga, *"Model order reduction of deep
structured state-space models: A system-theoretic approach"*, **2024 IEEE 63rd CDC, pp. 8620-8625**,
DOI `10.1109/CDC56724.2024.10886865`, free preprint `arXiv:2403.14833`. Authors, venue, year and the
page range all match, and **Remark 1 exists**, on p.5 of the preprint, read in full:

> "Remark 1 System (3) has an equivalent complex-conjugate representation: [Eq. 5a-5b] with
> `x~_k in C^{2n_x}`, which in turn may be transformed in a real Jordan form, with a block-diagonal
> state-transition matrix containing `n_x` 2x2 blocks, see e.g. Appendix E.3 of [17]. The
> complex-valued, diagonal representation (3) is preferred for its implementation simplicity and
> halved state dimension."

**But it does not support the claim we hang on it.** Remark 1 is a one-sentence pointer: it asserts
the equivalence and the block count, and never writes the rotation-scaling matrix, never uses
`r`/`cos w`/`sin w`, and never states that the spectral radius is exactly `r`. It also explicitly
**declines** the real form in favour of the complex diagonal one, which is the opposite of what our
`AugLRUBypass` does. The derivation is in the reference Remark 1 points at, **[17] = Orvieto et al.
ICML 2023, Appendix E.3, Eq. (26)**: for a conjugate pair with eigenvector `v`, the real basis
`V~ = [Re(v) Im(v)]` gives `A V~ = V~ [[Re lambda, -Im lambda], [Im lambda, Re lambda]]`, which is
our block in Cartesian rather than polar form. Orvieto attributes it to standard real-Jordan-form
texts, i.e. it is classical linear algebra, not an ML contribution.

**So the label should read:**

```
# THEORY: Orvieto et al., ICML 2023, Appendix E.3 Eq. (26) -- the real realisation of a
# complex-conjugate pair lambda = r e^{i w} is r[[cos w, -sin w],[sin w, cos w]], spectral
# radius exactly r, independent of state. Forgione, Mejari, Piga, IEEE CDC 2024
# pp. 8620-8625 Remark 1 is the control-venue pointer to it, not the derivation, and that
# paper prefers the complex diagonal form we do NOT use.
```

Caveat recorded honestly: the IEEE version of record was not read (TU/e SSO not authenticated); the
remark text above is from the 14-page arXiv v1. Remark numbering in the 6-page published version is
very likely identical but is not verified. This does not affect the substance, since the substance
is now attributed to Orvieto E.3 either way.

Two process notes from that run, kept because they are the kind of thing that causes a wrong
"already held locally": (a) `literature/stability-training/claude-deep-research-perstep-rollout-diagnostics.md:90`
cites a DIFFERENT Forgione/Mejari/Piga paper (arXiv:2206.12928), a near-match trap; (b) the skill's
own suggested fix was to add a "verify one citation" fast path (Crossref `query.author` +
`query.bibliographic=<venue> <year>`, then DOI content negotiation, then arXiv exact-title), since
dblp returned HTTP 500 and OpenAlex author enumeration was pure overhead. **Not applied to
`.claude/skills/deep-research/SKILL.md` in this session** (outside this task's scope, and the
standing rule is to modify only what was asked); recorded here so it is not lost.

### Still open

| citation | status |
|-|-|
| `AUG_LRU_B = 0.377` scale | `# HEURISTIC:` confirmed, no source claimed. The `1/sqrt(n_in)` SHAPE of `B_a` is Orvieto Sec. 3.3 (Glorot on `B`); the `0.377` multiplier on top of it is ours and unexplained in the code. If `AUG_LRU` survives F1, this number needs either a derivation or an explicit "tuned, value arbitrary" note. |
| `over_floor_db > 10` peak threshold | `# HEURISTIC:` confirmed, no source claimed, already labelled as such in `model.py`'s `lru_band_from_artifact`. |

---

## Deliverable table

One row per mechanism. `necessary` / `not necessary` / `moot` / `untested`, each with free-run RMS,
ablation ratio, provenance, and behaviour under noise. Filled as results land; a row that stays
`untested` stays `untested` rather than being argued into a verdict.

| # | mechanism | class | test | free-run RMS | ablation | provenance | under noise | verdict |
|-|-|-|-|-|-|-|-|-|
| 0 | **the result itself reproduces across the pole draw** | gate | step 0, seeds 0 and 1 | `3.795974e-07`, `4.8867311476e-07` | `5.2081x`, `4.5807x` | n/a | pending seeds 2-5 | **PASSES at n = 2.** Both below `6.0e-07`, both far above `2.0x`. The stop branch and the draw-dependence branch are closed |
| 1 | `nx_aug` 2 -> 8 | A | arm 1 vs arm 2, width-matched control | `1.379891e-06` -> `3.795974e-07` | `1.0183x` -> `5.2081x` | `# HEURISTIC:` capacity choice; Kessels needed `n_ext = 14` for a comparable machine | pending step 2b | **necessary** (established before this session; the width-matched control at 828 ANN params lands on arm 1, so it is not ANN width) |
| 2 | `AUG_LRU` live `A_aa` | A | F1 | pending | pending | `# THEORY:` Orvieto Sec. 3.3 p.8 (verified) | n/a if moot | pending |
| 3 | `AUG_LRU_B` (`B_a` input path, D-151) | A | F2, only if F1 fails | pending | pending | `# THEORY:` Orvieto Sec. 3.3 for the `1/sqrt(n_in)` shape; `# HEURISTIC:` the `0.377` multiplier | n/a | pending |
| 3b | **the band recipe existing at all** (residual-spectrum-derived init vs the published default) | A | **F4c**, Orvieto Lemma 3.2 as published | pending | pending | **none. This is the component with NO literature source**, and F4c is the arm that can delete it | the recipe's threshold is unsafe on coloured backgrounds (see the derivation below), which is the Telica case | pending, and it is the highest-value row in this table |
| 4 | data-derived FREQUENCY band | A | F4a, only if F1 fails | pending | pending | `# HEURISTIC:` phase-arc restriction is Orvieto Sec. 3.4 empirical, **not** Lemma 3.2 (verified) | band unchanged clean vs noisy, 24 dB margin lost of 38.8 dB headroom | pending |
| 5 | data-derived DAMPING band (`rho`) | A | F4b, only if F1 fails | pending | pending | `# THEORY:` `rho = exp(-zeta wn Ts)`; `# HEURISTIC:` the `over_floor_db > 10` peak threshold behind it | pending | pending |
| 6 | trainable vs frozen poles | A | F5, only if F1 fails | pending | pending | n/a, it is an ablation of 2 | pending | pending |
| 7 | `ENC_WA_ZERO` (`W^a` init value) | **B** | F3a vs F3b, **under noise** | pending | pending | `# THEORY, then REFUTED:` Hoekstra arXiv:2602.17297 p.9 Eq. (31) Xavier; refuted by arXiv:2602.13108 p.3 Eq. (7) under D-072 (both verified, D-152) | THE test; `1.013x` noiseless is not evidence against it (encoder amplifies input noise `1919.8x`) | pending |
| 8 | `W^a` trainable at all | **B** | F3c (`ENC_WA_FREEZE`), **under noise** | pending | pending | untested in the literature; Eq. (7) constrains the VALUE, not trainability | pending | pending |
| 9 | `closed_loop_rollout` (stabilized PEM, D-147) | **B** | **cannot be ablated as a factor**; step 2b is its first training-level test | pending | pending | `# THEORY:` Sugie & Maruta 2020 Eq. (8) Sec. 3, verified to the float32 floor across four decades of sigma (`alpha_cancellation.json`) | step 2b | pending |
| 10 | `CL_NOISE_CONSISTENT` (C5) | **B** | step 2b is its first run ever | pending | pending | `# THEORY:` Sugie & Maruta 2020, same | it IS the noise condition | pending |
| 11 | `AUG_LRU_NA_NB = 17` (T2's lag rule) | **B**-ish | not ablatable: D-072 fails away from 17 | n/a | n/a | `# THEORY:` Beintema, Schoukens, Toth 2023 Automatica 156:111210 Sec. 3.4 p.5 (na, nb are free design variables) | n/a | **necessary by construction**: at any other lag the arm starts from a model that is not the baseline and cannot be attributed at all (`d072_matrix_probe.json`) |
| 12 | LRU **exponential parameterisation** of `\|lambda\|` | A, but argued not run | not run; argued from sample rate + loop bandwidth | n/a | n/a | `# THEORY:` Orvieto Sec. 3.3 p.8 (verified) | **stronger on Telica than in simulation**: at 20 kHz the admissible annulus is `1.0e-3` wide with its edge `8e-6` from the unit circle, about 70 float32 ulps; see the pre-registered band section | **necessary on real data** if `AUG_LRU` survives F1 at all |

---

## Real-data procedure

### PRE-REGISTERED: the Telica band, stated before any result of this session is seen

Handoff step 5 item 4 requires this in writing first, because `lru_band_from_artifact` correctly
REFUSES on a dataset with no strong residual peak, and the hand-supplied `AUG_LRU_BAND` /
`AUG_LRU_RHO` is exactly where oracle knowledge would enter unnoticed. Written 2026-08-22, before
step 0 returned. Only two ingredients are allowed: **loop bandwidth and sample rate.**

**Sample rate.** Telica logs at `SamplingTime = 5e-5 s`, i.e. **20 000 Hz**
(`docs/kamtin-telica-schema.md`; `SamplingFrequency = 20000`, D-073 supersedes the old 10 kHz
assumption). Nyquist is 10 kHz, and the usual 10-samples-per-period floor for a discrete
second-order mode puts an upper bound of `fs/10 = 2000 Hz`. **That bound is not binding here**, so
the sample rate constrains only `rho`, not `f`.

**Loop bandwidth.** `telica_plant_frf.py` gives an identifiable band **below 83 Hz on X and below
55 Hz on Y**. Above that edge the measured FRF carries no usable plant information in these
closed-loop logs, so a pole initialised there cannot be identified no matter what the augmentation
does. An augmentation routed to BOTH axes is bounded by the smaller of the two.

**The band, therefore:**

| quantity | value | source |
|-|-|-|
| `f_hi` (both axes) | **55 Hz** | Y-axis identifiable edge, the binding one |
| `f_hi` (X-only variant) | 83 Hz | X-axis identifiable edge |
| `f_lo` | **5 Hz** | `# HEURISTIC:` below this the tracking-error record is dominated by the motion profile itself, and closed-loop output error is structurally near-blind below crossover (`landau2011adaptive` Eq. 9.78, `karimi1998bias`, already cited in problem-log section 15.5). Not derived; a stated engineering choice, to be replaced by the measured crossover when an open-loop-equivalent FRF exists. |
| `zeta` range | `[0.005, 0.06]` | `# HEURISTIC:` plausible damping for a machine structural mode. Anchored, not invented: the simulation's own true mode, `rho = 0.986982` at `157.8937 Hz`, inverts to `zeta = 0.05276` at `Ts = 2.5e-4`, so the upper end is set just above the one damping value in this project that can be checked against a known truth, and the lower end is a decade below it. |

`rho = exp(-zeta * wn * Ts)`, `wn = 2*pi*f / sqrt(1 - zeta^2)`
(`# THEORY:` discrete pole magnitude of a second-order mode, the same formula
`lru_band_from_artifact` uses). Evaluated over the corners of `f in [5, 55]` x `zeta in [0.005, 0.06]`
(`scratchpad/rho.py`, run against scipy):

| rate | `AUG_LRU_BAND` | `AUG_LRU_RHO` | `nu_log` span |
|-|-|-|-|
| Telica native, `Ts = 5e-5 s` | `"5,55"` | `"0.998962,0.999992"` | `[-11.75, -6.87]` |
| resampled to this project's `4000 Hz`, `Ts = 2.5e-4 s` | `"5,55"` | `"0.994820,0.999961"` | `[-10.15, -5.26]` |

**A finding that falls out of the pre-registration itself, and it is not about noise.** At Telica's
native 20 kHz the whole admissible annulus is `[0.998962, 0.999992]`, a width of `1.0e-3`, and its
upper edge sits `8e-6` from the unit circle. Orvieto's annulus draw
`r = sqrt(u*(r_max^2 - r_min^2) + r_min^2)` therefore draws from a ring whose upper edge is only
about 70 float32 ulps from `1.0` (`eps = 1.19e-7` at that magnitude): a direct `r` parameterisation
would resolve the top of the band to roughly two significant digits and could round a pole ONTO or
outside the unit circle, which is an instability, not an accuracy loss. The exponential
parameterisation removes this exactly: `nu_log = log(-log r)` spreads the same annulus over
`[-11.75, -6.87]`, a span of `4.9` in a variable of order 10, where float32 has full precision.

**So the LRU exponential parameterisation is not decoration on real data: it is what makes a 20 kHz
band drawable at all.** That argument is independent of every run below, rests only on the sample
rate and the loop bandwidth, and holds whichever way the factorial comes out. It is also the first
argument in this project for an `AUG_LRU` sub-feature that is STRONGER on Telica than in
simulation, where `Ts` is 5x larger and the annulus 5x wider.

**What is NOT decided by the above, and must not be smuggled in**: the band says where a pole MAY
be initialised, never that a mode EXISTS there. On Telica the residual `y - y_baseline` is
everything the FP model gets wrong, not a near-pure absorber signature at 65-168 dB over floor as
in simulation, so the peak-picker's dominant peak carries no guarantee of being a missing mode.
`lru_band_from_artifact` raising on Telica is correct behaviour and must be left in.

### Step 5 items 1-3, MEASURED 2026-08-22. Run unconditionally, per section 10.

All at `CL_RS_FILES=all` (18 records x 3 channels = 54 record-channels), base sigma
`8.544e-9, 7.762e-9, 6.539e-9` m, artefacts
`runs/cl_residual_spectrum_noise_{consistent,yonly}_x<scale>.json`.

**Item 1, the missing provenance: FIXED.** Every artefact now carries a `noise` block (sigma, base
sigma, scale, consistent flag, RNG description, what it was applied to) and a `config` block
(record list, `nperseg`, band, `K0`, `ts`). A separate `summary.band_recipe` block records what
`lru_band_from_artifact` WOULD return from that artefact, so the recipe's behaviour is readable off
the file instead of requiring a model build. A defect fixed on the way: the noise RNG was
constructed inside `residual_for` with a fixed seed, so **all 18 records received the byte-identical
noise sequence** -- one realisation repeated, not white noise across the set, which matters because
the band recipe aggregates over records. Now seeded per record from `crc32(name)`.

**Item 2, `u`-side noise: MEASURED, AND IT DOES NOT MATTER.** The handoff expected this to be a
blocker ("same asymmetry C5 fixed for the training drive, unfixed here"). At nominal sigma:

| | band [Hz] | `rho` | `over_floor_db` min / median |
|-|-|-|-|
| clean | `[149.90234, 164.06250]` | `[0.979412, 0.995575]` | 65.4 / 76.4 |
| noise, `y` only | `[149.90234, 164.06250]` | `[0.979412, 0.995577]` | 48.7 / 52.1 |
| noise, CONSISTENT (`u -= C_fb(v)`) | `[149.90234, 164.06250]` | `[0.979410, 0.995576]` | 49.1 / 52.4 |

**The band is bit-for-bit the same and the margin differs by 0.4 dB.** So the `u`-side asymmetry,
which is real and load-bearing for the TRAINING drive (C5), is irrelevant to the INITIALISATION
band. One of the two hardest-sounding Telica blockers is closed, and closed by measurement rather
than by argument. The physically-correct path is implemented and is now the one to use, but nothing
downstream depended on it.
(The `y`-only row also reproduces the handoff's recorded `76.4 -> 52.2` / `65.4 -> 48.8` to 0.1 dB,
which independently confirms the rewritten path against the old artefact.)

**Item 3, the breakdown SNR: MEASURED. This is the number that did not exist.**

| sigma scale | dominant peaks (of 54) | band [Hz] | `over_floor_db` min | recipe |
|-|-|-|-|-|
| 1 | 54 | `[149.90, 164.06]` | 49.1 | succeeds, band CORRECT |
| 10 | 54 | `[149.90, 164.06]` | 29.1 | succeeds, band CORRECT |
| 100 | 53 | `[149.90, 165.53]` | 11.5 | succeeds, upper edge has MOVED |
| 300 | 18 | `[153.32, 165.53]` | 24.2 | succeeds, both edges moved |
| 1000 | 18 | `[154.30, 165.53]` | 14.1 | succeeds, band wrong by 4.4 Hz at the low edge |
| 1500 | 18 | `[150.39, 166.99]` | 10.8 | succeeds |
| **2000** | **1** | **`[159.67, 159.67]`** | 10.5 | **succeeds, and returns a ZERO-WIDTH band** |
| 2500 | 0 | -- | -- | **raises** |

**The stated SNR requirement**: the recipe stops returning anything at **2500x** the Telica-derived
noise level, i.e. sigma `[21.4, 19.4, 16.3] um`, which is **68 dB** of margin. The handoff's
"38.8 dB of headroom" was inferred from two points and was pessimistic by nearly 30 dB.

**But the headline finding is not the breakdown point, and it is bad news the handoff did not
anticipate.** Read the table by band accuracy rather than by success:

* The recipe **degrades long before it fails**. By 100x the upper edge has moved a bin; by 300x the
  lower edge has moved 3.4 Hz; by 1000x the band is `[154.30, 165.53]` against a truth of
  `157.8937`, i.e. still containing the mode but 4.4 Hz wrong on one edge.
* At 2000x it returns `[159.67, 159.67]` -- **a single point, from a single surviving peak out of
  54 record-channels, with no width at all.** `lru_band_from_artifact` reports success. Every pole
  would then be drawn at exactly one frequency, and the "band" would be a hard-coded mode wearing
  the costume of a data-derived one.
* `over_floor_db` **is not monotone in sigma** (49.1, 29.1, 11.5, 24.2, 14.1, 10.8, 10.5) because
  the weak peaks fall below the 10 dB threshold first and the statistic is a min over the
  SURVIVORS. **So margin-in-dB cannot be used as a health check.** The monotone quantity is the
  peak COUNT: 54, 54, 53, 18, 18, 18, 1, 0.

**Consequence for the clean implementation, and it is a concrete design change.** The current
contract is "raise if there are NO strong peaks", which is a test for zero. That contract passes a
one-peak, zero-width band as valid. `lru_band_from_artifact` must additionally report
`n_dominant_peaks` and the band WIDTH, and refuse when the peak count is a small fraction of the
record-channels available:

```
# HEURISTIC: require dominant peaks on at least 1/3 of record-channels AND a band wider than one
# FFT bin. Measured 2026-08-22: at 2000x nominal noise the recipe returns a ZERO-WIDTH band from a
# SINGLE surviving peak out of 54 and reports success, which is indistinguishable at the call site
# from a well-determined mode. The zero test the contract currently performs does not catch this.
```

The fraction and the width are engineering choices and are labelled as such; what is measured is
that a test for zero is insufficient.

### The `over_floor_db > 10` threshold, DERIVED. It was not conservative; it was unsafe.

User direction 2026-08-22: literature grounding over engineering tuning, and no unexplained
numbers. The 10 dB peak threshold is the gate on the whole band recipe and is the number that will
be applied to Telica, where whether a mode exists is genuinely unknown. So it was calibrated rather
than argued.

**Method.** Monte Carlo of the null hypothesis "this channel contains NO lightly damped mode",
pushed through the EXACT pipeline: `signal.welch(nperseg=8192, detrend='constant')`, the
`find_peaks(prominence=0.3, distance=8)` filter, the half-power `zeta_ok` test, and the same floor
estimator. Simulation rather than the analytic `Gamma(K, 1/K)` result for a Welch ordinate, because
that result accounts for none of the prominence filter, the `zeta_ok` filter, the Hann window, the
50 % overlap or the median-over-band floor, and all five change the null distribution of the
MAXIMUM, which is the statistic actually used. 1500-2000 realisations per row, record length
`48000 - K0`, family-wise `alpha = 0.05` over the 54 record-channels by the Sidak correction.
Backgrounds swept as `1/f^a`, because the residual on real data is not white.

| background | T, **global-median floor** (current recipe) | T, **local running-median floor** |
|-|-|-|
| white, `a = 0` | 6.0 dB | 6.6 dB |
| `a = 0.5` | 14.5 dB | 6.9 dB |
| `a = 1` (pink) | **25.1 dB** | 6.9 dB |
| `a = 1.5` | 35.8 dB | 6.9 dB |
| `a = 2` | **46.6 dB** | 6.9 dB |

**Finding 1: `> 10` is unsafe for any background steeper than about `1/f^0.4`.** It is not the
conservative choice it looks like. On a pink background the false-alarm-controlled threshold is
25 dB and the recipe uses 10, so it fires on the broadband tilt and reports a mode. **This is
exactly the Telica case**: there the residual is everything the FP model gets wrong, dominated by
friction, drift and cable forces, i.e. steeply coloured at low frequency. The handoff's fourth
band-source problem, "the peak-picker will return the dominant peak with no guarantee it is a
missing MODE", is now quantified rather than feared.

**Finding 2: the cause is the FLOOR estimator, not the threshold.** The recipe estimates the floor
as the median over the whole `[5, 1900] Hz` band, so a broadband tilt is read as prominence. With a
LOCAL running median the threshold becomes **independent of the background** (6.6 to 6.9 dB across
two decades of colour) and is then a property of the ESTIMATOR alone, derivable once:

```
# THEORY: false-alarm-controlled detection threshold. T = 6.9 dB is the Sidak-corrected
# family-wise alpha = 0.05 quantile of max(over_floor_db) under the null "no lightly damped mode",
# Monte-Carlo'd through this exact estimator over 1/f^a backgrounds for a in [0, 2], where it is
# invariant (6.6-6.9 dB). Requires the LOCAL floor; with the global-band median the same quantile
# runs from 6.0 dB (white) to 46.6 dB (a = 2) and no single number is correct.
```

The local window is `401` bins `= 196 Hz`, chosen by separation of scales rather than by taste: an
order wider than the `~15 Hz` mode it must not suppress, an order narrower than the `1895 Hz` band
whose tilt it must track.

**This deletes a heuristic rather than retuning one**, which is the point. The threshold stops
being a chosen number and becomes a stated false-alarm rate.

**NOT APPLIED, deliberately.** Changing the floor estimator changes the artefact, which changes the
band, which would make every arm incomparable with seeds 0 and 1 and with the handoff's reference
table. It is recorded for the clean implementation. It may also be moot: F4a and F4c may delete the
band recipe outright, and with it the threshold.

### The `AUG_LRU_B = 0.377` scale: better sourced than I first said, and used out of its range

D-151 records it as data-derived, not arbitrary: *"the scale putting `RMS(x_a)` equal to
`RMS(x_phys)` in normalised coordinates"*, measured on **seed 0 at `nz = 11`**, with the explicit
condition *"It is data-derived per dataset and must be re-measured, never carried across datasets
as a constant."*

**`nz = 11` is `nx_aug = 2`. Every `nx_aug = 8` arm runs at `nz = 17`, and the scale was never
re-measured.** The drawn `||B_a||_F` differs accordingly: `5.55e-01` at `nx_aug = 2` against
`8.73e-01` at `nx_aug = 8` (seed 1). So arm 2, the `3.795974e-07` result, and every arm in wave 1
carry a `B_a` scale calibrated for a different configuration, in violation of D-151's own stated
condition.

This does not invalidate F2, which tests the path on/off rather than its scale, and it does not
change the step 0 verdict.

#### RE-MEASURED at `nz = 17`, 2026-08-22, on user instruction before submitting

D-151's criterion reproduced verbatim from `probe_input_injection.py:286-308`: `rms_phys` from the
encoder's physical rows at the first window, `rms_xa` from a 4-segment closed-loop rollout with
`B_a` at the raw `N(0, 1/nz)` draw, `scale = rms_phys / rms_xa` (exact, since `x_a` is linear in
`B_a` at init where `NL = 0`). Harness: `scratchpad/recal_ba.py`.

| config | `\|\|B_a\|\|_F` raw | `RMS(x_phys)` | `RMS(x_a)` | calibrated scale |
|-|-|-|-|-|
| `nx_aug = 2`, `nz = 11`, seed 0 | 1.4725 | 0.72123 | 1.8364 | **0.3927** |
| `nx_aug = 8`, `nz = 17`, seed 0 | 1.8604 | 0.72123 | 1.9844 | **0.3634** |

**The harness check did not reproduce D-151's `0.377` at `nz = 11`; it returned `0.3927`, 4.2 %
away.** That is explained rather than waved past: D-151's number was measured through
`probe_input_injection.attach_injection`, which builds `B_a` from its OWN generator seeded at 0,
whereas the production path in `model.py` uses `manual_seed(cfg.seed + 150)` and draws the POLES
first, so `B_a` consumes different random numbers. **The two were never the same draw.**

That prompted the measurement that actually settles it, the draw-to-draw scatter at `nz = 17`:

| seed | 0 | 1 | 2 | 3 | 4 |
|-|-|-|-|-|-|
| calibrated scale | 0.3634 | 0.3784 | 0.4285 | 0.3605 | 0.4396 |

**mean `0.3941`, sd `0.0373`, spread `20.1 %`.** `0.377` sits `4.3 %` below the mean, i.e. **well
inside the scatter of the quantity it is supposed to be**.

**Three conclusions.**

1. **Safe to submit unchanged.** `0.377` at `nx_aug = 8` gives
   `RMS(x_a)/RMS(x_phys) = 0.377 x 1.9844 / 0.72123 = 1.037`, i.e. it satisfies D-151's criterion to
   3.7 % on the seed-0 draw the arms actually use. The "used out of its range" defect is real as a
   provenance claim and immaterial as a numerical one.
2. **But no constant can be correct.** The scale is a property of the DRAW, not of the dataset, and
   the draw varies by 20 %. D-151's instruction to re-measure "per dataset" is too weak; it is per
   SEED. Any hard-coded value, `0.377` or `0.3941`, is wrong for four seeds out of five.
3. **So the clean implementation should not have this constant at all.** `RMS(x_a)` is linear in
   `B_a` at initialisation, so the correct scale is computable in closed form at build time from
   one rollout, exactly as this harness does. **Computing it deletes the heuristic instead of
   re-tuning it**, and it is the same move as the local-floor fix for the peak threshold. Recorded
   for the clean implementation; NOT applied now, because changing `B_a` changes every arm.

### RESIDUAL BLA: a parametric replacement for the band recipe. Measured 2026-08-22.

User proposal: initialise the augmented states from a BLA rather than from a peak-picked band. Two
BLAs are possible and only one is useful. The **BLA of the full system** (`u -> y`) has poles
dominated by the baseline dynamics we already hold exactly, so the missing mode would have to be
extracted as a difference of pole sets. The **BLA of the RESIDUAL** (`u -> r`, `r = y - y_baseline`)
has poles that ARE the missing dynamics. That is Ljung's Model Error Model, and it is the same
object the band recipe approaches with a PSD and a peak-picker.

Fitted as a MISO ARX per output channel, `A_j(q) r_j(k) = sum_i B_ij(q) u_i(k) + e_j(k)`, over all
18 records x 3 channels, orders `n_A = n_B` swept 4 to 32. `scratchpad/residual_bla.py`,
`residual_bla.json`. ARX specifically because Reinelt, Garulli & Ljung (Automatica 38(5):787-803,
2002) use an ARX error model, so this is MEM in its published form.

**Stabilisation diagram** (median over the 54 record-channels of poles in 100-250 Hz with
`0 < zeta < 0.2`):

| `n_A` | 4 | 8 | 12 | 16 | 20 | 24 | **28** | 30 | 32 |
|-|-|-|-|-|-|-|-|-|-|
| median `f` [Hz] | 157.221 | 157.595 | 157.759 | 157.848 | 157.881 | 157.888 | **157.8946** | 157.896 | 157.899 |
| median `zeta` | 0.0522 | 0.0527 | 0.0529 | 0.0529 | 0.0529 | 0.0527 | **0.05257** | 0.0512 | 0.0505 |

Monotone convergence to the truth, then `zeta` degrades past `n_A = 28` (over-fit). **The order
selects itself off the diagram; no threshold is involved anywhere.**

**Against the known absorber mode `157.8937 Hz`, `zeta = 0.05276`, `rho = 0.986982`:**

| method | frequency | error | damping | needs |
|-|-|-|-|-|
| peak-picker (current recipe) | band `[149.90, 164.06]` Hz, 14.2 Hz wide; fit `158.203 Hz` | `+0.20 %`, **1 FFT bin** and bin-quantised at 0.488 Hz | `rho` band `[0.9794, 0.9956]` | a 10 dB threshold, a half-power `zeta`, a scatter-to-band rule, and a random draw |
| **residual BLA, `n_A = 28`** | **`157.8946 Hz`** | **`+0.001 %`, i.e. `0.0009 Hz`** | **`zeta = 0.05257` (`-0.4 %`), `rho = 0.987029` (`+0.005 %`)** | **none of them** |

**The BLA is ~540x more accurate in frequency than one FFT bin**, returns damping directly to
`0.4 %`, and it is a POINT ESTIMATE rather than a band. Even at `n_A = 4`, the crudest order tried,
it lands at `-0.43 %`, still finer than the peak-picker's bin resolution.

**What this deletes, if it holds up.** Four heuristics at once, not by re-tuning any of them:

* `over_floor_db > 10` -- there is no threshold in a parametric fit
* the half-power `zeta` estimate, and with it Wu's bias correction
* "estimator scatter IS the band width" -- replaced by a proper stabilisation diagram
* **the random draw itself.** `A_aa` would be SET to the estimate, not sampled from an annulus, so
  Lemma 3.2's role disappears from our method and with it the seed-to-seed variation that step 0
  exists to worry about. The `AUG_LRU_B` scale's `20 %` draw scatter goes the same way, since a
  BLA supplies `B_r` as an estimate too.

The LRU *exponential parameterisation* stays regardless: it is what keeps the pole stable and
well-conditioned DURING TRAINING, which is a separate job from initialisation, and it is the one
argument that is stronger on Telica than in simulation.

**Three caveats, and the first is serious.**

1. **ARX is consistent here only because the data is NOISELESS.** With measurement noise, ARX
   biases the poles unless the noise enters through `1/A(q)`, which it does not. The published fix
   is ARMAX / OE / IV or a subspace method. **So this result does not yet transfer to the noise
   case, let alone to Telica**, and the noise re-run is the decisive next test rather than a
   formality.
2. **Closed-loop bias.** `r` is the closed-loop residual and `u` is the recorded input, so on real
   data the estimate needs a closed-loop-aware method (Forssell & Ljung), already flagged in
   problem-log section 15.5.
3. **The min/max across record-channels is WIDE** (`[134.4, 170.9]` Hz at `n_A = 28`) even though
   the median is exact, because poorly-excited record-channels still produce a qualifying pole. So
   the aggregation rule matters: **median, not `[min, max]`**. This is where Au's uncertainty law
   (MSSP 48, 2014) belongs -- weight each record-channel by its posterior variance instead of
   taking extremes. The current recipe's `[min, max]` rule is the wrong statistic even for the
   peak-picker.

#### Under noise: ARX biases the DAMPING, not the frequency, and IV fixes it

`CL_RS_NOISE_CONSISTENT=1` (the physical case, `u -= C_fb(v)`) at 1x, 10x and 100x the
Telica-derived sigma, ARX against a basic instrumental-variable estimator whose instruments come
from a noise-free simulation of the stage-1 model driven by `u` alone.
`scratchpad/residual_bla_noise.py`, `residual_bla_noise.json`.

| sigma | ARX `f` error | **IV `f` error** | ARX `zeta` error | **IV `zeta` error** |
|-|-|-|-|-|
| clean | `+0.001 %` | `+0.001 %` | `-0.4 %` | `+0.1 %` |
| **1x (Telica level)** | `+0.000 %` | `-0.005 %` | `+0.5 %` | `-0.0 %` |
| 10x | `-0.004 %` | `-0.024 %` | **`+4.8 %`** | **`+0.4 %`** |
| 100x | `-0.008 %` | `-0.079 %` | **`+8.8 %`** | **`+1.9 %`** |

**Finding 1: frequency is essentially immune.** Within `0.08 %` at 100x noise, against the
peak-picker's `+0.20 %` on CLEAN data. The mode's location survives two decades of noise.

**Finding 2: the predicted ARX bias is real, it is in the DAMPING, and it is one-sided.** ARX
`zeta` inflates monotonically with sigma (`0.05257 -> 0.05301 -> 0.05529 -> 0.05739`), i.e. noise
smears the resonance and ARX reads the broader peak as heavier damping. Exactly the failure mode
the theory predicts for output-additive noise entering a `1/A(q)` equation-error model.

**Finding 3: IV removes it, and cheaply.** At 10x, IV's `zeta` error is `+0.4 %` against ARX's
`+4.8 %`, a **12x** improvement; at 100x, `+1.9 %` against `+8.8 %`. `rho = exp(-zeta*wn*Ts)` is
what actually initialises `A_aa`, so a `9 %` damping error is a real init error and IV is not
optional under noise.

**Finding 4, non-oracle.** The "best `na`" above is selected by closeness to a truth we would not
have on Telica, so it is optimistic. The honest statistic is the SPREAD across the converged orders
`na in [20, 32]`, which needs no truth:

| sigma | ARX spread | IV spread |
|-|-|-|
| 1x | 0.02 Hz | 0.05 Hz |
| 10x | 0.37 Hz | **0.16 Hz** |
| 100x | 0.35 Hz | **0.18 Hz** |

IV is about **2x more stable across model order** as well as less biased, so the stabilisation
diagram itself is cleaner and order selection is easier without ground truth. ARX's per-order
sequence at 100x is erratic (`160.81, 158.20, 157.53, 157.76, 158.11, ...`) while IV's is monotone.

**Verdict: the clean-data result is NOT a simulation artefact.** It survives the noise level that
matters, and the one place it degrades has a standard, cheap, citable fix. **Use IV, not ARX.**
Recorded honestly: this is still simulation, and the closed-loop bias of caveat 2 is untested,
since noiseless-plus-noise is not the same as a genuinely closed-loop-identified BLA on real logs.

**Status: not an arm, and not applied.** Wave 1 is unaffected and should be submitted as is. This
is the follow-on that F4a/F4b/F4c feed into: those three ask whether our band recipe beats
Orvieto's published default, and this is the third option neither of them contains, with a
control-venue citation behind it (Schoukens & Toth, IFAC-PapersOnLine 53(2):310-315, 2020,
arXiv:2004.05040, BLA initialisation for **LFR models specifically**; PDF not held, fetch before
designing anything).

### BLA deep-research sweep, 2026-08-22. SECOND-HAND: nothing here was opened by this session.

Persisted because the handoff cannot carry it and the report otherwise exists only in a conversation.
**Every claim below is a subagent's, not a verified reading.** Verification is deliverable (i) of
`tasks/handoffs/2026-08-22-bla-initialisation-of-the-augmented-states.md`.

**What the BLA-init literature actually does with the added block.**
`schoukens2020lfr` Sect. 4.2 eq. (7): `A, B_u, C_y, D_yu` from the BLA; **`B_w = 0`, `D_yw = 0`**;
eq. (8) `C_z, D_zu` **random**, scaled to unit-std `z`; `D_zw` does not exist by construction,
"to prevent the presence of algebraic equations ... avoiding well-posedness problems"; everything
jointly re-estimated by LM on simulation error. `schoukens2021ssnn_init` Table II, gR-SS-NN column:
`W_x = W_y = b_x = b_y = 0` with `A,B,C,D = A_LTI,...` and hidden weights `~ U(-1,1)/sqrt(n)`.
**Pattern: the path OUT of the added block is zeroed, the path INTO it is kept alive and random.**

**Why a zeroed readout is not a dead zone there.** `schoukens2021ssnn_init` Sect. IV-B.2:
*"random weights in the nonlinear layer and zero weights in the linear layers ... using random
weights and biases in the nonlinear layer generates a pool of nonlinearly transformed outputs which
the estimator can pick from using the linear weights during optimization."* Neither paper ever names
the optimisation consequence; both justify the zeroing only by forward properties (initial model
equals the BLA, stability inherited). The named, analysed version is ML-side: ReZero,
arXiv:2003.04887, which this project already implements as `ANN_REZERO_GATE`.

**Novelty.** Initialising added dynamic states from a linear model of an FP residual: **NOVEL,
grade MODERATE**. `hoekstra2025lfr` Sect. 3.5 initialises per `schoukens2020lfr`, i.e. the BLA of
the FULL system; `hoekstra2026encoderinit`, whose whole subject is initialisation in the augmentation
setting, contains "residual" **zero times**. arXiv `abs:"model error model" AND "identification"` = 0,
`AND "neural"` = 0. Grade capped because Google Scholar returned `[]` on 15/15 queries across five
agents (shared-IP rate limit), losing the only full-text route. Nearest relative: Floren, Mamedov,
Noel, Swevers, IFAC-PapersOnLine 58(15):468-473, 2024, DOI `10.1016/j.ifacol.2024.08.573` - physics
baseline + residual ANN, but `n_x` fixed by the baseline, residual **static**, nothing initialised
from a linear residual model.

**Nobody subtracts a known part before partitioning a BLA.** `sjoberg2012whpartition` Algorithm 1
enumerates ALL pole/zero splits and ranks by `V_N`; prior knowledge enters only as cardinality
constraints, never attribution. Full 63-work forward-citation cone read: zero grey-box instances.
`schoukenstiels2017survey` Sect. 11 names "use prior pole/zero knowledge" as OPEN, pointing only at
soft regularisation toward prior pole/zero locations (Risuleo 2015; Tiels & Schoukens 2014).

**Closed-loop.** `pintelon2020bla_feedback` p6: the estimator "is the solution of the Wiener-Hopf
equation ... This property is inherited by (11) because it is the ratio of two best linear
approximations", i.e. an orthogonal **projection**, hence linear in its argument - which is what
licenses substituting a residual for the numerator. **Case A, admissible**: baseline simulated OPEN
loop on the recorded input, `e = y - G_base u`, gives `E{E R*}/E{U R*} = G_BLA - G_base` exactly,
sensitivity cancelling. **Case B, the trap**: baseline re-simulated INSIDE the loop from the
reference gives `G_BLA - T_base/G_RU`, contaminated by an inverse closed-loop BLA. Eq. (58) p12:
`var(G_BLA) ~ (1/|S|^2) sigma_P^2/|R|^2`, so **sensitivity suppression costs VARIANCE, not bias**.
`schoukens2016bla` pp. 37-38 describes forming `Y - G U` as *"an 'opening' of the closed loop"*.
Ljung's MEM is strictly open loop ("closed loop" appears once, never in the construction);
`oomen2008modelvalidation` p4 gives the literature's correction: *"additive structures for open-loop
systems and dual-Youla structures for closed-loop systems"*.

**Dual-Youla equivalence, the sweep's own algebra, sympy-verified, in NO paper.**
Target: `beta = D0 y - N0 u = D0 * rho` where `rho = y - P0 u`, exact. Regressor: theirs is
`alpha = D_K u + N_K y = D_K r`, ours is `u` - **not the same**, and Sugie p2 stresses `alpha` is
uncorrelated with the noise while our `u = S_u r - S_u K w` is not. With `L = 0` the factorisation
degenerates to `D0 = I, N0 = P0` and `beta = rho`, valid only if `P0` is stable; our baseline has
double integrators so `A0` contains `(1-q^-1)^2`, i.e. **the drift in `y - P0 u` is the missing `D0`
filter, not a modelling artefact**. `Q = H_K (A0 B - A B0)/(H (A S + B R))`: the denominator is the
TRUE closed-loop characteristic polynomial, so **the missing dynamics are `Q`'s ZEROS, not its
poles** - take initialisation poles from `Delta_hat = P_hat - P0`. The stabilised-by-`K` certificate
does NOT transfer to additive coordinates. Nobody uses a dual-Youla `Q` as an initialiser for a
learned model (arXiv `abs:"dual Youla" AND abs:"neural"` = 1, the held Boroujeni paper, which uses
it as a training parameterisation).

**Metadata-only, do NOT quote**: Vanbeylen 2012/2013, Besancon-Voda 2000 ("Model Error Model from
Identification in Closed-Loop", IFAC 33(15):163-168, DOI `10.1016/S1474-6670(17)39744-6`, the
highest-value unread item for closed-loop MEM), Douma & Van den Hof 2005, Oomen ACC 2013, Niemann
2003, Wu 2014, Brincker 2001, Au 2014, ReZero.

**PDFs added by the sweep**: `literature/BLA/{Schoukens_Tiels_2017_BlockOriented_Survey,
sjoberg2012_WH-init-BLA-partitioning_Automatica48,
schoukens2021_improved-init-state-space-ANN_ECC_arXiv2103.14516,
pintelon2020_bla-feedback-process-noise_arXiv2004.02579,
schoukens2016_linear-sysid-nonlinear-setting_arXiv1804.09587}.pdf`,
`literature/closed-loop-id/{oomen2008_disturbances-model-uncertainty-model-validation_CDC,
geerardyn2014_hinf-norm-local-rational_TUePure}.pdf`.

### Procedure (short or long form, filled after F1)

(pending F1; the band-source half above is complete and is independent of it)
