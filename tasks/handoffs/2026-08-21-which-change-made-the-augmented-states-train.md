# Handoff: which changes made the augmented states train, and is each one defensible under noise

**From**: session of 2026-08-21 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

An overnight run reached `3.795974e-07 m` free-run validation RMS with the augmented states
measurably load-bearing (ablation `5.2081x`), against a long-standing plateau of `1.3933793e-06`.
Six mechanisms were on at once, on NOISELESS simulation data, at a mode frequency that does not
exist in the real machine's identifiable band. **Determine which mechanisms are necessary, which
survive measurement noise, and what the resulting method does on Telica, where no residual peak
exists.** The deliverable is a table with one row per mechanism reading necessary / not necessary /
untested, each with its free-run RMS, its ablation ratio, its provenance as a verified citation or a
labelled heuristic, and its behaviour under noise, plus a written real-data procedure stating what
the method does when there is no identifiable peak. A mechanism earns a place in a clean
`model_augmentation/` implementation only if it is necessary AND its justification survives noise.

**The attribution question and the real-data question are the same question**, which is why they are
one task. If F1 (section 9) shows that eight plain ANN-written latent rows reach e-7 without
`AUG_LRU` at all, then no band recipe is needed and the two hardest Telica blockers disappear with
it. If `AUG_LRU` is necessary, they bite. Run F1 first for that reason.

## 2. Out of scope

* **Do not write the clean `model_augmentation/` implementation.** This task decides its contents.
  Writing it before the ablation is what this task exists to prevent.
* **Do not modify** `model_augmentation/`, `kamtin-fp-model/`, or
  `gantry_dynamic/{config,evaluation,orth_penalty,rezero_gate}.py`. The last four carry another
  session's P1/P1-e work. `gantry_dynamic/model.py` and `closed-loop-controller/*.py` are editable.
* **Do not commit, push, or stage** `model_augmentation/` or `gantry_dynamic/`. See section 13 for
  the snapshot mechanism the user wants instead.
* **Do not extend the orthogonal-projection penalty.** A verified finding (section 4) says it is
  blind to the dynamic augmentation. That is real and important and it is the NEXT task, not this
  one.
* **Do not chase the `1.215e-06` target further.** It is already beaten. The question is now
  attribution, not magnitude.
* **Do not re-run** the multiple-shooting defect, burn-in as a fix, or a gradient-coherence sweep.

## 3. Where things stand

Branch `Augmentation`, last commit `4cdb7c1`, tree dirty. Nothing in flight.

Modified: `model_augmentation/fit_systems/{closed_loop,pre_encoder}.py`,
`gantry_dynamic/{config,evaluation,model,orth_penalty,rezero_gate}.py`,
`closed-loop-controller/cl_train.py`. Untracked: `closed-loop-controller/transient-investigation/`
(all this track's), `gantry_dynamic/{bounded_integral_block,lipschitz,passive_ph_block,patches}`.

Full record of the overnight run: `tasks/overnight-2026-08-21-verdicts.md`. Run rows in
`docs/gantry-augmentation-problem-log.md` section 12. The snapshot of the frozen-path changes has
already been created and pushed (section 13); `model_augmentation/` and `gantry_dynamic/` remain
dirty and unstaged by design.

## 4. Established and verified

Numbers are free-run validation RMS on V1-V4, `closed_loop_free_run_rms`, 4 records x 12.000 s,
scored from `k0 = 17`. **It is a closed-loop rollout** (`u_data + C_fb(y_data - y_model)`), not an
open-loop simulation; every number below uses it, so they are mutually comparable.

| fact | value | evidence |
|-|-|-|
| untrained / D-072 gate | `2.1866011034177349e-06` | every probe reproduces it |
| previous plateau | `1.3933793e-06` | run table section 12 |
| **arm 1**: `AUG_LRU=1 AUG_LRU_B=0.377 ENC_WA_ZERO=1`, `nx_aug=2`, 520 updates | `1.379891e-06`, ablation **`1.0183x`** = decoration | `runs/arm_ablation_arm1_520upd.json` |
| **arm 2**: same but `nx_aug=8` with `AUG_LRU_NA_NB=17` | **`3.795974e-07`**, ablation **`5.2081x`** | `runs/arm_ablation_arm2_nx8_520upd.json` |
| **width-matched control**: `nx_aug=2`, `CL_NODES=20` (828 ANN params vs arm 2's 798) | `1.384274e-06` | `runs/cl_train_ctrl_width20_nx2.json` |
| planted (oracle) reference | `4.176627e-07`, ablation `6.010x` | `runs/representation_ceiling.json` |
| in-band split, arm 2 | intact in-band `2.858956e-07`; blinded `1.784171e-06` = **`6.241x`** in band against `3.409x` out of band | `runs/arm_ablation_arm2_nx8_band.json` |
| in-band split, arm 1 | `1.017x` in band against `1.020x` out of band, preference `0.997` | `runs/arm_ablation_arm1_nx2_band.json` |
| readout Jacobian `d w[0:6]/d x_a` | untrained **exactly `0.0`**; arm 1 `1.046098e-03`; arm 2 `1.083590e-03` | `runs/readout_jacobian_*.json` |
| trained poles move under `0.15 Hz` in both arms | arm 1 `154.543 Hz`; arm 2 `159.350 / 162.854 / 151.995 / 153.475 Hz` | same |
| **D-072 holds bit-identically ONLY at `na_nb = 17`** | `0.000e+00` at 17 and at (17, 8); `1.336e-04` at 32, `2.775e-04` at 64, `4.028e-04` at 103 | `runs/d072_matrix_probe.json` |
| true absorber mode, from the plant | `r = 0.986982` at `157.8937 Hz` | `probe_objective_sign.py`, prints both pole tables |
| the band recipe finds it **without oracle input** | recipe gives `[149.90, 164.06] Hz`, fit `rho = 0.98560` at `158.203 Hz`, i.e. **within one FFT bin** of truth | `lru_band_from_artifact` vs the above |
| the band is **bin-quantised at `0.48828 Hz`** | `fs/nperseg = 4000/8192`; `149.90234`, `158.203125`, `164.0625` are bins 307, 324, 336 exactly. The recipe cannot resolve finer, so "within one bin" is the honest accuracy claim, not sub-Hz | arithmetic on `cl_residual_spectrum.json` |
| the band is **unchanged under the recorded noise**, with the margin quantified | clean and noisy both give 54 dominant peaks and the identical band and `rho`; `over_floor_db` median falls `76.4 -> 52.2`, min `65.4 -> 48.8`, i.e. a **24 dB margin loss against a 10 dB threshold**, leaving `38.8 dB` of headroom | `cl_residual_spectrum.json` vs `cl_residual_spectrum_noisy.json` |
| objective damps a CORRECT mode | `dL/d(nu_log) < 0` on 7 of 8; `nu_log` monotone up, 100 % of steps | `runs/objective_sign_probe.json` |
| **orthogonality penalty is blind to the dynamic augmentation** | `dV_orth/dp` **exactly `0.0`** for `W^a`, `B_a`, `nu_log`, `theta_log`, ANN `x_a` input columns; controls non-zero (`1.393`, `27.66`) | `runs/orth_gauge_probe.json` |
| **the learned mode is OUT OF BAND on the real machine** | Telica identifiable band is **below 83 Hz on X and 55 Hz on Y** (`telica_plant_frf.py`); the mode learned here is at `157.89 Hz`. The specific result cannot transfer, only the METHOD can | `docs/augmentation-training-status.md` section 2 P7 |
| the `9.13e-07` "ceiling" is not a floor | a static correction removes 97.4 % of that out-of-band power; encoder startup transient `9.946e-09`, baseline model error `8.146e-09` | `runs/out_of_band_probe.json` |

**Arm 1 and arm 2 differ in exactly one factor**: `nx_aug`. The lag is pinned at 17 in both and ANN
width is controlled by the width-matched control, which lands on arm 1 (`1.384274e-06`) despite
carrying the most parameters of the three.

## 5. Assumed but not verified

* **That `AUG_LRU_B` (D-151, the `B_a` input path) is necessary at `nx_aug = 8`.** Never run off at
  `nx_aug = 8`. Settled by factor run F2 below.
* **That `ENC_WA_ZERO=1` is necessary.** C3 measured it worth `1.585x` on the WINDOW metric but only
  `1.013x` on the free run, on the planted model. Its effect at `nx_aug = 8` is unmeasured. F3.
* **That the band draw is necessary, and WHICH half of it.** C7 scored band vs full-circle draws by
  least-squares fit quality at init (median `1-R^2` `0.988` vs `0.999`, no overlap) but never trained
  one, and never separated the frequency band from the damping band. F4a and F4b.
* **That `AUG_LRU` itself (the live `A_aa`) is necessary at `nx_aug = 8`**, as opposed to eight
  ANN-written latent rows with no pole. F1.
* **Why arm 2 works.** The readout Jacobian is the SAME in both arms (3.6 % apart), so the gap is
  not explained by how strongly the ANN reads `x_a`. Working hypothesis, **not measured**: arm 1's
  single pole is `3.35 Hz` off the mode, about 40 cycles of phase drift over 12 s, so its injection
  averages away; four poles spanning `151.995` to `162.854 Hz` can be combined into something
  phase-coherent. The frozen-pole run (F5) tests the fixed-basis half of this directly.
* **That any of it survives noise.** `CL_NOISE_CONSISTENT=1` is implemented in `cl_train.py` and has
  never been run. Section 9 steps 2b and 3.
* ~~That 520 updates is enough to rank the factors.~~ **Now justified, keep 520.** Both arms are
  within 1 % of their 260-update value at 520 (arm 2 `3.822e-07 -> 3.795974e-07`, 0.7 %; arm 1
  `1.383192e-06 -> 1.379891e-06`, 0.2 %), so the RANKING is not update-limited even though both were
  truncated by the host. What 520 does not establish is the converged floor of any arm.
* **That `3.795974e-07` reproduces at all.** It is **n = 1**: one seed, one draw. The pole draw is
  stochastic (`manual_seed(cfg.seed + 150)` over the data-derived band), and seed 0 happened to give
  four poles bracketing the mode. Seeds 1 and 2 settle both reproducibility and draw-luck at once,
  and if it does not reproduce the whole factorial is measuring noise. Section 9 step 0.

### The trap in a noiseless factorial, and it would discard the mechanisms that matter most

The six mechanisms do NOT share a valid test condition. Splitting them is the single most important
structural point in this handoff:

| class | mechanisms | correct test |
|-|-|-|
| **A, representational** | `AUG_LRU`, `AUG_LRU_B`, `nx_aug`, the band draw | noiseless, because their claim is about representation and optimisation |
| **B, noise-motivated** | `ENC_WA_ZERO`, `closed_loop_rollout` (stabilized PEM), `CL_NOISE_CONSISTENT`, T2's lag rule | **under noise only. Noiseless data cannot show their value and will mark them unnecessary** |

Concretely: `ENC_WA_ZERO` is justified by the encoder amplifying input noise **1919.8x**
(`encoder_conditioning.json`), yet measures only `1.013x` on the noiseless free run (gate C3). A
noiseless F3 would mark it NOT necessary and delete the mechanism most likely to matter on hardware.
The same holds for `closed_loop_rollout`, whose entire justification is that the physical residual
stays pinned at the float32 floor across four decades of sigma while a non-cancelling estimator
scales linearly (`alpha_cancellation.json`); noiseless data shows none of that.

**So Class A is tested noiseless and Class B is tested under noise, and a Class B mechanism is never
judged on a noiseless run.**

## 6. Tried and failed

* **`CL_PROBE=0` for speed** -> validation froze at `2.186601103417735e-06`, bit-identical to
  untrained after 260 updates -> `cl_train.py:130` reads
  `CONCURRENT = bool(int(os.environ.get('CL_CONCURRENT', 0 if PROBE else 1)))`, so disabling probes
  ENABLES concurrent subprocess validation, and that child scores a stale model -> the serial path at
  the same 260 updates gave `1.383192e-06`. **Always set `CL_CONCURRENT=0`.** D-146 claims this path
  was verified; it is not.
* **Editing `get_encoder_dims` without running a build** -> `NameError: name 'ic' is not defined` ->
  the edit deleted `ic = Interconnect(...)`; `py_compile` passes because it is a runtime error.
  **A compile check is not a build check.**
* **`na_nb` above 17** -> D-072 fails, monotonically in `n` -> float32 conditioning of
  `W^b = A^n O_n^{-1}` -> `runs/d072_matrix_probe.json`. Cancels T2's derived `na_nb = 103`.
* **A weighting to fix the objective** -> derived, then proved impossible: the batch-consistent
  damping term is positive under EVERY non-negative weighting, and a narrowband prefilter multiplies
  the whole loss change by `|L(theta)|^2 > 0` -> section "T3" of the overview.
* Four earlier mechanisms, each verified to work and none of which moved the number: D-150 live
  `A_aa` alone (`-0.665 %`), burn-in `K=100`, multiple-shooting defect (degenerate), `na_nb` coherence
  sweep (probe artefact). Do not revisit.

## 7. Achieved

Implemented and validated: `AUG_LRU_NA_NB` in `model.py:get_encoder_dims`, env-gated, OFF by
default, `cfg.na_nb_override` still wins, D-072 line with the gate unset
`17 2 2.186601103417735e-06 rel dev 0.000e+00 PASS` (`runs/d072_noop_check.json`). `CL_NX_AUG` and
`CL_NODES` in `cl_train.py`, both printing an ANN-parameter-count line. `CL_NOISE_CONSISTENT` in
`cl_train.py`: implemented, **not run**.

Nine probes in `transient-investigation/`, each writing a JSON artefact:
`probe_d072_matrix`, `probe_representation_ceiling`, `probe_wa_freerun`, `probe_objective_sign`,
`probe_encoder_isolation`, `probe_out_of_band`, `probe_orth_gauge`, `probe_arm_ablation`,
`probe_readout_jacobian`.

## 8. The open question

**Which mechanisms are load-bearing, and does each one's justification survive noise and transfer
to a machine whose residual has no identifiable peak?** `nx_aug` 2 to 8 is established as necessary
and sufficient to move `1.38e-06` to `3.80e-07` GIVEN the other five on. Unknown is whether any of
the other five is necessary, and the answer is not a flat sweep: `AUG_LRU_B`, the frequency band, the
damping band and the frozen poles are all sub-features of `AUG_LRU`, so **F1 decides whether the
other four questions exist at all** (section 9).

The evidence that chooses, per factor: run it off at `nx_aug = 8`, read the free-run RMS and the
ablation ratio against the section 10 boundaries. A factor whose removal leaves both unchanged is not
earned and should not enter `model_augmentation/`.

Two riders, neither blocking. First, the readout Jacobian is equal in both arms, so the mechanism
behind arm 2 is not "the ANN reads `x_a` harder"; F5 tests the fixed-basis reading. Second, the two
`model_augmentation/` files that will actually ship (`closed_loop.py`, `pre_encoder.py`) are tested
only indirectly, by F3c and by step 2b; step 3b says why that is the honest limit rather than an
oversight.

## 9. Next action

**The factors are NESTED, not parallel.** `AUG_LRU_B`, the band draw and the frozen poles are all
sub-features of `AUG_LRU`: if there is no live `A_aa` there is no input path to it, no band to draw
from and nothing to freeze. So this is a decision tree with F1 at the root, and running it as a flat
five-row sweep would waste three runs answering questions F1 has already closed.

**Budget.** Each training run is ~50 min at 520 updates, each ablation ~25 min.
**Best case 7 runs, about 9 h** (F1 passes: seeds 2, F1 1, noise baseline 1, F3 3).
**Worst case 11 runs, about 14 h** (F1 fails: + F2, F4a, F4b, F5).
Plan against the worst case and against a host that kills jobs every 40-60 min. If the budget binds,
the documented drop order is F3c first, then F5, then F4b. Write the run-table row before each
launch, per the run-discipline rule.

**Write everything to `tasks/ablation-2026-08-22-what-earned-its-place.md`, appended after each run,
never composed at the end**, so it is correct if the session dies mid-way. That file holds the
deliverable table (one row per mechanism: necessary / not necessary / moot / untested, with free-run
RMS, ablation ratio, provenance, noise behaviour), the DEVIATIONS section required by section 13, and
the written real-data procedure. The overnight run worked largely because its equivalent file existed
from the first minute; do not keep results in the conversation.

### Step 0, 2 runs. Does the result exist?

Replicate arm 2 at `cfg.seed` 1 and 2, unchanged otherwise. Ablation on at least one of them.
`3.795974e-07` is n = 1 with a stochastic pole draw. Three outcomes, all defined:

* **Both reproduce below `6.0e-07`** -> the result is robust to the draw. Continue to step 1.
* **Neither reproduces** -> **STOP and report that.** The tree below would be measuring noise and
  nothing else in this handoff is worth running.
* **Exactly one reproduces** -> **do not average them and do not continue as if it were fine.** The
  method is DRAW-DEPENDENT, and that is itself the headline finding: it means the band recipe
  delivers a working basis only sometimes, which matters far more for real data than any factor
  attribution. Record both pole sets (`scratchpad/poles.py` pattern), then make the reliability of
  the draw the primary question: run 3 more seeds to estimate the success rate, and treat F1 as
  secondary. A method that works on 1 draw in 2 is not shippable regardless of which mechanism
  carries it.

### Step 1, 1 run. F1, the root

Drop `AUG_LRU` entirely (so no `AUG_LRU_B`, no band, no poles): eight plain ANN-written latent rows
at `nx_aug = 8`, `AUG_LRU_NA_NB=17`. Ablation as usual, **but WITHOUT `PROBE_PERPAIR=1`**, which is
meaningless here since there is no `A_aa` and therefore no complex-conjugate pair structure.

* **If F1 PASSES** (free run `<= 6.0e-07` AND ablation `>= 2.0x`): `AUG_LRU` is not needed.
  **Do not run F2, F4a, F4b or F5 - they are moot.** Skip step 2 AND step 5 entirely: with no band recipe,
  `cl_residual_spectrum.json` is never read, so all four band-source problems and Telica blockers 2
  and 3 vanish with it. Go to step 3. **This is the best available outcome**: it is both the simplest
  clean implementation and the most defensible on real data, because every real-data blocker we found
  attaches to `AUG_LRU`, not to `nx_aug`.
* **If F1 FAILS**: `AUG_LRU` is necessary. Run step 2, and step 5 becomes mandatory.

### Step 2, 0 or 4 runs, only if F1 failed

| id | change from arm 2 | tests |
|-|-|-|
| F2 | drop `AUG_LRU_B` | is the D-151 input path needed |
| F4a | `AUG_LRU_BAND="1,2000"`, `AUG_LRU_RHO` left at the artefact value | is the data-derived FREQUENCY band needed |
| F4b | `AUG_LRU_RHO="0.05,0.99"`, `AUG_LRU_BAND` left at the artefact value | is the data-derived DAMPING band needed |
| F5 | freeze `nu_log`, `theta_log` | are the poles a fixed basis, as the pole tables imply |

`AUG_LRU_BAND` and `AUG_LRU_RHO` are two separate outputs of the same peak fit: the frequency from
the peak location, `rho = exp(-zeta*wn*Ts)` from the fitted damping. Replacing both at once, as a
single F4 would, cannot say which one carried it, and T4's coverage analysis was entirely about
FREQUENCY and said nothing about `rho`. Note `model.py` asserts the two env vars must be set
TOGETHER, so each of F4a and F4b passes both, one at the artefact value and one wide.

**F5 needs code that does not exist yet.** There is no gate to freeze the LRU parameters; they are
created in `AugLRUBypass.__init__` (`model.py`). Add `AUG_LRU_FREEZE=1` setting
`requires_grad_(False)` on both tensors right after construction, env-gated and OFF by default, and
verify the D-072 line with it unset before using it. That is a small piece of named work, not a
config change.

### Step 2b, 1 run. The NOISE BASELINE, and it must precede step 3

Re-run the winner of steps 0-2 (arm 2's configuration if F1 failed, F1's if it passed) with
`CL_NOISE_CONSISTENT=1 CL_NOISE_SIGMA="8.544e-9,7.762e-9,6.539e-9"`, unchanged otherwise, plus its
ablation.

**This run has two jobs and both are load-bearing.**

1. **It is the reference against which step 3 is scored.** Every F3 arm runs under noise, so scoring
   them against `6.0e-07` -- a boundary derived from two NOISELESS runs -- would compare different
   quantities, and all three arms could land above it purely because noise was added. Section 10
   therefore scores Class B against THIS number, not against `6.0e-07`.
2. **It is the first training-level test of `closed_loop_rollout`** (step 3b). If the ablation ratio
   survives noise, that is evidence for the stabilized-PEM rollout, which until now has only ever
   been tested by the `alpha_cancellation` probe and never by a training run.

### Step 3, 2 or 3 runs. F3, and it must be judged under noise

`ENC_WA_ZERO` is Class B (section 5) and is independent of the F1 branch, since `W^a` initialises the
latents whether or not they have a pole. Judging it on a noiseless run would delete the mechanism
most likely to matter on hardware, so every arm here runs with
`CL_NOISE_CONSISTENT=1 CL_NOISE_SIGMA="8.544e-9,7.762e-9,6.539e-9"`.

**There are THREE options, not two.** `ENC_WA_ZERO` changes the initial VALUE and leaves `W^a`
trainable; whether it should be a free parameter at all is untested, and T1's derivation says only
that the optimal initial value is zero, nothing about trainability:

| arm | `W^a` init | trainable | note |
|-|-|-|-|
| F3a | random (kaiming) | yes | Hoekstra's stated convention, arXiv:2602.17297 p.9 Eq. (31) |
| F3b | zero | yes | `ENC_WA_ZERO=1`, what arm 2 ran |
| F3c | zero | **no** | needs `ENC_WA_FREEZE=1`, see below |

**F3c needs code that does not exist**: after the `ENC_WA_ZERO` block in `model.py`, add
`ENC_WA_FREEZE=1` calling `requires_grad_(False)` on `Wa_psi_y` and `Wa_psi_u`, env-gated, OFF by
default, D-072 line verified with it unset. If run time is short, F3c is the one to drop: it is the
least likely of the three to differ, since T1 measured `W^a`'s whole contribution at `1.013x` on the
free run.

### Step 3b, 0 runs. What CANNOT be ablated, and why that is honest rather than an oversight

`closed_loop.py` (+104, the stabilized-PEM rollout) and `pre_encoder.py` (+47, the `W^b`/`W^a`
split) are **the two files that must go into a clean `model_augmentation/` implementation**, and no
run above evaluates either. Do not let them ride in unexamined.

* **`pre_encoder.py`'s `W^a` block IS tested**, by F3 above: F3c with `W^a` frozen at zero is
  behaviourally the "no `W^a` block" case. Nothing further needed.
* **`closed_loop_rollout` cannot be ablated as a factor** and should not be. Turning it off means
  open-loop training, a different estimator settled by D-142/D-147, not a variant of this one. But
  its CLAIMED benefit is noise robustness, and that has never been tested in a TRAINING run, only in
  the `alpha_cancellation` probe (residual pinned at the float32 floor whole-record, degrading to a
  saturating `2.0x` at `nf = 400`). **Step 2b is that test**, and it costs no extra run because it is
  needed as step 3's baseline anyway. Report it as evidence for the rollout, not merely as a property
  of the winning arm.
* If time allows one extra run, the sharpest single check is the gradient-cosine test T6 proposed:
  one forward and one backward on a fixed batch under `xc = 0` versus `xc` carried, from the same
  checkpoint. Cosine `> 0.99` proves the windowing convention cannot be steering the optimisation
  and closes the `xc = 0` question without a training pair.

### Step 4. Provenance of every surviving mechanism

Record each as `# THEORY: <source, section/equation>` or `# HEURISTIC: <reason>`, checked at the PDF.
Known-good: Orvieto ICML 2023 Lemma 3.2 p7 (radius draw), Sec. 3.3 p8 (exponential parameterisation),
Eq. (7) Sec. 3.4 p10 (`gamma`); Forgione, Mejari, Piga, IEEE CDC 2024 pp. 8620-8625 Remark 1 (the 2x2
real Jordan form). Known heuristics: the `AUG_LRU_B = 0.377` scale, the `over_floor_db > 10` peak
threshold, and the phase-arc restriction, which is **Sec. 3.4 empirical, NOT Lemma 3.2**.

**A citation error is currently IN THE CODE and must be fixed before anything ships.**
`model_augmentation/fit_systems/pre_encoder.py:422` reads

```
# HEURISTIC, with no literature source: kaiming_uniform_ on both blocks.
```

That is false. Hoekstra, Gyorok, Verhoek, Toth, Schoukens, arXiv:2602.17297, **p.9 Sec. 5.4.2
Eq. (31)** specifies exactly this: *"the weights and biases of psi_aug are initialised by the Xavier
approach"*, with p.10 giving `m ~ U(-1,1)` for every matrix not fixed by baseline equality. So a
random `W^a` **is** Hoekstra's stated convention. We refute it by argument, not by absence of a
source: his own Eq. (7) (arXiv:2602.13108 p.3) defines the encoder as approximating
`E[x_a | u_hist, y_hist]`, and under D-072 the readout is exactly zero, so `x_a` is independent of
the window and that conditional mean IS zero. `model.py`'s `ENC_WA_ZERO` comment block repeats the
same wrong claim and needs the same correction. Whichever `W^a` arm wins in step 3, the comment must
say "Hoekstra's convention, refuted here by Eq. (7) under D-072", never "no literature source".

### Step 5, only if F1 failed. The band source under noise and on real data

This decides whether the method is a realistic application, and it outranks the factorial if the two
conflict. The whole `AUG_LRU` initialisation hangs off `cl_residual_spectrum.json`, the spectrum of
`y_data - y_baseline_sim`. Four problems, in ascending severity:

1. **The noisy artefact records no noise setting.** `cl_residual_spectrum_noisy.json` has no sigma
   field, so its provenance rests on the doc. Write the sigma into the artefact and re-generate.
2. **Only `y` was perturbed.** On a real machine the baseline sim is driven by MEASURED `u`, so
   `u`-side noise propagates into the residual too. Same asymmetry C5 fixed for the training drive,
   unfixed here. Extend `cl_residual_spectrum.py` and re-measure the band.
3. **No breakdown point is known.** The `38.8 dB` headroom in section 4 is inferred from two points.
   Sweep `CL_RS_NOISE_SIGMA` upward until dominant peaks fail `over_floor_db > 10` and the band moves
   or `lru_band_from_artifact` raises. That is a stated SNR requirement, a thesis deliverable.
4. **The deepest problem, and it is not about noise.** In simulation `deriv6` and `deriv8` differ by
   ONLY the absorber, so the residual is a near-pure absorber signature at 65 to 168 dB over floor.
   On Telica the residual is EVERYTHING the FP model gets wrong. The peak-picker will return the
   dominant peak of that with no guarantee it is a missing MODE. `lru_band_from_artifact` raising on
   Telica is correct behaviour, and **the hand-supplied `AUG_LRU_BAND`/`AUG_LRU_RHO` is where oracle
   knowledge enters unnoticed**. State the Telica band from loop bandwidth and sample rate, in
   writing, BEFORE seeing any result.

### Step 6. Final table

The noise re-run that used to be this step has moved to **step 2b**, because step 3 cannot be scored
without it. What remains here is the deliverable: fill
`tasks/ablation-2026-08-22-what-earned-its-place.md` with one row per mechanism, and write the
real-data procedure from step 5's numbers (or, if F1 passed, the shorter procedure that needs no
band at all).

## 10. Acceptance criterion

A factor is **necessary** if removing it at `nx_aug = 8`, 520 updates, moves the free-run RMS above
`6.0e-07` or drops the ablation ratio below `2.0x`.

`6.0e-07` is the midpoint of `3.795974e-07` (arm 2) and `1.379891e-06`/`1.384274e-06` (arm 1 and the
width-matched control), on a log scale, and is stated here so it is fixed before any run.
`2.0x` is the boundary already used for C8 and is a stated engineering choice, not a derived one;
the measured ratio goes in the table either way so it can be revisited. `# HEURISTIC:` both.

**Step 0 is a gate, not a factor**: if neither seed reproduces below `6.0e-07`, stop; if exactly one
does, the method is draw-dependent and that becomes the primary question (step 0).

**Step 1 (F1) is scored by the same rule, but read the sign carefully.** F1 "failing" the necessity
test means `AUG_LRU` IS necessary and there is more work to do; F1 "passing" it means `AUG_LRU` is
NOT necessary, which is the **best** outcome, because it deletes the band recipe, all four
band-source problems, and Telica blockers 2 and 3 at once.

**Class A factors (steps 1 and 2) are scored against `6.0e-07` and `2.0x`, both derived from
NOISELESS runs.**

**Class B factors (step 3) are scored against the step 2b NOISE BASELINE, never against `6.0e-07`.**
Every F3 arm runs with noise on, so a noiseless boundary would compare different quantities and could
mark a factor necessary purely because noise was added. Concretely: a Class B factor is necessary if
removing it moves the free-run RMS more than 1.5x above the step 2b baseline, or drops the ablation
ratio below `2.0x`. `# HEURISTIC: 1.5x` -- a stated engineering choice, and the measured ratio goes in
the table either way.

**Step 2b itself passes** if the ablation ratio stays above `2.0x` under noise; that is simultaneously
the test of `closed_loop_rollout` (step 3b).

**Step 5 has no pass/fail**: it produces two numbers that do not exist yet, the SNR at which the band
recipe breaks down, and the band the recipe returns when `u` is perturbed as well as `y`. Both belong
in the thesis whichever way they come out.

**Done when** every mechanism has a row reading necessary / not necessary / moot / untested, with its
free-run RMS, ablation ratio, provenance, and noise behaviour, and the real-data procedure is written
down. Not when a number improves.

## 11. Read these first

1. `tasks/overnight-2026-08-21-verdicts.md` - the full record; the header block, "What the headline
   metric actually is", and "THE SYNTHESIS" are the spine.
2. `scripts/gantry/gantry_dynamic/model.py:35-145, 240-360, 460-500` - `AugLRUBypass`,
   `lru_band_from_artifact`, the `AUG_LRU` wiring, `ENC_WA_ZERO`, `AUG_LRU_NA_NB`.
3. `docs/aug-lru-implementation.md` - the env contract and the checkpoint rule (a gated checkpoint
   loads only into a gated build).
4. `scripts/gantry/closed-loop-controller/transient-investigation/probe_arm_ablation.py` - the
   primary criterion, both ablation surfaces, and why the obvious surface is a no-op.
5. `tasks/snapshots/2026-08-21-augmented-states/MANIFEST.md` - the hunk-level ownership split of
   `model.py`, needed before editing it; then `docs/decisions.md` D-072, D-142, D-146, D-150, D-151.

## 12. Do not

* Do not run any arm with `CL_PROBE=0` unless `CL_CONCURRENT=0` is also set (section 6).
* Do not set `na_nb` away from 17; D-072 fails and the arm is uninterpretable.
* Do not compare arms at different update counts. 520 for every run.
* Do not pass `PROBE_PERPAIR=1` to F1's ablation: with `AUG_LRU` off there is no `A_aa` and no pair
  structure, so the per-pair split is meaningless.
* Do not run F2, F4a, F4b or F5 if F1 passes. They are sub-features of `AUG_LRU` and are moot.
* Do not set `AUG_LRU_BAND` without `AUG_LRU_RHO` or vice versa; `model.py` asserts they come in a
  pair. For F4a and F4b, pass both, one wide and one at the artefact value.
* Do not use `rho(A_aa)`, `RMS(x_a)`, gradient coherence, or parameter-movement counts as evidence;
  the ablation is the criterion (`augmentation-training-status.md` section 6.7).
* Do not plant the true mode (`r = 0.986982`, `157.8937 Hz`) into any training arm. It is permitted
  in a diagnostic gate only.
* Do not commit or stage `model_augmentation/` or `gantry_dynamic/`.

## 13. Operational

Env `GraduationProject`. Launch pattern, ~50 min per run at 520 updates, ~25 min per ablation:

```
cd scripts/gantry/closed-loop-controller
AUG_LRU=1 AUG_LRU_B=0.377 ENC_WA_ZERO=1 CL_NX_AUG=8 AUG_LRU_NA_NB=17 \
CL_EPOCHS=2 CL_LR=1e-5 CL_ADAM_EPS=1e-16 CL_STRIDE=10 CL_ITS_PER_VAL=epoch \
CL_PROBE=0 CL_CONCURRENT=0 CL_FLOOR=0 CL_BURNIN=0 CL_CONS_FRAC=0 CL_TAG=<id> \
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
  -n GraduationProject python -u cl_train.py
```

Ablation, with the SAME `CL_NX_AUG`/`AUG_LRU_NA_NB`/`CL_NODES` as the run that made the checkpoint:

```
cd scripts/gantry/closed-loop-controller/transient-investigation
AUG_LRU=1 AUG_LRU_B=0.377 ENC_WA_ZERO=1 CL_NX_AUG=8 AUG_LRU_NA_NB=17 \
PROBE_TAG=<id> PROBE_PERPAIR=1 \
PROBE_CKPT=<newest SSE_Interconnect_MultipleShooting_*_best.pth> \
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
  -n GraduationProject python -u probe_arm_ablation.py
```

Checkpoints land in `C:\Users\20203253\AppData\Local\deepSI\checkpoints\`; take the newest `_best`
after each run. `AUG_LRU=1` needs `runs/cl_residual_spectrum.json` (present).

### Execution discipline, learned from the 2026-08-21 overnight run

These four cost real time or nearly cost correctness last night. They are operational rules, not
suggestions.

* **Measure the wall clock of the first unit, then re-plan the schedule from that measurement and
  record the revised plan. Do not run the timings in this file as given.** Last night's spec assumed
  5-10 min gates; they took 25-50, and the schedule was not re-planned until several hours in.
* **Log every deviation from this spec in a DEVIATIONS section of the run log, at the moment you
  make it**: what you did, what the spec said, and why. **Substituting a pre-registered arm for one
  that turns out not to be runnable IS a deviation.** Last night's single most valuable result
  (arm 2, the e-7) came from exactly such a substitution and was recorded only in conversation, so
  from the artefacts alone it was indistinguishable from scope drift.
* **Any configuration change made for speed is a deviation and must be recorded BEFORE the run is
  launched, not after it returns.** Setting `CL_PROBE=0` for speed silently flipped validation to
  the concurrent path, which froze the reported number at the untrained value for 260 updates.
* **Prefer serial execution; measure before parallelising.** Four concurrent probes ran about 3x
  slower each than the same probes run one at a time. Parallelism felt efficient and was not.

**The host kills background jobs on a ~40-60 min cycle.** Launch one run at a time, shrink the unit
of work to fit the observed window rather than retrying at the same size, and treat a
killed run as truncated: read the last validation from the `.output` and record the update count
rather than discarding it.

**Snapshot: already created, verified and PUSHED. Do not recreate it.**
`tasks/snapshots/2026-08-21-augmented-states/` holds `patches/<file>.patch` (one `git diff` per
changed file, each separately applicable and separately rejectable), `files/` verbatim copies, and
`MANIFEST.md`. All seven patches pass `git apply --check` against `4cdb7c1`.
**Read `MANIFEST.md` before editing `model.py`**: it is the one file carrying BOTH tracks, and the
manifest gives the hunk-level split (patch lines 172-217 and 352-358 are another session's P1 work,
everything else is this track's). Keep `model_augmentation/` and `gantry_dynamic/` unstaged; if you
change `model.py`, regenerate its patch and copy rather than staging the live file.

## 14. Delegation

None. This is five sequential training runs plus citation checks in one context; an Explore
subagent would not help and the compute is serial. If the citation checks in **step 4** need PDFs
not in `literature/`, one `deep-research` subagent per unresolved citation, ceiling two.
