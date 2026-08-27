# What makes the augmented states learn: two experiments, 2026-08-22 and 2026-08-23

Scope: the 16-arm server attribution factorial and the local BLA-initialisation night. Both ran
against the same question and neither has been written up together before. This document exists so
that a session holding neither context can reach its own conclusion.

**How to read this file.** Sections 1 to 4 are **measurement only**: numbers, artefact paths, and
what was configured. Section 5 is **one reading of them, and it is labelled as such**. Section 6
lists what would overturn that reading. If you disagree with section 5, sections 1 to 4 are complete
enough to argue from without re-running anything.

**The two experiments are on different harnesses and their absolute numbers are NOT comparable.**
See section 4. Compare only within-night quantities; the cross-night fractions are also invalid.

## 1. What ran

| | server factorial | BLA night |
|-|-|-|
| when | 2026-08-22, SLURM arrays 77958 / 77959 | 2026-08-23, local |
| what | 16 arms, each removing or changing one component of the working `AUG_LRU` bundle | 3 arms: control, random init, BLA-fitted init |
| updates | 520 (`CL_EPOCHS=2`, `CL_STRIDE=10`), identical every arm | 4 epochs |
| untrained baseline | `2.186580e-06` (varies in the 7th digit by arm) | `2.534187e-06` |
| logs | `scripts/gantry/closed-loop-controller/server-results/closed-loop-controller/wave1_77958_*.out` | `tasks/overnight-2026-08-23-verdicts.md` |
| results | `scripts/gantry/closed-loop-controller/runs/cl_train_w1_*.json` | `scripts/gantry/BLA-Augmentation/runs/*.json` |
| arm definitions | `runners/run_ablation_wave1.sh`, the `case` block from line 107 | `scripts/gantry/BLA-Augmentation/DESIGN.md` D9 |

## 2. Server factorial: measured

Free-run validation RMS on V1-V4, field `final` in each `cl_train_w1_*.json`. Sorted best first.

| task | tag | final RMS | imp % | noisy | min |
|-|-|-|-|-|-|
| 9 | `F5_frozen_poles` | `3.790189e-07` | 82.67 | | 104 |
| 11 | `F3b_arm2_wa_zero` | `3.832755e-07` | 82.47 | **Y** | 97 |
| 12 | `F3c_arm2_wa_frozen` | `3.838424e-07` | 82.45 | **Y** | 94 |
| 4 | `seed5` | `4.200797e-07` | 80.79 | | 72 |
| 1 | `seed2` | `8.056957e-07` | 63.15 | | 72 |
| 10 | `F3a_arm2_wa_random` | `8.088446e-07` | 63.01 | **Y** | 94 |
| 3 | `seed4` | `9.588639e-07` | 56.15 | | 72 |
| 2 | `seed3` | `9.661937e-07` | 55.81 | | 72 |
| 8 | `F4b_wide_rho` | `1.253324e-06` | 42.68 | | 96 |
| 6 | `F2_no_Ba` | `1.339865e-06` | 38.72 | | 72 |
| 7 | `F4a_wide_freq` | `1.390669e-06` | 36.40 | | 71 |
| 13 | `F3a_F1_wa_random` | `1.397310e-06` | 36.10 | **Y** | 51 |
| 14 | `F3b_F1_wa_zero` | `1.397841e-06` | 36.07 | **Y** | 52 |
| 15 | `F3c_F1_wa_frozen` | `1.397867e-06` | 36.07 | **Y** | 53 |
| 5 | `F1_no_auglru` | `1.397913e-06` | 36.07 | | 65 |
| 16 | `F4c_orvieto_default` | `1.399598e-06` | 35.99 | | 96 |

**What each arm changed**, from the runner's `case` block:

* `F1` removes `AUG_LRU` and `AUG_LRU_B` entirely: eight plain ANN-written latent rows.
* `F2` removes `AUG_LRU_B` only.
* `F4a` widens the frequency band to `1-2000 Hz`, keeps the artefact `rho` `[0.9794, 0.9956]`.
* `F4b` keeps the artefact band `[149.90234375, 164.0625] Hz`, widens `rho` to `[0.05, 0.99]`.
* `F4c` widens both: Orvieto Lemma 3.2 exactly as published, full circle, no data input.
* `F5` sets `AUG_LRU_FREEZE=1`.
* `F3a/b/c` vary `W^a` random / zero / frozen, under noise
  (`CL_NOISE_CONSISTENT=1`, sigma `8.544e-9, 7.762e-9, 6.539e-9`), in two configurations:
  tasks 10-12 in the arm-2 configuration, tasks 13-15 in the `F1` configuration.
* Tasks 1-4 are seeds 2-5 of the pole draw, everything else at the arm-2 configuration.

**Historical seeds, for the same configuration**, from the earlier session: seed 0 `3.795974e-07`,
seed 1 `4.8867311476e-07`.

### 2b. The ablation ratios: the PRIMARY criterion. Measured 2026-08-22, added after first writing.

`probe_arm_ablation.py` zeroes the augmented route in the **trained** model and re-runs the free
run. Surface A blinds the ANN to `x_a`; surface B zeroes the ANN output columns that write the
augmented rows. **A ratio near `1.0` means the augmented states are decoration regardless of RMS.**
Reference: on the planted model both surfaces cost `6.010x`.

Read from `server-results/closed-loop-controller/wave2_*.out`. Artefacts on the server at
`transient-investigation/runs/arm_ablation_w1_*.json`.

| tag | intact RMS | ratio A | ratio B |
|-|-|-|-|
| `F5_frozen_poles` | `3.790189e-07` | **`5.2144x`** | `5.2145x` |
| `F3c_arm2_wa_frozen` | `3.837126e-07` | **`5.1551x`** | `5.1551x` |
| `seed5` | `4.200797e-07` | `4.4157x` | `4.4159x` |
| `seed2` | `8.056957e-07` | `2.2664x` | `2.2663x` |
| `seed3` | `9.661937e-07` | `2.1676x` | `2.1676x` |
| `F3a_arm2_wa_random` | `8.088260e-07` | `2.0278x` | `2.0279x` |
| `F4b_wide_rho` | `1.253324e-06` | `1.9381x` | `1.9382x` |
| `seed4` | `9.588639e-07` | `1.6806x` | `1.6806x` |
| `F2_no_Ba` | `1.339865e-06` | `1.0470x` | `1.0470x` |
| `F4c_orvieto_default` | `1.399598e-06` | `1.0255x` | `1.0254x` |
| `F4a_wide_freq` | `1.390669e-06` | `1.0139x` | `1.0139x` |
| `F3c_F1_wa_frozen` | `1.397741e-06` | `1.0001x` | `1.0001x` |
| `F3b_F1_wa_zero` | `1.397862e-06` | `1.0000x` | `1.0001x` |
| `F1_no_auglru` | `1.397913e-06` | `1.0000x` | `1.0000x` |

Surfaces A and B agree to four significant figures everywhere, consistent with `x_a` reaching `y`
only through the ANN.

**Still pending**: task 11 `F3b_arm2_wa_zero` and task 13 `F3a_F1_wa_random` went
`launch failed requeued held` and were resubmitted as `77991`. Their intact RMS are already visible
in the logs, `3.831160e-07` and `1.397292e-06`, matching wave 1.

### 2c. Per-pair ablation and the `W^a` contribution

`F5_frozen_poles`, from `wave2_77959_9.out`:

| blinded | RMS | ratio | in-band ratio |
|-|-|-|-|
| pair 0 only | `1.223454e-06` | `3.2280x` | `3.938x` |
| pair 1 only | `1.030416e-06` | `2.7186x` | `3.100x` |
| pair 2 only | `4.991938e-07` | `1.3171x` | `1.490x` |
| pair 3 only | `5.173478e-07` | `1.3650x` | `1.525x` |
| **`W^a` zeroed (trained value)** | `3.791065e-07` | **`1.0002x`** | |

`||W^a||_F` after training `1.983904e-02`, having started at exactly `0` under `ENC_WA_ZERO=1`.

`F3c_arm2_wa_frozen`, the noisy equivalent, gives the same shape: pairs `3.2747x`, `2.6067x`,
`1.2310x`, `1.4588x`, and `W^a` zeroed `1.0000x` with `||W^a||_F = 0.000000e+00`.

### 2d. In-band versus out-of-band

Band share of free-run error power, and the cost of the ablation split by band:

| tag | in-band ratio | out-of-band ratio | band share |
|-|-|-|-|
| `F5_frozen_poles` | `6.258x` | `3.408x` | `0.8143` |
| `F3c_arm2_wa_frozen` | `5.887x` | `3.614x` | `0.8160` |
| `seed5` | `5.297x` | `2.945x` | `0.8034` |
| `F4b_wide_rho` | `2.162x` | `1.435x` | `0.8075` |
| `F4a_wide_freq` | `1.048x` | `0.963x` | `0.6260` |
| `F4c_orvieto_default` | `1.053x` | `0.981x` | `0.6373` |
| `F2_no_Ba` | `1.064x` | `1.023x` | `0.6079` |

## 3. BLA night: measured

| arm | trained RMS | `F` | source |
|-|-|-|-|
| A0 control | `1.904981e-06` | `0.0007` | `runs/ablation_a0_clean.json` |
| A1 random | `2.015236e-06` | `0.022` | `runs/ablation_a1_clean.json` |
| A2 BLA-fitted | `2.038484e-06` | **`-0.096`** | `runs/ablation_a2_clean.json` |

`F` is the improvement fraction flowing through `x_a` (D-157), which replaced the `2.0x` ablation
threshold. `F = -0.096` means removing the fitted block **improves** the model: `A_blind`
`1.990814e-06` and `B_zero` `1.990788e-06` are both below the trained `2.038484e-06`.

**What the fit produced, `runs/fit_reduce.json`:**

* `selected_na = 28`
* `eps_splithalf_hinf = 0.00815401142380887` (D-156, the constant-free reduction tolerance)
* Hankel singular values: `3.679e-03, 3.059e-03, 1.466e-03, 1.227e-03, 3.519e-05, 1.873e-05, ...`
  i.e. a factor-35 gap after the fourth
* `nx_aug = 2`
* **the retained pole: `f_hz = 5.018812642664096`, `zeta = 0.09268363132157441`**

**What the fit had found before reduction**, `runs/pole_gate.json`: at 1x Telica sigma, differenced,
IV at `na = 28`, the nearest identified pole to the coupled-mode truth `157.8937 Hz` was
**`157.710 Hz`, `-0.116 %`**, with `zeta = 0.05208` against `0.05276`, and `rho(A_r) < 1`.

**Pre-flight**, `runs/preflight.json`: `dL/dW^a` is exactly `0.0` at step 0 and
`2.576e-11` at step 1; `blk.nu_log`, `blk.theta`, `blk.B` likewise `0.0` then non-zero. The D-130
dead zone is fixed. `blk.alpha` is non-zero at step 0 (`3.693e-06`), which is the ReZero gate
behaving as its paper describes.

**Other stated outcomes** (from `tasks/overnight-2026-08-23-verdicts.md`): the D10 falsifier fired,
the installed pole training away from the truth while RMS improved; no noise arms ran; the refusal
condition was exercised on a nothing-to-find case and fired correctly.

## 4. Why the two are not directly comparable, and other caveats

**Different untrained baselines.** The server factorial ran at `2.186580e-06`. The BLA night's
restored harness gives `2.534187e-06`, and `runs/preflight.json` records `bit_identical: false`
against the `2.186601103417735e-06` gate. That is a **15.90 % shift in the untrained model**.

**AUDIT CORRECTION, 2026-08-23: the cause is now located and the models are materially different.**
The snapshot patch
`tasks/snapshots/2026-08-22-working-implementation/patches/scripts_gantry_gantry_interconnect_dynamic.py.patch`
shows that the working factorial configuration changed the restored `4cdb7c1` configuration from
`joint_estimation=True` with a fourteen-entry `param_init_detune` vector of `0.9/1.1` multipliers to
`joint_estimation=False` with `param_init_detune=None`. `probe_preflight.py` replaces only `seed`,
`lr`, and `ann_route_ix`, so the BLA preflight inherited the restored **joint, detuned physical
plant**. `cl_train.py` inherited the snapshot's **fixed, nominal physical plant**. This is not a CPU
roundoff effect or a cosmetic metric change. The snapshot manifest's separate assertion that the
reset changed `closed_loop_rollout` is contradicted by its own archived patch: against `4cdb7c1`,
that patch adds `return_error` and window-diagnostic helpers but does not change the rollout or the
scalar path. The configuration difference is the demonstrated material difference.

The following arithmetic remains a description of each night internally:

* factorial plateau arms: `1.3979e-06 / 2.18658e-06` = **36 %** improvement
* A2: `2.0384e-06 / 2.53419e-06` = **19.6 %** improvement

**Do not compare even these fractions across nights.** They have different denominators because
they evaluate different physical parameterisations. In particular, do not read `2.04e-06` against
`3.79e-07` as a 5x gap. Only A0/A1/A2 comparisons within the BLA night and comparisons within the
factorial remain valid.

**Heterogeneous hardware.** Factorial arms ran across blade1, blade2, blade3 and blade4, which are
not the same CPU. D-072 passed on the submit node only. Each arm's `base` differs in the 7th digit
(`2.186517e-06` to `2.186603e-06`), consistent with that. Immaterial at the scale of the differences
being read, but it is why the bases are not identical.

**Wave 2 incomplete**, as in section 2. The factorial's primary criterion is missing.

**Wall clock.** Factorial arms took 51 to 104 minutes against a `25-30 min` estimate, from seven
concurrent arms on one blade. BLA arms took 2.5 to 2.9 hours against an `~85 min` estimate.

## 5. One reading of the above. This section is interpretation, not measurement.

**5.1 `AUG_LRU` is necessary.** `F1` sits at `1.397913e-06`, the plateau, and the three `F3*_F1_*`
arms sit within `0.0007e-06` of it. Removing the block returns the model to what the static ANN
correction achieves alone.

**AUDIT NARROWING, 2026-08-23.** The experiment establishes the necessity of a **driven recurrent
temporal feature path in this architecture**, not of the LRU parameterisation as a method. `F1`
removes `A_aa` and `B_a` together; `F2` separately shows that recurrence without external drive is
almost inert (`1.0470x`). No non-LRU driven recurrence was tested.

**5.2 The frequency band is the active ingredient; `rho` is secondary.** `F4a`, which widens only
the frequency arc, collapses to `1.390669e-06`. `F4b`, which widens only `rho`, reaches
`1.253324e-06`. Both narrow gives `3.79e-07`. So the `14.16 Hz` window
`[149.90234375, 164.0625]` is carrying the result.

**AUDIT CORRECTION, 2026-08-23.** Target-frequency overlap is load-bearing, but **band width was not
isolated and `rho` is not shown to be secondary**. The raw F4a draw is at `1321.76, 486.99,
1841.14, 288.80 Hz`: widening happened to put all four poles far from 158 Hz. F4b changes all four
radii to `0.8746, 0.5667, 0.6561, 0.6283`; its `1.9381x` shows persistence is also load-bearing.
Thus F4a tests an off-target bank and F4b tests a mostly short-memory bank. Neither is a matched
wide-versus-narrow or high-versus-low-`rho` contrast.

**SECOND AUDIT, 2026-08-23 (section 9.4). The last sentence above is wrong: F4a and F4b ARE matched
one-factor contrasts.** The draw maps two fixed uniform variates through the band, so changing only
`AUG_LRU_BAND` leaves `r_init` untouched and changing only `AUG_LRU_RHO` leaves `theta_init`
untouched, and `B_a` is drawn afterwards from the same generator state and is bit-identical. Checked
against the logs: F4a's radii `0.9920, 0.9847, 0.9865, 0.9859` are F5's exactly, F4b's frequencies
`159.26, 153.34, 162.94, 151.94` are F5's exactly, and `||B_a||_F = 7.0138e-01` in F5, F4a, F4b and
F4c alike. So F4a is a pure frequency contrast at fixed `rho` and fixed drive, and F4b is a pure
`rho` contrast at fixed frequency and fixed drive. The factorial is better controlled than either
section 5 or the first audit assumed. What remains un-isolated is band **width**, not band position
or `rho`.

**5.3 The published default is insufficient.** `F4c`, Orvieto Lemma 3.2 as written, gives
`1.399598e-06`: the plateau, and the worst arm of the sixteen. **The component that works has no
literature source, and the citable alternative is measurably worse.** The runner's own comment
records the hope that this arm would match, precisely so the initialisation could fall back to
something defensible. It did not.

**AUDIT NARROWING, 2026-08-23.** Claim 29 in `EVIDENCE.md` CONFIRMS that F4c implements the paper's
full-disk default, but this one seed and this readout establish insufficiency only for this task and
construction. They do not establish that every published-default draw is insufficient.

**5.4 `B_a` is necessary but not sufficient.** `F2` gives `1.339865e-06`, marginally above plateau.

**SECOND AUDIT, 2026-08-23 (section 9.9). Understated.** `F2` keeps the good seed-0 poles and its
per-pair ablation, never transcribed into 2c, is `1.0150, 1.0114, 1.0096, 1.0098`: **every pair
inert**. Correct pole geometry with no live input path contributes nothing at all, so `B_a` is not
one ingredient among several, it is what makes the band worth anything. It is also the precondition
in CONFIRMED `EVIDENCE.md` claim 4 and its own amendment, that a zero read-out trains *"provided ...
the states feeding it are excited"*; `B_a` is the only input path live at initialisation, the ANN
path being ReZero-gated to exactly zero. `F2` does not separate "too little amplitude" from "no
gradient at step 0"; `F2` with a non-zero initial gate does.

**5.5 `W^a` zeroed beats `W^a` random by 2.1x, under noise.** `F3b_arm2_wa_zero` `3.832755e-07`
against `F3a_arm2_wa_random` `8.088446e-07`, same configuration, same noise. `F3c_arm2_wa_frozen`
`3.838424e-07` says frozen and zero are equivalent. **This contradicts D-155**, which was written on
2026-08-22 from `EVIDENCE.md` claims 2, 3, 9 and 28 and concluded `W^a` should be random by Xavier.
D-152's original reasoning survives; its reversal does not.

**5.6 Noise is not the obstacle.** `F3b_arm2_wa_zero` is a noisy arm within 1.1 % of the best
noiseless arm.

**AUDIT NARROWING, 2026-08-23.** This rejects the proposition that **this one injected Telica-noise
realisation** prevents the result. It does not establish robustness to arbitrary noise levels or
realisations.

**5.7 The draw is unreliable.** Six seeds of the same configuration: `3.80, 4.89, 8.06, 9.66, 9.59,
4.20` (x 1e-7). Two of six reach the `3.80-4.89e-07` band; spread 2.5x; median about `6.5e-07`.
`3.795974e-07` was a good draw, not a typical one.

**AUDIT CORRECTION, 2026-08-23.** The **whole run seed** is unreliable; the pole draw alone is not
identified as the cause. Changing `cfg.seed` changes the four pole locations, four independent
`B_a` row projections, ANN and encoder initialisation, and minibatch order. The six outcomes cannot
be attributed to pole placement without holding those other draws fixed.

**5.8 The two nights locate the same failure.** `F4a` shows that poles outside the narrow window
give the plateau. The BLA night's reduction **kept a `5.019 Hz` pair and discarded the `157.710 Hz`
one the same fit had identified to `-0.116 %`**. So A2 was initialised into precisely the condition
`F4a` measures as fatal, and `F = -0.096` (the block is a net liability) is what that looks like
after training. **The BLA fit is not what failed. The step after it is.**

**AUDIT OVERTURNED, 2026-08-23.** That causal synthesis is false for two independent reasons.
First, `157.710 Hz` comes from `pole_gate.json`'s **noisy, differenced, IV, `na=28`** row;
`fit_reduce.json` selected a **clean, undifferenced ARX, `na=28`** model. They are not “the same
fit,” so the artefacts do not show that the reduction discarded that particular 158 Hz pair.
Second, section 4 now locates a materially different physical parameterisation between nights.
Within the BLA night one may say only that the installed 5.019 Hz reduced block was a liability;
the factorial cannot assign why.

**SECOND AUDIT, 2026-08-23 (section 9.11). The first reason above is wrong, but the replacement is
weaker than first written; the second reason stands.**
`fit_reduce.json` selected `clean_arx` and its own top-level `provenance` reads *"differenced
open-loop residual fit"*, so the condition is clean, **differenced**, ARX, `na = 28`, not
undifferenced.

**THIRD AUDIT, 2026-08-23. The `pole_gate` row is still not the same regression.** `fit_reduce.py`
line 329 is `difference2(u_f / std_u, rho_f / ystd)`, normalised then differenced;
`probe_pole_gate.py` line 114 is `difference2(u_f, rho_f)`, unnormalised. Channel scaling reweights
a shared-denominator least squares, so `158.10943 Hz` is not the pole of the fit that was reduced,
and the `d = 0.000267` computed from it is withdrawn. What survives is a faithful read-only
reconstruction of the normalised fit, which returns a nearest pole at **`157.9884 Hz`,
`zeta 0.05247`** (`d ~ 0.008` from truth). `fit_reduce.json` does not store its unreduced poles, so
this is a reconstruction, not an artefact identity. **The qualitative conclusion of 5.8 survives on
that reconstruction: the unreduced fit contained the mode and the step after it did not.** The
cross-night half stays withdrawn, on the detune.

**5.9 Why the reduction discarded the mode, and it is predictable.** `EVIDENCE.md` claim 15 records
that `2 * sum of discarded Hankel singular values` is a **uniform H-infinity ceiling over all
frequencies** and "does not promise that any *named* mode survives". After D1b the residual is
low-frequency dominated, so an H-infinity criterion spends its budget where the energy is. The
Hankel spectrum in `fit_reduce.json` shows the top four singular values two orders above the rest;
reducing to `nx_aug = 2` keeps the two largest, which are low-frequency. **The bound was satisfied
and the initialisation was useless.** The reduction objective and the design objective were never
the same quantity.

**AUDIT NARROWING, 2026-08-23.** Claim 15 CONFIRMS the uniform error bound and that it does not
preserve named modes. It does **not** say balanced truncation “spends its budget where the energy
is,” and `fit_reduce.json` does not associate individual Hankel singular values with the 5 Hz and
158 Hz modes before reduction. The retained 5.019 Hz outcome is measured; the proposed reason for
that outcome is a conjecture. A frequency-weighted or modal-participation calculation on the same
unreduced realisation would settle it.

~~**SECOND AUDIT, 2026-08-23 (section 9.11).** ... **Order 4 passes the same tolerance with a 17x
margin and would have kept both modes.** So the binding decision was an order-selection rule
preferring the smallest admissible order ...~~

**THIRD AUDIT, 2026-08-23. The order-4 conjecture above is FALSIFIED, and the real answer is worse
than it.** The inference was invalid on its face: `fit_reduce.py:127` `balanced_sp` returns
`Ar = A11 + A12 M A21`, a Schur complement, so the reduced poles are **not a subset** of the
unreduced spectrum and no Hankel-singular-value count licenses "order 4 would have kept the mode".
A read-only reproduction of the normalised differenced clean ARX-28 fit settles it:

| | nearest pole |
|-|-|
| unreduced | `157.9884 Hz`, `zeta 0.05247` |
| BSP order 2 | `5.0405 Hz` |
| BSP order 4 | two pairs at `~5.0400 Hz` |
| BSP order 6 | `~5.0400 Hz` pairs plus `496.79 Hz` |

**Balanced singular perturbation does not retain the absorber mode at any tested order.** So the
fault is not the order-selection rule and not the tolerance: it is that BSP on this realisation
relocates poles freely, which is exactly what a Schur complement is entitled to do. 5.9's core point
therefore stands and gets stronger, and the prescription changes from "use a larger order" to
**"select the pair modally from the unreduced fit and do not balance-reduce at all"**. That is the
change made to `fit_reduce.py` and recorded in D-159.

Caveat carried: the reproduced reduced pole is `5.0405 Hz` against the recorded `5.0188 Hz`, so the
original unreduced realisation was not preserved bit-exactly and the reconstruction is faithful
rather than identical.

**5.10 What this implies for a generalisable method.** The working recipe is: peak-pick the residual
spectrum with an unsourced `10 dB` threshold, then draw four poles in the resulting window. It is
not generalisable, and it cannot run on Telica at all: in simulation `deriv6` and `deriv8` differ
*only* by the absorber, so the residual is a clean modal signature at 65-168 dB over floor;
**Telica has no absorber**, and its residual is friction, stick-slip, cable forces and parameter
mismatch, with an identifiable band below 83 Hz on X and 55 Hz on Y.

But 5.2 says what the recipe contributes is **a narrow frequency window with several poles spanning
it**, not anything about peak-picking. And the pole gate shows the BLA fit produces such a window
from data where no peak-picker would work. **The untried combination is: use the residual fit to set
the band, keep the spanning draw.** That retains what `F4a` shows is load-bearing, replaces the
threshold with a parametric estimate, and never reduces to a single pair, which is the step that
failed. It is also consistent with the earlier observation that seed 1 drew a pole `0.01 Hz` from
truth, closer than seed 0's nearest, and landed **29 % worse**.

**AUDIT OVERTURNED, 2026-08-23.** The data do not imply this recipe. The fit supplies point poles,
not a validated band, and F4a does not isolate spanning. More importantly, seed 1 also has four
pairs and a different `B_a`, ANN/encoder draw and training order; it is not a one-pole exact-fit
arm. The proposal remains admissible as a conjecture, but it is no longer a consequence of the
factorial.

**SECOND AUDIT, 2026-08-23, amended by the THIRD AUDIT. The recipe should be reversed, but the
replacement is an experiment, not a settled recipe.** "Use the fit for the band, keep the spanning
draw" keeps the insurance and discards the information: the reconstructed normalised fit contained
the mode at `d ~ 0.008`, better placed than any of the 24 poles ever drawn (best drawn pole
`d = 0.047`, seed 1). What follows is: fit the residual, replace balanced reduction with **modal
selection**, install the identified pair, keep `B_a`, leave `W^a` at zero.

Two limits, both conceded to the third audit. **This has never been run**, so it is the next arm and
not an earned conclusion. And **`zeta_hat * f_hat` gives a width, not a warrant**: a poorly
identified friction or parameter-mismatch pole has a mathematically defined damping width without
representing omitted dynamics worth modelling, so 7.4's Telica question is *not* closed by 9.3. The
band half-width is the right scale once you have a target worth hitting; deciding whether Telica has
one is a separate and unanswered question, and `fit_reduce`'s own noisy-condition refusal
(out-of-sample VAF `-0.0136`, "the residual is not a linear object on this data") is the current
evidence that it may not.

## 5b. What the ablation ratios changed. Added after section 5 was first written.

**AUDIT CORRECTION, 2026-08-23.** Sections 5.2, 5.7-5.10 and 5b.1/5b.3/5b.5-5b.7 are narrowed or
overturned in place by this audit.

~~Nothing in section 5 was overturned. Two points became much stronger and one mechanism appeared.~~

**5b.1 Ratio and RMS are monotone across all fourteen arms**, with one inversion (`seed3` `2.1676x`
at `9.66e-07` against `seed4` `1.6806x` at `9.59e-07`). **Every arm that reaches `e-7` has
load-bearing augmented states and every arm at the plateau does not.** There is no decoration
anywhere in the set, which was section 6's largest stated risk. `F5_frozen_poles` returns `5.2144x`
against the planted reference of `6.010x`.

**AUDIT CORRECTION, 2026-08-23.** “No decoration anywhere” is false: the plateau arms immediately
listed in 5b.2 are decoration by this document's own near-1 criterion. The supported statement is
that every `e-7` arm measured so far has a load-bearing augmented route.

**5b.2 The plateau arms are not merely unhelped, their augmented states are inert.** `F1` `1.0000x`,
`F3b/c_F1_*` `1.0000x`/`1.0001x`, `F4a_wide_freq` `1.0139x`, `F4c_orvieto_default` `1.0255x`,
`F2_no_Ba` `1.0470x`. So 5.1, 5.3 and 5.4 are stronger than the RMS alone showed: removing
`AUG_LRU`, using Orvieto's published draw, or removing `B_a` does not degrade the augmented states,
it makes them do **nothing at all**.

**5b.3 The frequency band result is now unambiguous.** `F4a_wide_freq` `1.0139x` against the
artefact band's `5.2144x`. Widening the frequency arc does not weaken the contribution, it removes
it. `F4b_wide_rho` at `1.9381x` sits in between, confirming `rho` as secondary. This is 5.2, and it
is the single sharpest number in the set.

**AUDIT CORRECTION, 2026-08-23.** The contrast unambiguously establishes **target-frequency
overlap**, not narrow width, because F4a drew no pole near the target. F4b does not establish that
`rho` is secondary; changing every radius and losing most of the ratio establishes that long
memory is jointly necessary in this draw.

**SECOND AUDIT, 2026-08-23 (sections 9.4, 9.5).** F4b changes *only* the radii, at F5's exact
frequencies and bit-identical `B_a`, so it is a matched `rho` contrast after all, and its per-pair
ablation resolves it within the arm: the single pole whose `rho` leaves it inside the mode's
neighbourhood gives `1.8347x` and the other three give `1.0951x, 1.0035x, 0.9897x`. `rho` is not
secondary and not merely "memory": it is one of the two coordinates of the distance that decides
whether a pole is usable at all.

**5b.4 The band share separates the two groups.** Working arms sit at `0.80-0.82` band share with
in-band ablation costs of `5.3x` to `6.3x`; plateau arms sit at `0.61-0.64` with in-band costs of
`1.05x` to `1.06x`. **The augmented states, when they work, are doing in-band work specifically.**

**5b.5 The mechanism question is HALF answered: spanning is measured, why it works is not.**
Spanning is the observation, not the explanation, and the distinction matters because the two
candidate explanations imply opposite next experiments. See section 7.1. `F5`'s per-pair ablation:
pair 0 `3.2280x`, pair 1 `2.7186x`, pair 2 `1.3171x`, pair 3 `1.3650x`. **All four pairs contribute
and two contribute strongly.** The noisy arm `F3c_arm2_wa_frozen` reproduces it: `3.2747x`,
`2.6067x`, `1.2310x`, `1.4588x`. So the `14.16 Hz` window is not delivering one resonator that sits
on the mode; it is delivering **several poles that each carry part of the in-band correction**.
That is consistent with seed 1 having drawn a pole `0.01 Hz` from truth and landing 29 % worse, and
it is the mechanism section 5.10 needed and did not have.

**AUDIT CORRECTION, 2026-08-23.** Per-pair post-training ablation establishes that the readout uses
all four realised pairs, but not that **frequency spanning** caused their usefulness. Each pair also
has a different two-row `B_a` projection. The observation is equally compatible with input-direction
coverage, and the seed-1 comparison is not controlled. Section 8 gives the mechanism supported by
the existing isolation probes and the measurement that would separate the two readings.

**SECOND AUDIT, 2026-08-23 (section 9.5). "Equally compatible" understates the pole side.** Scoring
each pair by its pseudohyperbolic distance `d_k` to the true mode, over the nine per-pair blocks that
were never transcribed plus the two that were, 44 pairs in all: `Spearman(d_k, ratio) = -0.625`
pooled, and **six of six independent pole draws give a negative within-arm rank correlation**
(`p = 0.016` under a sign test). Eleven pairs sit at `d_k >= 0.9` and **not one exceeds `1.0951x`**,
against `3.2747x` for in-band pairs. `F4b` is the decisive case: four poles at F5's exact
frequencies with only `rho` changed, `d_k = 0.822, 0.954, 0.939, 0.944`, measured
`1.8347, 1.0951, 1.0035, 0.9897`. Only the pole inside the neighbourhood does anything, and it is
inside on `rho` alone. So pole geometry does predict which pairs are usable; what it does not predict
is how much of the usable set the read-out recruits, which is where the input-direction reading
belongs. Section 9.7 sets out both as necessary conditions rather than rivals.

**5b.6 `W^a` contributes nothing, which explains 5.5.** Zeroing the **trained** `W^a` costs
`1.0002x` in `F5` and `1.0000x` in `F3c_arm2`, despite `F5`'s `W^a` having trained from exactly `0`
to `||W^a||_F = 1.983904e-02`. It moves and it does not matter. So a random `W^a` is not adding
capability, it is injecting an initial `x_a` the optimiser has to work around, which is why
`F3a_arm2_wa_random` (`2.0278x`, `8.088e-07`) is beaten 2.5x on ratio by the frozen and zero
variants (`5.1551x`, `3.837e-07`). **D-155 is refuted and now also explained.**

**AUDIT NARROWING, 2026-08-23.** The long-free-run irrelevance is measured; “the optimiser has to
work around” a random initial state is only a conjecture. Section 8.4 supplies a horizon derivation
for why a trained `W^a` can move on 400-sample windows yet vanish from a 48,000-sample free run.

**5b.7 The draw problem extends to the ratio.** Seeds 2 to 5 give `2.2664x`, `2.1676x`, `1.6806x`,
`4.4157x`. Only `seed5` is in the same class as `F5`. So the draw determines not just how good the
result is but whether the augmented states are used at all.

**AUDIT CORRECTION, 2026-08-23.** As in 5.7, this is a **run-seed** effect. The data do not isolate
the pole draw from `B_a`, the neural initialisation, or training order.

## 6. What would overturn section 5

* ~~**Wave 2's ablation ratios.** If `F5_frozen_poles` returns a ratio near `1.0`, its
  `3.790189e-07` is decoration.~~ **RESOLVED 2026-08-22: `5.2144x`.** Not decoration. See 2b and 5b.
  This was the largest open risk and it closed in favour of section 5.
* **Tasks 11 and 13, still pending as `77991`.** Task 11 `F3b_arm2_wa_zero` is half of 5.5's
  comparison; its intact RMS `3.831160e-07` is already visible in the log, so a ratio near `5x` is
  expected and anything near `1.0` would make 5.5 and 5b.6 incoherent with `F3c_arm2_wa_frozen`'s
  `5.1551x`. Task 13 is a plateau arm and its triple-mates already read `1.0000x` and `1.0001x`.
* ~~**An explanation for the `2.5342e-06` versus `2.1866e-06` baseline shift.** If the restored
  harness differs materially rather than cosmetically, section 3's numbers describe a different
  model and 5.8 is comparing across it.~~ **RESOLVED 2026-08-23: it differs materially.** The BLA
  preflight used joint estimation with a ±10% detuned physical parameter vector; the factorial used
  a fixed nominal plant. Section 5.8 is overturned and cross-night comparisons are withdrawn.
* **A frequency-weighted reduction that keeps the 158 Hz mode and still fails.** That would move the
  fault from 5.9 to somewhere else, and would weaken 5.10.
* **A spanning-draw-from-BLA-band arm that reaches the plateau rather than `e-7`.** That would refute
  5.10 directly.
* **Any arm in which the frequency window is wide and the result is still `e-7`.** That refutes 5.2,
  which most of section 5 rests on.

## 7. Open. Ranked, because these are not equally important.

### 7.1 WHY does spanning beat exact placement? The central open question.

**AUDIT RESOLUTION, 2026-08-23: choose (b), with an important correction. Hypothesis (a) is
refuted for the omitted physical mode, and “spanning beats exact placement” was never the measured
contrast.**

The written-out frozen-plant check is direct. At fixed `Y`, `plant.py` defines

`A(Y) = [[0, I], [-M8(Y,0)^(-1) K4, -M8(Y,0)^(-1) C4]]`.

Substituting the stored plant constants and taking the positive-imaginary eigenvalues for
`Y = -0.4, -0.3, ..., 0.4 m` gives the same high pair at every point to the displayed numerical
precision: damped frequency `157.893656426711 Hz`, natural frequency `158.113874 Hz`, damping ratio
`0.052760`. Across that grid the damped high frequency varies by only `6.1e-08 Hz`, while the low
gantry mode ranges from `4.4053` to `5.1221 Hz`; the omitted absorber mode does
not. This calculation uses the generator's equations, not a residual estimator. The existing
residual artefact independently fails to support (a): `cl_residual_spectrum.json` reports the
18-record regression slope `+3.5607 Hz/m`, `R^2 = 0.02447`, and `10.2539 Hz` scatter. `pole_gate.json`
pools all fourteen training records and has no per-Y pole estimates, while `per_record.final` is a
prediction-error scalar and contains no frequency estimate. Claims 17-20 in `EVIDENCE.md` remain
CONFIRMED statements about local LPV identification, but their precondition is not the mechanism
here because the physical 158 Hz pair is invariant under frozen Y.

The fourth “fact” also needs correction. Seed 1 did not use a single exact pole; it used four pairs
(`157.90, 162.65, 152.15, 159.98 Hz`), just as seed 0 used four. Its seed also changed `B_a`, neural
initialisation and training order. The observation proves only that nearest-pole distance does not
rank two complete seeded runs. No existing training arm compares one exact pair with four pairs.

What the existing artefacts do identify is **drive/subspace mismatch**. In
`transient-investigation/runs/encoder_isolation_probe.json`, one pair placed at the true mode and
driven through the same random `B_a` convention leaves `1-R^2 = 0.98842`: frequency accuracy alone
explains only `1.16 %` of the in-band residual variance. Supplying that same two-state pair with the
**true absorber trajectory** instead leaves `1-R^2 = 0.35254`, explaining `64.75 %`. The pole is
therefore capable; the random driven trajectory is wrong. `band_draw_probe.json` agrees: eight
single-pair draws inside the band have median `1-R^2 = 0.98819`, barely different from the exact
pair, whereas full-circle draws are worse at `0.99889`. F2 adds the causal training result: without
`B_a`, the route is almost inert (`1.0470x`).

The one-sentence reading is: **the narrow, high-`rho` bank is a fixed resonant random-feature lift;
several distinct, continuously driven trajectories at the omitted mode's time scale improve the
finite-window feature subspace and its conditioning, while exact eigenfrequency cannot repair a
bad one-pair trajectory.** This is stronger than “they span and combine”: the proposed mechanism is
diversity and conditioning of the **driven regressors**, constrained to a useful temporal basis.
Whether that diversity comes mainly from frequency separation or independent `B_a` projections is
not identified. F4a explains the constraint—all
four of its diverse trajectories are off-target. F4b explains the memory requirement. The per-pair
ablations and `readout_jacobian_arm2_nx8.json` (all four nonzero, `||J_aug||/||J_phys|| = 1.342`)
show that the trained readout uses the available directions. They do not by themselves prove why.

**What would overturn this reading.** One offline, no-training, `2 x 2` feature-fit measurement on
the already stored windows is sufficient: compare one versus four pairs, and duplicated versus
independent `B_a`, while setting every pole exactly to the true pair; then repeat with one duplicated
`B_a` while spreading the four frequencies over the band. Score held-out best-linear-readout
`1-R^2`. If four identical-frequency pairs with independent `B_a` beat the duplicated-`B_a` bank
and approach the spread bank, input-subspace coverage is the mechanism. If only frequency spread
helps, the present reading is wrong and filter-bank frequency coverage is the mechanism. If neither
helps, the nonlinear joint-training dynamics—not the fixed feature geometry—produce the factorial
result. This is a conjecture until that matched measurement is run.

### 7.2 Why does `W^a` train to something useless?

Zeroing the **trained** `W^a` costs `1.0002x`, yet it trained from exactly `0` to
`||W^a||_F = 1.983904e-02`. It moves and it does not matter. 5b.6 explains the *consequence* for
`W^a` random versus zero; it does not explain why the encoder's augmented block converges to a
non-zero value that contributes nothing. Nobody has asked. It bears directly on whether the encoder
route is worth keeping at all.

**AUDIT RESOLUTION, 2026-08-23: this is a horizon mismatch, not evidence that the training
gradient is meaningless.** `W^a` changes only the window-start state. For two trajectories that
differ only in `W^a`, before nonlinear coupling their state difference is
`delta x_a(k) = A_aa^k delta x_a(0)`. Therefore

`sum_(k=0)^(N-1) ||delta x_a(k)||^2 <= ||delta x_a(0)||^2 / (1-rho^2)`.

At the largest factorial radius `rho = 0.9956`, the energy length is at most
`1/(1-rho^2) = 113.9` samples. Training re-encodes every `N = 400` samples, so that transient can
occupy up to `28.5 %` of every objective window. Free-run validation encodes once over approximately
`48,000` samples, reducing the same bound to `0.237 %` of the record. Continuous `B_a z_k` drive
does not receive this `1/N` dilution. The independent stored probe measures exactly the predicted
split: `wa_freerun_probe.json` gives a `1.5846x` window change from zeroing `W^a` but only `1.0127x`
in the free run. Thus a nonzero trained `W^a` can matter to the repeated-start training objective
and be negligible to the long-run ablation. What remains a conjecture is whether deleting the
encoder route would improve optimisation; only a matched training arm would settle that, and none
was launched here.

### 7.3 The `2.534187e-06` versus `2.186601103417735e-06` baseline shift

A 16 % change in the untrained model, logged as re-based and never explained
(`BLA-Augmentation/runs/preflight.json`, `bit_identical: false`). **Every cross-night comparison in
this document silently rests on those being the same model.** Cheap to settle, and nobody has.

**AUDIT RESOLUTION, 2026-08-23: they are materially different models; every cross-night numerical
comparison is withdrawn.** The archived entry-file patch shows the exact switch: the factorial
snapshot uses `joint_estimation=False`, `param_init_detune=None`; restored `4cdb7c1`, inherited by
the BLA preflight, uses `joint_estimation=True` and fourteen physical parameters multiplied by
`0.9/1.1`. `probe_preflight.py` does not override either field. The measured `15.90 %` baseline
shift is therefore unsurprising and cannot be attributed to hardware. Section 4 records the source
and also corrects the snapshot manifest's unsupported claim that a rollout-code change caused it.
Within-night ablation ratios remain valid; section 5.8's cross-night causal story does not.

### 7.4 The rest

* `F5_frozen_poles` is the best arm, and freezing has no literature support: `EVIDENCE.md` claim 10
  (`marconato2014init`, CONFIRMED) establishes the literature re-estimates after initialisation.
  The two best-performing components, the band and frozen poles, are the two with the weakest
  citations.
* No source initialises a dynamic added block from a residual fit. Recorded in the BLA night's
  verdict file as the design's own step, now with a measured negative attached.
* D1's move to an open-loop residual is contested and reopened (`DESIGN.md` D1, REOPENED block).
  Section 3's numbers rest on it.
* The baseline is not fitted on the real system, so a Telica residual carries parameter mismatch.
  Ordering, not caveat: fit the baseline, then augment.
* Tasks 11 and 13's ablation ratios, resubmitted as `77991`. Section 6 states what each outcome
  would mean. Everything else in wave 2 is in and is in section 2b.
* **What the working recipe cannot do**, restated here because it is the reason 7.1 matters rather
  than being of academic interest: the recipe peak-picks the residual spectrum with an unsourced
  `10 dB` threshold, and on Telica there is nothing to pick. `deriv6` and `deriv8` differ *only* by
  the absorber, so the simulation residual is a clean modal signature at 65-168 dB over floor;
  Telica has no absorber, its residual is friction, stick-slip, cable forces and parameter mismatch,
  and its identifiable band is below 83 Hz on X and 55 Hz on Y. **Whichever of 7.1's two
  explanations is right determines whether a Telica-runnable substitute exists**: (a) needs a
  per-Y envelope, which is derivable from data with no clean mode in it; (b) needs coverage of a
  band, which needs some other way to decide where the band is.

## 8. Independent reading, 2026-08-23

### 8.1 Pre-result prediction for tasks 11 and 13

This prediction was written **before inspecting** `wave2_77991_11.out` or
`wave2_77991_13.out`. The files had appeared in the local artefact tree by the time of this audit,
so this is a pre-inspection prediction rather than literally a pre-arrival prediction.

* **Task 11, `F3b_arm2_wa_zero`: above `5`, approximately `5.2x`.** Conjecture: it will match
  task 12 because task 12 already has exactly frozen-zero `W^a`, gives `5.1551x`, and the clean
  counterpart's trained nonzero `W^a` changes the ablation RMS by only `1.0002x`. Near `1.0`
  would mean its low intact RMS was achieved outside the augmented-state route and would overturn
  the claim that zero-initialised, trainable `W^a` is immaterial; `1-2` would mean weak or
  stochastic route use; above `5` would confirm strong augmented-state use independent of a useful
  `W^a`.
* **Task 13, `F3a_F1_wa_random`: near `1.0`.** Conjecture: it will match the other `F1` arms
  because its intact RMS is on their common `1.3978e-06` plateau and removing `AUG_LRU` leaves no
  fixed resonant state transition for a random encoder initial condition to sustain. Near `1.0`
  would confirm that random `W^a` does not rescue the plain ANN-written latent rows; `1-2` would
  show a weak route created by their initial condition; above `5` would overturn the present `F1`
  reading and show that encoder initialisation alone can make those rows essential.

### 8.2 Raw-number audit

Every numeric table was checked against the named primary artefacts rather than accepted from this
document.

* All sixteen `final` values in section 2 match `cl_train_w1_*.json`. The full-precision endpoints
  are `3.790189168858397e-07` for F5 and `1.399598364875e-06` for F4c; the displayed rounding is
  faithful. The stated base range also matches the run files.
* Every available intact RMS, A/B ratio, per-pair ratio, in/out-of-band ratio and band share in
  sections 2b-2d matches the corresponding `wave2_*.out`. Tasks 11 and 13 still have only intact
  RMS in `wave2_77991_11.out` and `wave2_77991_13.out`; no ratio was present at the time of this
  audit.
* All three BLA ablation records match `ablation_a*_clean.json`. One transcription omission was
  corrected in section 3: A0's raw trained RMS is `1.9049812841184182e-06`. The exact fractions are
  `0.000724001`, `0.021783760`, and `-0.096219935` for A0/A1/A2.
* `fit_reduce.json` confirms clean ARX, `na=28`, `eps=0.00815401142380887`, `nx_aug=2`, and the
  retained `5.018812642664096 Hz` pair. `pole_gate.json` confirms `157.7104498 Hz`, but under a
  different noisy/differenced/IV configuration; section 5.8 had conflated them and is corrected.
  `preflight.json` confirms `2.534187007593955e-06` and the stated step-0/step-1 gradients.
* Section 3's “other stated outcomes” cites a verdict handoff, not one of the primary raw artefacts
  supplied for this audit. In particular, the claimed D10 movement from `5.019` to `3.926 Hz` is
  **second-hand and remains unverified here**; it is not used in the diagnosis. Likewise, no raw
  noise-arm record exists because those arms did not run.

### 8.3 My reading, explicitly labelled

**READING.** The augmented states do not learn because a narrow band is intrinsically magic. They
learn when the architecture supplies continuously driven, sufficiently persistent temporal
features whose frequencies overlap the missing dynamics. With only one pair, a random `B_a`
projection usually produces the wrong two-dimensional trajectory even at the exact pole. Four
independent projections give the readout more chances to span a well-conditioned residual feature
subspace while the narrow high-`rho` constraint prevents those chances being spent on irrelevant
time scales.

This reading earns each part separately:

* **Continuous drive:** F2 is `1.0470x`; F1 is `1.0000x`. In the state equation
  `x_a(k+1)=A_aa x_a(k)+gamma B_a z(k)+...`, deleting `B_a` removes the only live exogenous term at
  initialisation. The route then cannot build a persistent data-dependent feature trajectory.
* **Relevant time scale:** F4a's actual four frequencies are all outside the target region and the
  route is `1.0139x`; full-circle F4c is `1.0255x`. The single-pair offline probe improves from
  median `1-R^2=0.99889` off-band to `0.98819` in-band. This establishes overlap, not a particular
  `14.16 Hz` optimal width.
* **Persistence:** F4b's mostly short-memory bank retains only `1.9381x`, materially below F5's
  `5.2144x`. For a radius `rho`, an impulse contribution decays as `rho^k` and has state-energy sum
  proportional to `1/(1-rho^2)`; changing `rho` from `0.99` to `0.63` reduces that memory factor
  from about `50.3` to `1.66`. Calling `rho` secondary was not justified.
* **Subspace rather than exact frequency:** the exact-pole random-drive probe explains `1.16 %` of
  in-band variance, while the actual two-state absorber trajectory explains `64.75 %`. That
  difference holds `A_aa` fixed and changes the trajectory supplied to the readout; it directly
  locates the deficiency in drive/state realisation rather than pole placement.
* **Several chances:** all four F5 pairs have nonzero post-training ablation cost and nonzero
  readout Jacobian. This is consistent with independent `B_a` projections supplying complementary
  directions. It is not yet causal proof, so the claim remains the conjectural part of the reading
  and the factorial measurement in 7.1 is the falsifier.

This chooses candidate (b) over (a), but it does **not** endorse the draft's stronger “frequency
spanning is the mechanism.” Frequency diversity and input-direction diversity were changed
together. Existing data locate a negative—one exact randomly driven pair is inadequate—and locate
the deficiency in its driven trajectory; they do not yet say whether frequency separation or
independent input projections are what make the four-pair remedy work.

### 8.4 Why 7.2 and 7.3 mattered to 7.1

`W^a` and `B_a` are not interchangeable input paths. `W^a` acts once at each encoded start and its
effect decays; `B_a` acts at every sample. The `1/(1-rho^2)` derivation in 7.2 and the measured
`1.5846x` window versus `1.0127x` free-run split explain why `W^a` can train nonzero without being
load-bearing in the long metric. That supports the continuously driven-feature reading of 7.1.

The section 7.3 result removes the apparent BLA corroboration entirely. Because the BLA night used
a detuned joint physical plant and the factorial used a fixed nominal one, A2's negative fraction
cannot be combined with F4a/F5 to diagnose the same failure. The BLA night remains a valid
within-night negative for its 5 Hz reduced construction and nothing more.

### 8.5 What would overturn my reading

* The matched offline `2 x 2` feature-fit in 7.1 overturns input-direction coverage if duplicated
  `B_a` plus frequency spread wins while independent `B_a` at identical exact poles does not.
* A per-Y eigenanalysis using a plant different from the checked `plant.py` that moves the physical
  158 Hz pair by several hertz would reopen candidate (a). Residual peak regressions alone would
  not: the settling measurement is the frozen physical eigenvalue or a per-Y local pole estimate
  with Y held, as required by CONFIRMED `EVIDENCE.md` claim 20.
* A controlled one-exact-pair training arm that reaches F5's RMS and ablation ratio with the same
  `B_a`, ANN initialisation and update order would refute the need for multiple feature directions.
* If task 11 lands near `1.0`, the current link from low RMS to a load-bearing driven route fails for
  the zero-`W^a` noisy arm. If task 13 lands above `5`, the assertion that recurrence/continuous
  drive is necessary fails for the plain latent rows. The intermediate `1-2` interpretations are
  stated in 8.1 and remain the preregistered boundary.

### 8.6 Literature status

The mechanism was named before literature search as **a resonant random-feature bank whose distinct
driven trajectories improve residual-subspace coverage**. Claims 17-20 were checked first and are
not the explanation because candidate (a) failed the frozen-plant calculation. One targeted search
was then run for this mechanism; no general sweep on pole placement, reduction or augmentation was
opened.

**CONJECTURE FROM AN UNVERIFIED LITERATURE LEAD, not an `EVIDENCE.md` claim.** The strongest located
source is Liu and Li, *Autocorrelation Matters: Understanding the Role of
Initialization Schemes for State Space Models* (2024),
[arXiv:2411.19455](https://arxiv.org/abs/2411.19455), held at
`literature/deep-ssm-init/2411.19455_AutocorrelationMatters.pdf`. Its
Section 4.3 studies the Gram matrix of fixed damped-sinusoid features with a trained readout:
distinct nonzero imaginary parts make that matrix positive definite, and separation improves its
conditioning, while moving frequencies away from the target increases approximation error. The
paper therefore names an **approximation-versus-conditioning tradeoff** and supports several
separated features near dominant target frequencies. This is a direct candidate mechanism for why
several near-band pairs can optimise better than one exact pair. Scope limitation: its result is
continuous-time, infinite-horizon, scalar-input and linear-readout; it does not prove this
finite-horizon nonlinear closed-loop case or the role of independent `B_a` directions.

**SECOND UNVERIFIED LITERATURE LEAD.** Gu et al., *On the Parameterization and Initialization of
Diagonal State Space Models* (NeurIPS 2022,
[arXiv:2206.11893](https://arxiv.org/abs/2206.11893)), held at
`literature/deep-ssm-init/2206.11893_S4D.pdf`, independently represents a diagonal SSM as a linear
combination of damped Fourier-like basis kernels and motivates spreading their imaginary parts. CONFIRMED
`EVIDENCE.md` claim 29 on Orvieto et al.'s LRU supplies only the existing stable full-disk default;
the additional phase/memory passages inspected for this audit were not quote-verified.

**Located negative:** no completed source was found for the stronger claim that independent MIMO
input projections improve “controllability coverage” in this construction. Existing CONFIRMED
claim 4 is only a preconditioned analogy: it supports a pool of excited random hidden features
selected by a learned linear readout. Accordingly, independent-`B_a` coverage remains a conjecture.
The settling measurement is the matched offline factorial in 7.1, augmented with the singular
spectrum, effective rank and condition number of each actual finite-window feature matrix. A larger
`sigma_min`/effective rank for independent `B_a` at identical exact poles supports input-direction
coverage; improvement only under separated frequencies supports temporal Gram conditioning.

No quotation from the targeted pass was run through `verify_pdf_quote.py`; **nothing from it may be
added to `EVIDENCE.md` as CONFIRMED**. The sources were inspected from held PDFs, and the external
metadata query failed in the sandbox. This is therefore a literature lead plus a located negative,
not a new verified claim.

**Research log.** Repository holdings and `EVIDENCE.md` were searched before fetching. The held
LRU, S4D, S5, DSS and Liu-Li PDFs were inspected. One exact-title OpenAlex request returned no
usable response; there were zero successful external queries, zero dblp queries and zero arXiv-API
queries. TU/e browser access was unavailable and unnecessary because the relevant papers were held
and openly available. Coverage gap: no external Scholar cross-check and no source for the
independent-MIMO-input claim.

## 9. A second independent reading, written in parallel with section 8. Interpretation, labelled as such.

Sections 1 to 4 were re-derived first-hand from `cl_train_w1_*.json`, `wave1_77958_*.out`,
`wave2_77959_*.out` and `BLA-Augmentation/runs/*.json` before section 5 was read. This section was
written concurrently with section 8 and without sight of it; where the two now agree that is
convergence, not adoption, and where they differ it is marked. It is numbered 9 only because 8 was
taken while this was being written.

**Audit outcome, independent of section 8.2 and agreeing with it.** All sixteen `final` values, all
fourteen ratio pairs in 2b, all five per-pair rows in 2c, all seven rows in 2d, the `pole_gate` row
(`157.71045`, `zeta 0.05208`, `rho_A 0.94972`), the `fit_reduce` block and every `preflight` number
reproduce exactly. Two further omissions not in 8.2 are recorded in 9.9 and 9.10.

### 9.1 The Y-invariance of 7.1(a) is an algebraic identity, not a numerical finding.

7.1's audit refutes (a) by evaluating `plant.py` on a `Y` grid. The stronger statement is available
in closed form from the generator's own equations of motion,
`Matlab-scripts/Augmentation/gantrySystemExtended.m` lines 29-32, with `K4 = diag(0, k_b, 0, k_a)`.
The absorber mode is the non-zero eigenvalue carried by `delta_a`, so `omega_a^2 = k_a (M^-1)_44`.
Subtract row 4 of `M` from row 3, which leaves `mh*[0, -d, 1, 0]`, use it to clear column 3 from
rows 2 and 4, then expand:

```
det M           = mh*ma*( mT*(J - (mh+ma) d^2) - S^2 )
det M(1:3,1:3)  = (mh+ma)*( mT*(J - (mh+ma) d^2) - S^2 )
(M^-1)_44       = (mh + ma) / (mh * ma)                      [confirmed symbolically]
```

with `mT = m1+m2+mb+mh+ma`, `S = (m1-m2)Lb/2 - (mh+ma)Y - ma(L0+delta_a)` and
`J = Jb+Jh+(m1+m2)Lb^2/4+(mh+ma)d^2+mh Y^2+ma(Y+L0+delta_a)^2`. **Every `Y` and `delta_a` term sits
in a bracket common to numerator and denominator and cancels exactly.** Hence
`f_n = fa*sqrt(mh_total/mh_rigid) = 150*sqrt(10.1/9.09) = 158.113883 Hz`, independently of `Y`, of
absorber deflection, of the offset `d` and of the beam inertia. 7.1's `6.1e-08 Hz` residual variation
is float noise on an exact zero.

This matters beyond tidiness: it makes (a) unrecoverable rather than merely unsupported for this
plant, and it says which parameter would resurrect it. Only `ma` or `mh_rigid` moving with `Y` could,
and neither does.

**A correction to `pole_gate.json`'s scoring, affecting no conclusion.** `f_true = 157.8937` is the
**damped** frequency; the identified `f_nearest` is the **undamped** one. The closed form gives
`f_d = 157.8941` against `f_n = 158.1139`, a gap of `+0.140 %`, and that gap appears as a systematic
offset in every `f_err_pct` in the file. The clean differenced-ARX fits are accurate to about
`0.01 %`, not the `0.14 %` displayed, and the noisy IV `na = 28` row quoted in section 3 as
`-0.116 %` is `-0.255 %` against `f_n`. Recorded so it is not later mistaken for estimator error.

### 9.2 What the 14.16 Hz band is: one fixed pole, estimated 54 times.

`lru_band_from_artifact` takes the `[min, max]` of the dominant peak over 18 records times 3
channels. Those 54 numbers run `149.9023` to `164.0625 Hz` and, by 9.1, are 54 estimates of **one
pole**. The function's docstring says so: *"the estimator scatter itself is the ring width"*. The
`rho` band `[0.9794, 0.9956]` is the same scatter in damping and brackets the true
`rho* = exp(-zeta wn Ts) = 0.98698` at `fs = 4000 Hz` (bin width `4000/8192 = 0.48828 Hz`, so the
band edges are FFT bins 307 and 336).

The band is therefore a **confidence region around a point**. Nothing in it is a range, and nothing
in it is spanning.

### 9.3 The mechanism, one sentence, with the derivation.

**The block is a fixed-pole rational basis, and the Blaschke product of pseudohyperbolic distances
from the target pole to the basis poles is a predictive SURROGATE for how usable a draw is.**

**Status, after the third audit. This is a surrogate, not a proven ceiling for this architecture.**
The identity below is exact for scalar Hardy-`H2` kernel approximation with freely selectable
residues over an infinite horizon. The trained block has real conjugate pairs, a vector-valued fixed
random `B_a`, finite recorded inputs rather than impulse-response `H2`, a nonlinear jointly trained
readout, and closed-loop feedback. So `|B(lambda*)|` is a pole-geometry statistic that is *derived*
in an idealised setting, *validated* in the regime where that setting nearly holds (9.5's
single-pair linear-readout probe, `Pearson = +0.960` at `n = 16`), and *extrapolated* to the trained
arms where it orders 4 of 4. The literature is consistent with that status and no stronger: it
states a **bound** and a proportionality, never an equality for a single target pole (see 9.14).

Each pair is `x_k[t+1] = lambda_k x_k[t] + gamma_k b_k u[t]` with `lambda_k` fixed (`F5` freezes them;
elsewhere the trained tables move under `0.15 Hz`), so the block's contribution lies in
`span{1/(z - lambda_k)}`. On the unit circle `<1/(z-a), 1/(z-b)> = 1/(1 - a conj(b))`, so the best
one-pole relative `H2` error is

```
1 - (1-|a|^2)(1-|b|^2) / |1 - conj(a) b|^2  =  d(a,b)^2 ,
d(a,b) = |a - b| / |1 - conj(a) b|                  (pseudohyperbolic distance)
```

and with `n` fixed poles the residual factorises as `|B(lambda*)| = prod_k d(lambda*, lambda_k)`.
Near the circle the metric's natural radius is `1 - rho*`, i.e. `0.0130` in modulus and the same in
angle, so

```
angular radius (1-rho*)/(2 pi Ts)         =  8.28 Hz   ->  full width 16.6 Hz
mode half-power bandwidth 2 zeta f_n      = 16.67 Hz
peak-picker scatter, the artefact band    = 14.16 Hz
```

**CONJECTURE, corrected by the third audit.** These three all scale as `zeta f0`, and here they agree
to `15 %`. The original text said "these are the same quantity"; that is not established. There is no
uniquely derived cutoff in the metric (`d` is continuous, so "neighbourhood" implies a boundary that
does not exist), and the claim that peak-picker scatter *must* scale as `zeta f0` is not derived at
all. The targeted literature search found **no source for the damping-radius half**: zero hits on
arXiv and OpenAlex. **Settling measurement, cheap and synthetic**: peak-pick a simulated resonance
over a range of `zeta` at a fixed noise floor and check whether the estimator scatter scales as
`zeta f0`. Until then, the agreement of `14.16` with `16.67` may be a meaningful coincidence rather
than a rule, and the "why the unsourced `10 dB` threshold happened to work" story rests on it.

### 9.4 The F4 family is a single-factor experiment on pole location. This closes the gap 8.3 flags in itself.

Section 8.3 states its own weakness plainly: *"Frequency diversity and input-direction diversity were
changed together."* In the `F4` arms **they were not.** `B_a` is drawn from the same dedicated
generator immediately after the two `torch.rand(n_pairs)` calls that produce `r_init` and
`theta_init`. Changing `AUG_LRU_BAND` or `AUG_LRU_RHO` changes the *values* those draws are mapped
to, not the number of draws, so the generator state at the `randn` call is identical and `B_a` is
**bit-identical**. The logs confirm it: `wave1_77958_9` (`F5`), `_7` (`F4a`), `_8` (`F4b`) and `_16`
(`F4c`) all print `||B_a||_F = 7.0138e-01`, while seeds 2 to 5 print `7.5552e-01`, `7.8064e-01`,
`7.5281e-01`, `7.3329e-01`.

With the drive held bit-identical and only the poles moved, `|B(lambda*)|` computed from the printed
draws and the closed-form `lambda*`, with **zero free parameters**, orders the primary criterion
exactly:

| arm | poles | `\|B(lambda*)\|` | ratio A |
|-|-|-|-|
| artefact band, **arm-2 seed 0** | `rho 0.985-0.992`, `151.9-162.9 Hz` | **`0.0062`** | **`5.2081x`** |
| `F4b_wide_rho` | same frequencies, `rho 0.567-0.875` | `0.6949` | `1.9381x` |
| `F4c_orvieto_default` | full disc | `0.9242` | `1.0255x` |
| `F4a_wide_freq` | right `rho`, `289-1841 Hz` | `0.9896` | `1.0139x` |

Four of four monotone, including the ordering of the two arms that sit within `0.012x` of each other.
`|B| -> 1` predicts **inert**, not weak, which is exactly the `1.0139x` that 5b.3 calls the sharpest
number in the set.

**Correction, third audit: the control must be arm-2 seed 0, not `F5`.** `F5` also sets
`AUG_LRU_FREEZE=1` while `F4a/b/c` leave the poles trainable, so "only the poles moved" is literally
true only against the historical arm-2 seed-0 run, `5.208138914155868x`
(`runs/arm_ablation_arm2_nx8_520upd.json`). `F5` gives `5.2144x`, a `0.12 %` difference, so nothing
in the ordering changes; the row above has been switched to the matched control and the sentence is
now true as written.

**So pole location is causal, at fixed drive.** That is a direct disagreement with 7.1's audit
sentence *"The pole is therefore capable; the random driven trajectory is wrong"* taken as the whole
story. It is not a disagreement with the `encoder_isolation_probe` measurement itself; see 9.7.

### 9.5 Per-pair, pooled over eleven arms and 44 pairs.

Section 2c transcribes two per-pair blocks. Nine more are in the `wave2` logs and were never carried
across: `seed2`, `seed3`, `seed4`, `seed5`, `F3a_arm2`, `F2_no_Ba`, `F4a`, `F4b`, `F4c`. Scoring each
pair by `d_k = d(lambda*, lambda_k)` from its own arm's printed draw:

* `Spearman(d_k, per-pair ratio) = -0.625` over all 44 pairs. **Descriptive, not an `n = 44` test**
  (third audit): four pair-ablations from one trained model are dependent and their marginal effects
  are not additive in a nonlinear model, and several arms reuse identical pole and `B_a` draws.
* Within-arm rank correlations are negative in every arm measured:
  `-0.600, -0.800, -0.400, -0.400, -0.400, -0.400`. **Five independent draws, not six** (third
  audit): `F4b` shares seed-0's uniform variates and its `B_a`, so it is not independent. Five
  same-sign outcomes give a one-sided `p = 1/32 = 0.031`, and the statistic was chosen after seeing
  the data. Suggestive, small-sample, post-hoc.
* **Eleven pairs have `d_k >= 0.9`. Not one exceeds `1.0951x`, and the mean is `1.0139x`.** In-band
  pairs reach `3.2747x`. The separation is categorical, not graded.
* `F4b_wide_rho` is the single-crossing case and it is the cleanest datum in the set:
  `d_k = 0.822, 0.954, 0.939, 0.944` against measured `1.8347, 1.0951, 1.0035, 0.9897`. **The only
  pair that does anything is the only one inside the neighbourhood**, and it is inside on `rho`
  alone, at a frequency it shares with three inert siblings. That is `rho` acting as a first-class
  variable through the metric, which is the correct version of 8.3's "persistence" bullet and of
  5.2's demoted `rho`.

### 9.6 Where 9.3 stops. A located negative.

**Within the band, representation is saturated and the mechanism explains nothing.** All six seeds
give `|B| <= 0.0066`, i.e. every draw can represent the mode to better than `0.7 %` relative `H2`
error, yet their RMS spans `3.80e-07` to `9.66e-07`. Against that spread, at `n = 6`:

| statistic | Spearman vs RMS |
|-|-|
| `\|B(lambda*)\|` | `+0.029` |
| `kappa(Gram)`, `G_jk = 1/(1 - lambda_j conj(lambda_k))` | `-0.257` |
| `sigma_1/sigma_2` of that Gram | `+0.086` |
| frequency spread of the four poles | `-0.086` |
| largest drawn `rho` | `-0.086` |

None is a result and `n = 6` cannot resolve anything weaker than `|rho_s| > 0.83`. The same
conclusion arrives from the other direction: `F3a_arm2_wa_random` and `F3c_arm2_wa_frozen` have
**identical poles** and give per-pair rankings `1 > 2 > 0 > 3` against `0 > 1 > 3 > 2`.

**`d_k` predicts usability. Nothing measured here predicts how much of the usable set is used.** That
residual is exactly where section 8's input-direction reading lives, and 9.6 is the reason to take it
seriously rather than a rebuttal of it.

### 9.7 Synthesis: two necessary conditions, and neither section alone has both.

The two readings are not competing explanations of one effect. They are two necessary conditions
measured on different axes, each with the other held fixed:

| condition | isolated by | drive held? | poles held? |
|-|-|-|-|
| the pole must lie in the hyperbolic neighbourhood | `F4` family, 9.4 | **yes, bit-identical `B_a`** | no |
| the drive must reach the residual subspace | `encoder_isolation_probe.json`, 7.1 | no | **yes, exact pole** |

`encoder_isolation_probe` shows an exact pole with one random `B_a` explains `1.16 %`; `F4a` shows
bit-identical `B_a` with wrong poles explains nothing at all. Both are true.

**Softened by the third audit.** The original text said `1 - |B|` *is* a ceiling and the drive is the
fraction of it attained. That is not established for this architecture (see 9.3's status note). The
defensible version is weaker and still useful: `|B(lambda*)|` tracks whether a pole is **usable**,
the drive tracks whether a usable pole is **used**, and the two are measured on orthogonal axes.
It remains consistent with 9.6's negative, in that once `|B| <= 0.0066` for every seed the pole axis
stops discriminating and all remaining variation is on the other one, but "the ceiling is no longer
binding" is an interpretation of that pattern rather than a measurement of it.

Where 9.3 does disagree with 8.3: **"several chances to span the residual subspace" is not what four
poles buy in the frequency dimension.** One in-band pole already gives `d = 0.047` (seed 1's best
pair) or `d = 0.025` (the BLA fit's own `157.710 Hz / zeta 0.05208`). Four poles buy insurance
against a bad `rho` draw, since the `rho` band spans resonator bandwidths of `5.6` to `26.5 Hz`
around a target of `16.7 Hz`. Whether four poles also buy input-direction coverage is untested and is
exactly 7.1's proposed `2 x 2` feature-fit; 9.3 makes no claim against it.

### 9.8 Two points where section 8 or 7's audit supersedes what this section would otherwise have said.

* **7.2's horizon answer beats an Adam-drift account, and the drift account is withdrawn.** The
  arithmetic is real: `dL/d(blk.alpha) : dL/dW^a = 1.754e-06 : 2.576e-11 = 68065 : 1` at step 1, and
  `||W^a||_F = 1.9839e-02` over `8 x 102` entries is `6.945e-04` per entry, `5.2e9` times what plain
  SGD would give at `lr = 1e-5` over 520 updates and squarely between Adam's `lr*sqrt(T) = 2.28e-04`
  and `lr*T = 5.20e-03`. But `wa_freerun_probe.json`'s `1.5846x` window change settles it the other
  way: `W^a` has a genuine gradient in the objective actually optimised, so no optimiser pathology is
  needed to explain the motion. The step-1 gradient ratio is measured where `x_a` is still zero and
  is not representative. **7.2's resolution stands and this account is not needed.**
* **7.1's audit already identifies the seed confound**, that `cfg.seed` changes the pole draw, `B_a`,
  the ANN initialisation and the update order together. Independently reached here and recorded only
  to note the agreement. The `||B_a||_F` spread across seeds in 9.4 is a direct measurement of one
  arm of that confound.

~~What survives from the encoder analysis is the **magnitude asymmetry**: the input path's resonant
state gain is `gamma/(1-rho) = 12.36`, giving `|x_a| ~ 12.6` against an encoder contribution of
`~0.071`, a factor `178`; and `F3a`'s `||W^a||_F = 5.71` gives `|x_a(0)| ~ 20.4`, `1.61x` the forced
response.~~

**WITHDRAWN, third audit. This was wrong, and wrong in a way the code states plainly.** Orvieto's
`gamma = sqrt(1-r^2)` makes the state variance under white input exactly `||b||^2`: it **cancels the
resonant gain by design**, which is the stated purpose of the normalisation in
`AugLRUBypass.forward`'s own docstring. So there is no `12.36x` amplification for broadband drive,
and the `178x` compounded that error with an unstated unit-variance assumption about `z` and `psi`.

It is also directly measured, and the measurement was on disk the whole time.
`probe_encoder_isolation.py:294` iterates `(('ON', B_a), ('OFF', None))`, so the probe's `xa_rms`
fields are exactly this ratio:

| encoder init | `B_a` ON | `B_a` OFF | ratio |
|-|-|-|-|
| `ENC` | `0.8187` | `0.5262` | **`1.56`** |
| `TRUE` | `0.7045` | `0.3083` | **`2.28`** |

Derived `178`, measured `1.56`. The encoder path is *comparable* to the input path in state RMS, not
two orders below it. The `1.61x` pricing of `F3a`'s random `W^a` goes with it.

**Consequence: 5.5's `2.1x` training penalty for a random `W^a` is back to unexplained.** 7.2's
horizon derivation explains why `W^a` moves and why *post-hoc* substitution is cheap in free run
(`wa_freerun_probe`: `1.0127x`), but the factorial's *trained-with-random* arm is `2.11x` worse
(`8.088e-07` against `3.833e-07`) and nothing here accounts for that gap. It is a second located
negative alongside 9.6, and it shares the same shape: an effect that appears during training and
not in any post-hoc probe.

Logging defect found while verifying: the per-pair probe prints *"it started at EXACTLY 0 under
ENC_WA_ZERO=1"* unconditionally, including on `F3a_arm2_wa_random` where `ENC_WA_ZERO` is unset and
`5.71` is the Xavier draw essentially untouched. Nothing was concluded from it, but "it moves" cannot
be claimed for that arm.

### 9.9 5.4 is understated, and `F2` has a CONFIRMED anchor.

`F2_no_Ba` keeps the good seed-0 poles and every one of its four pairs is inert:
`1.0150, 1.0114, 1.0096, 1.0098`, never transcribed into 2c. Good geometry, zero contribution, which
is the sharpest available statement that pole placement is necessary and not sufficient. `B_a` is the
only input path live at initialisation, since the ANN path into the augmented rows is `gamma * w`
behind a ReZero gate that starts at exactly zero. That is precisely the precondition in
**`EVIDENCE.md` claim 4 (CONFIRMED, with its own 2026-08-22 amendment)**: a zero read-out trains
*"provided ... the states feeding it are excited"*, and the amendment records that our configuration
deletes that path. **D-151's `B_a` restores claim 4's precondition and `F2` measures its removal.**

Honest limit: `F2` does not separate "too little amplitude" from "no gradient at step 0". `F2` re-run
with the ReZero gate initialised non-zero separates them.

### 9.10 7.3: the detune explanation is corroborated, and A0 adds to it.

7.3's audit finds the cause: `joint_estimation=True` with fourteen parameters at `0.9/1.1` on the BLA
night against `joint_estimation=False`, `param_init_detune=None` on the factorial. Two independent
pieces of support, neither in 7.3 or 8.2, and both are what a detuned plant predicts.

**The shift is not uniform.** It is concentrated in the record a parameter detune should hit hardest.

| record | BLA untrained | BLA A0 trained | server plateau | A0 / plateau |
|-|-|-|-|-|
| `V1_standstill_Yp10` | `2.4134e-06` | `1.5318e-06` | `1.3521e-06` | `1.133` |
| `V2_aprbs_Ylow` | `2.8668e-06` | `2.7270e-06` | `1.5035e-06` | **`1.814`** |
| `V3_ysweep_Yp10` | `2.4097e-06` | `1.5099e-06` | `1.3561e-06` | `1.113` |
| `V4_lissajous_Ym10` | `2.4164e-06` | `1.5662e-06` | `1.3745e-06` | `1.139` |

Excluding `V2` the untrained shift is `+10.36 %`; including it, `+15.90 %`. `V2` is the broadband
record at the extreme `Y`, i.e. the one that most excites mismatched physical parameters, and it is
the one A0 cannot train down at all.

**A0 is the best of the three BLA arms and still underperforms the server's worst.**
`ablation_a0_clean.json` gives `1.9049812841184182e-06`, `ratio_A = 1.0002122`:

| arm | trained | improvement vs `2.534187e-06` | ratio |
|-|-|-|-|
| A0 control | `1.904981e-06` | **`24.83 %`** | `1.0002x` |
| A1 random, Orvieto full disc | `2.015236e-06` | `20.48 %` | `1.0056x` |
| A2 BLA-fitted | `2.038484e-06` | `19.56 %` | `0.9766x` |

A0's `24.83 %` is below the server plateau's `36.07 %`. **Corrected, third audit: this is consistent
with the detune but is not independent confirmation of it**, because the two nights also differ in
`nx_aug` (2 against 8), epochs (4 against 2) and harness, as 9.10(c) itself records two paragraphs
below. Treating it as confirmation contradicted that paragraph. What it does establish on its own is
the weaker and still useful point that **fractions are not comparable either**, which section 4's
"compare fractions" instruction assumed they were.

One cross-night comparison does survive and is worth keeping: A1's `1.0056x`, a full-disc Orvieto
draw, reproduces `F4c_orvieto_default`'s `1.0255x`. Both are inert, on different harnesses, as 9.3
requires.

**Section 4's list of differences is still incomplete**: `a2_spec_clean.json` has `nx_aug = 2` while
every factorial arm ran `CL_NX_AUG 8` (`wave2_77991_11.out`, `[capacity] ablation built at
nx_aug=8`), and the nights differ in epochs, 4 against 2.

### 9.11 5.9 sharpened, 5.10 reversed, and a rate ambiguity nobody has flagged.

**5.9 is right and can be made specific.** The Hankel spectrum is `3.679e-3, 3.059e-3, 1.466e-3,
1.227e-3` then `3.5e-5`: **four significant states, i.e. two modes.** `bound_2sum_disc` is `5.870e-3`
at order 2 and `4.845e-4` at order 4, against `eps = 8.154e-3`. Order 4 passes with `17x` margin and
would have kept both modes; `nx_aug = 2` holds only one pair. **The binding decision was an
order-selection rule preferring the smallest admissible order, not the H-infinity criterion as
such.** Conjecture, settled cheaply by re-running `fit_reduce` with the order forced to 4 and reading
`poles_report`.

**5.10's recommendation should be reversed.** "Use the residual fit to set the band, keep the
spanning draw" keeps the insurance and discards the information. The pole gate already found
`157.710 Hz, zeta 0.05208` on noisy differenced data at `na = 28`, which is
`d(lambda*, lambda_fit) = 0.025`, better than any single drawn pole in any of the six seeds. What
follows from 9.3 is: fit the residual, do **not** reduce to the smallest admissible order, install the
identified pair directly, size any neighbourhood as half-width `zeta_hat * f_hat` from the estimate
itself rather than from a peak-picker, keep `B_a` (9.9), and leave `W^a` at zero (9.8).

**This changes the Telica prospect**, which is 7.4's reason for caring. 7.4 argues that (b) "needs
some other way to decide where the band is". Under 9.3 the band is not a free choice: its half-width
is `zeta_hat f_hat` for whatever the residual does contain inside the identifiable range, below
`83 Hz` on X and `55 Hz` on Y. A neighbourhood radius is derivable from any pole estimate, including
a poor one, because the radius scales with that estimate's own damping. The `10 dB` peak-pick is not
load-bearing. This answers the band half of 7.4's "the two best components have the weakest
citations"; it does **not** rescue `AUG_LRU_FREEZE`, which remains uncited.

~~**A rate ambiguity in `fit_reduce.json`.** ... If the regression ran at `fs_eff = 1000` like the
gate, the identified pole is at `1.2547 Hz` and installing its raw angle at `4000 Hz` is a `4x`
frequency error.~~

**WITHDRAWN, third audit. There is no rate ambiguity.** `fit_reduce.py:302` sets
`ts_fit = cfg.ts_new * d8.DEC` (`1/1000 s`) and `ts_model = cfg.ts_new` (`1/4000 s`), and
`rate_convert` at line 150 performs an explicit ZOH conversion between them via `logm`/`expm`, with
a refusal on the closed negative real axis. `f_exc_measured_hz = 2000.0` is measured on the original
`4 kHz` APRBS record and is report-only; it is not the regression rate, and inferring the regression
rate from it was invalid. The conjecture is deleted rather than softened.

### 9.12 Predictions for tasks 11 and 13, and they agree with 8.1 for a different reason.

Written before inspecting the ablation output; at the time `wave2_77991_11.out` and
`wave2_77991_13.out` contained only intact RMS, `3.831160e-07` and `1.397292e-06`.

* **Task 11, `F3b_arm2_wa_zero`: `5.10x` to `5.30x`.** Bit-identical poles and `B_a` to
  `F3c_arm2_wa_frozen` (`5.1551x`); only `W^a` trainability differs, and zeroing the trained `W^a`
  costs `1.0002x` in `F5` and `1.0000x` in `F3c`. (The prediction originally leaned on 9.8's
  factor-`178`, which is withdrawn; it now rests on the two measured `W^a`-zeroing costs.)
* **Task 13, `F3a_F1_wa_random`: `1.000x` to `1.005x`.** `F1` has no `AUG_LRU` block and therefore no
  basis poles, so `|B(lambda*)| = 1` by construction and 9.3 gives no path at all.

Outcome table, weakened by the third audit: the original wording claimed each off-prediction result
would refute a mechanism. Most would only bound one.

| result | task 11 | task 13 |
|-|-|-|
| near `1.0` | would show that `W^a` trainability is decisive despite costing `1.0002x` post-hoc in `F5`, and would be incoherent with `F3c` at bit-identical poles and drive. It would **not** by itself refute any gain argument, since gain was never shown to determine recruitment | as predicted, nothing new |
| `1` to `2` | something other than pole and drive geometry gates the block; 9.4's categorical separation weakens | a weak alternative latent route exists in the plain ANN-written rows. This **bounds** 9.3 rather than refuting it, since 9.3 is about `AUG_LRU` poles and `F1` has none |
| above `5` | as predicted; 9.3 and 9.7 hold and 5.5 stands with both halves measured | would be a genuine surprise and would reopen 5.1, since `F1`'s own `1.0000x` denies it |

### 9.13 What would overturn section 9

* **An arm with one pair at `(rho*, theta*)` and the seed-0 `B_a` that reaches the plateau rather
  than `e-7`.** Refutes 9.7's claim that `|B|` is a ceiling and 9.3's sufficiency reading of the
  neighbourhood. It is also the cheapest arm in the programme and it is the training-side complement
  of the offline `2 x 2` fit proposed in 7.1.
* **Any arm whose poles all sit at `d_k >= 0.9` and whose ablation ratio exceeds about `1.2x`.**
  Eleven pairs currently say this cannot happen; one counterexample breaks 9.4 and 9.5.
* **7.1's `2 x 2` feature-fit showing that four identical-frequency pairs with independent `B_a`
  match the spread bank.** That would not refute 9.4, which is measured at fixed `B_a`, but it would
  refute 9.7's ordering of the two conditions and make the drive the whole story after all.
* **A within-band statistic predicting the seed spread at `n > 12`.** 9.6 is a negative at `n = 6`
  and it is not strong. If the spread turns out to be pole geometry, 9.7's "ceiling no longer
  binding" reading fails.
* **A `V2_aprbs_Ylow` split that is not explained by the `0.9/1.1` detune.** 9.10's corroboration of
  7.3 would then be coincidence and the cause would still be open.
* **A per-`Y` residual fit on Telica showing a mode that moves.** 9.1 is an identity about the
  simulation's own equations and cannot fail there. On Telica it can, and 9.1 says nothing about
  Telica.

### 9.14 Literature status for the surrogate. The canonical source was on disk the whole time.

One targeted search was run, on the named mechanism only, per D-121. No general sweep on pole
placement, reduction or augmentation was opened. Nothing below has been added to `EVIDENCE.md`.

**The quantity has a name in control, and it is not "pseudohyperbolic".** It is the **Kolmogorov
measure**, and the product form is the **Blaschke product modulus**. `OpenAlex` returns 2 works for
`"pseudohyperbolic" "identification"`, both PDE inverse problems. Any future sweep phrased on
"pseudohyperbolic" will return nothing and read as novelty.

**Held, and verified `MATCH OK` with `verify_pdf_quote.py`:**

* `literature/books/Toth_2010_[12]_LPVModelingIdentificationBook.pdf`, **eq. (2.61), PDF p65**:
  `kappa_ng(z, Lambda) := |G_b,Lambda(z^-1)| = prod_j |z - lambda_j| / |1 - z lambda_j*|`. That is
  the object in 9.3, under the name Kolmogorov measure. Eq. (2.10) PDF p45 is the Blaschke product;
  Proposition 2.1 PDF p64 is the `n`-width theorem with worst-case error proportional to
  `rho^(n_e+1)`.
* Tóth, Heuberger, Van den Hof, *Asymptotically optimal orthonormal basis functions for LPV system
  identification*, **Automatica 45(6):1359-1370, 2009**, DOI `10.1016/j.automatica.2009.01.010`,
  free at TU/e Pure. **p4**: *"the distance between basis poles and the original system poles
  determines the convergence rate of the coefficients"*, and **p5**: *"one cannot improve on the
  worst-case error by adding new poles to the n_b basis poles."*

**Two narrowings the search itself imposed**, and they are why 9.3 is labelled a surrogate:

1. The literature states a **bound and a proportionality over a region**, never an equality for a
   single target pole. The equality in 9.3's derivation is the project's own and must be carried as
   a derivation, not a citation.
2. **The `zeta f0` radius has no source at all.** Zero hits on arXiv and OpenAlex. It is the
   conjecture flagged in 9.3.

**What this changes for 7.4 and 5.10.** Tóth 2010 **Chapter 8**, PDF pp. 219-242, is basis-pole
selection when the system poles are unknown: Definition 8.1 the Kolmogorov measure, Algorithm 8.1
the Fuzzy-Kolmogorov c-Max clustering, applied to a **sample pole cloud** obtained from data
(Sect. 2.4.6, "Pole Uncertainty of Model Estimates"). Our 54 scattered peak estimates are exactly
such a cloud. So the `10 dB` peak-pick has a published replacement with a min-max optimality
criterion, and 7.4's "the band is the component with the weakest citation" is answerable. This does
**not** answer whether a Telica residual contains a target worth clustering toward; see the
amendment to 5.10.

R. Toth is a supervisor on this thesis, and the two verified sources are his.

**Priority-4 negative, enumerated rather than asserted.** Whole-of-arXiv abstract counts, with two
positive controls in the same session (`abs:"linear recurrent unit"` = 18,
`abs:"orthonormal basis functions" AND abs:"identification"` = 8):
`"orthonormal basis functions" AND "recurrent"` = 0; `"rational orthonormal basis" AND "neural"` = 0;
`"linear recurrent unit" AND "basis"` = 0; `"Blaschke" AND "system identification"` = 0;
`"fixed poles" AND "neural network" AND "identification"` = 0. OpenAlex
`"orthonormal basis functions" "neural network"` = 22, all 22 off-target. **No source connects the
fixed-pole rational basis literature to an LRU or deep state-space architecture.** Google Scholar
returned empty twice and is recorded as unusable, not as zero, so grade this negative strong but not
exhaustive.
