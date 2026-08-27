# What creates the offset when the baseline replays MSD data open loop (2026-07-29)

**Question.** Input `u_total` is recorded from the closed-loop simulation of the plant WITH the
hidden MSD, then replayed open loop through the baseline (no MSD). An offset appears. The
augmentation is set up for a frequency-domain difference (the ~158 Hz mode the absorber adds),
so what produces the offset was not clear.

**Answer, in one line.** X and Y have different causes. The X offset is the `L0` constant
inertia error, a genuine parameter mismatch. The Y offset, which is ~100x larger, is the
absorber momentum that a 6-state model has no state to hold, and it is exactly
`ma*vdelta_a(t0)/cy`. Neither is the absorber's dynamics, and neither is what is blocking the
project.

**Status of this document.** No training was run. Everything below is a deterministic
open-loop simulation or a closed-form check, so seed counts do not apply in the usual sense;
record counts are stated per result instead. Section 8 lists what this corrects elsewhere and
section 9 states an unresolved provenance gap.

---

## 1. The mismatch, exactly

Subtracting the baseline 3x3 mass matrix (`gantrySystem.m`, full `mh`) from the truth's upper
3x3 block (`gantrySystemExtended.m`, `mh_rigid + ma`) leaves exactly two nonzero entries:

```
dM(1,2) = dM(2,1) = -ma*(L0 + delta_a)
dM(2,2)           =  ma*[2*Y*(L0 + delta_a) + (L0 + delta_a)^2]
```

plus the absorber columns `M(2,4) = -ma*d`, `M(3,4) = ma`, `M(4,4) = ma` with `ka`, `ca`.
`dM(1,1)`, `dM(2,3)` and `dM(3,3)` are identically zero: the mass split conserves total mass,
so at DC the Y axis sees the same inertia. That is why "the absorber only changes the frequency
response" is a reasonable prior, and why the offset needed explaining.

Three candidate mechanisms live in that expression:

| tag | term | can it produce a DC? |
|-|-|-|
| LIN | absorber columns + `ka`/`ca` | linear, constant-coefficient. Zero-mean in, zero-mean out |
| CONST | the `L0`-only part of `dM` | a parameter error. Residual `dM*qddot` is zero-mean |
| BILIN | the `delta_a`-dependent part | quadratic, so it rectifies band energy into DC |

BILIN was the first hypothesis and it is wrong. See section 4.

## 2. Parameters, confirmed from the data rather than assumed

The mat files store no absorber parameters (verified: no `ma`, `ma_frac`, `ka`, `ca`, `L0`,
`fa`, `zeta_a`, `mh_rigid`), so `gantry_dynamic/oracle.py` hardcodes them. Fitted instead, on
V1, oracle seeded from the true state and driven by the recorded `u_total`:

| `ma_frac` | RMSE(Y) [m], 2 s | `delta_a` RMSE / RMS |
|-|-|-|
| 0.050 | 2.40e-04 | 0.361 |
| 0.075 | 1.19e-04 | 0.188 |
| **0.100** | **2.37e-06** | **0.005** |
| 0.125 | 1.24e-04 | 0.194 |
| 0.150 | 2.45e-04 | 0.387 |
| 0.200 | 4.87e-04 | 0.741 |

Sharp minimum, two orders deep. `L0` is confirmed the same way on X: RMSE(X1) is 2.28e-06 at
`L0 = 0`, **2.60e-07 at `L0 = 0.10`**, 1.75e-06 at `L0 = 0.20`. Y is insensitive to `L0`, which
is what the algebra says since `L0` enters only the X-Theta block. 1 record.

## 3. X: the `L0` constant inertia error

Logical X row, truth variant minus baseline, 12 s, seeded at t = 0.02 s.

| arm | V1 mean | V1 @11.9s | T10 mean | T10 @11.9s |
|-|-|-|-|-|
| FULL | -2.325e-05 | -2.652e-05 | +5.096e-04 | +7.145e-04 |
| `delta_a` frozen in `M` | -2.423e-05 | -2.932e-05 | +5.093e-04 | +7.134e-04 |
| absorber IC `= 0` | -2.327e-05 | -2.657e-05 | +5.011e-04 | +7.026e-04 |
| **`L0 = 0`** | **+2.031e-08** | +5.159e-08 | **+9.297e-06** | +1.312e-05 |
| `L0 = 0` and IC `= 0` | +4.856e-09 | +9.559e-09 | -5.850e-08 | -1.127e-07 |

It IS a settled offset (86 to 97 percent DC). Zeroing the absorber initial condition changes it
by 0.1 to 2 percent. Zeroing `L0` removes 1145x (V1) and 55x (T10).

`L0 = 0.10 m` puts `-ma*L0 = -0.101 kg*m` on the X-Theta coupling (3.1 percent of the baseline
entry at `Y = 0.3`) and `+0.0707 kg*m^2` on the Theta inertia (1.5 percent).

**Mechanism.** `dM_const * qddot` is zero-mean but has a nonzero time integral. On an axis with
a pole at `s = 0`, `x(s) = f(s)/(s(ms+c))`, so `x(inf) = (integral of f dt)/c`. The final
position offset is set by the residual's IMPULSE, not by its mean. This is why the offset
coexists with D-A's finding that the residual is zero-mean to `|mean|/rms ~ 1e-04`. Both are
true and they are about different functionals.

2 records (V1, T10); the same ablation on T3 and T6 gives `L0 = 0` retaining 0.0 to 4.7 percent.

## 4. Y: the absorber initial condition, and nothing else

All four truth variants give an IDENTICAL Y offset to four significant figures (`+2.314e-03 m`
mean on V1). Neither `L0` nor the `delta_a`-dependence of `M` affects it at all.

Seeding the truth with `delta_a = vdelta_a = 0`, keeping all six physical states identical:

| seed time [s] | true `delta_a`, `vdelta_a` | `delta_a = vdelta_a = 0` |
|-|-|-|
| 0.02 | +2.526e-03 | -5.670e-07 |
| 0.60 | -3.841e-03 | -5.671e-07 |
| 1.00 | +1.584e-03 | -5.670e-07 |
| 2.00 | +8.202e-04 | -5.670e-07 |
| 4.00 | +4.320e-05 | -5.670e-07 |
| 8.00 | +9.698e-04 | -5.672e-07 |

**4450x drop**, and it becomes independent of seed time. With the true absorber state the offset
swings an order of magnitude and changes sign with seed phase, which is an initial condition,
not an accumulating force.

**Closed form.** At the seed instant the truth carries absorber momentum `ma*vdelta_a(t0)` that
a 6-state model has no state to hold. On a `K = 0` axis an initial momentum excess `p` leaves a
permanent offset `p/c`, since `integral(v dt) = (p/m)*(m/c) = p/c`, independent of `m`:

```
dY(inf) = ma * vdelta_a(t0) / cy
```

| `t0` [s] | `vdelta_a(t0)` [m/s] | `ma*vdelta_a/cy` [m] | observed dY [m] | ratio |
|-|-|-|-|-|
| 0.02 | +2.5016e-02 | +2.5266e-03 | +2.5260e-03 | **1.000** |
| 0.60 | -3.8027e-02 | -3.8407e-03 | -3.8412e-03 | **1.000** |
| 1.00 | +1.5690e-02 | +1.5846e-03 | +1.5841e-03 | **1.000** |
| 2.00 | +8.1263e-03 | +8.2075e-04 | +8.2015e-04 | 0.999 |
| 4.00 | +4.3347e-04 | +4.3780e-05 | +4.3198e-05 | 0.987 |
| 8.00 | +9.7948e-03 | +9.8927e-04 | +9.6981e-04 | 0.980 |

Consistent with a first-order settle at `tau_Y = mh/cy = 1.010 s`: the 2 s value is 0.861 of the
final one against a predicted `1 - exp(-2/1.01) = 0.862`. 1 record, 6 seed phases.

## 5. Rectification is NOT the mechanism

Freezing `M` at `delta_a = 0`, which removes the bilinear `delta_a*qddot` and `delta_a^2` terms
entirely, changes nothing:

| record | FROZEN keeps, X | FROZEN keeps, Y |
|-|-|-|
| V1 standstill | 104.2 % | 100.00 % |
| T3 standstill | 104.7 % | 100.00 % |
| T10 aprbs 60 | 99.9 % | 100.00 % |
| T6 ysweep slow | 112.1 % | 100.00 % |

The reasoning that `<delta_a*qddot> != 0` is sound in principle, but the term is too small to
matter: `delta_a` has an RMS of 22 micrometres against a moment arm of 0.4 m. 4 records.

## 6. It requires the MSD, but only its hidden state

| configuration | dY over 12 s |
|-|-|
| no MSD at all (baseline model vs baseline-only records) | ~1e-08 m |
| MSD dynamics present, absorber seeded at rest | 5.7e-07 m |
| MSD present, absorber seeded correctly | 2.5e-03 m |

The genuine dynamic mismatch contributes 0.02 percent of the offset. First row is 13 records
(`trajectory/augmentation/baseline/`), whose multisine is 1 to 7 Hz rather than 130 to 180 Hz;
see section 8.

**Closed loop hides it.** `ruleOfThumb` is `(s + 2*pi*fbw/6)/s * leadlag * lowpass`, i.e. a pole
at the origin, so the loop regulates the offset away. Opening the loop reveals it rather than
creating it.

## 7. Why this is not the blocker

V1, Y channel, 12 s free run, from `71167/gantry_results_71167.npz`:

| arm | Y RMS [m] |
|-|-|
| baseline, true 6-state `x0` at K0 | 7.33e-04 |
| baseline, **encoder-init** `x0` | **2.11e-04** |
| oracle, perfect 8-state `x0` | 9.02e-05 |
| after ONE epoch of ANN training (pooled V1-V4) | 2.11e-02 |

The encoder is already 3.5x better than seeding with the true physical state. It does this
WITHOUT estimating the absorber: `x_enc_ann` in that file is identically zero over all 48000
samples, so it compensates by biasing the `dY` estimate instead. Total headroom from perfect
8-state knowledge is **2.3x**. Training costs **127x**, with `loss_train[0]` and `loss_train[1]`
bit-identical at 1.36860286e-06.

**Consequence for the augmentation structure.** No parallel force block can supply a missing
initial condition, which is a sharper version of "the current structure does not address this"
than a frequency-content argument. But the offset is 2.3x from its floor and training is 127x,
so the offset is explained, not blocking.

**Observability, recomputed at current parameters.** Frozen-point PBH on the scaled system,
swept over `Y` in [-0.35, 0.35] and `delta_a` in [-1e-4, 1e-4]: worst `sigma_min = 1.12e-05`,
so the pair is structurally observable everywhere. But over the pipeline's encoder window
(`na+1 = 18` samples at 4 kHz = 4.5 ms = 0.71 absorber periods) `vdelta_a` produces 26,000x
less output energy than `Y`, and `cond(O) = 4.6e+05`. Least-squares estimability of
`vdelta_a` against its true RMS of 2.18e-02 m/s:

| noise | N=18 | N=80 | N=1200 |
|-|-|-|-|
| noiseless (float32 `y` storage) | 0.43 % | 0.14 % | 0.13 % |
| SNR 60 dB | 50x | 16x | 15x |
| SNR 55 dB | 89x | 29x | 26x |
| SNR 50 dB | 161x | 52x | 47x |

So on the current noiseless data the absorber state is recoverable (achievable `r2` about
0.99998 against the trained `r2_aug_lin` of 0.344 / 0.350), and at the project's own noise
levels it is not, by 15 to 160x, with no window length fixing it.

## 8. What this corrects elsewhere

1. **`simulations/gantry_subnet/diagnostics/system_dynamics.json`** records `msd.obs_rank = 3`
   of 8. Superseded: at current parameters the system is structurally observable. That file is
   dated 2026-06-16 and describes a 421.6 Hz absorber.
2. **`Matlab-scripts/Augmentation/diagnostics/PBH_observability_test_MSD.m`** uses `ka = 500`,
   `ca = 2`, i.e. a 3.5 Hz absorber, not `ka = ma*(2*pi*150)^2 = 8.97e+05`. Stale.
3. **`docs/status-overview-2026-07-27.md` section 5 item 4** validates `ma_frac = 0.10` via
   `150*sqrt(1 + ma/mh_rigid) = 158.1 Hz`. That formula is not a valid predictor here (it
   ignores the coupling to Theta and the crossarm); the measured `delta_a` peak on V1 is
   164.55 Hz and the linearised model eigenvalue is 157.89 Hz. The `ma_frac` fit in section 2
   is the decisive test and it agrees with 0.10 anyway.
4. **`gantry_dynamic/oracle.py` docstring** claims the oracle "reproduces y to discretization
   error", citing `oracle_vs_data_V1.npz`. That artifact holds RMSE/RMS of 0.110 / 0.128 /
   0.449, and it was run at `START = 0` (the D-087 contaminated seed its own script warns
   about) and `up_sample = 2` while the entry file uses 1. Seed choice alone moves RMSE(Y) 800x
   (5.51e-05 at index 0, 6.66e-08 at index 100). Regenerate at an interior seed.
5. **`docs/drift-diagnosis-status.md` section 4 side finding**, "even the true baseline driven
   open loop by a zero-mean multisine has a small constant drift (~1e-4 m/s)": not reproduced.
   Baseline model vs baseline records gives ~1e-08 m TOTAL over 12 s on 13 records. Caveat:
   those records carry a 1 to 7 Hz multisine, so this is not a like-for-like refutation.
6. **D-A's attribution of the residual DC to `M(Y)` rectification** (`drift-diagnosis-status.md`
   section 5) is not supported by the freeze ablation in section 5 above. D-A's measurement of
   the residual is not in question; the attribution of its DC to rectification is.

## 9. Provenance, and one gap

Numbers came from `oracle_check2.py`, `msd_offset_mechanism.py`, `msd_offset_y.py`,
`x_channel.py`, `observability.py`, `downsample_baseline.py`, `normcheck.py`, plus direct reads
of `71167/gantry_results_71167.npz` and `oracle_vs_data_V1.npz`, and an inspection of the
unzipped `gantry_additional_state_2025a.slx` XML.

**Gap:** those scripts were written to a session scratchpad and print to stdout rather than
writing artifacts. That is precisely the failure mode `status-overview-2026-07-27.md` section
4.1 documents. They should be moved into `scripts/gantry/msd-offset/` and made to write their
numbers, with horizons and units, to `simulations/gantry_subnet/diagnostics/`, before anything
here is cited as settled.

## 10. Open, and adjacent fixes made

* **Open:** why the linearised absorber sits at 157.89 Hz while the recorded `delta_a` spectrum
  peaks at 164.55 Hz, a 4.2 percent gap. The `ma_frac` fit is two orders deep so the parameters
  are not in doubt.
* **Fixed 2026-07-29, needs regeneration:** `gtd_run_simulation.m` read `delta_a` from the
  Simscape Multibody block (SID 47) while `q_aug` comes from the Extended ODE (SID 88), so every
  record paired a trajectory from one plant with a hidden state from another. Now reads
  `delta_a_ode`. The two agree to 0.50 percent at 4 kHz and to display precision at 20 kHz on
  V1, so nothing above is invalidated, but `vdelta_a` is the signal the section 4 result rests
  on. Regeneration should reproduce `y` and `u_total` bit-identically.
* **Not fixed:** `load_mat_aug` discards the stored `vdelta_a` and recomputes a backward
  difference at 4 kHz. On a 158 Hz signal consumed as an instantaneous value that is up to
  12.4 percent error. One line.
* **Not logged:** neither the `delta_a_ode` change nor the pending normalisation-source change
  has a `docs/decisions.md` entry.
