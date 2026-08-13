# Result of the gate

Run: `verify_controller.py`, 2026-08-12. Figure: `figures/controller_formula_check.png`.

## Verdict

**The closed-form controller of `DERIVATION.md` section 2 is MATLAB's `Cfb`.** Built in Python
from the physical parameters, `P`, `f_bw` and `ts` alone, with no MATLAB object and no exported
matrices, it reproduces the stored `u_fb` of the closed-loop records to `4.5e-09` relative in the
best-conditioned channel.

## Numbers

Applied to the stored `r_sim - y` and compared against the stored `u_fb`:

| record | ch | `u_fb` rms [N] | residual rms [N] | rel. broadband | rel. `<= 200 Hz` | float32 step on `y` [m] |
|-|-|-|-|-|-|-|
| `V1_standstill_Yp10` | X1 | 28.63 | 9.01e-07 | 3.147e-08 | **4.533e-09** | 9.09e-13 |
| `V1_standstill_Yp10` | X2 | 29.17 | 9.03e-07 | 3.095e-08 | **4.692e-09** | 9.09e-13 |
| `V1_standstill_Yp10` | Y  | 24.28 | 6.90e-01 | 2.841e-02 | **4.775e-04** | 7.45e-09 |
| `T10_aprbs_60` | X1 | 157.59 | 1.78e-01 | 1.128e-03 | **1.268e-04** | 7.45e-09 |
| `T10_aprbs_60` | X2 | 161.03 | 1.83e-01 | 1.134e-03 | **1.276e-04** | 7.45e-09 |
| `T10_aprbs_60` | Y  | 80.37 | 2.61e-01 | 3.246e-03 | **2.052e-04** | 2.98e-08 |

Normalisation gains, `K_j = 1/|sys_jj(iw) Cnorm(iw)|`:

| `Y_op` [m] | `K_X1` | `K_X2` | `K_Y` |
|-|-|-|-|
| 0.10 (V1) | 2.0632e+07 | 2.4163e+07 | 1.1556e+07 |
| 0.00 (T10) | 2.1764e+07 | 2.2298e+07 | 1.1550e+07 |

The `Y_op` dependence is real but small, about 5 % on X1 and 8 % on X2 between the two design
points, and under 0.1 % on Y. It enters only through `|sys_jj(i w)|`, as section 2 predicts.

## Why the residual is storage and not formula

**The mechanism.** `gtd_run_simulation.m:33` computes `u_fb = lsim(Cfb, r_sim - q_with)` with
`r_sim` and `q_with` in **double**, and `gtd_save_record.m:25-31` then stores `u_fb`, `y` and
`r_sim` as **single**. Python can only reconstruct `e` from the stored single values, so it drives
the controller with a quantised error while the stored `u_fb` came from an unquantised one. The
difference is a roughly constant bias in `e` of order one float32 step, and `Cfb` has a pole at
`z = 1`, so that bias is **integrated into a ramp** rather than appearing as broadband noise.

**The evidence.** Fitting a straight line to each residual:

| record | ch | res rms [N] | ramp explains | slope fitted [N/s] | slope predicted [N/s] |
|-|-|-|-|-|-|
| V1 | X1 | 9.01e-07 | 3.21 % | +4.66e-08 | n/a |
| V1 | X2 | 9.03e-07 | 0.02 % | -3.66e-09 | n/a |
| V1 | Y | 6.90e-01 | **99.99 %** | +1.991e-01 | **+1.718e-01** |
| T10 | X1 | 1.78e-01 | 27.57 % | +2.69e-02 | n/a |
| T10 | X2 | 1.83e-01 | 27.56 % | +2.77e-02 | n/a |
| T10 | Y | 2.61e-01 | **81.08 %** | +6.78e-02 | **+5.03e-02** |

The prediction is `kappa_j * w/54 * mean(e)`, the controller's integral gain times the measured
bias, with no fitted quantity. On the two ramp-dominated channels it matches the fitted slope to
14 % and 35 %. It is listed as n/a where the ramp fraction is small, because there `mean(e)` is
dominated by real signal content rather than by rounding bias and the formula does not apply.

V1 X1 and X2 are the exception that confirms the mechanism: their stage positions sit at zero, `r`
is exactly zero, and the float32 step is `9.09e-13 m`, four decades smaller. With almost no bias to
integrate, those two channels show only the broadband term, `9.0e-7 N` against `1.0e-6 N` predicted
from `step/sqrt(12)` times the rms controller gain.

**Correction to an earlier version of this file.** It argued that the residual is broadband
quantisation because the spectra look flat. That was wrong on four of the six panels. The spectra
looked flat because `scipy.signal.welch` defaults to `detrend='constant'`, which removes each
segment's mean and smears a ramp into a broad featureless floor. The figure now uses
`detrend=False` and overlays the fitted ramp.

**This floor is a property of the archive**, not of the controller, and it grows linearly with
record length because it is an integrated bias. It is removed entirely by the L4 test below.

## Exactness test against a double-precision export

`export_controller.m` (MATLAB, no Simulink, no record touched) exports `Cfb` itself in double:
per-channel `tf` coefficients, `ss(Cfb)` as `(A,B,C,D)`, the design scalars, and a deterministic
`e_test` with MATLAB's `lsim` response `u_test`. `test_controller_exact.py` then runs three
levels. This removes the single-precision storage floor of the record-based check above.

| level | compares | worst rel. err | tol | verdict |
|-|-|-|-|-|
| L1 coefficients | `C_j(z)` num/den, poles, zeros, `kappa_j`, `sys_jj(i wb)` | 9.586e-12 | 1e-10 | PASS |
| L2 realisation | MATLAB's `(A,B,C,D)` run in Python vs `u_test` | 1.458e-16 | 1e-11 | PASS |
| L3 end to end | our `num/den` from the formulas vs `u_test` | 1.137e-09 | 1e-7 | PASS |
| L4 record, num/den | our `num/den` vs MATLAB `lsim`, **same input bits** | 4.662e-10 | 1e-7 | PASS |
| L4ss record, same realisation | MATLAB's `(A,B,C,D)` in Python, **same input bits** | **1.898e-16** | 1e-11 | PASS |

**L4 is the machine-precision version of the record gate.** `export_record_reference.m` re-runs
MATLAB's own `lsim` on exactly the signal Python forms, `double(r_sim) - double(y)` built from the
stored single values, so both sides consume identical input bits. No Simulink, no record modified.
On the same record and the same channel where the stored comparison gives `2.760e-02`, this gives
`1.6e-16`:

```
V1_standstill_Yp10
    our num/den vs MATLAB lsim, same input   [5.891e-11 8.502e-11 4.320e-11]
    MATLAB (A,B,C,D) in Python, same input   [1.358e-16 1.238e-16 1.625e-16]
    for contrast, vs the STORED u_fb         [7.552e-08 6.889e-08 2.760e-02]
```

Fourteen orders of magnitude, with nothing changed but the precision of the input. That is the
proof that the residual documented above is the archive and not the reconstruction.

Design scalars, `Y_op = 0.10` m:

```
sys_jj(i wb)  MATLAB [1.4414159425e-07 1.2307872111e-07 2.5735466382e-07]
              python [1.4414159425e-07 1.2307872111e-07 2.5735466382e-07]   rel 3.26e-14
kappa         MATLAB [2.0632079796e+07 2.4162916607e+07 1.1555807189e+07]
              python [2.0632079796e+07 2.4162916607e+07 1.1555807189e+07]   rel 6.81e-14
```

**What the three levels together establish.** L2 at machine epsilon shows the arithmetic is exact
when both sides run the same realisation. L3 is seven decades looser, and L1 shows that gap is not
the formula: it is `Cfb = ss(Cfb)` at `gtd_build_plant.m:28`, a 9-state canonical realisation of a
transfer function with numerator coefficients of order `2.7e6` and a pole at `z = 1`, so the
conversion costs about seven digits. For bit-level agreement with the applied force, use the
exported `(A,B,C,D)` rather than the transfer function.

Thresholds are set by conditioning, not by machine epsilon: the L1 numerator is formed from
products of coefficients of order `kappa*10*w` followed by a bilinear transform, so `1e-11` is the
achievable floor and the tolerance sits a decade above it.

**Figure:** `figures/controller_exactness.png`, four rows, one confirmation each: frequency
response overlay, relative difference of the two responses, pole-zero map with the unit circle,
and a time-domain window.

Read row 2 with care. The relative difference of the two frequency responses rises to `1e-9` at
DC and past `1e-4` at Nyquist, and **neither rise is a disagreement**. `C_j(z)` has a pole at
exactly `z = 1` and a zero at exactly `z = -1`, so evaluating `den(z)` near DC and `num(z)` near
Nyquist cancels to zero and the relative difference there measures polynomial-evaluation
conditioning. Those two regions are shaded in the figure. In the clean middle band the two
responses agree at `1e-11` to `1e-12`, consistent with the coefficient result. The coefficient
comparison is immune to this because it never evaluates the polynomial, which is why L1 is the
level the claim rests on.

`u_test` reaches about `3000 N` rms, above the `2000 N` peak in `cfg.lim`. That is deliberate, the
step segment is there to drive the integrator, but it means `u_test` is not a physically
realisable force and must not be reused as an excitation elsewhere.

## Scope, and what this does not establish

- **It gates the controller, not the plant.** `G` inside `So = (I + G Cfb)^-1` and
  `Si = (I + Cfb G)^-1` is the frozen, rigid-nominal, `Y_op`-dependent discretised model from
  `gtd_build_plant.m`, not the 8-state plant that actually produced `q`. The error identity
  `e_cl = So Delta So_hat w` in `DERIVATION.md` section 4c is exact, but the sensitivity numbers
  quoted with it are evaluated on that design plant, so they describe the loop as designed rather
  than the loop as realised on the true 8-state system.
- **`Y_op` is not stored in the records.** It was read from the declarative table in
  `gtd_build_records.m` (`V1_standstill_Yp10` at 0.10, `T10_aprbs_60` at 0.00). Any new record
  needs that lookup, or the field should be added to `gtd_save_record.m`.
- Two records were checked, both from `trajectory/augmentation/`. The controller is frozen per
  record, so a record at a different `Y_op` re-derives `K_j` but not the structure.
