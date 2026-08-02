# True-init augmentation: can the ANN learn the absorber at all?

Autonomous session, 2026-08-02. Brief: `tasks/handoffs/2026-08-02-true-init-augmentation.md`.
Written as the work happened, including the parts that did not work.

---

## 0. Read this first

### What was run

Six diagnostics and three training arms, all on the frictionless `augmentation` dataset at
the pipeline's own 4 kHz, with two changes against the current pipeline and nothing else:
the SUBNET encoder is replaced by the truth's six physical states with **analytic**
velocities (integrator states of a 20 kHz RK4 replay of the 8-state truth from the rest IC,
not `gradient()` finite differences), and the baseline carries the truth's static mass
distribution at `delta_a = 0` (a CoG-corrected LFR block, gates C1a-C5 all pass). The
model's augmented rows 6-7 start at zero and are never seeded, per handoff section 8.

### The numbers, against the section 10 criterion

**The criterion is FAILED on five of the six physical states, and the handoff's section 5
assumption is falsified.** Per-window re-seeding from the exact 6-state IC gives, on V1:

```
state       exact seed     free-run floor     ratio     gain over the record's FD velocities
X          1.2703e-08       7.6940e-08        0.17          2.6x     PASSES
Theta      4.7217e-07       4.1339e-09      114.22          1.0x
Y          1.0278e-04       3.1469e-08     3266.24          1.0x
dX         2.2495e-07       5.7877e-09       38.87          2.9x
dTheta     8.8952e-06       1.1686e-07       76.12          1.0x
dY         2.0251e-03       3.1155e-05       65.00          1.0x
```

The free-run floors reproduce the 20 kHz figures the handoff quotes (X `7.69e-08` vs
`9.15e-08`, Theta `4.13e-09` vs `3.73e-09`, Y `3.15e-08` vs `2.98e-08`); the three velocity
floors are new and were derived the same way before being used. The pattern holds on all
four validation records (standstill, APRBS, Y-sweep, Lissajous).

**The exact velocities bought `2.6x` on X and `2.9x` on dX and exactly nothing on Theta, Y,
dTheta and dY.** The Y per-window DC scatter is `1.0278e-04 m` from the exact IC against
`1.0273e-04 m` from the record's finite differences. Those are the same number. The `3507x`
gap this experiment was meant to close by construction did not close.

### Mechanism, established four ways

1. **Regression.** Each window's mean error against the truth's `[delta_a(s), vdelta_a(s)]`:
   `R^2 = 1.0000` on Y and `0.9999` on dY, with fitted slopes matching the **unfitted**
   closed forms `-(ma/mh)*nf*Ts/2` and `-(ma/mh)` to 3.5 % and 5 %.
2. **The truth model against itself.** Re-seeded per window from its complete 8-state IC it
   collapses to the integrator floor (`Y 1.735e-08`). The same model re-seeded from the
   exact six states with the absorber zeroed reproduces the baseline's scatter to **1.000 on
   every state, to four digits.** No baseline, no CoG term and no LFR realization is involved
   in the residual at all.
3. **Observability.** The absorber state is recoverable from the encoder's 18-sample window
   at `R^2 = 1.0000` out of sample on every record class, and from the instantaneous
   `[x_phys(k), u(k)]` the ANN reads at `R^2 = 0.10` to `0.49`.
4. **Horizon.** The signal is flat at `2.19e-06 m` in-window RMS from 6 ms to 400 ms while
   the initial-condition ramp grows linearly with the horizon (`4x` the signal at `nf = 25`,
   `56x` at `nf = 400`), with `mean/RMS` pinned at the pure-ramp `sqrt(3)/2` throughout. No
   horizon both contains one absorber oscillation and keeps the IC error subdominant.

So the target's corruption is not a defect in this code and not an encoder problem. It is
that the truth has eight states and the model has six that can be initialised. **An encoder
that reconstructed the six physical states perfectly would land exactly on these numbers.**

### The training arm: it does not learn, and it does something worse

Three learning-rate arms (`1e-7`, `1e-6`, `1e-5`), killed externally at 47/47/43 of 90
epochs, so 1222/1222/1118 Adam updates instead of 2340. Recovered from the streamed logs
(`runlogs/`, `harvest_runs.py`); the per-epoch `_last` checkpoints survived because they were
written every epoch on purpose.

```
    lr  updates      ANN off         best   best %  final %  worst %  DC_Y best %   |w| final
 1e-07     1222   6.8486e-05   6.8312e-05    -0.25     1.02     1.59        -0.28    1.80e-06
 1e-06     1222   6.8486e-05   6.8037e-05    -0.65     2.16     8.54        -0.62    4.78e-05
 1e-05     1118   6.8486e-05   6.7924e-05    -0.82    -0.16   167.08        -1.40    4.17e-04
```

The best point of the best arm is `0.82 %` below ANN-off, while **91 %, 91 % and 93 % of all
validation points sit above it**. The Y per-window DC, the quantity the static analysis says
cannot be touched, improves by at most `1.40 %` over all three runs: `1.03e-04 m` before and
`1.03e-04 m` after. So P1 is a null in substance and P2 is confirmed.

**And the 12 s free run degrades by `253x` to `3575x` on the same checkpoint** (V1
`1.30e-06 -> 4.52e-03 m`). The free-run per-window DC on Y goes `3.56e-07 -> 4.24e-03 m`, a
factor `11900`: the ANN's small non-zero-mean output has nothing to relax it on a `K = 0`
row, so 48000 steps integrate what 400 steps hide.

**That last result is the one to carry forward.** The 120x train/select horizon gap has been
shown twice before with a confound each time (run 71167 had the encoder; the STAGE 1 black
box had no informed `x0`). Here the initial condition is exact by construction and the split
reproduces at full strength, so **initialisation is eliminated as an explanation of the
horizon gap.**

### Which section 8 candidate the results implicate

**Persistence, with an important qualification I got wrong at first and corrected in section
5.2b.** The augmented partition is not a static scratchpad: the ANN reads all eight state
rows (`connect_block_signals(ann_block, ["x","u"], [])` carries no selection matrix) and
writes rows 6-7, so `x_aug(k+1) = h_ann(x_phys(k), x_aug(k), u(k))` is a genuine learnable
recurrence. Gate G6's `0.000000e+00` is an **initialisation** result, not a structural one.

What is measured is that training does not leave that corner. The recurrence gain, the
largest singular value of `d x_aug(k+1)/d x_aug(k)`, is `0.0` untrained and, at the end of
the three arms, `5.16e-08`, `1.02e-05`, `1.57e-04` at lr `1e-7`, `1e-6`, `1e-5`. A lightly
damped 150 Hz absorber at 4 kHz needs `|lambda| ~ 0.99`. It is four to seven decades short
and scales with the learning rate rather than with the data. Coordinate pinning is not
implicated (rows 6-7 do end up carrying something, `|x_aug| ~ 1e-04` against exactly zero at
init, but `R^2` of the best affine map onto the truth's absorber state is `0.11-0.16`, so
there is nothing there to pin). Capacity is not implicated (the same architecture, optimizer
and budget fits a static control target to `R^2 = 0.9999`).

That sharpens the Györök `A_aug` case in a way the earlier reading did not: the object is
already in our wiring, but its gain is initialised at exactly zero, and Adam moves a weight
by at most `lr` per step, so reaching `0.99` from `0` needs of order `1/lr` consistent
updates. `A = alpha_bar * sigmoid(alpha) * A_bar` starts that gain at `0.5`. The value is in
where it starts, not only in that it exists.

### What was assumed

- That the pipeline's `oracle.py` 8-state EOM is the data-generating plant. Verified under
  D-097 and re-verified here: the replay reproduces the records' own positions to
  `5.37e-10 m` on X (V1), worst `e-9` across 22 records.
- That the absorber parameters are `ma_frac = 0.10`, `L0 = 0.10 m`, `fa = 150 Hz`,
  `zeta = 0.05`. Fitted from the data in a prior session, not re-derived here.
- That `route_ix = (0..7)` stays, per D-103. Nothing measured here forces otherwise.
- That `lr = 1e-7` is the right rate. It is the pipeline's documented value for this routing
  and the lr probe screened three rates over 10 updates only, which is too short to rank a
  2340-update run (the STAGE 1 lesson). Two arms were added to bracket it.

### What is still open

Section 8. The largest item is that nothing here tests the fix; the handoff puts it out of
scope and that is respected.

---

## 1. What was built

All new files are in `scripts/gantry/true-init-augmentation/`. Nothing under
`model_augmentation/`, `kamtin-fp-model/` or `scripts/gantry/coulomb-offset/` was touched.

| File | Role |
|---|---|
| `data_exact.py` | record loader (pipeline conventions) + the EXACT 8-state truth: a 20 kHz RK4 replay from the rest IC, decimated to 4 kHz. Velocities are integrator STATES, never difference quotients. |
| `plant_cog.py` | `Gantry_State_Block_CoG`: the LFR baseline carrying the truth's static mass distribution at `delta_a = 0`. The corrected `N0,N1,N2` and `d(Y)` are derived, not fitted. |
| `check_plant_cog.py` | gates C1a-C5 |
| `precompute_exact.py` | caches the exact truth for all 22 records |
| `diag_window_target.py` | task item (i): the per-window target check, three seeding arms |
| `diag_dc_mechanism.py` | section 4: what the residual DC actually is |
| `diag_static_representability.py` | section 5: is the correction a function of the ANN's inputs |
| `diag_absorber_observability.py` | section 5.3: window versus sample |
| `diag_aug_state_activity.py` | section 5.2b: did training leave the dead-beat corner |
| `diag_nf_sweep.py` | section 3.5: does a shorter horizon rescue the target |
| `diag_zeromean_cog.py` | section 3.4b: zero-mean on both baselines, all six states |
| `true_init_train.py` | task item (ii): the training arm, encoder replaced by the exact IC |
| `eval_freerun.py` | the 12 s free-run arm on a trained checkpoint |
| `make_figures.py` | `figures/true_init_summary.png` |

Diagnostic JSON goes to `simulations/gantry_subnet/diagnostics/true_init_*.json`, per the
project convention. The exact-truth caches under `figures/_exact_*.npz` are regenerable and
are gitignored.

### 1.1 The exact initial condition, and why it is not `x_logical`

`gtd_save_record.m:22` builds the stored velocity rows with MATLAB `gradient()`, i.e. a
central difference, and the same for `vdelta_a`. Seeding from those rows is what
`gantry_interconnect_dynamic.py:157` calls "True-x0 (oracle)". It is not exact, and on a
`K = 0` axis the error never decays.

The route taken here is to integrate the truth ourselves. The 8-state EOM is imported from
`scripts/gantry/gantry_dynamic/oracle.py` (the pipeline's own oracle, verified against the
Simulink data under D-097) rather than restated, because a second copy is a second thing to
be wrong. The one exactly known initial condition is the rest state
`[0, 0, Y_op, 0, 0, 0, 0, 0]` at `t = 0`; from there a 20 kHz RK4 replay with the recorded
ZOH input gives an exact state at every sample, and decimating by 5 lands on the training
grid. Every velocity read out afterwards is an integrator state.

This is validated rather than assumed: gate C5 compares the replayed POSITIONS against the
record's own positions, which our integration never saw.

### 1.2 The centre-of-gravity correction inside the LFR block

The coulomb-offset thread did this correction on the collapsed 3-DOF realization and noted
that the LFR block "builds M(Y) implicitly through its polynomial constants, so the CoG term
cannot be edited in one line there". It cannot be edited in one line, but it can be edited
exactly, because the correction preserves the rational structure. Writing
`A = alpha`, `B = beta - ma*l0`, `Gp = gamma + mh*d^2 + ma*l0^2` (with `gamma` excluding
`mh*d^2`, the `gantry_ss` convention), the corrected mass matrix

```
M_c(Y) = [[A,        B - mh*Y,                  0    ],
          [B - mh*Y, Gp + 2*ma*l0*Y + mh*Y^2,  -mh*d ],
          [0,        -mh*d,                     mh   ]]
```

is still quadratic in `Y`, so `adj(M_c)` and `det(M_c)` are still quadratic and the LFR
rational form `M^-1 = N(Y)/d(Y)` survives with

```
N0 = [[m*Gp - m^2*dd^2, -m*B,    -m*dd*B    ],  N1 = [[2*m*ma*l0, m^2, dd*m^2],
      [-m*B,             A*m,     A*m*dd    ],        [m^2,       0,   0     ],
      [-m*dd*B,          A*m*dd,  A*Gp - B^2]]        [dd*m^2,    0,   2*(A*ma*l0 + B*m)]]

N2 = [[m^2, 0, 0], [0,0,0], [0, 0, A*m - m^2]]
d(Y) = m * [ (A*Gp - A*m*dd^2 - B^2) + Y*2*(A*ma*l0 + B*m) + Y^2*m*(A - m) ]
```

At `ma = 0` every one of those collapses term for term onto `build_poly_constants` and
`d0 = mh*(alpha*gamma - beta^2)`. That is gate C1a and it is why this is a derivation and
not a re-parameterisation.

`deriv()` is restated rather than patched, following the `plant_coulomb.py` precedent:
`model_augmentation/` is Jan's framework and is not modified. The arithmetic is kept term
for term (Horner, divide after the matmul), because reassociating costs ~2 ulp and that is
exactly what broke the equivalent gate in the Coulomb thread (its trap T5).

---

## 2. Gate results

```
C1a ma = 0 constants vs framework, max rel  2.205e-16   PASS  (tol 1e-14, float64 both sides)
C1b ma = 0 vs stock block, max rel |dxdot|  1.305e-07   PASS  (tol 1e-6)
C2  max |M_c(Y) N_c(Y)/d_c(Y) - I|          4.441e-16   PASS  (71 Y in [-0.35, 0.35])
C3  max |M_c(Y) - M_truth(Y, da=0)[:3,:3]|  8.882e-16   PASS
C4  CoG on vs off, max rel |d xdot|         8.545e-03   PASS  (must be NON-zero)
C5  exact replay vs record, V1: X 5.3692e-10  Theta 2.7255e-11  Y 5.8799e-09   PASS
```

Three things worth keeping.

**C1b is 1.3e-07, not zero, and that is correct.** `gantry_ss` stores every physical constant
as `float32`, so the stock block's polynomial constants carry `float32` rounding while the
subclass derives them in `float64`. `1.305e-07` is one `float32` epsilon. The exact algebraic
statement is C1a (both sides in `float64`, `2.2e-16`); C1b only says the two agree to the
precision the framework itself keeps. Quoting C1b as a failure would have been quoting a
dtype difference as a physics difference.

**C3 caught a real bug, in the gate rather than in the block.** The independent numpy
`mass_matrix_cog` first omitted the `ma*l0^2` term from `M[1,1]` and C3 reported a flat
`1.010e-02 kg*m^2 = ma*L0^2`. The block was right; the checker was wrong. That is the whole
point of writing the checker as an independent expression rather than reusing the block's
own constants.

**C5 independently reproduces the coulomb-offset thread's `5.37e-10 m`.** Different code,
different folder, same number to three digits. Across all 22 records the worst X replay
residual is in the `e-9` range.

### 2.0b The training arm's model IS the pipeline's model

`true_init_train.py` assembles the interconnect directly rather than calling
`gantry_dynamic/model.py::build_model`, which is a reasonable choice (the encoder, the
multiple-shooting subclass, the orth penalty, the ReZero gate and the Lipschitz cap all
become dead weight once the encoder is deleted) and also a way to silently train a different
model. `check_model_equivalence.py` closes that: both models built from the same
`RunConfig` and the same `Norm`, the ANN weights copied across so only structure can differ,
then rolled 400 steps from the same state with the same input.

```
E1 ANN at zero init          max|dy| 0.000e+00  rel 0.000e+00  PASS
                             x_aug at segment end: ref |x| 0.000e+00, max|diff| 0.000e+00
E2 ANN perturbed off zero    max|dy| 0.000e+00  rel 0.000e+00  PASS
                             x_aug at segment end: ref |x| 2.800e-03, max|diff| 0.000e+00
```

Bit-identical, not merely close. E2 exists because E1 alone would pass against a model with
the ANN wired to nothing: with the output identically zero the entire augmentation path is
invisible. That is the same trap as T2 in the coulomb-offset log, a gate passing because the
thing under test vanished.

**E2 also independently confirms section 5.2b.** Perturbing only the ANN's final layer takes
the propagated `x_aug` at segment end from exactly `0.000e+00` to `2.800e-03`. The augmented
rows are not structurally dead; they are dead at initialisation.

The gate is run with `cog=False`, because the CoG correction is our change and must not be
smuggled into an equivalence claim. It has its own gates, C1a-C5.

### 2.1 How wrong the record's finite-difference velocities actually are

Measured on `V1_standstill_Yp10`, exact truth vs `x_logical[:, 3:]`:

```
                dX [m/s]  dTheta [rad/s]      dY [m/s]
max           9.4661e-06      6.2252e-05    1.0495e-04
rms           6.5670e-07      4.0065e-06    2.8039e-06
rel (rms)     5.8029e-04      5.9251e-04    6.3124e-04
```

A flat `~6e-04` relative error on every velocity channel, and it is not noise: for a
central difference the truncation error is `-(w*ts)^2/6` relative, which at 20 kHz over the
`[130, 180] Hz` excitation band is `3.7e-04` to `5.3e-04`. The measurement matches the
mechanism, so the FD velocity error is understood, not merely quantified.

Consequence on a `K = 0` axis: a seed velocity error `dv` displaces the window by about
`dv * nf * ts / 2` in the mean. At `dv = 6.6e-07 m/s` on X and `nf*ts = 0.1 s` that is
`3.3e-08 m`, i.e. the same order as the free-run floor, so this term is expected to matter
but not to dominate. Section 3 measures what it actually does.

---

## 3. Task item (i): the per-window target check

`diag_window_target.py`, all four validation records (one per excitation class), 476 windows
of `nf = 400` (0.100 s) each on a stride-100 start grid, 4 kHz, block-mean input,
`up_sample = 1`, CoG-corrected baseline, float64. Three seeding arms on ONE start grid so
the numbers are directly comparable: `record` (today's `x_logical[s]`, finite-difference
velocities), `exact` (this experiment), and `freerun` (one continuous run from the rest IC,
chunked on the same windows).

### 3.1 The result, V1

```
state   unit       record seed     exact seed       free run  exact/free     gain
X       m           3.3000e-08     1.2703e-08     7.6940e-08        0.17     2.6x
Theta   rad         4.7928e-07     4.7217e-07     4.1339e-09      114.22     1.0x
Y       m           1.0273e-04     1.0278e-04     3.1469e-08     3266.24     1.0x
dX      m/s         6.4943e-07     2.2495e-07     5.7877e-09       38.87     2.9x
dTheta  rad/s       8.8896e-06     8.8952e-06     1.1686e-07       76.12     1.0x
dY      m/s         2.0241e-03     2.0251e-03     3.1155e-05       65.00     1.0x
```

and the `gain` column (record seed / exact seed, i.e. what the analytic velocities bought)
across all four classes:

```
                        X     Theta       Y       dX   dTheta      dY     Y exact/free
V1 standstill        2.6x      1.0x    1.0x     2.9x     1.0x    1.0x         3266
V2 aprbs             1.2x      1.0x    1.0x     1.2x     1.0x    1.0x          230
V3 ysweep            2.1x      1.0x    1.0x     2.2x     1.0x    1.0x         2347
V4 lissajous         1.7x      1.0x    1.0x     1.4x     1.0x    1.0x          657
```

The pattern is identical on every excitation class: the exact velocities buy a factor on X
and dX and exactly nothing on Theta, Y, dTheta, dY. This is not one record's accident.

### 3.2 The acceptance criterion is FAILED on five of six states

**X passes and passes well** (`0.17x` the free-run floor, and `1.27e-08` against the
handoff's `9.147e-08` 20 kHz figure). **Every other state is 39x to 3266x above its floor.**

**And the exact velocities bought nothing where it matters.** The `gain` column is the whole
story: replacing the record's finite-difference velocity rows with the truth's own
integrator states improves X by `2.6x` and dX by `2.9x`, and improves Theta, Y, dTheta and
dY by **`1.0x`, i.e. not at all**. The Y per-window scatter is `1.0278e-04 m` from the exact
IC against `1.0273e-04 m` from the finite-difference IC. Those are the same number.

This **falsifies the handoff's section 5 assumption.** F1's zero-mean result does not survive
per-window re-seeding from the exact physical IC. It was right to be listed as an assumption.

### 3.3 The free-run floor is real, and it is the 20 kHz figure

The `freerun` column reproduces the coulomb-offset config-B numbers at a different rate, a
different integrator setting and a different (CoG-corrected) baseline:

```
              this run, 4 kHz     coulomb-offset, 20 kHz
X               7.694e-08              9.147e-08
Theta           4.134e-09              3.730e-09
Y               3.147e-08              2.979e-08
```

So the floor is a property of the model mismatch and not of the measurement, and the
velocity floors, which were never measured before, are

```
dX  5.788e-09 m/s      dTheta  1.169e-07 rad/s      dY  3.116e-05 m/s
```

derived the same way (free run at the exact IC, per-window means, same window grid), as
handoff section 10 requires them to be before use.

### 3.4 Three by-products worth keeping

**float32 is not the limit.** The dtype training actually runs in changes the exact-seed
scatter by less than 0.01 % on every state. It does inflate the free-run floor on Y
(`3.15e-08 -> 9.85e-08` on V1, `4.49e-08 -> 5.57e-07` on V3), which is round-off accumulating
over a 12 s single-precision rollout, but the per-window quantity is untouched. No reason to
pay for float64 training.

**The CoG correction earns its place on the re-seeded arms and only there.**

```
state       exact CoG on   exact CoG off     free CoG on    free CoG off
X             1.2703e-08      6.0257e-07      7.6940e-08      1.0073e-08
dX            2.2495e-07      1.2268e-05      5.7877e-09      1.7556e-07
Y             1.0278e-04      1.0278e-04      3.1469e-08      3.1474e-08
```

On the per-window arm the correction is worth `47x` on X and `55x` on dX, which is what
"stops polluting the X and Theta rows" means quantitatively. On Y it changes nothing to five
digits, exactly as the row structure predicts (the `L0` terms never enter the Y row). On the
continuous free run it makes X position `7.6x` worse while making dX `30x` better; that is
consistent with coulomb-offset F4's finding that the correction is not a fix for a settled
offset, and it is why the correction is applied here as confound removal and nothing else.

**The free run is NOT zero-mean on X and dX.** See section 3.4b, which measures this on both
baselines because the question of whether the CoG correction causes it is not answerable from
the CoG-ON arm alone.

### 3.4b Zero-mean, on which states, and does the CoG correction change it

`diag_zeromean_cog.py`. Newey-West HAC t on the 476 per-window means, both baselines, both
records, all six states. Added after the first pass because `diag_window_target.py` stored
bias and t only for the CoG-ON arms.

**Per-window re-seeding from the exact IC is zero-mean on every state, with and without the
correction.**

```
exact-seed t         CoG ON (V1 / V3)        CoG off (V1 / V3)
X                     0.09 /  0.56            0.09 / -1.19
Theta                 0.11 /  0.67            0.51 /  0.66
Y                    -0.09 / -0.66           -0.09 / -0.66
dX                    0.21 /  0.49            0.09 / -1.22
dTheta                0.20 /  0.64            0.08 /  0.72
dY                   -0.12 / -0.68           -0.12 / -0.68
```

Every value is inside `|t| < 2`, and the record-seed arm is the same (`|t| <= 1.84`). **This
is the important half of the result and it is easy to read the wrong way round.** The short
windows are as clean in the MEAN as the 12 s free run is; what they are not is clean in the
VARIANCE, and the variance is what corrupts the training target. Coulomb-offset F4 said "it
is variance, not bias" for the K0-seeded case; that survives exact initialisation, on all six
states, unchanged.

**On the 12 s free run the answer depends on the correction.**

```
free-run t           CoG ON (V1 / V3)        CoG off (V1 / V3)
X                    11.44 / 10.71          -11.70 / -14.46
Theta                 2.30 /  2.06            0.37 /   0.12
Y                     1.29 /  0.10            1.29 /   0.06
dX                   31.41 / 23.87           -0.65 /  -0.10
dTheta                0.04 /  0.07            0.19 /  -0.04
dY                   -0.13 / -0.01           -0.13 /  -0.01
```

Without the correction, five of six states are zero-mean and only X is not. With it, X and dX
are not and Theta goes marginal. So the correction does **not create** the X free-run bias,
which is there either way, but it **flips its sign and amplifies it about 7x**
(`-1.61e-08 m`, `t = -11.70` -> `+1.20e-07 m`, `t = +11.44`) and it **does create** the dX
bias outright (`t = -0.65 -> +31.41`), i.e. with the correction X no longer settles but is
still drifting at the end of the record.

**Two caveats, so this is not over-read.** The sizes are `1e-08` to `1e-07 m` over 12 s,
three decades below the `1.03e-04 m` this task is about. And it does not contradict
coulomb-offset F4's `|t| < 1.3`: that was measured at 20 kHz where the X free-run floor is
`9.1e-08 m`, and against config A's `6.3e-07 m` per-window X scatter a `1.6e-08 m` bias is
not resolvable. This refines F4, it does not overturn it. The operational consequence is that
the CoG correction should be used as the handoff frames it, confound removal for the
RE-SEEDED arms where it is worth `47x` on X, and not carried into a free-run claim without
this caveat attached.

### 3.5 A shorter training window does not rescue it either

The obvious response to sections 3 and 4 is that `nf = 400` is what makes the ramp large, so
a shorter horizon would give a cleaner target. `diag_nf_sweep.py` measures that, on V1, 200
windows per horizon, each horizon compared against the free run **at the same horizon** so
the window length itself is not the confound.

```
    nf   [ms]    Y DC exact    Y DC free   DC ratio   Y RMS exact   Y RMS free  RMS ratio   DC/RMS
    25    6.2    6.7403e-06   1.9412e-07       34.7    8.5516e-06   2.1252e-06        4.0    0.788
    50   12.5    1.3110e-05   1.8108e-07       72.4    1.5536e-05   2.1584e-06        7.2    0.844
   100   25.0    2.6075e-05   1.2483e-07      208.9    3.0288e-05   2.2192e-06       13.6    0.861
   200   50.0    5.3265e-05   6.2937e-08      846.3    6.1904e-05   2.1854e-06       28.3    0.860
   400  100.0    1.0699e-04   3.1908e-08     3353.2    1.2335e-04   2.1988e-06       56.1    0.867
   800  200.0    2.0212e-04   1.4785e-08    13671.3    2.3187e-04   2.1820e-06      106.3    0.872
  1600  400.0    3.9775e-04   8.5484e-09    46529.0    4.5261e-04   2.1988e-06      206.8    0.879
```

Three things fall out.

**The signal is flat and small.** The free-run in-window Y RMS is `2.19e-06 m` at every
horizon from 6 ms to 400 ms. That is the absorber's ongoing contribution, the thing the ANN
is meant to learn, and it does not grow with the window because it is a bounded oscillation.

**The corruption grows linearly with the horizon**, because it is a ramp: `1.23e-04 m` at
`nf = 400`, `56x` the signal, and `4.53e-04 m` at `nf = 1600`.

**And `DC/RMS` stays pinned at the pure-ramp value.** A ramp `a*t` over `[0, T]` has
`mean/RMS = sqrt(3)/2 = 0.866`. Measured: `0.788` at `nf = 25` rising to `0.879` at
`nf = 1600`. So even at 6 ms the per-window error is 79 % ramp.

**Consequence.** Bringing the IC error down to the signal level would need
`nf ~ 400/56 = 7` samples, i.e. `1.75 ms`, against an absorber period of `6.7 ms`. There is
no horizon that both contains one oscillation of the thing to be learned and is short enough
for the initial-condition error not to dominate it. Shortening `nf` trades the ramp against
the ring-down at a fixed, unfavourable rate; it does not escape.

---

## 4. Why the target is still dirty: the mechanism

`diag_dc_mechanism.py`, `V1_standstill_Yp10`, same 476 windows.

### 4.1 The per-window DC is the absorber initial condition, with the predicted slope

Regression of the measured per-window mean error on `[delta_a(s), vdelta_a(s)]`, the truth's
absorber state at the window start:

```
state          R^2    corr vda     slope vda     predicted    ratio
X           0.8252      0.9082    5.4154e-07          n/a      n/a
Theta       1.0000      0.9998    2.2158e-05          n/a      n/a
Y           1.0000     -0.9998   -4.8236e-03   -5.0000e-03    0.965
dX          0.9955      0.9977    1.0532e-05          n/a      n/a
dTheta      0.9959      0.9979    4.1655e-04          n/a      n/a
dY          0.9999     -0.9999   -9.5024e-02   -1.0000e-01    0.950
```

`R^2 = 1.0000` on Y and `0.9999` on dY, and the fitted slopes match the closed form to
3.5 % and 5 %. The closed form is not fitted: a missing absorber momentum `vdelta_a(0)` is a
velocity deficit `(ma/mh)*vdelta_a(0)` on the payload row, and on a `K = 0` axis it
integrates, so the mean over an `nf`-window is `-(ma/mh)*vdelta_a(0)*nf*ts/2` on Y and
`-(ma/mh)*vdelta_a(0)` on dY. This is the coulomb-offset thread's F3 (`corr -1.000`, slope
to 3 %) reproduced independently, but now per window rather than once, on all six states,
and with the CoG-corrected baseline.

X is the exception at `R^2 = 0.825`, and X is also the one state that already passed the
acceptance criterion. Its residual DC is `1.27e-08 m`, three decades below the floor-relevant
scale, so what is left there is not worth attributing.

### 4.2 The decisive control: the truth model against itself

The same window grid, the same integrator, the TRUTH's own 8-state model on both arms, and
the only difference is the absorber initial condition:

```
state        T8 exact-8   T6 abs zeroed  baseline exact-6  T6 / baseline
X            5.3130e-09      1.2703e-08        1.2703e-08          1.000
Theta        5.1401e-10      4.7216e-07        4.7217e-07          1.000
Y            1.7350e-08      1.0278e-04        1.0278e-04          1.000
dX           1.4395e-08      2.2494e-07        2.2495e-07          1.000
dTheta       5.7684e-07      8.8951e-06        8.8952e-06          1.000
dY           1.3194e-07      2.0258e-03        2.0251e-03          1.000
```

**T8**: re-seeded from the complete 8-state IC, the per-window scatter collapses to the
integrator floor on every state. So the window grid, the statistics and the re-seeding
procedure are sound. Per-window re-seeding is not intrinsically dirty.

**T6**: the same truth model, re-seeded from the exact SIX states with the absorber zeroed,
reproduces the baseline's scatter to **1.000 on every one of the six states, to four
digits**. There is no baseline model in that arm at all: no CoG term, no LFR realization, no
6-vs-8-state mismatch. The entire per-window DC is the absorber initial condition and
nothing else.

T6 is a measurement of the mechanism, not an implementation of the "seed rows 6-7" fix the
handoff rejected in section 8. It runs on the TRUTH's own absorber coordinates, where they
mean what they say. The model's latent rows 6-7 are not touched anywhere in this file.

### 4.3 What this does to the experiment's premise

The handoff's framing was: remove the initialisation confounds and the target becomes clean,
so a failure to learn must be structural. The first half of that is now false. Removing the
encoder and the finite-difference velocities leaves the target **as dirty as it was**, on
every state except X, because the confound that dominates was never an encoder problem: the
model has no absorber state to initialise, so no initialisation procedure whatsoever can
remove it. An encoder that reconstructed the six physical states perfectly would land exactly
on this experiment's numbers.

That is a stronger statement than the handoff anticipated, and it changes what the encoder
thread can hope for. `docs/dc-accumulation-*` and the F6 result (`vDelta_a` recoverable from
the encoder's inputs at `R^2 = 1.0000`) say the information is there to be had; this says
that recovering it into rows 6-7 is worth nothing while those rows are overwritten every
step (G6). The two are the same finding seen from opposite ends.

---

## 5. Is the correction even a function of what the ANN can see?

`diag_static_representability.py`, `V1_standstill_Yp10`, 47999 samples. The object measured
is `Delta(k) = x6_truth(k+1) - Phi_base(x6_truth(k), u(k))` in normalised state units: the
`xp` contribution the ANN would have to make, at that step, for the model to be exact given
a correct current state. The ANN's input is `z = [x_norm (8), u_norm (3)]` with rows 6-7
identically zero (G6), so eleven columns of which nine carry information.

```
L  R^2 of Delta on z            overall 0.879342
   per state  X 0.9647  Theta 0.9664  Y 0.8692  dX 0.7835  dTheta 0.9454  dY 0.8793
A  R^2 of Delta on [z, da, vda] overall 0.999994
   per state  X 0.9649  Theta 0.9670  Y 0.9999  dX 0.7938  dTheta 0.9633  dY 1.0000
C  R^2 of the CONTROL on z      overall 1.000000
N  nearest-neighbour consistency over the 5 % closest pairs in z (median pair distance
   4.427e-02 against 1.051e-01 over all pairs)
     Delta   2.1434        control 0.0769
```

Three readings, and the third is the one that generalises.

**Most of the correction IS learnable.** `R^2 = 0.879` of the ideal per-step correction is a
linear function of the ANN's own inputs, before any nonlinearity is allowed. A static ANN can
therefore be expected to reduce the in-window fit error, which is the bulk of the training
loss. This is a prediction for the training arm, registered before it runs.

**What is missing is exactly the absorber state.** Adding `[delta_a, vdelta_a]` takes the
overall `R^2` from `0.879` to `0.999994`, and on the two rows that carry the DC it goes
`0.8692 -> 0.9999` (Y) and `0.8793 -> 1.0000` (dY). Nothing else is missing. Nothing else
needs to be.

**No static function of `z` can supply it, whatever its capacity.** Pairs of samples that are
near-duplicates in `z` (median distance 2.4x closer than typical) have `Delta` values that
differ by 2.14 in units where 1.0 is "as different as two random samples". A continuous
`f(z)` must return near-equal outputs for near-equal inputs, so this falsifies every static
map from `z`, not just linear ones and not just this architecture. The control target, which
is a static function of `z` by construction, scores `0.0769` on the identical machinery, so
the test is not passing for the wrong reason. (Caveat on the exact value: the metric pools
the six rows and is dominated by dY, whose scale is largest; the defensible statement is the
28x separation between `Delta` and the control, not the number 2.14 itself.)

### 5.1 The record class matters, and it matters a lot

The same measurement on `T9_aprbs_30`, a broadband training record:

```
              V1 (narrowband 130-180 Hz)   T9 (aprbs, broadband)
L  Delta on z            0.879342                 0.032856
A  Delta on [z, da, vda] 0.999994                 0.999994
N  Delta / control       2.1434 / 0.0769          1.8272 / 0.1438
```

On the narrowband standstill record most of the correction is statically representable,
because in a narrow band the absorber state is close to a fixed linear function of the
payload state and a static map of the current state can proxy it. On broadband excitation
that proxy collapses: `R^2 = 0.033`. Both records land on `0.999994` once the absorber state
is supplied, so the missing quantity is identical; what changes is how well the current state
stands in for it. The training set is five standstill records, three Y-sweeps, four APRBS and
two Lissajous, so it is a mixture and the ANN cannot lean on the narrowband proxy throughout.

### 5.2 The capacity test, and its own caveat

```
                        R^2 train    R^2 val
V1  Delta                  0.3511     0.3730
V1  control                0.9999     0.9999
T9  Delta                  0.0372    -0.0423
T9  control                0.9998     0.9999
```

Identical architecture (the pipeline's own 2x16 tanh `zero_init_feed_forward_nn`), identical
optimizer, identical 3000-step budget, identical 80/20 split. The static control reaches
four nines on both records; the real correction reaches `0.37` and `-0.04`. **The
architecture is not the limit.**

**Caveat, stated because it is against the reading**: on V1 the MLP's `0.373` is BELOW the
plain linear least squares' `0.879`, so the MLP fit is not converged and its `Delta` number
is a lower bound, not an estimate of what the architecture could reach. What survives the
caveat is the comparison, not the level: the same budget that leaves `Delta` at `0.37`
takes the control to `0.9999`. The honest conclusion is "capacity and budget are not what
separates these two targets", not "the ANN can only reach 0.37".

---

### 5.2b CORRECTION: the augmented partition IS a recurrence, and this changes the claim

Found while writing section 6 and load-bearing, so it is stated before the conclusions rest
on it.

`model.py:131` wires the ANN's input as `connect_block_signals(ann_block, ["x","u"], [])`
with **no selection matrix**, unlike the physical block and the output block which both get
`selection_matrix(PHY_IX, nxd)`. So the ANN reads all `nxd = 8` state rows, 6-7 included
(`nz = nxd + nu = 11`, which is what the diagnostics print), and `model.py:132` routes its
output back into those same rows. The augmented partition is therefore a learnable nonlinear
**recurrence**

```
x_aug(k+1) = h_ann( x_phys(k), x_aug(k), u(k) )
```

not a state "rebuilt from scratch by a static feedforward net every sample", which is how
`scripts/gantry/coulomb-offset/IMPLEMENTATION-LOG.md` section 12.2 words it. That folder is
read-only for this task, so the correction is recorded here.

**What this does and does not change.** Gate G6's `x_aug = 0.000000e+00` is an
INITIALISATION result: the zero-initialised final layer makes the ANN output exactly zero,
so the recurrence is dead-beat and its Jacobian is exactly zero at init. Sections 4 and 5
set rows 6-7 to zero, which is exactly right for the model as initialised and as trained
here, and it is what the training arms actually start from. But the sentence "no static
`f(z)` can supply the correction" is a statement about **the model while those rows stay
zero**, not about the model class. A trained ANN could in principle carry the absorber state
in rows 6-7. So the section 8 verdict is narrower than "persistence is structural": the
model starts at an exact dead-beat corner and has to learn its way out from zero.

**And that is measurable, so it was measured.** `diag_aug_state_activity.py`, `V1`, 64
windows, on the checkpoints as they stood:

```
checkpoint                   |x_aug| rms   |x_aug| max   J gain mean   J gain max  R2 vs [da,vda]
(untrained)                   0.0000e+00    0.0000e+00    0.0000e+00   0.0000e+00          0.0000
main_lr1e-7_last  (7 epochs)  5.9634e-08    1.5171e-07    1.0268e-08   1.1072e-08          0.1482
main_lr1e-6_last  (7 epochs)  2.2912e-06    6.5442e-06    1.7493e-06   1.8713e-06          0.1481
main_lr1e-5_last  (3 epochs)  7.0200e-05    3.1265e-04    7.5729e-05   7.9356e-05          0.1296
```

`J gain` is the largest singular value of `d x_aug(k+1) / d x_aug(k)`: the recurrence gain
the ANN has actually built. A lightly damped 150 Hz oscillator at 4 kHz needs
`|lambda| ~ 0.99`. Measured `1e-08` to `8e-05`, four to eight decades short, and it tracks
the learning rate linearly rather than tracking anything about the data. `R^2` of the best
affine map from `x_aug` to the truth's `[delta_a, vdelta_a]` is `0.13` to `0.15`, so what
little the rows carry is not the absorber.

**Consequence for the Györök proposal, and it is a different argument from the one in the
coulomb-offset log.** That log argued the framework HAS the object and we omitted it. The
measurement says something sharper: the object exists in our wiring, but its gain is
initialised at exactly zero and Adam moves a weight by at most `lr` per step, so reaching
`|lambda| ~ 0.99` from `0` needs of order `1/lr` consistent updates. The parameterisation
`A = alpha_bar * sigma_A * A_bar` with `sigma_A = sigmoid(alpha)` starts at
`sigmoid(0) = 0.5`, i.e. it starts the only dynamic path at order 1 instead of at zero. The
value is in where it starts, not only in that it exists.

### 5.3 The one-line version: the information is in the window, not in the sample

`diag_absorber_observability.py`. Two least-squares fits of the truth's absorber state, on
identical data, with a held-out tail (70/30, the 30 % is the end of the record, not
interleaved). `inst` is what the ANN sees: `[x_phys(k) (6), u(k) (3)]`, with rows 6-7 omitted
because they are identically zero. `window` is what the encoder sees:
`y[k-17..k]` and `u[k-17..k]`, `na = nb = 17` being Jan's `2*(nx_phys+nx_ann)+1`.

```
record                target      inst train   inst test   window train  window test
V1_standstill_Yp10    delta_a         0.8687      0.8713         1.0000       1.0000
                      vdelta_a        0.4781      0.4899         1.0000       1.0000
V2_aprbs_Ylow         delta_a         0.1370      0.0889         1.0000       1.0000
                      vdelta_a        0.0843      0.0970         1.0000       1.0000
V3_ysweep_Yp10        delta_a         0.0500      0.0045         1.0000       1.0000
                      vdelta_a        0.3030      0.2868         1.0000       1.0000
V4_lissajous_Ym10     delta_a         0.0396      0.0538         1.0000       1.0000
                      vdelta_a        0.3034      0.2641         1.0000       1.0000
```

**From an 18-sample window the absorber state is recoverable exactly, on every record class,
out of sample, by ordinary least squares.** That independently reproduces the coulomb-offset
thread's F6 (`R^2 = 1.0000`).

**From the instantaneous state and input it is not.** `vdelta_a`, the quantity that sets the
entire per-window DC, is 49 % explained on the narrowband standstill record and 10 to 29 %
on the other three.

So the thread reduces to one sentence. *The information the augmentation needs is present in
the encoder's inputs and absent from the ANN's inputs, and the only rows that could carry it
from one to the other are overwritten to exactly zero every step.* Encoder work cannot fix
that, because the encoder is already able to see it; a bigger ANN cannot fix it, because the
signal is not in its input; only giving the augmented partition its own dynamics can.

---

## 6. Task item (ii): the training arm

### 6.1 The decision to run it, and why

Handoff section 10 says that if the target is dirty, diagnose first and then decide. The
diagnosis (sections 4 and 5) says the corruption is not a defect in this code: it is the
absorber initial condition, it is deterministic, and it is fully characterised. The arm was
run anyway, for four reasons. It is task item (ii) and it is cheap at stride 100. A
mechanism argument that is never confronted with a run is weaker than one that is. The static
analysis makes two falsifiable predictions about the run (P1, P2 in the run-log row), so the
run is a test of the analysis rather than a fishing trip. And whether the ANN does active
HARM under a clean IC is a genuinely open question that no static analysis can answer.

### 6.2 What the loss actually consists of, measured before launch

ANN-off val, `nf = 400` free run from the exact IC over V1-V4:

```
val nf-RMS            7.1698e-05 m       per channel  X1 3.533e-07  X2 3.863e-07  Y 1.2420e-04
val per-window DC     2.700e-07 / 2.903e-07 / 1.0790e-04 m
```

The Y channel is 300x the X channels and carries essentially the whole metric. And its shape
identifies it: a per-window error that is a pure ramp `a*t` has RMS `a*T/sqrt(3)` and mean
`a*T/2`, so mean/RMS must be `sqrt(3)/2 = 0.866`. Measured: `1.0790e-04 / 1.2420e-04 =
0.869`. The ramp slope predicted from the absorber momentum is
`a = (ma/mh)*std(vdelta_a) = 0.1 * 2.1653e-02 = 2.165e-03 m/s`, giving an RMS of
`a*0.1/sqrt(3) = 1.25e-04 m` against the measured `1.242e-04`.

**So the training loss under a true initial condition is, to within a few per cent, entirely
the absorber initial-condition ramp.** That is the sharpest form of the section 3 result: it
is not that the target is somewhat corrupted, it is that under exact 6-state initialisation
the target IS the absorber IC error.

### 6.3 The learning-rate probe

10 updates per arm. The STAGE 1 lesson (a 10-update probe cannot rank a 2600-update run)
applies and is why this is reported as a screen, not a ranking.

```
lr      step 0 train MSE   step 1        step 9        val vs ANN off
1e-7    1.139229e-07       1.182759e-07  1.185691e-07  +0.69 %
1e-5    1.139229e-07       5.452921e-05  1.866650e-05  +332.6 %
1e-3    1.139229e-07       7.996051e-01  1.405187e-01  +27832.9 %
```

The step-1 jump scales with lr, which is the signature coulomb-offset section 11.4 predicted
and did not get to test at more than one rate: Adam's update is gradient-normalised, so a
zero-initialised final layer takes a full `lr`-sized step no matter how small its gradient
is. At `1e-3` that is a `7e+06x` loss increase in one update. Two mitigations exist (a warmup,
or plain SGD whose first step is proportional to the gradient) and NEITHER is tested here,
because both add a mechanism the pipeline does not have and this experiment already carries
enough changes.

`1e-7` is the pipeline's own documented rate for `(0..7)` routing (D-101/D-102) and is the
primary arm; `1e-6` is run as a bracket.

### 6.4 Result

**The runs were killed externally at roughly half their epoch budget** (47, 47 and 43 of 90),
by something outside this session. They are not lost: `_last` was written every epoch,
deliberately, because D-130 was written from a run whose trained weights survived only in a
`_last` artefact. `harvest_runs.py` recovers the series from the streamed logs, which are
kept in `runlogs/`. So what follows is 1222, 1222 and 1118 Adam updates rather than the 2340
that were budgeted, and that limitation is stated wherever it matters.

```
    lr  epochs  updates      ANN off         best   best %  final %  worst %  DC_Y best %   |w| final
 1e-07      47     1222   6.8486e-05   6.8312e-05    -0.25     1.02     1.59        -0.28    1.80e-06
 1e-06      47     1222   6.8486e-05   6.8037e-05    -0.65     2.16     8.54        -0.62    4.78e-05
 1e-05      43     1118   6.8486e-05   6.7924e-05    -0.82    -0.16   167.08        -1.40    4.17e-04
```

Percentages are against each run's own ANN-off value, which is exact rather than approximate
(`max|w| = 0.000e+00` at init, measured, so the untrained model IS the baseline).

**P1, "val nf-RMS improves", is satisfied only in the most technical sense and should be
read as a null.** The best point of the best arm is `0.82 %` below ANN-off. Meanwhile
**91 %, 91 % and 93 % of all validation points sit ABOVE the ANN-off value**, the `1e-6` arm
peaks `8.5 %` worse and the `1e-5` arm peaks `167 %` worse. A metric that a model beats on
7 % of its checkpoints by under one per cent has not been learned; it has been jittered.

**P2, "the Y per-window DC does not improve materially", is confirmed.** The best DC
improvement over the whole of all three runs is `1.40 %`, on the arm whose ANN output is
largest. The DC is `1.03e-04 m` and it stays `1.03e-04 m`.

**P3 therefore holds**, and section 5.2b's measurement says why in the sharpest possible
form. Re-run on the final checkpoints:

```
checkpoint                   |x_aug| rms   |x_aug| max   J gain mean   J gain max  R2 vs [da,vda]
(untrained)                   0.0000e+00    0.0000e+00    0.0000e+00   0.0000e+00          0.0000
main_lr1e-7_last              3.4297e-07    5.5798e-07    5.1617e-08   5.6430e-08          0.1135
main_lr1e-6_last              9.0050e-06    3.4561e-05    1.0155e-05   1.0837e-05          0.1553
main_lr1e-5_last              1.0234e-04    4.7123e-04    1.5656e-04   1.6424e-04          0.1359
main_lr1e-5_best              8.1944e-05    3.7377e-04    1.0735e-04   1.1248e-04          0.1317
```

After 1118 updates at the largest usable learning rate the recurrence gain is `1.57e-04`
against the `~0.99` a damped 150 Hz absorber needs, and it is still tracking the learning
rate linearly rather than tracking the data. The augmented rows do carry something by the end
(`|x_aug| ~ 1e-04`, up from exactly zero) but `R^2` of the best affine map onto the truth's
`[delta_a, vdelta_a]` is `0.13`, so what they carry is not the absorber.

**Honest limits on this arm.**

- Half the budgeted updates. Doubling them moves an Adam random walk by `sqrt(2)`, which does
  not close a four-decade gap, but the series is what it is and the truncation is stated.
- One seed per arm. The `+-1 %` band the arms wander in is not resolved against seed noise,
  which is precisely why the reading rests on the 91-93 % figure and on P2 rather than on the
  sign of a `0.8 %` best point.
- Neither a warmup nor plain SGD was tried, and both would defuse the first-update Adam
  artefact that forces the learning rate down in the first place. That is the most obvious
  thing this arm did not test.
- `nf = 400` and `stride = 100` throughout, so nothing here speaks to a different horizon
  beyond what section 3.5 measured without training.

### 6.5 The 12 s free-run arm, and it is the most important number here

Backlog item 1. Whole-record free run from the exact rest IC (no encoder, no initialisation
error of any kind), ANN off against the best checkpoint of the `1e-5` arm, all four
validation records.

```
record                arm         RMS [m]           X1           X2            Y     win DC Y
V1_standstill_Yp10    off      1.2979e-06   4.3280e-08   4.3026e-08   2.2472e-06   3.5573e-07
V1_standstill_Yp10    on       4.5155e-03   1.3458e-04   1.3442e-04   7.8187e-03   4.2381e-03
                      ratio     3479.02   DEGRADES
V2_aprbs_Ylow         off      2.1459e-06   2.0149e-06   2.0152e-06   2.3861e-06   5.8202e-07
V2_aprbs_Ylow         on       5.4212e-04   8.1702e-05   8.1667e-05   9.3185e-04   9.2863e-04
                      ratio      252.64   DEGRADES
V3_ysweep_Yp10        off      1.3602e-06   4.5577e-08   4.5381e-08   2.3550e-06   3.9330e-07
V3_ysweep_Yp10        on       4.8624e-03   1.2306e-04   1.2291e-04   8.4201e-03   4.4681e-03
                      ratio     3574.84   DEGRADES
V4_lissajous_Ym10     off      3.1762e-06   2.6254e-06   2.6252e-06   4.0597e-06   1.6299e-06
V4_lissajous_Ym10     on       9.9342e-04   3.5441e-05   3.5257e-05   1.7199e-03   7.9547e-04
                      ratio      312.76   DEGRADES
```

**The same checkpoint improves the 0.100 s windowed metric by `0.82 %` and degrades the 12 s
free run by `253x` to `3575x`.** It was selected as "best" on the windowed metric. That is
the 120x train/select horizon gap (D-129) in its purest available form: **no encoder, no
initialisation error, an exactly known initial condition, and the split reproduces at full
strength anyway.**

This matters beyond this task. The gap has been demonstrated twice before and each time a
confound survived: run 71167 had the encoder in the loop, and the STAGE 1 black box removed
the baseline but kept the encoder and had no informed `x0`. Here the initialisation is exact
by construction, so **initialisation is eliminated as an explanation of the horizon gap.**

The mechanism is visible in the numbers rather than inferred. The ANN's output is
`|w| ~ 4e-04` in normalised state units per step, and it writes the K = 0 rows directly. A
non-zero-mean component of that output has nothing to relax it, so 48000 steps integrate it:
the free-run per-window DC on Y goes from `3.56e-07 m` (ANN off) to `4.24e-03 m` (ANN on),
a factor of `11900`. The windowed loss cannot see it, because inside 400 steps the same DC
contributes only 1/120 as much displacement. This is the DC-accumulation picture that
`docs/dc-accumulation-*` and MS12 built, now reproduced with the initialisation confound
removed.

**Also worth recording: the ANN-off free-run RMS from the exact rest IC is `1.30e-06` to
`3.18e-06 m` across the four records.** That is what the CoG-corrected baseline is worth over
12 s with a correct initial condition, and it is the number an augmentation has to beat.

---

## 7. Proposed amendment to D-130, DRAFTED ONLY

`docs/decisions.md` is deliberately NOT edited. The handoff forbids amending D-130 with
nobody available to approve it (its section 12), so the text is parked here.

D-130 says the `W^a` dead zone "follows from the wiring, not from a hyperparameter" and calls
it structural. Two measurements now bear on that, and they pull in opposite directions.

**The word "structural" is wrong, and that was already established.** `verify_rezero_gate.py`
(coulomb-offset section 11) measured that the plain zero-init recovers `W^a` gradient at
optimizer step 1: `dL/dW_final = <dL/dw, h>` is non-zero even at `W_final = 0`, so the final
layer moves on the first step and the input-Jacobian is alive on the second. Measured
`|g W^a|` goes `0.000000e+00 -> 1.586e-05` between step 0 and step 1. The dead zone is a
one-step transient. That refutation is a month older than this session and D-130 has not yet
been amended for it.

**But the conclusion D-130 draws survives, for a different reason, and this session is what
supplies it.** D-130's operative claim is that the augmented rows are inconsequential. That
remains true, and it is not about gradient at all:

- the propagated `x_aug` is exactly `0.000000e+00` at every step, because `model.py:132`
  connects only the ANN to rows 6-7 and there is no identity path from `x` (gate G6);
- so the ANN is a static map of `z = [x_phys, 0, 0, u]`;
- and the correction it would have to produce is provably not a function of that `z`
  (section 5: `R^2` 0.879 / 0.033 on `z`, 0.999994 once `[delta_a, vdelta_a]` are added,
  and a nearest-neighbour test that rules out every static map);
- so filling rows 6-7 correctly at window start would change nothing, because they are
  overwritten before they are used.

**Proposed replacement wording for the "Why" of finding (1):**

> The dead zone at step 0 follows from the wiring (`h_base` and `f_base` are both wired with
> `selection_matrix(PHY_IX, nxd)`, `model.py:123` and `:127`, and the ANN's final layer is
> zero-initialised, `torch_nets.py:113-114`, so its input-Jacobian is exactly zero at
> initialisation). It is a ONE-STEP TRANSIENT, not a structural barrier: `dL/dW_final =
> <dL/dw, h>` is non-zero at `W_final = 0`, so the final layer moves at step 1 and the
> Jacobian is alive at step 2 (measured, `verify_rezero_gate.py`, `|g W^a|` `0.000000e+00 ->
> `1.586e-05`). What IS structural is downstream of the gradient: nothing but the ANN writes
> rows 6-7, so the propagated `x_aug` is identically zero (gate G6) and the ANN is a static
> map of the physical state and the input. The correction it must produce is not a function
> of those (`scripts/gantry/true-init-augmentation/`, section 5), so the augmented rows are
> inconsequential whatever the encoder puts in them. The fix is therefore autonomous
> dynamics on the augmented partition (the Györök contracting `A_aug`), not encoder
> initialisation and not a gate.

**Proposed addition to "Ruled out":**

> (e) "Initialising the encoder's augmented rows well would make the augmented states
> useful" - refuted independently of any encoder: seeding the six physical states EXACTLY,
> with analytic rather than finite-difference velocities, leaves the per-window DC scatter at
> 1.0x its previous value on Y, Theta, dTheta and dY, because the DC is the absorber initial
> condition and the model has no state that survives one step in which to carry it
> (`true-init-augmentation/IMPLEMENTATION-LOG.md` sections 3-4).

**Proposed addition to "Constrains":**

> Any claim that an initialisation improvement will reduce the per-window DC must first show
> that the augmented partition retains state across a step. Until it does, the per-window DC
> is `-(ma/mh)*vdelta_a(0)*nf*ts/2` on Y regardless of how the physical rows are initialised
> (`R^2 = 1.0000`, measured on 476 windows).

---

## 8. What is still open

1. **Whether a learnable contracting `A_aug` fixes it.** Everything here points at it and
   nothing here tests it. The handoff puts it out of scope (its section 2) and that is
   respected. What this session adds to the case is that the requirement is now a
   measurement rather than an inference: the augmentation needs a state that survives a
   step, because the quantity it must reproduce is `R^2 = 1.0000` explained by
   `[delta_a, vdelta_a]` and `R^2 <= 0.49` by anything instantaneous.
2. **The X and dX rows.** They are the two states the exact velocities did improve, and X is
   the only state that meets the acceptance criterion. Their residual DC is `1.27e-08 m` and
   `2.25e-07 m/s`, is only 83 % explained by the absorber state, and has not been attributed.
   It is three decades below the Y scale so it was not chased.
3. **The free-run X bias.** Measured on both baselines (section 3.4b): X is non-zero-mean on
   the 12 s free run with AND without the CoG correction, and the correction flips its sign
   and amplifies it 7x while creating a dX bias that was not there. Sizes are `1e-08` to
   `1e-07 m`. The mechanism is not established. The plausible candidate is the truth's
   `-ma*delta_a` X-Theta coupling and its `ma*2*(Y+L0)*delta_a` inertia term, which are
   first order in an oscillating `delta_a` and therefore rectify into a slow force on a
   `K = 0` axis; the corrected baseline has the `L0` part and not the `delta_a` part, so it
   changes which way the rectification points. Not tested.
4. **The `1e-5` arm's first-epoch excursion.** Whether it is the Adam-over-zero-init
   artefact alone or something the routing adds was not separated. A warmup arm or an SGD
   arm would separate them and neither was run.
5. **Real data.** Everything here is the frictionless `augmentation` simulation dataset. The
   Karnopp dataset and the Telica logs were not touched.

---

## 9. Running state

- A1-A3 done, gates C1a-C5 all pass.
- Exact-truth cache: all 22 records, worst X replay residual in the `e-9` range.
- Item (i) done on all four validation records. **The acceptance criterion FAILS on five of
  six states, and the handoff's section 5 assumption is falsified.**
- Mechanism identified three independent ways (section 4) and the representability question
  answered without training (section 5).
- Training arm: three lr arms, killed externally at 47/47/43 of 90 epochs. Series recovered
  from `runlogs/` by `harvest_runs.py`; both checkpoints survived for every arm. Sections 6.4
  and 6.5 are the result. **Not restarted**: the series is flat within `+-2 %` over 1200
  updates and doubling an Adam random walk moves it by `sqrt(2)`, which does not close a
  four-decade recurrence-gain gap. The truncation is stated wherever a number depends on it.
- Session complete.
