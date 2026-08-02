# True-init augmentation: can the ANN learn the absorber at all?

Autonomous session, 2026-08-02. Brief: `tasks/handoffs/2026-08-02-true-init-augmentation.md`.
Written as the work happened, including the parts that did not work.

---

## 0. Read this first

_(filled in at the end of the session; see section 9 for the running state while the run is
in flight)_

---

## 1. What was built

Four new files, all in `scripts/gantry/true-init-augmentation/`. Nothing under
`model_augmentation/`, `kamtin-fp-model/` or `scripts/gantry/coulomb-offset/` was touched.

| File | Role |
|---|---|
| `data_exact.py` | record loader (pipeline conventions) + the EXACT 8-state truth: a 20 kHz RK4 replay from the rest IC, decimated to 4 kHz. Velocities are integrator STATES, never difference quotients. |
| `plant_cog.py` | `Gantry_State_Block_CoG`: the LFR baseline carrying the truth's static mass distribution at `delta_a = 0`. The corrected `N0,N1,N2` and `d(Y)` are derived, not fitted. |
| `check_plant_cog.py` | gates C1a-C5 |
| `precompute_exact.py` | caches the exact truth for all 22 records |
| `diag_window_target.py` | task item (i): the per-window target check |
| `true_init_train.py` | task item (ii): the training arm, encoder replaced by the exact IC |

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

_(in progress)_

---

## 9. Running state

- A1-A3 done, gates C1a-C5 all pass.
- Exact-truth cache for all 22 records: running.
- Item (i) and the training arm: pending.
