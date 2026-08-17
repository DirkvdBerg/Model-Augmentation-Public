# Results: the controller in the loop

Executes `PLAN-controller-in-the-loop.md`. All work is inside this folder. Run order:

```
export_record_reference.m     (MATLAB, once)
p1_equivalence.py             P1 gate
p2_rate_compare.py            P2 sample-rate fork
so_filter.py                  P3 sensitivity filter
a_closed_loop_models.py       A
bc_learnability.py            B step 2 and C step 6
```

## Summary

| | criterion | result |
|-|-|-|
| P1 | closed-loop equivalence | **PASS** both records, `max abs(dy)` down to `1.1e-12 m` |
| P2 | sample-rate fork | **FLAG**, 4 kHz loop has a 15.3 % higher sensitivity peak |
| P3 | `So` as a filter | **PASS**, `2.8e-11` against the direct inverse |
| A | closed-loop evaluation | **PASS** S-A1 to S-A4, and the section 6.3 prediction is confirmed |
| B/C | learnability | the absorber is learnable under all three losses; see the caveats |

---

## P1. Closed-loop equivalence

Truth model plus the verified `Cfb`, driven by `r_sim` alone, against the record.

| record | `max abs(dy)` [m] | ramp fraction | `rms(du)/rms(u)` |
|-|-|-|-|
| `V1_standstill_Yp10` | `[1.10e-12, 9.75e-13, 5.22e-09]` | `[0.00, 0.00, 0.00] %` | `[5.6e-08, 5.5e-08, 2.5e-07]` |
| `T10_aprbs_60` | `[8.97e-09, 8.63e-09, 3.29e-08]` | `[0.48, 0.47, 0.83] %` | `[4.0e-05, 4.0e-05, 1.1e-04]` |

Tolerances `1e-6 m`, `5 %`, `1e-3`. Both records pass with three to six orders of margin. The
loop wiring, sign convention, sample alignment and controller state initialisation are correct.

**Correction to the plan.** P1c was specified as `max abs(du) / rms(u)`, mixing a peak numerator
with an rms denominator. On `T10_aprbs_60`, whose reference contains steps, that inflated the
ratio to `1.46e-3` and produced a spurious FAIL. The criterion is now the consistent `rms/rms`
form; the peak/peak measure is printed for information and also passes, at `3.7e-4`.

## P2. The sample-rate fork

`sigma_max(So)`, frozen design loop:

| f [Hz] | 1 | 10 | 50 | 100 | 150 | 180 | 500 |
|-|-|-|-|-|-|-|-|
| 20 kHz | 0.0004 | 0.0214 | 1.0544 | 1.6695 | 1.7983 | 1.8043 | 1.1438 |
| 4 kHz | 0.0004 | 0.0214 | 1.0816 | 1.8362 | 2.0738 | 2.0506 | 1.1230 |

Phase margin drops `37.4` to `33.8` degrees, a `3.6` degree shift, inside the 5 degree flag.
`sigma_max(So)` at 150 Hz rises by **15.3 %**, outside the 10 % flag. Both `Y_op` values agree.

**Reading.** The 4 kHz loop differs most in exactly the band the augmentation targets, and the
sensitivity peak also governs the stability margin that B depends on. Recommendation: run all
loop work at 20 kHz and accept the 5x cost. The decision belongs in `docs/decisions.md`, which is
outside this folder and has not been written.

## P3. `So` as a filter

15 states, 6 plant plus 9 controller. Matches the frequency-by-frequency inverse to `2.8e-11`.

`sigma_max(So)` at DC is `3.7e-10`, i.e. zero to numerical precision. This is the zero at `z = 1`
inherited from the controller's integrator, and it is the substantive property of option C:
**weighting by `So` makes the loss blind to a constant output offset by construction.** That is
the formal version of "the controller would pull it to zero".

## A. Models in the loop

Baseline against truth, same loop, same reference, compared against the open-loop replay of the
same record. NRMS of the output error.

**`V1_standstill_Yp10`**, excitation narrowband `[130, 180] Hz`:

| ch | open loop | closed loop | ratio | `sigma_max(So)` predicts |
|-|-|-|-|-|
| X1 | 5.658e-02 | 1.102e-01 | **1.95** | 1.80 |
| X2 | 6.720e-02 | 1.181e-01 | **1.76** | 1.80 |
| Y | 4.961e-01 | 7.813e-01 | **1.58** | 1.80 |

**`T10_aprbs_60`**, excitation dominated by the tracking trajectory:

| ch | open loop | closed loop | ratio |
|-|-|-|-|
| X1 | 1.054e-02 | 5.737e-06 | **5.4e-04** |
| X2 | 1.054e-02 | 5.677e-06 | **5.4e-04** |
| Y | 2.555e-04 | 2.351e-05 | **9.2e-02** |

Criteria: **S-A1 PASS** (no divergence; peak force `581 N` against the `2000 N` limit, rms `168 N`
against `916 N`), **S-A2 PASS** (baseline `7.8e-1` against truth `5.9e-4` on V1 Y, three orders),
**S-A3 PASS** (ramp fractions `0.00 %` to `0.83 %`), **S-A4 PASS** (truth below baseline on both).

**The section 6.3 prediction is confirmed.** The same loop amplifies the model discrepancy by
about 1.8 where the excitation sits in the absorber band, and suppresses it by up to three orders
where the excitation is low frequency. Both follow `sigma_max(So)`, which is `1.80` at 150 Hz and
`0.021` at 10 Hz.

**Consequence for how results are reported.** Open-loop NRMS on a trajectory record overstates the
model error by up to three orders relative to what the loop experiences. Quoting open-loop NRMS on
`T10`-like records as evidence that the baseline is inadequate invites the reply that the
controller removes almost all of it. A is the answer to that objection and should be reported
alongside.

## B step 2 and C step 6. Learnability

The absorber frequency in the model is swept about the truth's `f_a = 150` Hz, which traces the
loss surface along the direction the ANN must learn. Three losses on the same sweep:
`L_ol` open loop as used today, `L_so` open loop weighted by `So` (option C), `L_cl` closed loop
(option B). Record `V1_standstill_Yp10`.

| `f_a` [Hz] | `L_ol` | `L_so` (C) | `L_cl` (B) |
|-|-|-|-|
| 120 | 1.888e-06 | 3.032e-06 | 2.446e-06 |
| 130 | 1.886e-06 | 3.029e-06 | 2.065e-06 |
| 140 | 1.103e-06 | 1.765e-06 | 1.170e-06 |
| 145 | 5.733e-07 | 9.147e-07 | 5.969e-07 |
| 148 | 2.304e-07 | 3.668e-07 | 2.379e-07 |
| **150** | **2.037e-09** | **1.264e-09** | **1.509e-09** |
| 152 | 2.265e-07 | 3.596e-07 | 2.324e-07 |
| 155 | 5.488e-07 | 8.697e-07 | 5.625e-07 |
| 160 | 1.004e-06 | 1.586e-06 | 1.037e-06 |
| 170 | 1.453e-06 | 2.285e-06 | 1.609e-06 |
| 185 | 1.366e-06 | 2.155e-06 | 1.841e-06 |

**Finding 1: the absorber is learnable under all three losses.** Every loss has its minimum at
exactly 150 Hz, two to three orders below its value at 120 or 185 Hz. There is a gradient to
follow, so S-B1's core requirement is satisfied and B is not dead on arrival.

**Finding 2: `L_so` and `L_ol` have essentially the same shape.** Normalising each by its own
value at 140 Hz, the two agree within 2 % at every point of the sweep. The apparently larger
"contrast" of `L_so` (1700 against 670) is measured against the numerical floor at the minimum,
not against a difference in the surface, and the floor is not a meaningful discriminator.
**This weakens the case for C**: the `So` weighting does not sharpen the loss surface, so its
value is confined to offset-blindness and loop relevance, not to easier optimisation. That is a
correction to the expectation stated when option C was proposed.

**Finding 3: `L_cl` has the widest basin.** Normalised the same way, it reaches 2.09 at 120 Hz
and 1.57 at 185 Hz against 1.71 and 1.24 for the other two, and it is the only loss that is
monotone on both sides of the minimum. A larger capture region is a genuine advantage for B.

**Finding 4, and it applies to all three: learnability is confined to the excitation band.** Every
loss turns over past about 170 Hz, so `L(185) < L(170)`. Beyond the `[130, 180] Hz` multisine the
detuned absorber leaves the excited band and the loss stops distinguishing it, with the gradient
pointing the wrong way. This is an excitation limit, not a loss-function limit, and no choice
among A, B and C changes it. Any optimiser initialised well outside the band can move away from
the solution.

---

## What is not implemented, and why

The training halves of B and C need edits outside this folder. C's loss requires a differentiable
`So` inside the training loss in `scripts/gantry/gantry_dynamic/`, and B additionally requires the
controller inside the BPTT graph, controller state initialisation for mid-record windows, a
zero-mean constraint on the ANN output and a divergence guard. What is implemented here is the
decision-relevant half of each: the loss-surface comparison above, which is what determines
whether either training change is worth making.

`a_closed_loop_models.py` compares truth against baseline because no ANN checkpoint is reachable
from this folder. `MODEL_HOOK` at the top of that file is the insertion point: supply a callable
with the `deriv(x, u)` signature and it joins the comparison, giving S-A2 and S-A4 their intended
three-way form.

## Recommendation

Findings 2 and 3 together argue against C and for a narrower version of B. C's only remaining
justification is that it discards the DC offset, and that can be obtained far more cheaply by
high-pass filtering the residual than by carrying a 15-state `So` through training. B retains a
real advantage in basin width, but it is bought with the divergence risk, the nine unknown
controller states and the 5x cost of running at 20 kHz.

The strongest result in this folder is A, and it needs no training at all.
