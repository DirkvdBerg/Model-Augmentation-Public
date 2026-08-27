# Augmentation training: verified status, open problems, uncommitted code

**Date**: 2026-08-20. **Scope**: why the ANN plus encoder does or does not train on the gantry
closed-loop augmentation, what is measured, what is assumed, and what sits uncommitted.

## 0. How to read this file

This file holds **conclusions**. Measurements stay in their session folders and are append-only.
Where an earlier document disagrees with this one, this one is later and the disagreement is
listed in section 6 rather than left implicit.

| Document | Role now |
|-|-|
| this file | current conclusions, the one place they are revised |
| `docs/gantry-augmentation-problem-log.md` §12 | the run table, append-only, hypothesis before launch and outcome after |
| `docs/gantry-augmentation-problem-log.md` §15 | the long-form 2026-08-20 record with full tables |
| `scripts/gantry/closed-loop-controller/transient-investigation/runs/*.json` | raw measurements, never revised |
| `ANN-learning-issue/{RESULTS,HYPOTHESES-AND-SOLUTIONS}.md` | session records, **contain superseded claims**, see §6 |
| `transient-investigation/RESULTS.md` | session record, **contains a false citation**, see §6 |
| `docs/decisions.md` | D-072, D-142, D-150, D-151 |

Every number below was measured on this machine and names the artefact it came from. Literature
claims name the page or equation and were checked against the PDF in the session that wrote them.

## 1. What works

The **static** augmentation works. This is not in doubt and it is the bulk of the result so far.

| Quantity | Value | Source |
|-|-|-|
| Free-run validation RMS, untrained | `2.1866011e-06 m` | `closed_loop_free_run_rms`, V1-V4 |
| Free-run validation RMS, trained | `1.3933793e-06 m`, **-36.3 %**, best at epoch 3 | run table §12 |
| Acceptance target | `1.215e-06 m` (45 % of headroom) | run table §12 |
| Data-derived floor | `2.81e-08 m` | run table §12 |
| ANN correction rows, gradient | `8.55e-06`, coherence `0.954` over 8 disjoint batches | `consistency_probe.json` |
| Physical encoder `W^b`, gradient | `4.09e-06`, coherence `0.825` | `consistency_probe.json` |
| D-072 baseline equality | `2.186601103417735e-06`, bit-identical with D-150 and D-151 | `bootstrap_probe.json` |

That improvement comes **entirely through the ANN's correction on the physical rows**, a function of
`(x, u)`. Nothing in section 2 touched it.

The closed-loop rollout is also on solid theoretical ground, see §4.

### 1.1 What the error is made of, and why the target is reachable

`error_budget.json`, untrained model, free run over the full validation records:

| | value |
|-|-|
| free-run error RMS, aggregate | `2.188726e-06 m` |
| in-band `[140, 175] Hz` component | `1.989087e-06 m` |
| **band share of error POWER** | **0.826** |
| per-record share, min / median / max | 0.718 / 0.809 / 0.837 |
| error with the in-band component removed entirely | **`9.132650e-07 m`** |

**Five sixths of the error is the augmented mode**, consistently across every validation record and
all three channels. Two consequences.

1. **The acceptance target is reachable through this mode.** Fully accounting for the band gives
   `9.13e-07`, below the `1.215e-06` target. The augmented-state effort is aimed at the right part
   of the spectrum.
2. **The data is not the constraint.** The thing we are trying to learn is already the dominant
   feature of the residual, so no amount of extra excitation helps; more excitation cannot add
   information about a signal that already accounts for 83 % of the error. The constraint is the
   method, which points at capacity (P6), the objective (P3) and the encoder (P4), not at the
   dataset. This supersedes the pessimistic reading in P7.

It also resolves the tension noted in P7: `cl_band_split.py` reporting the loss weighing the 150 Hz
peak two to three decades above everything else is exactly 82.6 % of the power sitting in a 35 Hz
slice of a 2000 Hz band. The two statements agree.

**Open follow-up**: this is measured on the UNTRAINED model. The ANN correction enters the state
update as a function of the full state, so it acts as state feedback and can shift the closed-loop
poles, which is the likely source of the 36 %. Measuring the in-band share at the plateau checkpoint
would say how much in-band energy is left for the augmented states to claim. Not run.

## 2. The problems, stated

### P1. The augmented states contribute nothing

Two structural facts at initialisation, both measured, not inferred:

* **Unobserved.** `y` reaches `x_a` only through ANN output rows 0-5, whose final layer is exactly
  zero (D-072), so `dy/dx_a` is exactly zero. Measured: `||grad||` from the settled output loss is
  **exactly `0.0000e+00`** on `W^a` and on `nu_log`/`theta_log`, with and without D-151.
* **Undriven.** `gamma * NL = 0` at init, so `x_a` is an autonomous ringing seeded only by
  `W^a psi`. Nothing in the data drives it, so it carries no plant information.

This is a bootstrap, not a permanent dead end: after one update the readout leaves zero and the
Jacobian `dw/dx_a` climbs (`0 -> 1.71e-05 -> 1.35e-04` over 40 steps). The problem is what the
readout grows **onto**.

### P2. Without excitation, the pole gradient is noise

Coherence `||mean_b g_b|| / mean_b ||g_b||` over 8 disjoint batches, isotropic-noise reference
`1/sqrt(8) = 0.354`, positive controls in the same table:

| | control | D-151 injected, `na_nb = 17` | injected, `na_nb = 32` |
|-|-|-|-|
| LRU `nu/theta` | **0.086** | 0.827 | **0.994** |
| `nu_log` sign agreement | 50 % | 100 % | 100 % |
| `theta_log` sign agreement | 50 % | 62 % | **100 %** |
| `W^a` | 0.473 | 0.535 | 0.700 |
| ANN MLP (control) | 0.954 | 0.712 | 0.949 |

Source: `consistency_probe.json`, `na_sweep_probe.json`.

### P3. The objective prefers to remove the augmented mode

With D-151 the pole finally has a direction, and the direction is `dL/d(nu_log) < 0` on **every**
batch, so descent increases `nu_log`, decreases `r`, and damps the mode. A randomly driven
resonator uncorrelated with the residual is a nuisance regressor and the cheapest local
improvement is to attenuate it. One snapshot, 8 batches; not yet confirmed over a training run.

### P4. The encoder cannot reconstruct the augmented state

`na_nb` defaults to `(nx_phys + nx_ann)*2 + 1 = 17` samples (`model.py:154`, Jan's `nxd*2+1`, a
**state-count** rule with no timescale content). At `fs = 4 kHz` that is 4.25 ms against a mode
period of 6.47 ms, i.e. 0.66 of a period.

The full sweep (`na_sweep_probe.json`), coherence by `na_nb`:

| group | 17 (0.66 p) | 32 (1.24 p) | 64 (2.47 p) | 128 (4.94 p) |
|-|-|-|-|-|
| W^b | 0.620 | 0.696 | 0.649 | 0.709 |
| W^a | 0.535 | 0.700 | 0.564 | **0.775** |
| encoder net | 0.691 | 0.568 | 0.462 | 0.558 |
| ANN MLP | 0.712 | 0.949 | 0.972 | 0.732 |
| LRU `nu/theta` | 0.827 | **0.994** | 0.938 | **0.289** |
| `theta_log` sign | 62 % | 100 % | 88 % | 62 % |

**Verdict: inconclusive, and the "one period" reading is not supported.** After 32 and 64 looked
like a clean threshold story (`theta_log` 62 % -> 100 % as the window crosses one period), 128 at
4.94 periods collapses to `0.289`, below the isotropic-noise reference, with `theta_log` back to
62 %. A mechanism that needs at least one cycle cannot behave that way.

The likely reason is a flaw in the probe rather than in the encoder: **each `na_nb` builds a
different model** (different encoder parameter count, different analytic `W^b`, a recomputed
orthogonal-projection basis) and each runs its own independent 40-step warm-up. The four settings
are four training trajectories, not a controlled sweep of one variable, so the coherence at step 40
partly reflects where each trajectory happens to be. A defensible version would fix the warm-up or
average over seeds.

What survives: `na_nb = 17` places the encoder window below one period of the mode, and the rule
that chose it is a state-count heuristic with no timescale content. That remains a reasonable
suspicion. It is **not** established that changing it helps.

**But a second, cleaner line of evidence says the lag is under-set, and it has a citation.**
`encoder_conditioning.json`, exact rather than Monte Carlo (the encoder's linear block gives
`dx_b = Wb_psi_y @ v`, so the per-channel amplification is the row norm):

| na_nb | vel/pos ratio | ‖velocity rows‖ | vs na=17 |
|-|-|-|-|
| 17 | **29.7x** | 1.4119e+03 | 1.000x |
| 32 | 15.7x | 5.5915e+02 | 0.396x |
| 64 | 7.6x | 1.9342e+02 | 0.137x |
| 128 | 3.5x | 6.0945e+01 | 0.043x |

The amplification falls as `n^-1.57`, which is the classical `n^{-3/2}` law for estimating a
**slope** from `n` noisy samples. The encoder is doing least-squares velocity estimation from noisy
positions and obeys the textbook scaling for it. See §4.3 for the per-channel breakdown and the
`K = 0` connection.

And SUBNET's own authors prescribe exactly this step. Beintema, Schoukens and Tóth 2023
(Automatica 156:111210) §3.2: *"While `n = nx - 1` is often the minimal required number of past IO
samples to obtain an unbiased estimator, the variance of estimate `x_hat` can be rather significant
and further reduced by increasing `n`. The underlying mechanism is similar to the concept of
**minimum variance observers** (Darouach & Zasadzinski, 1997)."* Their user guidelines repeat it
(*"increase `n` to reduce the variance of the initial state estimate"*), and §5.6 links it directly
to our P5: a longer lag gives *"a variance reduction in the state estimate (i.e. encoder as a
minimum variance observer) which **reduces the average transient error**"*, with diminishing returns
in their example beyond about **20x the minimum**.

Our `na_nb = 17` against a minimum of `nx - 1 = 7` is **2.4x the minimum**. Jan's `nxd*2 + 1` is a
rank rule; the method's authors say the rank minimum buys unbiasedness and that `n` should then be
raised for variance. **That step was skipped.** This makes the lag a `# THEORY:`-backed design
quantity rather than a tuning knob, which is what the coherence sweep failed to establish.

### P5. This is the 87.9 % transient, restated

A *correct* model's window loss is 87.9 % startup transient, and the planted model carries 3.9x
more startup energy than the plateau model. That is the definition of an encoder that cannot
initialise the latent state. **P4 and the transient problem are the same problem**, which collapses
two of the four failure links into one.

### P6. Capacity is untested, and P2/P3 now predict it is the binding one

`nx_aug = 2` is one complex pair: **one** resonator, with **one** random input direction, at one
frequency. Everything measured in P2 and P3 is what that predicts. The pole gradient is a coin flip
without excitation because a single undriven random mode correlates with nothing; with excitation
it becomes consistent and asks for the mode to be **damped**, because a single random filter that
happens not to match the residual is a nuisance regressor and the cheapest local improvement is to
attenuate it. A random-feature basis of size one is a lottery ticket.

Two independent reasons to think this is the binding constraint rather than a footnote:

* **The D-150 band recipe assumes coverage, not a sample.** The initialisation draws poles uniformly
  on an annulus over the data-derived band (`# THEORY: Orvieto et al. ICML 2023 Lemma 3.2`). With
  `n_pairs = 1` we draw a single sample from that distribution and commit to it. The recipe only
  does what it was designed to do when several pairs cover the band. Our drawn pole sits at
  `154.52 Hz` inside a band of `[149.90, 164.06] Hz` by luck, not by design.
* **The only directly comparable industrial case disagrees with our setting.** Kessels found
  `n_ext = {0, 2}` "lack sufficient complexity" (p.172), swept to 34, and settled on **14** on an
  ASMPT stage. Jan's `nx_aug = 2` worked because his missing physics was exactly one mode in a
  3-DOF MSD, which is not our situation.

Untested here (H5). This is the largest untried lever, it is a single hyperparameter, and it is
entirely inside SUBNET.

### P7. The reference never excites the band the mode lives in

The applicable MIMO informativity condition for closed-loop data **with** a reference is Colin,
Bombois, Bako, Morelli (Automatica 121:109171, 2020) Thm 3: informative iff, for every
model-difference pair, `dW_y - dW_u K = 0` **and** `E||dW_u r||^2 = 0` implies `dW = 0`. Two
channels, a controller-kernel channel and a reference-excited input channel.

(Bazanella et al. 2010 Thm 3.2, the `kmin > lmax` controller-complexity condition, is **not** our
condition. Read at source: it is stated for identification *"without external excitation"* and is
the requirement for the noise-only case. Our records are reference-driven. An earlier note in
`docs/references.md` framed it as ours; that was reading the theorem for the wrong experiment.)

Measured on the second channel (`reference_excitation.json`, exogenous drive recovered as
`alpha = u_data + C_fb(y_data)`, the identity verified in §4.1), band `[140, 175] Hz` which brackets
the residual band, as a fraction of total power over 18 records x 3 channels:

| | value |
|-|-|
| min | 5.05e-11 |
| median | 5.54e-09 |
| max | 4.43e-06 |

Six to ten decades down, in every record. The record names agree: `aprbs_30/60/100` are band-limited
far below 154 Hz, and the sweeps and lissajous profiles are motion profiles.

A second finding, not looked for: **the standstill and ysweep records are near-rank-one in
excitation**, `cond(cov)` from `1.2e+10` to `5.6e+13`, with channel 3 exceeding channels 1 and 2 by
three orders (`2.26e+06` against `3.47e+03`). The best-conditioned records are `T12_aprbs_yaw`
(`3.8e+04`) and `T14_lissajous_yaw` (`5.9e+04`).

**Reading, CORRECTED by §1.1.** The first draft of this section concluded that the reference
channel of Colin Thm 3 carries essentially nothing about the augmented mode and that discrimination
therefore rests on the controller-kernel channel alone. **That implication is wrong.** Band
*fraction* of the drive is the wrong statistic: a lightly damped resonance turns tiny excitation
into a dominant output feature, and §1.1 measures the mode at **82.6 % of the free-run error
power**. The signal is enormous and sitting in `y`.

What survives is narrow and still worth recording: the exogenous drive is band-limited well below
154 Hz in every record, so the mode is rung by broadband transients (APRBS steps) rather than by
designed excitation, and the standstill and ysweep records are near-rank-one in excitation
(`cond(cov)` up to `5.6e+13`, channel 3 exceeding the others by three orders). The second fact is a
real property of the training set and may matter for the physical parameters; it does not limit the
augmented mode.

**Consequence for the plan**: adding excitation at 154 Hz is not a fix. It would not transfer to
Telica (`telica_plant_frf.py`: identifiable band below 83 Hz on X, below 55 Hz on Y), and more
importantly it is unnecessary, since the mode already dominates the residual.

The tension with `cl_band_split.py` is resolved in §1.1: 82.6 % of the power in a 35 Hz slice of a
2000 Hz band **is** a peak two to three decades above the rest.

## 3. What was tried, and why each did not move the number

| Fix | Mechanism worked? | Result |
|-|-|-|
| D-150 live `A_aa` | yes, `rho` 0 -> 0.9920 | free run `-0.665 %` |
| Burn-in `K = 100` | yes, discrimination 1.56x -> 11.56x | never trained successfully |
| Multiple-shooting defect | yes, `||grad W^a||` 0 -> 1.83e-01 | epoch 1 validation **2.8x worse** |
| D-151 `B_a` injection | yes, pole coherence 0.086 -> 0.827 | not yet run as an arm |

The defect deserves its own line: it is **degenerate**, not merely mis-weighted. With `x_a`
undriven, `d_a,j = enc_a(psi_j) - A_aa^nf_seg enc_a(psi_{j-1})` is minimised by `enc_a == 0`.
Measured `<grad_{W^a} L_defect, W^a> = +1.983` (positive, so descent shrinks `W^a`), and both
`L_defect` and `RMS(enc_a)` fall about 2.3 % per 15 updates on matched batches. It should be
deleted, not retuned.

**The pattern is itself a finding.** Four mechanisms, each verified to do what it was designed to
do, and the free-run number unchanged.

## 4. Noise and real data

### 4.1 The estimator is the right one, and this is now measured

`closed_loop_rollout` drives the model with `u_data + C_fb(y_data - y_model)`. That is the
**stabilized PEM** of Sugie & Maruta 2020 (`literature/closed-loop-id/sugie2020_dual-youla-simplified.pdf`,
section 3, Fig. 6), which they derive from and prove equivalent to dual Youla. Their Eq. (8) gives
the reason it is noise-robust: a specific combination of the recorded `u` and `y` reconstructs the
exogenous signal and is therefore free of measurement noise.

Tested on our own code (`alpha_cancellation.json`) by perturbing the record the way a real machine
would, `y -> y + v` **and** `u -> u - C_fb(v)`, across four decades of sigma:

| sigma [m] | A whole-record, physical | A, noise on `y` only | A cancellation |
|-|-|-|-|
| 1e-8 | 8.14e-08 | 1.49e-07 | 1.8x |
| 1e-7 | 9.67e-08 | 1.21e-06 | 12.5x |
| 1e-6 | 8.08e-08 | 1.20e-05 | 149x |
| 1e-5 | 8.13e-08 | 1.20e-04 | 1477x |
| 1e-4 | 9.54e-08 | 1.20e-03 | 12590x |

The physical residual is **pinned at the float32 roundoff floor across four decades** while the
naive one scales linearly. The cancellation is exact. Sugie's property is ours.

**And this is load-bearing for SUBNET's own consistency, not a bonus.** Beintema, Schoukens and
Tóth 2023 Thm 4 (Convergence), on which their consistency Thm 8 rests, assumes *"a quasi-stationary
`u` independent of the white noise process `e`"*. That is an **open-loop** assumption and it fails
in closed loop, where `u` is a function of `y` and therefore of `e`. So SUBNET's consistency result
does **not** cover closed-loop training as written. What restores it is exactly the stabilized-PEM
structure: the model is driven by the reconstructed exogenous signal, which is independent of `e`,
measured above to the float32 floor. The closed-loop rollout is therefore a requirement of the
statistical argument, not an implementation detail. This belongs in the thesis.

### 4.2 But our windowing throws most of it away

Same run, configuration B, windowed at `nf = 400` with `xc = 0` at every window start (D-142):

| sigma [m] | B physical | B cancellation |
|-|-|-|
| 1e-8 | 1.02e-07 | 1.4x |
| 1e-7 | 6.15e-07 | 2.0x |
| 1e-6 | 6.08e-06 | 2.0x |
| 1e-5 | 6.08e-05 | 2.0x |

B's residual scales **linearly** with sigma and the cancellation saturates at **2.0x**. The two
configurations differ in nothing but the controller-state reset. So resetting `xc` every 400
samples converts an exact noise cancellation into a factor of two. D-142 argued `xc = 0` is the
unique consistent choice for the residual form; that argument stands, but its cost was never
quantified and it is large.

### 4.3 The encoder is an unprotected noise channel

`||dx0||/||x0|| = 5.30e-04` for an input perturbation of `2.76e-07`: an amplification of
**1919.8x**, constant to four significant figures across four decades of sigma, so it is a
condition-number property of the reconstructability map, not a nonlinearity. The encoder reads the
`y` history directly and no cancellation is available to it.

**It is a `K = 0` effect, and it is entirely in the velocity rows.** Per-channel amplification at
`na_nb = 17`, relative to each state's own RMS (`encoder_conditioning.json`):

| state | relative amplification |
|-|-|
| `dY` | **7.76e+03** |
| `dX` | 3.40e+03 |
| `dTheta` | 1.28e+03 |
| `X` | 33 |
| `Y` | 0.28 |
| `delta_a`, `vdelta_a` | 0.92, 0.73 |

X and Y have no stiffness, so their position/velocity pairs are double integrators: position is
measured but velocity lives in the differences, of order `Ts*v = 2.5e-04 * v` at 4 kHz, and
inverting the observability matrix for it is badly conditioned. `W^b = A^n O_n^{-1}` is that
inverse. The same effect is already recorded independently in `gantry_dynamic/data.py` for direct
differencing (SNR 60 gives `dTheta` 193x). The augmented rows show **no** amplification (`0.92`,
`0.73`) because `W^a` is a random draw rather than an observability inverse.

Consequences. The decoupling argument that justifies burn-in (initial-condition error decays, model
error persists) is weakest exactly here: for the `K = 0` axes that decay comes entirely from the
controller, not from the plant, whose `z = 1` modes never decay, and a velocity error on an
integrator produces a position **ramp**, which is low-frequency content where the loop is slowest.
Worse, a small ANN force correction on a `K = 0` axis double-integrates into a large position
change, so the channel with the highest gain from "ANN compensates" is the same channel where the
encoder is worst conditioned. That is the concrete mechanism by which the encoder and the ANN
absorb each other's errors, and D-103 requires routing to X and Y, so it cannot be closed off.

This reframes the project's history: D-066, D-067 and D-068 treated `K = 0` as a **drift** problem
in forward simulation. It is also an **estimation** problem in the encoder, and that has never been
addressed.

Note this is a **noise** argument, and the simulation data is noiseless, so it predicts a large
effect on Telica and possibly none in simulation.

### 4.4 The existing noise gate does not test the physical scenario

`cl_train.py:246-248` adds noise to `sd.y` only and leaves `sd.u` as recorded from a noiseless
simulation. On a real machine the controller sees the noisy output and its reaction is already
inside the recorded `u`. So the gate injects a spurious `+C_fb(v)` into the model's drive that a
real machine would have cancelled, i.e. it tests a harder and non-physical problem. D-150's
"survives Telica-level noise" rests on this gate and should be re-read accordingly.

### 4.5 MIMO: SUBNET does not treat it, and our MIMO problems are measured

Read at source (Beintema, Schoukens, Tóth 2023): **MIMO appears only as motivation** in the
introduction (*"well applicable for multiple-input multiple-output (MIMO) systems"*). No
MIMO-specific analysis, no per-channel weighting, no MIMO experiment. SUBNET is simply
vector-valued. So there is no MIMO gap in the method, and equally no MIMO help from it.

Our MIMO problems are ours to solve, and both are measured:

* **Nine decades across correction rows.** `model.py:263-266`, citing `cl_capability.py`: the eight
  corrections span nine decades in normalised units, and regressing this architecture onto the exact
  target fits the absorber rows to `1-R^2 = 9.6e-05` **with** a per-row scale and fails completely
  (`0.98`) without one. That is why the per-row ReZero gate exists.
* **Per-channel encoder conditioning spans `2.8e+04`** within one encoder (§4.3).

The nearest precedent is Kessels Eq. (5.15), which weights each output by `1/y_RMS[i]`. Our `ystd`
normalisation already does that **for outputs**. Nothing does it for **state rows**, which is where
the nine decades are.

## 5. Uncommitted code

Nothing has been committed or pushed. `git diff --stat` on the framework:

```
model_augmentation/fit_systems/closed_loop.py  | 104 +++++++++++++-
model_augmentation/fit_systems/pre_encoder.py  |  47 ++++++-
```

plus `scripts/gantry/gantry_dynamic/model.py` (D-150 `AugLRUBypass`, D-151 `B_a`) and the
`transient-investigation/` and `ANN-learning-issue/` folders, all untracked.

| Item | Provenance | Verdict |
|-|-|-|
| `closed_loop.py` `closed_loop_rollout` | Kessels Eq. (5.13d) p.157, checked; and §4.1 shows it is Sugie's stabilized PEM | **keep**, now the best-supported piece in the tree |
| `closed_loop.py` `xc = 0` | labelled `# HEURISTIC:` in code, real derivation (unique consistent choice in residual form) | keep, but §4.2 quantifies its cost and that belongs in D-142 |
| `pre_encoder.py` `W^b` | Hoekstra Eqs. 16-17, `# THEORY:` labelled | **keep** |
| `pre_encoder.py` `W^a` kaiming | labelled `HEURISTIC, with no literature source`, flagged load-bearing in the file itself | keep; matches Hoekstra Eq. (31)'s Xavier draw in spirit |
| `model.py` D-150 `AugLRUBypass` | Orvieto ICML 2023 Sec. 3.3/3.4 and Lemma 3.2; split is Hoekstra S-DP | **keep** |
| `model.py` D-151 `B_a` | Hoekstra Sec. 5.4.3 keeps the linear part live; scale is `HEURISTIC` | keep, env-gated and off by default |
| `transient-investigation/train_combined_arm.py` defect | measured degenerate, §3 | **delete before committing** |
| burn-in `K = 100` | citation false, see §6 | keep the code, fix the label to `HEURISTIC` |

The three marker mechanisms required by `CLAUDE.md` are present: `closed_loop.py` carries one
`# CHANGED:` marker, `pre_encoder.py` seven. **Not audited by me**: whether every one of the 145
changed lines is marked, and the contents of the other untracked scripts.

**Recommendation**: commit the rollout, the encoder init and D-150 with the citation fixed and the
defect removed, and record the negative results in the run table as negative results. Do not
commit anything phrased as "solved". Blocked in any case on another session's work in
`gantry_dynamic/{config,evaluation,orth_penalty}.py`.

## 6. Corrections to earlier records

1. **The Kessels burn-in citation is false.** `transient-investigation/RESULTS.md` §6 credits
   Kessels with an "evaluation delay `tau = n_o`" separating startup transients. Kessels Eq. (5.14),
   thesis p.160, is `V_N = 1/(N - n_o) sum_{k=n_o+1}^{N} ||W(y_bar_k - y_hat_k)||^2` with
   `tau = n_o` the **encoder history length** (Fig. 5.4, p.158). The sum starts there because the
   model cannot be simulated earlier. No transient argument exists, and the training loss `V_T`
   (5.12) has no offset at all. **Burn-in is ours and must carry `# HEURISTIC:`.**
2. **"The defect solves the gauge and oracle problem"** (`transient-investigation/RESULTS.md` §3.2)
   is false; it is degenerate, see §3.
3. **`lambda_defect = 1e-5`** (same file, §5.2) is wrong by two to three decades; per-group parity
   spans `2.2e-08` to `4.8e-06` and there is no single parity value.
4. **Parameter-movement counts are not evidence.** "108/108 `W^a` entries moved" and "3016/3130
   encoder parameters moved" measure gradient non-zeroness; under Adam any non-zero gradient moves
   every parameter by about `lr`. The epoch-1 `Wa_delta = 4.22e-02` is exactly `416 x lr_enc`.
5. **"`rho` held at 0.9920" is not evidence of health.** Its gradient was exactly zero.
6. **`x_a` is a gauge, `rho` is not.** `x_a` is defined only up to an invertible transformation, so
   it has no ground truth and does not map 1:1 onto the absorber state. Eigenvalues are invariant,
   which is the only sense in which the D-150 band initialisation is meaningful.
7. **Superseded acceptance criterion.** "`rho(A_aa)` above 0.5" is passed by a model whose augmented
   state is identically zero, and `RMS(x_a)` is gauge-dependent. Use an **ablation test**: zero the
   readout's augmented columns in the trained model and re-run the free run. No degradation means
   the augmented states are decoration.
8. **"D-151 costs the working paths coherence"** (problem log §15, written earlier today) was an
   artefact of `na_nb = 17`. At `na_nb = 32` the ANN MLP returns to `0.949`.
9. **`chen2024dualiop` does not help us.** It avoids coprime factorisation by also discarding the
   nominal plant ("neither requires a doubly-coprime factorization of the controller nor a nominal
   plant"), and the nominal plant is the object whose parameters we need.

## 7. Open questions, and what would settle each

| Question | What settles it |
|-|-|
| Do the augmented states help at all? | A D-151 training arm with the **ablation** test of §6.7, pre-registered, stop if epoch 1 is worse than untrained |
| Is `na_nb = 17` wrong? | The sweep is inconclusive (§2 P4). A clean version fixes the warm-up or averages seeds; the arm at 17 and 32 answers the only question that matters, which is accuracy |
| What does the `xc = 0` reset cost in accuracy, not just noise rejection? | §4.2 measured the noise cost; the accuracy cost needs a whole-record versus windowed training comparison |
| Does D-072 survive `na_nb != 17`? | Free-run gate per setting; `W^b = A^n O_n^{-1}` is exact for any `n` above the observability index but conditioning varies |
| Is the objective's sensitivity weighting the real ceiling? | The CLIE / effort term (`landau2002duality` Eq. 32, `zang1995iterative` Eq. 27), measured at 11.27x against 10.14x for burn-in alone |
| Is `nx_aug = 2` enough? | H5, untested |
| Is the encoder lag under-set for **variance**, per Beintema §3.2? | Measured `n^{-3/2}` in §2 P4; the arm at 17 versus a larger `n` answers whether it costs accuracy. Note this is a noise argument and the sim is noiseless |
| ~~Is the reference channel empty?~~ | **Answered in §1.1**: the mode is 82.6 % of the error power. Excitation is not the constraint and no dataset change is warranted |
| ~~Does `cl_band_split.py` contradict P7?~~ | **Answered in §1.1**: no, the two agree |
| How much in-band energy is left at the PLATEAU, not untrained? | Re-run `probe_error_budget.py` on the plateau checkpoint. Decides how much of the remaining headroom the augmented states must actually claim |
| Do the state rows need per-row weighting (nine decades, §4.5)? | The per-row ReZero gate already exists (`ANN_REZERO_GATE=row`); whether the **loss** needs it is untested |
| Does any of this survive real noise? | The gate in §4.4 must be fixed to perturb `u` and `y` consistently before it means anything |

## 8. The next arm: `nx_aug`

**Status: PROPOSED, not decided, and it depends on two gates the user has not approved.**
`AUG_LRU` (D-150, the pole) and `AUG_LRU_B` (D-151, the injection) are env-gated and default OFF.
Neither is in the implementation. Nothing below is agreed, and §8.0 is the decision that comes
first.

### 8.0 The prior decision: what the DEFAULT implementation does

With both gates off, which is the current committed behaviour, the augmented rows are written by
the zero-initialised ANN output alone. That means `rho(A_aa) = 0` **exactly**: the augmented
"states" have no dynamics at all, they are a one-step hold of an ANN output that is itself zero at
initialisation, and `W^a` receives exactly zero gradient. In the default implementation the dynamic
augmentation is structurally incapable of representing a mode, so the ceiling is the 36 % static
result in §1. That is not a tuning problem and no objective fixes it.

So the real decision is **how the augmented rows get dynamics at all**.

**What baseline equality actually requires in our wiring** (corrected 2026-08-20; an earlier draft
of this section got it wrong). In our interconnect `x_{k+1}` rows 0-5 are the baseline plus the
ANN's routed output, rows 6-7 are the ANN's routed output alone, and `y = Cd_norm x + Dd u` where
`Cd_norm` has **zero columns on the augmented states** (the trainable `C_aug` was removed,
`model.py:242-244`). Therefore D-072 requires exactly one thing: **rows 0-5 of the ANN output are
zero at initialisation.** Rows 6-7 may be anything and `y` is untouched.

That is what `AugLRUBypass` does. So "keep the learning function's linear part live while the
physical rows stay zero", which is Hoekstra Sec. 5.4.3 (`phi_aug(z_a) = 0 + W_a z_a`, ResNet, with
baseline equality carried by the LFR matrix conditions `B̃ = I`, `C̃ = I`), **is not an unattempted
alternative structure. It is what D-150 plus D-151 already is.**

The genuine difference is narrower, and it is how that live linear map is drawn:

| | Hoekstra as written | D-150 + D-151 |
|-|-|-|
| `A_aa` | random, from LFR matrices `~ U(-1,1)` | stable by construction, `lambda = exp(-exp(nu) + i exp(theta))`, ring-initialised over a data-derived band |
| `B_a` | random, same draw | `N(0, 1/nz)`, scaled to `RMS(x_phys)` |
| stability | not guaranteed | `abs(lambda) < 1` by construction |

Ours is a **more constrained** version of the same structure, and the constraint is motivated:
Orvieto et al. ICML 2023 is the argument that a dense random `A` is the wrong draw for a recurrent
block. An unconstrained draw with `abs(lambda) > 1` inside a 400-step closed-loop rollout diverges,
a risk an open-loop 3-DOF MSD tolerates more easily than we do.

**So the open question is not "restructure or not", it is whether our constraints earn their keep**,
and that is testable with no new code: `AUG_LRU_BAND` and `AUG_LRU_RHO` already override the band,
so a Jan-faithful draw (phase over the full circle, radius over a wide annulus) can be run against
the band-initialised draw directly. If the plain draw does as well, the band recipe is not
contributing and the novelty claim in `ANN-learning-issue/HYPOTHESES-AND-SOLUTIONS.md` §7 weakens.
If the band draw wins, that claim gets its first direct evidence.

**The question for Jan is correspondingly narrower**: his initialisation draws the augmented
dynamics at random and ours draws them from a residual band under a stability constraint; is the
constraint sound, and did he ever observe instability from the unconstrained draw?

The two real options remain: accept a static augmentation and report 36 %, or keep a dynamic one,
in which case some live draw on the augmented rows is mandatory and the only question is which.

### 8.1 The proposed arm (conditional on the above)

Varies **augmented-state capacity**, not another initialisation detail. Rationale in §2 P6: the measured behaviour of the pole gradient is what a
single random resonator predicts, the D-150 band recipe is a distribution that one pair cannot
cover, and the one comparable industrial case (Kessels) rejects `n_ext = {0, 2}` and settles on 14.

**Configuration.** `nx_aug` in `{2 (control), 8, 14}`, even by construction since D-150 pairs the
augmented states. D-151 injection ON at the measured scale, D-150 band init, burn-in `K = 100` with
Strategy B, `lr_enc = lr_ann = 1e-5`, Adam `eps = 1e-16`, **no multiple-shooting defect** (§3).
Everything else at defaults.

**Pre-registered acceptance**, in this order:

1. **Ablation**, the primary and gauge-free test: zero the readout's augmented columns in the
   trained model and re-run the free run. If the error does not degrade, the augmented states are
   decoration and capacity was not the constraint either.
2. Free-run sim-RMS on V1-V4 against `1.3933793e-06` (current plateau) and `1.215e-06` (target).
3. Epoch 1 worse than `2.1866011e-06` is a **stop condition**, not a phase to sit through.

**Confound to record before launching.** `na_nb` is derived as `(nx_phys + nx_aug)*2 + 1`, so
raising `nx_aug` also lengthens the encoder window: `nx_aug = 8` gives `na_nb = 29` (1.12 periods)
and `nx_aug = 14` gives `41` (1.58 periods). Capacity and encoder lag therefore move together. That
is Jan's rule operating as designed, so the first arm should let it, but the result cannot separate
P4 from P6. If capacity wins, a follow-up with `na_nb` pinned at 17 separates them.

**Also to verify per setting**: D-072 bit-identity (the free-run gate), since `W^b = A^n O_n^{-1}`
is rebuilt at every `nx_aug`, and the orthogonal-projection basis is recomputed.

**Run-table row goes in `docs/gantry-augmentation-problem-log.md` §12 before launch**, stating this
hypothesis, per the run-discipline rule.

Everything else in this file is diagnostics about diagnostics. Four favourable diagnostics have
already failed to move the free-run number, so the next thing run should be an arm with a
falsifiable primary criterion, and the ablation test is that criterion.
