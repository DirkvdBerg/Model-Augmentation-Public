# Handoff: which term of the baseline-versus-MSD equation difference produces the open-loop offset

**From**: session of 2026-07-30 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

Replaying the MSD plant's recorded input open loop through the 6-state baseline produces a
position offset on X and on Y. Determine, term by term, which part of the difference between the
two models' equations of motion produces it, on each output channel. The difference is fully
enumerable and is written out in section 4: it is three candidate terms and nothing else. **The Y
channel is solved and the mechanism is exact. The X channel is not, and Theta has never been
examined at all.** Finish the attribution on X and Theta, and then bring
`scripts/gantry/msd-offset/ISSUE.md` and `docs/msd-offset-mechanism-2026-07-29.md` into line with
what the artefacts support, including the four corrections in section 6.

## 2. Out of scope

- **The drift-isolation programme** (`scripts/gantry/drift-isolation/`, tests T1 to T6). That
  thread is about the *unbounded* drift the augmentation introduces. This one is about the
  *bounded* offset the baseline shows with the ANN at zero. Do not modify anything there.
- **The ANN causal experiment** (arms A/B/C on latent seeding). Not authorised.
- **Extending the latent-eigenvalue measurement to other checkpoints.** Belongs to
  drift-isolation; see section 8.
- **Regenerating the MATLAB records.** `gtd_run_simulation.m` was fixed this session; regeneration
  is a user decision, not a step here.
- **`load_mat_aug`'s `vdelta_a` recomputation.** Real defect, only meaningful after regeneration.
- Do not modify `scripts/gantry/gantry_dynamic/*` beyond the comment already added to `data.py`.

## 3. Where things stand

Branch `Augmentation`, last commit `42b3396`. Tree dirty in `scripts/gantry/msd-offset/` (new,
untracked), `Matlab-scripts/Augmentation/data/gtd_run_simulation.m` (one line),
`scripts/gantry/gantry_dynamic/data.py` (comment only), `docs/` (one new file). **No runs in
flight.**

## 4. Established and verified

### 4.1 The equation difference, complete

Truth is `Matlab-scripts/Augmentation/gantrySystemExtended.m`, 4 DOF `q = [X, Theta, Y, delta_a]`,
payload split `mh_rigid + ma`. Baseline is `kamtin-fp-model/03 Simulink gantry/functions/gantrySystem.m`,
3 DOF, rigid payload at the full `mh`. Both freeze `M` at the current state, so neither carries
Coriolis or centrifugal terms.

Subtracting the baseline 3x3 mass matrix from the truth's upper 3x3 block leaves **exactly two
nonzero entries**:

```
dM(1,2) = dM(2,1) = -ma*(L0 + delta_a)
dM(2,2)           =  ma*[2*Y*(L0 + delta_a) + (L0 + delta_a)^2]
```

`dM(1,1)`, `dM(1,3)`, `dM(2,3)` and `dM(3,3)` are identically zero: the mass split conserves total
mass, so at DC the Y axis sees the same inertia as the baseline.

The truth additionally has the absorber column and its own row, absent from the baseline entirely:

```
M(2,4) = -ma*d      M(3,4) = ma      M(4,4) = ma      K(4,4) = ka      C(4,4) = ca
```

And the property that turns any of this into a permanent offset rather than a transient:

```
K(1,1) = K(3,3) = 0        (X and Y have no stiffness: poles at s = 0)
K(2,2) = kb1 + kb2         (Theta is sprung)
```

That is the entire difference. Three candidate mechanisms live in it:

| tag | term | character |
|-|-|-|
| **A** | the absorber column `M(3,4) = ma`, `M(2,4) = -ma*d`, with `ka`, `ca` | linear, constant coefficients |
| **B** | the `L0`-only part of `dM(1,2)` and `dM(2,2)` | constant parameter perturbation on the X-Theta block |
| **C** | the `delta_a`-dependent part of the same two entries | bilinear in `delta_a` and `qddot` |

Numerically at `ma_frac = 0.10`, `Y = 0.3`: B is `-ma*L0 = -0.101 kg*m` on the X-Theta coupling
(3.1 % of the baseline entry) and `+ma*(2*Y*L0 + L0^2) = +0.0707 kg*m^2` on the Theta inertia
(1.5 %).

### 4.2 Y is solved: mechanism A, through momentum

The truth's Y row reads `(mh_r+ma)*Yddot + ma*delta_addot - (mh_r+ma)*d*Thetaddot + cy*Ydot = F_Y`.
Integrating once, the truth's Y momentum is `mh*Ydot + ma*delta_adot`; the baseline's is `mh*Ydot`.
The deficit is `ma*delta_adot`. With `K(3,3) = 0`, nothing restores position, so an initial
momentum excess `p` leaves a permanent shift `p/c` (because `integral(v dt) = (p/m)*(m/c) = p/c`,
independent of `m`):

```
dY(inf) = ma * vdelta_a(t0) / cy            slope ma/cy = 0.101 s
```

| record | R2 against the unfitted line | max residual | seeds |
|-|-|-|-|
| V1_standstill_Yp10 | **1.000000** | 5.8e-07 m | 60 |
| T10_aprbs_60 | **0.999997** | 8.2e-06 m | 60 |

Artefact: `simulations/gantry_subnet/diagnostics/msd_offset_figures_{V1_standstill_Yp10,T10_aprbs_60}.json`,
key `F2`. It is available to build on; re-checking it is cheap and legitimate, particularly given
the four errors in section 6, but it is not the task.

Two confirmations of the same mechanism from different directions:

- **Zeroing the absorber's initial state removes it.** Settled Y error over a 12 s free run, mean
  of the last 1 s: V1 `+7.84e-04` at K0 against `-1.22e-09` seeded at `t=0` with the analytic IC;
  T10 `+7.79e-04` against `-2.43e-07`. Key `F1`.
- **With the absorber at rest the two models are numerically identical.** V1 Y at `t=0` exact IC:
  baseline minus data `-1.22e-09`, truth minus data `-1.78e-09`. At K0 the same pair is
  `+7.84e-04` and `-3.86e-06`, a 200x separation. So Y's offset is not about the absorber's
  *dynamics* at all, only its initial momentum. Artefact: `msd_offset_x_discrepancy.json`, where
  every sum check `(truth-data) - (truth-baseline) - (baseline-data)` lands at `1e-19`.

### 4.3 Mechanism C is refuted on every channel

Freezing `delta_a` at zero inside `M`, which deletes the bilinear `delta_a*qddot` and `delta_a^2`
terms while leaving A and B intact, retains **100.00 %** of the Y offset and 99.9 to 112 % of X, on
V1, T3, T6 and T10. The reason is size: `delta_a` has an RMS of 22 micrometres against a moment
arm of 0.4 m, so the product never reaches the DC. Key `F6`.

### 4.4 X: mechanism B is the leading candidate but is not established

One draw per arm at seed K0. `L0 = 0` reduces the settled X1 error from `3.91e-06` to `3.55e-09` on
V1 and from `6.38e-04` to `4.86e-05` on T10, while zeroing the absorber IC changes X by 0.1 to
2 %. So the *direction* is clear. What is missing is separation from seed scatter:

| record | `truth - baseline` on X1, 40 seeds in [0.005, 3.0] s |
|-|-|
| V1 | mean `-7.43e-07`, std `1.99e-05`, min `-4.69e-05`, max `+3.46e-05`, **sign changing** |
| T10 | mean `+5.04e-04`, std `2.54e-04`, min `+2.22e-05`, max `+1.02e-03` |

**V1's X has no stable value and is withdrawn.** A standstill record has no sustained X motion, so
mechanism B's contribution, which is the accumulated impulse of `dM_const*qddot`, never builds up;
what remains is oscillatory and its running integral is a random-walk endpoint. T10's X is a real
model difference (at `t=0` exact IC the truth reproduces the data to `+2.46e-07` m while the model
difference is `+7.07e-04` m), but a one-draw 13x reduction against a std of `2.54e-04` on a mean of
`5.04e-04` does not attribute it.

### 4.5 Theta has never been examined

`K(2,2) = kb1 + kb2`, so Theta is sprung and a state error there should decay rather than offset.
Both `dM` entries touch the Theta row, so B and C both act on it. Nothing in this session measured
it. The stage outputs X1 and X2 are `X +/- (Lb/2)*Theta`, so Theta's contribution is mixed into
both of them and is not separable from the current figures.

### 4.6 Supporting facts

- **`ma_frac = 0.10` and `L0 = 0.10` are fitted from the data, not assumed.** RMSE(Y) at 2 s over
  `ma_frac` 0.050 / 0.075 / **0.100** / 0.125 / 0.150 is 2.40e-04 / 1.19e-04 / **2.37e-06** /
  1.24e-04 / 2.45e-04. For `L0`, RMSE(X1) is 2.28e-06 at 0, **2.60e-07** at 0.10, 1.75e-06 at 0.20.
  The mat files store no absorber parameters.
- **D-087 extends to X.** Seeding at sample 0 with the stored `x_logical[0]` rather than the
  analytic IC gives truth-minus-data `+1.72e-05` on X1 against `+2.50e-07`, a 69x penalty from the
  one-sided `gradient()`.
- **`K0 = 17`** is the encoder history requirement, `na = nb = 2*(nx + nx_aug) + 1`, 4.25 ms at
  4 kHz. The pipeline cannot seed earlier, and by then `vdelta_a` is at 2.5e-02 m/s, essentially
  its full RMS. The `t=0` seeding is a diagnostic, not a fix the pipeline can adopt.
- **Observability at current parameters**, superseding `diagnostics/system_dynamics.json` (dated
  2026-06-16, describes a 421.6 Hz absorber): frozen-point PBH over `Y` in [-0.35, 0.35] gives
  worst `sigma_min = 1.12e-05`, so structurally observable; but over the 18-sample encoder window
  `vdelta_a` produces 26,000x less output energy than `Y`, `cond(O) = 4.6e+05`, and least-squares
  estimability of `vdelta_a` against its 2.16e-02 m/s RMS is 0.43 % noiseless, 50x at SNR 60 dB,
  161x at SNR 50 dB, at any window length.
- **The Simulink records mix two plants.** `y` and `x_logical` come from `q_aug` (SID 88, the
  Extended ODE) while `delta_a` and `vdelta_a` come from SID 47, the Simscape Multibody subsystem;
  the ODE's own `delta_a_ode` was computed and discarded. The two agree to 0.50 % at 4 kHz and to
  display precision at 20 kHz **on V1 only**.

## 5. Assumed but not verified

1. **That mechanism B (`L0`) owns the X offset.** Section 4.4. This is the task.
2. **That Theta carries no offset.** Argued from `K(2,2) > 0`, never measured. Section 4.5.
3. **That the latent-eigenvalue result generalises past run 71167.** On
   `gantry_drift_71167_last.pth` the augmented rows' 2x2 self-Jacobian has `|lambda|` mean `6e-05`
   against the `0.987` at 14.2 deg/step needed to hold the 158 Hz mode, so the latents are nonzero
   but memoryless. That run's objective moved only 2 to 4.5 % over 20 epochs (drift-isolation T1),
   so this may describe the run rather than the method. Artefact `msd_offset_latent_eigs.json`,
   grade SINGLE. Settled by pointing `diag_latent_eigs.py` at
   `scripts/gantry/ARTBP/data/72659/ckpt_poly6_seed0_ep8_h1600_b256_last.pt`.
4. **That the 0.50 % Simscape-versus-ODE agreement holds on moving records.** Only V1, which is
   standstill, where the terms the ODE drops are negligible by construction.
5. **The downsampling figures have no artefact.** 20 kHz versus 4 kHz measured at `3.09e-08` m
   pooled against a baseline error of `1.59e-03` m, i.e. 0.0019 %, over 18 records. Printed to
   stdout only, script gone. Re-derive before citing.

## 6. Tried and failed

- **Mechanism C (rectification) as the offset source** -> freezing `delta_a` in `M` retains
  100.00 % of the Y offset -> the term is real but four orders too small, `delta_a` RMS 22
  micrometres against a 0.4 m arm -> key `F6` in both record JSONs.
- **"Rebuild the encoder init from the 8-state map so it can estimate `vdelta_a`"** -> wrong
  target -> the encoder-init baseline already beats true-`x0` seeding 3.5x (2.11e-04 against
  7.33e-04 on V1 Y) and the total headroom to the 8-state oracle is 2.3x, against 127x from one
  epoch of training -> `71167/gantry_results_71167.npz`, keys `baseline_encinit_rms`,
  `baseline_rms`, `rms_oracle`, `loss_val`.
- **"`x_enc_ann` is identically zero, so the latents never learn"** -> the array is exactly 0.0
  over 47983 valid samples, but it is an artefact -> `deepSI/fit_systems/fit_system.py:486`
  restores `_best` at the end of `fit()` and run 71167's best epoch is 0, so the evaluated model
  was the zero-init ANN -> trap already recorded in `docs/status-overview-2026-07-27.md` section 8.
- **"There is no identity term on the latent rows, so the memory is not built in"**, framed as a
  defect -> a learned latent transition is the method -> withdrawn.
- **Treating F1's V1 X1 (`-3.24e-08`) versus F6's (`-3.91e-06`) as a discrepancy** -> not one ->
  the quantity has no stable value on a standstill record, and those are two draws from zero-mean
  scatter -> `msd_offset_x_discrepancy.json`, `x_seed_sweep`.
- **The "157.89 versus 164.55 Hz absorber anomaly"** -> not an anomaly -> the eigenvalue was
  computed at `Y = 0.3` and the spectrum measured at `Y = 0.10`, and drift-isolation T4's dataset
  verification records peaks of 161.74 / 160.22 / 155.94 Hz across records -> **strike from
  `ISSUE.md`**.

## 7. Achieved

**Implemented and validated.** `scripts/gantry/msd-offset/`: `plant.py` (both models, RK4, record
loading), `make_figures.py` (F1 to F6, 10 PNGs in `figures/`), `diag_x_discrepancy.py` (three-way
decomposition, sum checks at `1e-19`), `diag_latent_eigs.py` (checkpoint linearisation). Numbers
land in `simulations/gantry_subnet/diagnostics/msd_offset_*.json` with units and horizon.

**Implemented, not validated.** `gtd_run_simulation.m` now reads `delta_a_ode`. Needs regeneration.

**Written, needs the section 6 corrections.** `ISSUE.md`, `README.md`,
`docs/msd-offset-mechanism-2026-07-29.md`.

**Known figure defects, recorded in `README.md`.** F1 draws truth first so it is hidden where the
others overlay it; F1's `t=0` error renders as a flat line on an axis set by the K0 failure; V1's
X1 and X2 panels show one draw of a zero-mean quantity; F6's log bar chart has meaningless bar
lengths.

## 8. The open question

**Which term of `dM` produces the X offset, and does Theta carry one at all?**

Candidates and what chooses between them:

- **B (`L0`)**: on T10 the `L0 = 0` arm's seed-mean must separate from FULL's by more than the seed
  scatter permits. Section 10.
- **A (the absorber column, via the `-ma*d` coupling into Theta and thence into X1 and X2)**: the
  absorber-IC arm changes X by only 0.1 to 2 % at one seed, so this is unlikely, but it has not
  been tested across seeds either.
- **Neither, i.e. the X offset is not attributable at this magnitude**: then F6's X bars are
  withdrawn on both records and X joins Theta as unexamined.

Theta needs a separate look because both `dM` entries touch its row and the stage outputs mix it
into X1 and X2 as `X +/- (Lb/2)*Theta`. Reporting the logical channels alongside the stage ones
would separate it at no extra simulation cost.

**A larger question exists and the decision is the user's.** If the latent-eigenvalue result
(section 5 item 3) holds on other checkpoints, the augmentation is a static residual force acting
on marginally stable axes, which would bear on whether drift-isolation T2's ladder is climbing
toward the right thing. That is a bigger question than this task and it belongs to that thread.

## 9. Next action

Extend `fig6_attribution` in `scripts/gantry/msd-offset/make_figures.py` to evaluate all four arms
(FULL, `delta_a` frozen in `M`, `L0 = 0`, absorber IC `= 0`) over the 40 seed instants
`diag_x_discrepancy.py` already uses (`np.linspace(20, int(3.0/ts), 40)`, 8 s horizon each),
reporting mean and standard deviation per arm per channel, and to report the **logical** rows
`[X, Theta, Y]` alongside the stage rows `[X1, X2, Y]` so Theta is separable. Replace the log bar
chart with a dot plot carrying standard-error bars.

Rationale: one change answers both halves of section 8. The seed sweep settles B versus A on X, and
adding the logical channels exposes Theta for the first time. Both halves of the machinery already
exist in the folder.

## 10. Acceptance criterion

**X, on T10, channels X1 and X2**: the `L0 = 0` arm's seed-mean must sit below FULL's seed-mean by
more than **three standard errors of the difference**, with `N = 40` per arm and the standard error
computed from the measured per-arm scatter. FULL's scatter is already known (mean `+5.04e-04`, std
`2.54e-04`), so the threshold comes from the data, not from the model. Pass attributes X to
mechanism B; fail withdraws the X bars on both records.

**Theta**: report the seed-mean and std of the settled logical-Theta error for all four arms. The
criterion is whether the mean exceeds two standard errors. Expect it not to, since `K(2,2) > 0`;
if it does, that is a new finding and the argument in section 4.5 is wrong.

**V1**: no criterion, that channel is withdrawn.

Report every number either way. A fail on X is a usable result, not a problem. Grade SINGLE unless
a third record is added.

## 11. Read these first

1. `Matlab-scripts/Augmentation/gantrySystemExtended.m` and
   `kamtin-fp-model/03 Simulink gantry/functions/gantrySystem.m` -- the two equation sets whose
   difference is the subject. Section 4.1 is the difference already worked out; read the sources to
   confirm it rather than trusting the transcription.
2. `scripts/gantry/msd-offset/ISSUE.md` -- issue statement and confirmed/open split. Still carries
   the four errors in section 6.
3. `scripts/gantry/msd-offset/make_figures.py`, `fig6_attribution` -- the function to change.
4. `scripts/gantry/msd-offset/diag_x_discrepancy.py` -- the seed sweep to reuse, instants and
   horizon.
5. `scripts/gantry/drift-isolation/CONCLUSIONS.md`, T5 and cross-cutting finding 3 -- establishes
   independently that the baseline's offset is bounded and that the true-`x0` baseline is not an
   upper bound. This session re-derived both, so they carry two-source agreement.

## 12. Do not

- Retry mechanism C, the encoder-init rebuild, or any argument resting on `x_enc_ann` being zero.
  Section 6.
- Quote V1's X1 or X2 offset numbers, or the "157.89 versus 164.55 Hz" gap.
- Load `gantry_ckpt_71167.pt` expecting a trained model. It is the epoch-0 initialisation; use
  `gantry_drift_71167_last.pth`.
- Touch `scripts/gantry/drift-isolation/`.

## 13. Operational

```
cd "scripts/gantry/msd-offset"
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
    -n GraduationProject python -u make_figures.py
```

Launch in the background per the live-output rule and read the streamed `.output` file. The current
script takes about 14 minutes, dominated by F2's 60-seed scatter; four arms over 40 seeds at an 8 s
horizon adds 10 to 15 minutes, so budget 30. Consumes
`data/gantry/matlab/trajectory/augmentation/{V1_standstill_Yp10,T10_aprbs_60}.mat` at 20 kHz.
Figures to `figures/`, numbers to
`simulations/gantry_subnet/diagnostics/msd_offset_figures_<record>.json`.

`conda run python -c` cannot take a multi-line argument; write snippets to a scratchpad file. This
session lost a step to it.

## 14. Delegation

None. Two files in one directory, both already summarised here.
