# Handoff: derive and numerically test the condition under which OBC recovers the true parameters
**From**: session of 2026-09-18 | **Branch**: Augmentation | **Effort suggested**: xhigh

## 1. Task

Supervisor Maarten Schoukens raised a claim about the orthogonal-by-construction (OBC) method:
projection makes the baseline/augmentation decomposition **unique**, but it recovers the **true**
physical parameters only if the true missing dynamics are themselves orthogonal to the baseline's
parameter sensitivities. Work this out properly in
`scripts/gantry/orthogonal-by-construction/baseline-and-added-dynamics/` (currently empty, created
by the user for this task). Produce three things: a symbolic derivation in Python using sympy that
establishes the condition and the closed-form parameter bias; a written statement of it that
connects to the existing OBC documentation; and a numerical evaluation on this project's actual
gantry augmentation data that reports how non-orthogonal the true added dynamics (the hidden MSD
absorber) actually are, and what parameter error that predicts. Then compare the predicted bias
against the parameter error the completed OBC training run really reached. Work autonomously to
completion; the user is away and will read the result cold.

## 2. Out of scope

- **Do not launch, cancel or resubmit cluster jobs.** Everything here is offline and local. No
  training run is needed or wanted.
- **Do not modify** `model_augmentation/fit_systems/obc.py`, `obc_projection.py`, or
  `scripts/gantry/gantry_dynamic/obc_gantry.py`. Import and call them. `obc_projection.py` is a
  superseded eleven-column prototype ruled out by D-192 and is not the affine arm.
- **Do not change any training config, protected space, reference set or basis-refresh policy.**
- **Do not re-analyse the three-arm comparison.** It is finished and written up; its numbers are in
  section 4 here and its figures in `scripts/gantry/meeting/meeting-18-09-2026/figures/OBC/`. Use
  them as the comparison target, do not redo them.
- **Do not start the conditional follow-ups** in the OBC implementation plan (fixed initial basis,
  oracle parameter point, alternative velocity reference).
- **Do not implement a separate parameter learning rate (`lr_theta`).** That is a live and agreed
  next experiment but it belongs to a different session; see section 8.

## 3. Where things stand

Branch `Augmentation`, last commit `c315e2f`, tree dirty across `model_augmentation/fit_systems/`,
`scripts/gantry/gantry_dynamic/`, `scripts/gantry/orthogonal-by-construction/`,
`scripts/gantry/meeting/meeting-18-09-2026/` and `gantry_interconnect_dynamic.py`. Nothing from the
OBC work is committed.

No jobs in flight. The three-arm comparison (`noproj` 84032, `obc` tangent 84033, `obc_affine`
84034/84037) has finished for the first two arms; the affine arm's logs and checkpoints are a
snapshot taken mid-run at 93 percent. All logs, per-run output folders and `_best.pth` checkpoints
are local at `scripts/gantry/meeting/meeting-18-09-2026/server/useable/`.

The target folder `scripts/gantry/orthogonal-by-construction/baseline-and-added-dynamics/` exists
and is empty.

## 4. Established and verified

**The mathematical claim, as this session reconstructed it.** Write the truth as
`f_truth = f_base(theta*) + Delta*`, let `J = d f_base / d theta` at `theta*` over the reference
set, and let `P = J (J^T J)^-1 J^T`. Under the OBC constraint `f_ann ⟂ range(J)`, matching
`J dtheta + f_ann = Delta*` forces `J dtheta = P Delta*` and `f_ann = (I - P) Delta*`, so with `J`
of full column rank

```
theta_hat - theta* = J^+ Delta*
```

and `theta_hat = theta*` **iff** `J^T Delta* = 0`, i.e. iff the true added dynamics are orthogonal
to the sensitivity span. This is first order in `theta` and exact only if `f_base` is affine in
`theta`. The successor should re-derive this symbolically rather than trusting the paragraph.

**The reference set already carries everything needed to build `Delta*` from data.**
`GantryReferenceSet` in `scripts/gantry/gantry_dynamic/obc_gantry.py:49-61` has fields `Z_step`,
`Z_field`, `x_phys (N,6)` physical `[q; qdot]`, `u_stage (N,3)` physical stage forces,
`x_phys_next (N,6)` "the RECORDED next logical state", and `x_abs (N,2)` recorded
`[delta_a, vdelta_a]`. So the truth's one-step map is **recorded**, not simulated: `Delta*` can be
formed as `x_phys_next` minus the baseline one-step prediction, with no oracle model in the loop.
This is a data-derived quantity, which the Control Engineering Stance prefers over an
oracle/model-based one.

**The primitives exist and must be reused, not reimplemented.** In
`model_augmentation/fit_systems/obc.py`: `build_stacked_sensitivity(step, vbar, Z, row_ix, chunk,
device)` at line 341 returns `J (N * n_rows_sel, p)` in float64 via `jacfwd` over the batched step;
`evaluate_step_rows(step, vbar, Z, row_ix, ...)` at line 367 returns `step(vbar, z)` restricted to
the protected rows, stacked sample-major, which is exactly the second term of `Delta*`. On
`OBCBasis`: `coefficient`, `project_onto`, `project_out`, `r_overlap`, `r_perp`. Reusing these
guarantees the same inner product as training, which is the whole point.

**The basis the runs used.** `scripts/gantry/gantry_dynamic/obc_gantry.py:326` builds it as
`build_stacked_sensitivity(lambda v, z: blk64.transition_from_free(v, z), ...)`. Protected rows are
`(0,1,2,3,4,5)` (ANN cols `(0..5)`), reference set 671944 tuples from 14 records at stride 1,
`J` is `4031664 x 10`, rank 10 of 10, condition 8.9250e+02, largest column correlation
`cg1~cg2 = -0.9817` (measured at epoch 0 of the tangent arm; recorded in the prior handoff
`tasks/handoffs/2026-09-18-obc-three-arm-results.md` section 4).

**The truth system is fully pinned.** `ma_frac = 0.10`, so `ma = 1.01 kg`, `mh_rigid = 9.09 kg`,
`fa = 150 Hz`, `ka ≈ 8.97e5 N/m`, `zeta_a = 0.05`, `ca ≈ 95.3 Ns/m`, `L0 = 0.10 m`. Exact 8-state
EOM at `Matlab-scripts/Augmentation/gantrySystemExtended.m`, state
`[X, Theta, Y, delta_a, dX, dTheta, dY, vdelta_a]`, `mh` argument is `mh_rigid`. Stage vs logical:
`u_logical = P @ u_stage`, `q_stage = P^T @ q_logical`, `P = gantry_ss.P`.

**The three-arm result this must be compared against** (all measured this session from the logs and
`_best.pth` checkpoints). `combo-err` is the RMS over the ten relative parameter errors in percent;
every arm starts at 9.49 percent from the same detuned initialisation.

| arm | final combo-err | minimum | val sim-RMS |
|-|-|-|-|
| noproj (84032) | 7.20 % @ 19500 | 7.14 @ 15600 | 5.8609e-06 |
| obc tangent (84033) | 7.69 % @ 19500 | 6.98 @ 9100 | 5.8905e-06 |
| obc affine (84037) | 8.47 % @ 18200, unfinished | 6.66 @ 10400 | 5.9842e-06 |

Per-parameter percentage errors at matched budget (epoch 100, it 13000), and the change in absolute
error each projected arm produced relative to `noproj`:

| param | noproj | obc | affine | d\|err\| obc | d\|err\| affine |
|-|-|-|-|-|-|
| cg1 | +15.36 | +0.98 | -0.28 | **-14.38** | **-15.08** |
| mh | -6.69 | -4.10 | -5.17 | -2.59 | -1.52 |
| J_eff | -6.45 | -5.77 | -5.13 | -0.68 | -1.32 |
| cy | +4.02 | +4.00 | +0.96 | -0.02 | -3.06 |
| m_total | +2.28 | +2.31 | +2.08 | +0.03 | -0.20 |
| cg2 | -5.77 | -5.71 | -7.27 | -0.06 | +1.50 |
| cb_sum | -9.18 | -10.24 | -10.03 | +1.06 | +0.85 |
| m_diff | -0.12 | -1.13 | -1.13 | +1.01 | +1.01 |
| kb_sum | +4.21 | +11.88 | +3.22 | **+7.67** | -0.99 |
| d | +6.60 | +13.41 | +15.89 | **+6.81** | **+9.29** |

The two OBC arms agree on the sign for 9 of 10 parameters despite different protected spaces, so
this redistribution is a reproducible effect of projecting, not noise. Validation fit differs
between arms by only 0.5 percent while individual parameters differ by up to 15 percentage points,
which says the arms differ largely along near-flat directions of the objective.

**Parameter order is not the same everywhere.** `params_init` / `params_learned` /
`param_names` in `gantry_results_*.npz` are ordered
`[kb_sum, cg1, cg2, cy, cb_sum, mh, m_total, m_diff, J_eff, d]`. The log's two `[combos]` lines
print `[mh, m_total, m_diff, J_eff, d]` then `[kb_sum, cg1, cg2, cy, cb_sum]`. Align explicitly by
name, never by position.

**Physical parameters are stored in log space.** `physical = params_init * exp(free_params)`,
verified to rtol 1e-4 against `params_learned` on run 84032. `free_params` is
`hfn.connected_blocks[0].free_params`, a 10-vector. Consequence for this task: a bias `dtheta` in
the free parameterisation **is** the predicted relative error, directly comparable to the
percentage columns above, with no Jacobian of the parameterisation needed.

**Loading a checkpoint requires two path entries.** `SSE_Interconnect_Composed_<id>_best.pth` are
pickles of the fit-system `__dict__`. `torch.load(..., weights_only=False)` fails with
`ModuleNotFoundError: gantry_dynamic` unless both the repo root **and** `scripts/gantry` are on
`sys.path`.

**`Probe_V_orth` and `Probe_orth_frac` are NaN at every entry in all four checkpoints**, noproj
included. There is no recorded in-training orthogonality measure to compare against.

**Post-training gradient norms** (`gantry_grad_norms_*.npz`, single batch): `hfn.0` is
`free_params` at 2.020e-09 for run 84032, against `hfn.7` (an ANN weight) at 6.999e-06 and the whole
`hfn` group at 7.070e-06.

**Timescale separation**, `noproj`, normalised progress: at epoch 10 the network has completed 77.4
percent of its total fit improvement while the parameters have removed 1.1 percent of their initial
error; at epoch 30, 92.4 against 2.9 percent; at epoch 150, 100 against 24.1 percent.

**Figures and their data cache already exist** at
`scripts/gantry/meeting/meeting-18-09-2026/figures/OBC/`: `extract_obc_data.py` writes
`obc_data.json` (all four arms, per-validation `combo_err`, `val`, `train`, `rho`, rank, and the ten
per-parameter percentage errors), and `plot_obc.py` draws from that cache. **Read
`obc_data.json` rather than re-parsing any log.** House figure style is
`scripts/gantry/drift-isolation/figures/figstyle.py`, imported not copied.

## 5. Assumed but not verified

- **That `x_phys_next` is the recorded truth next state at the sample aligned with `Z_step`, with no
  off-by-one.** The field comment says "the RECORDED next logical state" and marks it
  "diagnostics only". Settled by a null test: with the baseline evaluated at the **true** parameters
  and the absorber absent, `x_phys_next - step_base` on a record generated without the absorber
  should be at the derivative-stencil floor and not at the signal level. If no absorber-free record
  exists, settle it instead by checking that the position rows of `Delta*` are orders of magnitude
  below the velocity rows, since a one-sample misalignment would corrupt the position rows first.
- **That the units of `J`'s rows and of `Delta*` match.** `J` is built on the normalised block input
  through `blk64.transition_from_free`, so its rows are in whatever units that step emits, while
  `x_phys_next` is physical. The normalisation (`norm.x_mean`, `norm.std_x`) must be applied to one
  side. **Getting this wrong silently changes the inner product and makes `rho*` meaningless**, so
  it is the single most important thing to pin down before any number is reported. Settled by
  checking that `evaluate_step_rows(step_base, theta*, Z_step, rows)` reproduces the normalised
  `x_phys_next` to stencil accuracy on a record where the absorber contribution is small.
- **That the epoch-0 `vbar` convention carries over.** The runs' first refresh prints
  `vbar = 0` (ten zeros) and later epochs print small nonzero values, so `vbar` is an offset in the
  free parameterisation relative to the current parameters, not an absolute parameter vector.
  Confirm against `GantryOBCCorrection.expansion_point` (`obc_gantry.py:221`) before using it.
- **That `theta*` (the true parameter vector) is recoverable exactly.** This session derived it as
  `params_learned / (1 + pct/100)` from run 84032 and got
  `[kb_sum 3975.08, cg1 14.50, cg2 20.30, cy 10.00, cb_sum 18.00, mh 10.10, m_total 43.70,
  m_diff -0.5067, J_eff 3.7962, d 0.1000]`. Nine of ten are clean round numbers and are almost
  certainly right. **`m_diff` is not**: reconstructing it this way disagrees with the log's own
  `[combos]` percentage, because `m_diff` crosses zero and the log normalises it differently. Find
  the authoritative truth vector in `kamtin-fp-model/` or `Matlab-scripts/Augmentation/gtd_config.m`
  rather than trusting this reconstruction, and treat `m_diff` as suspect until you do.
- **That the first-order statement survives the hidden absorber state.** `Delta*` depends on
  `delta_a`, which is not part of the reference tuple `z_k`. So `Delta*` is a trajectory-realised
  signal, not a function of `(x, u)`, and "orthogonal" is therefore a statement about the realised
  signal on this reference set, not about an abstract function. This is weaker than the clean iff
  and must be stated as such. `documentation/obc-io-additional-state-chapter.tex` is the existing
  treatment of why the additional state is needed at all; the derivation must agree with it.
- **That finite ANN capacity does not dominate.** The identity assumes `f_ann` can represent
  `(I - P) Delta*` exactly. It cannot (3 hidden layers, 24 nodes). So `J^+ Delta*` is a **lower
  bound** on the achievable parameter error, not a prediction of equality. Say so when reporting.

## 6. Tried and failed

- Reconstructing per-parameter errors for the affine arm from checkpoint `free_params` via
  `physical = init * exp(free)` -> 9 of 10 parameters matched the log to 0.01 percentage points,
  `m_diff` came out `+30.87` against the log's `-1.46` -> `m_diff` crosses zero, so a relative error
  is normalised differently there -> use the log's `[combos]` values, or `obc_data.json` which
  carries them.
- Arguing from Adam's step budget that the parameters are not learning-rate limited (19500 updates
  at `lr=1e-5` gives 0.195 of drift in log space, 0.095 needed, 0.087 observed) -> **the argument is
  wrong and was withdrawn** -> it assumes `m_hat / sqrt(v_hat) ~ 1`, which fails when the gradient
  is small relative to minibatch noise, and net displacement is not path length when the parameters
  demonstrably reverse direction (`cg1`, `d`) -> do not revive it.
- Expecting `Probe_V_orth` / `Probe_orth_frac` to provide an in-training orthogonality measure ->
  NaN at all 16 entries in all four checkpoints -> never populated -> there is no such series to
  plot.
- Expecting the stored time-domain arrays in `gantry_results_*.npz` to give clean signal plots ->
  `y_hat_enc` is the **open-loop free run** and is diverged (NRMS 94 / 98 / 148 for noproj,
  43 / 69 / 1607 for obc, against a baseline of 0.032 / 0.032 / 0.017) -> the runs are selected on a
  closed-loop criterion and no closed-loop time series is stored -> do not build time-domain figures
  from these files.
- `conda run python -c` with a multi-line snippet -> `AssertionError: Support for scripts where
  arguments contain newlines not implemented` -> write the snippet to the scratchpad and run the
  file, per the Python-environment rule.

## 7. Achieved

Nothing in the target folder; it is empty. What exists and is usable as input:

- The three-arm comparison is complete and written up, with figures at
  `scripts/gantry/meeting/meeting-18-09-2026/figures/OBC/` (`obc_fig1_fit_vs_parameters.pdf`,
  `obc_fig2_per_parameter.pdf`) and the extraction cache `obc_data.json`.
- The OBC one-step path is implemented and gated, with 14 behavioural gates at
  `scripts/gantry/orthogonal-by-construction/implementation/08-one-step/test_behavior.py` and
  results at `implementation/results/2026-09-17/test_behavior.log`.
- Existing derivations to build on, not duplicate, in
  `scripts/gantry/orthogonal-by-construction/documentation/`: `state-level-obc-derivation.tex`,
  `obc-io-additional-state-chapter.tex`, `gyorok-to-gantry-state-space-obc.tex`,
  `obc-gantry-one-step-implementation-plan.tex`.

## 8. The open question

**Is the hidden MSD absorber orthogonal to the gantry baseline's parameter sensitivities on this
data, and does the resulting predicted bias `J^+ Delta*` account for the parameter error the OBC run
actually reached?**

Candidate answers and what chooses between them, all from `rho* = ||P Delta*|| / ||Delta*||` and
`J^+ Delta*`:

- *The absorber is nearly orthogonal* (`rho*` small, predicted `combo-err` well below 7 percent).
  Then OBC should have recovered the parameters and did not, so the limit is elsewhere: the
  timescale separation in section 4, or excitation. The interpretability claim survives and the
  experiment was run in the wrong regime.
- *The absorber is strongly non-orthogonal* (`rho*` large, predicted `combo-err` near 7 percent).
  Then the OBC arm converged approximately where theory says it must, the residual error is the
  predicted bias rather than a failure, and the thesis gains a testable a-priori criterion for when
  an augmentation is interpretable at all. **This is the outcome that would most change the story**,
  so be especially careful about the units question in section 5 before claiming it.
- *Intermediate.* Report the per-parameter predicted bias against the per-parameter observed error
  and say which parameters the condition explains and which it does not. `cg1` is the interesting
  one: the two OBC arms move it by 14 to 15 percentage points relative to `noproj`, and its
  sensitivity column is 98 percent correlated with `cg2`.

One thing to raise with the user in a sentence and then leave alone: a separate learning rate for
the ten physical parameters (`lr_theta`) is the agreed next experiment for the timescale problem.
`lr_theta` does not exist in the live `gantry_dynamic` pipeline (only in
`scripts/gantry/orth-projection/improvement-trials/` and in the August snapshots under
`tasks/snapshots/`), so it needs a `param_groups` split. D-095 records that the lr knob was once
silently disconnected and every run trained at 1e-3 instead of its stated rate, so any new lr
plumbing needs a connectivity check: two different lrs must produce different first-iteration
losses. Also `CL_ADAM_EPS` sets `eps` on all param groups after `build_model` and would silently
override a per-group `eps_theta`. That work belongs to a different session.

## 9. Next action

**Write and run
`scripts/gantry/orthogonal-by-construction/baseline-and-added-dynamics/derive_condition.py`, a
sympy derivation that establishes the bias identity symbolically before any numerics are attempted.**
Set up a baseline affine in its parameters, `f_base(theta) = f0 + J theta`, an added component
`f_ann` constrained to `J^T f_ann = 0`, solve `J dtheta + f_ann = Delta` for `(dtheta, f_ann)`, and
show symbolically that `dtheta = J^+ Delta` and that it vanishes iff `J^T Delta = 0`. Then repeat
with `f_base` nonlinear in `theta` to show the statement holds to first order and to obtain the
second-order remainder term, which is what quantifies how far the 10 percent initial detuning
invalidates it. Print the results; the printed derivation is the deliverable, alongside a short
`.tex` or `.md` statement of it in the same folder.

Do this first because it is cheap, it has no dependency on the data pipeline, and it fixes the
notation and the sign conventions that the numerical step must then match. Then build `Delta*` and
report `rho*` and `J^+ Delta*` as described in section 4, settling the units question in section 5
before reporting any number.

## 10. Acceptance criterion

Two numbers, both data-derived:

1. **`rho* = ||P Delta*|| / ||Delta*||`** on the 671944-tuple reference set with `J` built by the
   project's own `build_stacked_sensitivity`, evaluated at the true parameter vector. Report it with
   the null-test residual from section 5 alongside it, because `rho*` is only meaningful if the
   baseline-at-truth residual on an absorber-free signal sits at the stencil floor.
2. **Predicted per-parameter bias `J^+ Delta*` in percent**, and its RMS as a predicted `combo-err`,
   against the observed `obc` arm value of **7.69 percent final / 6.98 percent minimum** and the
   shared starting value of **9.49 percent**.

The comparison counts as explaining the observed floor if the predicted RMS lands between the
arm's minimum and its final value, 6.98 to 7.69 percent. It counts as ruling the explanation out if
the predicted RMS is below 2 percent, which is roughly a quarter of the distance travelled and well
outside anything the second-order remainder could account for. Anything between is inconclusive and
must be reported as such rather than rounded toward either story. Report the predicted bias
per parameter as well as in aggregate, since section 4 shows the aggregate hides a large and
reproducible redistribution.

Also run the same computation with `J` built at the **detuned initial** parameters as well as at the
true ones, and report both. The gap between them is the sensitivity of the whole conclusion to the
expansion point, and with a 10 percent initial detuning it is not obviously small.

## 11. Read these first

1. `scripts/gantry/gantry_dynamic/obc_gantry.py`, lines 49 to 170 (the reference set and its
   construction) and 308 to 400 (the basis builder, reference field and `attach_obc`). This is where
   `Delta*` gets built and where the units question is decided.
2. `model_augmentation/fit_systems/obc.py`, lines 160 to 340 (`OBCBasis`, including `coefficient`,
   `project_onto`, `r_overlap`) and 341 to 390 (`build_stacked_sensitivity`, `evaluate_step_rows`).
   Reuse these; do not write a second projector.
3. `scripts/gantry/orthogonal-by-construction/documentation/obc-io-additional-state-chapter.tex`.
   The hidden-state subtlety in section 5 is its subject; the new derivation must agree with it.
4. `scripts/gantry/meeting/meeting-18-09-2026/figures/OBC/obc_data.json`, the cached three-arm
   result. Read this instead of any `.out` log.
5. `tasks/handoffs/2026-09-18-obc-three-arm-results.md`, sections 4 and 5, for the basis health
   numbers and the provenance of the run configuration.

## 12. Do not

- Do not reimplement the projector, the basis builder or the reference set. Import them.
- Do not use `gantry_results_*.npz` time series for anything; they are the diverged open-loop run.
- Do not quote `combo_err` from the end-of-run npz; it is not in there. It is in the log, in
  `Probe_combo_err` inside `_best.pth`, and in `obc_data.json`.
- Do not trust the reconstructed `m_diff` truth value from section 5 without confirming it.
- Do not revive the Adam step-budget argument from section 6.
- Do not treat the `rho ~ 0.93` logged during training as `rho*`. The logged `rho` is the overlap
  between the **network's update direction** and `range(J)`; `rho*` is the overlap between the
  **truth's missing dynamics** and `range(J)`. Different objects, and conflating them would invert
  the conclusion.
- Do not touch `kamtin-fp-model/` or read anything under `kamtin-data/Data Telica/`.

## 13. Operational

Env `GraduationProject`. Everything is local and offline; no cluster, no GPU needed, though the
Jacobian build over 671944 tuples benefits from one if available (`build_stacked_sensitivity` takes
a `device` argument and chunks at 16384).

Per the live-output rule, anything expected to run more than a few seconds goes `run_in_background:
true` with
`PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output -n GraduationProject python -u <script>`,
and tell the user the `.output` path. `conda run python -c` cannot take a multi-line argument; write
the snippet to the scratchpad and run the file.

Artefacts this task consumes: the training `.mat` records under `data/gantry/matlab/trajectory/
augmentation/` (via `build_reference_set`), `obc_data.json` for the comparison target, and
optionally `SSE_Interconnect_Composed_84033_best.pth` under
`scripts/gantry/meeting/meeting-18-09-2026/server/useable/` if the learned parameter vector is
wanted directly (remember the two `sys.path` entries).

Artefacts this task produces, all in
`scripts/gantry/orthogonal-by-construction/baseline-and-added-dynamics/`: the sympy derivation
script and its printed output, a written statement of the condition, the numerical evaluation
script, and its results including the per-parameter predicted-versus-observed comparison. A figure
is welcome but optional; if you make one, import `figstyle.py` as
`scripts/gantry/meeting/meeting-18-09-2026/figures/OBC/plot_obc.py` does rather than restyling.

Per the run-discipline rule this is analysis rather than training, so no run-table row is required,
but a `D-` entry in `docs/decisions.md` is warranted before implementing the units convention chosen
for `Delta*`, since that is a non-trivial choice the rest of the result depends on.

## 14. Delegation

None. Every file this task needs is named in sections 4 and 11, so there is no search to fan out.
The project default is one Explore subagent for a genuinely wide search, and this is not one.
