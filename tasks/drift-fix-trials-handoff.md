# HANDOFF: drift-fix trial campaign (overnight, autonomous, kill-tolerant)

**Created** 2026-07-24. **Runs in**: `scripts/gantry/drift-fix-trials/`.
**Reads first**: `tasks/lessons.md` (Step 0), then that folder's `README.md` and `CHECKLIST.md`.
**Evidence base**: `docs/drift-critical-analysis.md` (what is established, over-claimed, contradictory) and
`docs/drift-research-report.md` (method candidates, cited, marked against the five requirements).

---

## 0. Launch, and the rules that keep it running

**Launch command (user runs this, then leaves):**
```
/loop work tasks/drift-fix-trials-handoff.md autonomously
```
No interval, so the loop self-paces. Task completions already re-invoke the session automatically; `/loop`
is the safety net for the case where a run is killed and nothing wakes the session. When the queue is
genuinely finished, end the loop deliberately rather than letting it idle.

**Wakeup pacing**: when self-scheduling, use a LONG fallback (1200 s or more). Do not poll every minute for
work that will notify you anyway.

### THE ONLY THINGS THAT MAY STOP THIS SESSION
The four ASK gates in `CHECKLIST.md`. Nothing else. In particular, these are NOT stops, and treating any of
them as one has ended previous autonomous sessions early:

1. **Finishing a task.** Roll straight into the next one. Never write "task N done, shall I proceed".
2. **Wanting to show a script skeleton.** Show it in the transcript and CONTINUE in the same turn.
3. **An inconclusive or failed result.** That is data. Log it, branch per §5, continue.
4. **An error or a crash.** Diagnose from the traceback plus one or two targeted file reads, fix, re-run.
   If it cannot be fixed in two attempts, log the blocker, mark the unit FAILED in the manifest, and move to
   the next task. Do not stop the queue for one broken unit.
5. **A killed run.** Expected. Units are resumable; relaunch and skip completed units.
6. **An empty queue.** See §5.4: spend remaining time widening the most informative result, not idling.

**Do not batch a progress report per step.** Work continuously and surface a consolidated report when a task
completes or a branch changes the plan.

---

## 1. Mission and definition of done

Attack the six diagnosed issues with the methods below, on ONE frozen rig, running unattended overnight,
accumulating a picture across multiple seeds and settings rather than declaring a verdict off single points.

**Done** = §0 of `PROGRESS.md` is filled with measured reference numbers; every task in the queue has either
a filled R1-R5 row or a written reason it was pruned; at least the T1 question (can a penalty bite at all
under Adam) is answered decisively; and a single named recommendation exists for the supervisor gate.

**Not done** = a pile of logs. Every cycle writes its scorecard row, its dated finding, and its §12 outcome.

## 2. The overriding constraint

**R2 is the most important requirement: the fix must not make any dynamics unlearnable, because the
deliverable is real nonlinear data with an unknown residual.** Any candidate that lowers the windowed
%learned on the Coulomb rig is PRUNED regardless of how well it stops drift. A no-drift result with no R2
number attached is not a result.

## 3. The six issues (full evidence in the folder README §1)

I1 truncation-bias DC drive. I2 optimizer step geometry (parks a stiff direction ~lr from `b*`, and caps any
soft penalty at that scale). I3 anti-damping Y feedback. I4 drift and friction are not separable by
direction or by information. I5 soft penalties saturate in beta under Adam. I6 encoder-init floor (a floor,
not a drift; it is the reference, not a target).

## 4. Task queue

Work top down. Each task states its hypothesis, the knob check that must pass first, its arms, its metric,
its size, and its decision rule. Sizing rule: **the minimum that answers the question**; the DC forms by
~step 13, so one epoch is enough for every DC-mechanism probe.

### 4.0 THE MINIMUM VIABLE NIGHT (read this first; protect these five in order)
The queue below has grown to roughly thirteen arms. If time, kills or failures eat the night, these five
still produce a campaign that CONCLUDES something. Everything else is extension.

1. **T0.1** rig frozen, reference numbers and per-unit cost measured. Blocking: without it nothing else
   is interpretable.
2. **T0.5** velocity-reversal audit. Nearly free, and it decides whether T3a is even possible on this data.
3. **T1 arms (a) + (b2)** only: in-loss control versus implicit prox. This is the family-gating question in
   its smallest honest form.
4. **T3a on the null, then its T5 R2 gate on the Coulomb rig.** The strongest candidate, with the R2 number
   that makes it a fix rather than a drift score.
5. **T2 arm (b0) or (b1)**, whichever is cheaper to stand up. The cheapest attack on the dominant DC carrier.

Rank order for everything beyond these five: T1 (b3) and (c), T3b, T4, T3c, T2 (b2), T3d, T6.
If an item in the five fails or is ruled impossible (for example T0.5 says there are no reversals), record
that and move to the next item rather than substituting an unranked one.

### T0 — Freeze the rig and reconcile (blocking, cheap)
- **T0.1 (BLOCKING, do this first).** Write `rig.py` with the frozen constants (records, MAXLEN, STRIDE, nf,
  eval-window set, seed list, Coulomb constants). Measure and record in `PROGRESS.md` §0: ANN-off floor per
  axis, perfect-match true-x0 floor, unconstrained-Adam control (per-axis drift as a multiple of floor,
  per-row DC, Jacobian self-feedback entries), and on the Coulomb rig the ANN-off windowed residual and the
  unconstrained-Adam %learned. **Nothing later is interpretable until this exists.**
  Also MEASURE AND RECORD the cost of one unit (one arm, one setting, one seed, one epoch) on the null rig
  and on the Coulomb rig, in seconds. Every later sizing decision comes from those two numbers, per §4.1.
- **T0.2** Settle `tau_X` / `tau_Y` (1.55/1.01 s versus 21.05/0.049 s) closed-form from the model matrices.
  Seconds. The bounded-impulse proof depends on it.
- **T0.3** Settle Problem 2, exponential versus marginal, on the second checkpoint
  (`ARTBP/data/72659/ckpt_poly6_seed0_ep8_h1600_b256_best.pt`) using `test_efolding`. Free-run only, no
  training. It decides whether I3 needs a structural fix or conditioning.
- **T0.4** Re-run and log `dC` (train with the telescoping parametrization) on the frozen rig, to give the
  reference arm a real, provenance-clean number. Log the §12 row.
- **T0.5 (PREREQUISITE for T3, nearly free, no training).** Velocity-reversal / parity-observability audit.
  Under UNIDIRECTIONAL motion `sign(v)` is constant, so a Coulomb force `Fc*sign(v)` is EXACTLY collinear
  with a constant, and no method can separate them on that data. At nf=400 (0.1 s) a window may contain zero
  reversals, in which case the confounding is exact per window, not approximate. Measure and record, per
  axis and per record: the number of velocity sign changes per window, and `mean(sign(v))` per window
  (magnitude near 1 = unidirectional = unseparable; near 0 = well balanced). **If most windows carry no
  reversal, T3's parity arm cannot work at nf=400 and must move to longer windows or to
  reversal-containing records; say so rather than running it blind.** The collinearity argument is
  elementary algebra, not a citation; bidirectional-excitation practice in friction identification is
  SEARCH-LEVEL (Sensors 26(1):78). See `research/thread-CD-bias-through-integrator.md`.

### 4.1 BUDGET RULE: size the campaign from the MEASURED cost, never from a guess
The seed counts written in the tasks below (3 per arm) are a placeholder. **Replace them with an allocation
computed from T0.1's measured seconds-per-unit.** Do this once, immediately after T0.1, and write the
arithmetic into `PROGRESS.md` so the choice is auditable.

1. **Compute the budget**: `units_affordable = available_seconds / measured_seconds_per_unit`, using the
   null-rig cost for null tasks and the Coulomb-rig cost for T5. Leave ~20% headroom for re-runs, because
   some units will be killed and some knob checks will fail.
2. **Floor on seeds: 3 per arm, non-negotiable.** A verdict from fewer is not reportable. This project has
   been misled twice by single-seed results (the 24-47x ARTBP variance claim, later corrected to 2-5x by a
   5-seed grid; and the 2.4x velocity-routing number). If the budget cannot afford 3 seeds on a task, cut
   the TASK or its ARMS, never the seeds.
3. **Spend surplus on the DECISIVE tasks first**, not uniformly. Priority for extra seeds: T1 (it gates a
   whole family), then T3a (the strongest candidate if it passes), then T2, then T3b/T4. A wide T1 and a
   narrow T6 is a better night than three seeds everywhere. This ordering must agree with §4.0; if you ever
   find them in conflict, §4.0 wins.
4. **If short, cut in this order**: T6 first (already optional), then T0.4, then narrow T1's beta grid from
   three values to two (keep the extremes), then reduce T4 to arms (a), (b), (d).
5. **Never cut**: the control arm of any task, the cheap-alternative-explanation arm (T4 arm (b),
   decaying-lr alone), or the T5 R2 gate. A candidate without an R2 number is not a fix, so cutting T5
   forfeits the entire campaign's conclusion.
6. **Re-check the budget once mid-campaign** against the wall-clock actually consumed, and re-allocate the
   remainder using the same rules. Log the re-allocation.

### T1 — Can a penalty bite at all under Adam? (targets I5; gates the whole soft-regularizer family)
- **Hypothesis**: the step4 saturation is caused by applying the penalty INSIDE the adaptive preconditioner.
  Applying it proximally (after the Adam step), or replacing it with an update-space projection, restores a
  monotone response.
- **Knob check first**: confirm on one arm that the in-loss penalty reproduces saturation on THIS rig
  (bit-identical or near-identical across beta). If it does not reproduce, the rig differs from step4's and
  the whole task must be re-scoped.
- **Arms** (revised after research thread A, `research/thread-AB-optimizer-mechanics.md`):
  (a) in-loss penalty = CONTROL;
  (b1) EXPLICIT decoupled shrink (the naive AdamW-style step). Expect shrinkage `(1 - lr*beta)` on the
  protected component: NOT monotone in beta, exact annihilation at `lr*beta = 1`, DIVERGENT for
  `lr*beta > 2`. Keep it as the arm that demonstrates why explicit is the wrong form, and stay inside the
  stable band when sweeping;
  (b2) IMPLICIT / exact prox, shrinkage `1/(1 + lr*beta)`: monotone in beta, tends to 0, no instability.
  **This is the primary arm**; it is what "apply the penalty proximally" has to mean;
  (b3) implicit prox taken in ADAM'S OWN metric (preconditioner-scaled), not Euclidean. For our rank-6
  quadratic penalty against a diagonal `V_t` this is a diagonal-plus-low-rank solve, one 6x6 solve per step
  via Woodbury. Basis: ProxGen (Yun, Lozano, Yang, NeurIPS 2021) improves on Euclidean prox precisely by
  "incorporating the effect of preconditioners in the proximal mapping computations" (ABSTRACT-VERIFIED);
  (c) update-space projection (delete the update component along the protected direction; no beta at all).
- **Metric**: the ANN's DC along the protected direction as a function of beta (monotone = bites, flat =
  saturates), plus per-axis free-run drift versus floor, PLUS the residual velocity-row DC in the
  ORTHOGONAL complement after the prox. The last one is not optional: a SEARCH-LEVEL result
  (arXiv:2006.06650) claims projection in constrained nonconvex adaptive methods induces an
  iteration-independent stochastic bias, i.e. we could trade one DC offset for another and call it a win.
- **Size**: 1 epoch x beta in {1e3, 1e6, 1e9} x seeds per the §4.1 budget rule. Arms (c) has no beta.
  Running only the Euclidean prox risks an uninformative null, so (b2) and (b3) are both required.
- **Decision**: (b2), (b3) or (c) monotone / at floor -> promote to T5. All arms flat -> the soft-penalty
  family is dead under Adam; record that as a finding in its own right, drop T6, and put the weight on T2
  and T4. If (b1) diverges as predicted while (b2) is monotone, that contrast is itself a reportable result
  and explains the original step4 saturation.
- **Note for the write-up**: I5 has a published name and mechanism, scale-freeness (Zhuang et al.
  arXiv:2202.00089) plus sign-dominance (Balles and Hennig arXiv:1705.07774). The falsifiable prediction is
  that prox shrinkage depends on beta EXPLICITLY rather than through a gradient, so saturation must vanish.
  That prediction is what this task measures.

### T2 — Curvature-aware step restricted to the output-DC subspace (targets I2, I1)
- **Hypothesis**: Adam parks the DC ~lr from a minimizer at `b* = -g/H ~ 1e-11`. A Newton step in that small
  subspace lands on `b*`, so the drift collapses toward the floor, with no class restriction.
- **Knob check first**: confirm the chosen parameter group actually spans the measured drift direction
  (regress the trained ANN's routed output onto that group). If it does not, the group is wrong and the test
  would be inconclusive by construction.
- **ACCEPTANCE GATE, run BEFORE trusting any second-order step**: measure the minibatch variability of the
  6-dimensional gradient and compare it against the smallest eigenvalue `lambda_min = 1.95e-2`. A near-zero
  eigenvalue amplifies gradient noise through the inverse. Note also that `b* ~ 1e-11` is plausibly BELOW
  the gradient-noise floor, in which case "land on `b*`" is not a meaningful target and the task must be
  re-scoped to "reduce the parked offset by a stated factor" instead. Report the number either way.
- **Arms** (revised after research thread B, `research/thread-AB-optimizer-mechanics.md`):
  (a) Adam = CONTROL;
  (b0) STAGED refinement, the cheapest arm and the field-standard shape (Adam first, then a short refinement
  phase acting only on the 6 parameters). It sidesteps the entire concurrent-group-interaction risk class,
  for which the agent found NO literature at all. Run this first;
  (b1) VARIABLE ELIMINATION / variable projection. If the 6 parameters enter the windowed loss linearly,
  their conditional minimizer is an exact least-squares solve: Newton-exact in one step, no damping, no
  trust region, unconditionally stable for that block. **Our own measurement says this holds in the null**:
  `curvature_sensitivity.py` found `L(b)` eps-invariant, i.e. the map bias -> output -> yhat is affine and
  `L(b)` is EXACTLY quadratic. Basis: Golub and Pereyra (Inverse Problems 19 (2003) R1, SEARCH-LEVEL);
  elimination "reshapes the critical point structure" so saddles become local maxima in the reduced
  landscape (Gan et al., arXiv:2511.01234, ABSTRACT-VERIFIED). **Best R2 story of any arm**: the 6
  directions are SOLVED, not removed, so a Coulomb DC stays fully representable. CAVEAT: exactness is a
  null-regime property; with a trained nonzero ANN and state feedback it becomes an approximation, so
  re-check the quadratic assumption on the Coulomb rig before trusting it there;
  (b2) DAMPED Newton with Levenberg-Marquardt damping, **damping in from the start, not as a fallback**.
  Our block's condition number is ~3.8e6 (eigenvalues 1.95e-2 to 7.43e4), so an undamped step amplifies the
  softest and least reliably measured direction ~51x while shrinking the stiffest by ~1.3e-5. Sweep
  `lambda` over roughly 1e-1 to 1e2. Basis: Clarke and Hernandez-Lobato, arXiv:2310.14963 (PRIMARY-READ by
  the agent): second-order heuristics are "essential components, without which the optimiser will perform
  unstably or ineffectively", and LM damping supplies the trust region, positive-definiteness, and
  prevention of large updates in low-curvature directions.
- **Two design elements to include in every second-order arm** (from Sophia, arXiv:2305.14342,
  ABSTRACT-VERIFIED): clip the 6-dimensional step, and RE-MEASURE the 6x6 Hessian on a cadence rather than
  reusing one snapshot, because clipping exists precisely to tame rapid change of the Hessian along the
  trajectory and our matrix is one checkpoint's snapshot.
- **Metric**: DC magnitude versus `b*` (or versus the gradient-noise floor if `b*` sits below it), per-axis
  drift versus floor, plus step-size and stability diagnostics per arm.
- **Size**: 1 epoch per arm, seeds per §4.1.
- **Decision**: DC reduced to the stated target and drift at floor -> promote to T5 (the R2 gate is
  MANDATORY here, because the SGD route died on exactly this failure mode, inaction). Still drifting -> the
  displacement is not the only carrier; raise T3 per the branch table. If (b0) or (b1) already succeeds,
  do NOT spend the night tuning (b2).

### T3 — The separation task: parity and power sign (targets I4, and I3; revised after research thread C)
Two mechanisms, COMPLEMENTARY, not alternatives. Coulomb friction `F = -Fc*sign(v)` is exactly ODD in `v`;
a spurious constant is exactly EVEN; an anti-damping feedback `F = +k*v` is also ODD. So the parity split
separates the constant exactly but is BLIND to anti-damping, while the power sign catches anti-damping
exactly (`F.v = k*v^2 > 0` always) but catches a constant only on half the trajectory (`c*v` alternates).
Run both, score both, and expect them to fix different issues. Depends on T0.5.

**T3a — parity / even-part penalty (targets I1, I2, I4).**
- **Hypothesis**: split the routed residual at fixed scheduling `p` into
  `f_even = (f(x,v,p) + f(x,-v,p))/2` and `f_odd = (f(x,v,p) - f(x,-v,p))/2`, penalize only the
  v-INDEPENDENT part of `f_even`, leave `f_odd` completely free. Friction's even part is identically zero,
  so friction is untouched; the spurious constant is entirely even, so it is pinned.
- **Why this beats direction-pinning**: it is a STATE-SPACE split, not a time-mean, so a genuine friction
  impulse over an asymmetric duty cycle is NOT forbidden. That is exactly the property whose absence
  demoted the mean penalty and the telescoping block.
- **Knob check first**: T0.5 must show enough velocity reversals per window; without them the even and odd
  parts are not separable on this data and the arm is inconclusive by construction.
- **Honest costs, state them in the results doc**: (i) it forbids a genuine constant velocity-EVEN force,
  which the near-unit-root argument says is not consistently estimable from this data anyway; (ii)
  ASYMMETRIC Coulomb (`Fc+ != Fc-`) has a nonzero even part, so mitigate by penalizing only `f(x, v=0, p)`,
  and note that this variant then also penalizes a genuine static preload.
- **Metric (null)**: per-row DC and per-axis drift. **Metric (Coulomb, MANDATORY)**: windowed %learned,
  which must not fall; this is the arm's whole claim.

**T3b — soft power-sign steering (targets I3). NOT NOVEL: this is a published baseline that FAILED.**
- **Prior art, established 2026-07-25, cite it and do not claim otherwise**: DiLaR-PINN (Long, Solak,
  Ajoudani, arXiv:2604.18277, IFAC 2026) uses exactly this penalty as its "DiLaR-Soft" baseline, with our
  problem statement as its motivation, and reports it FAILING against their hard variant (test RMSE 0.4726
  versus 0.0504, "gradually deviates from the true system behavior"). The same mechanism is long
  established in thermodynamics-informed ML (Jones, Frankel, Johnson, arXiv:2111.14714: the Macauley
  bracket "is identical in form to the ReLU activation function"). A contrast citation worth one line in
  the write-up: arXiv:2604.14678 Eq. (15) uses a RAW SIGNED difference, unbounded below, which rewards
  unlimited energy reduction and would inflate learned friction; that is why the one-sided form matters.
- **Revised hypothesis (this is now the interesting question)**: DiLaR-Soft may have failed for the SAME
  reason our step4 penalty saturated, namely that a soft penalty applied INSIDE an adaptive optimizer caps
  at the step scale (I5). If so, applying it with the implicit prox from T1 should rescue it, and we would
  be EXPLAINING a published negative result rather than merely reproducing it. Treat this as a hypothesis,
  not a claim: their setting, system and optimizer differ from ours, and the transfer is unverified.
- **Not** "the only criterion that separates drift from friction": T3a is exact where this is partial, and
  this is exact where T3a is blind.
- **Knob check first**: confirm the null's anti-damping Jacobian entries are measurable above their noise on
  this rig (prior: `dW_dY/ddY = +1.43e-8`, sd ~1e-11).
- **Metric (null)**: the Jacobian self-feedback entries must fall toward zero or go negative; plus per-axis
  drift. **Metric (Coulomb, MANDATORY)**: windowed %learned not worse than the control.

**T3d — W-PGNN with a MATRIX-valued informativity weight (added 2026-07-25, small and unclaimed).**
The supervisors' own W-PGNN (arXiv:2405.10429) uses a SCALAR reciprocal-KDE weight
`w_j = 1/(sum_k h_k(z_j) + eps)`, `h_k = exp(-||z_k - z_j||^2 / 2 sigma^2)`, applied to squared norms of
BOTH the state and output function differences. Verified: it enforces no stability and restricts no model
class, and directional or frequency weighting is NOT discussed in the paper. Because `w_j` multiplies a
squared norm, replacing it with a MATRIX `W_j` concentrated on the X/Y velocity rows is a small, unclaimed
modification that is R2, R3 and R5 safe and reuses a supervisor-adjacent method. Cheap to try as an
additional application mode once T1 has settled how a penalty must be applied. Score it like the others.

**T3c — parametric Coulomb reference fit (cheap label, no new mechanism).**
Fit a parametric `-Fc*sign(v)` term alongside the network on the same data. If the PARAMETRIC fit also
lands a persistent component, the component is evidence-consistent with genuine friction; if only the
network does, it is an artefact. This gives the spurious-versus-genuine label cheaply and independently of
any regularizer, and it is worth running before or alongside T3a.

- **Application**: use whichever application mode T1 found to actually bite.
- **Size**: 1 epoch null per arm; short Coulomb run for the R2 gate. Seeds per §4.1.
- **Decision**: T3a pins the constant AND friction is still learned -> strongest candidate in the campaign,
  widen it. T3b reduces the Jacobian entries AND friction is still learned -> the Y half is addressed;
  the two together are the composition to recommend. Either one suppressing friction -> R2 fail for THAT
  arm, prune it and say so; do not soften and retry more than once.

### T4 — ARTBP poly-tail paired with a decaying lr (targets I1 and I2 jointly)
- **Hypothesis**: ARTBP removes the gradient bias, but unbiasedness buys convergence only at a vanishing
  step size (the paper's own influence-balancing demonstration uses `eta_t = eta_0/sqrt(1+t)`), and our drift
  is proportional to lr. The PAIRING should beat either alone. Everything run so far has been constant-lr.
- **Arms**: (a) fixed-window Adam at constant lr = CONTROL, (b) **decaying lr ALONE** (the cheap alternative
  explanation, mandatory, because drift is proportional to lr), (c) ARTBP poly6 at constant lr, (d) ARTBP
  poly6 + decaying lr.
- **Metric**: DC endpoint, per-axis drift versus floor, windowed nf-RMS.
- **Size**: reuse `ARTBP/train_artbp.py`; 1 epoch per arm x 3 seeds.
- **Decision**: (d) beats both (b) and (c) -> the pairing is real. (b) alone explains it -> do NOT credit
  ARTBP; report the schedule as the active ingredient.

### T5 — R2 expressivity gate on the Coulomb rig (every promoted candidate)
Same frozen rig, same eval windows, same seeds. Report per axis: windowed %learned versus the
unconstrained-Adam control (R2), `pole_check` on the TRAINED map (R3), free-run envelope ratio (R4).
**Any candidate that materially reduces %learned is pruned here, whatever its drift number.**

### T6 — Free-run-consequence penalty (OPTIONAL, only if T1 says penalties bite and T3 leaves residual Y growth)
Price the accumulation over a horizon instead of naming a direction. Same gate, same scorecard.

### J1 — Separate workstream: does `orth_beta` bite under Adam during joint estimation?
Apply the T1 beta-sweep instrument to the REAL orth penalty in a joint-estimation config. This serves the
thesis contribution (physical-parameter interpretability, non-negation), **not** R4, and is never scored on
drift. Run it once the drift queue is moving; it is cheap.

## 5. Reflect, document, and re-plan

### 5.1 The documentation contract (do this for EVERY test, it is not optional)
Three artifacts per task, all written BEFORE moving on:
1. **Per-unit JSON**, `data/<task>_<arm>_<setting>_seed<k>.json`, written the moment the unit finishes, with
   a fixed schema: task, arm, setting, seed, rig hash, git commit, per-axis metric values, the floor values
   they are referenced against, wall time, status. Append the unit to `data/manifest.json`.
2. **One results document per test**, `results/T<k>-<name>.md`, from `results/TEMPLATE.md`. This is the
   durable record: hypothesis, what would have refuted it, arms, the per-axis numbers table, the verdict
   with its reasoning, what it changes in the queue, and the artifact paths. A task is not finished until
   this file exists.
3. **The rolling record**: scorecard row updated in `PROGRESS.md`, a dated one-paragraph finding appended,
   and the §12 run-table outcome written.

Write numbers per axis, always against the measured floor, never against an oracle. If a run was killed
mid-task, say how many units completed and out of how many; a partial result is reportable, a silently
partial result is not.

### 5.2 The reflection loop (after EVERY run)
1. Read the actual output; extract per-axis numbers; note every error.
2. Run the POST-RUN CRITICAL ASSESSMENT in `CHECKLIST.md`. Do not write a verdict until it passes.
3. Write the three artifacts of §5.1.
4. Branch: PASS -> promote to the T5 R2 gate. FAIL -> prune and re-order. INCONCLUSIVE -> redesign the knob
   (prove it moves the target first), do not re-run the same thing.
5. Anti-churn: after two confounded or failed attempts on the same question, stop multiplying methods; fix
   the experimental design or write the incompatibility argument, then move on.

### 5.3 Pre-declared re-planning branches (so re-planning at 3am is deterministic, not improvised)
Apply these automatically; each one is a plan change you make without asking.

| Trigger | Re-plan |
|---|---|
| **T0.1 cannot produce a stable unconstrained-Adam control** (drift bounces more than ~3x between seeds) | The rig is noise-dominated. Increase the deterministic window bank / use full-batch before ANY candidate runs. Everything downstream is uninterpretable otherwise. |
| **T1: no arm shows a monotone beta response and projection does not reach floor** | The soft-penalty family is dead under Adam. DROP T6 entirely. Reallocate its budget to more seeds on T2 and T4. Record this as a first-class finding, it kills a whole family. |
| **T1: arm (c) update-space projection reaches floor but arm (b) does not** | Use projection as the application mode for T3, not a penalty. Note the rank-1 dodge still applies. |
| **T2: DC lands near b\* but drift persists** | The displaced DC is not the only carrier on this rig. RAISE T3 above T4 in priority; the feedback component dominates. |
| **T2: unstable / diverging Newton step** | Switch to arm (c) damped Newton with a trust region immediately; do not tune the undamped version twice. |
| **T3: friction suppressed on the Coulomb rig** | R2 fail. PRUNE T3 permanently and say so in the scorecard. Do not soften the penalty and retry more than once. |
| **T3: Jacobian entries fall AND friction retained** | This is the strongest candidate. Widen it: more seeds, and add the excited-record variant before anything else in the queue. |
| **T4: decaying-lr ALONE explains the improvement** | Do not credit ARTBP. Report the schedule as the active ingredient and re-test the best other candidate WITH the schedule, since it may be a free additive gain. |
| **T0.3 says Problem 2 is EXPONENTIAL (pole > 1)** | The Y half needs a structural or sign constraint, not conditioning. RAISE T3 to immediately after T1. |
| **T0.3 says Problem 2 is MARGINAL** | Conditioning may suffice. Keep T4 where it is and treat T3 as the separation test rather than the Y fix. |
| **Any candidate passes null R4 but has no R2 number** | It is NOT a fix. It cannot be promoted or recommended until T5 gives it an R2 number. |

### 5.4 If the queue empties before morning
Do not idle and do not invent new methods. In this order: (a) widen the seed count on the single most
informative result, since most verdicts here rest on 3 seeds; (b) re-run the strongest candidate on a
second, excited record to test whether the verdict is record-specific; (c) run J1 if it has not run;
(d) fill any empty scorecard cell that a cheap existing script can fill. Log what you chose and why.

## 6. Run mechanics

- Runs are unattended and may be killed. A unit is `(task, arm, setting, seed)`; write its JSON and append to
  `data/manifest.json` the moment it completes; skip completed units on restart. A kill costs one unit.
- `PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output -n GraduationProject python -u ...`
- Every run gets its §12 run-table row with the hypothesis BEFORE launch (D-090).
- Seeds must reach the model build: `dataclasses.replace(CFG, seed=k)`, and confirm two seeds differ.
- Set experiment flags explicitly; do not inherit `orth_observe` / `orth_beta` from the production `CFG`.

## 7. ASK gates (only these halt; see `CHECKLIST.md`)

Modifying `model_augmentation/` or `kamtin-fp-model/`; real Telica data; the R4-empirical acceptance and
Y-scheduling supervisor decisions; production training on the deliverable configuration. Finishing a task is
not a gate. Wanting to show a skeleton is not a gate: show it and continue.

## 8. What NOT to do

The refuted and eliminated list is in the folder `README.md` §4. In particular: no longer fixed windows, no
lr tuning as a fix, no Adam-versus-SGD swap, no Lipschitz or contraction cap, no mean/zero-mean penalty, no
Theta-only routing, no velocity-domain loss or velocity-only input, and do not resurrect the re-aimed
orthogonal projection as a drift cure. Do not re-derive the diagnosis; it is closed.
