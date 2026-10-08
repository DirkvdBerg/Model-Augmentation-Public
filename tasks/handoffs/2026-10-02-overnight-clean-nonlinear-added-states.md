# Handoff: overnight search, falsification, and prototype of a clean nonlinear added-state learning method
**From**: session of 2026-10-02 | **Branch**: Augmentation | **Effort suggested**: xhigh

## 1. Task
Work autonomously through the overnight window to identify the best-supported clean method for
making the additional states in Hoekstra's dynamic parallel LFR augmentation learn missing
dynamics, including nonlinear missing dynamics. Start from the evidence in this handoff, perform
the required primary-literature research, compare at most three principled candidate methods,
falsify them with the cheapest informative local experiments, and implement an isolated prototype
only for a candidate that passes its preliminary mechanism tests. Use the existing known-controller
closed-loop simulator and the deployed nonlinear S-DP state map. Do not use the absorber frequency,
damping, poles, true absorber states, band filters, or any other oracle information in training,
model selection, schedules, or pass criteria. Continue through research, local runs, monitoring,
analysis, and documentation without stopping after a plan. A rigorous negative result is an
acceptable outcome if no candidate earns implementation.

In the user's words, the core method must support **nonlinear additional states**, must not be
**monkey patching**, and must not deliberately implement the absorber or any other oracle-known
missing dynamic. Operationally, this means the method may use data-derived predictive targets and
temporary training components, but it may not use planted dynamics, known bands, known periods,
truth states, or target and schedule choices reverse-engineered from this absorber. Prefer a
training-method extension that leaves the deployed Hoekstra S-DP model unchanged. Rank any method
that changes the deployed architecture below an otherwise adequate training-only method, and
require an architecture-only control if such a change is essential.

The user explicitly authorises this successor session to create experiment-local code, run and
monitor local experiments overnight, and select the strongest method supported by the evidence.
The objective is the best-supported method among the candidates examined, not a claim of global
optimality or guaranteed supervisor acceptance.

The required morning deliverable is deliberately narrower than a proof of generality: either an
evidence-backed gantry prototype with a fully specified nonlinear generality experiment, or a
precise negative result. Run the nonlinear benchmark overnight only if the gantry mechanism and
paired controls finish with enough time remaining. Do not rush it merely to satisfy a checklist.

## 2. Out of scope

- Production integration into `model_augmentation/` or the established
  `scripts/gantry/gantry_dynamic/` pipeline. Produce an isolated prototype and an integration plan
  first.
- Server or cluster submissions. This task uses local computation only.
- Contacting supervisors, sending messages, opening a pull request, committing, or pushing.
- Changing the physical baseline, controller, OBC definition, thesis validation selector, or data
  generation.
- A broad hyperparameter search. Search only parameters required to distinguish a candidate's
  mechanism.
- Claims of generality based only on the 212 Hz absorber.
- Editing the two existing proposals
  `IMPLEMENTATION-PLAN.md` and `NONLINEAR-PREDICTIVE-STATE-IMPLEMENTATION-PLAN.md`. Critique them in
  the new report.
- Reading `kamtin-data/Data Telica/` or otherwise bypassing the repository data-access policy.

## 3. Where things stand

Branch `Augmentation`, HEAD `c9d096c`. The worktree is very dirty across many unrelated tracked
and untracked paths. Preserve every existing change. No run from this session is in flight.

The relevant experiment directory already contains many untracked scripts, logs, checkpoints, and
two proposal documents. Create all new overnight artefacts under:

`scripts/gantry/absorber-learning-diagnosis/overnight-clean-method/`

Use unique filenames and do not overwrite any existing result. Append the run hypothesis and
result where required by the run-discipline rule. Before implementing a nontrivial selected
method, append one narrowly scoped entry to `docs/decisions.md`, preserving all existing content.

The production configuration uses the known-controller closed-loop training simulator:
`scripts/gantry/gantry_interconnect_dynamic.py` sets `closed_loop=True`. Recorded-input replay is a
different and worse-conditioned problem for the free X and Y integrators and is not the primary
training path for this task.

## 4. Established and verified

1. **Current model.** Hoekstra-style S-DP is implemented as a physical baseline plus one
   zero-output-initialized tanh MLP. Its input is \([x_b,x_a,u]\); routed physical rows are additive
   corrections and routed added rows are the complete next value of \(x_a\). The encoder has a
   baseline-informed linear part, random augmented rows, and a zero-output nonlinear part.

2. **Dead start.** With the complete ANN output layer zero, \(x_a\) starts dead. Plain thesis
   training learns static corrections or real lags but has not learned the missing oscillatory
   mode. U1 measured the effective \(x_a\) recurrence gain around 0.055 to 0.087.

3. **Optimizer-only changes failed.** Increasing ordinary-loss learning rates, early
   Levenberg-Marquardt, and related speed changes did not create the missing pole. The missing
   direction, not merely update magnitude, is the established problem. See
   `scripts/gantry/absorber-learning-diagnosis/DECISIONS.md` and
   `tasks/handoffs/2026-10-02-review-ordered-innovation-opening.md`.

4. **Random live dynamics are a strong negative prior.** U2 was neutral. AE1 found that only
   0.42 percent of literal random draws produced a near-band lightly damped pair. Landscape probes
   L1, W1, and X1 found little pull toward the missing mode when the initial dynamics were far from
   it. In the stable linear recurrence of
   `model_augmentation/fit_systems/augmented_dynamics.py`, planted poles moved less than 0.15 Hz
   over 520 updates and behaved as a fixed basis rather than an adaptive resonator.

5. **The only positive local mechanism uses a mode-carrying target.** E5 and E9 used a temporary
   head that predicts the next output-residual samples from \(x_a\), followed by a detached-teacher
   fit of the deployed added-state map and then the ordinary thesis loss. They learned local pairs
   near 208 to 209 Hz in two completed local outcomes. The fixed 300-update schedule completed one
   of three seeds; extending the innovation phase rescued one previously partial seed. This is not
   a two-of-four robustness rate.

6. **That result is not yet a general method.** The tested head was linear, the schedules and
   rate factors were hand selected, the target moved with the current model, one partial run
   self-oscillated on quiet records, and no server-scale or OBC result establishes transfer.

7. **The prior NPSP proposal is rejected as the primary direction.** Output simulation plus
   encoder-transition consistency supplies no better-conditioned mode-selection signal than the
   output-error path that already failed. Its consistency loss permits collapse, it used
   recorded-input replay instead of the production closed-loop simulator, and it changed the
   deployed architecture and encoder substantially. It remains a negative candidate or ablation,
   not the default implementation plan.

8. **Hoekstra references.** In arXiv:2602.17297v1, Table 1 on p. 3 defines S-DP with separate
   \(f_{\mathrm{aug}}\) and \(g_{\mathrm{aug}}\); Eqs. (5) and (6) are on p. 4; Eqs. (14) and (15)
   define ANN and ResNet learning functions; Eq. (22) on p. 8 is the encoder-initialized truncated
   simulation criterion; Eq. (29) and Eq. (31) are on p. 9. Section 5.4.3 sets the nonlinear part
   to zero for its initialization construction. Section 6.3 uses feedforward networks for the
   parallel experiments.

9. **Closed-loop identification matters.** A residual derived from closed-loop signals can carry
   controller dynamics and feedback-correlated noise. Any consistency or identifiability claim
   must state its assumptions. Ramos, Mercere, and Markovsky, CDC 2016, is relevant to
   innovation-based linear closed-loop subspace identification but does not validate a nonlinear
   objective automatically.

10. **Paper/code initialization provenance remains open.** D-233 records an ambiguity between
    Hoekstra's paper construction and Jan's dynamic-parallel example code, and Jan's answer on the
    initialization used for the reported S-DP result is pending. This affects the historical
    explanation, but it does not override the measured behaviour of the present gantry pipeline.

## 5. Assumed but not verified

- A nonlinear residual-predictive representation can shape \(x_a\) without merely learning
  periodic multisine phase. Settle with unseen phase or nonperiodic validation and an added-state
  clamp.
- A detached or alternating teacher arrangement can prevent collapse while still teaching the
  deployed nonlinear \(g_{\mathrm{aug}}\). Settle by gradient analysis and a collapse control.
- A literature-grounded method exists that is stronger than the present linear reduced-rank
  interpretation for input-driven nonlinear systems. The existing literature reports may conclude
  only partial support; do not convert analogy into theorem.
- A method that works on the gantry absorber will also work when the missing state transition is
  genuinely nonlinear. Design one separate nonlinear missing-dynamics benchmark using the same
  method and selection rules. Run it only after the gantry gates and paired controls pass.
- A temporary training head can be acceptable to the supervisors if it is a principled
  predictive-state estimator, trains the deployed nonlinear transition, uses no oracle
  information, and is discarded cleanly. This is a scientific judgment to support, not assume.

## 6. Tried and failed

- Plain zero-initialized output-error training -> static corrections or real lags, no missing
  oscillatory mode -> the added-state path has no useful initial state or pole gradient -> U1 and
  the main diagnosis logs.
- Large learning rates on the plain loss -> no useful pole -> speed cannot manufacture the missing
  direction -> U1/U4.
- Random live initialization -> neutral result and almost no near-band draws -> output error is
  locally too weak to move far-away dynamics -> U2, AE1, L1/W1/X1.
- Equation-error fitting to the random encoder alone -> slow real lags -> the random encoder state
  contains baseline-like slow content rather than the missing mode -> E1/E2.
- Simultaneous fast residual-state and state-map training from update one -> slow content is fitted
  before a predictive state forms -> E4.
- NPSP as currently documented -> rejected before implementation because output error plus
  consistency repeats the failed signal path and adds collapse and architecture confounds ->
  `NONLINEAR-PREDICTIVE-STATE-IMPLEMENTATION-PLAN.md` plus the critique in the source session.

Do not present any of these as a new remedy without a materially different mechanism and a cheap
test that isolates that difference.

## 7. Achieved

- A working local diagnostic harness exists in
  `scripts/gantry/absorber-learning-diagnosis/`. Reuse `u_probe.py` with `U_INIT=hoekstra` and an
  explicitly recorded `U_SKIP_SCALE` for a live-random control; `g2_steal.py` for baseline stealing;
  `ac_band.py` for post-selection frequency diagnostics; and `ad_valfree.py` for whole-record and
  quiet-record behaviour. Reuse the existing E4 to E9 opening code, pole tracking, checkpoints,
  and logs rather than rebuilding them.
- The ordered innovation opening demonstrates that a residual-predictive target can organize two
  added states into a useful oscillatory representation in some runs.
- Three targeted literature reports already exist under `literature/added-states-training/` and
  should be used before new searches.
- `tasks/handoffs/2026-10-02-lit-nonlinear-residual-predictive-state.md` already frames the wider
  nonlinear predictive-state literature search. This overnight task supersedes that separate
  literature-only task: reuse its N1 to N4 questions, do not run both handoffs independently, and
  produce one integrated research result.
- The rejected NPSP plan provides useful engineering requirements for OBC, L-BFGS, checkpoint,
  dtype, device, and routing isolation even though its central learning objective is unsupported.

## 8. The open question

What is the cleanest literature-grounded, data-only training method that gives Hoekstra's
additional nonlinear states a predictive target, makes the deployed nonlinear transition
propagate that state, works with the known-controller closed-loop objective, and shows useful
added-state memory on both the gantry case and a nonlinear missing-dynamics case without
absorber-specific information?

Candidate answers may include nonlinear residual-predictive state learning with a detached or
alternating teacher, latent-state multiple shooting followed by encoder amortization, or another
method supported more strongly by the literature. Output-error-only live initialization is the
required negative control, not a leading candidate. Compare at most three method families.

## 9. Next action

Execute the following single evidence-gated selection pipeline from start to finish.

### Gate A: frame and audit

1. Read the five items in Section 11.
2. Treat `tasks/handoffs/2026-10-02-lit-nonlinear-residual-predictive-state.md` as subsumed by this
   task. Use its N1 to N4 framing and the existing LA, LB, and LC reports before searching. Run the
   project's deep-research skill with FRAME first only for gaps that remain. Time-box new research
   to the unresolved decision: how a nonlinear latent state is anchored to future predictive
   information while being constrained to propagate under its deployed state transition, with
   inputs and closed-loop data.
3. Produce a candidate matrix with at most three rows and these columns: deployed model change,
   objective, mode-carrying signal, collapse defence, closed-loop assumptions, nonlinear support,
   zero-step gradient path, system-specific choices, literature guarantee, and cheapest
   falsification.
4. Rank the candidates before running new training. Write the predicted outcome of every planned
   arm before launch.
5. For every candidate, declare before its first run how the predictive horizon, auxiliary-loss
   scale, learning rates, and any switch condition are chosen. `M=40`, the E5/E9 `100x` and `1000x`
   rate factors, and their update counts are prior gantry findings, not general defaults. A choice
   must follow from a cited method, an existing framework quantity such as encoder or rollout
   length, or a predeclared data-only calibration that never inspects pole proximity. If no clean
   rule exists, count that as a limitation of the candidate rather than hiding the choice.

### Gate B: mechanism analysis

For each candidate, derive the gradient path at the current zero initialization and answer:

1. What trains \(\psi_a\) before the model reads \(x_a\)?
2. What forces \(x_a\) to contain temporally predictive information rather than a constant?
3. What trains the deployed \(g_{\mathrm{aug}}\) to propagate that information?
4. Which tensor is detached, fixed, alternated, or independently anchored?
5. Can \(f_{\mathrm{aug}}(x_b,u)\) bypass \(x_a\)?
6. Does the known-controller closed-loop rollout remain the optimized object?

Reject a candidate at this gate if these questions expose an exact dead path, an unopposed
constant-state solution, oracle dependence, or a different plant-input replay objective.

### Gate C: cheap local falsification

1. Reuse the exact assets named in Section 7. Do not first build a new training framework. Audit
   existing U2, AE1, E4 to E9, clamp, steal, and quiet-record results before deciding that a control
   is missing.
2. The required negative control is frozen physical parameters, live random added-state
   initialization, production closed-loop simulation, and output error only. It already has an
   implementation in `u_probe.py` with `U_INIT=hoekstra`. Run only the missing matched seed or scale
   needed to answer the selected candidate's comparison; do not automatically repeat three full
   seeds.
3. For the top candidate, implement the smallest experiment-local change that isolates its new
   learning signal. Start with one seed and at most 150 updates. Continue that seed to at most 600
   updates only if the predeclared intermediate mechanism appears.
4. Continue a candidate only if its intended target becomes predictable, the deployed added-state
   map fits or improves that representation, and clamping \(x_a\) worsens held-out prediction.
5. Abandon it immediately if the state collapses, only a temporary head improves, the deployed
   map remains unused, quiet records self-oscillate, or the gain comes from a changed architecture
   control.
6. Run up to three paired local seeds only for a candidate that passes the one-seed mechanism
   gate. Count continuations as the same run. Keep the total overnight budget to at most six new
   training starts: one missing control, one short candidate screen, its continuation, two paired
   seeds, and one architecture or detach ablation. Spend fewer when existing artefacts settle a
   comparison.

### Gate D: nonlinear generality check

For a gantry candidate that passes Gate C, write an executable design for one cheap input-driven
synthetic example whose omitted subsystem has a genuinely nonlinear internal state transition.
Jan's three-DOF hardening-spring MSD is not sufficient by itself: its omitted mass supplies extra
states with linear internal dynamics even though the overall plant has a nonlinear spring. A valid
example may use a hidden Duffing-type oscillator, hysteretic internal state, or amplitude-dependent
hidden damping, provided no truth state, parameter, or frequency enters training or selection.

Apply the same objective, detach rules, horizon-selection rule, and learning-rate rule as on the
gantry. Ground truth may be used only after selection for diagnosis. Compare against ordinary S-DP
and the simplest linear predictive-head control. Implement and run Gate D only if Gates A to C and
the paired gantry diagnostics finish with enough overnight time remaining. Otherwise the complete
benchmark specification, data split, equations, predictions, commands, and pass criteria are the
required deliverable.

### Gate E: consolidate

1. If one candidate passes Gates A to C, clean the experiment-local prototype, add focused tests,
   and write a
   minimal production-integration plan. The final deployed model must remain Hoekstra S-DP unless
   the evidence shows a model change is essential; any such change needs an architecture-only
   control and must be stated as part of the contribution.
2. If none passes, stop implementing and write the negative result, including which assumption
   failed and the single most informative future experiment.
3. Classify the outcome as framework candidate, gantry-supported prototype, or negative result
   using Section 10. Do not call a gantry-only result generalizable.
4. Write the morning report at
   `scripts/gantry/absorber-learning-diagnosis/overnight-clean-method/REPORT.md`.
   End it with a supervisor-ready method statement: the gap in Hoekstra's framework, the proposed
   equations and training algorithm, what remains unchanged, measured evidence, and claims that
   are explicitly not yet supported.

The report's opening should follow this example style:

> Outcome: Candidate B passed the gradient, collapse, added-state clamp, and paired gantry gates.
> Gate D is specified but not yet run, so nonlinear generality and closed-loop noise consistency
> remain unproved. It is a gantry-supported thesis prototype, not yet a framework contribution.

## 10. Acceptance criterion

The overnight task is complete when the report and artefacts support exactly one of the following
three outcome levels. Lack of wall-clock time for Gate D does not convert a passing gantry method
into a negative result.

### Gantry-supported prototype

All of the following are present:

1. A primary-literature-grounded mathematical objective and an exact statement of which part is a
   new heuristic extension.
2. No oracle quantity, absorber-derived band, planted pole, true missing state, or known-mode
   horizon appears in training, tuning, stopping, or checkpoint selection.
3. The production known-controller closed-loop simulator is used for gantry training.
4. A zero-step gradient and detach-path account shows how \(\psi_a\), the deployed
   \(g_{\mathrm{aug}}\), and the physical readout learn.
5. The method beats its paired ordinary S-DP and architecture-only controls on held-out free-run
   performance in at least two of three local seeds, with the median also improved.
6. On the fixed held-out calibration records, the paired per-record or per-window increase in
   free-run error after clamping \(x_a\) has a bootstrap confidence interval for its median that
   excludes zero. The resampling unit respects record boundaries. This demonstrates that the
   deployed model uses added-state memory without introducing an oracle threshold.
7. Quiet-record stability is no worse than the paired control and all failed or unstable seeds are
   reported.
8. Gate D is specified completely, including why its missing internal transition is genuinely
   nonlinear and why no truth quantity enters selection.
9. The final report states closed-loop, identifiability, excitation, state-gauge, order-selection,
   and novelty limitations.

This outcome may be called a gantry-supported prototype, not a general nonlinear framework
extension.

### Framework candidate

All gantry-supported criteria hold, Gate D is implemented, and the same method and predeclared
selection rules improve over ordinary S-DP on the genuinely nonlinear missing-dynamics example.
The result may be called a candidate extension to Hoekstra's framework, while retaining all stated
theoretical and closed-loop limitations. It is still not a proof of universal generalization.

### Negative completion

Every candidate is rejected by a predeclared gate with a mathematical reason or measured result,
all runs and failures are preserved, and the report identifies one remaining research question
without presenting an unsupported implementation as the solution.

Absorber pole proximity is never an acceptance threshold. It may be reported only after model and
checkpoint selection as a mechanism diagnostic.

## 11. Read these first

1. `tasks/handoffs/2026-10-02-review-ordered-innovation-opening.md`: exact E4 to E9 method,
   evidence, confounds, and existing artefacts.
2. `scripts/gantry/absorber-learning-diagnosis/DECISIONS.md`, overnight section through E9: raw
   predictions and measured outcomes.
3. `tasks/handoffs/2026-10-02-lit-nonlinear-residual-predictive-state.md`, then
   `literature/added-states-training/deep-research-LA-latent-mode-training.md` and the sibling LB
   and LC reports: the already framed nonlinear search and existing research on latent-mode
   training, bias, and closed-loop innovations. This task supersedes the separate handoff.
4. `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`: the framework being
   extended, especially Table 1 and Sections 5.1 and 5.4.
5. `scripts/gantry/absorber-learning-diagnosis/IMPLEMENTATION-PLAN.md` and
   `NONLINEAR-PREDICTIVE-STATE-IMPLEMENTATION-PLAN.md`: two rejected or unvalidated proposals;
   extract useful engineering constraints but do not inherit their conclusions.

## 12. Do not

- Use 212 Hz, its damping, an absorber state, a bandpass around it, a planted resonator, a pole
  bank selected around it, or any truth-model quantity in learning or selection.
- Treat a linear recurrence or hand-selected basis as the core general nonlinear solution.
- Reject all temporary heads merely because they are temporary. Reject them only if they lack a
  principled role, fail to train the deployed state map, or do not transfer.
- Reuse output-error-only NPSP as a leading candidate; it is the negative control from Section 6.
- Replace the closed-loop objective with recorded-input replay.
- Tune against official validation or test records. Use training-internal calibration where a
  choice is unavoidable and report the reduced effective training data.
- Select on learned pole proximity or inspect truth poles before selection.
- Resample failed initializations, omit unstable runs, or extend only the seeds that look
  promising without reporting the selection.
- Build a large framework, new artifact format, or nine-file pipeline before a one-seed mechanism
  test passes.
- Inherit `M=40`, the E5/E9 learning-rate multipliers, or their switch counts as defaults. They are
  system-informed prior results unless a new system-independent rule independently selects them.
- Modify or delete existing user files outside the narrow append-only logs authorised in Section
  3 and the new overnight directory.

## 13. Operational

Use conda environment `GraduationProject`. The established local probes are CPU-only; do not call
`fs.cuda()` or run two training jobs concurrently. Available memory is limited, so one detached CPU
job at a time is the safe default.

Long runs must survive a session or shell exit. Use the established
`scripts/gantry/absorber-learning-diagnosis/run.cmd` pattern and launch a dedicated experiment or
chain command file with hidden `Start-Process`. `run.cmd` selects the environment and enables
unbuffered output. Redirect each run to a unique log under
`scripts/gantry/absorber-learning-diagnosis/overnight-clean-method/outputs/` and write an explicit
done marker. Do not rely on a terminal-owned background process. Poll logs at declared update
milestones or expected completion times while doing research or analysis between polls; do not
busy-poll an hour-long run.

Start with existing local probe code and copy only the minimum needed into the overnight
directory. Record for every run: hypothesis, candidate, seed, exact configuration, update budget,
prediction, pass/fail rule, runtime, checkpoint, and outcome. Save machine-readable histories for
the primary metric, auxiliary objective, added-state clamp, and quiet-record check.

Use wall-clock-efficient successive halving: analytical gate, one short seed, one full seed, then
paired seeds. Existing opening runs take about 4 to 5 seconds per update at batch 64, while some
ordinary controls are slower. Budget from measured timing before each launch. Continue analysis
while a detached job runs, and terminate only under its predeclared failure rule.

No server credentials, external data, or cluster dependency is authorised.

## 14. Delegation

This is a wide research and engineering task, but most of the literature and code mapping already
exist. Use at most two literature subagents, divided between nonlinear predictive-state theory and
latent-state estimation or grey-box residual dynamics. Do not use a code Explore subagent unless a
specific required component cannot be located with targeted search. The main session must read the
selected papers, make the candidate decision, own all code changes, run the experiments, and
synthesize the report. Do not delegate independent reviews of the same result.
