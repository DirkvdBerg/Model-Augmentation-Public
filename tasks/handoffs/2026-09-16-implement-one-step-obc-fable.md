# Handoff: implement one-step orthogonal-by-construction augmentation for the gantry

**From**: session of 2026-09-16 | **Branch**: `Augmentation` | **Effort suggested**: `high`

Run this prompt with Claude Fable 5.1 at high effort. This is an unattended implementation task.
The user will not be available to answer questions during the run.

## 1. Task

Implement the first complete one-step orthogonal-by-construction, or OBC, path for the gantry in
`model_augmentation/` and `scripts/gantry/gantry_dynamic/`. Follow Gyorok's construction at the
level of one discrete plant state transition, using the ten identifiable parameters of
`Reduced_Gantry_State_Block`. Keep the existing physical plus augmentation state model. Do not
introduce delayed input or output states. Deliver reusable OBC code, gantry integration, focused
tests, numerical diagnostics, and runtime measurements. Do not launch a training campaign.

Read the mathematical specification at
`scripts/gantry/orthogonal-by-construction/documentation/obc-gantry-one-step-implementation-plan.tex`
before editing code.

## 2. Out of scope

* Do not run an OBC training experiment. The next session chooses whether to train from the
  diagnostics produced here.
* Do not redesign or regenerate the gantry dataset. Qualify the active dataset only to the extent
  needed for this implementation and report any limitation.
* Do not implement multistep, input-output, closed-loop-Jacobian, or controller-Jacobian
  orthogonality. The first construction is one plant timestep.
* Do not introduce a delay-coordinate state. Continue with the physical plus augmentation states.
* Do not change the default behavior of `gantry_interconnect_dynamic.py` or `RunConfig` when OBC is
  disabled.
* Do not modify `kamtin-fp-model/`.
* Do not use, modify, or delete the untracked remnant
  `model_augmentation/fit_systems/obc_projection.py`.
* Do not repair the older stage tests that import that remnant. Reconciliation is a separate task.
* Do not extend the primary protected space with an offset column. The primary construction protects
  `range(J)` for the ten reduced parameters.

## 3. Where things stand

The current branch is `Augmentation`. The starting commit is
`a52dd5584065f87bdeceef4b6f4a3ccdf05fb6cd`.

The working tree is heavily dirty across unrelated experiments. Treat it as user work. Modify only
the paths allowed in section 12. Do not clean, restore, move, or commit unrelated files.

The active pipeline is `scripts/gantry/gantry_interconnect_dynamic.py`. It currently uses:

* dataset mode `augmentation_ma50_b140-230_a6_z03`;
* `joint_estimation=False`;
* `physics_parameterization='reduced'`;
* ten combination-space detuning factors;
* closed-loop training;
* `nx_ann=8` and all fourteen interconnected state rows routed;
* CUDA with `compile_mode='reduce-overhead'`.

No run is currently part of this handoff.

## 4. Established and verified

1. `Reduced_Gantry_State_Block` directly trains the ten identifiable combinations. Use its free
   coordinates and `COMBO_NAMES`. Do not reconstruct a fourteen-parameter raw estimation path.
2. The active dataset contains fourteen training records listed in
   `scripts/gantry/gantry_dynamic/data.py`. They cover five fixed Y positions, Y sweeps, mixed-axis
   excitation, APRBS records, and Lissajous records.
3. The active dataset was selected to expose the added absorber pole and anti-resonance in the
   140 to 230 Hz band. D-188 records that purpose. It was not qualified specifically for practical
   joint identifiability of all ten reduced parameter combinations.
4. An existing sampled geometry reported rank ten, but that does not establish practical
   identifiability on the complete active training set.
5. Training uses the recorded total plant force `u_total`. The controller may remain active in the
   rollout, but the controller derivative is absent from the one-step OBC basis.
6. `orthogonal-IO-augm-main/orthogonalx_augm/src.py` is the local authority for Gyorok's
   by-construction lifecycle: build the baseline basis outside the objective, recompute the
   auxiliary coefficient from the current ANN inside each objective, and allow gradients through
   that coefficient.
7. `orthogonal-augmentation-main` is useful for state-space stacking and sensitivity conventions.
   It is a soft regularization predecessor and does not define the present dynamic-state OBC path.
8. The state-equation baseline has zero entries in the additional-state rows. Those rows therefore
   need no subtraction in this one-step definition. Their accumulated influence on future physical
   outputs is outside this first condition.
9. The production compiler compiles one interconnection step, not the whole rollout. Any OBC seam
   must preserve the compiled path and avoid recompilation per objective.

## 5. Assumed but not verified

1. The complete active training set supplies a numerically usable rank-ten baseline tangent space
   at both the nominal and ten-percent-detuned parameter points. The dataset qualification in
   section 9 must settle this without starting a separate dataset study.
2. A coefficient tensor can be threaded into the compiled interconnection step without graph
   breaks or recompilation.
3. A forward-mode JVP through the reduced RK4 transition compiles under PyTorch 2.5.1 with
   `reduce-overhead` and `fullgraph=True`.
4. The full reference set fits the target device and its once-per-objective ANN evaluation has an
   acceptable cost.
5. Projecting only the physical ANN rows against `range(J)` improves parameter separation for this
   discrepancy. The requested `rho` diagnostic measures whether that premise is plausible, but no
   training conclusion is allowed in this task.

## 6. Tried and failed

* Delay-state realization of Gyorok's finite-memory input-output model was derived, but the
  supervisor rejected it as an unnecessary model change. The gantry already has physical and
  augmentation states. Use that representation directly.
* Epoch-frozen auxiliary coefficients were considered. They fail the by-construction lifecycle
  because the ANN changes inside the epoch while the subtraction would remain fixed.
* Recomputing the complete coefficient inside every timestep preserves values but repeats the
  reference ANN pass and autograd graph for every rollout step. Compute it once per objective.
* The remnant `obc_projection.py` uses the wrong lifecycle and an eleven-column basis. It is not a
  starting point.
* Forming a normal-equation inverse is numerically weaker and unnecessary. Use an economy SVD.
* Treating a rank-ten result as proof that the dataset supports accurate joint estimation confuses
  structural rank with practical excitation. Report conditioning and contribution diagnostics.

## 7. Achieved

The derivation and implementation strategy are documented in
`scripts/gantry/orthogonal-by-construction/documentation/obc-gantry-one-step-implementation-plan.tex`.

The reduced gantry block and combination-space detuning configuration already exist. The active
entry point contains the intended ten detuning factors in `COMBO_NAMES` order:
`[kb_sum, cg1, cg2, cy, cb_sum, mh, m_total, m_diff, J_eff, d]`.

No production OBC module satisfying this handoff has been completed.

## 8. The open question

The only implementation-level uncertainty is whether the exact JVP correction can remain inside
the current compiled step without graph breaks, per-objective recompilation, or unacceptable
runtime. Resolve this with the compile probe first. Use the fallback ladder in section 9 only if
the exact form fails.

Dataset suitability is a reported qualification result, not the open implementation question. A
poorly conditioned active dataset does not authorize dataset generation and does not cancel the
generic OBC implementation.

## 9. Next action

Implement the complete one-step OBC path under the following contract.

### 9.1 Mathematical invariant

For fixed reference tuples and the current epoch's baseline expansion point, define the stacked
baseline sensitivity and current ANN physical contribution as

```text
J       = col_k d f_base^d(v, x_b,k, u_k) / d v evaluated at vbar
F_aug   = col_k f_aug^b(x_b,k, x_a=0, u_k; theta_a)
K_J     = J^+ from an economy SVD
theta_aux = K_J F_aug
F_tilde = F_aug - J theta_aux
```

`theta_aux` must be computed exactly once per loss or objective evaluation from the current ANN,
then reused unchanged at every timestep of that rollout. Gradients must flow through
`theta_aux` to the ANN weights. It must not be frozen across optimizer updates and must not be
recomputed inside the per-timestep block call.

At a rollout point, apply

```text
f_tilde_aug^b(xhat_k, u_k)
    = f_aug^b(xhat_k, u_k) - J_b(vbar, x_b,k, u_k) theta_aux
```

Evaluate the product as a parameter-direction JVP. Leave the additional-state rows unchanged.

### 9.2 Protected coordinates and dimensions

* Physical state dimension: 6.
* Additional learned state dimension in the active configuration: 8.
* Input dimension: 3.
* Output dimension: 3.
* Protected parameter dimension: 10 reduced free coordinates.
* Protect the physical rows to which the ANN writes. With the active all-row routing, this is all
  six physical rows.
* Work in the normalized state and input coordinates already used by the block and ANN. Do not add
  a second state scaling.

### 9.3 Fixed reference tuples

Use every valid sample from all fourteen files in `TRAIN_FILES`. For each record, construct

```text
q_k       = P^{-T} y_k
qdot_k    = fourth-order central difference of q within that record
x_b,k     = col(q_k, qdot_k)
x_a,k     = 0
u_k       = recorded u_total after the pipeline's existing resampling
```

Use the fourth-order central derivative only where its full stencil exists. Trim two samples from
each end of each record. Never cross a record boundary. Label this numerical stencil with the
required `# THEORY:` comment.

These reference tuples remain fixed during training. At each epoch boundary, rebuild `J` at a
detached copy of the current reduced physical parameter vector `vbar`. Hold that `J`, its SVD, and
`K_J` fixed during the epoch. This moving tangent basis is distinct from the fixed data-derived
reference states.

The controller does not enter this construction. The actual training rollout remains closed loop.
The recorded `u_total` values define the plant inputs at the reference tuples.

### 9.4 Dataset qualification kept inside the implementation

While constructing the basis, report the following at the nominal and ten-percent-detuned reduced
parameter points:

* all ten singular values;
* numerical rank under the declared SVD tolerance;
* condition number after the block's existing coordinate parameterization;
* normalized column norms and pairwise column correlations;
* contribution of each of the fourteen records to each column norm;
* rank and smallest singular value after leaving out each record once.

This is a bounded preflight, not a new dataset campaign. Do not compare every dataset directory and
do not generate data. If the active set is rank deficient or severely ill conditioned, finish the
generic code and all diagnostics that remain meaningful, refuse any claim that the dataset is
qualified for joint estimation, and state which directions and records caused the limitation.

### 9.5 Numerical construction

Build `J`, its economy SVD, and its pseudoinverse in float64 under `no_grad`, outside the hot path.
Declare and report the rank tolerance. Cast the stored basis and solve factors once to the pipeline
dtype. Never form or invert `J.T @ J`.

At an epoch refresh, if numerical rank falls below ten, retain the previous valid basis and its
`vbar`, report the spectrum, and continue. Abort OBC training readiness if this occurs at two
consecutive refreshes. Since this task does not run training, test this lifecycle synthetically.

Measure the attainable float32 orthogonality floor by projecting random fields with the same shape
and norm. Derive the acceptance threshold from that floor. Do not hard-code an idealized float64
threshold.

### 9.6 Generic and gantry-specific boundaries

Create `model_augmentation/fit_systems/obc.py` with the required project-origin marker. It owns:

* the SVD basis representation and rank report;
* the differentiable auxiliary-coefficient solve;
* the once-per-objective lifecycle;
* generic orthogonality diagnostics.

The generic module accepts `step(v, z) -> x_plus`. It must not know about gantry parameter names,
`free_params`, P transforms, controllers, trajectory files, or absorber states.

Put the reference construction, reduced gantry step adapter, epoch refresh, dataset qualification,
and gantry diagnostics under `scripts/gantry/gantry_dynamic/`.

### 9.7 Compile seam and fallback ladder

Start with a standalone CUDA compile probe. Pass `theta_aux` as a tensor input through the compiled
step so its address and graph contract are explicit. Run at least two calls with different
coefficient values. Report graph breaks, recompiles, and corrected versus uncorrected step time.

Use this fallback order:

1. Exact JVP with `fullgraph=True` and no per-objective recompile.
2. If it recompiles, fix the argument seam so the coefficient is lifted as an input.
3. If exact JVP cannot trace, measure `fullgraph=False` before changing the mathematics.
4. If that is unusably slow, test a pure-PyTorch directional derivative against the exact eager
   JVP. A finite-difference fallback must be labelled `# HEURISTIC:` and its error reported.

Do not carry the coefficient as additional model states. Do not compile the complete Python rollout
loop as a workaround.

### 9.8 Diagnostics required before any future training

Report

```text
rho(B, delta) = ||P_B delta|| / ||delta||
```

in a two-by-two table. Cross basis `J` and `[J | Gamma]` with recorded absorber states and zero
absorber states. The primary implementation remains `J`; `[J | Gamma]` is diagnostic only.

Measure separately:

* the once-per-objective reference ANN pass plus coefficient solve;
* the per-step JVP correction;
* total update time relative to the compiled disabled path;
* resident memory for the reference tuples, basis, solve factors, and retained ANN graph.

If reference decimation is measured, compare its principal angles and singular values with the full
basis. Do not adopt decimation in the primary implementation. The complete training dataset is the
definition requested by the user.

### 9.9 Configuration surface

Add explicit OBC configuration fields without changing existing defaults. Provide three future
experiment configurations:

1. joint estimation without projection;
2. joint estimation with the existing 2025 soft penalty;
3. joint estimation with one-step OBC.

All use `physics_parameterization='reduced'`, the existing ten combination-space detuning factors,
and no raw-parameter detuning. Create the configurations but run none of them.

### 9.10 Required behavioral tests

Produce focused tests or executable diagnostics showing:

1. the coefficient is evaluated once per objective;
2. gradients pass through the coefficient to ANN weights;
3. detaching the coefficient changes the ANN gradient;
4. stacked orthogonality reaches the measured numerical floor;
5. orthogonality still holds after at least one optimizer update;
6. additional-state outputs are bit-identical with correction on and off;
7. the disabled path is bit-identical to the current implementation;
8. all ten reduced physical parameters retain nonzero training gradients;
9. save and reload do not leave functorch tensor wrappers in mutable block caches;
10. at least ten consecutive objective evaluations cause no recompilation.

### 9.11 Documentation

Before nontrivial implementation, add one decision entry to `docs/decisions.md` covering the OBC
lifecycle, protected space, reference tuples, SVD construction, compile seam, and selected fallback
rung. Update
`scripts/gantry/orthogonal-by-construction/implementation/PROGRESS.md` in its existing format.
Do not edit `tasks/todo.md`.

## 10. Acceptance criterion

The task is complete when:
* the generic and gantry-specific OBC paths are implemented;
* all ten behavioral checks in section 9.10 pass with numerical results;
* the compile probe identifies the working rung and shows no per-objective recompilation;
* the active dataset qualification is reported at nominal and detuned parameters;
* the two-by-two `rho` table is produced;
* both runtime components and memory are measured on the available CUDA device;
* the three future experiment configurations exist and no training run was launched;
* disabled OBC behavior is bit-identical to the starting pipeline;
* the decision and progress entries are written.

If the dataset qualification fails, that does not fail the generic implementation. Mark the gantry
training configuration as not dataset-qualified and report the exact singular directions that block
the claim.

## 11. Read these first

1. `scripts/gantry/orthogonal-by-construction/documentation/obc-gantry-one-step-implementation-plan.tex`
   for the accepted mathematics and scope.
2. `orthogonal-IO-augm-main/orthogonalx_augm/src.py` for Gyorok's objective lifecycle.
3. `model_augmentation/fit_systems/blocks.py`, specifically `Reduced_Gantry_State_Block`, for the
   reduced parameter coordinates and normalized transition.
4. `scripts/gantry/gantry_dynamic/controller.py` and
   `model_augmentation/fit_systems/interconnect.py` for the compiled step seam.
5. `scripts/gantry/gantry_dynamic/data.py` and D-188 in `docs/decisions.md` for the exact active
   records and their augmentation-focused design intent.

## 12. Do not

* Do not ask the user to choose routine implementation details. Follow this contract.
* Do not touch files outside the paths authorized below.
* Do not use delayed inputs or outputs as model states.
* Do not include controller derivatives in `J`.
* Do not constrain the additional-state rows beyond leaving them unchanged.
* Do not claim multistep or closed-loop input-output orthogonality.
* Do not treat full rank alone as proof of practical identifiability.
* Do not switch datasets, generate trajectories, or tune excitation.
* Do not silently truncate parameter directions.
* Do not form normal equations.
* Do not freeze `theta_aux` across optimizer updates.
* Do not recompute `theta_aux` per rollout timestep.
* Do not disable compilation without first recording the failed exact probe.

Authorized paths:

* create `model_augmentation/fit_systems/obc.py`;
* create narrowly scoped OBC modules under `scripts/gantry/gantry_dynamic/`;
* create probes and tests under
  `scripts/gantry/orthogonal-by-construction/implementation/`;
* modify the minimum compile-seam files in `model_augmentation/fit_systems/` and
  `scripts/gantry/gantry_dynamic/`;
* modify `scripts/gantry/gantry_dynamic/config.py` only for explicit OBC fields;
* append the required entry to `docs/decisions.md`;
* update `scripts/gantry/orthogonal-by-construction/implementation/PROGRESS.md`.

## 13. Operational

Use the `GraduationProject` conda environment. Follow the repository live-output convention for any
probe lasting more than a few seconds. Store generated diagnostic results under
`scripts/gantry/orthogonal-by-construction/implementation/results/` with a dated subdirectory.

Compare performance with the disabled compiled CUDA step in the same process and configuration.

No cluster submission and no training launch are part of this task.

## 14. Delegation

One Explore subagent is allowed only for tracing the objective and compile call graph across
`model_augmentation/fit_systems/` and `scripts/gantry/gantry_dynamic/`. Do not delegate the
mathematical construction, implementation, or final verification. Do not spawn additional agents.

Work autonomously to completion. If one compile rung fails, proceed through the pinned fallback
ladder and finish every independent deliverable. End with a concise report listing changed files,
the selected compile rung, orthogonality and gradient numbers, dataset-conditioning results, runtime
and memory, and anything that remains blocked.
