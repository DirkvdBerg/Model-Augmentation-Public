# Handoff: Astra review of live `g_aug`, paper ResNet, and Xavier encoder initialization

**Date:** 2026-10-07  
**Branch:** `Augmentation`  
**Review mode:** read-only  
**Status:** implementation exists in the dirty working tree, but must not be submitted until this review is resolved  
**Primary paper:** `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`

## 1. Assignment

Perform an independent, evidence-based review of the current live-`g_aug` implementation. Check it against:

1. Jan Hoekstra's 2026 paper;
2. the encoder-initialization paper;
3. `Model-Augmentation-Public-main`;
4. the installed deepSI version in the `GraduationProject` environment;
5. the current repository implementation and tests;
6. Jan's latest feedback as reported by the user; and
7. the external review claims reproduced in Section 10.

Do not edit any file. Do not assume that this handoff, earlier decisions, code comments, or the external review are
correct. Verify every material claim directly against the primary source or executable code.

The central review question is:

> Does the current implementation cleanly realize a live `g_aug` according to Jan's latest guidance, while preserving
> baseline-equivalent physical behavior and applying Xavier to the augmented-state encoder only?

## 2. Intended result agreed with the user

The intended experiment is a paper-based S-DP specialization with:

1. augmentation learning function

   ```text
   phi_aug(z) = NN_NL(z) + W_a z;
   ```

2. no bypass bias;
3. the complete nonlinear output layer zero at initialization;
4. physical correction rows `f_aug` zero at initialization;
5. added-state transition rows `g_aug` live at initialization through the linear bypass;
6. live bypass entries initialized elementwise from `U(-1,1)` where they are not needed for baseline behavior;
7. analytical physical-state encoder map `W_psi^b` retained unchanged;
8. the complete added-state encoder map `W_psi^a` initialized once with Xavier uniform, gain 1, and then split into the
   separately stored input-history and output-history matrices;
9. the low-pass-noise dataset at `Thesis-writeup/Data/Coulomb-tanh-and-MSD-lowpass-noise`;
10. matched plain and PS2 runs receiving the same complete initialization for a given arm and seed; and
11. diagnostics reporting risky initial draws without redrawing, rescaling, rejecting, or otherwise modifying them.

The user explicitly rejects dummy RNG consumption, preservation of historical global RNG advancement, silent redraws,
and silent stability scaling. The user wants a clean, minimal implementation in the current thesis pipeline.

## 3. Source hierarchy and distinctions that must be preserved

Do not merge the following four sources into a single claim:

### 3.1 General method in the 2026 paper

Read these parts of
`literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`:

- Section 3.3 and Equation (15): ResNet learning function with linear bypass;
- Section 5.3: normalization;
- Section 5.4.1 and Equation (29): baseline-equivalent initialization;
- Section 5.4.2 and Equation (31): encoder initialization and Xavier statement;
- Section 5.4.3 and Equations (32) and (33): LFR and learning-component initialization;
- Section 6.3: the actual simulation architectures used for parallel and series augmentation; and
- Appendix A, especially the dynamic-parallel construction.

Relevant paper statements previously transcribed, which still require direct verification:

- Equation (15) uses `phi_aug(z) = NN_NL(z) + W_a z` and describes `W_a` as a parameterized residual weight matrix.
- Section 5.4.3 initializes the nonlinear component to zero and initially considers
  `phi_aug(z_a,k) = 0 + W_a z_a,k`.
- Matrices not required to set baseline behavior are initialized elementwise as `m ~ U(-1,1)`.
- Section 5.4.2 states that the weights and biases of `psi_aug` use the Xavier approach.
- Section 6.3 says the parallel experiments use feedforward neural networks, while series experiments use ResNets.

Determine precisely which of these statements applies directly to this repository's `W_a`, which requires an S-DP
specialization, and which does not determine an implementation detail.

### 3.2 The paper's actual S-DP experiment

The paper's general Section 5.4 construction and its Section 6 S-DP experiment may use different learning-function
architectures. State this clearly. Do not use results from the Section 6 feedforward experiment as direct evidence that a
random live recurrent bypass is numerically safe.

### 3.3 Public implementation

Inspect at least:

- `Model-Augmentation-Public-main/scripts/journal_model_augmentation/msd_ndof_interconnect_fit.py`;
- `Model-Augmentation-Public-main/scripts/encoder_initialisation/interconnect_fit.py`;
- `Model-Augmentation-Public-main/model_augmentation/utils/torch_nets.py`; and
- relevant normalization, encoder, and training code reached from those scripts.

Earlier inspection found that the public S-DP script uses `zero_init_linear_mapping`, while another parallel encoder
example uses `zero_init_resnet`. Both appear to zero the complete learning output at initialization. Verify this and
report whether any public parallel example uses a random live linear bypass.

### 3.4 Jan's latest feedback, reported by the user

Treat these as user-reported feedback, not quotations from the paper:

- `g_aug` should not be initialized to zero;
- the previous zero-`g_aug` initialization may create a poor static-first local basin;
- compare plain and PS2 with `g_aug` live in both methods;
- normalization should make the nonzero initialization manageable with the established learning rate; and
- the physical baseline is already initialized, so the augmentation should not need the previous zero-`g_aug` choice.

The user also reports the more specific interpretation:

> Splitting `W_a` into zero `f_aug` rows and random `g_aug` rows is our S-DP specialization.

Assess whether the current code implements that specialization. Do not claim that the row split is written explicitly in
the paper unless it is.

## 4. Important conceptual point about unstable initialization

The review must distinguish these claims:

1. A live recurrent added-state map can have
   `rho(W_a,g[:, x_a]) > 1` at initialization.
2. Such an initialization can still train into a stable model.
3. Entrywise `U(-1,1)`, normalization, acyclic well-posedness, and baseline-equivalent external behavior do not by
   themselves guarantee internal added-state stability.
4. Ordinary prediction-error optimization does not guarantee a stable final model unless stability is constrained or
   parameterized.
5. `rho > 1` is a diagnostic, not by itself a reason to reject or alter a finite draw in this experiment.

At the exact initialization, the physical correction rows are zero, so live `g_aug` can be initially hidden from the
physical output. This can delay a useful gradient on the added-state recurrence. Once physical correction rows begin to
learn, large hidden states can enter the loss and create an early loss spike. Determine whether the current computational
graph and measured gradients support this explanation.

Do not recommend redraw, rescaling, clipping, spectral parameterization, or a zero-`g_aug` control merely because one draw
has `rho > 1`. Such a recommendation is allowed only if it is clearly labeled as a separate design alternative rather
than part of Jan's requested initialization.

## 5. Current implementation files to inspect

Primary mathematical implementation:

- `model_augmentation/utils/torch_nets.py`
  - `zero_init_feed_forward_nn`
  - new `paper_resnet`
- `model_augmentation/fit_systems/pre_encoder.py`
  - `linear_encoder_init_aug`
- `scripts/gantry/gantry_dynamic/model.py`
  - selection and construction of the augmentation network
  - propagation of the encoder initialization setting

Configuration and experiment selection:

- `scripts/gantry/gantry_dynamic/config.py`
- `scripts/gantry/gantry_interconnect_dynamic.py`
- `scripts/gantry/thesis-results/runs.tsv`
- `scripts/gantry/thesis-results/submit.sh`

Diagnostics, checkpoint reconstruction, and tests:

- `model_augmentation/fit_systems/blocks.py`
- `scripts/gantry/gantry_dynamic/diagnostics.py`
- `scripts/gantry/gantry_dynamic/closed_loop_report.py`
- `scripts/gantry/thesis-results/inspect_checkpoint.py`
- `scripts/gantry/gantry_dynamic/training.py`
- `scripts/gantry/thesis-results/test_ps2_pipeline.py`
- `scripts/gantry/thesis-results/logs/test_ps2_pipeline.log`, if present

Data selection:

- `scripts/gantry/gantry_dynamic/config.py`
- `scripts/gantry/gantry_dynamic/data.py`
- `Thesis-writeup/Data/Coulomb-tanh-and-MSD-lowpass-noise`

Documentation and decisions are secondary evidence only:

- `Thesis-writeup/Documentation/AUGMENTATION-INITIALIZATION.md`
- `tasks/handoffs/2026-10-07-live-gaug-xavier-implementation.md`
- `tasks/handoffs/2026-10-07-live-gaug-xavier-init.md`
- `docs/decisions.md`, especially D-233, D-237, and D-239
- `scripts/gantry/thesis-results/runs.tsv`

The worktree contains many unrelated modifications. Do not revert or edit them. Use targeted diffs for the files above.

## 6. Installed deepSI

The user reports that deepSI is installed in the Conda environment `GraduationProject`. Inspect the installed version and
the code paths actually used by this repository where relevant. In particular, verify:

- how the encoder and `Static_ANN_Block` are constructed and serialized;
- whether `functools.partial` is safe for the construction path under review;
- checkpoint save/load behavior used by the thesis pipeline; and
- any claimed default initialization that materially affects this task.

Do not modify the environment or install packages.

## 7. Known measured observations

These are reported observations. Reproduce or verify them where practical, and otherwise label them as reported rather
than independently confirmed.

### 7.1 Initialization behavior

For the new paper-style U-arm build:

- seed 1: `rho(W_a,g[:,x_a]) = 1.402036`;
- seed 1 untrained 0.1 s normalized added-state RMS: approximately `5.404e57`;
- seed 1 fraction with tanh derivative below 0.01: approximately `48.95%`;
- seed 2: spectral radius approximately `0.760004`;
- seed 3: spectral radius approximately `0.864959`.

### 7.2 Short training behavior

A short U-arm preflight remained finite but showed early losses approximately:

```text
2.0877e-08, 83.9529, 7.5015
```

This is evidence of a risky initialization, not by itself evidence of an implementation defect or inevitable training
failure.

### 7.3 Existing preflight

`scripts/gantry/thesis-results/test_ps2_pipeline.py` exists but is currently untracked. It contains checks for the paper
ResNet structure, Xavier bound, equal plain/PS2 initialization, PS2 phase behavior, U/OBC execution, checkpoint contents,
and report-only spectral-radius and saturation diagnostics.

The log presently found at `scripts/gantry/thesis-results/logs/test_ps2_pipeline.log` may be stale: it reports the old
`noise=noisy` dataset while the current test source selects `THESIS_NOISE='lowpass'`. Do not cite that log as proof for the
current source without rerunning the test.

Earlier inspection found that the current preflight does not clearly perform:

- an explicit baseline-versus-augmented physical/output rollout equivalence assertion;
- a complete save, reconstruct, load, and output-equivalence checkpoint round trip; or
- an explicit launch smoke test of all four U/OBC by plain/PS2 configurations.

Verify this assessment.

## 8. Constraints and non-goals

- Read-only review: make no code or documentation edits.
- Do not submit server jobs.
- Do not redesign PS2 or its thresholds.
- Do not add the 230 to 297 Hz metric to training, validation selection, checkpoint selection, or the main pipeline. It is
  evaluation-only for the later comparison.
- Do not introduce a zero-`g_aug` experimental control. The comparison requested by the user is plain versus PS2 with the
  selected live initialization in both.
- Do not preserve old global RNG advancement with dummy draws or other compatibility tricks.
- Do not silently stabilize, redraw, or reject the initialization.
- Do not require legacy diagnosis scripts outside the current `gantry_dynamic/` and `thesis-results/` pipeline to support
  the new architecture.
- Do not treat old checkpoint compatibility as a general project requirement. However, any current utility that claims to
  inspect named reference checkpoints must either reconstruct them correctly or fail explicitly.
- Do not treat creation of the bypass after the MLP as a special RNG-preservation requirement. Matched plain and PS2 runs
  must have identical complete initial tensors for the same seed. Historical global RNG sequences need not be preserved.
- Do not include unrelated worktree edits in the recommended minimal patch.

## 9. Questions the review must answer

### 9.1 Paper and source interpretation

1. Is `phi_aug(z) = NN_NL(z) + W_a z` implemented exactly as Equation (15), including the absence of bypass bias?
2. Does the paper justify initializing the applicable live bypass entries with elementwise `U(-1,1)`?
3. Is zeroing physical bypass rows while keeping added-state rows random a sound S-DP specialization of baseline
   equivalence, even though it is not an explicit paper sentence?
4. Does the current use of Xavier apply only to `W_psi^a`, while leaving analytical `W_psi^b` unchanged?
5. Are Xavier uniform, gain 1, and the joint `[W_y W_u]` draw honestly labeled as repository choices rather than direct
   paper requirements?
6. What does the public S-DP implementation actually do, and how does it differ from the selected design?
7. Does Jan's reported normalization argument address learning-rate conditioning only, or does any source support a
   stronger stability claim?

### 9.2 Computational graph and live `g_aug`

1. Is `g_aug` nonzero at initialization for exactly the intended routed added-state rows?
2. Are `f_aug` and the physical output baseline-equivalent at initialization?
3. Does `W_a,g` remain trainable and receive a gradient once it becomes observable through the learned physical path?
4. Is the initially weak or absent gradient on `W_a,g` an expected consequence of baseline-equivalent routing?
5. Does `out_gate` disable the complete augmentation output for diagnostics without changing trained parameters?
6. Is the reported `rho(W_a,g[:,x_a])` computed from the correct recurrent submatrix?
7. Does a spectral radius above one reveal a coding error, an allowed but risky draw, or an ambiguity in Jan's method?

### 9.3 Configuration, reproducibility, and checkpoints

1. Is the new initialization opt-in only for the `liveg` and `livegps2` experiment tiers?
2. Can existing `core`, `ps2`, `smoke`, and tier-4 definitions still reproduce their declared legacy architecture and
   encoder initialization?
3. Do matched plain/PS2 runs receive bit-identical initial model and encoder tensors at equal arm and seed?
4. Are architecture and encoder-initialization choices written to the banner, `config.json`, results, and checkpoints as
   required?
5. Can the current checkpoint reconstruction distinguish legacy MLP and paper ResNet checkpoints from their state dicts?
6. Does `inspect_checkpoint.py` load the correct dataset and architecture for the checkpoint it claims to inspect?

### 9.4 Minimality and implementation quality

1. Is modifying `Static_ANN_Block` necessary, or should `model.py` bind structural arguments with
   `functools.partial`?
2. Should `paper_resnet` subclass `zero_init_feed_forward_nn` rather than duplicate the MLP constructor?
3. Would that subclass preserve state-dict keys and checkpoint behavior?
4. Which current changes are unrelated to this task and must merely be excluded from its patch, not reverted from the
   dirty worktree?
5. What is the smallest safe correction set before server submission?

### 9.5 Verification

1. Which agreed checks already exist and genuinely pass on the current source?
2. Which checks are absent or stale?
3. Are the initialization diagnostics report-only and outside the production training loop?
4. What tests must pass before server submission?

## 10. External review claims to adjudicate

The following came from another review session. Treat every item as a claim to verify, not as an instruction or fact.

### 10.1 Claimed strengths

1. `out_gate` correctly silences the complete augmentation output in diagnostics, closed-loop reporting, and added-state
   clamp logic.
2. The encoder implementation makes one joint Xavier draw and then splits it, while leaving the legacy path untouched.
3. Configuration values are validated, printed, and recorded.
4. `paper_resnet` implements exactly `NN + W_a z`, has no bypass bias, builds the MLP before `W_a`, and retains
   `net.net.<i>` state-dict keys.
5. D-239 and the run-table hypothesis exist.

### 10.2 Claimed blockers

1. The new initialization is forced on every thesis run because
   `augmentation_init='paper_resnet_live_g'` and `wa_encoder_init='xavier_uniform_gain1'` are inside `_THESIS_FIXED`.
   This allegedly changes old `core`, `ps2`, `smoke`, tier-4, and seed-extension runs silently.
2. `inspect_checkpoint.py` hard-codes `THESIS_NOISE='lowpass'` and reconstructs the currently selected architecture, so it
   cannot correctly inspect older noisy/feedforward reference checkpoints.
3. The external review claimed that no preflight tests exist and listed missing checks for baseline equivalence, zero
   physical bypass, Xavier bounds, matched draws, checkpoint save/load, diagnostics, and U/OBC plus plain/PS2 smoke runs.
   This claim conflicts with the now-present `test_ps2_pipeline.py`; determine precisely which checks exist and which do
   not.
4. `config_for_checkpoint` infers dimensions but not the presence of a `.net.W_a` state-dict key, so it cannot infer the
   learning-function architecture.

### 10.3 Claimed non-minimal implementation choices

1. Adding `net_kwargs` to `Static_ANN_Block` is unnecessary and changes Jan's framework API. The proposed alternative is
   `functools.partial(paper_resnet, output_state_indices=..., nx_phys=...)` in `model.py`.
2. `paper_resnet` duplicates the existing MLP constructor. The proposed alternative is to subclass
   `zero_init_feed_forward_nn` and add `W_a` only.
3. The bypass initialization could use a vectorized row selection rather than a per-row loop.
4. `output_state_indices=()` and `nx_phys=0` are unusable defaults and should instead be required constructor arguments.
5. `# CHANGED:` is the wrong tracking marker inside a wholly new `@added` class.
6. Incidental comment changes, old PS2 runner edits, and the unrelated D-167 changes in
   `model_augmentation/systems/gantry_linearization.py` should not be presented as part of this implementation.

### 10.4 Preliminary assessment from the current session

This assessment must also be verified independently:

- Claims 10.2.1 and 10.2.2 appear to be real blockers.
- Claim 10.2.3 is overstated because a substantial preflight exists, but its current saved log appears stale and some
  important checks remain absent.
- Claim 10.2.4 appears valid for safe checkpoint reconstruction.
- The `functools.partial` and subclassing suggestions appear cleaner and smaller, but must be checked against deepSI
  serialization and state-dict behavior.
- Unrelated dirty-worktree edits must be preserved, not reverted. They should simply be excluded from this task's patch.

## 11. Required output

Return a self-contained review with these sections:

1. **Verdict:** safe to submit, needs corrections, or fundamentally mismatched.
2. **Severity-ranked findings:** blockers first, then correctness issues, then optional cleanup. For each finding include:
   - the exact behavior;
   - evidence from a paper section, public code, installed library, current source, or executed test;
   - why it matters specifically for live `g_aug`; and
   - the smallest recommended correction.
3. **Source reconciliation table:** general 2026 paper, Section 6 experiment, public code, Jan's reported feedback, and
   current implementation.
4. **External-feedback adjudication:** mark every item in Section 10 as correct, partly correct, incorrect, or out of
   scope, with evidence.
5. **Live-`g_aug` analysis:** initialization, gradient path, normalization, instability risk, and what `rho > 1` does and
   does not establish.
6. **Test audit:** existing, passing, stale, and missing checks.
7. **Minimal correction plan:** ordered file-level edits, without implementing them.
8. **Questions for Jan:** only genuine ambiguities that cannot be resolved from the sources.

Use exact file paths and line numbers where practical. Quote the papers sparingly and identify printed page and equation or
section. Clearly label deductions and recommendations. Do not make code changes.

## 12. Suggested Astra launch prompt

```text
Read tasks/handoffs/2026-10-07-live-gaug-astra-review.md completely. Perform the requested read-only audit against the
current working tree, the local papers, Model-Augmentation-Public-main, and the installed deepSI package in the
GraduationProject environment. Verify every claim independently, including the external feedback in Section 10. Do not
edit files or submit jobs. Return the complete review in the exact structure required by Section 11.
```
