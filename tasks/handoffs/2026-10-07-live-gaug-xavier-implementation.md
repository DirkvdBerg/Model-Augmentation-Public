# Handoff: implement the paper-style live `g_aug` ResNet and Xavier added-state encoder

**From:** session of 2026-10-07  
**Branch:** `Augmentation`  
**Status:** design agreed; implementation has not started  
**Primary source:** `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`

## 1. Objective

Implement the initialization described by Hoekstra et al. for the augmentation learning function and augmented-state
encoder, specialized to this repository's S-DP model:

1. replace the present plain augmentation MLP by the paper's ResNet form
   `phi_aug(z) = NN_NL(z) + W_a z`;
2. keep the nonlinear branch zero at initialization;
3. keep the physical correction `f_aug` zero at initialization, while making the added-state dynamics `g_aug` live;
4. initialize the added-state encoder map `W_psi^a` with Xavier;
5. use the new low-pass-noise dataset at
   `Thesis-writeup/Data/Coulomb-tanh-and-MSD-lowpass-noise`.

Jan Hoekstra's latest feedback, reported by the user, is that `g_aug` should not be initialized to zero. The design below
implements that feedback through the paper's linear bypass, not by randomizing rows of the existing nonlinear MLP.

## 2. What the paper says exactly

The implementation and thesis must distinguish direct statements from the paper from deductions and repository choices.

### 2.1 Learning-function architecture — direct paper statement

Section 3.3, Equation (15), printed page 6, extends the learning function to a ResNet:

```text
phi_aug(theta_aug, z_a,k)
    = theta_w,q+1 xi_q(k) + theta_b,q+1 + W_a z_a,k,
xi_0(k) = z_a,k.
```

Thus the paper's learning function has two additive paths:

```text
phi_aug(z) = NN_NL(z) + W_a z
```

`W_a` here is the residual linear weight matrix of the **augmentation learning function**. It is not the encoder matrix
`W_psi^a`.

### 2.2 Required behavior at initialization — direct paper statement

Section 5.4.1, Equation (29), printed page 9, requires the initialized LFR model with encoder to behave equivalently to
the baseline model. The construction is stated under:

1. an acyclic LFR computational graph; and
2. learning functions parameterized as the ResNets of Equation (15).

"Equivalent to the baseline" concerns the represented baseline input-output/physical-state behavior; it does not require
the added hidden state to remain zero.

### 2.3 Encoder initialization — direct paper statement

Section 5.4.2, Equation (31), printed page 9, extends the baseline encoder with an augmented-state encoder `psi_a` and
states:

> the weights and biases of `psi_aug` are initialised by the Xavier approach.

The paper does **not** specify Xavier-uniform versus Xavier-normal, a gain, an RNG stream, or how to split a single map
across separately stored input-history and output-history matrices.

### 2.4 Learning-function initialization — direct paper statement

Section 5.4.3, printed pages 9–10, says to:

1. initialize the nonlinear component to zero;
2. initially consider only the linear component, `phi_aug(z_a,k) = 0 + W_a z_a,k`;
3. initialize all LFR matrices not required to impose baseline behavior elementwise as `m ~ U(-1,1)`.

Consequences:

- Xavier is stated for the augmented-state **encoder**, not for the learning-function bypass `W_a`;
- the paper's random rule for an unconstrained part of `W_a` is `U(-1,1)`, not Xavier;
- Equation (15) has no bias on the bypass: it is exactly `W_a z`.

### 2.5 S-DP structure — direct paper statement plus explicit deduction

Appendix A, Equation (A.2), printed page 16, realizes dynamic parallel state augmentation as

```text
[x_b,k+1]   [f_base(x_b,k, u_k)]
[x_a,k+1] = [0                 ] + phi_aug(x_b,k, x_a,k, u_k).
```

Partition the learning output and bypass rows as

```text
phi_aug(z) = [f_aug(z)] = [NN_f(z)] + [W_a,f] z
             [g_aug(z)]   [NN_g(z)]   [W_a,g]
```

The following is our direct S-DP deduction from Equations (29) and (A.2), not a sentence quoted from the paper:

- baseline physical behavior requires `f_aug^(0) = 0`;
- `g_aug^(0)` may be nonzero because it evolves only the added hidden state at initialization;
- therefore set the physical bypass rows `W_a,f = 0` and leave the added-state bypass rows `W_a,g` live.

This is also consistent with Jan's newer instruction not to zero `g_aug`.

### 2.6 Paper/code caveat that must remain documented

Section 6.3 says that the paper's parallel augmentations use feedforward networks and its series augmentations use
ResNets. Jan's public S-DP example also uses `zero_init_feed_forward_nn`, whose complete final layer is zero. Therefore,
the public S-DP example does not visibly implement the live linear bypass described by the general Section 5.4 method.

This task follows the **general paper method in Sections 3.3 and 5.4 plus Jan's latest feedback**. Do not claim that it
reproduces the public example code byte-for-byte.

## 3. Exact repository design

### 3.1 Augmentation network

Current code (`scripts/gantry/gantry_dynamic/model.py`) constructs one `Static_ANN_Block` using
`zero_init_feed_forward_nn`. Its input is

```text
z = [x_b, x_a, u]
```

and its routed output contains the physical-state and added-state updates. Its final layer is completely zero, so the
current initialization has both `f_aug = 0` and `g_aug = 0`. There is currently no `W_a z` path.

Implement a network with

```text
phi_aug(z) = nonlinear_mlp(z) + linear_bypass(z)
```

where:

- `nonlinear_mlp` retains the current depth, width, activation and hidden-layer initialization;
- its complete final weight and bias remain exactly zero at initialization;
- `linear_bypass` is a trainable weight matrix without bias;
- bypass rows are classified using `cfg.ann_route_ix`, never by assuming that the final two output rows are added-state
  rows;
- a routed output whose state index is `< cfg.nx_phys` is an `f_aug` row and is initialized exactly zero;
- a routed output whose state index is `>= cfg.nx_phys` is a `g_aug` row and is initialized elementwise from `U(-1,1)`;
- both bypass partitions remain trainable after initialization.

At initialization this gives

```text
f_aug(z) = 0
g_aug(z) = W_a,g z
```

Do not initialize the MLP's `g_aug` final rows with Xavier. That was the superseded design and is not Equation (15) or
Section 5.4.3.

### 3.2 Encoder

The repository uses `linear_encoder_init_aug` rather than the paper's fully ANN-based encoder. It stores:

- analytical baseline-state maps `Wb_psi_y` and `Wb_psi_u`;
- random added-state maps `Wa_psi_y` and `Wa_psi_u`;
- a shared nonlinear encoder correction whose final layer is zero.

Implement the following specialization:

- keep `Wb_psi_y` and `Wb_psi_u` exactly unchanged;
- keep the nonlinear encoder correction unchanged;
- treat `[Wa_psi_y Wa_psi_u]` as one logical matrix mapping the stacked history to `x_a`;
- initialize that logical matrix with **Xavier uniform, gain 1**, then split it into the two stored parameters;
- initialize any added-state encoder bias consistently if one exists; the current linear map appears to have no such
  bias, so do not introduce one merely for this task.

The choice of Xavier **uniform with gain 1** is a repository convention needed because the paper only says "Xavier".
The thesis must word it as: Xavier is prescribed by Hoekstra et al., while the uniform variant and gain 1 are our
implementation choices.

### 3.3 RNG policy

- Use the run's ordinary deterministic seed and explicit initialization calls.
- Do not perform dummy Kaiming draws, construct dummy matrices, or otherwise imitate the old global RNG advancement.
- Do not claim the new architecture is RNG-stream-identical to historical runs.
- Same code + same seed must reproduce the new initialization and training order.

This is a new architecture/initialization experiment, so historical parameter-by-parameter bit identity is neither
possible nor scientifically required.

### 3.4 Dataset

Add a new, explicit dataset mode for:

```text
Thesis-writeup/Data/Coulomb-tanh-and-MSD-lowpass-noise
```

Recommended name: `thesis_taf_lowpass_noise`, exposed through a distinct thesis noise option such as
`THESIS_NOISE=lowpass`. Include the mode in the same anti-aliasing and record-trimming paths as the other thesis TAF
datasets where appropriate.

Do not silently repoint the existing `thesis_taf_noisy` mode. Existing runs must keep their original data provenance.

## 4. Scientific comparison and provenance

The old core runs 41–46 and PS2 runs 111–116 used a different noisy dataset and the old zero-`g_aug` architecture. They
are historical context only; they are not matched controls for the new low-pass-noise runs.

The primary matched comparison on the new dataset is:

```text
plain S-DP with paper-style initialization
versus
PS2 S-DP with the same paper-style initialization
```

Match arm, seed, data, model, routing and all other hyperparameters. Do not claim that this comparison isolates the
effect of the new initialization. Isolating that effect would require rerunning the old initialization on the new
dataset, which is not currently requested.

The previously proposed run IDs 141–146 and 151–156 may be reused only after `runs.tsv` is checked for collisions. Their
old `reference` fields must not point to 41–46 or 111–116 as if those were matched-data controls.

## 5. Implementation steps

1. Log the decision in `docs/decisions.md`, explicitly separating paper statements, the S-DP deduction, and repository
   conventions.
2. Add a reusable augmentation ResNet module implementing `NN_NL(z) + W_a z`.
3. Integrate it into `scripts/gantry/gantry_dynamic/model.py` without changing routing or the state/output equations.
4. Add explicit configuration for the paper-style initialization. Defaults must preserve the existing pipeline unless
   the new experiment flag is selected.
5. Add the Xavier option to `linear_encoder_init_aug`; its existing Kaiming behavior remains the compatibility default.
6. Add the new dataset mode and thesis environment option without changing existing modes.
7. Record architecture, bypass initialization, encoder initialization and dataset mode in checkpoint metadata and the
   run banner.
8. Update/add tests before preparing server runs.

Suggested semantic configuration values:

```text
augmentation_init = "legacy_zero_mlp" | "paper_resnet_live_g"
wa_encoder_init   = "legacy_kaiming"  | "xavier_uniform_gain1"
```

Avoid calling the combined setting merely `live_xavier`: Xavier applies to `W_psi^a`, while the live `g_aug` bypass uses
the paper's `U(-1,1)` rule.

## 6. Acceptance criteria

### 6.1 Compatibility mode

1. With legacy/default flags, model type, parameters, initial output and fixed-batch loss remain identical to the current
   pipeline.
2. Existing checkpoints load through the legacy path without new required keys.
3. Existing dataset modes resolve to their existing directories.

### 6.2 Paper-style augmentation

1. A distinct trainable bias-free bypass exists, and forward evaluation equals `nonlinear_mlp(z) + W_a z`.
2. Every nonlinear final-layer weight and bias is exactly zero at initialization.
3. Every `W_a,f` entry is exactly zero.
4. `W_a,g` entries are within `[-1,1]`, are reproducible for a fixed seed, and are not all zero.
5. At initialization, the full augmented model's physical/output rollout matches the baseline to float64 round-off even
   though the added state evolves.
6. Test row classification with non-default `ann_route_ix`, including reordered and partially routed rows.
7. Verify that both `W_a,f` and `W_a,g` receive gradients once the loss connects them; do not require every group to have
   a nonzero gradient at update zero.
8. Record the initial added-state Jacobian/eigenvalues and rollout scale as diagnostics. The paper does not guarantee
   stability of a random `U(-1,1)` bypass, so do not silently rescale it. Stop and report before server submission if the
   initialization immediately diverges.

### 6.3 Encoder

1. `Wb_psi_y` and `Wb_psi_u` are identical between legacy and Xavier modes when compared before unrelated RNG effects.
2. Concatenated `[Wa_psi_y Wa_psi_u]` obeys the Xavier-uniform gain-1 bound computed from the logical combined matrix.
3. Splitting and recombining the two stored matrices exactly recovers the initialized logical matrix.
4. Encoder initialization is reproducible for a fixed seed.

### 6.4 Data and pipeline

1. The new mode resolves only to `Coulomb-tanh-and-MSD-lowpass-noise` and loads Training, Validation and Test records.
2. Saved metadata identifies the new dataset unambiguously.
3. Plain and PS2 CPU smoke runs complete for both U and OBC arms.
4. `test_ps2_pipeline.py` passes or is extended for the new architecture.

## 7. Out of scope

- Changing PS2's algorithm, thresholds or phase definitions.
- Changing augmentation routing, activation, loss, OBC, physical parameterization or output map.
- Changing the analytical `W_psi^b` formulas.
- Initializing the bypass with Xavier; the paper only assigns Xavier to the augmented encoder.
- Dummy RNG consumption or preservation of historical global-RNG advancement.
- Silently rescaling `U(-1,1)` to obtain stability.
- Claiming the public S-DP example already implements this paper-style bypass.
- Submitting server jobs from the implementation session; the user moves files and submits.

## 8. Files to inspect first

1. `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`: Eq. (15), Sec. 5.4.1–5.4.3, Eq. (31),
   Appendix A Eq. (A.2), and Sec. 6.3.
2. `tasks/handoffs/2026-10-07-live-gaug-xavier-init.md`: earlier investigation and measured background.
3. `scripts/gantry/gantry_dynamic/model.py`: current augmentation construction and routing.
4. `model_augmentation/utils/torch_nets.py`: `zero_init_feed_forward_nn`.
5. `model_augmentation/fit_systems/pre_encoder.py`: `linear_encoder_init_aug`.
6. `scripts/gantry/gantry_dynamic/data.py` and `config.py`: dataset routing and experiment configuration.
7. `scripts/gantry/thesis-results/test_ps2_pipeline.py`: compatibility/pre-flight test patterns.
8. `Model-Augmentation-Public-main/`: comparison source only; do not modify.

## 9. Required reporting after implementation

Report separately:

1. what was taken directly from the paper;
2. what was deduced for S-DP (`f_aug^(0)=0`, live `g_aug` allowed);
3. what was chosen by this repository (Xavier-uniform gain 1, configuration names, RNG mechanics);
4. the exact initialized tensor shapes, bounds and row partitions;
5. the baseline-equivalence and reproducibility test results;
6. the new dataset manifest/counts and the final matched run table.

Do not summarize the result as merely "changed Kaiming to Xavier." The encoder Xavier change and the augmentation
ResNet/live-`g_aug` change are separate mechanisms with separate initialization rules.

## 10. Delegation

None requested. Keep the implementation localized and review the dirty worktree before every edit.
