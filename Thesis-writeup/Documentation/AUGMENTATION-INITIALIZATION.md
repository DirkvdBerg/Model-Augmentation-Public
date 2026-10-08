# Augmentation and encoder initialization

Decision date: 2026-10-07. Implementation status: specified, not yet implemented.

This document fixes the initialization selected for the thesis experiments. The target is the general ResNet initialization method in Hoekstra et al. (2026), specialized to the structured dynamic-parallel state augmentation used for the gantry. It is not a reproduction of the paper's Section 6 S-DP experiment.

Primary source: `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`, Equation (15), Sections 5.4.1 to 5.4.3, Equation (31), and Appendix A Equation (A.2).

Encoder source: `literature/augmentation/Encoder initialisation methods in the model augmentation setting.pdf`, arXiv:2602.13108, Equation (8) for the linear and nonlinear encoder decomposition and Equations (16) and (17) for the analytical physical-state map.

## 1. Selected learning function

Use the ResNet from Equation (15), printed page 7:

\[
\phi_{\mathrm{aug}}(\theta_{\mathrm{aug}},z_{a,k})
=
\theta_{w,q+1}\xi_q(k)+\theta_{b,q+1}+W_a z_{a,k},
\qquad
\xi_0(k)=z_{a,k}.
\]

For this project, write it as

\[
\phi_{\mathrm{aug}}(z)
=
\mathrm{NN}_{\mathrm{NL}}(z)+W_a z.
\]

The first-principles model is not Equation (15). It remains the separate baseline function \(f_{\mathrm{base}}\). Equation (15) only parameterizes the learned augmentation.

The linear bypass is selected because the missing mass-spring-damper mode contains important linear dynamics. The paper motivates Equation (15) generally as a way to represent unknown linear dynamics directly instead of forcing nonlinear activation functions to approximate them. It does not claim that every S-DP model requires a ResNet.

The paper states in Section 5.4.1 that the ResNet condition is required to feasibly create series augmentations with baseline behavior. Its Section 6 parallel experiments use ordinary feedforward networks. Applying Equation (15) to this S-DP model is therefore a deliberate use of the paper's general architecture, motivated by the missing linear mode, rather than a requirement imposed by the S-DP topology.

## 2. Structured S-DP specialization

The selected model is the dynamic-parallel state augmentation from Appendix A:

\[
\begin{bmatrix}
x^b_{k+1}\\
x^a_{k+1}
\end{bmatrix}
=
\begin{bmatrix}
f_{\mathrm{base}}(x^b_k,u_k)\\
0
\end{bmatrix}
+
\begin{bmatrix}
f_{\mathrm{aug}}(z_k)\\
g_{\mathrm{aug}}(z_k)
\end{bmatrix}.
\]

Accordingly, partition the ResNet output and bypass as

\[
\phi_{\mathrm{aug}}(z)
=
\begin{bmatrix}
\mathrm{NN}_f(z)\\
\mathrm{NN}_g(z)
\end{bmatrix}
+
\begin{bmatrix}
W_{a,f}\\
W_{a,g}
\end{bmatrix}z.
\]

Implement this structured S-DP equation directly. This is an exact structured realization of the paper's LFR, not a routing workaround and not a separately invented topology.

## 3. Initialization of the augmentation

### 3.1 Direct statements from the paper

Section 5.4.1 requires baseline-equivalent behavior at initialization. Section 5.4.3 initializes the nonlinear component to zero, leaving

\[
\phi_{\mathrm{aug}}^{(0)}(z)=0+W_a z.
\]

Section 5.4.3 also initializes matrices not required to impose baseline behavior elementwise from \(\mathcal U(-1,1)\).

### 3.2 S-DP specialization and Jan's feedback

Use

\[
W_{a,f}^{(0)}=0,
\qquad
W_{a,g}^{(0)}\sim\mathcal U(-1,1).
\]

Therefore,

\[
f_{\mathrm{aug}}^{(0)}(z)=0,
\qquad
g_{\mathrm{aug}}^{(0)}(z)=W_{a,g}^{(0)}z.
\]

This row partition is our S-DP deduction, not a sentence stated explicitly in the paper:

* the physical-state correction must initially be zero to preserve the baseline physical trajectory;
* the added hidden state may evolve without changing the initial baseline input-output behavior;
* keeping \(g_{\mathrm{aug}}\) live follows Jan Hoekstra's latest feedback reported on 2026-10-07.

Applying the paper's \(\mathcal U(-1,1)\) rule specifically to \(W_{a,g}\) is also our S-DP interpretation of its general rule for matrices not required for baseline behavior.

The bypass has no bias because Equation (15) contains \(W_a z\), not an affine bypass.

All weights and biases in the final layer of \(\mathrm{NN}_{\mathrm{NL}}\) are initialized exactly to zero. The complete nonlinear branch therefore contributes zero initially. Both bypass partitions remain trainable after initialization.

## 4. Encoder initialization

The project retains its linear and analytical encoder structure. Partition the shared nonlinear correction as \(\widetilde\psi=[\widetilde\psi_b^\top\;\widetilde\psi_a^\top]^\top\):

\[
x_b
=
W_{\psi,y}^b y_{\mathrm{hist}}
+W_{\psi,u}^b u_{\mathrm{hist}}
+\widetilde\psi_b(y_{\mathrm{hist}},u_{\mathrm{hist}}),
\]

\[
x_a
=
W_{\psi,y}^a y_{\mathrm{hist}}
+W_{\psi,u}^a u_{\mathrm{hist}}
+\widetilde\psi_a(y_{\mathrm{hist}},u_{\mathrm{hist}}).
\]

Initialization is:

1. Keep \(W_{\psi,y}^b\) and \(W_{\psi,u}^b\) at the existing analytical reconstructability initialization. This uses the known linearized first-principles model to initialize the physical state directly.
2. Treat \([W_{\psi,y}^a\;W_{\psi,u}^a]\) as one logical added-state history map.
3. Initialize this logical \(W_\psi^a\) matrix with Xavier uniform, gain 1, and then split it into the two stored parameters.
4. Keep the existing zero-output nonlinear encoder correction unchanged.
5. Do not introduce a separate bias into the linear map. The selected linear encoder equations do not contain one.

The source and implementation choice must be distinguished:

* Equation (31) of the 2026 paper states that the augmented encoder uses Xavier initialization.
* The paper does not specify Xavier uniform versus normal, the gain, or the treatment of separately stored history matrices.
* Xavier uniform with gain 1 and initialization of the combined logical matrix are project choices.
* Xavier initialization is defined for weight matrices, not one-dimensional bias vectors. The selected linear map has no bias, so the paper's wording about encoder biases cannot be transferred literally to this component.

## 5. Explicit differences from the 2026 paper

### 5.1 Baseline encoder

The paper fits a general baseline encoder \(\psi_b\) against simulated baseline states using Equation (30). This project instead uses the analytical reconstructability initialization for \(W_\psi^b\).

Reason: the linearized first-principles model is known, so the physical state can be initialized directly without a separate encoder-pretraining optimization. This also preserves the intended physical meaning of the baseline-state coordinates.

### 5.2 Augmented encoder form

The paper presents a general ANN augmented encoder with weights and biases. This project applies its Xavier rule to the selected linear added-state map and retains the existing zero-output nonlinear correction.

Reason: the thesis already uses the linear encoder formulation. Adding an affine bias or replacing it with a separate ANN would change the selected encoder model rather than merely correct its initialization.

### 5.3 General LFR versus structured S-DP realization

The general derivation expresses and initializes a parameterized LFR interconnection. This project implements the paper's structured S-DP equations from Appendix A directly.

Reason: the intended augmentation topology is known. The direct S-DP equations and their corresponding structured LFR realization describe the same interconnection, so materializing a larger general LFR matrix is unnecessary.

### 5.4 Paper experiment architecture

Section 6 uses ordinary feedforward networks for its parallel augmentation experiments and ResNets for its series augmentations. This project instead applies the paper's general Equation (15) ResNet to S-DP.

Reason: the objective is to use the latest paper's general initialization method, not reproduce the particular Section 6 S-DP benchmark. Equation (15) is motivated generally by unknown linear dynamics, and the missing gantry dynamics include a linear mass-spring-damper mode. This choice also provides a direct way to keep the nonlinear component zero while leaving the added-state dynamics live.

The paper states that the ResNet condition is needed for feasible baseline initialization of series augmentations, not that it is required for S-DP. Jan's public S-DP example uses a purely linear learning map, which is relevant precedent for a direct linear path but is not the same two-path ResNet selected here.

### 5.5 Project-specific training and data

The gantry dataset, PS2 opening phase, and OBC and U comparison arms are not part of the paper's experiment.

Reason: they address the thesis research questions. They must be reported as project-specific additions, not attributed to Hoekstra et al. (2026).

## 6. Normalization

Retain the established pipeline normalization: training-record means and standard deviations, normalized first-principles state-space matrices, and the encoder's coordinate-convention correction. This keeps the baseline states, histories, and learning inputs in compatible numerical ranges.

Normalization does not determine whether the augmentation is a feedforward network or a ResNet. It supports the use of a common learning rate by preventing state and input scales from dominating the gradients.

## 7. Safety diagnostic

For the autonomous added-state part of the initial linear bypass, report

\[
\rho\!\left(W_{a,g}[:,x_a]\right),
\]

where \(\rho\) is the spectral radius and \(W_{a,g}[:,x_a]\) is the submatrix mapping \(x_a(k)\) to \(x_a(k+1)\).

This is a project safety diagnostic, not a requirement stated by the paper. The paper's baseline-equivalent initialization does not guarantee internal added-state stability. Acyclicity guarantees well-posed evaluation, and normalization controls numerical scale, but neither guarantees that this spectral radius is below one.

Also report, over one untrained 0.1 s training window:

\[
\frac{\operatorname{RMS}(x_a)}{\operatorname{RMS}(x_b)}
\]

and the fraction of first-layer tanh preactivations with magnitude above 3. At this magnitude, the tanh derivative is below approximately 0.01, so the fraction is an interpretable saturation diagnostic.

These diagnostics are implemented in a preflight or comparison utility, not in the production training loop. They do not change Jan's prescribed initialization. Do not reject a finite seed, redraw it, or rescale its matrices solely because its spectral radius exceeds one. Report the initialized value and the observed hidden-state behavior. If a run becomes nonfinite, record it as a failure of that initialized run rather than substituting another seed silently.

## 8. Implementation requirements

Sections 9 to 15 contain the complete implementation and verification contract. They are authoritative for module placement, configuration, encoder initialization, checkpoint behavior, data selection, and testing. This section intentionally does not repeat those requirements.

## 9. Current implementation and required change

### 9.1 Current augmentation

`scripts/gantry/gantry_dynamic/model.py` currently constructs one `Static_ANN_Block` using `zero_init_feed_forward_nn`. Its input is the concatenated vector

\[
z=[x_b,x_a,u],
\]

and its output is expanded into the state-update rows selected by `ann_route_ix`. The current network has no residual bypass. Its complete final layer is zero, which gives

\[
f_{\mathrm{aug}}^{(0)}=0,
\qquad
g_{\mathrm{aug}}^{(0)}=0.
\]

### 9.2 Target augmentation

Add a reusable network module with the same constructor and forward interface as the existing feedforward network:

```text
paper_resnet(z) = nonlinear_mlp(z) + linear_bypass(z)
```

The module must contain:

* the existing MLP architecture and activation;
* a trainable bias-free linear bypass with input dimension `nxd + nu` and output dimension `len(ann_route_ix)`;
* initialization metadata or buffers sufficient to identify the physical and added-state output rows;
* no separate interconnect block for the bypass.

The bypass must be a child module or parameter of the network held by the existing `Static_ANN_Block`. This preserves the following behavior:

* `ann.parameters()` includes both paths, so PS2 sees the complete learning function;
* `ann(z)` returns their sum, so OBC acts on the complete physical correction;
* the interconnection still contains one augmentation block;
* checkpoint traversal can locate all trainable augmentation parameters in the same block.

The legacy feedforward network remains available for historical checkpoint compatibility. The paper-style experiment explicitly selects the new ResNet.

## 10. Configuration contract

Use separate semantic settings for the two independent initialization decisions:

```text
augmentation_init = "legacy_zero_mlp" | "paper_resnet_live_g"
wa_encoder_init   = "legacy_kaiming"  | "xavier_uniform_gain1"
```

Requirements:

1. Existing defaults retain the historical model unless the paper-style experiment is selected explicitly.
2. The thesis runner selects both `paper_resnet_live_g` and `xavier_uniform_gain1` for the new experiment.
3. Do not name the combined setting only `live_xavier`. Xavier applies to the encoder, whereas the live bypass follows a different initialization rule.
4. Record both settings separately in the run banner, checkpoint metadata, saved result metadata, and run table.
5. Validate invalid combinations and values at configuration construction rather than failing during training.

The exact environment-variable spelling may follow the existing thesis runner convention, but it must map unambiguously to these two semantic settings.

## 11. Detailed encoder implementation

Modify `linear_encoder_init_aug` through a new initialization argument whose compatibility default retains the current behavior.

For `xavier_uniform_gain1`:

1. Determine the actual stored shapes of `Wa_psi_y` and `Wa_psi_u` at model construction.
2. Allocate one logical matrix with

   \[
   \mathrm{fan\_in}
   =\mathrm{cols}(W_{\psi,y}^a)+\mathrm{cols}(W_{\psi,u}^a),
   \qquad
   \mathrm{fan\_out}=n_{x_a}.
   \]

3. Apply Xavier uniform with gain 1 once to that complete matrix.
4. Split its columns into `Wa_psi_y` and `Wa_psi_u` without drawing either block independently.
5. Leave `Wb_psi_y`, `Wb_psi_u`, normalization offsets, and the nonlinear correction network unchanged.

The resulting Xavier bound is

\[
a_\psi=\sqrt{\frac{6}{\mathrm{fan\_in}+\mathrm{fan\_out}}}.
\]

This formula and its application to the combined logical matrix are implementation choices that must be labeled as such in code and reporting.

## 12. Interaction with PS2, OBC, and diagnostics

### 12.1 PS2

PS2 stage S2 obtains `ann.parameters()` from the connected augmentation block. Because the bypass is registered inside that block, S2 optimizes it together with the nonlinear branch.

The implementation must test, rather than assume, which parameter groups receive gradients in each PS2 phase. Do not change the PS2 loss, thresholds, phase definitions, or handover logic as part of this task.

### 12.2 OBC

OBC evaluates the connected augmentation block and projects or corrects its routed physical output. Since `ann(z)` returns the sum of both ResNet paths, OBC must see

\[
f_{\mathrm{aug}}(z)=\mathrm{NN}_f(z)+W_{a,f}z.
\]

The added-state output \(g_{\mathrm{aug}}\) remains outside the physical parameter-sensitivity projection, as intended.

The current OBC reference field is defined on \(x_a=0\). OBC therefore orthogonalizes the physical addition on that designated reference manifold. It does not orthogonalize \(x_a\) or \(g_{\mathrm{aug}}\), and it does not provide an exact off-manifold guarantee for physical corrections that depend on a nonzero \(x_a\). In particular, the term \(W_{a,f}^{x_a}x_a\) vanishes on the reference manifold and is not visible there. This is the declared scope of the existing OBC construction, not a reason to change OBC in this task.

The new trainable physical bypass creates a direct linear route that could imitate baseline mass, damping, or stiffness changes in the U arm. Report physical-parameter drift and compare U with OBC. This negation risk already exists for a flexible nonlinear augmentation, but the bypass makes the linear route explicit.

### 12.3 Existing utilities

Several utilities currently assume that `ann.net.net` is one `nn.Sequential` and that its last `nn.Linear` is the complete augmentation output layer. Review and update every such assumption, including:

* checkpoint architecture inference;
* ANN-zeroed and clamp diagnostics;
* closed-loop reports;
* ReZero experiment hooks;
* Lipschitz experiment hooks;
* parameter-group and gradient diagnostics.

Silencing the augmentation for a diagnostic must silence both the nonlinear and bypass paths. Inspecting or zeroing only the MLP final layer would leave the new bypass active and produce an invalid comparison.

## 13. Checkpoint and RNG behavior

### 13.1 Checkpoints

The selected architecture is part of the checkpoint identity.

* Historical checkpoints must reconstruct the legacy feedforward architecture.
* New checkpoints must reconstruct the ResNet and include its bypass parameters.
* Loading must use saved architecture metadata rather than the currently checked-out default configuration.
* Missing new keys must not be silently ignored when a checkpoint declares the ResNet architecture.
* Diagnostics that deep-copy or rebuild a model must preserve the selected architecture.

### 13.2 Random-number generation

Use the run's normal deterministic seed and explicit initialization calls. Do not consume dummy Kaiming draws, create dummy matrices, or attempt to reproduce historical global RNG advancement.

Required reproducibility claim:

> The new implementation is reproducible for a fixed code version, configuration, and seed.

Do not claim parameter-by-parameter or minibatch-order identity with historical architectures unless it is measured independently.

For every matched plain and PS2 pair, use the same model seed and the same model-building configuration. PS2 must not alter model construction before its training phase begins. A preflight test must build both variants and verify exact equality of their initial `W_a,g` tensors. This is a one-time comparison test, not additional production metadata.

## 14. Data and experiment definition

Add a distinct data mode for

```text
Thesis-writeup/Data/Coulomb-tanh-and-MSD-lowpass-noise
```

Recommended internal name:

```text
thesis_taf_lowpass_noise
```

Requirements:

* do not repoint or rename the existing noisy-data mode;
* include the new mode in the appropriate anti-aliasing and record-trimming logic;
* verify the Training, Validation, and Test file counts against the dataset manifest;
* save the resolved data mode and source directory in result metadata;
* keep data selection independent of the two initialization settings.

The primary new-data comparison is:

\[
\text{plain S-DP with the paper-style initialization}
\quad\text{versus}\quad
\text{PS2 with the same initialization}.
\]

Match the arm, seed, dataset, model dimensions, normalization, routing, optimizer, and training budget. Historical runs on the previous noisy dataset are context only and are not matched controls.

Every compared run uses a live \(g_{\mathrm{aug}}\). Do not add a zero-\(g_{\mathrm{aug}}\) control arm. The experiment is not intended to isolate the causal effect of changing the initialization. Its purpose is to determine whether PS2 remains beneficial once both methods use the selected paper-style initialization.

The primary offline comparison endpoint is the prediction error in the 230 to 297 Hz band containing the missing mode:

* keep the existing checkpoint selection and convergence logic based on validation free-run error;
* after training, score the already selected checkpoints with the external comparison tooling;
* use validation-band error for the method comparison and test-band error once for final reporting;
* report full-band free-run error as a guardrail against improving the target band at the expense of the remaining response;
* compare plain and PS2 runs by matched arm and seed.

Retain PS2 only if it improves the 230 to 297 Hz score consistently beyond run-to-run variation without degrading free-run stability or overall prediction error. If plain training performs equivalently or better, prefer the simpler plain method and conclude that PS2 is unnecessary under this initialization.

The 230 to 297 Hz calculation is not part of the main model, training loop, validation callback, or checkpoint-selection rule. The comparison protocol and its run-table entries are prepared separately before server launch.

## 15. Verification and acceptance tests

### 15.1 Legacy compatibility

1. With legacy settings, the model type, state dictionary, initial parameters, fixed-batch output, and fixed-batch loss match the current pipeline.
2. Historical checkpoints load with their historical architecture.
3. Existing dataset modes still resolve to their original directories.

### 15.2 ResNet construction

1. For arbitrary test input \(z\), the module output equals `nonlinear_mlp(z) + linear_bypass(z)`.
2. Every nonlinear final-layer weight and bias is exactly zero initially.
3. Every physical bypass entry is exactly zero initially.
4. Added-state bypass entries lie within `[-1, 1]`, are not all zero, and reproduce exactly for a fixed seed.
5. The bypass has no bias.
6. Row classification works for reordered and partially routed `ann_route_ix`, not only the default ordering.

### 15.3 Baseline equivalence and hidden-state behavior

1. Before training, the physical-state and output rollout match the baseline to float64 round-off.
2. The added state evolves and `g_aug` is not identically zero.
3. An external preflight utility reports the spectral radius, added-state to physical-state RMS ratio, and first-layer tanh saturation fraction from Section 7 over one 0.1 s training window for every planned seed.
4. Do not reject, redraw, or rescale a finite initialization based only on these diagnostics.
5. Record a nonfinite rollout or training run as a failed initialized run. Do not replace its seed silently.

### 15.4 Encoder

1. The analytical `Wb_psi_y` and `Wb_psi_u` values are unchanged by selecting the new added-state initialization.
2. Recombining `[Wa_psi_y Wa_psi_u]` recovers the single logical matrix that was initialized.
3. Every entry respects the Xavier bound \(a_\psi\).
4. Initialization reproduces exactly for a fixed seed.

### 15.5 Integration

1. PS2 sees the bypass parameters through `ann.parameters()` and completes its smoke path.
2. OBC evaluates the combined physical correction.
3. ANN-zeroed and clamp diagnostics silence both ResNet paths.
4. Checkpoint save, load, deep copy, resume, and evaluation preserve the architecture.
5. Plain and PS2 CPU smoke runs complete for both U and OBC arms on the new data mode.
6. A preflight test builds matched plain and PS2 models with the same seed and verifies that their initial `W_a,g` tensors are exactly equal.
7. Preflight tests, rather than the production loop, can report gradient norms by parameter group for the nonlinear branch, physical bypass, added-state bypass, physical encoder map, added-state encoder map, and physical parameters.
8. The existing offline comparison tooling can score the selected checkpoints over 230 to 297 Hz without changing model training or checkpoint selection.

## 16. Implementation sequence

1. Record the decision in the project decision log before changing framework code.
2. Add the ResNet module with the same external network interface as the legacy MLP.
3. Add configuration and metadata fields for augmentation and encoder initialization.
4. Integrate the ResNet into model construction and classify bypass rows from `ann_route_ix`.
5. Add combined-matrix Xavier initialization to `linear_encoder_init_aug`.
6. Update every utility that assumes a single sequential ANN.
7. Add the new dataset mode without changing existing modes.
8. Add unit and integration tests for Sections 15.1 to 15.5.
9. Run local CPU preflight tests before preparing the separate server comparison plan.
10. Report initialized shapes, bounds, row partitions, matched-draw equality, spectral radii, baseline-equivalence error, and smoke-test results.

Do not submit server jobs as part of the implementation session. The user moves the verified files and submits the confirmed runs.

## 17. Out of scope

* Changing the S-DP state equations or augmentation routing.
* Changing the physical baseline model or output map.
* Changing the analytical formulas for \(W_\psi^b\).
* Changing PS2 stages, losses, thresholds, or handover logic.
* Changing OBC's mathematical projection.
* Adding an off-manifold OBC diagnostic to the production pipeline.
* Changing validation, convergence, or checkpoint selection to use the 230 to 297 Hz score.
* Adding initialization-scale or saturation diagnostics to the production training loop.
* Applying Xavier to the augmentation bypass.
* Adding a bypass bias.
* Silently rescaling or redrawing an unstable initialization.
* Preserving historical RNG advancement through dummy operations.
* Replacing an existing dataset mode with the new low-pass-noise data.

## 18. Terminology for the thesis

Use:

> The general ResNet initialization of Hoekstra et al. (2026), specialized to the structured S-DP augmentation and the analytical/linear encoder used in this work.

Do not call the implementation an exact reproduction of the paper's S-DP experiment. Do not summarize it as only replacing Kaiming by Xavier. The live augmentation bypass and the Xavier added-state encoder are separate changes with separate source arguments.
