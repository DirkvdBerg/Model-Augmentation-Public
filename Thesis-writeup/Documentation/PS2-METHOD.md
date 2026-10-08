# PS2: predictive-state initialisation of the added states (method documentation)

Companion to `TRAINING-DESIGN.md` and `RESULTS-DESIGN.md`. Decisions: D-234 (method choice), D-235 (cubic-absorber test),
D-236 (production implementation and its correction), D-237 (Jan's public code). Implementation design:
`docs/ps2-implementation.md`. Prototype evidence: `scripts/gantry/absorber-learning-diagnosis/overnight-clean-method/REPORT.md`
and `RUNS.md`. Status on 2026-10-04: implemented in the production pipeline, locally verified, server runs in progress
(interim results in Section 10.4). Open points are marked (?).

This revision documents the code inspected on 2026-10-04. It does not implement the proposed safeguards in Section 11.
Bracketed source names refer to the linked references in Section 12.
Equations and numerical results attributed to this project are distinct from results established in those papers.

## 1. Summary

PS2 is an experimentally supported residual-prediction initialisation of the latent added states in Hoekstra's
dynamic-parallel augmentation (S-DP), motivated by predictive-state learning. It leaves the deployed model, the ordinary
simulation-loss definition, the encoder architecture, the routing, the orthogonality construction
(OBC) and the checkpoint selection unchanged. During the first few hundred optimiser updates of an otherwise ordinary run it
performs two supervised stages (S2 itself can take thousands of additional regression steps):

- **S1, predictive state.** The encoder's added-state outputs and a temporary decoder head are trained to predict the current
  model's next M closed-loop output residuals from the encoder's measurement window. This encourages the states to encode
  information useful for residual prediction; predictive sufficiency and a Markov realisation are not guaranteed.
- **S2, state map.** The deployed added-state transition of the augmentation network is fitted, once, as a one-step
  regression that propagates those encoder states (next encoder state from current encoder state and input).
- **S3.** The temporary head is discarded and ordinary thesis training continues from the modified parameters (Adam,
  L-BFGS polish, free-run checkpoint selection). S1/S2 use additional objectives and optimisers, so the overall training
  procedure and optimisation trajectory differ from ordinary training.

On the simulated gantry configurations examined, PS2 improves held-out closed-loop free-run error relative to the
examined plain S-DP controls. The interim production comparisons suggest approximately sixfold lower validation error
in the inspected U and OBC runs, with qualifications about reference checkpoints and unfinished runs in Section 10.4.
Latent Jacobian estimates are secondary diagnostics. Nonlinear-state identification, generality across plants/controllers,
and protection against negative transfer remain unestablished.

## 2. The gap PS2 addresses

### 2.1 The model

Hoekstra et al. [LFR, Table 1, printed p. 3] define the state-level dynamic-parallel augmentation. With this pipeline's
unchanged baseline output map, the deployed U-arm equations are

$$
x_{b,k+1} = f_{\mathrm{base}}(x_{b,k}, u_k) + f_{\mathrm{aug}}(x_{b,k}, x_{a,k}, u_k), \qquad
x_{a,k+1} = g_{\mathrm{aug}}(x_{b,k}, x_{a,k}, u_k), \qquad
y_k = h_{\mathrm{base}}(x_{b,k}, u_k),
$$

with physical states $x_b$ (six in the gantry: X, Theta, Y and their rates, logical coordinates), added states $x_a$
($n_{\mathrm{add}} = 2$ in the thesis main results) and input $u$ (three stage forces). In the thesis pipeline (following Jan's code)
$f_{\mathrm{aug}}$ and $g_{\mathrm{aug}}$ are the physical and added-state rows of ONE network,

$$
\begin{bmatrix} f_{\mathrm{aug}} \\ g_{\mathrm{aug}} \end{bmatrix}(z) = W_2\,\tanh\!\big(W_1 \tanh(W_0 z + b_0) + b_1\big) + b_2,
\qquad z = [x_b;\ x_a;\ u],
$$

a 2 x 16 tanh MLP whose output layer $(W_2, b_2)$ is initialised to zero (`zero_init_feed_forward_nn`), so that the
augmented model has baseline input-output behaviour at the start ([LFR, Sec. 5.4, Eq. 29]). The OBC arm additionally
subtracts its existing projection correction from the physical transition; its added-state rows are untouched by that
subtraction. These equations display the U arm, not the complete OBC transition.

The initial state of every training window comes from a SUBNET-style encoder [SUBNET] on 30 input and output samples.
Write their lengths as $H_u$ and $H_y$, distinct from $n_{\mathrm{add}}$. The implementation slices
$\bar u_k=\operatorname{vec}(u[k-n_b:k+n_{b,r}])$ and
$\bar y_k=\operatorname{vec}(y[k-n_h:k+n_{h,r}])$, with the upper index exclusive;
$n_h$ denotes the code's `fs.na`, not the number of added states. Thus $H_y=n_h+n_{h,r}=30$ and similarly $H_u=30$.
The production right offsets are one sample, so the window includes the measurement at $k$.

$$
\hat x_k = \psi(\bar u_k,\bar y_k) = \begin{bmatrix} W^b_{\psi,u}\,\bar u + W^b_{\psi,y}\,\bar y \\
W^a_{\psi,u}\,\bar u + W^a_{\psi,y}\,\bar y \end{bmatrix} + \psi_{\mathrm{NL}}(\bar u, \bar y),
$$

where $W^b$ is built analytically from the baseline's reconstructability map ([ENC, Eqs. 16, 17], a separate paper):

$$
W^b_{\psi,u}=-A^n O_n^\dagger T_n+r_n,\qquad W^b_{\psi,y}=A^n O_n^\dagger.
$$

Here $O_n$ is the baseline observability matrix, $T_n$ its input-output convolution matrix, and $r_n$ its input-state
convolution matrix, with stacking/time conventions as in [ENC] and `pre_encoder.py`. The implementation also applies
the existing input/output/state offset conversions; the compact equation above suppresses those fixed offsets.
$W^a$ is random (`kaiming_uniform_`) and $\psi_{\mathrm{NL}}$ has a zero output layer. This is a pipeline convention:
[LFR, Eq. 31] prescribes Xavier for the augmented encoder, rather than this Kaiming/zero-output combination (D-152).
[LFR, Eq. 30] describes fitting a baseline encoder to simulated baseline states; the analytical construction here is [ENC].
Eqs. 16 and 17 of [LFR] concern LFR algebraic well-posedness and must not be cited for $W^b$.
The ordinary training objective is the
closed-loop truncated simulation error over windows of $n_f = 400$ samples (0.1 s at 4 kHz) with the known controller K1
wrapped around the model (the project's closed-loop adaptation of [LFR, Eq. 22], D-220). The paper itself supplies an
input-driven truncated simulation criterion, not this controller-wrapped implementation. The thesis arms omit the
baseline-parameter regulariser. OBC changes the model's physical correction, not the squared-error loss definition.

### 2.2 The dead start

At initialisation $W_2 = 0$, $b_2 = 0$, hence $g_{\mathrm{aug}} \equiv 0$ and $f_{\mathrm{aug}} \equiv 0$:

- in every rollout $x_{a,j} = 0$ for $j \ge 1$; only $x_{a,0} = \psi_a(\cdot)$ is non-zero;
- the output does not depend on $x_a$ at all, because $\partial y / \partial x_a$ passes through $\partial f_{\mathrm{aug}}/\partial x_a
  = W_2\,(\ldots)$, which is zero; so the loss gives the encoder's added-state part ($W^a$) exactly zero gradient;
- the added-state rows of $W_2$ receive a gradient only through the effect of $x_a$ on later physical rows, which again
  passes through the physical readout rows of $W_2$: the added-state output rows have zero loss gradient at the exact
  zero start. Their influence involves a product of transition and readout weights.

This is an initial gradient gate, not a proof that ordinary training can never activate the states. Physical output rows
can learn first, and subsequent updates can open the path to the added-state transition and encoder. Measured on the gantry
(diagnosis in `scripts/gantry/absorber-learning-diagnosis/DECISIONS.md`): plain training learns static corrections or fast
real lags; larger learning rates and early Levenberg-Marquardt do not create the missing pole (U1, U4, L1); random live
initialisations rarely place a pole near the missing mode and the loss has little pull toward it from far away (AE1, L1,
W1, X1). At server scale the thesis run 42 (U arm, seed 1) ends with an added-state self-map that has only a real
eigenvalue (|z| 0.67, no complex pair) and leaves 88 to 91 % of the 230 to 297 Hz error in place (Section 10.4).

### 2.3 What the paper and Jan's code do

- The paper's general construction permits a live start: the learning function can be a ResNet with a linear part
  (Eq. 15) and Sec. 5.4.3 initialises matrices not needed for the baseline behaviour randomly (D-233).
- The public-code inspection recorded in D-237 (fresh download, quoted files checked there) found zero initialisation in the inspected
  example: the paper's S-DP script uses `zero_init_linear_mapping` (a linear layer with zero weight and bias) with two added
  states; even `zero_init_resnet` zeroes its linear part; no public script uses a random linear part for parallel
  augmentation. This supports a dead start in the inspected published configurations, not a theorem about all versions
  or all Hoekstra models. That script also uses a default MLP encoder rather than this pipeline's analytical $W^b$ encoder.
  Which configuration produced the paper's
  S-DP table is not determinable from the code (question to Jan open (?)).

### 2.4 The gap in one sentence

The pipeline supplies a baseline-derived initial encoder for $x_b$ but random initial coordinates for $x_a$.
PS2 adds residual-prediction supervision and transition fitting to encourage useful added-state coordinates.
This extends the motivation of baseline-informed initialisation; it does not establish an analytical reconstructability
map or a physical interpretation for the additional states. Ordinary training runs throughout S1.

## 3. Notation

| Symbol | Meaning |
|-|-|
| $k$ | sample index at 4 kHz |
| $n_{\mathrm{add}}, n_y$ | number of added states (2) and measured outputs (3) |
| $H_u,H_y$ | input/output encoder-window lengths (30); distinct from state dimension |
| $\bar u_k, \bar y_k$ | flattened normalised encoder windows, indexed explicitly in Section 2.1 |
| $\psi = [\psi_b; \psi_a]$ | encoder; $\psi_a$ its added-state outputs |
| $\theta_\psi^a$ | encoder parameters selected for S1 updates: $W^a_{\psi,u}, W^a_{\psi,y}$ and the last nonlinear layer; only its added rows receive S1 gradients. Hidden encoder layers also affect $\psi_a$ but are not updated by the auxiliary optimiser |
| $F_\theta$ | one step of the deployed model (physics block plus network), state $\to$ next state |
| $g_\theta$ | the added-state rows of $F_\theta$ (only the network writes them) |
| $\theta_g$ | the network parameters ($W_0, b_0, W_1, b_1, W_2, b_2$) |
| $\mathcal{S}$ | residual-form closed-loop simulator; receives recorded input/output, controller row and encoded initial state |
| $M$ | S1 prediction horizon, = the encoder's past output window (30 samples) |
| $e_k \in \mathbb{R}^{M n_y}$ | the current model's next $M$ closed-loop output residuals from window start $k$ |
| $h_\phi$ | temporary S1 decoder, input dimension $n_{\mathrm{add}}$ |
| $\mathcal{R}_{\mathrm{fit}}, \mathcal{R}_{\mathrm{cal}}$ | records used for auxiliary fitting and withheld from that fitting for phase decisions; both participate in ordinary training |

## 4. The method

### 4.1 Algorithm

```
build the thesis model and the closed-loop simulator as usual          (model equals the baseline)
attach PS2 to the training loop (declared seam, Section 7.2)
for every optimiser update t = 1, 2, ... of the ordinary training:
    ordinary thesis update (closed-loop rollout loss, Adam, unchanged)
    if PS2 is in phase S1:
        one S1 step on its own minibatch with its own Adam               (Section 4.2)
        every 25 updates: calibration check, screen, S1 end rule         (Section 4.5)
        at the S1 end: S2 once, then phase S3                            (Section 4.3)
        if the screen fails: phase S3 without S2 (fall-back)             (Section 4.6)
    phase S3: no auxiliary update; ordinary training from the modified model
L-BFGS polish and free-run checkpoint selection, unchanged
```

### 4.2 S1: predictive state

**Target.** For a training window starting at $k$ with encoder state $\hat x_k = \psi(\bar u_k, \bar y_k)$, the residual of
the CURRENT model over the next $M$ samples, using the same residual-form closed-loop prediction path as validation,
detached from the graph. The actual call supplies recorded $u$, recorded $y$ and the controller-bank row $c_k$:

$$
\tilde e_k = \operatorname{vec}\big(y_{k:k+M-1} - \mathcal{S}_\theta(\operatorname{sg}(\hat x_k);\ u_{k:k+M-1},y_{k:k+M-1},c_k)\big),
\qquad e_k = \tilde e_k \Big/ \sqrt{\tfrac{1}{B M n_y} \textstyle\sum_{\text{batch}} \|\tilde e\|^2}.
$$

Here $\operatorname{sg}$ means stop gradient. The controller reconstructs the model input through the existing relation
$u_{\mathrm{model}}=u_{\mathrm{data}}+C_{\mathrm{fb}}(y_{\mathrm{data}}-y_{\mathrm{model}})$, including its dynamic filters.
PS2 does not directly pass a reference trajectory to the head. The normalisation removes a common residual-amplitude
scale within a batch, but also makes the target depend on batch composition. The denominator has no zero/finite guard
in the inspected code. The target changes with both the model parameters and its encoded initial state, including
changes caused by S1 itself. Detachment blocks a derivative through the target; it does not freeze its numerical value.
Removal of early baseline mismatch is a possible effect of ordinary training, not an imposed ordering guarantee.

**Head.** $h_\phi:\ \mathbb{R}^{n_{\mathrm{add}}} \to \mathbb{R}^{M n_y}$, one tanh layer of width 16 (the deployed network's width),
output layer initialised to zero (it starts predicting "no error").

**Loss and parameters.**

$$
\min_{\phi,\ \theta_\psi^a}\ L_{S1} = \frac{1}{B M n_y} \sum_{\text{batch}} \big\| h_\phi\big(\psi_a(\bar u_k, \bar y_k)\big) - e_k \big\|^2 .
$$

Trained: $\phi$ and $\theta_\psi^a$ only, with their own Adam (learning rate $10^{-3}$, Adam's default; Kingma and Ba
2015), on minibatches of 64 windows drawn from $\mathcal{R}_{\mathrm{fit}}$ only. The gradient is computed with
`torch.autograd.grad` and applied by this optimiser alone; the training optimiser's gradient buffers are saved and
restored around the step, so the thesis update is never mixed with it.

The head sees $\psi_a$ only. It is not conditioned explicitly on $x_b$, future inputs, future reference or controller
state. Accurate prediction may therefore depend on information about excitation or controller behaviour encoded in
$x_a$. The current procedure does not distinguish missing plant memory from those correlations.

**Horizon rule.** $M=H_y=\texttt{fs.na}+\texttt{fs.na_right}=30$ samples (7.5 ms).
Reusing the encoder length reduces the number of independent settings; it does not prove that this horizon is sufficient
for the missing dynamics. Equal past/future lengths are a design convention, not a general subspace-identification
requirement. The earlier M = 40 proposal was dropped (D-234).

**Literature and scope.** [2SR] motivates regression from history to future information. Its S1A and S1B estimate
conditional expectations of future and extended-future features; S2 fits a linear operator between them. PS2 has no S1B
or matching instrumental-regression construction. Its nonlinear encoder bottleneck, changing residual targets and
nonlinear transition regression prevent direct application of that paper's Theorem 2.

With a linear encoder/head, fixed targets and a fixed normalisation, the S1 composition has a rank-limited linear
regression interpretation (affine if biases are included). The implemented changing, batch-normalised targets do not
make it exactly ordinary reduced-rank regression. [PSD, Eq. 4] supports auxiliary future decoding from recurrent states;
PS2 instead supervises encoder states with model residuals and uses separate stages/optimisers.

### 4.3 S2: state map

At the S1 end, once:

1. **Snapshot.** Draw 8192 window pairs from $\mathcal{R}_{\mathrm{fit}}$ and 2048 from $\mathcal{R}_{\mathrm{cal}}$; for
   each, the encoder state at $k$ and at $k+1$ (the window shifted by one sample) and the input $u_k$, all under
   `no_grad`:

$$
(X_i, U_i, T_i) = \big(\psi(\bar u_k, \bar y_k),\ u_k,\ \psi_a(\bar u_{k+1}, \bar y_{k+1})\big).
$$

2. **Regression** of the deployed added-state rows on the frozen pairs:

$$
\min_{\theta_g}\ L_{S2} =
\frac{\frac{1}{B n_{\mathrm{add}}}\sum_{i=1}^{B}\big\|g_\theta(X_i,U_i)-T_i\big\|_2^2}
{\frac{1}{n_{\mathrm{add}}}\sum_{j=1}^{n_{\mathrm{add}}}\operatorname{Var}_{i}(T_{ij})}.
$$

   Trained: all network parameters $\theta_g$ with a fresh Adam ($10^{-3}$), minibatches of 512 from the fit snapshot.
   The denominator is computed on the targets of each evaluated batch/snapshot using `T.var(0).mean()` (PyTorch's
   default sample-variance correction). Thus it is not one globally fixed scale for all minibatches. There is no
   zero-variance guard in the inspected code. Calibration is checked every 100 steps.

3. **Stop.** After at least 500 steps, when the best of the last five checks improved by less than 1 % (relative) on the
   best before them; at the latest after 8000 steps.

**Weight selection as implemented.** `_fit_map` initialises `best=r0` but `state=None`. It saves parameters only when a
checked calibration residual is strictly smaller than `r0` or the subsequent best. If at least one improvement occurs,
it restores the best saved weights. If none occurs, it leaves the final weights in place even though `residual_best`
still reports `r0`. This is a known defect: counting the starting weights as the initial best would allow rollback.
There is no minimum fit-quality threshold, finite-value rejection, multi-step acceptance screen or S2 rejection status.
`after_step` reports `completed` after the routine returns regardless of fit quality. These safeguards are proposed,
not implemented.

**Mechanism hypothesis.** S2 supplies direct one-step supervision without differentiating a long rollout. It may avoid
weak long-horizon gradients, but inherits the adequacy and propagatability of the S1 coordinates. [PSRNN, Sec. 4.2]
provides precedent for initialisation followed by BPTT; its failure of random initialisation is an observation about its
own PSRNN architectures and experiments, not a theorem about S-DP. PS2 is a distinct adaptation.

**What S2 changes.** $L_{S2}$ depends on the network only through its added-state outputs, so the physical rows of $W_2,
b_2$ get exactly zero gradient and stay unchanged. The shared hidden layers $W_0, W_1$ do move; that changes the physical
correction even though its output weights are preserved. The magnitude is not bounded by the implementation. Changes
to the added-state trajectories can also change physical corrections indirectly. The inspected logs show an initial
validation deterioration followed by recovery, but this is empirical and is not guaranteed. The physical readout may
already have moved during the preceding ordinary updates.

### 4.4 S3: refinement

No further PS2-specific updates: `_head` and `_opt` are set to `None`, the training-phase seam is detached when `fit()` returns,
and the run continues with the unchanged closed-loop objective, the ordinary Adam learning rate ($10^{-5}$), the L-BFGS
polish and the free-run checkpoint selection. The $10^{-5}$ rate refers to production Adam refinement; auxiliary S1/S2
use $10^{-3}$ and L-BFGS has its own settings. S3 can adapt the readout, transition, encoder and physical parameters
jointly. Interpreting its effect as recovering plant dynamics from a closed-loop predictor is a hypothesis (Section 6.3).

The ordinary Adam's moments are retained across auxiliary changes to overlapping parameters. Separate gradient buffers
prevent loss-gradient mixing, but do not make the optimisers independent: each subsequently evaluates gradients at
parameters changed by the other. Retaining/resetting moments after S2 is an untested algorithmic choice.
The opening object's `_aux` list still retains references to the head parameters until that object is released;
the head is no longer evaluated or updated and is not part of the deployed model or its checkpoint.

### 4.5 Phase switching (data-only rules)

- **Calibration records.** Record-level split of the training set: the last record of every record class with at least
  three records is held out from S1 and S2 fitting (class = record name without its trailing number). On the thesis set:
  TR-S5, TR-Y3, TR-P4, TR-T4; S1 and S2 fit on the other 14 records. These records still take part in the ordinary
  training. Rationale: the predictive state of one excitation class need not transfer to another (prototype P1, D2:
  the original T-only calibration exposed a cross-class prediction failure). The implemented split covers classes with at
  least three records; it does not include the L class, which has only two. It is representative of selected classes,
  not every class. These calibration records are withheld from auxiliary fitting only, not independent of model training.
- **Calibration check.** Every 25 updates: unexplained fraction of the head on 128 fixed calibration windows,

$$
U = \frac{\sum \| h_\phi(\psi_a) - e \|^2}{\sum \| e \|^2} .
$$

- **S1 end.** Plateau (after at least 100 updates, the best of the last four checks improved by less than 2 % on the best
  before them) or the cap of 350 updates.
- **Screen.** If $U > 0.90$ at update 150 (and S1 has not ended), or at the S1 end: fall-back (Section 4.6).

The auxiliary switch/screen rules do not inspect poles, frequencies, the absorber, the thesis validation set or truth
quantities. Ordinary best-checkpoint selection still uses thesis validation throughout. The phase decisions depend on
data and hand-set thresholds/caps, so calling them automatic or free of scheduled switch counts would overstate them.

### 4.6 Fall-back

The code labels a failed S1 screen `fallback`. Operationally, it skips S2, deletes the temporary head/auxiliary optimiser,
and continues ordinary training from the current parameters. It retains S1's encoder updates and their effects on the
interleaved ordinary updates. It does not restore a plain-training trajectory and supplies no guarantee against worse
performance. A failed screen means this particular predictor did not meet the threshold by the deadline; it does not
prove that the residual contains no predictable dynamics.

Restoring the encoder's initial weights would also erase ordinary-training updates. Subtracting accumulated auxiliary
updates would not reconstruct the plain trajectory, because later ordinary gradients depended on those updates.
An exact fallback would require a different design, such as an isolated pretraining stage with rollback or a separately
advanced control trajectory. In the reported gantry/cubic runs this branch did not trigger. Gate D used an earlier
stop-on-screen protocol, so its prematurely stopped runs are not full-budget comparisons (Section 10.3).

### 4.7 What is detached, fixed or separate

| Quantity | Treatment | Why |
|-|-|-|
| S1 target $e_k$ | detached, recomputed from the current model each update | a target, not a path into the model |
| S1 input to the rollout $\hat x_k$ | detached, recomputed | prevents target-path derivatives, not numerical dependence on encoder parameters |
| S2 inputs $X_i, U_i$ and targets $T_i$ | frozen snapshot, `no_grad` | prevents joint shrinkage of encoder targets during S2; does not prove non-collapse or predictability |
| PS2 optimisers | separate Adam instances with overlapping model parameters | ordinary update uses its own gradient; step frequency/rates still set relative influence. Adam's finite epsilon prevents exact scale invariance |
| head $h_\phi$ | no longer used after S1; excluded from deployed model/checkpoint | learned encoder/transition changes remain, but no decoder is deployed |

### 4.8 Why parameter selection suffices (no masks)

$L_{S1}$ reaches the encoder through $\psi_a$: $W^b$ and the last layer's physical rows have zero auxiliary gradient,
while shared encoder hidden layers can have nonzero gradients but are excluded from the auxiliary optimiser.
$L_{S2}$ reaches the ANN through its added-state rows, giving its physical output rows zero gradient.
Adam with a zero gradient and zero moments makes an
exactly zero update ($m = v = 0$, step $= 0/(0 + \epsilon) = 0$). Selecting the parameter tensors is enough; the
pre-flight test asserts bit-identity of every protected tensor across S1 steps and S2 (Section 8, T3).

This statement concerns the auxiliary updates only. Ordinary training can update these tensors between auxiliary steps.
The encoder's hidden layers are excluded from the auxiliary optimiser, not intrinsically zero-gradient paths.
S2's shared ANN hidden layers are included and can move. Zero physical-output-row gradients preserve those rows because
the fresh S2 Adam has zero moments and no weight decay; they do not preserve the full physical-output function.

## 5. Rules and their origin

The numerical stopping/sampling rules are fields of `PS2Spec`. Horizon and head width are derived from the model.

| Rule | Value | Label and origin |
|-|-|-|
| S1/S2 learning rate | $10^{-3}$ | HEURISTIC: recommended default in [ADAM, Algorithm 1], not an optimal-rate theorem for PS2; the code's `THEORY` label should be corrected separately |
| S1 horizon $M$ | 30 samples | framework quantity: the encoder's past output window (equal past and future horizons) |
| head width | 16 | framework quantity: the deployed network's width |
| S1 batch | 64 windows | HEURISTIC: the prototype's batch; the opening's own, independent of the training batch (512) |
| check cadence | 25 updates | HEURISTIC |
| S1 minimum / cap | 100 / 350 updates | HEURISTIC; the cap ended S1 in every run so far (?) |
| S1 plateau | 2 % over 4 checks | HEURISTIC |
| screen | $U \le 0.90$ at 150 and at the S1 end | HEURISTIC |
| calibration windows | 128 | HEURISTIC |
| S2 snapshot | 8192 fit / 2048 calibration pairs | HEURISTIC |
| S2 batch / check / min / cap | 512 / 100 / 500 / 8000 steps | HEURISTIC |
| S2 plateau | 1 % over 5 checks | HEURISTIC |
| calibration split | last record in dataset order of each class with 3 or more records | HEURISTIC, reasoned (Section 4.5); ordering is not a numerical suffix sort |

The tolerances and caps have no literature value. The cap of 350 updates, not the plateau rule, ended S1 in every gantry
run (local and server): S1 was still improving about 1 % per 25 updates. A principled S1 stopping rule is open (?).

## 6. Mechanism analysis

### 6.1 Gradient paths at the zero start

1. **What trains $\psi_a$ before the model reads $x_a$?** $L_{S1}$: $\partial L_{S1} / \partial \theta_\psi^a = J_h^\top (h - e)
   \,\partial\psi_a/\partial\theta_\psi^a$. At update 1 it is zero (head output layer zero); the head's output layer gets
   a batch gradient proportional to $-\sum_i e_i a_i^\top$ (with $a_i$ the hidden activation).
   Unless that gradient cancels, later S1 steps can train $\psi_a$ without an active deployed readout.
2. **What forces $x_a$ to carry temporally predictive information rather than a constant?** A constant $x_a$ lets the head
   predict a constant vector. If residual variation is learnable by the chosen encoder/head, a predictive representation
   can lower the loss. This is an incentive, not a guarantee that optimisation finds it or preserves every latent dimension.
3. **What trains $g_{\mathrm{aug}}$ to propagate it?** $L_{S2}$, a supervised regression, first order in $W_2$ and in the
   added-output weights. Hidden-layer gradients also depend on the added-output weights: they vanish if those weights
   are exactly zero, but need not vanish after the ordinary updates preceding S2. This does not require a long rollout.
4. **Can $f_{\mathrm{aug}}(x_b, u)$ bypass $x_a$?** In S3, as in any S-DP training; the clamp evaluation and the band
   analysis measure it (Section 9).
5. **Is the closed-loop rollout still the optimised object?** Yes: the thesis objective runs unchanged throughout, and the
   S1 target itself comes from the closed-loop prediction path.

### 6.2 Collapse

There is no general collapse-prevention guarantee. S1 can favour informative representations over constant predictions,
but can still encounter collapsed/degenerate solutions, insufficient signal, saturation or local optima. In S2, freezing
the encoder stops it shrinking jointly with the transition; an already uninformative snapshot or an unsuccessful
transition fit remains possible. Zero target variance also makes the normalised S2 loss undefined.

[SPRL, Theorem 3] establishes a non-collapse result under linear encoder/transition assumptions, continuous-time
training and a transition at a stationary point. Those conditions do not describe this nonlinear Adam procedure.
The paper also reports collapse with detached targets in some nonlinear experiments. Its theorem is background
motivation for target treatment, not a PS2 guarantee. The rejected NPSP consistency-only construction admitted constant
states, but rejecting it does not establish that the replacement cannot collapse.

### 6.2.1 Predictive information versus a propagatable state

A low S1 loss says that the encoded variable helps predict a finite residual window. It does not establish that
$x_{a,k+1}=g_\theta(x_{b,k},x_{a,k},u_k)$ can propagate the representation without re-encoding measurements.
The encoder may store excitation phase, controller correlations or history-window features. S2 tests one-step fit only;
small one-step errors may accumulate on rollout. A proper state-adequacy check compares recursively propagated states
or decoded predictions with independently re-encoded states at several horizons on unseen records. Neither S1 nor the
implemented S2 acceptance logic currently guarantees this Markov/rollout property.

### 6.3 Closed loop

For scalar LTI loops with controller $C$, true plant $P_0$, model $P_x$, additive output disturbance $v$ and zero-state
transfer responses, define $S_0=(1+P_0C)^{-1}$ and $S_x=(1+P_xC)^{-1}$. The loop equations give

$$
y-y_{\mathrm{sim}}=S_0(P_0-P_x)S_xCr+S_0v.
$$

This is an illustrative project derivation (the local LC research note, Sec. 0), not an equation attributed to [LFR].
Its transfer poles generally reflect the true/model closed loops, subject to cancellations. Nonzero initial conditions
add transient terms. Matrix-valued loops require the appropriate left/right sensitivity ordering; the scalar product
must not be applied unchanged to the coupled nonlinear gantry. Finite windows, current encoder states and batch
normalisation further distinguish the implemented target from this ideal transfer expression.

The observed latent Jacobian frequencies changed after S2 and during S3 (prototype: roughly 233/206 Hz after S2 and
211/203/208 Hz later; inspected server diagnostics: roughly 213 Hz). This is consistent with a changing predictor,
but does not prove that S1 identified closed-loop poles or that S3 recovered plant poles (Section 6.4).

Measured closed-loop inputs/outputs can be correlated with disturbance, so residual prediction may also learn noise
or controller correlations. [CL-ID], [CL-IV] and [IVNN] provide background on bias and conditional consistency of
particular estimators. Reference instruments require independence, relevance/rank and appropriate estimator/model
assumptions. They do not automatically remove all bias here, nor make a closed-loop residual equal to plant mismatch.
No instrumental-variable objective or noise model is implemented in PS2.

### 6.4 State gauge

PS2 does not supervise absorber displacement, velocity, mass, stiffness, damping or frequency. Its added states remain
latent. A useful representation need not equal a physical state or be an invertible transform of the true missing state;
invertibility and sufficiency have not been established. Fixed network capacities and auxiliary losses also bias which
coordinates are easy to learn.

For a valid smooth invertible coordinate change $z=T(x)$ of a discrete-time model $x_{k+1}=F(x_k,u_k)$,

$$
D_z\widetilde F(z_k,u_k)=DT(x_{k+1})\,D_xF(x_k,u_k)\,DT(x_k)^{-1}.
$$

At a fixed point these factors give a similarity transformation, preserving eigenvalues. Constant linear coordinate
changes do so everywhere. A nonlinear change at a general trajectory point need not preserve instantaneous eigenvalues.
Moreover, `inspect_checkpoint.py` computes only $\partial g/\partial x_a$, holding $x_b$ and $u$ fixed, rather than the
full coupled state Jacobian. Its frequencies/damping characterise that block in this learned coordinate system.
They are suggestive diagnostics, not coordinate-invariant physical poles or proof of absorber identification.

## 7. Implementation

### 7.1 Files

| File | Change | Lines |
|-|-|-|
| `model_augmentation/fit_systems/ps2_opening.py` | new (`__project_origin__ = "added"`): `PS2Spec`, `calibration_split`, `PS2Opening` | about 255 |
| `model_augmentation/fit_systems/interconnect.py` | declared seam and checkpoint exclusion, all marked `D-236` | +18 |
| `scripts/gantry/gantry_dynamic/config.py` | `RunConfig.ps2: bool = False`, recorded in config.json (`PS2`) | +4 |
| `scripts/gantry/gantry_dynamic/training.py` | attach before `fit()`, detach after, summary in the checkpoint meta | +18 |
| `scripts/gantry/gantry_interconnect_dynamic.py` | `THESIS_PS2 = 0 / 1` in the thesis block; `PS2` in the run line | +5 |
| `scripts/gantry/thesis-results/runs.tsv`, `submit.sh` | tier `ps2` (ids 111 to 116), `ps2smoke` (131, 132) | |

### 7.2 The framework seam

`SSE_Interconnect` declares a class attribute `training_phase = None`. In `fit()`, directly after the optimiser step and the
NaN check, the loop calls `self.training_phase.after_step(self)` if it is not None. With None the loop is unchanged
(tested bit-identical, T1). `SSE_Interconnect_Composed` excludes `training_phase` from checkpoints and restores it across
deepSI's end-of-fit `self.__dict__ = torch.load(...)`, exactly as it treats `obc` and `traj_penalty`. The seam is generic: any
staged training procedure can use it.

### 7.3 `PS2Opening`

- `PS2Opening.for_pipeline(fit_sys, data, n_xb, n_a, width, spec)` builds the opening from the pipeline's data bundle: the
  normalised training records, their controller rows (`fit_sys.simulator.train_ctrl_rows`), the record names and the
  calibration split.
- `after_step(fit_sys)`: the seam method (S1 step, checks, switch, S2, fall-back); a no-op once `done`.
- `done`, `summary()`, `summary_json()`: state and a plain-number record (status, updates, S1 end, S2 residuals,
  calibration records, history, spec).
- Internals: `_sample` draws windows with the deepSI window convention on the model's device and dtype; `_residual`
  predicts through `_sync_prediction_state_for_validation()` plus `simulate()` without `obc_pass` (the documented
  prediction path, so the OBC projection is included as in validation and the compiled training rollout is not used);
  `_s1_step`; `_cal_unexplained`; `_fit_map` (S2).

### 7.4 Pipeline switch and logs

- Run with PS2: `THESIS_PS2=1` (plus the usual `THESIS_*` variables); default 0 is the unchanged thesis run.
- The run line ends in `PS2`; the opening prints `[ps2] opening started` (spec and calibration records), `[ps2] update N:
  calibration unexplained U` every 25 updates, `(S1 ends: plateau|cap)`, `[ps2] S2 state-map fit: calibration EE residual
  r0 -> r in s steps`, `[ps2] opening completed` (or the fall-back reason) and, after `fit()`, `[ps2] summary: {...}`; the
  summary is also stored as `ps2_summary` in the checkpoint `.npz` meta.
- Attachment requires `cfg.ps2`, `done_epochs == 0` and `start_phase == 'adam'`. A resume with positive recorded epochs
  skips the opening. A checkpoint recorded at epoch zero can satisfy the condition and rerun it. The auxiliary
  optimiser/head state is not checkpointed, so interruption/resume during S1 is not supported as exact continuation.
- `THESIS_SMOKE=1` uses a short spec (S1 cap 4 updates, small snapshots) to test the launch path in minutes.

### 7.5 Server runs

`bash scripts/gantry/thesis-results/submit.sh ps2smoke` (launch test, both arms), then the tier `ps2` (U and OBC, seeds 1 to
3). References: the existing thesis core runs 41 to 46 (same arms, seeds and settings without PS2; no new control runs, user
decision, D-236 correction). The OBC runs are slow on blade1 (oahu); a 72 h wall time and the faster partitions (hawaii,
lanai, molokai) are used for them.

### 7.6 Checkpoint attribution

The ordinary selector can save checkpoints before S2 as well as afterwards. At the end of `fit()` deepSI reloads the
best checkpoint, while the live opening summary describes the completed run. A summary with `status=completed`
therefore does not establish that the selected weights contain the S2 implant. The current summary does not record
the selected checkpoint's PS2 phase. Check its selected update/epoch against the S2 boundary before attributing a
result to the complete S1/S2/S3 procedure. No post-S2-only selection rule is implemented.

### 7.7 Cost

S1: about 350 extra steps, each an encoder pass, an M-step closed-loop prediction on 64 windows and a small head
(on the GPU a fraction of a thesis update; on the OBC arm one extra prediction-coefficient solve per step). S2: a few
thousand one-step regressions on a snapshot (minutes in the local checks). Relative cost depends on hardware, OBC
coefficient solves and the ordinary rollout horizon/batch. A claimed percentage overhead requires measured timings;
the separate update counts alone do not establish an overhead below 5 %.

## 8. Verification

| Id | Check | Result |
|-|-|-|
| T1 | the seam has no side effect: per-update losses and final weights identical with `training_phase=None` and a do-nothing phase | PASS |
| T2 | S1 steps at the zero start leave the training loss unchanged (model still equals the baseline) | PASS |
| T3 | S1 writes only $W^a$ and the added-state rows of the encoder's last layer; S2 writes only the network, physical output rows bit-identical | PASS |
| T4 | S2 lowers the calibration one-step residual | PASS (1.007 to 0.086 on the smoke-sized snapshot) |
| T6 | training loss on a fixed batch before and after S2 (information only) | identical, 1.785e-8 (S2 moves the hidden layers, x_a not yet read) |
| T5 | checkpoints contain no `training_phase`; the phase survives the save | PASS |
| T7 | inside the real `fit()` the opening completes and is done when `fit()` returns | PASS |
| T8 | calibration split = TR-S5, TR-Y3, TR-P4, TR-T4 | PASS |
| T9 | S1 and S2 run on the OBC arm through the prediction path | PASS |
| server | GPU, float64, compiled training | server smoke and runs: `training rollout COMPILED`, opening completed |

Tests: `scripts/gantry/thesis-results/test_ps2_pipeline.py` (log `logs/test_ps2_pipeline.log`). Entry-script smoke on CPU
(both arms) and on the server (jobs 87699 U, Quadro RTX 6000; 87698 OBC, A100). Real-data mechanism check on CPU with the
production code at the thesis learning rate (`check_ps2_mechanism.py`): S1 calibration 0.6046 at the cap (prototype 0.605),
S2 residual 1.011 to 0.0074 (prototype 0.0087).

These are recorded checks, not proofs of predictive sufficiency, identifiability, rollout stability or generality.
T1 compares the new seam's `None` and no-op paths; it is not a comparison against a separately executed historical
checkout. T6 is measured at the zero physical-output start and does not bound S2's effect on a trained physical
correction. No existing preflight test establishes exact fallback, S2 rollback on failure, safe zero-variance handling,
selected-checkpoint phase attribution or exact mid-opening resume.

## 9. How to evaluate a PS2 model

- **Primary:** closed-loop free-run error on held-out records (validation for selection, test records for the claim),
  against the reference run with the same arm and seed.
- **Band analysis:** error per band (0-20, 20-106, 106-140, 140-230, 230-297, 297-2000 Hz) per channel
  (`scripts/gantry/thesis-results/band_compare.py`).
- **Quiet records:** no record worse than the model with the learned block off.
- **Added-state use:** zeroing the network's added-state rows (`inspect_checkpoint.py`). Caveat: the network's hidden layers
  and physical output weights are unchanged by this intervention, but their inputs change as $x_a$ becomes zero after
  the first transition. The physical correction can therefore change along the rollout. The encoded initial $x_a$
  remains nonzero. A performance change indicates dependence on added-state propagation; it does not isolate a physical
  subsystem, establish slow memory, or exclude a fast/static correction routed through those states.
- **Diagnosis only (never a criterion, never for selection):** eigenvalues of the local added-state self-map at encoder
  states (frequency and damping of the least-damped pair), compared with the truth only after all decisions.
- **State consistency:** `overnight-clean-method/d4_rollout_consistency.py` encodes the initial state, propagates the
  complete deployed map with recorded inputs, and compares its $x_a$ with independently re-encoded states at horizons
  1, 10, 30 and 100. It is an input-driven coordinate-consistency diagnostic, not the thesis closed-loop evaluation.
  Encoder noise and drift of the physical states can contribute to disagreement; a large error weakens a self-propagating
  representation claim but does not by itself disprove useful input-output dynamics.
- **Nonlinear generalisation:** compare unseen amplitudes with trajectory shape/phase held fixed and compare a genuinely
  linear latent transition against the nonlinear map. A linear temporary head does not make the deployed transition
  linear. Existing mixed extrapolation records do not isolate amplitude dependence.

## 10. Evidence

### 10.1 Prototype, linear absorber (local, 600 updates, batch 64; D-234)

Paired with exact controls (same initialisation and batch sequence), validation free run over six records:
seed 0 3.12e-6 vs 1.145e-5; seed 2 9.39e-6 vs 1.148e-5; seed 3 3.20e-6 vs 1.140e-5 (3 of 3 better, median ratio 0.28). Learned
pairs (coordinate-dependent diagnosis) 211, 203 (underdamped, zeta 0.33), 208 Hz. One seed (P1) first failed its screen because the calibration
records were the no-multisine T class only; corrected by the class-stratified split (reported as a post-failure correction).
The corrected seed-0 rerun is exploratory; seeds 2 and 3 followed the revised protocol. Prototype ordinary-training
rates were $10^{-3}$ for the added-output rows and $10^{-5}$ elsewhere in both arms, rather than production's uniform
$10^{-5}$ Adam rate. The local and production experiments are not identical training protocols.

### 10.2 Cubic absorber, nonlinear missing transition (local; D-235)

Gantry truth with a hardening absorber spring ($k_a \delta + k_{a3} \delta^3$, cubic force 0.33 of the linear at RMS, equal at
peak stroke), 12 training and validation records plus 5 test records. Plain S-DP formed no slow mode (it routed a fast
correction through $x_a$). PS2 against plain S-DP: validation 0.55 / 0.50 / 0.54 (3 seeds); unseen test records 0.66 / 0.51 /
0.56, better on every record class in at least 2 of 3 seeds; on the larger-stroke record the advantage shrinks in 2 of 3
seeds (part of what is learned behaves like an averaged mode; no evidence that the hardening itself is represented).

Gate 1 failed its predeclared clamp-effect rule: ordinary S-DP's clamp effect was 42 %. The experiment continued after
the pole diagnostic motivated a different interpretation; this is a logged exploratory deviation, not a passed
preregistered gate. The quoted force ratios describe the generated truth/data and are not supplied to PS2 training.

The prototype D4 diagnostic on PS2 seeds 0/2/3 reported one-step normalised state discrepancies 0.200/0.176/0.138, but
at 30 steps 0.956/0.842/1.063 and at 100 steps 1.358/0.820/0.925 (`outputs/d4_cubic.log`). The denominator is encoder
state RMS, not centred variance. These large longer-horizon discrepancies do not support claiming that S1's encoded
coordinates propagate accurately without re-encoding. The held-out output improvements remain separate evidence.

### 10.3 Gate D, hidden Duffing mass in Jan's 3-DOF chain (local; negative)

Plain S-DP already uses its added states effectively on that input-driven benchmark and has lower nominal test error
than the reported PS2 endpoints in all seeds. Several PS2 arms terminated early on the screen under the earlier
protocol, so those endpoints are not equal-budget comparisons. The completed nonlinear-head seeds also lost on nominal
test error, despite a limited amplitude-extrapolation hint. PS2 is not shown to improve systems where plain training
already succeeds, and the production skip-S2 branch does not establish a remedy for negative transfer.

### 10.4 Production, server (interim, 2026-10-04; best checkpoints of runs in progress)

| Arm, seed | PS2 job | Validation free-run RMS, PS2 | Reference | Ratio |
|-|-|-|-|-|
| U s1 | 87705 | 1.597e-6 m | 1.027e-5 m (run 42) | 0.16 |
| OBC s1 | 87703 | 1.992e-6 m | 1.153e-5 m (run 41, local copy before its best) | 0.17 |
| OBC s2 | 87704 | 1.643e-6 m | 1.149e-5 m (run 43, local copy before its best; log best 1.028e-5) | 0.14 |
| OBC s3 | 87700 | 1.808e-6 m | 1.035e-5 m (run 45) | 0.17 |

This table preserves the reported 2026-10-04 snapshot; it is not an automatically refreshed final result. Two reference
artifacts predate their best checkpoint, so their ratios can exaggerate the improvement relative to final controls.
These are matched arm/seed/settings comparisons to older runs, not demonstrated same-code paired reruns. Final
comparisons must use the correct selected artifacts, selection stage (Adam or polish), code provenance and test data.

- Opening (four logged runs): S1 ended at the cap of 350 updates with calibration unexplained 0.58 to 0.68; S2 calibration residual from about 1.00 down to between 0.004 and 0.011, in 3100 to 3300 steps; status completed in every run.
- Bands, multisine validation records, PS2 / reference, Y: 230-297 Hz 0.08 to 0.11 (the references remove 9 to 20 % of that
  band's error, PS2 about 90 %); 140-230 Hz 0.05 to 0.12; 106-140 Hz 0.05 to 0.22; below 106 Hz 0.33 to 0.56; 297-2000 Hz
  0.23 to 0.42. X: 230-297 Hz 0.27 to 0.37; below 106 Hz unchanged (0.94 to 1.04, at the no-augmentation level for both
  models); 297-2000 Hz 1.0 to 2.6 at about 1e-7 m.
- No validation record worse than the learned block off (U s1, OBC s3); the references are worse on the quiet records VA-T1,
  VA-T2. On the quiet records PS2 is 2 to 5x better below 106 Hz and slightly higher above 230 Hz at 2e-8 to 3e-8 m (about
  50x below those records' totals).
- Diagnosis only: added-state pair at all 24 encoder points, 213 Hz, zeta 0.045 (U s1) and 0.034 (OBC s3); true absorber
  211.9 Hz, zeta 0.043; run 42: real eigenvalue only (|z| 0.67).
- Open: U seeds 2 and 3, final polished checkpoints, test-record results, OBC physical-parameter estimates vs the
  references (?).

## 11. Limitations

- **Theory.** PS2 differs from the algorithms underlying the cited guarantees. [2SR] permits nonlinear S1 but retains
  an operator-regression structure absent here; [PSRNN]'s discrete consistency result and [SPRL]'s linear theorem do
  not establish consistency or non-collapse for PS2, including a linear-head variant. Its staged/interleaved optimisation
  has no proof of convergence to the missing plant dynamics.
- **Closed loop and moving targets.** Residuals depend on controller, excitation, current model and current encoder.
  The head does not explicitly condition on future excitation. There is no noise-bias treatment or proof that S3 removes
  controller/excitation shortcuts.
- **Heuristics.** The S1 cap decided every run; the switch tolerances and the calibration split have no literature value.
- **Order/horizon.** $n_{\mathrm{add}}=2$ matches the known absorber order in these simulations. PS2 does not select state
  dimension or establish sufficiency of $M=30$; these remain study-design assumptions rather than inferred quantities.
- **Generality.** One plant, one controller, simulation. No gain where plain S-DP already works (Gate D). The nonlinear
  missing transition (cubic absorber) is handled at the output level, but its amplitude dependence is not shown to be
  represented.
- **Shared network.** In Jan's single network the hidden layers serve both the physical correction and the added-state
  transition; S2 moves them, and the clamp test cannot separate the two roles.
- **Provenance.** The server logs record no git commit (the cluster repository is not a git working copy).
- **Safety.** Screen failure retains S1 changes. S2 lacks initial-best rollback, fit-quality rejection and rollout
  acceptance. Zero residual RMS, zero target variance and non-finite values are unguarded. The spec also lacks validation
  of record lengths, positive added-state dimension and check/screen cadence alignment.
- **Optimiser/checkpoint lifecycle.** Ordinary Adam moments remain after auxiliary parameter changes. Auxiliary state is
  excluded from checkpoints. The final opening summary does not establish the selected checkpoint's phase.
- **State adequacy.** One-step fit is not a Markov or multi-step consistency guarantee. D4 showed substantial disagreement
  at longer horizons. No physical coordinate identification or coordinate-invariant pole claim follows.

For future code revisions, priority safeguards are: save the initial S2 weights as the initial best; reject non-finite
and inadequate fits; record the selected checkpoint's phase/update; validate numerical denominators and sampling rules;
measure S2's physical-function change at trained weights; and explicitly choose retained/reset ordinary Adam moments.
An exact plain-training fallback requires a separate design. These are recommendations, not features of the code
documented here. Existing jobs/results must retain their original implementation attribution.

## 12. Sources, exact attribution and verification scope

The relevant equations/passages of the principal method papers below were checked for this revision. This is not a
claim that every theorem assumption or every cited publication has been independently reproduced. In particular,
closed-loop identification references are background, not a consistency proof for PS2. Equation numbers refer to
the linked versions; the two Hoekstra papers must not be merged into one reference.

| Key | Source | Supported use here; boundary of attribution |
|-|-|-|
| [LFR] | J. H. Hoekstra, B. Györök, R. Tóth, M. Schoukens (2026), [Learning-based augmentation of first-principle models: A linear fractional representation-based approach](https://arxiv.org/abs/2602.17297), arXiv:2602.17297; [PDF](https://arxiv.org/pdf/2602.17297). | Table 1: S-DP. Eq. 15: ResNet bypass. Eq. 22: input-driven truncated simulation criterion. Sec. 5.4, Eqs. 29–31: baseline-preserving initialisation and encoder construction, including Xavier for the added-state encoder. Eqs. 16–17 are **not** the analytical baseline-encoder formulas. PS2 and this project's closed-loop loss are not prescribed by this paper. |
| [ENC] | J. H. Hoekstra, B. Györök, R. Tóth, M. Schoukens (2026), [Encoder initialisation methods in the model augmentation setting](https://arxiv.org/abs/2602.13108), arXiv:2602.13108; [PDF](https://arxiv.org/pdf/2602.13108). | Sec. 3.1, Eqs. 16–17: analytical baseline encoder through observability/reconstructability for the stated linear baseline setting. These are the source of the $W^b$ formulas in Section 2.1, not [LFR]. Neither source establishes the PS2 latent-state fit. |
| [SUBNET] | G. I. Beintema, M. Schoukens, R. Tóth (2023), [Deep subspace encoders for nonlinear system identification](https://doi.org/10.1016/j.automatica.2023.111210), *Automatica* 156, 111210; [preprint](https://arxiv.org/abs/2210.14816). | Encoder-based initial-state estimation and truncated simulation training. This supports the underlying estimation architecture, not PS2's residual opening or its closed-loop guarantees. |
| [2SR] | A. Hefny, C. Downey, G. J. Gordon (2015), [Supervised Learning for Dynamical System Learning](https://arxiv.org/abs/1505.05310), *NeurIPS 2015*. | Predictive-state learning via supervised regressions; nonlinear first-stage regression is permitted. S1A, S1B and operator-regression S2 are distinct from PS2. Theorem 2's assumptions and error bound do not transfer to PS2. |
| [PSRNN] | C. Downey, A. Hefny, B. Li, B. Boots, G. J. Gordon (2017), [Predictive State Recurrent Neural Networks](https://arxiv.org/abs/1705.09353), *NeurIPS 2017*. | Sec. 4.2: 2SR initialisation followed by recurrent optimisation. Reported random-initialisation failures concern their models/experiments, not all S-DP models. Their qualified discrete consistency result is not a PS2 theorem. |
| [PSD] | A. Venkatraman, N. Rhinehart, W. Sun, L. Pinto, M. Hebert, B. Boots, K. Kitani, J. A. Bagnell (2017), [Predictive-State Decoders: Encoding the Future into Recurrent Networks](https://arxiv.org/abs/1709.08520), *NeurIPS 2017*. | Eq. 4: an auxiliary decoder predicts future-observation features from recurrent hidden states. Related motivation, but different targets, state construction and optimisation; it does not establish PS2's separate S2. |
| [SPRL] | T. Ni, B. Eysenbach, E. Seyedsalehi, S. Ma, C. Gehring, A. Mahajan, P.-L. Bacon (2024), [Bridging State and History Representations: Understanding Self-Predictive RL](https://arxiv.org/abs/2401.08898), *ICLR 2024*; [version checked](https://arxiv.org/html/2401.08898v3). | Theorem 3's non-collapse argument is restricted to its linear, continuous-time-training setting and other stated assumptions. Detached targets alone are not a general nonlinear non-collapse guarantee; PS2 also uses different losses and interleaving. |
| [ADAM] | D. P. Kingma, J. Ba (2015), [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), *ICLR 2015*. | Algorithm 1 and the suggested default $10^{-3}$. This motivates a convention, not a PS2-optimal learning rate. Finite $\epsilon$ limits exact gradient-scale invariance. |
| [CL-ID] | U. Forssell, L. Ljung (1999), [Closed-loop identification revisited](https://doi.org/10.1016/S0005-1098(99)00022-9), *Automatica* 35(7), 1215–1241. | Background on closed-loop identification. Full text was not independently re-fetched in this revision; attribution is limited to the recorded literature review, not an equation-specific PS2 claim. |
| [CL-IV] | R. A. González, S. Pan, C. R. Rojas, J. S. Welsh (2024), [Consistency analysis of refined instrumental variable methods for continuous-time system identification in closed-loop](https://arxiv.org/abs/2404.08955), *Automatica* 166, 111697; [DOI](https://doi.org/10.1016/j.automatica.2024.111697). | Metadata/abstract verified here; background on conditional consistency of particular closed-loop estimators. PS2 implements neither their estimator nor its assumptions. |
| [IVNN] | J. Kon, M. Heertjes, T. Oomen (2022), [Neural Network Training Using Closed-Loop Data: Hazards and an Instrumental Variable (IVNN) Solution](https://arxiv.org/abs/2202.05337), *IFAC-PapersOnLine* 55(12), 182–187; [DOI](https://doi.org/10.1016/j.ifacol.2022.07.308). | Metadata/abstract verified here; background on closed-loop training hazards and an IV solution. No IVNN step is implemented in PS2. |

Further reading only: A. Hefny, R. Marinho, W. Sun, S. Srinivasa, G. J. Gordon (2018),
[Recurrent Predictive State Policy Networks](https://arxiv.org/abs/1803.01489). Metadata only; no result from this paper
is used as justification of PS2.

**Public implementation versus papers.** [Jan's public repository](https://github.com/JanHHoekstra/Model-Augmentation-Public)
is a separate implementation source. Section 2.3 reports the inspection recorded in D-237, not a freshly verified
commit-pinned reproduction of every public example. Paper statements, public-example choices and this project's
pipeline choices are therefore attributed separately. For reproducibility, future comparisons should record the
public commit and exact example script, as well as this project's code revision.

## 13. Where everything is

| What | Path |
|-|-|
| method code | `model_augmentation/fit_systems/ps2_opening.py` |
| seam | `model_augmentation/fit_systems/interconnect.py` (search `D-236`) |
| pipeline switch | `scripts/gantry/gantry_dynamic/config.py`, `training.py`, `scripts/gantry/gantry_interconnect_dynamic.py` |
| tests, checks, evaluation | `scripts/gantry/thesis-results/test_ps2_pipeline.py`, `check_ps2_mechanism.py`, `inspect_checkpoint.py`, `band_compare.py`, logs in `logs/` |
| server tiers | `scripts/gantry/thesis-results/runs.tsv`, `submit.sh` |
| server logs and checkpoints (interim) | `scripts/gantry/meeting/meeting-08-10-2026/server/` |
| reference checkpoints | `scripts/gantry/meeting/meeting-01-10-2026/server/Checkpoints/` |
| prototype and its experiments | `scripts/gantry/absorber-learning-diagnosis/overnight-clean-method/` |
| decisions | `docs/decisions.md`: D-233, D-234, D-235, D-236, D-237 |
| implementation design | `docs/ps2-implementation.md` |
