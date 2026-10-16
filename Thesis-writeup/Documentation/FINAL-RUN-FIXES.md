# Fixes before the final runs

Code changes and checks found while structuring thesis Section III (2026-10-09). Each fix gets a
`docs/decisions.md` entry before it is implemented, and the matching `\todo` in
`Writing/sections/03_augmentation.tex` is resolved once the rerun exists.

## Code fixes

1. **Encoder map $W^b$ from the start model, not the truth.**
   - Now: `GD/model.py::build_model` (line 207) calls `gantry_linearize_and_discretize(dt=TS_NEW)`, which
     linearises with the nominal module constants of `SYS/gantry_ss.py`. These equal the simulated truth,
     so in the detuned runs the encoder starts from the true parameters while $\theta_{\mathrm{base}}$
     starts 10 % off.
   - Fix: linearise at the start $\theta_{\mathrm{base},0}$ (the detuned combinations). At rest at
     $Y=0$ the linearisation needs only $M(0)$, $C$ and $K$, which depend on $\theta_{\mathrm{base}}$
     alone, so it can be built from the combinations.
   - Check: for a correct-start run the new $W^b$ equals the old one to rounding.
   - Thesis: III-B todo.

2. **Polish acceptance without truth information (D-255).**
   - Now: `FS/lbfgs_polish.py::decide` (line 94) keeps the L-BFGS polish only if the validation error
     improves and no meter worsens by more than 0.1 %. The meters include `combo_err`
     (`GD/training.py::_gantry_meters`, line 652), the error of the ten combinations against the true
     parameters, and `param_loss` (`FS/lbfgs_polish.py::generic_meters`, line 168), the drift from the
     start parameters, if the block exposes it.
   - Fix: accept on the validation error and non-oracle meters only; keep `combo_err` as a logged
     diagnostic. Check whether `param_loss` is active with `param_prior=False`; if so, drop it from the
     acceptance too, since from a detuned start a move toward the truth increases the drift.
   - Thesis: III-D todo.

5. **PS2 S1 target disjoint from the encoder window** (numbered 5 so the references to 1 to 4 stay valid).
   - Now: `FS/ps2_opening.py::_sample` gives the encoder the history up to and including $k$
     (`y[k-na:k+na_r]`, `na_r = 1`) and `_residual` simulates the target over `y[k:k+M]`, so sample $k$ is
     on both sides of the S1 regression. Hefny et al. 2015 (Sec. 2, p. 3) keep history and future
     disjoint so that the history is uncorrelated with the future noise; our records are noisy, so that
     reason applies.
   - Fix: keep the encoder state at $k$, simulate $M+1$ samples from it (`u[k:k+M+1]`, `y[k:k+M+1]`) and
     drop the first residual, so the target covers $k+1$ to $k+M$. The sampling bound in `_sample` needs no
     change: `rng.integers(k0, len(u) - M - 1)` already gives $k \le N-M-2$, so the $M+1$ samples fit
     (checked 2026-10-12). The decoder output size $M n_y$ is unchanged.
   - Check: on one batch, the new target equals the old target shifted by one sample.
   - Thesis: III-E todo; `eq:ps2_s1` then sums from $\tau+1$, and the sentence on matching $V$ goes.
6. **Live start of $g_{\mathrm{aug}}$ (D-248).**
   - Now: the 2026-10-11 runs R1 to R7 use `G_INIT` zero (`zero_init_feed_forward_nn`, final layer all zero), so
     $g_{\mathrm{aug}}=0$ and the added states are zero after the first step.
   - Fix: launch with `THESIS_ARCH=mlp`, `THESIS_INIT=live` (`live_g_feed_forward_nn`): added-state rows of the output
     layer keep nn.Linear's draw, physical rows and biases zero. No code change; config.json must show `G_INIT` live and
     `LIVE_G_WEIGHT_INIT` `kaiming_uniform_a_sqrt5`.
   - Check: initial 400-sample loss equals the baseline's; plain and PS initial parameters bitwise equal (D-240 test);
     no overflow in the first updates on all seeds.
   - Thesis: III-D `eq:aug_init` and its attribution todo; III-E todo on the current runs.
7. **Added-state encoder rows on the centred window.**
   - Now: `FS/pre_encoder.py::linear_encoder_init_aug.forward` adds the D-055 offsets `u_off`, `y_off` before both
     $W^b$ and $W^a$, so the initial added states carry a constant $W^a\chi_0$ (about 3 % of $W^a\chi^{\mathrm n}$
     on the noisy training data, `scratchpad/bb_offset.py`, 2026-10-12). The offset has a reason only for $W^b$:
     Jan's map works in deviations from rest (encoder paper, text after Eq. 32), the model in deviations from
     the data mean, which gives the bias $b^b = W^b\chi_0 - T_x\mu_x$.
   - Fix: apply the offsets to the $W^b$ rows only; $W^a$ acts on the centred window, as Jan's $\psi_{\mathrm{aug}}$
     sees centred data (2026 Sec. 5.3). Physical rows unchanged.
   - Check: physical encoder output bitwise unchanged; added-state output differs by exactly $W^a\chi_0$.
   - Thesis: III-B `eq:aug_enc_lin` (clean form) and its `\revnote`.
8. **Encoder window ends at $\tau-1$ (D-250).**
   - Now: `GD/model.py::get_encoder_dims` sets `na_right = nb_right = 1` (D-054), so the window is $\tau-n..\tau$ and the
     first loss term ($k=\tau$) scores a noisy sample the encoder has seen.
   - Fix: `na_right = nb_right = 0` AND `na = nb = n + 1 = 30`, so the window $\tau-1-n..\tau-1$ keeps the displayed
     $n+1 = 30$ samples (Hoekstra 2026 Eq. 22e with $n_a = n_b = n+1$; changing only the right extensions gives 29); in
     `FS/pre_encoder.py::linear_encoder_init_aug` build $W^b_y = A^{n+1}O_n^+$ and $W^b_u = \bar r_n - A^{n+1}O_n^+\mathcal T_n$,
     $\bar r_n = [B\;AB\cdots A^nB]$ (oldest-first storage reverses the blocks). Check every caller that builds encoder
     windows (`closed_loop_free_run_rms_batch`, PS2 `_sample`, item 5, diagnostics) uses `na_right` from the model.
   - Check: noiseless linear trajectory reconstructs $x_\tau$ to rounding (`scratchpad/wb_shift_check.py`: 6e-17).
   - Thesis: III-B `eq:aug_encoder`, `eq:aug_wb` and the `\revnote` after the encoder where-clause.
9. **One normalisation for encoder and rollout.**
   - Now: `GD/data.py::compute_normalization` stores $\sigma_i + 10^{-8}$, used by the rollout, the output map and the
     D-055 offsets; `GD/model.py::build_model` builds $W^b$ through `normalize_linear_ss_matrices`, which recomputes the
     raw $\sigma_i$ from `norm.x_all`, `u_all`, `y_all`. Relative scale mismatch up to $2\times10^{-5}$, $W^b$ off by
     $4.5\times10^{-6}$ relative (second-audit finding F2, 2026-10-14).
   - Fix: scale $(A_d, B_d, C_d)$ with the stored `norm.std_x`, `norm.std_u`, `norm.ystd`, so the thesis's single
     $T_x, T_u, T_y$ is what the code uses.
   - Check: $W^b$ changes by about $10^{-5}$ relative and no more; physical encoder output matches the rollout scaling.
   - Thesis: none (the equations already state one common $T$).
   - Superseded by item 10 (note 2026-10-14): the data-based fit replaces the model-based $W^b$ that
     `normalize_linear_ss_matrices` scales, so after item 10 this mismatch no longer reaches the final runs. Check
     during item 10 that no remaining caller uses the recomputed scales; no `\revnote` (second Section IV critique
     raised it; the OBC reference set and rollout already share the stored scales).
10. **Data-based initialisation of the physical encoder rows (D-251).**
   - Now: `GD/model.py::build_model` builds $W^b$ from the reconstructability map of the $Y=0$ linearisation
     (`linear_encoder_init_aug`, D-054).
   - Fix: before training, simulate the baseline at $\theta^0_{\mathrm{base}}$ in closed loop (`FS/closed_loop.py` step,
     $f_{\mathrm{aug}}=0$, controller bank of the training records) over every training record; collect windows
     $(\hat y, u_{\mathrm{cl}})$ over $\tau-1-n..\tau-1$ (item 8) and the simulated state at $\tau$, in the centred
     normalised coordinates; fit the PHYSICAL ROWS OF THE WHOLE ENCODER to these states, as Hoekstra 2026 Eq. (30)
     (D-251 amendment, option A): $W^b$, the hidden layers of $\tilde\psi$ and the physical rows of its output layer
     (weights and bias), minimising the squared state error. No separate physical-row bias $b^b$: the encoder is
     encoder paper Eq. (8) as printed, and the output bias of $\tilde\psi$ carries any constant (D-249 amendment). Use a randomly initialised physical encoder
     for this pretraining, as in the nonlinear-baseline method of encoder paper Eq. (35); no linear least-squares
     warm start is adopted. Train the shared hidden layers and physical output rows; keep the added-state output
     rows zero and $W^a$ at its Xavier draw. Retain the fitted physical nonlinear rows and hidden layers at the
     start of joint training; do not reapply the old all-row zero initialisation of $\tilde\psi$.
     Remove the D-055 offsets (items 7, 9 then concern only $W^a$ and the scales).
     Initial state per record: open (D-251); candidate measured positions with zero velocity.
   - Check: pretraining reduces the simulated-state error; $W^a$ is unchanged and the added-state nonlinear output
     remains zero. Encoder histories and targets use the stored normalisation scales and the past-only window.
     The initial augmented transition reproduces the baseline from the same physical state, with $f_{\mathrm{aug}}=0$.
     True-state validation comparisons are simulation-only diagnostics, not a criterion for accepting the encoder
     pretraining. The earlier 0.11-std result belongs to the linear fit, not to this not-yet-run nonlinear fit.
   - Thesis: III-B `eq:aug_enc_fit` and its `\revnote`; III-D `eq:aug_init` must keep the fitted physical nonlinear
     rows and zero only the added-state nonlinear output rows; III-E sentence near line 930.
11. **OBC reference tuples with a successor (D-252).**
   - Now: `GD/obc_gantry.py::build_reference_set` keeps every sample with a full fourth-order stencil,
     `k = np.arange(2, len(u) - 2)`, so the last tuple of each record has no next tuple. The recovery result of
     Section IV (minimiser beta* + Phi^+ eps, independent of the network) needs the coefficient and the discrepancy on
     the SAME stack; with the coefficient on all tuples and eps only on tuples with a successor, a network-dependent
     term remains (critique 2026-10-14: three-row counterexample shifts the minimiser by -9/70 at eps = 0).
   - Fix (amended after the second critique, 2026-10-14): the successor only needs a reconstructible state, not to
     be a reference tuple itself. `k = np.arange(2, len(u) - 3)` AND align the velocities to it:
     `x_ref = np.hstack([q[k], qd[k - 2]])` (`qd` is aligned with `q[2:-2]`; changing `k` alone breaks the
     `hstack`). One tuple fewer per record.
   - Fix, discrepancy diagnostic: the successor target of `eq:obc_eps` is the RECONSTRUCTED next state
     `np.hstack([q[k + 1], qd[k - 1]])`, the same reconstruction as the tuples; store it beside the reference set.
     Now `x_phys_next = x_log[k + 1]` holds the simulator's true next state, and `GD/obc_diagnostics.py::discrepancy_fields`
     uses it for the 'data' discrepancy; keep that as a separate, simulation-only diagnostic.
   - Check: tuple count drops by exactly one per record; Phi of the remaining tuples is bitwise unchanged; rank 10;
     the reconstructed next state of tuple k equals the reconstructed state of tuple k+1 wherever both exist.
   - Thesis: IV-A reference-set `\revnote` (remove after the rerun).
12. **Polish compared with the validation error after the OBC basis rebuild (D-253).**
   - Now: `FS/interconnect.py::fit` restores the best checkpoint and rebuilds the OBC basis at its beta
     (`_sync_prediction_state_for_validation(refresh_basis=True)`), which changes the corrected model;
     `GD/training.py::run_lbfgs_polish` then passes `val_before=bestfit`, measured BEFORE the rebuild, so
     `FS/lbfgs_polish.py::lbfgs_polish` compares the polish with a different model.
   - Fix: when `cfg.obc` is on, pass `val_before=None`, so `lbfgs_polish` measures the validation error of the rebuilt
     model (`_validate()`) before polishing. Cost: one closed-loop free run (about 162 s). No change without OBC.
   - Check: for an OBC run the logged `val_before` differs from `bestfit` and equals a fresh validation of the
     restored, rebuilt model; for a U run it is unchanged.
   - Thesis: IV-B `\revnote` on the comparison score (remove after the rerun).

Items 13 to 18: error relative to the servo error, $\mathrm{NRMS}_{e,j}=\mathrm{RMS}(\hat y_j-y_j)/\mathrm{RMS}(y_j-r_j)$
per record and stage axis $j$ (D-256, Dirk 2026-10-15). Reason in one line: the reference is given, so the servo error
$e=y-r$ is what a closed-loop model must predict; RMS in m hid that the final model is at 3 to 8 % on every test record
and axis while the baseline is above 100 % on Y on the multisine records.

13. **Log validation RMS per record and axis at every validation.**
   - Now: `FS/closed_loop.py::ClosedLoopSimulator.validation_error` (line 649) keeps `self.last_per_record` but the
     run saves only the pooled score per validation (`loss_val` in `gantry_results_<job>.npz`) and per-record values only
     for the final model (`cl_val_rms`). Re-ranking checkpoints by another rule afterwards is therefore impossible
     (tried 2026-10-15 on the 2026-10-11 runs).
   - Fix: at every validation append the per-record, per-axis RMS in m (6 records x 3 axes) and RMS(y - r) of each
     validation record to the run's results npz (e.g. `val_rms_hist` of shape (n_val, 6, 3)). Same for the black box
     (`scripts/gantry/blackbox-cl-trainable/train_bb_clmap.py`).
   - Check: the last entry equals `cl_val_rms` of the selected model when no polish is accepted.
14. **Checkpoint selection by NRMS_e.**
   - Now: `FS/closed_loop.py::ClosedLoopSimulator.validation_error` returns the record-length-weighted mean of the
     per-record RMS in m (`np.average(per_record, weights=len(sd.u))`, line 689).
   - Fix: return the mean over the 6 validation records and 3 axes of RMS(y_hat_j - y_j) / RMS(y_j - r_j), both from
     k0; RMS(y - r) per record and axis computed once from the validation data (`r_sim` decimated as y). Keep the old
     value in the log for comparison.
   - Check: on a saved checkpoint the new score equals the mean of `cl_val_rms / RMS(e)` computed offline.
   - Thesis: III-D selection sentence (to update, Dirk to approve).
15. **Loss weighted per training record and axis.**
   - Now: `FS/interconnect.py::SSE_Interconnect_Composed.loss` (line 1418): `mse_loss(yfuture, y_pred)` on the
     normalised output, so each axis is weighted by 1/std of the training POSITION.
   - Fix: multiply the error of each window by the fixed weight $w_{\rho,j}=\sigma_{y,j}/\mathrm{RMS}(y_j-r_j)$ of its
     record $\rho$ and axis $j$ (the $\sigma_{y,j}$ undoes the position normalisation of the error), computed once from
     the training records (noisy data, no oracle) and carried with each window. Not per window (zero in standstill).
   - Check: with all weights equal to one the loss is bitwise the old one; on one batch the weighted loss equals the
     offline mean of $(w(\hat y-y))^2$.
   - Thesis: III-D `eq:aug_loss` gets the weight (to update, Dirk to approve).
16. **L-BFGS tolerance for the new loss scale.**
   - Now: `FS/lbfgs_polish.py` `tol_grad = 1e-12` (line 62) is absolute and commented "the loss here is of order 1e-9";
     the weighted loss is of a different order.
   - Fix: measure the opening loss and gradient norm of the weighted loss on the pilot and set `tol_grad` relative to
     them (same ratio as now), or confirm that 1e-12 never triggers before the 100 iterations.
17. **Polish acceptance on NRMS_e.** `FS/lbfgs_polish.py::decide` (line 94) and `GD/training.py::run_lbfgs_polish` compare
   validation errors; they must use the score of item 14 (with OBC: after the basis rebuild, item 12; D-255).
18. **Black box on the same loss and selection.** `scripts/gantry/blackbox-cl-trainable/train_bb_clmap.py` uses deepSI's
   `fit` with `validation_measure='sim-RMS'` (line 146) and the unweighted window MSE. Give it the weights of item 15 and
   the selection of item 14 (custom loss and validation hook), and the selection rule in
   `runners/thesis_r8_grid.conf` ("lowest best-validation sim-RMS") becomes NRMS_e. Without this the comparison is not
   matched.
19. **OBC basis with the offset column $\Gamma$ (D-257).** The 2026-10-11 runs protect $\operatorname{ran}\Phi$ only
   (D-254). The final OBC runs protect $\operatorname{span}[\Phi\ \Gamma]$, with
   $\Gamma=F_{\mathrm{base}}(\beta_{\mathrm{lin}})-\Phi\beta_{\mathrm{lin}}$ stacked over the same tuples (Gyorok 2025
   Eq. 18), so $\theta_{\mathrm{aux}}\in\mathbb R^{11}$ and every closed-loop step subtracts $[\varphi\ \gamma]\theta_{\mathrm{aux}}$.
   Code location not yet chosen: check first whether the earlier affine variant (D-186, D-190, D-192) is still in
   `obc_gantry.py` / `obc.py` and reuse it. Check after the change: $[\Phi\ \Gamma]^\top\tilde F=0$ to numerical
   precision on the tuples. Evaluation: add the negation measure $\cos\angle(F_{\mathrm{aug}},F_{\mathrm{base}}-\tilde X)$
   (baseline one-step increment) on the held-out I1 to I6 tuples to `Code/Evaluation/eval_negation.py`.

## Checks before launch

3. **D-232 leak check on the lowpass data.** `scripts/gantry/anti-alias/measure_aliasing.py` (line 40) still
   reads `Coulomb-tanh-and-MSD`; rerun it on `Coulomb-tanh-and-MSD-lowpass-noise` (open in
   `NOISE-INJECTION.md` Sec. 10.6). Reference: 0.2 to 0.4 nm leak with the filter on the previous noise.
   Done 2026-10-09 (`measure_aliasing_lowpass.py`/`.log`): Q5 leak with the filter 0.22 to 0.50 nm (4 to 10 % of
   the in-band noise; point sampling 1.3 to 2.8 nm); the order-8 Butterworth at 1.6 kHz still suffices, no change.
4. **Stale `pair` field.** The noise-free files' `pair` field still names the old noisy folder
   (`NOISE-INJECTION.md` Sec. 10.6). Harmless unless a script follows it.

## Checked, no change

- PS2 S2 regresses $g_{\mathrm{aug}}$ with the recorded $u_k$ (`FS/ps2_opening.py::_fit_map`). Correct: it is
  the force that took the machine from $k$ to $k+1$, and it equals the closed-loop input
  $u_k + D_{\mathrm{fb}}(y_k-\hat y_k)$ when the encoder reproduces $y_k$. To be specified in III-E.
- Hidden layers of $\phi_{\mathrm{aug}}$: PyTorch's default `nn.Linear` initialisation. Output layer all zero in the
  2026-10-11 runs (`model_augmentation/utils/torch_nets.py::zero_init_feed_forward_nn`, lines 96 to 113); in the final
  runs only its physical rows and biases are zero (`live_g_feed_forward_nn`, item 6, D-248).
