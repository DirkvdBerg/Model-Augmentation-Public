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

2. **Polish acceptance without truth information.**
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
     drop the first residual, so the target covers $k+1$ to $k+M$; widen the sampling bound in `_sample`
     by one. The decoder output size $M n_y$ is unchanged.
   - Check: on one batch, the new target equals the old target shifted by one sample.
   - Thesis: III-E todo; `eq:ps2_s1` then sums from $\tau+1$, and the sentence on matching $V$ goes.

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
- Hidden layers of $\phi_{\mathrm{aug}}$: PyTorch's default `nn.Linear` initialisation, output layer zero
  (`model_augmentation/utils/torch_nets.py::zero_init_feed_forward_nn`, lines 96 to 113).
