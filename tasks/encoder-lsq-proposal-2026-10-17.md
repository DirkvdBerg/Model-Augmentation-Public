# Least-squares encoder start in the final pipeline (D-271), revised after review 2026-10-17

## Why
Evidence: D-270 ablation logs, `scripts/gantry/meeting/meeting-19-10-2026/ps2-ablation/`.

| Case | Mean pre-fit MSE (normalised states) | nf val at update 0 | PS2 at update 150 |
|-|-|-|-|
| Production Adam, 20 epochs (D1, E1, final R4) | 1.548e-1 | 1.91e-4 m | fallback |
| Least squares at the detuned theta^0, F1 | 2.1e-8 | 1.894e-5 m | 0.671, pass |
| Least squares, failed solve, F2 | 1.47 (state 4: 8.79) | 2.094e-5 m | fallback |
| True-parameter map, C1 | n/a | 1.892e-5 m | 0.678, pass |

- F1 shows that one truth-free least-squares start was sufficient for PS2 to pass; necessity and robustness
  over seeds, arms and n_a are not shown.
- Neither live g_aug nor the sigma_e weighting prevented success in the tested C configuration.
- Production Adam does not reach the minimum of eq:aug_enc_fit: 0.155 at 20 epochs, 9.1e-3 at 200 (task 10).
- Suspected numerical-rank problem in the default `torch.linalg.lstsq` call: the same intended problem (seed 1,
  blade6) gave 0.155 (task 10) and 2.1e-8 (F1); F2 exceeds the constant-predictor bound in one state. The logs hold
  no singular values, so the mechanism is not proven; the new code logs them.
- F2's output diagnostic looks fine while one state is wrong: the nf val cannot certify the encoder.

## Method (thesis III-B, our adaptation)
A linear restriction of data-based encoder initialisation: W^b and an intercept are the least-squares map of encoder
paper Eq. (33) on the windows of the closed-loop baseline simulation at theta^0 (eq:aug_enc_sim, unchanged). The
physical output rows of psi_tilde start with zero weights and a bias that carries the intercept (a constant output);
the hidden layers keep their U(+-1/sqrt(m)) draw; W^a Xavier and the added-state rows zero (eq:aug_init, unchanged).
All parameters are then trained jointly (eq:aug_loss). No Adam pre-fit.
Appendix clause: "Eq. (33) solved as the numerical-rank-truncated SVD solution in standardised regressors
(LAPACK gelsd, cutoff max(N, m) eps, the PyTorch default)."

## Code
1. `gantry_dynamic/model.py::_prefit`: window construction unchanged; the Adam loop replaced by
   - Phi = [yh.flatten(1), uh.flatten(1)] (the encoder's input order); centre and scale per column (std 0 -> 1);
     centred targets;
   - `torch.linalg.lstsq(Z, x_c, rcond=max(N, m) eps, driver='gelsd')` in float64 on CPU; W = W_s / s,
     b = mean(x) - mu W; singular values from the same call give the condition number and effective rank;
   - W^b set, physical output rows of psi_tilde zero, their bias = b - enc(0, 0)[:nx] (absorbs D-055 offsets);
   - checks (RuntimeError): all inputs, coefficients and residuals finite; encoder predictions equal the solved
     predictions element-wise (relative 1e-6 of the target scale); readiness: every state's MSE <= 1e-3 x its
     population variance in the windows (HEURISTIC: R^2 >= 0.999 per state; F1 about 1e-7, failures 0.04 to 8.8);
     W^a and the added rows unchanged.
2. Rename `encoder_init='data_fit'` to `'data_lsq'` (one name, one meaning; new runs get their own folder
   `thesis_taf_lowpass_noise_data_lsq`, so `discover_runs` never sees the Adam-start runs); evaluation default folder
   and `run_eval.sbatch` comment follow.
3. Remove `encoder_fit_epochs`, `encoder_fit_lr` (config fields, validation, JSON keys) and `_ENCODER_FIT_EPOCHS`
   with the smoke override.
4. Retire `ablate_ps2_fallback.py`: it ran against the code before D-271; a guard refuses to run it; the logs are the
   record (D-270).
5. Final-run logs go to `thesis-results-4` (the Adam-start campaign stays in `thesis-results-3`).

## Tests (criteria stated in advance)
- T1, local, 2 s records, CPU, the production `prefit_physical_encoder` through the pipeline: element-wise
  reproduction, finiteness, readiness check passes; deterministic (two calls give identical W^b); the readiness check
  raises on a deliberately broken target; encoder builds and pre-fit run for n_a = 0, 2, 8.
- T2, server, full data, initialisation only, every distinct pre-fit problem of the final campaign: (detuned, seed 1)
  = R1, R4, R13, R14; (correct, seed 1) = R2, R3, R5, R6, R7; plus (detuned, seed 2) as an independent case; each
  twice. Pass: readiness check passes and both runs give identical W^b (hash).
- Relaunch diagnostics (read, not rules): update-0 nf val, PS2 screen, S2 residual. No run is cancelled on these.

## Not changed
PS2 rules, OBC, loss, weighting, selection, polish, horizons, run table.
