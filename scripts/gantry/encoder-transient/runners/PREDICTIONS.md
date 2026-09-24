# Server runs prepared by encoder-transient (ET-008): predictions, written before any server result

None of these is submitted. Launch from one unedited checkout; each script's header has the command.
Local evidence behind each prediction is in `../REPORT.md` (G1, G2, G3) and the smoke rows in `../RUNS.md`.

**Encoder linear map frozen in the G2 arms (ET-013).** The two smoke updates showed ONE Adam step
raising the closed-loop window error 8x (sim) / 30x (Telica), with `grow` collapsing to 0.43 / 0.01,
i.e. an initial-state error. So `g2static`, `g2refresh` and the Telica arm run with W^b, W^a, C_k at
`requires_grad = False` (the correction net trains). `g2refresh` keeps its map consistent by the
per-epoch rebuild. `ENC_TRAIN_LINEAR=1` is the paired unfrozen arm. Prediction for that pair: the
unfrozen arm's first-epoch `[nf val]` rms rises above its epoch-0 value, and its `grow` falls below
0.5; the frozen arm does neither.

## (a) `sim_arm.sh`: does the fix hold in training? (frictionless dataset, production config)
| arm | encoder | burn-in | prediction |
|-|-|-|-|
| `burnin` | production | 100 | reference, as D-178 |
| `g2static` | G2 map (baseline source, deg 2), W^a = 0 | 0 | tracks the production `K = 0` loss within 10 % in the first epochs (G2: the untrained error is model mismatch, 2.0535e-05 vs 2.0548e-05 m); the transient returns once the ANN uses its latents, because nothing rebuilds W^a |
| `g2refresh` | model-Jacobian map, deg 6, all 14 rows, rebuilt every epoch | 0 | `grow` in the `[nf]` lines rises from 0.46 to 0.81 toward >= 0.9 once the ANN has moved; best validation sim-RMS <= the `burnin` arm's at equal epochs |

Falsifier of the fix in training: `g2refresh` keeps `grow < 0.8` after the ANN has moved. Readings:
the per-epoch map does not track the learned latent realisation (try `REFRESH_EVERY` = 1/4 epoch), or
the affine offset at rest (printed on every `[refresh]` line) is not negligible, i.e. the ANN is not
zero at rest and a linear map without a bias cannot place `x0`.

## (b) `r2_ckpt.sh`: R2 against a trained checkpoint
x0 = exact / the checkpoint's encoder / the model-Jacobian map of the trained model.
Prediction: the model-Jacobian map gives Y free-run RMS <= the checkpoint encoder's (H 400 and 4000)
and a lower closed-loop window transient share. If the exact seed is worse than the checkpoint encoder
(D-197 R2 on the untrained model: 1.52e-03 vs 1.43e-04 m on V1), that measures the trained model's
residual Y mismatch, which the encoder is then compensating.

## (c) `telica_arm.sh`: the telica-real production arm with the G4 encoder (ET-012)
The encoder that passed G4 (attempt 3): map from the model the Telica encoder feeds (recovered-block
Jacobians, its tanh friction as input correction, Y schedule deg 3) at `na = 7` (1.6 ms, via
`ET_NA_NB=7`). Measured on Telica (G4): physical-state error 4x to 30x below the production encoder on
every channel at every ypos. Paired reference: telica-real's own `server_run.sh` default run.
Predictions: (1) a lower training loss from the first epoch, because the `x0` of every 40 ms window
is closer to the state (a 40 ms window at 5 kHz is 200 steps and the production encoder's initial
error dominates the first ones, as in simulation); (2) the `[nf]` line's `grow` closer to 1 than in
telica-real's run. (3) NOT predicted: the final validation free run, which is initialised once per
record, so the encoder barely enters it. A worse free run with a better training loss would mean the
ANN had been absorbing the production encoder's initial-state error (the D-197 compensation, in
reverse).
