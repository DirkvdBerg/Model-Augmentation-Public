# Downsampling findings

Dataset: `augmentation_ma50_a5` (22 records).  Master rate: 20000 Hz.
The 20 kHz oracle/data maximum gate errors were no worse than
`5.561e-08` in stage coordinates.

The decision ratio is the exact-oracle rate floor divided by the six-state
FP-to-eight-state-truth discrepancy.  `0.10` and `0.25` are diagnostic guide
lines, not statistical confidence bounds.

## Open loop: exact-state 100 ms windows

| domain | rate | oracle RMS | FP RMS | RSS ratio | worst-record ratio |
|---|---:|---:|---:|---:|---:|
| band | 4000 | 4.322e-08 | 7.299e-06 | 0.006 | 0.009 |
| band | 2000 | 1.831e-07 | 7.480e-06 | 0.024 | 0.026 |
| band | 1000 | 8.805e-07 | 7.314e-06 | 0.120 | 0.127 |
| time | 4000 | 8.039e-08 | 5.102e-04 | 0.000 | 0.000 |
| time | 2000 | 3.291e-07 | 4.979e-04 | 0.001 | 0.001 |
| time | 1000 | 1.418e-06 | 5.044e-04 | 0.003 | 0.004 |

## Open loop: full-record free run

| domain | rate | oracle RMS | FP RMS | RSS ratio | worst-record ratio |
|---|---:|---:|---:|---:|---:|
| band | 4000 | 4.821e-08 | 8.130e-06 | 0.006 | 0.014 |
| band | 2000 | 1.966e-07 | 8.025e-06 | 0.024 | 0.027 |
| band | 1000 | 9.268e-07 | 7.621e-06 | 0.122 | 0.127 |
| time | 4000 | 2.304e-06 | 2.870e-03 | 0.001 | 0.285 |
| time | 2000 | 9.364e-06 | 2.867e-03 | 0.003 | 1.089 |
| time | 1000 | 3.556e-05 | 2.859e-03 | 0.012 | 1.892 |

## Current co-rate controller: exact-state 100 ms windows

| domain | rate | oracle RMS | FP RMS | RSS ratio | worst-record ratio |
|---|---:|---:|---:|---:|---:|
| band | 4000 | 7.452e-08 | 1.239e-05 | 0.006 | 0.007 |
| band | 2000 | 3.802e-07 | 1.434e-05 | 0.027 | 0.029 |
| band | 1000 | 3.052e-06 | 1.835e-05 | 0.166 | 0.181 |
| time | 4000 | 8.946e-08 | 1.517e-05 | 0.006 | 0.006 |
| time | 2000 | 4.366e-07 | 1.676e-05 | 0.026 | 0.028 |
| time | 1000 | 1.221e-05 | 2.068e-05 | 0.591 | 0.614 |

## Current co-rate controller: full-record free run

| domain | rate | oracle RMS | FP RMS | RSS ratio | worst-record ratio |
|---|---:|---:|---:|---:|---:|
| band | 4000 | 8.400e-08 | 1.398e-05 | 0.006 | 0.007 |
| band | 2000 | 4.123e-07 | 1.554e-05 | 0.027 | 0.029 |
| band | 1000 | inf | 1.940e-05 | inf | inf |
| time | 4000 | 8.986e-08 | 1.498e-05 | 0.006 | 0.006 |
| time | 2000 | 4.398e-07 | 1.662e-05 | 0.026 | 0.029 |
| time | 1000 | inf | 2.064e-05 | inf | inf |

## 20 kHz controller with held low-rate output: exact-state 100 ms windows

| domain | rate | oracle RMS | FP RMS | RSS ratio | worst-record ratio |
|---|---:|---:|---:|---:|---:|
| band | 4000 | 1.133e-06 | 1.259e-05 | 0.090 | 0.104 |
| band | 2000 | 3.056e-06 | 1.505e-05 | 0.203 | 0.231 |
| band | 1000 | 9.765e-06 | 2.246e-05 | 0.435 | 0.485 |
| time | 4000 | 2.443e-05 | 2.885e-05 | 0.847 | 0.927 |
| time | 2000 | 5.507e-05 | 5.770e-05 | 0.954 | 0.980 |
| time | 1000 | 1.178e-04 | 1.194e-04 | 0.986 | 0.997 |

## 20 kHz controller with held low-rate output: full-record free run

| domain | rate | oracle RMS | FP RMS | RSS ratio | worst-record ratio |
|---|---:|---:|---:|---:|---:|
| band | 4000 | 1.232e-06 | 1.421e-05 | 0.087 | 0.096 |
| band | 2000 | 3.219e-06 | 1.629e-05 | 0.198 | 0.218 |
| band | 1000 | 1.029e-05 | 2.373e-05 | 0.434 | 0.478 |
| time | 4000 | 2.405e-05 | 2.843e-05 | 0.846 | 0.927 |
| time | 2000 | 5.413e-05 | 5.675e-05 | 0.954 | 0.980 |
| time | 1000 | 1.145e-04 | 1.168e-04 | 0.981 | 0.992 |

## Numerical verdict

- Open-loop 2 kHz preserves the 130--180 Hz learning target: its exact-oracle
  100 ms floor is `0.024` of the FP discrepancy.  At 1 kHz this is
  `0.120`, above the 10% guide line.
- With the current co-rate controller, the 100 ms time-domain ratios are
  `0.026` at 2 kHz and `0.591` at 1 kHz.  The exact-oracle
  arm diverged in every 1 kHz full-record co-rate run, while the six-state FP
  arm remained finite.  Consequently no paired full-run ratio exists there.
- The naive 20 kHz-controller/ZOH-model interface is already dominated by its
  intersample convention at 4 kHz (`0.847` time-domain ratio).
  It is therefore rejected as an implementation, not used to reject keeping
  the real controller at 20 kHz.  A useful 20 kHz-controller design needs a
  high-rate model-output reconstruction rather than a hold.

### Co-rate 1 kHz full-run divergence

| record | exact oracle | FP baseline |
|---|---:|---:|
| E1_resonance_sweep | 1.123 s | none |
| V1_standstill_Yp10 | 0.366 s | none |
| V2_aprbs_Ylow | 0.356 s | none |
| V3_ysweep_Yp10 | 0.397 s | none |
| V4_lissajous_Ym10 | 0.359 s | none |

Correction-force and total applied-force RMS/peak values for every arm are in
`controller_force_summary.csv` and the unflattened `controller.json`.

## Interpretation constraints

- Open loop is the primary sampling-rate gate: the recorded force already
  contains the action of the original 20 kHz controller.
- `corate` is the current training implementation and changes the discrete
  controller with model rate.
- `controller20k_zoh` keeps the controller at 20 kHz but necessarily chooses an
  intersample model-output convention.  Its zero-order hold is deliberately
  explicit; a different multirate interface can produce different numbers.
- No encoder and no ANN are used.  Every window begins from the reconstructed
  exact eight-state truth, so these numbers are a numerical/data-conditioning
  floor rather than an estimation result.
