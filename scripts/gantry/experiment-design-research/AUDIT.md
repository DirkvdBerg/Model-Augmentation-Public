# AUDIT: our datasets against the experiment-design conditions

DRAFT: measured sections final; condition table pending the literature merge.

## Measured quantities (reproducible)

Data: `augmentation_ma50_z03_b140-230_a6_telica` (T records bit-identical to
`augmentation_ma50_b140-230_a6_z03`, ED-004). Scripts in this folder, run under the watchdog:
`audit_signals.py` -> `outputs/audit_signals.json`, `audit_sensitivity.py` ->
`outputs/audit_sensitivity.json`, `audit_crossspectrum.py` -> `outputs/audit_crossspectrum.out`.

### M1. Injected excitation, measured from the records (not from the config)
| family | multisine RMS [f_sym N, f_anti N m, f_Y N] | lines per channel | band | share of `u_total` power in 140 to 230 Hz [sym, anti, Y] |
|-|-|-|-|-|
| standstill T1 to T5 | 240.0, 87.0, 180.0 | 1081 | 140.0 to 230.0 Hz, df = 1/12 Hz | 0.999, 0.999, 0.999 |
| sweep T6 to T8 | same | 1081 | same | 0.94 to 0.999, 0.998 to 0.999, 0.88 to 0.994 |
| APRBS T9 to T12 | same | 1081 | same | 0.25 to 0.88, 0.97 to 0.998, 0.39 to 0.95 |
| Lissajous T13, T14 | same | 1081 | same | 0.64 to 0.85, 0.995, 0.88 to 0.98 |
| Telica-profile TP1 to TP4 | 0, 0, 0 | 0 | none | 0.000 |

- Classical PE order of the injected signal: 2 x 1081 = 2162 per channel, three channels.
- Cross-channel structure (`audit_crossspectrum.out`): the three channels share the same 1081
  lines, so the single-record input spectral matrix `X(k) X(k)^H` is **rank 1 at every line**; summed
  over the band it is well conditioned (cond 1.06 to 1.12, max off-diagonal coherence 0.024 to
  0.049). Union of T1 to T5: cond 1.05.
- No excitation below 140 Hz except the reference motion itself: at standstill 99.9 % of the
  input power sits in the band.
- D-207 defect absent from this frictionless folder (amplitudes match `amp_rms`).

### M2. Parameter informativity: sensitivity `J` of the ten combinations, per family (stride 7)
| group | tuples | rank | cond(J) | collinearity index gamma (Brun Eq. 13) | cos(cg1, cg2) | most collinear pair | weakest direction |
|-|-|-|-|-|-|-|-|
| T1 (Y = -0.30) alone | 6857 | 10 | 9.40e4 | 82.5 | +0.571 | kb_sum / J_eff -0.962 | cb_sum |
| T2 (Y = -0.15) alone | 6857 | 10 | 1.22e5 | 109.8 | +0.673 | kb_sum / J_eff -0.961 | cb_sum |
| T3 (Y = 0) alone | 6857 | 10 | 1.32e5 | 118.4 | +0.686 | kb_sum / J_eff -0.962 | cb_sum |
| T4 (Y = +0.15) alone | 6857 | 10 | 1.15e5 | 102.6 | +0.623 | kb_sum / J_eff -0.961 | cb_sum |
| T5 (Y = +0.30) alone | 6857 | 10 | 8.34e4 | 74.2 | +0.460 | m_total / m_diff -0.962 | cb_sum |
| standstill T1 to T5 | 34285 | 10 | 2.65e4 | 24.5 | +0.575 | kb_sum / J_eff -0.961 | cb_sum |
| sweep T6 to T8 | 20571 | 10 | 4.23e3 | 6.6 | -0.959 | kb_sum / J_eff -0.961 | cg1 |
| APRBS T9 to T12 | 27428 | 10 | 656 | 8.2 | -0.981 | cg1 / cg2 | cg1 |
| Lissajous T13, T14 | 13714 | 10 | 431 | 7.3 | -0.979 | cg1 / cg2 | cg2 |
| Telica-profile TP1 to TP4 | 27428 | 10 | 679 | 10.3 | -0.988 | cg1 / cg2 | cg1 |
| E4 (APRBS, multisine OFF) | 6857 | 10 | 1.05e3 | 10.8 | -0.990 | cg1 / cg2 | cg2 |
| all T (T1 to T14) | 95998 | 10 | 776 | 7.6 | -0.977 | cg1 / cg2 | cg2 |
| all training (T + TP), stride 14 | 123426 | 10 | 703 | 7.8 | -0.980 | cg1 / cg2 | cg2 |

Calibration: all-T at stride 7 gives cond 776 and `rho*` 0.2125 against 766 and 0.2046 at stride 1
in `baseline-and-added-dynamics.md` section 4 (normalisation now over T + TP, ED-004).

### M3. Condition 4 (Gyorok Eq. 17) per family: `rho*` and the implied bias `J^+ Delta*`
Linearised percent (ED-010: nine log coordinates, `m_diff` relative to its own magnitude).
| group | rho* | kb_sum | cg1 | cg2 | cy | cb_sum | mh | m_total | m_diff | J_eff | d |
|-|-|-|-|-|-|-|-|-|-|-|-|
| T1 alone | 0.689 | +62 | +2 | -1 | +1684 | -6 | +111 | -26 | +1959 | -34 | +1 |
| T3 alone | 0.534 | +53 | +8 | -7 | +1190 | -2 | +112 | -26 | +250 | +1 | +1 |
| T5 alone | 0.742 | +80 | +0 | +5 | +1638 | -6 | +111 | -26 | -1455 | -18 | +1 |
| standstill T1 to T5 | 0.168 | +2343 | -127 | +442 | +2502 | -48 | +4 | +4 | +40 | +3 | +1 |
| sweep | 0.175 | +2129 | +10 | +6 | +9 | 0 | +6 | +4 | +47 | +2 | 0 |
| APRBS | 0.349 | 0 | -2 | -2 | -3 | +7 | +16 | +15 | +174 | 0 | -1 |
| Lissajous | 0.213 | 0 | -3 | -2 | +3 | +4 | +6 | +5 | +103 | +1 | -1 |
| Telica-profile | 0.9996 | +5 | +1 | +1 | 0 | +6 | 0 | 0 | +249 | +2 | 0 |
| E4 (multisine off) | 0.999 | -10 | 0 | 0 | 0 | -4 | 0 | 0 | +251 | +1 | 0 |
| all T | 0.2125 | 0 | -4 | -3 | +1 | +8 | +7 | +7 | +104 | +1 | -1 |
| all training (T + TP) | 0.316 | 0 | -1 | -1 | +2 | +7 | +6 | +6 | +159 | +1 | 0 |

### M4. Scheduling coverage (Y) and state-space coverage
- Training Y range: [-0.30, +0.30] m against the enforced stroke +-0.40 m; no training sample with
  |Y| > 0.30 m. Time share per 0.05 m bin over the 18 training records: 0.030 to 0.114 inside
  [-0.35, 0.35], zero outside.
- Frozen points: five (T1 to T5) at -0.30, -0.15, 0, +0.15, +0.30 m; validation at 0.10, test at 0.22.
- Y-rate: RMS 0.25 (T6) to 0.84 m/s (T7, T14); Telica profiles 0.33 to 0.44 m/s. (Y, Ydot) grid
  cells (0.05 m x 0.1 m/s) occupied: standstill 20, sweep 228, APRBS 362, Lissajous 193, Telica
  profile 388; union of test records 445.
- Absorber activation (`delta_a` RMS): 4.4e-5 m with the multisine (99 to 100 % of its power in
  140 to 230 Hz), 1.4e-5 to 1.8e-5 m in the Telica profiles and E4 with **0.000** of its power in
  the band: without the multisine the absorber follows the payload quasi-statically.
- Input-space coverage (HEURISTIC, ED-007): NN distance of val/test tuples to the training tuples
  over the within-training median NN distance, median / p95: V1 0.93 / 1.51, V2 1.93 / 3.74, V3
  1.00 / 1.56, V4 2.32 / 3.34, VP1 0.21 / 1.61, VP2 0.21 / 1.72, E1 0.31 / 0.63, E2 0.94 / 1.19,
  E3 2.02 / 5.15, E4 1.23 / 3.70, EP1 0.21 / 1.77; no tuple beyond 10x in any record.

### M5. Telica real data (from `docs/kamtin-telica-schema.md`, `telica-real/REPORT.md`, no loader run)
- 15 operating points (X start in {-60, -135, -210} mm, Y start in {-200, -120, -40, +40, +120} mm),
  each a fixed 40 mm X / 80 mm Y move; iterations differ only in ILC feedforward; test points
  inside the training grid.
- Reported by telica-real: Theta-row quantities (J_eff, d, kb, cb) "not identifiable from these
  moves (relative SE 0.3 to 1.8)"; viscous terms absorb the Coulomb excess, "not separable with
  this excitation". Repeatable residual power exceeds non-repeatable by 3.1 to 5.9x (G5).
