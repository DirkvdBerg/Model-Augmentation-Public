# Why the added states did not learn the absorber, and the remedy that works (updated 2026-10-02)

Handoff: `tasks/handoffs/2026-10-01-overnight-added-states-remedies.md`. Every number below has its prediction written first in `DECISIONS.md` section 4 (ids AA1, AB1, E4 to E9, AC1, AD1). Constraints respected throughout: Jan's `Static_ANN_Block` with its zero-init MLP output, routing, encoder method and baseline-at-start unchanged; no burn-in; no initialisation, parameterisation, band, pole or system information; only objective (temporary terms), schedule and optimiser rates change, all temporary, and the run ends on the pure thesis loss.

## 1. Cause

Three gates in series, each measured:

1. **Gradient gate (known before tonight).** With the zero-init output, the pole direction of the added states gets no first-order output-error gradient until drive and readout exist (Steiglitz-McBride 1965, App. II Eqs. 6 to 8). Plain training therefore buys static maps and real lags (U1, U3, run 42: dominant added pole +0.60, no in-band pair at any of 672 points). Optimiser changes alone (U1, U4, L1) do not change the direction.
2. **Target gate.** A temporary equation-error (EE) term opens the gradient path, but it can only teach what the encoder's x_a contains. A random encoder W^a (Hoekstra 2026 Eq. 8) gives slow, baseline-like content (E1, E2: real lags). The innovation head (x_a(k) linearly predicting the model's next 40 output errors) supplies the right content without system information. AA1 at E3's end: x_a is 90 % in 150 to 300 Hz, its own EE optimum is a 224 Hz pair at |z| 0.93 (IV agrees: no noise bias), and the innovation's optimal 2-d state is a 232 Hz pair with zeta 0.05, i.e. the absorber seen through the closed loop.
3. **Fit gate (E3's radius gap).** E3's state map never fitted that target: EE residual 0.68 of the x_a variance against a linear optimum of 0.002, A_aa radius 0.21 to 0.25 (AA1, AB1). With the thesis MLP's hidden layers fixed and the added-row output at 1e-3, the EE fit needs over 3000 updates and reaches the band at only about half the points (fixed features). With the hidden layers at 1e-3 and the added-row output at 1e-2 it reaches the linear optimum and a 217 to 224 Hz pair at 96 to 100 % of points within 500 updates (AB1).

Plus an **order** condition: making the state map fast before the innovation term has shaped x_a (E4) locks it onto the early, slow content of the random encoder (dominant real pole 1.0 to 1.3, a loss spike to 4.9e-8 at update 200, final in-band pair zeta 0.72).

So the defended cause: the added states did not learn the absorber because nothing in the thesis objective gave them a target with the mode before the pole gradient opened (gates 1 and 2), and in E3, which supplied that target, the zero-init MLP could not fit it within the opening's step budget (gate 3). Neither the model class nor the objective was the problem (G1, K1).

Corrections of earlier readings: E3's gap was not a duration effect of the terms alone and not a noise or symmetry cap of the EE target (AA1: LS = IV; the target is a lightly damped pair); it was the state map's fit budget plus order.

## 2. The remedy that works locally: the ordered innovation opening (E5, E9)

Zero start, thesis model unchanged, batch 64. Best form (E9):
- updates 1 to 500: thesis loss + innovation term only (encoder W^a and head at 1e-3, everything else at E3's rates);
- 501 to 800: + EE on the added rows (encoder states detached), with the shared hidden layers at 1e-3 and the added-row output at 1e-2 (opening-phase rates only);
- 800 to 950: both terms decay linearly to 0;
- after 950: pure thesis loss (added-row output and W^a 1e-3, rest 1e-5).
E5 is the same with the phases at 300 / 600 / 750 (900 updates).

| Measure | U1 control | E3 | E4 | **E5** | E6 (seed 2) | E8 (seed 3) | **E9** (seed 2, 500) | Truth |
|-|-|-|-|-|-|-|-|-|
| eval loss, 72 train windows | 6.91e-9 | 7.05e-9 | 6.31e-9 | **1.16e-9** | 3.19e-9 | 2.14e-9 | **1.18e-9** | G1 1.35e-9, K1 7.0e-10 |
| added in-band pair, share of points | 0 % | 46 % | 46 % | **96 %** | 71 % | 79 % | **100 %** | |
| pair frequency, zeta | none | 117 Hz, 0.99 | 180 Hz, 0.72 | **208 Hz, 0.00** | 222 Hz, 0.05 | 205 Hz, 0.33 | **209 Hz, 0.09** | 211.9 Hz, 0.043 |
| val MSE on / off (0.1 s windows) | 0.441 | 0.441 | 0.422 | **0.077** | 0.236 | 0.142 | **0.078** | |
| Y error removed 230-297 Hz (val, proxy) | 40 % | 41 % | 37 % | **93 %** | 60 % | 82 % | **92 %** | |
| whole-record free-run sim-RMS (m) | 1.15e-5 | | | **3.80e-6** | 9.05e-6 (self-oscillates) | 6.03e-6 | **3.79e-6** | block-off 1.67e-5 |
| physical-pole shift, max combination change | 5.4e-4, 0.5 % | 5.1e-4, 0.5 % | 2.9e-4, 0.4 % | 5.0e-4, 0.7 % | 8.7e-4, 0.8 % | 1.0e-3, 0.8 % | 8.9e-4, 0.9 % | |

Robustness: with a 300-update innovation phase, 1 of 3 seeds is complete (E5); the other two learn an in-band pair at 71 to 79 % of points, and one of them (E6) self-oscillates on the quiet validation records. With 500 updates the weak draw becomes complete (E9). The innovation phase length is the robustness knob: make the state map fast only once the encoder target has formed. Damping: on the pure thesis loss the learned pair drifts toward marginal (E5 0.06 -> 0.00; E7, 600 more updates, |z| 1.000) but stays stable over 12 s whole-record free runs (AD1).

## 3. Ranked remedies (allowed families only)

1. **Ordered innovation opening with opening-phase rates (E9 form).** Demonstrated locally from the zero start: complete in 2 of 4 runs, E5 (seed 1, 300-update innovation phase) and E9 (seed 2, 500); with 300 updates seeds 2 and 3 were partial (E6, E8), and 500 made seed 2 complete (E9); 500 on seeds 1 and 3 untested; mechanism measured at each gate (AA1, AB1, E4, E5 to E9, AD1). Uses no system information: the innovation target is the model's own output error, and the switch points are update counts. Literature analogues: free or encoded latent targets before map fitting (Ouala et al. 2020, Chaos 30:103121, Eq. 6; Otto and Rowley 2019, SIADS 18(1), Eq. 25); equation error before output error (Shynk 1989, Eqs. 3 to 5). No paper found reports a learned lightly damped latent mode from input-driven, closed-loop, noisy data (provisional; `literature/added-states-training/`).
2. **E3 as prepared for the server (unordered, E3 rates).** Expected to reproduce E3's gap (radius about 0.25) unless its 2000-update opening at batch 512 compensates for the fit budget; AB1 says the output layer alone reaches the band at only about half the points. It is already submitted (job 87446); read it as the control for item 1.
3. **Noise-robust EE (IV, reference instruments).** Not needed on the present data (AA1: IV = LS); keep it for real data, where the innovation target is regressed on measured outputs in closed loop (Forssell and Ljung 1999 projection, Eqs. 76 to 79; Gonzalez et al. 2024 Automatica 166:111697 Eq. 7; Kon et al. 2022, IFAC-PapersOnLine 55(12), Eqs. 16 and 17).
4. Closed: optimiser-only changes (U1, U4), early or late LM (L1, Z1), EE without the innovation target (E1, E2), multi-horizon short-first (M1, inconclusive). Multi-rate remains untested and incompatible with the unchanged discrete block as argued before.

## 4. Open points

- The OBC arm (not tested locally; needs the pipeline's epoch basis refresh).
- A data-derived switch point (for example the innovation head's unexplained fraction levelling off) instead of fixed counts; 300 was too early for 2 of 3 seeds, 500 was enough for the one tested.
- Transfer to the server settings: batch 512, 33k updates, best-checkpoint selection. The opening is defined in update counts at batch 64 locally.
- The learned pair's damping drifts low under the pure thesis loss (above).
- Real data: the innovation target in closed loop carries closed-loop poles and noise correlated with the encoder's measured inputs. The final pure thesis phase maps the mode back to the plant; it worked here on noisy simulated data.

## 5. Recommended next server experiment

See `DECISIONS.md` section 4, morning report.
