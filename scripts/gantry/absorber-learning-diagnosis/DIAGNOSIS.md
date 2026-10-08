# Absorber-learning diagnosis (2026-09-30)

Question: why the added states capture little of the absorber on the thesis data, against the old data.
Predictions and thresholds: `DECISIONS.md` (written before each number). Scripts and outputs: this folder.

## Answer so far
- The added states DO learn the absorber below 230 Hz (run 42: 90 % of the error energy removed in 106 to 140 and 140 to 230 Hz; old 84032: 93 % in 140 to 230 Hz).
- They fail ABOVE the 212 Hz pole (230 to 297 Hz: 22 % removed), where the thesis band carries the largest part of the absorber error; the old band (140 to 230 Hz) never contained it.
- The learned correction is systematic (coherence 0.98 to 1.00) but the wrong shape: a broad, over-damped bump, not the sharp pole.
- Same shape in runs 42 (seed 1, U, 2 states), 48 (seed 1, OBC, 4 states), 44 (seed 2, U, 2 states), within 0.02 in gain: systematic, not seed, capacity or projection.
- The training objective itself prefers the sharp absorber: on the windowed loss the truth scores 3.8x better than the model even with its absorber state unknown (set to zero) at every window start. Training systematically reaches a much worse solution: an optimisation problem, not the objective or the encoder.

## Difference table: old working setup vs thesis setup
| Factor | Old (84032 / 81265) | Thesis (41 to 49) | Outcome |
|-|-|-|-|
| Truth | FP + absorber, no friction | FP + absorber + tanh Coulomb (g 1000) | friction small in absorber band; drives damping drift and route damage |
| Absorber | ma 0.5 mh, fa 150 Hz, pole 212 Hz, zeta 0.03 | same | - |
| Multisine band | 140 to 230 Hz | 106 to 297 Hz, 0.5 Hz grid | the cause, in the form: band extends above the pole |
| Amplitude | A_prod (6x) | A_prod (6x) | - |
| Records | 14 train (T1 to T14), 4 val | 18 train incl. 4 routes, 6 val incl. 2 routes | routes carry friction, not absorber |
| Controller | per-record bank (Y_op) | one K1 (Y 0) | absorber visibility 1.02x: refuted |
| Noise | none | encoder noise (noisy set) | floor < 1.2e-7 m in band: refuted |
| Conditioning | point sampling | anti-alias + 5 ms trim | pipeline reproduces old checkpoint to 5 digits |
| Loss weighting | ystd X 3.2e-2, Y 1.90e-1 | X 8.3e-2, Y 1.81e-1 | Y weight 1.1x: refuted |
| Precision | float32 | float64 | not isolated (update cosine 0.999, entry-file note) |
| States / net | 8 / 3x24 (84032); 2 / 2x16 (81265) | 2 / 2x16 (4, 8 in 48, 49) | states 2 vs 4 not the lever (F2) |
| Parameters | true (81265) or 10 % detuned, joint | 10 % detuned, joint | no delay on old data; masses recover, damping drifts |
| Projection | none / OBC tangent / OBC affine | U / OBC tangent | same learned response (F2) |
| Coordinate | pre-D-229 m_diff | D-229 m_diff | evaluation identical after exact conversion |

## Decision table
| Row | Verdict | Deciding number | File |
|-|-|-|-|
| Too few epochs | refuted as main cause | U curves flat in last quarter (run 42: 1.031e-5 -> 1.027e-5 m); above-pole shape same at 22k and 30k updates | logs 41 to 49; F2 |
| Nothing to learn | refuted | ceiling 1.42e-5 m = 124 % of the na0 plateau | `b_budget_noisy.npz` |
| Band (signal size) | refuted | s_abs 3.6e-5 vs 3.5e-5 m total | B1 |
| Band (above the pole) | supported | 230 to 297 Hz: 2.52e-5 -> 2.23e-5 m (22 %); old band has < 1e-6 m there | E1, F1 |
| Friction masks absorber | refuted in band | s_fr / s_abs 0.01 to 0.03 in 140 to 230 Hz | B1 |
| Noise | refuted | floor 5e-8 to 1.2e-7 m in band | B1 |
| Record mix | refuted for the absorber; explains routes | routes: absorber 1e-7, friction 1e-6 m | B1 |
| Controller | refuted | visibility 1.02x | B1 |
| Parameter substitution | refuted | tangent span 34 % thesis vs 65 % old | C2 |
| OBC projection | refuted | OBC and U give the same response | F2 |
| Capacity (states) | refuted for 2 -> 4 | |H| within 0.02 at 250 to 280 Hz | F2 |
| Seed / detune direction | refuted | run 44 within 0.02 of run 42 | F3 |
| Closed-loop peak 264 Hz | refuted | |S_YY| smooth 1.28 to 1.56, no feature | F1 |
| Pipeline change | refuted | 84032: 5.8588e-6 m reproduced | D1 |
| Windowed objective prefers damped shape | refuted: optimisation problem | training loss: truth with absorber state zero 1.35e-9 vs model 5.09e-9 (0.27x) | G1 |

## Ceiling and captured fraction
- Val metric: thesis ceiling 1.42e-5 m, captured 8 to 9 % (dominated by the 230 to 297 Hz band); old ceiling 2.05e-5 m, captured 72 %.
- Y by band, run 42 (energy removed): 106 to 140 Hz 90 %, 140 to 230 Hz 89 %, 230 to 297 Hz 22 %.
- Learned / true correction, runs 42, 44, 48: |H| 1.0 at 150 Hz, 0.60 at 212, 0.26 at 264, 0.42 at 296 Hz; phase error to +125 to +132 deg.

## Route records (VA-T1, VA-T2)
- Model 4.0e-6 m Y, learned block removed 1.5e-6 to 2.2e-6 m, static part only 6.0e-6 to 9.6e-6 m, baseline at true 1.0e-6 to 1.1e-6 m.
- The static friction / damping correction, fitted on multisine records (errors about 30x larger), generalises badly to routes; the added states partly repair it.
- Damping drift (cg1, cg2, cy, cb_sum +13 to +32 %, also with n_a 0): friction's tanh slope (1.7e4 N s/m at v 0) and viscous damping substitute on this data.

## Proposed next steps
- Test 2 done (G1): optimisation problem.
- A literature session on Hoekstra's framework with the agreed question (claims and sub-questions to be agreed with the user first).
- Band change to 140 to 230 Hz: would hide the defect rather than fix it; last resort.
- Open (?): remaining checkpoint band tables (47, 46, 45, 41, 43, last); critic pass (phase F).

## Mechanism (2026-09-30, phase 2 of the diagnosis)
- Added states realise no resonance in any checkpoint; the class can (member 7x better); the learned static block occupies the absorber's place (correct absorber on top gains 15 % vs 94 % without it); parameter drift does not block. Full statement, evidence and the proposed experiment: `MECHANISM.md`.
