# Why training does not learn the absorber resonance (2026-09-30)

Handoff `tasks/handoffs/2026-09-30-overdamped-absorber-mechanism.md`. Every number below has its prediction written first in `DECISIONS.md` (section 3); runs in `RUNS.md`; run 42 (U, n_a 2, seed 1) unless stated.

## Mechanism
The added states never become a resonator: in every checkpoint they realise only fast real poles (|z| <= 0.6, no pair near 212 Hz), and the "over-damped bump" is the closed-loop response of a static correction plus one or two samples of memory. The model class can carry the resonance and the objective rewards it (a class member with the 212 Hz pole scores 7x lower), and the pole landscape itself is benign once the added state is driven and read out. What blocks it is competition from the learned static path: from zero init the physical-row correction receives gradient from the first update while the added-state rows receive exactly none until that readout exists, so the static path learns first and imitates the mass-like part of the absorber (below the pole). Once it has, the trained learned block occupies the absorber's place: a complete, correct absorber added on top gains only 15 % (94 % without the learned block), and the damped resonators that gradient descent would pass through while growing a pole raise the loss. The trained point is therefore a basin against the resonator. Parameter drift is not part of the block.

## Evidence
| Claim | Deciding number | File |
|-|-|-|
| No learned resonance | 0 of 672 encoder points with a pole pair in 95 to 380 Hz, runs 42, 44, 48; added poles 0.0007 / -0.60 (42), 0.024 / -0.57 (44), <= 0.36 (48); old 84032 also none (R032, R041 flat FRF) | H1, H1b; `encoder-transient/RUNS.md` |
| Class can represent it | exact member (true combinations + the same 2x16 tanh net as a linear map, error 1.5e-6): pole 211.8 Hz, zeta 0.048 (truth 211.9 Hz, 0.043); training loss 7.0e-10 with the trained encoder vs trained model 5.09e-9 | K1 |
| Objective and window reward it | truth with absorber state zero 1.35e-9 (G1); member with x_a zero 1.65e-9 (K1); window = 21 periods, 5.7 decay times | G1, K1 |
| Pole landscape benign | loss / member: r 0 33.7, r 0.6 20.3 to 21.0, r 0.8 11.3 to 13.4, r 0.9 5.7 at 212 Hz; well about +-40 Hz wide at r 0.9 to 0.95 | L1 |
| Learned block blocks it | full correct absorber grafted: model without learned block 1.21e-8 -> 7.5e-10 (0.06x, monotone); trained model 5.05e-9 -> 4.3e-9 (0.85x); damped graft r 0.6 on trained up to 1.30x | M1c |
| Drift does not block | drifted combinations without learned block: 1.28e-8 -> 7.7e-10 (0.06x), same as true combinations | M1c |
| Static path first | at init dy/dx_a = 0 exactly, gradient on the added path 0.0 (problem log 15.2 item 1); the physical-row path trains from update 1 | problem log 15.1 to 15.2 (measured on the old setup, by construction on this one) |

## Alternatives
| Alternative | Verdict | Deciding number / source |
|-|-|-|
| Data, band size, noise, friction, controller, record mix, weighting, pipeline | refuted | `DIAGNOSIS.md` table |
| Too few epochs | refuted as main cause | curves flat, shape unchanged 22k to 30k updates (F2) |
| Capacity (2 -> 4 states), seed, OBC vs U | refuted | F2, F3; H1b (no pair in any) |
| Representation (class cannot carry it) | refuted | K1 |
| Encoder / window length | refuted | G1; K1 lin_enc 7.0e-10 |
| Narrow well, flat far field (Zucchet and Orvieto 2024 Fig. 1C; Hayes et al. 2023 Fig. 2) | refuted for this system | L1: flat in f only at r <= 0.6 (3.4 %), but a steep radial slope at every f |
| Local damping preference of a mis-tuned sharp mode | refuted | L1: 240 Hz r 0.984 = 5.15 vs r 0.6 = 20.3 |
| Parameter substitution / drift blocks | refuted | M1c: drifted host reaches 7.7e-10; C2 span 34 % |
| Adam step budget (lr 1e-5) as the cause | contributes, not the cause | J1: added output rows used 65 to 70 % of lr N; the pole that moves is the negative real one, toward Nyquist; the gain available at the trained point is only 15 % (M1c) |
| Band was the problem (140 to 230 Hz worked) | refuted as a cause; the old band hid the defect | 84032 has no pole either; `DATA-DESIGN.md` l.14 derives 106 to 297 Hz from the FRF differences |

Literature context (Q1, Q2, Q3 reports): Hoekstra EJC 2025 Table 1 uses the same zero-init feedforward net for the added states; arXiv:2602.17297 Sec. 5.4, Eq. 29 requires a ResNet form (random linear part) and co-estimates the added states in the encoder; trained at Adam lr 1e-3; no pole, damping or FRF check of learned added dynamics in any of their examples. Projection gap found on the way: `obc_gantry.py` builds the reference set with x_a = 0 (l.147), so the orthogonality constrains the static field only; the added-state path is not projected.

Not verified: the ORDER claim (static path first) is by construction and from the old-setup measurement (problem log 15.2), not measured on the thesis runs; the block is measured at the end point (M1c).

## Decisive experiment (for the user to choose; not launched)
**Two-phase training (user's proposal, 2026-09-30).** Phase 1: the physical rows may be corrected only through the added states (readout from x_a; no static function of (x, u) on the physical rows), parameters frozen at the detuned start, added states free (unprojected, as the OBC is today). Phase 2: release the static path and the parameters (U or OBC as in the thesis). Needs a split of the Static_ANN_Block into a state net (x, u, x_a -> x_a+) and a readout net (x_a -> physical rows), in a copy of the pipeline. Switch at the end of phase 1 by the pole check of `h_learned_poles.py`, not by epoch count.
- Prediction (mechanism): without the static competitor, the only way to lower the loss is the added states, so they form a pole pair in the band.
- Pass, phase 1: a pair at 190 to 235 Hz with zeta <= 0.15 at >= 90 % of the encoder points within 10k updates, and training loss <= 5.09e-9 (the trained static solution) with parameters frozen.
- Pass, phase 2: the pair persists (zeta <= 0.15), Y energy removed in 230 to 297 Hz >= 60 % (today 22 %, E1), validation below 1.03e-5 m.
- Fail: no complex pair in the band after 10k updates in phase 1 (added poles stay real and fast): competition is not sufficient, and the bottleneck is the bootstrap speed (second-order start, lr 1e-5 budget, J1) or the parameterisation (arXiv:2602.17297 Eq. 29).
- Risk to watch: with parameters frozen at the detuned start and the added path unprojected, phase 1 may spend the added states on the detuning instead (fast poles, like today); the fail criterion catches it.
- Cost: code, a copy of the model builder with the split block (about half a day); local pre-flight of at most 50 updates (gradient reaches the readout and the added rows; poles printed), about 30 min; server: phase 1 up to 10k updates at the measured 1.72 s per update (run 42: 57471 s for 33.4k updates, validation included) = about 5 h, phase 2 up to 20k updates = about 10 h.

## Update after the phase-1 pre-flight (N1, O1)
- Phase 1 (static net gated off, parameters and encoder training, Jan's routing unchanged, 400 updates at lr 1e-5): the route through the added states opens and carries all the learning (fixed-window loss 1.51e-8 -> 1.22e-8), but the added states learn first as a delayed static map: dominant eigenvalue on the negative real axis (rho 0.0025), as in the trained models (-0.6).
- At that point, growing a resonance is uphill at every tested frequency (loss / L0 at r 0.6: 2.5x; r 0.9: 11 to 23x, least bad at 260 Hz), and the local descent direction for the memory grows the negative real pole, not a pair in the band.
- Revised mechanism: the blocking is not only a separate static path winning the race. Whatever path learns first learns the memoryless (static) part of the absorber correction, and the drive and readout it settles on make a resonance uphill. Removing the separate static path moves the static imitation into the added states; it does not remove it.
- Status of the two-phase experiment: not sufficient on its own at this point; not proposed for the server as is. Caveat: 400 updates only (output weights about 3e-2), and one realisation of the memory perturbation.
