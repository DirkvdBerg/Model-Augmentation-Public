# Run 2 delivery report (saved by the evaluator from the agent's final message)

## Length
About 1,650 source words, 7 displays (11 before), 2 figures; agent estimate 2.1 pages. Evaluator compile: Section III runs p. 5 to 7, same as the original.
Cut pass by the agent: Kessels "network weights only" comparison, a redundant unmeasured-states sentence, two init-stability sentences merged, "exact from noise-free histories", "needs neither r nor u_ff", Drenth "realised on the gantry", a duplicate Eq. (5.12) citation.

## Open points
1. III-A routing: the two reasons for writing all six physical rows (motor forces act on every axis; a held force moves the position within a sample) are the agent's own derivation; D-103 is a supervisor directive. Confirm.
2. III-B: no recorded reason why the encoder estimates the measured positions instead of taking them from the measurement (Kessels Remark 5.3).
3. III-D: no recorded reason for linearising at Y = 0.
4. III-D possible information leak: gantry_linearization.py builds W^b from the nominal module constants, also in detuned-start runs; per DATA-DESIGN.md the nominal values are the physical part of the simulated truth. Disclose in Setup or build W^b from the detuned start.
5. III-C: no technical reason recorded for full-window scoring (burn-in excluded by D-220).
6. III-D: Hoekstra 2026 Sec. 5.4.3 allows a random linear part; we zero the whole output layer (D-233/D-237, Jan's public code).
7. Setup: W^a Kaiming uniform in code vs Xavier in Hoekstra Eq. (31).
8. figures/tex/augmentation_structure.tex shows past-only encoder histories; eq:aug_encoder includes the current sample. Old caption todos remain.
9. Notation: u = P u_act as in II and IV; controller output u-hat_act; output map h_base as in the figure. Check.
10. Kessels Remark 5.6 (noise bias of open-loop identification on closed-loop data) left out; applicability to the noisy thesis data unchecked.
11. III-C: open-loop drift argument is structural; measured evidence is a Results item.
12. References: hoekstra2026encoder published version; D-142 wording ("x^c = 0 IS Remark 5.4") to correct; "self-scheduled" dropped from the Drenth sentence (Sec. 5.2 not checked for it).

## Moved to Setup or dropped
To Setup (Section V does not yet carry all): added-state count, network size, encoder lag 29, window n_f = 0.1 s (D-220), stride, 4 kHz, controller re-discretisation bias (D-141), W^a distribution, controller residual check, Adam and L-BFGS schedule.
Dropped as displays: RK4 stages, MLP layer formula, zero-init display, W^b formula, ZOH integral, second line of the recorded-loop equations.
Left out as mechanics or history: gradient path to added states after zero init, Kaiming vs Xavier, earlier Theta-only routing.
