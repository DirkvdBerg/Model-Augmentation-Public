# Run 3 delivery report (saved by the evaluator from the agent's final message)

## Length
Agent estimate 2.7 pages (about 1,850 words, 11 displays, 2 figures). Evaluator compile: Section III runs p. 5 to 7, same as the original, so the estimate was too high.
Cut pass: five "values in Setup" pointers, the D-180 Jacobian-block sentence, an unsourced reason for the held input.
Displays: removed RK4 stages, MLP layer formula, A_d/B_d integral; added eq:aug_norm (normalisation).
Structure: III-A model, III-B encoder and initial state, III-C closed-loop rollout, III-D identification criterion (new label sec:aug_objective).

## Open points
1. Notation: III now uses u_act as model input (as the code); Section IV writes f_base(theta, x, u) with u = P u_act and would need to change.
2. Figure augmentation_structure: past-only encoder labels vs current sample in text and code (na_right = 1); no reference input; cramped.
3. Missing reasons: encoder estimates the measured positions (vs Kessels Remark 5.3); linearisation at Y = 0 (D-167 built a Y-scheduled W^b(Y) that model.py does not use).
4. W^b uses the nominal parameters, not the detuned start; the truth has the nominal parameters, so information about the truth may reach the encoder. Confirm and disclose in Setup.
5. Full-window scoring reason ("the encoded initial-state error stays in V, so the encoder is trained on it") taken from D-178 notes and D-220; confirm.
6. D-142 wording ("IS Kessels' Remark 5.4") vs closed_loop.py:276 ("a definition"); correct D-142.
7. Residual-form exactness assumes the same discrete-time controller; records at 20 kHz, model loop re-discretised at 4 kHz (D-141); single K1 designed at Y = 0 (D-221). State in Setup.
8. Kessels Remark 5.6 not claimed; check applicability now that data are noisy (D-140 rejected it for noiseless data only).
9. No evidence claimed that open-loop position errors dominate on the thesis data.
10. W^a Kaiming vs Xavier; zero output layer follows Jan's code (D-237), paper Sec. 5.4.3 allows a random linear part. Question to Jan open.
11. No bibliography entry for RK4 or the optimiser.
12. Check hoekstra2026encoder (IFAC version) and hoekstra2026lfr (Automatica status).

## Moved to Setup
Added-state count, MLP size, encoder history n, window n_f, stride, validation horizon, RK4 substeps, model rate and controller re-discretisation, optimiser (Adam then L-BFGS), n_f vs closed-loop memory (D-220).
Dropped: six-attribution opening, D-180 Jacobian-block statement.
