# Run 6 (v5, Section III) delivery report, saved by the evaluator

Structure: roadmap; III-A model; III-B encoder; III-C closed-loop rollout and objective; III-D initialisation (new label sec:aug_init). About 1,700 words, 10 displays (RK4 stages removed; MLP layer formula and ZOH integral reduced to phrases), 2 figures. Evaluator compile: III on p. 5 to 7, as original.

Open points
1. III-A: Section II defines u = P u_act; Section III uses u_k = u_act (kept; Dirk's decision).
2. III-A: no source for writing the position rows, not only the velocity rows (D-103 covers X/Y authority).
3. III-B: no reason for encoding the measured positions instead of taking them from the measurement (Kessels Remark 5.3).
4. III-D: no reason for linearising at Y = 0; gantry_linearization.py builds W^b from the nominal gss parameters even when theta_base starts detuned (possible truth leak into detuned arms).
5. III-C: full-window scoring supported only by "as in Eq. (22)"; real basis is D-220. Proposal: the encoder is estimated jointly, so the window start is where its error is penalised.
6. III-C: training controller is K1 re-discretised at 4 kHz (+15.3% sigma_max(S_o) at 150 Hz, D-141); "same linear controller" holds only approximately; needs a Setup sentence.
7. Figure augmentation_structure: cramped, no setpoint generator, past-only encoder labels vs current sample in eq:aug_encoder.
8. Bibliography: check hoekstra2025lfr / hoekstra2026lfr metadata, published hoekstra2026encoder; no refs.bib entries for RK4 and Kaiming; Drenth Sec. 5.2 and Kessels citations not re-read this run.
9. D-142 wording ("x^c = 0 is Kessels' Remark 5.4") to correct.
10. Carried over: Results evidence that open-loop position errors dominate; Setup controller residual check at model rate; Kessels Remark 5.6 noise bias.
11. Zero start of the added states matches Jan's public S-DP code (D-237); question to Jan open.
Evaluator note: "We connect the learned part in parallel ... because the orthogonality construction acts on an additive correction" is a reason not traced to a source or decision.

Moved: to Setup n_xa = 2, network 16 x 2 tanh, n_a = n_b = 29, n_f = 0.1 s, window basis (D-220), validation records, 4 kHz bias; to Results the feedback-hides-error test; contributions to the Introduction.
