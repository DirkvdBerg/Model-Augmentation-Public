# Run 1 delivery report (saved by the evaluator; the agent could not write files outside its outputs)

## Length
About 2.4 pages estimated (1,850 words of prose, 10 displayed equations, 2 figures) against the 1.55-page budget. Evaluator compile: Section III runs p. 5 to 8 (original p. 5 to 7).
Cut candidates named by the agent: eq:aug_direct_controller to one sentence; eq:aug_lin to the appendix; the two gradient-at-initialisation paragraphs (end of III-A, end of III-B).

## Open points
1. III-A routing: D-103 gives the X/Y reason; nothing records why the position rows are written as well as the velocity rows. Text states the fact only.
2. III-A zero init follows Jan's public code (D-233, D-237); his Sec. 5.4.3 also allows a random linear part. Mention or not?
3. III-A normalisation: state scale comes from measured positions and first differences (data.py::compute_normalization), not a baseline simulation as in Hoekstra Sec. 5.3; no reason recorded. Code comment warns first differences of noisy positions inflate the velocity scale (up to 193x on dTheta); check std_x of the reported runs before relying on Section IV's inner product.
4. III-B the encoder also estimates the measured positions (alternative: Kessels Remark 5.3); no reason recorded.
5. III-B linearisation at Y = 0: no reason recorded (any rest position is an equilibrium; K1 also designed at Y = 0).
6. III-B possible oracle input: gantry_linearize_and_discretize builds W^b from the module's nominal parameters, not the run's start; if nominal equals simulation truth, detuned arms get their encoder from true parameters. Check against the no-system-information rule.
7. III-B W^a: Kaiming uniform in code vs Xavier in Hoekstra Eq. (31) (D-233); report in Setup.
8. III-C full-window scoring: user decision (D-220, burn_in=0) without a recorded reason.
9. III-C Kessels Remark 5.6 noise-bias argument left out (not measured on thesis data).
10. III-C D-142 says x^c_tau = 0 is "exactly Kessels' Remark 5.4"; closed_loop.py:276 says "NOT Remark 5.4". Correct D-142.
11. III-C evidence that open-loop position errors dominate is still a Results item.
12. Notation: Coulomb friction omission is stated in III but only in a todo in II; II calls the output of G "q" while III uses y = q_act = P^T q; u = P u_act now used in II, III, IV, and both III and IV say P is applied inside the physical block (keep one). Renamed: C_n to C_y, r_n to R_n, A_k..D_k to A_fb..D_fb (none used outside III).
13. Figure augmentation_structure labels show past-only histories; the equation includes the current sample with one n. Old layout todo still stands.
14. hoekstra2026encoder is arXiv v1; check for a published IFAC version.

## Moved out
To Setup: controller residual check, 4 kHz re-discretisation bias (D-141), values of n, n_f, validation length, added-state count, network size, Kaiming vs Xavier, parameter starts. Removed: RK4 stage display, MLP layer formula, opening attribution row and contribution sentences (Introduction owns them). Left out: burn-in history, Remark 5.6.
