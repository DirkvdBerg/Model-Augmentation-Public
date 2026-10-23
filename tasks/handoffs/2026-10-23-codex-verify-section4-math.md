# Handoff (Codex): verify the math of thesis Section IV against derivation, papers and code
**From**: Claude session of 2026-10-22/23 (Section IV cut pass and IV-A/IV-B walk) | **Branch**: Augmentation | **Effort suggested**: xhigh (derivations by hand, index and dimension checks, code reading; no runs)

## 1. Task
Verify, read-only, every equation and every mathematical claim in `\section{Dynamic Augmentation of the Physics Baseline}` (`Thesis-writeup/Writing/sections/03_augmentation.tex`, from the `\section` line to the end), in the appendix subsection `PS details` (`99_appendix.tex`, label `app:ps_stop`) and in the residual-form paragraph of `99_appendix.tex` subsection `Processing and checks` (label `app:setup_checks`, the paragraph starting "The residual form"). For each claim decide: correct, wrong, or correct only under a condition the text does not state. Check it three ways where they apply: (a) derive it by hand from the equations it depends on, (b) compare it with the cited paper passage, (c) compare it with the implementation in `Clean/`. Also verify the proposed text in 4.3, which the author has not yet applied. Report in your reply (format 4.4). Edit nothing.

## 2. Out of scope
- Any file edit, LaTeX build (thesis or figure), git write, or numerical run. If a claim can only be settled numerically, say so and say which quantity would settle it.
- Wording and style: the author handles those. Report wording only if it makes a statement mathematically wrong or ambiguous.
- Sections I to III and V onwards, except to check a symbol or equation that Section IV uses (`eq:system`, `eq:system_loop`, `eq:Ptransform`, `eq:eom`, `eq:base_ss`, `eq:phi`, `eq:matrices`/`eq:stiffness_matrix`, `eq:lfr_admissibility`, `eq:mdelta_map`).
- Re-arguing decisions in 4.2 unless they make the math wrong.
- `kamtin-data/`, `kamtin-fp-model/`.

## 3. Where things stand
Branch `Augmentation`, last commit `4df8faa` "section 4-A". Uncommitted edits in `03_augmentation.tex` (IV-B walk), `99_appendix.tex`, `figures/tex/ps2_windows.tex`, `Code/Figures/closed_loop_same_reference.tikz.tex`. The author may be editing `03_augmentation.tex`; read it once and review that snapshot. Change marks: read the source in its final reading as described in `tasks/handoffs/2026-10-22-codex-review-section4.md` section 4.1 (`\chg{old}{new}` = new, `\del` = gone, `\mv` = present, `\delmv` = gone, `\todo`/`\revnote`/`\ifdraft` = not printed). Equations carry no marks; superseded equation versions are kept as `%` comments above them and are not part of the text.

## 4. Established and verified

### 4.1 Claims to check (labels as in the file; equation numbers in the PDF differ)
Augmented model (IV-A)
1. `eq:aug_transition` (14) is the dynamic parallel structure of Hoekstra 2026 Table 1 (S-DP), with h_aug = 0.
2. `eq:aug_norm` (16) and its where-clause: means and standard deviations from D_N; positions converted by q = P^{-T} y, velocities by finite difference divided by T_s.
3. `eq:aug_norm_maps` (17): rows 1 to 3 are (14) in the coordinates of (16); row 4 defines f^n_base = T_x(f_base(theta, x, u_act) - mu_x) with row 5 the inverse map. Claims in the text around it: f^n_aug = T_x f_aug (no mu_x); g^n_aug = g_aug because x-bar is not normalised; C^n x^n has no offset because the position means equal P^{-T} mu_y; "we include the means ... since we do not assume zero-mean signals as in [Hoekstra Sec. 5.3]"; the affine version is exact for any f_base (incl. M(Y)).
4. "f_base is one RK4 step of F over T_s with the input u = P u_act held constant"; for each of the four RK4 sub-steps M(Y) uses Y of that stage's state.
5. The phi_aug display: the network maps (x-hat^n_k, u^n_k) to (f^n_aug, g^n_aug).

Closed-loop simulation (IV-B)
6. "The linearised RK4 step has a pole at z = 1 on each of [the X and Y] axes, making it marginally stable: stable, but not asymptotically stable." Check against K in `eq:stiffness_matrix` (X and Y rows zero), C (damping present), the RK4 stability function R(0) = 1, and simplicity of the eigenvalue (Lyapunov stable, not a Jordan block). State for which Y it holds.
7. `eq:aug_two_loops` (18) and the step "subtract (2) from (18) to cancel r and u_ff", x^c = x-hat^fb - x^fb driven by y - y-hat, x^c_tau = 0 sets the model controller state equal to the system's at tau, giving `eq:aug_residual_loop` (19), "which only needs the recorded u_act and y". Check the sign conventions and the D_fb terms.
8. `eq:aug_cl_step` (20): epsilon = T_y^{-1}(y^n - y-hat^n), u-hat^n = u^n + T_u(C_fb x^c + D_fb epsilon), the output means cancel in epsilon, and computing y-hat before u-hat avoids an algebraic loop through D_fb (Kessels Remark 5.1).
9. Appendix residual-form paragraph: (19) equals the model loop plus (u_act - u-bar_act); "the model loop contains no noise"; "only through this term does the measurement noise reach the model". Check whether the encoder start is a further noise path the paragraph omits.

Identification (IV-C)
10. `eq:aug_loss` (22): normalisation 1/(|W| n_f n_y), the constraints match (17) and (20), W_r = diag(sigma_y,j / sigma_e,r,j) makes each residual "in metres divided by the standard deviation of its servo error".
11. `eq:aug_encoder` (23): chi_tau stacks samples tau-n-1 ... tau-1, i.e. n+1 samples, so chi_tau in R^{6(n+1)}; "n+1 is the number of samples in the window".
12. `eq:aug_enc_lin` (24): dimensions of W^b, W^a; theta^a_enc = W^a plus the added-state output rows of psi-tilde.
13. SUBNET assumptions as stated (input independent of the noise, Beintema Sec. 2.1; incremental exponential output stability, Cond. 1) and "the open-loop gantry system does not satisfy the second".
14. Non-uniqueness of the baseline/network split (Hoekstra Sec. 5.2).

Initialisation (IV-D)
15. `eq:aug_enc_sim` (25) baseline simulated in closed loop with theta^0_base; records start from q = P^{-T} y with zero velocities.
16. `eq:aug_enc_fit` (26) least squares with intercept b; chi^0 built as in (23) from y-hat^{n,0}, u-hat^{n,0}; D = window ends at stride 10.
17. `eq:aug_init` (27): with W^{x-tilde}_L = 0 and b_L = 0 the augmented model gives the same x-tilde and y-hat as the baseline from the same physical state; Xavier uniform bound a = sqrt(6/(n_xbar + 6(n+1))) for W^a in R^{n_xbar x 6(n+1)}; U(-1/sqrt(m), 1/sqrt(m)) for hidden layers of phi_aug and psi-tilde and W^{x-bar}_L (n_h for the last).

Added-state initialisation (IV-E) and appendix PS details
18. `eq:ps2_s1` (28): e_tau over tau ... tau+H-1 with H = n+1, decoder p, minimised over theta_p and theta^a_enc; "the expected squared error is minimised by the conditional mean E[e_tau | chi_tau]"; the decoder sees only x-bar, which forces the encoder to put predictive information in x-bar.
19. `eq:ps2_alt` (29): alternation of one Adam step on (22) and one on (28) with its own optimiser; written-back theta^a_enc.
20. `eq:ps2_s2` (30): the cost equals the fraction of variance of the next added states not explained; the physical rows of W_L, b_L get zero gradient; shared hidden layers change f^n_aug too.
21. `eq:ps2_handover` (32) and "the models with and without PS have the same number of parameters".
22. Appendix: decoder dimensions; nu (`eq:ps2_nu`); the S1 stopping rule set; the S2 input argument (at x^c = 0, u-hat = u + D_fb(y - y-hat), equal to the recorded force when the encoder reproduces y_{k_i}); "chi_{k_i+1} uses the data up to k_i".

### 4.2 Decisions taken with the author (do not re-argue)
D_N of Section II is the estimation set; a separate validation set D_val is defined in IV-C; the test set only in Section V. Normalisation stays in IV-A. (17) presents the normalised model and defines f^n_base (rewritten 2026-10-23, same math as the old version). One reason for closed-loop simulation (marginal stability, Jan49/Jan50); noise is treated as a consequence, not a reason. Decoder named p (d is a baseline parameter).

### 4.3 Proposed text, not yet applied (verify it as claims 9a to 9d)
After "The model is still simulated with respect to the reference r, but r and u_ff are not used in simulation." in IV-B:
> (9a) The model is then driven only by r and u_ff, which are independent of the noise v (Section II). (9b) The correlation of u_act with v therefore does not enter the model input. (9c) If the controllers between the system and simulation differ, the noise enters through the difference in their forces (Appendix). (9d) The noise also enters through the encoder, which uses the recorded u_act and y.
Also check the condition in the data: the data controller ran at f_sim with the recorded y filtered and decimated (`Clean/gantry_dynamic/data.py`, `controller.py`), so whether "the controllers are the same" can hold at all.

### 4.4 Report format (one example row; give every finding, most severe first)
| # | Claim (number in 4.1/4.3) | Where (label, first words, file:line) | Verdict (correct / wrong / conditional) | Evidence (derivation step, paper page, code file:line) | Smallest fix (exact replace-with LaTeX, or "none") |
|-|-|-|-|-|-|
Then one line per claim in 4.1 and 4.3 that you found correct, with its evidence pointer.

### 4.5 Sources
Papers: `Thesis-writeup/References/index.csv` maps cite keys to PDFs (`hoekstra2026lfr`: Sec. 2 and Table 1 PDF p2-3, Sec. 5 PDF p8-10; `beintema2023subnet`: Sec. 2-4; `kessels2025ai`: Ch. 5, Eqs. (5.12)-(5.13), Remarks 5.1, 5.3, 5.4; `hoekstra2026encoder`: Eqs. (8), (33), (35); `hefny2015supervised` Sec. 2; `downey2017psrnn` Sec. 4.2). Code: `Clean/model_augmentation/fit_systems/blocks.py` (RK4 `_rk4`, `_deriv_with`, denormalisation), `closed_loop.py` (residual rollout), `pre_encoder.py` (encoder fit), `ps2_opening.py` (PS, calibration split, separate Adam), `interconnect.py`; `Clean/gantry_dynamic/data.py` (means, finite differences, filtering), `controller.py` (Tustin at `cfg.ts_new`), `config.py` (`up_sample = 1`), `model.py`.

## 5. Assumed but not verified
- That the code in `Clean/` is what produced the thesis results. If a file differs from the text, report both and do not decide which is right.
- That the PDF equation numbers match the labels as assumed above; refer to labels.

## 6. Tried and failed
None for this check. The earlier Codex review (`2026-10-22-codex-review-section4.md`) found and the author fixed: the S2 input claim missing its condition, the chi index (k_i, not k_i+1), the decoder's separate Adam state, f_aug vs f^n_aug naming.

## 7. Achieved
Section IV cut pass done; IV-A and the start of IV-B walked on 2026-10-23. Not compiled.

## 8. The open question
Nothing blocked.

## 9. Next action
Read `03_augmentation.tex` Section IV, `99_appendix.tex` (`app:setup_checks` residual paragraph, `app:ps_stop`), then check claims 1 to 22 and 9a to 9d in order, and report per 4.4.

## 10. Acceptance criterion
Every claim in 4.1 and 4.3 has a verdict with evidence (derivation, paper page or code line); every non-correct verdict has an exact fix.

## 11. Read these first
1. `Thesis-writeup/Writing/sections/03_augmentation.tex`, Section IV (final reading).
2. `Thesis-writeup/Writing/sections/99_appendix.tex`, the two passages named in 1.
3. `Thesis-writeup/Writing/sections/02_system_baseline.tex`, `sec:problem` to `sec:params` (symbols IV uses).
4. `Clean/model_augmentation/fit_systems/blocks.py` and `closed_loop.py`.
5. The cited paper pages in 4.5 as needed.

## 12. Do not
- Edit any file, build anything, run git write commands or numerical scripts.
- Propose re-adding the noise-correlation reason for closed-loop simulation or the indirect approach.
- Print PDF text to the console (cp1252); write extractions to a UTF-8 file in a temp folder.

## 13. Operational
Python if needed for reading PDFs: `conda run -n GraduationProject python -I <script>`, PyMuPDF, output to a UTF-8 file outside the repo. No other runs.

## 14. Delegation
None; sequential careful reading.
