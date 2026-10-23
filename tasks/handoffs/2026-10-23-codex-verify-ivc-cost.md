# Handoff (Codex): verify the math of thesis Section IV-C (Identification), above all the cost function (22), against the source papers and the code
**From**: Claude session of 2026-10-23 (IV-C walk) | **Branch**: Augmentation | **Effort suggested**: xhigh (term-by-term comparison of one cost function with two papers and the implementation; no runs)

## 1. Task
Verify, read-only, every equation and mathematical claim in subsection `\subsection{Identification}\label{sec:aug_closed_loop}` of `Thesis-writeup/Writing/sections/03_augmentation.tex` (up to `\subsection{Initialisation}`), with the cost function `eq:aug_loss` (shown as (22) in the PDF) as the main target. For each claim decide: correct, wrong, or correct only under a condition the text does not state, and check it three ways where they apply: (a) by hand from the equations it depends on, (b) against the cited source passage (page and equation number), (c) against the implementation in `Clean/`. Where the thesis says "as in", "adapted from" or "following" a source, state precisely what is the same and what differs, and whether the text names every difference. Report in your reply (format 4.3). Edit nothing.

## 2. Out of scope
- Any file edit, LaTeX build, git write, or numerical run. If a claim can only be settled numerically, say which quantity would settle it.
- Wording and style, unless they make a statement mathematically wrong or ambiguous.
- Subsections IV-A, IV-B, IV-D, IV-E and other sections, except to check a symbol or equation IV-C uses (`eq:aug_norm`, `eq:aug_norm_maps`, `eq:aug_cl_step`, `eq:aug_residual_loop`, `eq:phi`, `eq:mdelta_map`, `eq:lfr_admissibility`, `eq:system`).
- The decisions in 4.2.
- `kamtin-data/`, `kamtin-fp-model/`.

## 3. Where things stand
Branch `Augmentation`, last commit `4df8faa`; uncommitted edits in `03_augmentation.tex` and `99_appendix.tex` from the walk of 2026-10-23. The author may be editing; read the file once and review that snapshot. Read the source in its final reading as described in `tasks/handoffs/2026-10-22-codex-review-section4.md` section 4.1 (`\chg{old}{new}` = new, `\del`/`\delmv` = gone, `\mv` = present, `\todo`/`\revnote`/`\ifdraft` = not printed). A previous Codex math check of the whole section (`tasks/handoffs/2026-10-23-codex-verify-section4-math.md`) found IV-C largely correct; its findings 8 (validation metric uses the standard deviation of the servo error) and 11 (smoothness citation) are already applied. Do not repeat that report; go deeper on the cost function.

## 4. Established and verified

### 4.1 Claims to check
Cost function `eq:aug_loss` (22) and its where-clause
1. Normalisation 1/(|W| n_f n_y), the double sum over tau in W and k = tau ... tau+n_f-1, the squared 2-norm. Does the code average over exactly these (windows, samples, outputs), including the sample k = tau? Is any sample excluded (encoder history, warm-up)? Are windows overlapping, and at which stride?
2. The weight W_r = diag(sigma_y,j / sigma_e,r,j) applied to (y^n - y-hat^n): check that W_r T_y = diag(1/sigma_e,r,j), so each residual "in metres is divided by the standard deviation sigma_e,r,j of its servo error y_j - r_j". Check how sigma_e,r,j is computed in code (per record r of D_N, per axis j; mean removed or not) and whether "standard deviation" or RMS is used.
3. The constraints: x-tilde^n_{k+1|tau} = f^n_base(theta_base(beta), ...) + f^n_aug(...), x-bar_{k+1|tau} = g^n_aug(...), x-hat^n_{tau|tau} = psi(theta_enc, chi_tau), y-hat^n and u-hat^n from (20). Consistent with (17) and (20)? Is the controller state x^c reset per window and "the controller fixed" (no trainable controller parameters)?
4. vartheta = (beta, theta_aug, theta_enc): exactly the trained parameters in code? theta_base(beta) as defined in Section III-C (`eq:phi`, `eq:mdelta_map`).
5. "We adapt this from [Hoekstra 2026, Eq. (22)], adapted for the closed loop of Section IV-B and the encoder": compare term by term with Hoekstra 2026 Eq. (22) and (22a)-(22e) (PDF p8) and Sec. 5.2 baseline parameter regularisation (PDF p8-9). Name every difference: closed-loop input instead of the recorded input, the weight W_r, any regularisation term on theta_base that Hoekstra has and the thesis omits (or vice versa), normalisation constants, window definition, encoder input range (samples up to tau-1, Eq. (22e)).
6. Compare with Beintema 2023 Sec. 3.1, Eqs. (8)-(9) (PDF p4): the truncated prediction loss and its normalisation.
7. "jointly estimated, as in [Hoekstra Eq. (22)], while the encoder paper keeps theta_base fixed [hoekstra2026encoder, footnote 1]": check both citations.

Encoder (23), (24)
8. chi_tau indices tau-n-1 ... tau-1 (n+1 samples), chi_tau in R^{6(n+1)}, "n+1 is the number of past samples used by the encoder"; normalised as in (16); matches `pre_encoder.py` and `model.py`.
9. (24): psi = [W^b; W^a] chi + psi-tilde(chi), dimensions, theta^a_enc definition; Hoekstra encoder paper Eq. (8).
10. "The encoder estimates all 6 + n_xbar initial states, which includes the measured positions" and "we estimate the positions instead of using the measured y_tau to keep the positions consistent with the estimated velocities": does the code estimate the positions (not overwrite them with y)?

Windows, assumptions, non-uniqueness
11. "Estimating x-hat_tau as a separate parameter for each window would increase the number of parameters as the data grows" (Beintema Sec. 3.1); "keeps the computation per gradient step independent of the record length for fixed batch size and window lengths" (Sec. 3.1, Eq. (9)); "Using short simulation windows increases the loss function smoothness" (Sec. 3.1, 4.2, Theorem 9).
12. SUBNET assumptions as cited (Beintema Sec. 2.1, Cond. 1; Hoekstra 2026 Sec. 5.5) and "the open-loop gantry system does not satisfy the second".
13. "Each window starts from a new encoder estimate and x^c_tau = 0, so the window is not affected by model errors before tau."
14. Non-uniqueness (Hoekstra 2026 Sec. 5.2).

Optimiser and selection
15. Adam on minibatches of windows, "the gradient is calculated over the full window" (backpropagation through all n_f steps?), then L-BFGS "on a fixed set of windows", "adapted from the Adam and L-BFGS-B sequence of Drenth et al. [Sec. 4.1]", "Because every beta satisfies (13), unconstrained L-BFGS works without a bound". Check against `lbfgs_polish.py`, `training.py`, `interconnect.py` and Drenth Sec. 4.1.
16. Selection on D_val: "the parameters with the lowest validation error are kept" (during Adam only?); the error definition (mean over records of D_val and axes of RMS simulation error divided by the standard deviation of the servo error), simulated in closed loop "with the initial state estimated once by the encoder and x^c = 0", scored from which sample on; "We only keep the polish if it lowers the validation error".

### 4.2 Decisions taken with the author (do not re-argue)
D_N is the estimation set and consists of several records; a separate validation set D_val is defined in IV-C; the test set appears only in Section V. Closed-loop simulation has one reason (marginal stability, stated in IV-B). Normalisation is defined in IV-A.

### 4.3 Report format (most consequential first)
| # | Claim (4.1 number) | Where (first words, file:line) | Verdict (correct / wrong / conditional) | Evidence (derivation step; paper, PDF page, equation; code file:line) | Smallest fix (exact replace-with LaTeX, or "none") |
|-|-|-|-|-|-|
Then a short term-by-term table for claim 5: each term or element of Hoekstra Eq. (22)/(22a-e), its counterpart in thesis (22), its counterpart in code, and same/different/missing. Then one line per remaining claim found correct, with its evidence pointer.

### 4.4 Sources
`Thesis-writeup/References/index.csv` maps cite keys to PDFs (`hoekstra2026lfr` Sec. 5.1-5.5 PDF p8-10; `beintema2023subnet` Sec. 2.1 p3, Sec. 3.1-3.2 p4-5, Sec. 4.2 p8; `hoekstra2026encoder` Eq. (8), footnote 1; `drenth2025lpvlfr` Sec. 4.1; `kessels2025ai` Ch. 5). Code: `Clean/model_augmentation/fit_systems/interconnect.py` (loss, around :500-530), `closed_loop.py` (rollout, validation scoring), `lbfgs_polish.py`, `pre_encoder.py`; `Clean/gantry_dynamic/training.py`, `data.py` (normalisation, sigma_e), `controller.py` (loss weights, around :190-200), `model.py`, `config.py`.

## 5. Assumed but not verified
- That `Clean/` is the code that produced the thesis results. If code and text differ, report both and do not decide which is right.
- That PDF equation numbers match the labels; refer to labels.

## 6. Tried and failed
None for this check.

## 7. Achieved
IV-C walked on 2026-10-23: intro sentence (D_N, cost function), record definition at W_r, encoder-window wording, repeats of IV-B struck, D_val sentence, validation metric (standard deviation). Not compiled.

## 8. The open question
Nothing blocked.

## 9. Next action
Read IV-C in its final reading, then Hoekstra 2026 Eq. (22) with Sec. 5.1-5.2 and Beintema Sec. 3.1, then the loss code in `interconnect.py` and `controller.py`; check claims 1 to 16 and report per 4.3, cost function first.

## 10. Acceptance criterion
Every claim in 4.1 has a verdict with evidence; the term-by-term table for claim 5 is complete; every non-correct verdict has an exact fix.

## 11. Read these first
1. `03_augmentation.tex`, subsection Identification (final reading).
2. Hoekstra 2026, PDF p8-9 (Eq. (22), Sec. 5.1-5.2).
3. `Clean/model_augmentation/fit_systems/interconnect.py` loss and `Clean/gantry_dynamic/controller.py` loss weights.
4. Beintema 2023, PDF p4 (Sec. 3.1, Eqs. (8)-(9)).
5. `closed_loop.py` validation scoring and `lbfgs_polish.py`.

## 12. Do not
- Edit, build, run, or write to git.
- Repeat the earlier whole-section report; go deeper on IV-C.
- Print PDF text to the console (cp1252); extract to a UTF-8 file in a temp folder.

## 13. Operational
If needed: `conda run -n GraduationProject python -I <script>` with PyMuPDF for PDF text, output to a UTF-8 file outside the repo.

## 14. Delegation
None.
