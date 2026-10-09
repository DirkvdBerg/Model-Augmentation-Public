# Critic (checker lens), cycle 04, Section III

Score: 7 / 10

Compile: `latexmk -pdf main.tex` exits 0 and `build/main.log` has no line starting with "!". The only undefined references (`eq:Msplit`, `eq:lfr_chain`) are in `99_appendix.tex`, not this section. No em-dashes in the file; no semicolons in prose.

Verified against the code (thesis path, `THESIS_*` block of `ENT`): routing to all 6 + n_a rows (`ann_route_ix=tuple(range(6+_na))`); 2 tanh hidden layers; one RK4 step per sample (`up_sample=1`); state statistics from `P^-T y` and its scaled first difference, every signal centred (`compute_normalization`); `T_x f_aug = phi` (ANN adds in normalised state space); zero output layer (`g_init='zero'`); W^a Xavier uniform gain 1; encoder window `[tau-29, tau]` (`na_right=1`); W^b blocks match encoder paper Eqs. 16, 17; m_Delta map matches `combinations_from_free` and `_m_diff_bound`; criterion = MSE over every window sample in normalised output; selection = record-length-weighted mean of per-record RMS (`validation_error`); polish rule = score improves AND no meter regresses (`decide`); PS2: decoder one hidden tanh layer with zero output layer, horizon M = na + na_right = n + 1, unit batch RMS, S1 step after every update with its own Adam, S2 on frozen encoder pairs with the recorded input, calibration records withheld. Citations all match `citation-log.md` rows.

Strengths: roadmap in Hoekstra's form, one job per subsection, the identification problem, normalisation, initialisation, encoder map, residual loop, training coordinates and both PS2 stages are displayed with spaces in "where" clauses; trained versus fixed is explicit; open choices are `\todo`s in the required form.

## Issues

1. MAJOR (F, B). Encoder coordinates misstated. Lines 330 to 338 and 360: the text says the linearisation is taken "in the coordinates of eq:aug_norm" and displays psi as `[W^b; W^a][y; u] + psi_tilde` on centred histories. The code (`pre_encoder.py`, D-055/D-119) applies W^b and W^a to scaled but uncentred histories (adds `y_off`, `u_off`) and subtracts `x_off = T_x mu_x` from the physical rows only; `normalize_linear_ss_matrices` scales without centring. The displayed map therefore omits the constant `W^b col(T_y mu_y, T_u mu_u) - T_x mu_x`, which is not zero because the mean input is no equilibrium of a model with free X and Y. Fix: state that W^b acts in scaled, uncentred coordinates (as Hoekstra Eq. 28) and show the offset in eq:aug_encoder. Rule cause: README "Math standard" item 4 requires the display to show which coordinates each map uses, but the III-B source-map row does not point the writer at the D-055 offset convention in `pre_encoder.py::forward`.

2. minor (B). Identification problem not one display. Eq. aug_loss displays only the cost; the constraints are listed in prose ("subject to eq:aug_transition with u_k = ..., eq:aug_encoder and eq:aug_loop"). Hoekstra 2026 Eq. 22 displays them. Rule cause: README "Math standard" item 3 says "one display" but does not say whether citing earlier displays by number satisfies it.

3. minor (A, D). Forward reference: `vartheta = (xi, ...)` uses `xi` "below" (line 501) before eq:mdelta_map defines it. Put the training-coordinates paragraph before the criterion, as Hoekstra Sec. 5 orders problem then coordinates. Rule cause: skill "Order", "Known before new"; the "One job per subsection" grouping lists the criterion first, which invites this order.

4. minor (F, D). Polish acceptance stated vaguely: "no monitored quantity regresses" (line 532) does not give the condition as the code states it (meters with a relative tolerance), and hides the oracle meter the draft's own `\todo` flags. Rule cause: skill "Content", reader test exception for "a rule that defines the method ... with its condition as the code states it".

5. minor (F). PS2 switching rules imprecise versus `PS2Spec`: S1 also has a minimum update count before a plateau can end it (`s1_min`), and the screen at update 150 ends S1 and falls back, not only "S2 is skipped"; S2 restores its best calibration checkpoint, unstated. Rule cause: same reader-test exception; README PS2 sketch allows one sentence per switching rule but does not list the conditions.

6. minor (D). PS2 problem argued only at the exact start: the zero gradient of V in W^a and the added-state rows of W_L holds at iteration 0; after the first update of the f_aug rows it is nonzero. "We therefore initialise ..." needs the claim that the added states stay uninformative (PS2-METHOD Sec. 2.2), else a `\todo`. Rule cause: skill "Reasons and claims" (claim as strong as its evidence); not missing, not applied.

7. minor (E, C). PS2 subsection exceeds the agreed sketch (about 2 displays and 6 sentences plus switching rules): about 17 sentences, and includes optimisation mechanics the sketch excludes ("from the encoder state without gradient", "with a separate optimiser", record-withholding bookkeeping in two places). Rule cause: README "Example (PS2, sketch agreed 2026-10-08)" and skill "Content" reader test; the skill's "Length" rule treats the count only as a target, so nothing forced the cut.

8. minor (E). Restatement in III-A: line 197 announces "parallel, augments the state only, and corrects every physical row", then each is restated as its own sentence ("The output map is not augmented." repeats "state only" and the caption's `h_aug = 0`). "Training starts from the baseline." and the sentence after eq:aug_zero_init say the same thing. Rule cause: skill "Say each fact once" and "Deletion test"; the packing pass caught packing but no pass targets summary-then-itemise.

9. minor (A, E). Topic-sentence drift: the paragraph opened by "The map exists although the baseline is not stable" ends with the convergence benefit of the model-based init; the paragraph opened by "The residual form needs no controller initialisation" ends with the step order and the D_fb algebraic-loop argument. Rule cause: skill "Order", "Topic sentences".

10. minor (F). Symbol clash: `rho` is the margin in eq:mdelta_map but Section II (line 304) uses `rho(D_zw)` as the spectral radius; no `\todo` names it here (Section IV's `\todo` does). Rule cause: skill "Notation" (quoted operators are checked like introduced symbols).

11. minor (F). Forssell cited as `[Sec. 5.2.1]` against the Automatica 1999 bib entry, while the citation log and source map verify that numbering only in the LiU report and mark the Automatica numbering unchecked. Rule cause: README "Reference verification gate"; the source-map caveat is not turned into a requirement (cite the verified version or add a `\todo`).

12. minor (F). The `\todo` on state statistics (line 242) gives "own reasoning" as candidate and calls D-119 merely the removal of the simulated-state source; D-119 records a reason (one frame for encoder and state block, and the baseline-simulation file never existed for these modes). Rule cause: skill "Reasons and claims" (every reason traces to a decision when one exists).

13. minor (F). Subsection title "Initialisation of the Added States" is title case; the other subsections, and Section II's, use sentence case. Rule cause: none in the rules (missing consistency check for heading case).
