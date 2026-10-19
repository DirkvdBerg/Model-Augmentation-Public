# Comparison of the two step-1 inventories (2026-10-19)

Inputs: `inventory-claude.md` (coverage trace of the 9 final rows on lowpass, plus 221 and 224 on noise-free) and
`inventory-codex.md` (static reading). Part A was crossed function by function by a script (Claude A.3 against Codex
A4); Parts B and C were compared by reading. Disagreements were settled against the code where a grep or read could do it.

## Part A. Required code

**Agreement.** 538 definitions appear in both lists. 502 agree outright: 232 executed and required, 270 not executed
and not needed. The two lists also agree on every config field, on the 51 unused classes, and on `common/` (nothing in
it is needed for training).

**Label-only differences (no action).** 22 functions run during training but have no numeric effect: the nf probe's
printing parts, `WindowErrorStats`, the polish meters, `_note_update_time`, `memory_bytes`, `print_run_banner`,
`identifiable_combinations`, `load_mat_aug`. Claude labels them "required, no numeric effect" and Codex labels them
"diagnostic". Both mean the same thing: they can be dropped without changing numbers.

**Real disagreements, settled:**

| Item | Claude | Codex | Settled | Evidence |
|-|-|-|-|-|
| `data.load_traj` | not needed | evaluation interface | not needed | no use in `Thesis-writeup/Code/Evaluation/` (grep) |
| `OBCBasis.apply`, `jt`, `r_perp`, `frobenius`, `relative_change` | required (refresh print, validation meters) | diagnostic | diagnostic | `obc.py:567-568` returns theta before `apply`/`r_perp` run; `:602` is in `stacked_diagnostics` |
| `_combos_from_raw` | required | diagnostic | evaluation interface only | called by `param_table`, `identifiable_combinations` and the training meters (`training.py:430, 742`), never by the loss |

Agreed as "not executed on CPU, but required": `_is_oom` and `OBCBasis.to` (GPU or failure paths). Agreed as "needed
by the evaluation, not by training": `load_checkpoint`, `resolve_checkpoint`, `_assert_same_architecture`,
`_assert_same_dtype`, `estimation_parameters`, `free_from_combinations`, `param_table`,
`closed_loop_report._without_learned` and `_true_free`.

**Dependency both flag:** the installed `deepSI` package (`System_data`, `SS_encoder_general`, `System_fittable`,
`My_Simple_DataLoader`). It stays a dependency, or is vendored only if the `default_rng(0)` in-place shuffle is
reproduced exactly.

## Part B. Thesis against code

**Found by both:**
1. PS2 S2 cost uses `T.var(0)` (divides by B-1): (B-1)/B times the thesis ratio, same minimiser (`ps2_opening.py:226-227`).
2. PS2 S2 keeps the LAST iterate when no check beats the start; the thesis says it keeps the lowest held-out cost (`ps2_opening.py:233-251`).
3. Test records: the loader gives 12 (E5, generated with gain 1.2, included); the thesis says 11. E5 is skipped when scoring (`data.py:79-89`).
4. OBC offset column: the code appends `f_base(beta_lin)`, while the thesis has gamma = f_base - Phi beta_lin. Same span and correction; the first 10 coefficients differ by alpha beta_lin (`obc_gantry.py:332-338`).
5. `eq:setup_decim` does not describe the noise-free mode, which point-samples y (`data.py:172-182`).

**Found by Claude only:**
6. "Eighth-order Butterworth" applied with `sosfiltfilt` has the magnitude of a squared eighth order (`data.py:181-182`).
7. With OBC, the final model is the rebuilt-basis model even when it scores worse than the selected checkpoint; not stated in the thesis (`interconnect.py:1148-1149`).
8. The last 900 Adam updates (33400 mod 1300) are never validated, so they cannot be selected.
9. Adam batch order uses deepSI `default_rng(0)`, independent of the run seed.

**Found by Codex only:**
10. Appendix line 124 says both polynomials use Horner; the code expands the denominator (`blocks.py:848-855`).
11. Section V line 319, "100 % equals y_hat = r": exact only if mean(e) = 0, since the numerator is not centred.
12. Section II line 268 calls Y "measured"; the simulation schedules on predicted Y.
13. `eq:obc_overlap`, c_neg for OBC: the evaluator uses `corr.vbar` as the operating point, the thesis the current beta (`eval_negation.py:71-89`).
14. Nominal parameters and P are float32 in `gantry_ss.py`; the controller uses doubles. Not stated in the thesis.
15. The PS auxiliary Adam uses the default eps; the hyperparameter table does not state it.
16. A stale label in `evaluation.py:166-181` (the NRMS_e curve is labelled in metres); diagnostic only.

For Dirk to decide: which side wins for 1, 2, 3, 6, 7, 10, 11 and 12. These are either thesis-text fixes or code
changes. 4, 5, 8, 9 and 13 to 16 are thesis-detail additions or out of the training path.

## Part C. Equivalence hazards

The two lists agree on the points that matter for the harness:
- The global torch stream draws in a fixed order after the seed at `entry:409-410`: live g_aug MLP, then the W^b draw
  (its values are overwritten by least squares, but the draw advances the stream), then W^a Xavier, then the encoder
  net, then the PS2 decoder at the first update. Removing or reordering any draw changes every later weight.
- With n_a = 0 the g_aug constructor still consumes draws.
- `Dtype_DataLoader` shuffles with `default_rng(0)`. The PS2, polish-subset and probe generators are local and seeded.
- No diagnostic draws from a global stream before the final checkpoint is written, so dropping diagnostics changes no number.
- Float32 materialisation (nominal tensors, `to_tensor`, `nn.Linear` defaults cast to float64) must be reproduced, not "fixed".
- `torch.compile` and the CUDA branches are not reachable on CPU, so the harness covers them only by a GPU smoke run.

## Decisions (Dirk, 2026-10-19; D-274)

Item 4 (filter) is already fixed in the appendix ("applied forward and backward"); dropped.

| Item | Decision |
|-|-|
| 1 PS2 variance | code kept (B-1), comment in code; thesis unchanged |
| 2 PS2 keep best | clean code restores the start parameters when no check beats the start |
| 3 Test records | clean code skips E5 (11 test records) |
| 5 OBC polish | rebuild at the checkpoint's beta as now; polish kept only if below the selected checkpoint's score (`bestfit`), otherwise the rebuilt model is kept (D-274 amendment: no reruns); change in `training.py` only; clarifying todo in the OBC section |
| 6 Horner | thesis: "evaluates the numerator with Horner's scheme and $d(Y)$ term by term" |
| 7 100 % | thesis: "An NRMS of 100\,\% means the model performs no better than predicting the mean servo error." |
| 8 measured Y | thesis: "measured" removed in Section II (two sentences) |

Items 10 to 16 need no change (thesis detail or outside the training path).

## Next

Part A disputes are settled and the Part B rulings are in, so step 2 (the design document) can start. Open input: the
polish outcome of runs 221 and 222 (server logs), which decides whether D-274 change 3 alters any final checkpoint.
