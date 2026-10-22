# Check of the Clean comments and docstrings

2026-10-22. Reviewed the `4e3c4c9..1107587` pass and all 24 Python files in `Clean/`, against `comments-naming-review-codex.md` and the five requested thesis sources. The working copy of `Clean/` matches `1107587`. Only this report is written. No training, harness execution, project imports, git writes, evaluation edits or protected-data reads were performed.

The two previously important OBC formula corrections are applied: the SVD coefficient includes `D^-1`, and the gantry basis explanation distinguishes `[Phi F_base^n(beta_lin)]` from `[Phi Gamma]`, including the coordinate conversion. Remaining findings concern false guarantees and behavior descriptions, several ineffective error remedies, and the comment standard. No deployed numerical formula needs changing as a result of this review.

## 1. Wrong formulas and false claims

Findings are ordered by consequence within this section. A qualified branch description is still required even when the thesis presets only exercise one branch. Proposed text below is documentation, not a request to change behavior.

### F01. The averaged input is not an exact coarse-rate ZOH equivalent

`Clean/gantry_dynamic/data.py:87`:

> u_total is a zero-order hold at FS_ORIG (the 20 kHz controller), so the exact input of a zero-order-hold model at FS_NEW is the mean force over each FS_NEW hold interval.

Lines 92 to 94 compute the arithmetic mean of five fine-rate samples. This preserves the interval's force integral, but generally not its state transition. Even for an LTI system the fine-input contribution is `integral exp(A(T-s)) B u(s) ds`; the input samples carry different state-transition weights. For the simple plant `qddot = u`, inputs `1, 0` over two unit subintervals give final velocity 1 and position 1.5 from zero; the mean input 0.5 over the full interval gives velocity 1 and position 1. The gantry also has damping and state-dependent inertia.

Replace with: `# THEORY: block averaging preserves the force integral over each model interval; using that mean as a held force approximates the fine-rate input.`

The same exactness claim occurs in `99_appendix.tex`, after `eq:setup_decim`; matching that sentence does not make the claim mathematically true. `05_setup.tex` explicitly acknowledges force changes inside a model hold. Leave the thesis untouched under this task's scope.

### F02. The force is not strictly band limited, and averaging does filter it

`Clean/gantry_dynamic/data.py:89`:

> The force is band-limited far below the 2 kHz Nyquist frequency, so this is not anti-aliasing.

`u_total` includes feedback from noisy measurements, not just the low-frequency injected excitation. The loader imposes no band limit. The averaging at lines 92 to 94 is itself a boxcar filter before downsampling, although not a sufficient general anti-alias filter. A high-rate ZOH also does not establish the asserted band limit. The controller-rate and within-hold checks in `99_appendix.tex` concern this distinction.

Replace with: `# block averaging also applies a boxcar low-pass response before decimation; it is not a dedicated anti-alias filter.`

The intervening claim that point sampling necessarily leaves a mean force error and produces a position offset is too categorical: this depends on the signal and residual bias. Replace that sentence with `# a persistent bias in the resampled force can cause drift on the axes without stiffness to ground.`

### F03. The selection score has no metre unit

`Clean/model_augmentation/fit_systems/closed_loop.py:157`:

> sim-NRMSe: the mean over records and axes of RMS / sigma_e of the closed-loop free run, in metres.

Line 165 returns `mean(last_per_channel / val_sigma_e)`. Both operands have metre units, so the ratio is dimensionless. The separate `last_per_channel` and `last_sim_rms` diagnostics are in metres. This agrees with `05_setup.tex`, `eq:setup_rms`.

Replace with: `"""Dimensionless sim-NRMSe: mean over records and axes of closed-loop RMS divided by servo-error standard deviation."""`

### F04. Restoring gradients does not leave the next training update unaffected

`Clean/model_augmentation/fit_systems/ps2_opening.py:176`:

> the gradients go to the opening's optimizer only; the .grad of the shared parameters is saved and restored, so the next Adam step of the training is unaffected

Line 182 performs an optimizer step on shared encoder parameters. Their new values affect the next training loss and gradient. Restoring `.grad` preserves those slots; it does not undo the S1 parameter update or isolate the next training update from it. The main training closure also clears gradients before backward. The alternating updates in `03_augmentation.tex`, `eq:ps2_alt`, expressly change the encoder between main updates.

Replace with: `# S1 updates shared encoder parameters with its own Adam state, then restores the existing gradient slots.`

### F05. Two error remedies cannot enable the advertised configuration

`Clean/gantry_dynamic/controller.py:122`:

> set RunConfig.controller_gain to score it with its own controller

The normal construction at lines 146 to 147 never passes `cfg.controller_gain` to `build_controller_bank`, so its default `use_gain=False` remains in force. The setting is also a ClassVar, not a dataclass constructor field. Changing the class value alone does not fix this path. Direct evaluation callers that explicitly pass `use_gain=True` are different.

Replace the suffix with: `use a caller that explicitly passes use_gain=True to build_controller_bank, or restrict this run to unit-gain records`.

`Clean/gantry_dynamic/training.py:117`:

> set use_f64=%s to match the checkpoint

`RunConfig.use_f64` is a ClassVar, while `dtype_pt` and `dtype_np` are independently fixed to float64 in `config.py:128`. `build_model` uses `dtype_pt`; changing `use_f64` does not select float32. The message is actionable in a configurable mainline implementation, but not here.

Replace with: `checkpoint dtype %s does not match this run's %s; use a matching float64 checkpoint or the mainline implementation configured for the checkpoint precision`.

### F06. Compilation is also used during polish

`Clean/model_augmentation/fit_systems/closed_loop.py:104`:

> compiled only for the training loss on CUDA (gradients on); everything else runs eagerly

`Clean/gantry_dynamic/controller.py:156`:

> compiled here and handed to the simulator, which uses it only for the training loss on CUDA, so validation stays eager

The condition in `ClosedLoopSimulator.__call__` is a compiled pair, CUDA input, and gradients enabled. The L-BFGS closure meets it too, including the initial determinism probe. Validation uses a separate eager path.

Replace the first with `# use the compiled pair for gradient-enabled CUDA rollouts, including the polish` and the second with `# compile the pair for gradient-enabled CUDA objectives; validation uses the eager rollout`.

### F07. The training docstring promises an unconditional full Adam budget

`Clean/gantry_dynamic/training.py:233`:

> Adam for hp['epochs'] epochs with selection on the validation score, then the L-BFGS polish

Lines 251 and 277 to 292 run only the remaining epochs on continuation, skip Adam for a polish-only start or a spent budget, and permit `cfg.n_its` to override the update count in `fit`. `hp['epochs']` is the total nominal budget, not the number of epochs this call necessarily runs.

Replace with: `"""Run the remaining Adam budget when requested, then polish and save the selected model; return its final score."""`

### F08. N records do not necessarily use one rollout

`Clean/model_augmentation/fit_systems/closed_loop.py:169`:

> Closed-loop free-run RMS of N records in one rollout

Lines 186 to 188 group by scored length, and line 205 calls the rollout once for each group. Equal-length thesis records happen to share a rollout; unequal lengths do not.

Replace with: `"""Return closed-loop RMS per record and axis and per record, batching equal lengths; optionally return physical predictions."""`

### F09. The basis-builder identity is conditional on space

`Clean/gantry_dynamic/obc_gantry.py:151`:

> J = [Phi F_base^n(beta_lin)] in float64 on the reference tuples.

Line 159 appends the transition column only for `space == 'affine'`. Other values return `Phi`. The configured thesis arm is affine, but the function accepts and implements the other branch.

Replace with: `"""Return a float64 basis builder: J = Phi, with F_base^n(beta_lin) appended when space is affine."""`

### F10. The controller-bank return is not always a list of positions

`Clean/gantry_dynamic/controller.py:128`:

> The ControllerBank of a set of records, each record's row in it, and the distinct design positions.

Line 135 returns `(Y_op, gain)` keys as its third result when `use_gain=True`, positions otherwise.

Replace with: `"""Return the bank, record rows, and distinct positions or (position, gain) keys when use_gain is enabled."""`

### F11. The OOM predicate also accepts non-CUDA failures

`Clean/model_augmentation/fit_systems/lbfgs_polish.py:123`:

> True for a CUDA out-of-memory error.

Lines 124 to 125 also return true for any RuntimeError containing `out of memory`, including a CPU allocation failure. This predicate feeds diagnostic labels and cache handling, so its actual recognition rule is useful.

Replace with: `"""Recognise CUDA OOM exceptions and RuntimeErrors containing 'out of memory'."""`

### F12. The L-BFGS outer loop can pass max_iter

`Clean/model_augmentation/fit_systems/lbfgs_polish.py:245`:

> L-BFGS in blocks of inner_iter iterations, until it stops inside a block or max_iter is reached

Lines 246 to 250 allow `ceil(max_iter / inner_iter)` blocks, each with the unchanged `inner_iter` optimizer limit. For `max_iter=21, inner_iter=20`, the permitted total is 40. The default 100/20 and smoke 2/1 settings divide evenly, so they do not expose the distinction.

Replace with: `# allow ceil(max_iter / inner_iter) blocks, stopping early if a block uses fewer than inner_iter iterations`

### F13. The layer-stack formula needs a positive-depth condition

`Clean/model_augmentation/utils/torch_nets.py:12`:

> [Linear, activation] n_hidden_layers times, then a Linear with zero bias

Line 14 always creates one hidden layer. For zero or negative `n_hidden_layers`, the code still has one, rather than the claimed count. Thesis runs use two.

Replace with: `"""Build max(1, n_hidden_layers) hidden Linear/activation pairs and a final Linear with zero bias; return both modules."""`

### F14. A diagnostic labels an update count as an epoch

`Clean/model_augmentation/fit_systems/interconnect.py:470`:

> which happened on epoch {best_it}

`best_it` is assigned `it_count + 1` at line 447, so it counts updates. `epoch_id[-1]` in the same print is the latest validation epoch, not necessarily the selected epoch. This print predates the pass but remains a false behavior claim in the reviewed source.

Replace that part with `selected at update {best_it}; last validation epoch_id={self.epoch_id[-1]:.2f}`. The initial selection uses `best_it=0`. This is a proposed print-text correction only.

### F15. Zero difference state does not erase the recorded controller's integral memory

`Clean/model_augmentation/fit_systems/closed_loop.py:57`:

> the integral memory of the recorded run is lost.

The state initialized at line 58 is the residual controller state xc, not the recorded controller state. Its zero initialization sets the model-to-recorded controller state difference to zero. The recorded input still includes the recorded controller's output and its integral-memory contribution. What is discarded is the controller-state difference that preceding model errors would have created. This is the distinction made in `03_augmentation.tex` after `eq:aug_residual_loop`, and in the residual-loop derivation in `99_appendix.tex`.

Replace the three-line comment with: `# HEURISTIC: reset the controller-state difference xc at each window start, dropping the effect of earlier model errors; the recorded controller output remains in u_data.`

## 2. Misleading wording and missing qualifications

### W01. Scope the OBC guarantee to the reference stack

`Clean/model_augmentation/fit_systems/interconnect.py:76`:

> so the network cannot take over what the baseline parameters explain

The subtraction removes a fitted component on the reference stack at `x_bar=0`, with a fixed linearisation point. It does not forbid all baseline-like behavior at other states, nonzero added states, or across nonlinear closed-loop trajectories. `04_obc.tex`, `eq:obc_coef` and the paragraphs after it, explicitly limit the guarantee.

Replace with: `# CHANGED: remove the fitted baseline component so the corrected network is orthogonal on the reference stack at x_bar = 0`.

### W02. K and C depend on more than the two sums

`Clean/model_augmentation/fit_systems/blocks.py:276`:

> they depend on kb1 + kb2 and cb1 + cb2 only

The intended statement is that each pair enters only through its sum. Taken literally it omits `cg1`, `cg2`, `cy` and `Lb`, which appear at lines 289 to 291 and in the thesis damping matrix.

Replace with: `"""Build K and C; the joint stiffness and damping pairs enter only through kb1 + kb2 and cb1 + cb2."""`

### W03. The PS2 screen is a check at an update, not a deadline proved over history

`Clean/model_augmentation/fit_systems/ps2_opening.py:23`:

> screen: calibration unexplained fraction <= screen_max by this update

Line 94 checks the current `u` at `screen_at`, and again on S1 termination; it does not accept a previous crossing. With custom `screen_at` outside the `eval_every` grid it does not screen at that update at all. The default grid is consistent. `99_appendix.tex`, `app:ps_stop`, uses checked updates.

Replace with: `# HEURISTIC: check the current calibration fraction at this update, which must be on the evaluation grid`

### W04. PS2 sampling weights are eligible-start counts

`Clean/model_augmentation/fit_systems/ps2_opening.py:137`:

> drawn in proportion to their length

Lines 142 to 146 use `len(record) - k0 - M - 1`, not the full record length. They also omit the final two starts allowed by ordinary future-only deepSI windows, even with `pair=False`. The history/future indexing convention matches deepSI, but the population of sampled starts does not equal every available deepSI window.

Replace with: `"""Sample windows using eligible-start counts, reserving one future successor and an extra end sample; optionally include shifted histories."""`

### W05. A content mismatch does not prove a controller-row error

`Clean/model_augmentation/fit_systems/closed_loop.py:151`:

> so the controller rows per window are misplaced

The comparison at line 149 proves that an input window differs from the expected record slice. A value mutation or an unexpected transform also triggers it without proving a positional controller error.

Replace the message with: `window %d does not match its record at offset %d; check the window values and offsets before trusting the controller-row assignment`.

### W06. The rank error should name redundant columns, not rows

`Clean/model_augmentation/fit_systems/obc.py:120`:

> the protected rows and the baseline column are not redundant

The failed condition is full column rank of the stacked basis. Repeated state rows or tuples are not by themselves fatal if the remaining rows still separate all columns. The first clause about exciting coordinates is useful.

Replace that suffix with: `check reference excitation and linear dependence among the sensitivity columns and the baseline column`.

### W07. The tangent docstring confuses a function value with its derivative

`Clean/model_augmentation/fit_systems/blocks.py:180`:

> the derivative of _deriv_with and its directional derivative

The first return is `_deriv_with`'s value, the time derivative `xdot^n`, not the derivative of that function. The second is its directional derivative.

Replace with: `"""Return xdot^n and its directional derivative along the state tangent sx and matrix tangent dmats."""`

### W08. Limit the JSON assurance to the types actually converted

`Clean/gantry_dynamic/training.py:18`:

> without numpy scalars, so json.dumps cannot fail on the meta write

Only `np.integer` and `np.floating` are converted, recursively through dict/list/tuple values. `np.bool_`, complex scalars, arrays, and non-string dictionary keys are not handled. The current polish outcome uses the supported types, so no current serialization failure is established.

Replace with: `"""Convert NumPy integer and floating values recursively to native values for JSON metadata."""`

### W09. An exhausted-budget fit would validate and save, not update Adam

`Clean/gantry_dynamic/training.py:280`:

> fit(epochs=0) would still run

With its default `n_its=None`, `fit` computes zero updates at `interconnect.py:398`; it still validates, saves and reloads. The comment should identify those side effects rather than imply Adam would train.

Replace with: `# skip fit's validation and checkpoint cycle when the Adam budget is spent`

### W10. Two formulas need their local conventions stated

`Clean/gantry_dynamic/model.py:169`:

> chi from u_hat^n and y_hat^n at s .. s+k0-1

This holds as written because `na=nb=k0` in this model. Lines 171 to 172 actually take `nb` input samples and `na` output samples starting at `s`. Replace with `# with na = nb = k0, use histories at s .. s+k0-1 to fit x_tilde^n at s+k0`.

`Clean/gantry_dynamic/model.py:215`:

> 1 - R^2 per state

The identity needs nonzero target variance. For a constant state the code divides by zero and the later acceptance check rejects it. Replace with `# 1 - R^2 for states with nonzero target variance`.

## 3. Comment-standard findings

### S01. The entry-point module docstring has seven lines

`Clean/gantry_interconnect_dynamic.py:1` starts:

> Entry point of the thesis training runs: a fresh run, a resume or a polish-only rerun.

Its command example and defaults extend the docstring to seven lines. The standard permits one or two. Replace the module docstring with `"""Launch a fresh thesis run, resume, or polish-only rerun; defaults to R1 lowpass on CUDA."""`. Keep usage in argparse help or external documentation.

### S02. Public function docstrings remain multiline

The coverage ledger below quotes every complete docstring, so the following table specifies the affected public entries and proposed one-line replacements. The quoted start identifies the existing text. Formula details that merit preservation belong in a nearby short comment. Constructors are listed separately where their own existing docstring is multiline; class docstrings need not be reduced under the function rule.

| Location | Existing start | Proposed one-line docstring |
| - | - | - |
| `Clean/gantry_dynamic/config.py:57` | `Samples dropped at each record end` | `Return samples trimmed at each FS_NEW record end; sample i maps to FS_ORIG index (i + trim) D.` |
| `Clean/gantry_dynamic/config.py:145` | `A run's config.json, clean or mainline` | `Read a supported run configuration, taking roots and start_phase from defaults or overrides.` |
| `Clean/gantry_dynamic/config.py:242` | `The config.json of a run` | `Return mainline configuration keys in their original order, then Clean keys and GIT.` |
| `Clean/gantry_dynamic/controller.py:139` | `The ClosedLoopSimulator of this model` | `Build the simulator with controller rows in file order; val_data is retained for callers and unused.` |
| `Clean/gantry_dynamic/data.py:85` | `The plant force from FS_ORIG to FS_NEW` | `Average each block of cfg.d input samples to obtain the model-rate force.` |
| `Clean/gantry_dynamic/data.py:122` | `(System_data, e) of one record` | `Load a record once and return (System_data, servo error), using folder when supplied.` |
| `Clean/gantry_dynamic/data.py:133` | `sigma_e per record and axis` | `Return servo-error population standard deviations over k0 .. N-1, shape (n_records, ny).` |
| `Clean/gantry_dynamic/obc_gantry.py:43` | `qdot_k = (-q_{k+2}` | `Return fourth-order central differences aligned with q[2:-2].` |
| `Clean/gantry_dynamic/obc_gantry.py:172` | `ann -> F` | `Return a differentiable reference-field evaluator stacking the routed physical network outputs sample-major.` |
| `Clean/gantry_dynamic/training.py:31` | `Save the torch components` | `Save component state dictionaries to <base>.pt and metadata arrays to <base>.npz.` |
| `Clean/gantry_dynamic/training.py:233` | `Adam for hp['epochs'] epochs` | Use F07. |
| `Clean/gantry_interconnect_dynamic.py:77` | `One run: data, model` | `Build and train a fresh run or the requested resume/polish-only rerun; return its final validation score.` |
| `Clean/model_augmentation/fit_systems/blocks.py:433` | `x_tilde^n_{k+1}` | `Return the normalised baseline transition at beta = free without changing the block.` |
| `Clean/model_augmentation/fit_systems/closed_loop.py:121` | `deepSI's four window arrays` | `Check window alignment and append each window's controller row and loss weights.` |
| `Clean/model_augmentation/fit_systems/closed_loop.py:169` | `Closed-loop free-run RMS` | Use F08. |
| `Clean/model_augmentation/fit_systems/interconnect.py:237` | `Connect an input signal` | `Connect two signals using the selected concatenation or additive mapping.` |
| `Clean/model_augmentation/fit_systems/interconnect.py:339` | `Roll the model forward` | `Return (y_pred, x_final) through the attached simulator, with y_pred shaped (batch, nf, ny).` |
| `Clean/model_augmentation/fit_systems/interconnect.py:360` | `deepSI's batch loop` | `Run Adam with periodic validation, save best and last models, and reload the selected model.` |
| `Clean/model_augmentation/fit_systems/interconnect.py:498` | `Pair theta_frozen` | `Synchronise theta_frozen, rebuilding on request or a missing basis; return whether OBC is attached.` |
| `Clean/model_augmentation/fit_systems/obc.py:95` | `reference_field: network -> F` | `Store the reference-field evaluator and the basis builder returning (J, seconds).` |
| `Clean/model_augmentation/fit_systems/obc.py:105` | `Rebuild the basis` | `Refresh at detached vbar, retaining a valid basis on rank loss until the configured abort limit.` |
| `Clean/model_augmentation/fit_systems/obc.py:136` | `theta_aux for this objective` | `Solve theta_aux for the current reference field on the fixed basis, differentiably through F.` |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:37` | `Record-level split` | `Hold out the last listed record of each name class containing at least three records.` |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:53` | `The normalised (u, y)` | `Prepare normalised training records and the decoder specification, with n_xb physical and n_a added states.` |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:76` | `One step after an Adam update` | `Update S1 and check calibration on its evaluation grid, running S2 when S1 ends successfully.` |

The private multiline helper `_without_learned` (`Clean/gantry_dynamic/closed_loop_report.py:12`) is called outside its module too, so the same rule applies. Existing text starts `A deep copy of the model without the learned part`. Proposed replacement: `Return a copy with the network and OBC correction disabled, optionally changing the physical training coordinate.`

Private multiline helpers called only inside their module do not violate the externally-called function rule merely by being multiline.

### S03. Missing one-line descriptions at cross-module call surfaces

These findings have no existing comment to quote; the quoted signature identifies the missing docstring. This is a standard finding, not a behavior defect. Avoid documenting every trivial constructor or property simply to increase the count.

| Location and signature | External use | Proposed docstring |
| - | - | - |
| `Clean/model_augmentation/utils/deepSI_corrections.py:41`, `transform(self, sys_data)` | model fitting, polish and PS2 | `Return normalised WindowedSystemData, or a list of normalised records.` |
| `Clean/model_augmentation/fit_systems/pre_encoder.py:64`, `forward(self, uhist, yhist)` | model initialization and fitting | `Estimate physical and added initial states from input and output histories.` |
| `Clean/model_augmentation/fit_systems/interconnect.py:55`, `forward(self, x, u, ...)` | rollouts in `closed_loop.py`, PS2 and prefit | `Return (y, x_next), subtracting the attached OBC correction from the next state.` |
| `Clean/model_augmentation/fit_systems/blocks.py:40`, `forward(self, z)` | interconnect | `Apply the network and its per-output gate, returning (batch, nw, 1).` |
| `Clean/model_augmentation/fit_systems/blocks.py:62`, `forward(self, z)` | interconnect | `Return C x + D u from the concatenated state and input.` |
| `Clean/model_augmentation/fit_systems/blocks.py:71`, `forward(self, z, mats=None)` | interconnect | `Evaluate the subclass's nonlinear map with optional per-objective matrices.` |
| `Clean/model_augmentation/utils/torch_nets.py:40`, `forward(self, X)` | network block | `Return the physical correction and added-state update from X.` |

Class docstrings already describe several of these maps; if Dirk allows the class documentation to satisfy the callable's description, the five `forward` additions can be omitted to reduce duplication. The literal stated function rule does not currently say that.

### S04. Repeated attribution and missing notation bridge

`Clean/gantry_dynamic/model.py:29` quotes `(Hoekstra et al.)`; `Clean/model_augmentation/fit_systems/pre_encoder.py:49` quotes `Xavier uniform as Hoekstra et al. prescribe for psi_a`. The earlier review requested one attribution for the encoder method. Keep the attribution at the encoder and change the model comment to `# THEORY: na = nb history samples precede the window start; the current sample is excluded`.

`Clean/model_augmentation/fit_systems/pre_encoder.py:49` uses `n_a` in `a = sqrt(6 / (n_a + (n+1)(nu+ny)))`, and `Clean/gantry_dynamic/config.py:27` says `n_a added states`. No comment bridges that symbol to the thesis's `n_{bar x}`, as requested by earlier R10. The formula is correct for this equal-history encoder; the thesis uses `n_{bar x}` and `nu+ny=6`. Replace the config phrase with `n_a = n_{bar x} added states`, once. Keep `na`, `nb` distinct from `n_a`.

### S05. Routine restatement and historical justification remain

| Location | Quote | Why it breaks the standard; proposed replacement |
| - | - | - |
| `Clean/gantry_dynamic/model.py:46` | `fixed output map, not learned` | This inline repeats `Linear_Output_Block`'s buffer contract; remove it, or give the reason `# preserve the measured-position relation`. |
| `Clean/gantry_dynamic/config.py:66` | `the five axes` | The dataclass fields already enumerate them; remove. |
| `Clean/gantry_dynamic/config.py:166` | `axis properties` | A routine label over short properties; remove. |
| `Clean/gantry_dynamic/config.py:191` | `derived quantities` | A routine label over hp; remove. |
| `Clean/gantry_interconnect_dynamic.py:100` | `data and normalisation` | Restates the next calls in a long function; remove this particular step label. |
| `Clean/model_augmentation/fit_systems/interconnect.py:410` | `Adam` | Restates the optimizer loop; remove. |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py:187` | `start validation and the fixed batch` | Restates the next two calls; remove. |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py:278` | `final validation` | Restates `_validate`; remove. |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:131` | `a fixed seed of its own for the calibration windows` | Restates explicit `default_rng`; remove unless the reason for independence is needed. |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:198` | `a fixed seed of its own for the snapshot windows` | Same issue; remove. |
| `Clean/model_augmentation/utils/torch_nets.py:13` | `the layers draw their default initialisation in this order` | The order is visible; if preserving draws is the real reason, use `# keep constructor draws in this order to preserve seeded initialisation`. |
| `Clean/gantry_dynamic/config.py:114` | `tolerances 1e-12 and 1e-14 of the unweighted loss, times 2.6e7, the measured factor` | Retains calibration history and implies a universal relation between objective and gradient scaling; the factor is empirical. Use `# HEURISTIC: fixed gradient and loss-change tolerances for the servo-weighted objective`. The products are numerically correct. |

The five restricted-copy provenance headers are deliberate exceptions explicitly accepted in the earlier review; do not delete them as historical clutter. Likewise, changed-block markers and comments preserving initialization draw conventions have a real purpose.

### S06. Literal no-dashes claim is too broad

There are no em dashes, decision numbers, job identifiers or thesis equation/section references in Clean comments or docstrings. Hyphenated prose remains, for example `Clean/gantry_dynamic/config.py:40`, `run-environment fields`, and `Clean/gantry_dynamic/config.py:105`, `initial square-root loss`. If the writer's claim meant no prose hyphens, these are exceptions. Proposed phrases: `run environment fields` and `initial square root loss`. Mathematical minus signs, indexing such as `q[2:-2]`, file paths, genuine CLI flags and established method names such as L-BFGS must remain intact. The substantive standard bans em dashes, not mathematical subtraction.

### S07. One assertion has no remedy

`Clean/model_augmentation/utils/deepSI_corrections.py:48`:

> System_data is already normalized

The condition is correctly reported, but the message does not say what to do. Replace with `System_data is already normalized; pass the original unnormalised data`. Assertions that check pure internal tensor shapes in `interconnect.py:61` have no message; adding prose there is lower priority than the user-facing normalisation error.

## 4. Earlier-review application checklist

### Naming recommendations, sections 1 and 3

| Earlier item | Applied? | Evidence |
| - | - | - |
| R1 `AugmentedModel` | Yes | `interconnect.py:474`, model construction, `_dumps`; harness keeps the mainline spelling on its mainline path. |
| R2 `OBCProjection`, `projection` | Yes | `obc.py:89`, `obc_gantry.py:197`. |
| R3 `AugmentedEncoder` | Yes | `pre_encoder.py:14`, import, construction and explicit super call. |
| R4 `AugmentationNetwork` | Yes | `torch_nets.py:29`, model network argument and explicit super call. |
| R5 `ValidationHistory` | Yes | `training.py:153`, construction at line 273. |
| R6 `_sync_obc_coefficient` | Yes | Definition and the six calls across interconnect, model, polish and PS2; refresh/missing-basis behavior is now described. |
| R7 `_deriv_with_tangent` | Yes | Definition plus four RK4 calls in `blocks.py`. |
| R8 `_CONTROLLER_METADATA` | Yes | Registry assignment, registration and two lookups; values correctly described as metadata. |
| R9 `_tf_to_block_diag_ss` | Yes | Controller helper definition and call. |
| R10 locals `n_a`, `nx`, `raw_names`, `raw_nominal`, `initial_combinations` | Yes for names; notation bridge not applied | `model.py:26` to 39; literal key `NX_ANN` retained. See S04 for `n_{bar x}`. |
| R11 config `signs` | Yes | `config.py:189`. |
| R11 checkpoint locals `optimizer`, `optimizer_state`, `optimizer_state_dropped` | Yes | `training.py:68` to 71 and 141. |
| R11 selection locals `adam_rows`, `last_by_event` | Yes | `training.py:308`, 315. |
| R11 save locals `polish_accepted`, `polish_only`, `drop_optimizer_state`, `drop_reason` | Yes | `training.py:326` to 338; `polish_only` still tests start_phase, distinct from adam_ran. |
| R11 scorer `parameter` | Yes | `closed_loop.py:175`. |
| R11 interconnect `model_dtype`, `selected_checkpoint`, `correction`, `theta_aux` | Yes | Loader, reload, coefficient-sync and objective blocks. |
| R11 history-value counts | Yes | `pre_encoder.py:47` to 48. |
| R11 entry-point `fh` | Yes | Config file reads and writes. |
| R12 `state_map_loss`, `use_compiled` | Yes | `ps2_opening.py:207`, `closed_loop.py:105`. |
| Section 2 rejected keep: `WindowedSystemData`, `SystemDataNormalizer` | Yes | Definitions, constructor uses and imports all renamed; inherited method names retained. |
| Section 3 `GantryStateBlock`, `_TrainableGantryStateBlock`, `IdentifiableGantryStateBlock` | Yes | Definitions, inheritance and imported references updated. |
| `DtypeDataLoader` | Yes | Definition and construction in interconnect. |
| `build_mlp` | Yes | Definition and network/encoder calls. |
| `_replace_with_retry` | Yes | Definition and checkpoint-save call. |
| `mass_matrix_at`, `stage_frequency_response`, `normalized_controller_coefficients` | Yes | Definitions and controller calls. |
| `PHYSICAL_STATE_IX`, `_MAX_ENCODER_UNEXPLAINED_FRACTION` | Yes | Definitions and all uses in model. |
| OBC construction locals `candidate_basis`, `baseline64`, `correction`, `reference` | Yes | Refresh and attach paths. Existing helper argument names `blk64`, `ref` retained, as scoped recommendation allowed. |
| Prefit locals `as_tensor`, `as_vector`, `input_windows`, `output_windows`, `state_targets` | Yes | Simulation and regression construction. |
| PS2 `_decoder`, `_fit_added_state_transition`, `_added_states` | Yes | Definitions and calls, no payload-key renaming. |
| `FixedWindowBatch` | Yes | Definition and polish construction. |

No requested identifier rename is missing. The missing R10 comment bridge is separate from identifier spelling. Preserved Jan names, module filenames, parameter and buffer attributes, dictionary keys and evaluation interfaces match the earlier keep instructions. `_REGISTRY` and `_tf_to_ss_batch` were the actual previous spellings behind R8 and R9 in the reviewed base.

### Marker plan, section 5

| Item | Applied? | Evidence |
| - | - | - |
| 1 retain existing provenance | Yes | The 12 existing `@added` decorations remain on the corresponding renamed definitions, including `Linear_Output_Block` and `to_tensor`; six new-file origin markers remain in model_augmentation. |
| 2 replace 26 trim comments with five headers | Yes | Headers follow module docstrings in blocks, interconnect, pre_encoder, torch_nets and utils. |
| 3 keep changed-block reasons | Yes, wording partly needs work | 25 `# CHANGED:` occurrences remain, including import dependencies and `norm` initialization. See W01 for the subtraction reason. |
| 4 add seven decorators | Yes | `Interconnect._apply`, `_signal_cone`, `output_only`; `SSE_Interconnect.simulate`, `make_training_data`, `cal_validation_error`, `fit`. `_propagate` remains decorated. |
| 5 remove redundant markers but retain import provenance | Yes | Added definitions/new files no longer carry redundant internal markers; changed import groups retain markers. |
| 6 retain theory/heuristic labels with correct content | Partly | Counts are now 20 and 17. Formula labels are retained; F01/F02 and S05 identify content needing correction. |
| 7 do not add a new CLAUDE rule | Yes in the reviewed commit | The specified commit diff is confined to Clean and the three harness adaptations; existing unrelated workspace edits to CLAUDE.md are not attributed to this pass. |

### Sample corrections, section 6

| OBC sample item | Applied? | Evidence |
| - | - | - |
| 1 sensitivity plus offset in module description | Yes | `obc.py:1`. |
| 2 scaled basis and numerical rank, including rejected candidates | Yes | `obc.py:16` to 20. |
| 3 include D^-1 and full-rank condition | Yes | `obc.py:39`; formula exactly matches division at line 43. |
| 4 distinguish affine basis and coordinates | Yes | `obc_gantry.py:160` to 162; `(a,c)` becomes `(a+beta_lin c,c)` on the thesis basis; correction is `varphi a + c f_base^n`. |
| 5 Phi sensitivity, beta/beta_lin | Yes | `obc.py:59`, connected to `transition_from_free` at `blocks.py:433`. Generic `step` notation is appropriate for this generic function. |
| 6 step values are transitions, not gamma | Yes | `obc.py:77`. |
| 7 rank threshold belongs to scaled J | Yes | `obc.py:27` to 29. |
| 8 configurable rank-loss policy and differentiation through F | Yes factually; one-line rule not applied | `refresh` uses max_consecutive_rank_loss, coefficient keeps autograd. See S02. |
| 9 single-sentence actionable rank errors | Yes structurally; wording partly | Messages now include actions and the affine column; W06 tightens which dependencies matter. |

| Model sample item | Applied? | Evidence |
| - | - | - |
| 1 one-line build_model docstring | Yes | `model.py:25`. |
| 2 normalised predicted-state/output formulas | Yes | `model.py:60` to 62. Added states use x_bar, as the thesis does; they have no separate normalisation. |
| 3 dimensionally correct row-wise updates including OBC | Yes | Same block explicitly separates physical and added rows. |
| 4 normalised physical-state row description | Yes | `model.py:20`. |
| 5 sample convention and one attribution | Partly | 30 versus 29 convention corrected; repeated encoder attribution remains, S04. |
| 6 encoder linear term and nonlinear-input permutation | Yes | `pre_encoder.py:68` and 76, model formula at line 78. |
| 7 zero physical output weights/bias, live added rows | Yes | `torch_nets.py:30` to 37, with shared final-bias initialization in build_mlp. |
| 8 physical start before normalisation and row-batch transpose | Yes | `model.py:149` to 153. Recorded y_0 distinguished from norm.y0. |
| 9 recorded closed-loop input and regression alignment | Yes for configured equal histories | `model.py:157`, 169; general wording qualification in W10. |
| 10 affine approximate fit and transposed returned W | Yes | `model.py:184`; W has regressors by physical states. |
| 11 standardized regressors and restrained SVD-driver claim | Yes | `model.py:192` to 199; no last-bit reproducibility guarantee remains. |
| 12 physical rows become affine map, unexplained threshold | Yes on nonconstant targets | `model.py:205` to 228; W10 qualifies R^2 at zero variance. |
| 13 remove experimental history from fit-acceptance heuristic | Yes | `model.py:179`. |
| 14 remove routine comments | Partly | Routine grouping comment removed; fixed-output-map inline remains, S05. |
| 15 CUDA and Adam eps errors actionable | Yes | `model.py:88` to 99; literal CLI spellings preserved. |

### Section 7 decisions and syntax/string requirements

All three are applied: Kessels is named once for the rollout idea (`closed_loop.py:1`), the constants module docstring precedes its origin marker (`gantry_ss.py:1`), and the seven missing decorators are added. The residual equations and zero controller-difference initial condition are documented near the rollout, without claiming Kessels supplied that exact initialization.

All 24 files parse. A structural AST comparison matches 24/24 after removing standalone documentation strings, normalising the observed scoped identifier renames, accounting for the seven identity decorators, allowing changed print/error arguments, the OBC rank-error prefix local and the approved class-derived dump filename. The old misplaced constants description is a standalone string, not a module docstring. This is a qualified static equivalence check, not a claim of literally identical source ASTs or numerical validation.

Separately compared non-docstring string constants: dictionary keys, event strings, PS2 statuses and payload strings, polish outcome reasons, config values, and state attribute names remain unchanged. Differences are error messages, ordinary console prints, the approved `_dumps` class prefix, and the former standalone constants description. In particular `training rollout COMPILED` and `PS2` in the unfinished-opening error remain. The harness diff adapts only the clean/mainline class lookup and dump-name comparison/use in `compare_l3bc.py`, `gpu_smoke.py`, and `run_l3bc.py`. No harness was run or edited here.

The claimed counts are confirmed when counting actual decorators: 25 changed markers, 19 added decorators (12 before, seven new), 20 theory labels and 17 heuristic labels. Literal `@added` text appears 23 times because four occurrences are explanatory text. The five framework-copy headers and four empty package initializers are confirmed. No D-numbers, job IDs, em dashes or thesis section/equation references remain in comments/docstrings. Hyphenated prose remains as explained in S06.

`python -m pyflakes Clean` could not run because this environment has no pyflakes module. Thus the writer's earlier pyflakes-clean claim is not independently re-established here. No packages were installed. No project object was imported to test callable behavior; the decorator's unchanged identity implementation and AST checks support the static provenance conclusion.

Empirical statements about the measured initial loss, filter transient across 36 records and scaling calibration are not proven by nearby code. Their constants and intended checks are consistent with the cited thesis setup material, but no data or experiment was rerun. The Butterworth formula and its quoted gain errors were checked arithmetically with fs = FS_ORIG = 20000: approximately 1.67755e-12 at 300 Hz and 4.39410e-4 at 1 kHz. The filter is designed at that original rate in data.py, despite the cutoff being chosen relative to the training-rate Nyquist frequency.

## 5. File verdicts

Paths below are relative to `Clean/`. A pass means no additional factual defect in that file's remaining comments, not approval of every style choice elsewhere. The full coverage ledger follows.

| File | Verdict |
| - | - |
| `gantry_dynamic/__init__.py` | Pass: package-root guard explanation and remedy match the comparison. |
| `gantry_dynamic/closed_loop_report.py` | Factual pass; S02 for cross-module private helper docstring length. |
| `gantry_dynamic/config.py` | Formulas and current values pass; S02, S04 to S06 for standard/notation. Empirical calibration not rerun. |
| `gantry_dynamic/controller.py` | F05, F06, F10; S02. Controller formulas, scaling and state-space assembly otherwise pass. |
| `gantry_dynamic/data.py` | F01/F02; S02. Stencil-independent first-difference statistics, normalization and output scaling otherwise pass. |
| `gantry_dynamic/model.py` | W10, S04/S05. Architecture, OBC sign, normalization, prefit transpose and initialization otherwise pass. |
| `gantry_dynamic/obc_gantry.py` | F09, S02. Stencil, successor indices, sample-major order, affine coordinate mapping and correction otherwise pass. |
| `gantry_dynamic/training.py` | F05/F07, W08/W09, S02. Checkpoint formats, optimizer-state merging, selection and save policies otherwise pass. |
| `gantry_interconnect_dynamic.py` | S01/S02/S05. Fresh-run guard, rerun archiving and CLI defaults otherwise pass. |
| `model_augmentation/__init__.py` | Pass: empty. |
| `model_augmentation/fit_systems/__init__.py` | Pass: empty. |
| `model_augmentation/fit_systems/blocks.py` | W02/W07, S02/S03. RK4, tangent operations, parameter-coordinate formulas and gauge otherwise pass. |
| `model_augmentation/fit_systems/closed_loop.py` | F03/F06/F08/F15, W05, S02. Controller normalization, residual-loop sign, scorer indexing and RMS averaging otherwise pass. |
| `model_augmentation/fit_systems/interconnect.py` | F14, W01, S02/S03/S05. Fixed-basis coefficient differentiation, checkpoint resync and marker changes otherwise pass. |
| `model_augmentation/fit_systems/lbfgs_polish.py` | F11/F12; S02/S05. Rollback, finite-loss and strict-validation acceptance otherwise pass. |
| `model_augmentation/fit_systems/obc.py` | W06; S02. Scaled SVD formula, numerical rank and configurable failure policy pass. |
| `model_augmentation/fit_systems/pre_encoder.py` | S03/S04. Initialization bounds and encoder permutation/formula pass for the configured equal histories. |
| `model_augmentation/fit_systems/ps2_opening.py` | F04, W03/W04; S02/S05. Residual normalization, calibration ratio, S2 Bessel factor and retained-best rule otherwise pass. |
| `model_augmentation/systems/__init__.py` | Pass: empty. |
| `model_augmentation/systems/gantry_ss.py` | Pass: constants match the stated MATLAB source, polynomial inverse and LFR blocks match the thesis algebra. |
| `model_augmentation/utils/__init__.py` | Pass: empty. |
| `model_augmentation/utils/deepSI_corrections.py` | S03/S07. Normalization and dtype-preserving window construction otherwise pass. |
| `model_augmentation/utils/torch_nets.py` | F13, S03/S05. Physical-zero versus live added-state initialization passes. |
| `model_augmentation/utils/utils.py` | Pass: identity decorator, tensor pass-through and selection/expansion formulas match the code. |

## 6. Notation reconciliation

The thesis uses `tilde x^n` for six normalized physical states, `bar x` for added states without independent normalization, and `hat x^n = col(tilde x^n, bar x)`. Code `n_a`/`nx_ann`/`nx_aug` is the thesis `n_{bar x}`; code `na=nb=30` is its encoder window `n+1`, so the encoder constructor receives 29. Code `u^n` is normalized actuator force, while the unadorned `u` in the baseline thesis mechanics is generalized force `P u_act`; blocks.py's physical-units formula uses `P u` with that local `u` meaning actuator force. Code `C` in a controller formula denotes feedback output matrices, while `C` in gantry_ss/physics denotes damping and `C^n` in the model output denotes the normalized position map. These are local aliases, not identical matrices.

Code free_params/free is the thesis beta; vbar is beta_lin. The rational inverse's scalar alpha, beta, gamma in gantry_ss and blocks are polynomial constants, so that scalar beta is not the ten-dimensional training coordinate, and gamma_ is not the thesis OBC offset gamma. Code m_total is thesis m_Sigma, m_diff is m_Delta, kb_sum/cb_sum are k_b,Sigma/c_b,Sigma; M_DIFF_RHO/rho is the thesis admissibility margin gamma. Code a in the acceleration calculation is qddot; OBC a is a ten-vector of fitted sensitivity coefficients. Code E embeds protected physical rows into the full augmented state, whereas the thesis f_obc^n is itself six-dimensional.

In obc.py, generic J is the gantry `[Phi F_base^n(beta_lin)]` on selected rows, and generic F is stacked F_aug^n. With D = diag(column_scale), J D^-1 = U S V^T and theta_code = D^-1 V S^-1 U^T F for the accepted full-column-rank basis. The code coefficient `(a,c)` maps to thesis theta_aux `(a+beta_lin c,c)` on `[Phi Gamma]`; vbar and theta_frozen retain code coordinates. This relation is correctly documented in the new comments. A rank-deficient candidate uses truncated factors but is not accepted as the live basis.

PS2 `_M` is thesis H, not the mass matrix M(Y); its residual sigma_e is a shared batch RMS of normalized model residuals, not the per-record physical servo-error standard deviation used by the outer objective. `03_augmentation.tex` defines those roles separately. S2's target variance uses correction=1, so for a fixed batch its implemented ratio is `(B-1)/B` times the population-denominator thesis cost, as its comment says. In the encoder least-squares fit, Phi is a history-regressor matrix, unrelated to OBC Phi; returned W is the transpose of thesis W^b. No independent symbol mismatch beyond these local aliases and the missing S04 bridge was found.

## 7. Per-comment and per-docstring coverage ledger

Each row below is one docstring or one contiguous comment block; all its lines were checked. `Pass` means no additional finding for that unit after the local notation mappings above. Findings elsewhere that concern an absent docstring or an error/print string are listed in the main review and do not create a comment token in this ledger. Entries marked S02 are the length violations identified above. Private internal multiline docstrings are not automatically length violations. Text is quoted as stored, with line breaks collapsed for scanning.

### `Clean/gantry_dynamic/__init__.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 | docstring | The gantry side of the training code: configuration, data, controller, model, OBC and training. | Pass |
| 8 to 9 | comment | # this package must run with the model_augmentation of the same folder; a launch with the repository root first # on sys.path would mix it with the mainline one without an error | Pass |

### `Clean/gantry_dynamic/closed_loop_report.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 | docstring | Two helpers of the evaluation: the model without its learned part, and the coordinate of the true parameters. | Pass |
| 12 to 13 | docstring | A deep copy of the model without the learned part (network gate and theta_frozen zeroed), with the physics block at training coordinate free if given; the model passed in is not changed. | S02 |
| 28 | docstring | The training coordinate beta at which the physics block holds the true combinations. | Pass |

### `Clean/gantry_dynamic/config.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 | docstring | Run configuration: the five axes of a thesis run, the run environment and the fixed settings. | Pass |
| 19 to 21 | comment | # HEURISTIC: order 8, fc = 0.8 of the 2 kHz Nyquist frequency of the 4 kHz training rate # THEORY: zero-phase Butterworth magnitude 1 / (1 + x^16), x = tan(pi f / fs) / tan(pi fc / fs) (the order-8 #   response squared by the forward-backward pass): DC gain 1, error 1.7e-12 at 300 Hz and 4.4e-4 at 1 kHz | Pass |
| 23 to 24 | comment | # HEURISTIC: the filter's edge transient stays above 0.01 of the in-band noise for at most 4.6 ms on the 36 # records; rounded up to whole ms | Pass |
| 27 | comment | # the thesis runs: OBC or unconstrained arm, correct or detuned start, n_a added states, PS2 opening or plain | S04 |
| 40 | comment | # config.json keys of the run-environment fields, read back by from_config_json | S06 |
| 45 | comment | # config.json keys only the clean code writes | Pass |
| 50 | docstring | The rule that brings y to FS_NEW, as recorded in config.json. | Pass |
| 57 to 58 | docstring | Samples dropped at each record end at FS_NEW: loader sample i is the recorded FS_ORIG sample (i + record_trim) D. | S02 |
| 64 | docstring | One thesis run: the five axes, the run environment and the fixed settings. | Pass |
| 65 | comment | # the five axes | Pass |
| 71 | comment | # run environment: production defaults, changed only by the CLI flags, main (start_phase) and the harness | Pass |
| 83 | comment | # roots: recorded in config.json, never read back from it | Pass |
| 87 to 88 | comment | # fixed settings of the thesis runs under the mainline names; config_json_dict writes them and the # evaluation reads several | Pass |
| 105 | comment | # HEURISTIC: the measured initial square-root loss; only the Lambda_combo buffer uses it, and no loss reads it | S06 |
| 114 to 115 | comment | # HEURISTIC: tolerances 1e-12 and 1e-14 of the unweighted loss, times 2.6e7, the measured factor by which the # servo-error weighting scales the loss and its largest gradient entry | S05 |
| 123 | comment | # derived; nf = steps per training window (n_f), also the length of the gradient path | Pass |
| 131 | docstring | Check the axis and environment values, which arrive from the CLI, a config.json or the harness. | Pass |
| 145 to 146 | docstring | A run's config.json, clean or mainline, as a RunConfig; a setting outside the thesis runs raises. Roots and start_phase come from overrides or the defaults. | S02 |
| 166 | comment | # axis properties | S05 |
| 187 to 188 | comment | # HEURISTIC: 10 % per combination, the sign of each drawn from the seed, so a seed fixes the direction # and its noisy and noise-free runs share it | Pass |
| 192 | comment | # derived quantities | Pass |
| 195 | docstring | The hyperparameter dict; its keys and order are those of the checkpoint .npz field hp. | Pass |
| 210 | docstring | Output folder of the run family; the caller creates the run folder below it. | Pass |
| 216 | docstring | The code that produced this run, e.g. '0bded3d Augmentation dirty:302 diff:3b437e0c'. | Pass |
| 242 to 243 | docstring | The config.json of a run: every mainline key in mainline order (options the clean code lacks as the values the thesis runs wrote), then the clean-only keys and GIT. | S02 |

### `Clean/gantry_dynamic/controller.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 | docstring | The known feedback controller Cfb, from the physical parameters to the ControllerBank of the closed loop. | Pass |
| 15 | comment | # torch.compile backend of the training rollout; the mode is cfg.compile_mode | Pass |
| 18 | comment | # THEORY: physical parameters of the controller design (gtd_config.m) | Pass |
| 23 | comment | # THEORY: gtd_config.m, design bandwidth [Hz] | Pass |
| 26 | comment | # THEORY: stage-to-logical transform P and the damping and stiffness matrices (getss.m) | Pass |
| 35 | docstring | Mass matrix M(Y) at the frozen position Y_op. THEORY: gtd_build_plant.m. | Pass |
| 44 | docstring | P^T G(s) P at s, with G the logical plant frozen at Y_op. THEORY: getss.m. | Pass |
| 53 | docstring | Cnorm(s) = 10 W (s + W/6)(s + W/3) / (s (s + 3 W)(s + 10 W)) as (num, den). THEORY: ruleOfThumb.m. | Pass |
| 60 | docstring | Three discrete SISO controllers, one per stage channel, at sample time ts. | Pass |
| 66 | comment | # THEORY: k_j = 1 / \|G_jj(jW) Cnorm(jW)\|, unit loop gain at the bandwidth (ruleOfThumb.m) | Pass |
| 74 | docstring | Per-channel (b, a) to one block-diagonal (A, B, C, D) of the diagonal Cfb. | Pass |
| 80 | docstring | Cfb at sample time ts as one 3-input, 3-output state space (A, B, C, D), scaled by gain. | Pass |
| 86 | docstring | A record's file name without folder and .mat suffix: the key of _CONTROLLER_METADATA. | Pass |
| 91 | comment | # stem -&gt; (design Y, gain) of a stored controller the bank can build, or the description of one it cannot | Pass |
| 96 | docstring | Register a record's stored controller: (design Y, gain) if the bank can build it, else its description. | Pass |
| 102 | docstring | Controller design position Y of a record, by file name with or without .mat. | Pass |
| 117 | docstring | Stored gain of a record's controller, 1.0 unless stored otherwise. | Pass |
| 128 | docstring | The ControllerBank of a set of records, each record's row in it, and the distinct design positions. | F10 |
| 139 to 140 | docstring | The ClosedLoopSimulator of this model; assign it to fs.simulator. Validation pairs the records with the rows of val_files in order; val_data is unused and kept for the callers. | S02 |
| 156 to 157 | comment | # compiled here and handed to the simulator, which uses it only for the training loss on CUDA, so # validation stays eager | F06 |
| 171 | comment | # loss weight W_r = diag(sigma_y / sigma_e,r) per training record | Pass |

### `Clean/gantry_dynamic/data.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 | docstring | Data loading (the manifest split of one data set), resampling to the training rate and normalisation. | Pass |
| 18 | comment | # the thesis data sets below cfg.data_root, one folder per noise setting | Pass |
| 20 | comment | # manifest block -&gt; split | Pass |
| 25 | docstring | (train, val, test) file names relative to traj_dir(cfg), in manifest order. | Pass |
| 30 | comment | # K1 records only: every thesis record is generated with the controller K1 | Pass |
| 37 | docstring | Register the record's stored controller (meta.controller) for the closed-loop bank. | Pass |
| 47 | docstring | The records of the three splits: System_data at FS_NEW, file names and servo errors e = y - r. | Pass |
| 63 | docstring | Normalisation constants of the physical states, inputs and outputs, and the output map C^n. | Pass |
| 75 | docstring | Folder of the data set of cfg.noise. | Pass |
| 80 | docstring | The plant input u_total = [F_X1, F_X2, F_Y] of a record. | Pass |
| 85 to 91 | docstring | The plant force from FS_ORIG to FS_NEW as the mean over each hold interval.  # THEORY: u_total is a zero-order hold at FS_ORIG (the 20 kHz controller), so the exact input of a # zero-order-hold model at FS_NEW is the mean force over each FS_NEW hold interval. Point sampling u[::D] # leaves a mean force error, which the stiffness-free X and Y axes turn into a position offset. The force is # band-limited far below the 2 kHz Nyquist frequency, so this is not anti-aliasing. | F01, F02, S02 |
| 98 | docstring | y at FS_NEW: zero-phase anti-alias filtering of y - r on the lowpass data set, point sampling otherwise. | Pass |
| 104 | comment | # only the servo error is filtered; the reference r is added back unfiltered | Pass |
| 109 | docstring | [u, y, *fields] of one record at FS_NEW on the same samples, trimmed at both ends. | Pass |
| 122 to 123 | docstring | (System_data, e) of one record from one read of its file, e = y - r at FS_NEW in float64; read from folder if given, else from traj_dir(cfg). | S02 |
| 133 to 134 | docstring | sigma_e per record and axis: the population standard deviation of e = y - r over the scored samples k0 .. N-1, shape (n_records, ny). | S02 |
| 135 | comment | # THEORY: the NRMS normalisation by a standard deviation (Beintema et al.), here that of the servo error | Pass |
| 144 | docstring | Load the train, val and test records of the manifest split. | Pass |
| 173 | docstring | Normalisation constants of the six physical states, the inputs and the outputs, from the training records. | Pass |
| 181 | comment | # logical positions q = P^-T y, velocities by a first difference | Pass |
| 183 | comment | # THEORY: var of a first difference of white noise = 2*sigma_n^2, so sigma_v = sigma_n*sqrt(2)*fs | Pass |
| 196 | comment | # C^n = diag(1 / sigma_y) C diag(sigma_x), so y^n = C^n x_tilde^n: the offsets cancel, since y0 = C x_mean | Pass |

### `Clean/gantry_dynamic/model.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 | docstring | Model construction (interconnect and encoder) and the least-squares start of the encoder. | Pass |
| 20 | comment | # rows of the physical state x_tilde^n (normalised positions and velocities) in the model state x_hat^n | Pass |
| 25 | docstring | Build the untrained augmented model with its encoder, and on the OBC arm its projection. | Pass |
| 30 to 31 | comment | # THEORY: the encoder reads the na = nb samples before the window start, not the start sample itself # (Hoekstra et al.) | S04 |
| 36 | comment | # the baseline is trained in the ten identifiable combinations; the detuned arm starts them off nominal | Pass |
| 46 | comment | # fixed output map, not learned | S05 |
| 60 to 62 | comment | # the augmentation structure, row by row, with x_hat^n = (x_tilde^n, x_bar) and phi_aug = (f_aug^n, g_aug^n): #   x_tilde^n_{k+1} = f_base^n(x_tilde^n_k, u^n_k) + f_aug^n(x_hat^n_k, u^n_k)   (- f_obc^n on the OBC arm) #   x_bar_{k+1} = g_aug^n(x_hat^n_k, u^n_k),   y_hat^n_k = C^n x_tilde^n_k   (D = 0) | Pass |
| 78 to 79 | comment | # psi(chi) = [W^b; W^a] chi + psi_tilde(chi); prefit_physical_encoder fits its physical rows before training. # The class reads n + 1 samples for its argument n, so na - 1 gives the na-sample window. | Pass |
| 88 | comment | # no fallback to the CPU: config.json records the device the run asked for | Pass |
| 109 | docstring | Move the model, built and kept on the CPU, to cfg.device for the work inside the block. | Pass |
| 120 | docstring | Start the encoder's physical rows at a least-squares fit to a closed-loop simulation of the start model. | Pass |
| 126 | docstring | Simulate every training record in closed loop, then fit the encoder to the simulated physical states. | Pass |
| 149 to 150 | comment | # start state: q_0 = P^-T y_0 from the first recorded output, zero velocity, then normalised; zero # added states | Pass |
| 157 | comment | # the state and the closed-loop input u_hat^n = u^n + u_fb^n of every step | Pass |
| 169 | comment | # window s: chi from u_hat^n and y_hat^n at s .. s+k0-1, target x_tilde^n at s+k0, k0 = max(na, nb) | W10 |
| 179 | comment | # HEURISTIC: require R^2 &gt;= 0.999 for every physical state on the simulated encoder-fit windows | Pass |
| 184 | docstring | Fit x_tilde^n ~ W^b chi + b and set the encoder's physical rows to it; returns (W, b), W = (W^b)^T. | Pass |
| 192 | comment | # least squares on centred, standardised regressors, mapped back to chi below | Pass |
| 195 | comment | # THEORY: rcond = eps max(M, N), the numpy.linalg.lstsq default | Pass |
| 197 | comment | # the SVD driver, for the rank-truncated least-squares fit | Pass |
| 209 to 210 | comment | # physical rows of the output layer of psi_tilde: zero weights, and the bias that makes the physical # rows W^b chi + b (it removes the encoder's output at chi = 0) | Pass |
| 215 | comment | # 1 - R^2 per state | W10 |

### `Clean/gantry_dynamic/obc_gantry.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 to 2 | docstring | Gantry part of the orthogonal projection: the reference tuples, the basis of the baseline transition, the network on the tuples and the correction f_obc^n. | Pass |
| 23 | comment | # the float64 P^-T: measured stage positions y to logical positions q = P^-T y | Pass |
| 29 | docstring | The fixed reference tuples (x_tilde^n_i, x_bar_i = 0, u^n_i) over all training records. | Pass |
| 43 to 47 | docstring | qdot_k = (-q_{k+2} + 8 q_{k+1} - 8 q_{k-1} + q_{k-2}) / (12 ts) for k = 2 .. N-3, aligned with q[2:-2].  # THEORY: fourth-order central difference (Fornberg 1988); the caller uses it only where the full stencil # lies inside one record. | S02 |
| 53 | docstring | The reference tuples of the training records files. | Pass |
| 62 | comment | # one training record in float64, resampled as for training | Pass |
| 65 | comment | # x_tilde_i = (q_i, qdot_i), q_i = P^-T y_i, at every sample where qdot exists at i and i+1 | Pass |
| 94 | docstring | f_obc^n = E (varphi(beta_lin) a + c f_base^n(beta_lin)) at the current state and input, theta_aux = (a, c). | Pass |
| 106 | comment | # E: the protected physical rows of the model state | Pass |
| 112 | comment | # theta_aux = (a, c): p sensitivity coefficients and the coefficient of the baseline transition | Pass |
| 116 | docstring | The current training coordinate beta, attached; OBCProjection.refresh detaches it. | Pass |
| 121 | docstring | Store theta_aux and beta_lin for the prediction path. | Pass |
| 126 | docstring | (matrices at beta_lin, their tangent in the direction a, c) for theta_aux = (a, c), once per objective. | Pass |
| 132 | docstring | f_obc^n at (x, u); obc_pass is the tangent_pair of this objective, or None for theta_frozen. | Pass |
| 137 | comment | # f_base^n(beta_lin) and its directional derivative varphi(beta_lin) a, from a zero state tangent | Pass |
| 144 | docstring | A float64 copy of the physics block for the basis build; the live block keeps the training dtype. | Pass |
| 151 | docstring | vbar64 -&gt; (J, seconds) for OBCProjection, J = [Phi F_base^n(beta_lin)] in float64 on the reference tuples. | F09 |
| 160 to 162 | comment | # the extra column is the baseline transition F_base^n(beta_lin) itself, not the offset # Gamma = F_base^n(beta_lin) - Phi beta_lin: the same span, so the same projection, but other # coordinates: with coefficients (a, c) here, those on [Phi Gamma] are (a + beta_lin c, c) | Pass |
| 172 to 173 | docstring | ann -&gt; F: the network on the reference tuples, its routed physical columns stacked sample-major, with autograd. | S02 |
| 175 | comment | # the reference inputs, cached per device and dtype | Pass |
| 191 | docstring | Attach the reference set, the correction module and the projection to a built model. | Pass |

### `Clean/gantry_dynamic/training.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 | docstring | Training: Adam with the PS2 opening, checkpoint selection, the L-BFGS polish and the checkpoint files. | Pass |
| 18 | docstring | The outcome dict without numpy scalars, so json.dumps cannot fail on the meta write. | W08 |
| 31 to 32 | docstring | Save the torch components to &lt;base&gt;.pt (the deepSI model has no state_dict of its own) and meta to &lt;base&gt;.npz, one array per value. | S02 |
| 44 | docstring | Read either checkpoint format to (hfn_sd, encoder_sd, optimizer_sd or None, meta, format). | Pass |
| 48 | comment | # read and closed: a rerun moves this file right after loading it | Pass |
| 58 to 59 | comment | # the epoch count is cumulative over chained continuations (epochs_before is the count the dump's job # started from); ps2_opening = (phase, updates) of the opening at these weights | Pass |
| 73 | docstring | Restore the optimizer's moment estimates and keep this run's hyperparameters. | Pass |
| 88 | docstring | Raise naming the mismatch, rather than load_state_dict's list of keys. | Pass |
| 111 | docstring | Refuse a checkpoint whose dtype differs from the run's. | Pass |
| 123 | docstring | Restore the weights, and on an 'adam' start the Adam moments, from either format; returns the meta. | Pass |
| 127 to 128 | comment | # a dump holds no state of the opening itself (decoder, optimizers, calibration, random state), so a # continuation can neither resume it nor restart it on weights it already changed | Pass |
| 154 to 157 | docstring | The validation RMS per record and axis at every validation, and the selection events, in val_hist.npz.  Held here and not on the model, since fit's reload of the best checkpoint overwrites the model's __dict__; hist_path is rewritten at every entry. | Pass |
| 165 to 166 | docstring | Append a row: from the simulator's last validation (events adam, rebuilt, polish_candidate), or a copy of an earlier row passed as row (selected, final). | Pass |
| 177 | docstring | The history as arrays, one entry per row, for val_hist.npz. | Pass |
| 190 | docstring | Validation-probe entry: record the validation that just ran. | Pass |
| 195 | docstring | The L-BFGS polish on cfg.device; adam_ran is False when this job ran no Adam update. | Pass |
| 205 to 207 | comment | # the polish must beat the selected checkpoint's score bestfit, on the OBC arm too. There fit() has # rebuilt the basis at the restored parameters, so the rebuilt model scores slightly differently: it is # measured once for the log and the history, and it is the model kept when the polish is rejected. | Pass |
| 210 to 211 | comment | # without Adam in this job, one eager validation completes the lazy initialisation before the # compiled probe of the polish | Pass |
| 233 to 234 | docstring | Adam for hp['epochs'] epochs with selection on the validation score, then the L-BFGS polish; returns the final score. resume = (path, meta) of the checkpoint main has loaded into fit_sys. | F07, S02 |
| 247 to 248 | comment | # the dumps of this job count on from here; weights and Adam moments continue, the batch stream and an # unfinished PS2 opening do not | Pass |
| 257 | comment | # Adam, with the PS2 opening after each update and the validation history as a probe | Pass |
| 261 to 262 | comment | # a dump written after the opening ended never restarts it, even when its count rounds to 0; without a # record, the count decides | Pass |
| 280 | comment | # the Adam budget is spent, and fit(epochs=0) would still run; polish directly | W09 |
| 303 | comment | # pre-polish checkpoint and the selected row (the latest of equal scores, as fit keeps the latest best) | Pass |
| 313 | comment | # L-BFGS polish; the final row is the accepted polish, else the rebuilt basis (OBC), else the selection | Pass |
| 324 to 325 | comment | # final checkpoint, without Adam state after an accepted polish or in a polish-only job (it would not # belong to these weights) | Pass |

### `Clean/gantry_interconnect_dynamic.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 to 8 | docstring | Entry point of the thesis training runs: a fresh run, a resume or a polish-only rerun.  python -u Clean/gantry_interconnect_dynamic.py [preset] [noise] [--cpu] [--smoke] [--resume RUN_DIR [--from {best,last}] \| --polish-only RUN_DIR] [--data-root P] [--out-root P]  Without arguments: R1 lowpass on CUDA with reduce-overhead compilation; without CUDA use --cpu. The presets R1 to R7, R13 and R14 are the thesis runs (gantry_dynamic/config.py). | S01 |
| 29 | docstring | Paths of the deepSI dumps _best.pth and _last.pth of a run; deepSI names them after the model class. | Pass |
| 35 to 36 | docstring | (config.json, run id, source checkpoint) of a rerun, which continues in its run folder under the folder's run id; nothing moves here. | Pass |
| 54 to 55 | docstring | Move the previous outcome, and for a resume both deepSI dumps, to previous/&lt;rerun_id&gt;/ and add a RESUMES entry to config.json; returns the source's path in the archive. | Pass |
| 77 to 79 | docstring | One run: data, model, closed loop, encoder start, Adam, polish and checkpoints; returns the best score.  rerun = ('resume', run_dir, 'best' \| 'last') or ('polish-only', run_dir, None); None for a fresh run. | S02 |
| 80 | comment | # run folder: a fresh run never writes into an existing one | Pass |
| 100 | comment | # data and normalisation | S05 |
| 114 | comment | # model and closed loop; reseeded, so the initial draws of the model start from the seed | Pass |
| 125 to 126 | comment | # encoder start (it simulates the closed loop, so it follows the controller bank); a rerun restores # the fitted encoder instead | Pass |
| 131 | comment | # the run folder is archived only once data, device and the whole checkpoint have loaded | Pass |
| 141 | docstring | Parse the command line into a RunConfig and a rerun, then start main. | Pass |

### `Clean/model_augmentation/__init__.py`

Empty file; no comments or docstrings.

### `Clean/model_augmentation/fit_systems/__init__.py`

Empty file; no comments or docstrings.

### `Clean/model_augmentation/fit_systems/blocks.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 | docstring | The blocks of the augmented model: Jan's block types and the gantry physics blocks of the baseline. | Pass |
| 2 to 3 | comment | # CHANGED: restricted to the thesis training path; full implementation in # model_augmentation/fit_systems/blocks.py at the repository root | Pass |
| 7 | comment | # CHANGED: explicit names instead of the star import, and gantry_ss for the added gantry blocks | Pass |
| 13 | docstring | Basic block implementation with variables that need to be defined for interconnect. | Pass |
| 23 | docstring | w = net(z), times a gate per output: the network block of the augmented model. | Pass |
| 27 | comment | # CHANGED: net_kwargs reach the network's constructor (the number of physical rows) | Pass |
| 36 to 37 | comment | # CHANGED: a gate per output, 1 in training, applied in forward; _without_learned zeroes it to remove the # network (persistent=False: not in the state_dict) | Pass |
| 48 | docstring | w = C x + D u, with C and D as buffers: they follow .to() and are not trained. | Pass |
| 70 | docstring | w = nonlinear_function(z), defined by the subclass. | Pass |
| 72 | comment | # CHANGED: mats, the matrices built once per objective, are passed on to nonlinear_function | Pass |
| 78 to 79 | docstring | The gantry LPV-LFR baseline (Drenth et al.): the continuous-time model with M(Y)^-1 in rational form, integrated by RK4 over one sample in normalised coordinates. | Pass |
| 89 | comment | # the nominal LFR matrices of gantry_ss; the trainable subclasses rebuild them at every objective | Pass |
| 123 | docstring | up_sample classical RK4 steps of h = Ts / up_sample, with u held. | Pass |
| 134 | docstring | xdot^n at (x^n, u^n) with the matrices mats. | Pass |
| 138 | comment | # physical units inside: x = x^n sigma_x + x_mean, u = u^n sigma_u + u_mean | Pass |
| 147 | comment | # f = P u - K q - C qdot | Pass |
| 153 | comment | # frozen scheduling, set by the evaluation: a = N_op f / d_op at Y = Y_op | Pass |
| 157 | comment | # a = qddot = M(Y)^-1 f = (N0 + Y N1 + Y^2 N2) f / d(Y) | Pass |
| 169 | comment | # LFR: w = Y z, z = (a, Y a), xdot = Ax x + Bw w + Bu P u = A_combined (x, w, P u) | Pass |
| 180 to 181 | docstring | (xdot^n, its tangent): the derivative of _deriv_with and its directional derivative along the state tangent sx and the matrix tangent dmats. | W07 |
| 192 | comment | # every s_* below is the directional derivative of * along (sx, dmats) | Pass |
| 241 | docstring | _rk4 for the state and its tangent together; returns (x_next, sx_next). | Pass |
| 259 | docstring | The gantry block with trainable parameters: its matrices are rebuilt from a raw fourteen-vector. | Pass |
| 270 | comment | # the scale of the subclass's Lambda_combo buffer | Pass |
| 276 | docstring | K and C of a raw parameter vector, built as in gantry_ss; they depend on kb1 + kb2 and cb1 + cb2 only. | W02 |
| 296 | docstring | M1, M2 of M(Y) = M0 + Y M1 + Y^2 M2; they depend on mh only. | Pass |
| 311 | docstring | The matrices of one objective at a raw vector p: (K, C, A_combined, mh, alpha, beta, gamma_, N0, N1, N2). | Pass |
| 323 | docstring | One sample of the transition; mats are the matrices of this objective, else rebuilt here. | Pass |
| 329 | docstring | The matrices at the current parameters, built once per objective. | Pass |
| 335 to 336 | docstring | The gantry block trained in the ten identifiable combinations, through training coordinates beta (free_params) that are zero at the start. | Pass |
| 343 | comment | # HEURISTIC: m_diff stays within 0.99 of its admissible bound, see _m_diff_bound | Pass |
| 359 to 360 | comment | # m_diff = bound tanh(eta0 + scale beta_m): beta_m = 0 gives combo_init with slope m_diff,0, as the log # coordinates of the other nine have | Pass |
| 371 | docstring | The ten combinations of a raw fourteen-vector, as a tensor (autograd safe). | Pass |
| 382 to 383 | docstring | One raw fourteen-vector with the given combinations, a gauge fixing and not an estimate: equal splits of the sums, mb and the ratio Jb : Jh from raw_init. | Pass |
| 399 | docstring | rho (2 / Lb) sqrt(m_total J_eff), the contracted bound on \|m_diff\|. | Pass |
| 404 | docstring | The ten combinations at training coordinate free (..., 10): c0 exp(beta) for nine, m_diff by tanh. | Pass |
| 412 | docstring | The training coordinate of admissible combinations (..., 10), the inverse of combinations_from_free. | Pass |
| 429 | docstring | The ten combinations at the current training coordinate. | Pass |
| 433 to 434 | docstring | x_tilde^n_{k+1} = f_base^n(beta, x_tilde^n_k, u^n_k) at training coordinate beta = free, without side effects. | S02 |
| 439 | docstring | (matrices, their directional derivative along dfree) at training coordinate free. | Pass |
| 443 | docstring | The matrices at training coordinate free, from the raw vector of gauge_section. | Pass |
| 448 | docstring | Names, start values and current values of the identifiable combinations. | Pass |

### `Clean/model_augmentation/fit_systems/closed_loop.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 to 2 | docstring | Closed-loop model simulation with a known controller, following Kessels: the controller bank, the rollout and the simulator of training and validation. | Pass |
| 12 to 16 | docstring | The distinct controllers of a data set in normalised coordinates, gathered per window.  # THEORY: the controller seen from the normalised model: B^n = B diag(sigma_y), C^n = diag(1/sigma_u) C, # D^n = diag(1/sigma_u) D diag(sigma_y), applied once here. | Pass |
| 19 | docstring | A (K,nc,nc), B (K,nc,ny), C (K,nu,nc), D (K,nu,ny), all physical; ystd (ny,), std_u (nu,). | Pass |
| 29 | comment | # one step is one product: (u_fb^n, xc_next) = [C^n; A] xc + [D^n; B^n] e^n | Pass |
| 38 to 41 | docstring | One controller step for a batch: u_fb^n = C^n xc + D^n e^n, xc_next = A xc + B^n e^n.  # THEORY: the controller is a separate state equation beside the model, not part of the model state. | S02 |
| 50 | docstring | The closed-loop rollout of training, validation and selection; returns (y_hat^n, x final, xc final). | Pass |
| 55 to 57 | comment | # HEURISTIC: xc = 0 at every window start. The controller here filters the model error y^n - y_hat^n, which # does not exist before the window, and xc = 0 is the value for which its correction vanishes when the # model is exact; the integral memory of the recorded run is lost. | F15 |
| 62 to 63 | comment | # THEORY: the controller sees the model output; in this residual form e^n = y^n - y_hat^n and the model # input is u_hat^n = u^n + u_fb^n | Pass |
| 72 | docstring | The closed loop of training and validation; assigned to fit_sys.simulator. | Pass |
| 76 | comment | # loss() receives these per-window arrays by name | Pass |
| 78 | comment | # sigma_e per validation record and axis (n_val, ny), the denominator of sim-NRMSe | Pass |
| 80 | comment | # (hfn, output_only) compiled by torch.compile, or None for eager | Pass |
| 83 | comment | # W_r per training record, one (ny,) row each, appended to every window as loss_w | Pass |
| 88 | docstring | The state without the compiled pair, a cache that cannot be pickled. | Pass |
| 95 | docstring | Run through the uncompiled pair inside the block, then restore. | Pass |
| 103 | docstring | (y_hat^n, x final) of the closed-loop rollout of a batch of windows from x. | Pass |
| 104 | comment | # compiled only for the training loss on CUDA (gradients on); everything else runs eagerly | F06 |
| 116 | comment | # copies out of the CUDA graph's buffers, which the next step overwrites | Pass |
| 121 to 122 | docstring | deepSI's four window arrays plus the controller row and loss weight of every window, checked against the data. | S02 |
| 135 | comment | # same record order and counts as ctrl_ix, so the checks above cover it | Pass |
| 141 | docstring | Each record's first and last window against its raw data at the derived offset. | Pass |
| 157 | docstring | sim-NRMSe: the mean over records and axes of RMS / sigma_e of the closed-loop free run, in metres. | F03 |
| 160 to 162 | comment | # (n_val, ny), metres # THEORY: deepSI's multi-record sim-RMS, the mean of the per-record RMS weighted by the full record # length (System_data_list.RMS) | Pass |
| 169 to 170 | docstring | Closed-loop free-run RMS of N records in one rollout: (per record and axis, per record), and the physical predictions with return_pred. | F08, S02 |
| 195 | comment | # filled with return_pred only | Pass |
| 200 | comment | # x_hat^n at k0 from the samples before it; the scored samples are k0 .. N-1 | Pass |

### `Clean/model_augmentation/fit_systems/interconnect.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 | docstring | Jan's interconnect of blocks and his encoder fit system, with the augmented model of the thesis. | Pass |
| 2 to 3 | comment | # CHANGED: restricted to the thesis training path; full implementation in # model_augmentation/fit_systems/interconnect.py at the repository root | Pass |
| 9 to 10 | comment | # CHANGED: os, time, math, Dataset and the deepSI imports below serve the added fit loop, the epoch index of the # OBC refresh and the checkpoint saving | Pass |
| 19 to 20 | comment | # CHANGED: explicit names instead of the star imports: the @added marker, the blocks the added code looks for, # and the normaliser of the training windows | Pass |
| 26 | docstring | Blocks joined by signal connection matrices into one state-space step (x, u) -&gt; (y, x_next). | Pass |
| 43 | comment | # CHANGED: the OBC correction f_obc^n, set by attach_obc on the OBC arm, else None | Pass |
| 48 | docstring | nn.Module._apply that also moves array_connection_matrices, a plain list, with the module. | Pass |
| 56 to 57 | comment | # CHANGED: obc_pass and pass_mats, built once per objective, are arguments, so torch.compile sees them # as graph inputs | Pass |
| 68 | comment | # CHANGED: the signal loop is _propagate, shared with output_only, on the device and dtype of x | Pass |
| 74 to 75 | comment | # CHANGED: subtract f_obc^n, the part of the network along the baseline sensitivity and offset # directions, so the network cannot take over what the baseline parameters explain | Pass |
| 86 to 87 | docstring | The output signals after computing those in order (forward: all of them, output_only: the output's dependency cone); each block input is written before any block reads it. | Pass |
| 99 | comment | # the physics block takes the matrices built once per objective | Pass |
| 130 | comment | # order of the block computations; x and u are available from the start | Pass |
| 138 | comment | # CHANGED: Python ints, so torch.compile branches on them without a graph break | Pass |
| 146 | comment | # CHANGED: the output's dependency cone, resolved once for output_only | Pass |
| 148 | comment | # CHANGED: the block that takes the per-objective matrices pass_mats, resolved once | Pass |
| 152 | comment | # CHANGED: the connection matrices on the module's device and dtype | Pass |
| 159 | docstring | Output-signal indices that must be computed to obtain output signal target_output_ix. | Pass |
| 173 | docstring | y from the output signal's dependency cone alone, without advancing the state. | Pass |
| 228 | docstring | Add a block; its output and input get the next signal index. | Pass |
| 237 to 238 | docstring | Connect an input signal to an output signal, by concatenation into a block or by addition into x_next and y. | S02 |
| 250 | docstring | Connect every signal of input_signal_list into block, and block into every signal of output_signal_list. | Pass |
| 274 | docstring | Object to hold information regarding connection between two signals in the interconnect | Pass |
| 285 to 286 | docstring | os.replace, retried while another process (a sync client, a virus scanner) briefly holds dst open, which Windows reports as PermissionError. | Pass |
| 298 | docstring | My_Simple_DataLoader that keeps the training windows at the model's dtype (the original casts to float32). | Pass |
| 306 | comment | # the batch order of My_Simple_DataLoader: a generator fixed at seed 0 | Pass |
| 312 | docstring | Jan's fit system: an encoder and an interconnect, trained on deepSI's simulation windows. | Pass |
| 313 | comment | # CHANGED: the rollout of the loss and the validation, set by build_closed_loop (the closed loop) | Pass |
| 316 | comment | # CHANGED: callables run after every validation (the validation history) | Pass |
| 319 | comment | # CHANGED: a callable run after every Adam update (the PS2 opening), or None | Pass |
| 327 | comment | # CHANGED: a normaliser whose windows keep the data's dtype; build_model sets its values | Pass |
| 331 | docstring | Set nu and ny, build the optimizer over every parameter and reset bestfit. | Pass |
| 339 to 340 | docstring | Roll the model forward from x through the attached simulator; returns (y_pred, x_final), y_pred (batch, nf, ny). | S02 |
| 345 | docstring | deepSI's training windows plus the arrays the attached simulator appends per window. | Pass |
| 351 | docstring | The simulator's validation score, then the validation probes (side effects only). | Pass |
| 360 to 361 | docstring | deepSI's batch loop: Adam updates with a validation every its_per_val updates, the best model dumped as _best and the last as _last, then the best reloaded. | S02 |
| 375 | comment | # the dump holds the tensors on the training device; resolve_checkpoint maps them to the CPU | Pass |
| 377 to 378 | comment | # _last.pth at every validation, so a stopped run can resume from its last weights and Adam moments # (not from its batch stream or an unfinished PS2 opening) | Pass |
| 383 | comment | # training windows, data loader and the initial validation | Pass |
| 402 | comment | # DtypeDataLoader, since My_Simple_DataLoader casts the windows to float32 | Pass |
| 410 | comment | # Adam | S05 |
| 418 | comment | # the simulator's extra arrays reach loss() by name | Pass |
| 432 | comment | # the PS2 opening after every update, and how far it got at these weights, for the dumps | Pass |
| 455 to 456 | comment | # last dump, then the best reloaded; on the OBC arm the basis is rebuilt at the reloaded parameters and # the selected dump rewritten with it | Pass |
| 475 to 476 | docstring | The augmented model of the thesis: the W_r-weighted closed-loop objective, the OBC coefficient per objective, and dumps without the training-time state. | Pass |
| 480 | comment | # training-time state: left out of every dump, kept across a dump load | Pass |
| 487 | comment | # the network block and the block that takes pass_mats | Pass |
| 493 | docstring | The integer epoch the OBC refresh counts in. | Pass |
| 494 | comment | # + 1e-6: the epoch counter sums 1 / N per update and can end just below an integer | Pass |
| 498 to 499 | docstring | Pair theta_frozen with the current network on the basis; refresh_basis, or a missing basis, rebuilds the basis first. Returns False without OBC. | S02 |
| 512 | docstring | The objective of a batch of windows: the mean of (W_r (y^n - y_hat^n))^2 over windows, steps and axes. | Pass |
| 514 | comment | # the matrices of this objective: pass_mats, and on the OBC arm the tangent pair of theta_aux | Pass |
| 521 to 522 | comment | # theta_aux of this objective, differentiable in the network; the tangent pair after set_frozen, # which updates vbar | Pass |
| 526 | comment | # loss_w (windows, ny): W_r of each window's record, used by the objective, not by the rollout | Pass |
| 529 | comment | # THEORY: w = sigma_y / sigma_e turns the normalised error into the error over sigma_e, per record and axis | Pass |
| 533 | docstring | deepSI's dump of the model, written atomically. | Pass |
| 534 to 535 | comment | # deepSI's save into &lt;file&gt;.tmp, renamed into place once complete, so a process killed while saving keeps # the old dump; the dump is __getstate__, without the training-time state | Pass |
| 541 | docstring | deepSI's dump load, keeping the training-time state of this process. | Pass |
| 550 | docstring | The state without the training-time keys, as every dump holds it. | Pass |
| 553 | comment | # deepSI stores the optimizer as _optimizer in the dumps; the property keeps that attribute | Pass |

### `Clean/model_augmentation/fit_systems/lbfgs_polish.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 | docstring | The L-BFGS polish after Adam: one fixed batch of windows, kept only if the validation score improves. | Pass |
| 14 | docstring | The settings of the polish. | Pass |
| 25 | docstring | Clone every parameter for an exact rollback. | Pass |
| 36 to 37 | docstring | Pair theta_frozen of the OBC arm with the parameters as they are now, keeping the basis; every loss() overwrites it, so after a restore it can describe another network. A no-op without OBC. | Pass |
| 42 | docstring | How many windows make_training_data would return, without building them. | Pass |
| 48 to 49 | docstring | The one deterministic batch the polish optimises: windows at a stride scaled to about n_windows, then a seeded subset if there are still more. | Pass |
| 87 | comment | # loss(*args, **kw): the four window arrays, then the loss kwargs and the simulator's extras | Pass |
| 99 | docstring | The L-BFGS closure over the fixed batch; counter[0] counts its calls. | Pass |
| 105 | comment | # a detached value, so the graph is freed before L-BFGS calls again | Pass |
| 111 | docstring | The objective at the current weights: one forward pass over the batch, without gradient. | Pass |
| 117 | docstring | (iterations, closure evaluations) as torch counts them, cumulative over step() calls. | Pass |
| 123 | docstring | True for a CUDA out-of-memory error. | F11 |
| 129 | docstring | Evaluate the closure twice without a step; returns (identical, first value, error or None). | Pass |
| 136 | comment | # without its traceback, so the frames and tensors of the failed rollout are freed | Pass |
| 146 | docstring | Polish fit_sys in place with L-BFGS on one fixed batch; returns the outcome dict. | Pass |
| 156 | comment | # every parameter deepSI hands its optimizer, in its order (encoder, then hfn) | Pass |
| 173 to 174 | docstring | Abandon the polish: restore and resync the start weights, and release the CUDA cache after an out-of-memory error; the pre-polish result stands. | Pass |
| 184 to 185 | comment | # one restore-and-resync boundary from the start validation to the final scoring: an exception the inner # handlers do not catch aborts the polish with the weights restored | Pass |
| 187 | comment | # start validation and the fixed batch | S05 |
| 202 to 203 | comment | # determinism probe, compiled first if available: the strong-Wolfe line search needs a loss that # reproduces | Pass |
| 219 | comment | # the failed attempt's optimizer and allocations go before the eager retry | Pass |
| 245 | comment | # L-BFGS in blocks of inner_iter iterations, until it stops inside a block or max_iter is reached | F12 |
| 278 | comment | # final validation | S05 |
| 286 | comment | # handled here, once the except block has released the failed frames and their tensors | Pass |
| 292 | comment | # keep the polished weights only with a finite loss and a validation score that improved | Pass |

### `Clean/model_augmentation/fit_systems/obc.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 to 2 | docstring | Orthogonal projection of the network output onto the baseline sensitivity and offset directions (Gyorok et al.), the model-independent part: the basis factorisation and the coefficient theta_aux; the gantry code supplies J. | Pass |
| 12 | docstring | The projection basis lost column rank at the first build, or at too many refreshes in a row. | Pass |
| 16 | docstring | Economy SVD of a column-scaled projection basis J, with its numerical rank. | Pass |
| 19 to 20 | docstring | J (n_rows, n_cols) in float64; vbar, the detached float64 point beta_lin it is taken at. A rank below n_cols is recorded and the factors truncated, so the caller can reject the basis. | Pass |
| 27 to 28 | comment | # THEORY: numerical rank of the column-scaled J, tol = S_max max(M, N) eps (Golub and Van Loan; the # numpy.linalg.matrix_rank default) | Pass |
| 39 to 40 | docstring | theta_aux = D^-1 V S^-1 U^T F with J D^-1 = U S V^T, D = diag(column_scale): for a full-rank J this is (J^T J)^-1 J^T F. In float64, differentiable in F. | Pass |
| 59 | docstring | Phi: d step(beta, z_i) / d beta at beta_lin = vbar on the rows row_ix, stacked sample-major, float64. | Pass |
| 77 | docstring | The transition step(vbar, z_i) on the rows row_ix, stacked sample-major, float64 without gradient. | Pass |
| 90 | docstring | The projection basis with its rank policy and refresh, and the coefficient theta_aux of the network. | Pass |
| 95 to 96 | docstring | reference_field: network -&gt; F, its stacked output on the reference tuples; basis_builder: vbar -&gt; (J, seconds). | S02 |
| 105 to 106 | docstring | Rebuild the basis at a detached copy of vbar; a basis without full column rank keeps the previous one, and raises at the first build or after max_consecutive_rank_loss refreshes in a row. | S02 |
| 131 | docstring | Refresh at the first objective and at the first objective of every new epoch. | Pass |
| 136 to 137 | docstring | theta_aux for this objective: the network on all reference tuples, solved on the fixed basis and differentiable through F. | S02 |

### `Clean/model_augmentation/fit_systems/pre_encoder.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 | docstring | The encoder of the augmented model. | Pass |
| 2 to 3 | comment | # CHANGED: restricted to the thesis training path; full implementation in # model_augmentation/fit_systems/pre_encoder.py at the repository root | Pass |
| 8 | comment | # CHANGED: the @added marker and the layer stack of the added encoder | Pass |
| 15 | docstring | SUBNET encoder (Beintema et al.) of the physical and added states: psi(chi) = [W^b; W^a] chi + psi_tilde(chi). | Pass |
| 39 | comment | # the windows hold n + 1 samples | Pass |
| 41 to 42 | comment | # THEORY: W^b ~ U(-1/sqrt(m), 1/sqrt(m)), m = (n+1)(nu+ny), the nn.Linear default; drawn although # prefit_physical_encoder overwrites it, so the later draws stay the same | Pass |
| 49 to 51 | comment | # THEORY: W^a ~ U(-a, a), a = sqrt(6 / (n_a + (n+1)(nu+ny))), Xavier uniform as Hoekstra et al. prescribe # for psi_a # HEURISTIC: gain 1, and one draw for [W^a_y W^a_u] together; the paper does not fix these details | S04 |
| 59 to 60 | comment | # the added-state output rows of psi_tilde start at zero; its physical rows keep their draw until # prefit_physical_encoder sets them | Pass |
| 68 | comment | # [W^b; W^a] chi, chi = (y history, u history) | Pass |
| 76 | comment | # psi_tilde reads (u history, y history), a fixed permutation of chi | Pass |

### `Clean/model_augmentation/fit_systems/ps2_opening.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 | docstring | PS2, the predictive-state opening: a data-driven initialisation of the added states during the first Adam updates. | Pass |
| 15 | docstring | The settings of the opening; none of them is a property of a particular plant. | Pass |
| 16 to 21 | comment | # HEURISTIC: the opening's own batch, not the training batch # THEORY: Adam's default rate (Kingma and Ba) for both supervised stages # HEURISTIC: calibration check cadence, in optimizer updates # HEURISTIC: no S1 switch before this many updates # HEURISTIC: S1 ends here at the latest # HEURISTIC: plateau = the best of the last s1_window checks improved by less than s1_tol (relative) | Pass |
| 23 | comment | # HEURISTIC: screen: calibration unexplained fraction &lt;= screen_max by this update | W03 |
| 29 | comment | # HEURISTIC: S2 plateau = the best of the last 5 checks improved by less than s2_tol (relative) | Pass |
| 37 to 42 | docstring | Record-level split: the last record of every record class with at least 3 records is held out.  A class is the record name without its trailing number, 'TR-S5' -&gt; 'TR-S'. # HEURISTIC: the calibration set covers every such class, since the predictive state of one excitation class # need not transfer to another. | S02 |
| 49 to 50 | docstring | The opening after every Adam update until it ends: S1 fits a decoder of the residual on the added states, S2 the network's added-state transition. | Pass |
| 53 to 54 | docstring | The normalised (u, y) of the training records in the simulator's order, with their controller rows and names; n_xb, n_a: physical and added states; width: the decoder's hidden width, that of the network. | S02 |
| 76 to 77 | docstring | One step after an Adam update: an S1 update, and every eval_every updates a calibration check that can end S1 and run S2. | S02 |
| 107 | docstring | The opening's record, written as ps2_summary. | Pass |
| 118 to 119 | docstring | The decoder (output layer at zero), its optimizer over the decoder and theta^a_enc, and the calibration windows. | Pass |
| 127 | comment | # theta^a_enc: W^a and the output layer of psi_tilde, of which only the added-state rows get a gradient | Pass |
| 131 | comment | # a fixed seed of its own for the calibration windows | S05 |
| 137 to 138 | docstring | B windows of the given records, drawn in proportion to their length: encoder history, M future samples, controller row and, with pair, the history one sample later; deepSI's window convention. | W04 |
| 158 | docstring | e_tau / sigma_e: the current model's next M closed-loop output residuals over their RMS on the batch. | Pass |
| 166 | docstring | x_bar of a model state: the n_a rows after the physical ones. | Pass |
| 170 | docstring | One S1 update: the decoder on x_bar predicts the residual; trains the decoder and theta^a_enc. | Pass |
| 176 to 177 | comment | # the gradients go to the opening's optimizer only; the .grad of the shared parameters is saved and # restored, so the next Adam step of the training is unaffected | F04 |
| 186 | docstring | nu: the fraction of the residual the decoder leaves unexplained on the calibration windows. | Pass |
| 193 to 194 | docstring | S2: one-step regression of the network's added-state transition on frozen encoder pairs, kept at its best calibration cost. | Pass |
| 198 | comment | # a fixed seed of its own for the snapshot windows | S05 |
| 208 to 209 | comment | # the one-step error of x_bar over the target variance; T.var(0) divides by B - 1, which scales the # cost by (B - 1) / B and keeps its minimiser | Pass |
| 216 | comment | # the start is a candidate too, so S2 never ends worse than it began on the calibration pairs | Pass |

### `Clean/model_augmentation/systems/__init__.py`

Empty file; no comments or docstrings.

### `Clean/model_augmentation/systems/gantry_ss.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 to 2 | docstring | Physical constants and LFR matrices of the dual-gantry system in float32, from kamtin-fp-model/03 Simulink gantry/main.m (the immutable ground truth). | Pass |
| 32 | comment | # M(Y) = M0 + Y M1 + Y^2 M2 | Pass |
| 40 | comment | # damping C and stiffness K of the logical coordinates q | Pass |
| 51 | comment | # stage-to-logical transform: q = P^-T y, y = P^T q | Pass |
| 59 | comment | # output map y = Cd x = P^T q | Pass |
| 66 | docstring | alpha, beta, gamma and N0, N1, N2 of the rational inverse M(Y)^-1 = (N0 + Y N1 + Y^2 N2) / d(Y). | Pass |
| 92 | docstring | Ax, Bw, Bu and A_combined = [Ax Bw Bu] of the LFR, with M0^-1 = N0 / d0. | Pass |

### `Clean/model_augmentation/utils/__init__.py`

Empty file; no comments or docstrings.

### `Clean/model_augmentation/utils/deepSI_corrections.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 | docstring | deepSI data classes whose training windows keep the data's dtype. | Pass |
| 8 | docstring | System_data whose training windows keep the data's dtype: deepSI's stride-1 path casts them to float32. | Pass |
| 14 | docstring | (uhist, yhist, ufuture, yfuture) of every window, built by deepSI's loop for every stride. | Pass |
| 34 | docstring | deepSI's normalisation u^n = (u - u0) / ustd, y^n = (y - y0) / ystd, returning WindowedSystemData. | Pass |

### `Clean/model_augmentation/utils/torch_nets.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 | docstring | The layer stack of the networks and the augmentation network phi_aug. | Pass |
| 2 to 3 | comment | # CHANGED: restricted to the thesis training path; full implementation in # model_augmentation/utils/torch_nets.py at the repository root | Pass |
| 7 | comment | # CHANGED: the marker of the added definitions | Pass |
| 12 | docstring | [Linear, activation] n_hidden_layers times, then a Linear with zero bias; returns (Sequential, final layer). | F13 |
| 13 | comment | # the layers draw their default initialisation in this order | S05 |
| 30 to 31 | docstring | phi_aug = (f_aug^n, g_aug^n): an MLP whose physical output rows start at zero, so f_aug^n = 0 at the start, while the added-state rows keep their random draw. | Pass |
| 36 | comment | # the routed outputs are the states in order, so the physical rows are the first nx_phys | Pass |

### `Clean/model_augmentation/utils/utils.py`

| Lines | Kind | Quoted text | Verdict |
| - | - | - | - |
| 1 | docstring | Small helpers: the @added marker, tensor conversion and the selection and expansion matrices. | Pass |
| 2 to 3 | comment | # CHANGED: restricted to the thesis training path; full implementation in # model_augmentation/utils/utils.py at the repository root | Pass |
| 9 | docstring | Mark a class or function as added by this project, not part of Jan's framework (grep "@added"). | Pass |
| 15 | docstring | input as a tensor; a tensor is returned as it is. | Pass |
| 20 | docstring | E (n, len(ix)) with E[ix[j], j] = 1: puts a len(ix) signal on the rows ix of an n signal. | Pass |
| 26 | docstring | S (len(ix), n) with S[j, ix[j]] = 1: selects the rows ix of an n signal. | Pass |

Coverage total: 201 docstrings and 175 comment blocks across 24 files, including the four empty initializers. Error and print findings are in sections 1 to 3.