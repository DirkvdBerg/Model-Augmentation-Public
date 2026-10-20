# Codex judgment of Clean

2026-10-20. Executed `tasks/handoffs/2026-10-20-clean-code-judge-codex.md` against the working tree, including the modified implementation report and current D-274/D-275 amendments.

Verdict: the numerical implementation is a faithful copy-and-trim of the amended design on the specified paths. It is substantially smaller, but not yet minimal under rule G. I would address the run-folder collision and continuation-budget issues before using the CLI for unattended continuations. Polish-only also retains an exception to the checkpoint comparison rule. These are not failures of the established CPU equivalence results: several are faithfully inherited problems that an equivalence test cannot distinguish from correct behavior.

I read all 24 files, the implementation report, design, decisions, comparison, inventory Part B, thesis Sections II to V and relevant appendix passages. I compared retained definitions with mainline by AST, ignoring documentation strings, and checked all 24 files with in-memory Python compilation. Syntax passed. No pipeline imports, training, harness, evaluation or simulation was run. No package was installed, no git write was performed, and no protected data was read. This verdict is the only file created. GPU execution and server checkpoint compatibility remain unverified by execution.

References below use current working-tree line numbers. `GD` means `scripts/gantry/gantry_dynamic/`, `MA` means `model_augmentation/`, `V` means `Thesis-writeup/Code/Verification/clean-code/`, and `S2` to `S5` mean the four requested thesis files in `Thesis-writeup/Writing/sections/`. A design-conforming finding is explicitly identified; its proposed fix is a review recommendation, not an edit or a reversal of an agreed decision.

## Findings ranked by consequence

### F1. A fresh launch can overwrite an existing run without archiving it

`Clean/gantry_interconnect_dynamic.py:71` and `:94` | mainline `scripts/gantry/gantry_interconnect_dynamic.py:384`, `:387`, `:404`; design:267 and :893; D-275 D6 at `docs/decisions.md:8375` | **high**.

The fresh path accepts an existing folder through `exist_ok=True`, then replaces its config and eventually its checkpoints/history. The archive operation is only in the explicit rerun path. A reused SLURM job id, for example a requeue launched with the original command, or an explicitly reused `run_id`, therefore bypasses D6's preservation guarantee. Timestamp collisions are another possible trigger. This is inherited from mainline and copied as designed, but it conflicts with the stronger clean run-folder lifecycle.

Fix: require a new directory for a fresh run and reject a collision before writing anything, directing the caller to the explicit resume or polish-only operation. Keep the agreed rerun archive behavior.

### F2. A continuation with no epochs remaining runs a full Adam budget

`Clean/gantry_dynamic/training.py:272`, `:308` | mainline `GD/training.py:862`, `GD/model.py:487`; design:390 and :434 | **high**.

When a dump reports `done_epochs == hp['epochs']`, `epochs_remaining` is zero, but `epochs_remaining or hp['epochs']` substitutes the entire budget. A finished 200-epoch run continued from its final last dump runs another 200 epochs. A best dump rounded to the final epoch can do the same. Negative remaining epochs are also not handled explicitly. The layer-3c budget deliberately avoids the zero case, as design:1139 explains, so its PASS does not cover this behavior. This is an acknowledged mainline expression, faithfully merged, not an implementation deviation.

Fix: distinguish a missing epoch argument from zero; explicitly skip Adam when the remaining budget is nonpositive, then perform the appropriate selection/polish steps. Do not use truthiness to choose an epoch budget.

### F3. Polish-only bypasses the selected-checkpoint acceptance threshold

`Clean/gantry_dynamic/training.py:228`, `:235` | mainline `GD/training.py:790`, `:797`; D-274 amendment at `docs/decisions.md:8349`; design:433 | **medium**.

Even though the pre-polish pair supplies a finite `bestfit`, `start_phase == 'lbfgs'` discards it. With OBC, the comparison becomes the rebuilt model's remeasured score. If the selected score is 0.2926736, the rebuilt score is 0.2926741 and the candidate is 0.2926739, an ordinary run rejects the candidate but a polish-only run accepts it. This is exactly the interval D-274's comparison change is meant to exclude. The expression is explicitly retained by the design, so the implementation follows the design while the design leaves a method-rule exception. I am not proposing to restore the old epoch basis or undo the agreed rebuild rule.

Fix: use the finite selected score from the pre-polish metadata for polish-only too, and remeasure the start separately for history and lazy initialization. For U, eager initialization must still happen before a compiled probe even when the acceptance threshold is supplied.

### F4. Repeated continuations lose cumulative progress; early PS continuations are not phase continuations

`Clean/gantry_dynamic/training.py:56`, `:264`, `:281`; `Clean/model_augmentation/fit_systems/interconnect.py:448`, `:609` | mainline `GD/training.py:107`, `:853`, `:873`; `MA/fit_systems/interconnect.py:938`, `:1289`; design:901 to :913 | **medium**.

Only component state dictionaries and optimizer moments are transferred into a newly built system. The saved epoch/batch histories and counters are not installed. The new fit starts its epoch counter at zero, while the next dump again derives `done_epochs` solely from that local counter. Example: a first continuation starts after 150 epochs, runs 20 more and stops; its last dump describes about 20, so a second continuation calculates about 180 remaining rather than 30. Rounding fractional epochs also loses progress at every interruption.

The PS opening is deliberately excluded from dumps. Before half an epoch, rounding can yield zero and recreate S1 from scratch; later, any positive rounded count skips PS entirely, even if the interruption occurred during S1. Restored Adam moments do not restore the auxiliary optimizer, decoder, calibration state or batch-stream position. These behaviors are inherited and partly documented by the design. They qualify the claim that the periodic last dump continues where a time-limited run stopped; they do not contradict equivalence between the two resume implementations.

Fix: retain cumulative update/epoch offsets across continuations. For interrupted PS openings, either persist the opening state or explicitly reject/document that unsupported continuation point. Do not describe the current behavior as uninterrupted-trajectory continuation.

### F5. OOM rollback covers probes and optimizer steps, but not the whole polish

`Clean/model_augmentation/fit_systems/lbfgs_polish.py:239`, `:333`, `:339` | mainline `MA/fit_systems/lbfgs_polish.py:633`, `:765`, `:773`; design:618 | **medium**.

FixedBatch construction/device transfer is outside the probe and step exception handling. The final loss evaluation and final validation are also outside the rollback handler. An OOM in batch allocation aborts the run before its final pair is written, although the pre-polish pair exists. An exception during final scoring occurs after parameter updates and does not execute the restore/resync path. The outer pipeline finally only moves the model to CPU. This is faithful to mainline, but should not be described as general CUDA OOM recovery.

Fix: put the whole mutating polish phase, including terminal scoring, under one restore/resync exception boundary; classify allocation failure as an aborted polish and preserve the pre-polish result. Release failed optimization state before any retry that needs memory. Keep the copied compiled/eager decision rules.

### F6. Rerun preparation changes the live folder before it establishes a runnable continuation

`Clean/gantry_interconnect_dynamic.py:45`, `:50`, `:62`, `:77` | design:883 to :900; checkpoint checks at mainline `GD/training.py:277` | **medium**.

Preparation checks source existence, then moves all outcomes and, for Adam resumes, both dumps. Only afterwards does the program load data, check CUDA availability, decode the checkpoint and check its architecture/dtype. An ordinary failure such as unavailable CUDA or a truncated dump leaves the live folder without its old outcome. A retry with the same command then cannot find its source in the normal location. The old bytes remain in previous, so this is an availability/recovery problem rather than data deletion. Moving all old outputs is explicitly the design; the failure behavior around it is new orchestration.

Fix: preflight data/environment and checkpoint readability/compatibility before moving artifacts, or restore the archive on initialization failure. Preserve D6's one-folder and archive rules.

### F16. Periodic last-dump replacement is not atomic

`Clean/model_augmentation/fit_systems/interconnect.py:436`, `:616` | design:585 describes the inherited direct `torch.save(self.__dict__, file)` operation; D-275 amendment 2 at `docs/decisions.md:8383` | **medium**.

Every validation replaces the same last-dump file directly through deepSI. If a time limit kills the process during that write, the newest dump can be truncated and the previous valid last dump has already been replaced. This does not change training state or invalidate the equivalence check, but it weakens the specific interruption-recovery purpose of the new periodic save. Best may still be available; it is intentionally not a silent fallback for an explicitly requested last source.

Fix: write a temporary checkpoint beside the destination and atomically replace the destination only after a successful close. Keep the save at exactly the agreed point and preserve the explicit source-selection behavior. This is an inherited I/O limitation exposed more often by the new periodic write, not a missing device or RNG operation.

### F7. A removed THEORY label contradicts the marker-preservation claim

`Clean/gantry_dynamic/data.py:189` | mainline `GD/data.py:350`; design:146 to :148; implementation-report.md:89 to :96 | **low**.

The first-difference calculation remains, but its associated `# THEORY:` label is gone. It occurs inside mainline's longer noise discussion, which was removed wholesale. This is the one missing label found by the definition-level label-presence scan on retained computations. The other scan hit, the THEORY label of the dropped `RunConfig.na_nb` rule, belongs to removed code and is correctly absent. The report's assertion that every label of a kept definition survives is too broad.

Fix: retain the relevant label beside the kept first difference, without restoring the obsolete noise-mode narrative. This is a required-label finding, not a comment-wording review.

### F8. Internal guards and their helper remain despite rule G

`Clean/model_augmentation/fit_systems/interconnect.py:138`; `Clean/model_augmentation/utils/utils.py:23`; `Clean/model_augmentation/fit_systems/blocks.py:500`; `Clean/gantry_dynamic/closed_loop_report.py:34` | mainline `MA/fit_systems/interconnect.py:226`, `MA/fit_systems/blocks.py:1442`, `GD/closed_loop_report.py:77`; design:128 to :135 | **low**.

The algebraic-loop assertion checks the graph constructed by the fixed `build_model`; removing it also removes `detect_algebraic_loop` and its import. The transition input-shape assertion checks the clean reference builder's own tensors. The true-free round-trip assertion checks a value the helper itself just constructs. The design explicitly copies these definitions/statements, so this is a remaining inconsistency in its application of D5, not an unlisted implementation trim.

Fix: remove these internal rechecks and the now-unneeded helper. Keep checks for external checkpoint coordinates, evaluation-provided combinations, data readiness and deepSI window alignment. The forward batch-size assertion has an external evaluation caller and is justified.

### F9. OBC refresh bookkeeping has no remaining consumer

`Clean/model_augmentation/fit_systems/obc.py:141`, `:144`, `:148`, `:158`, `:176` | mainline `MA/fit_systems/obc.py:431`, `:443`, `:448`, `:458`, `:494`; design:630 to :638 | **low**.

The refresh/rank-loss totals, timing dictionary, entry dictionary and entry fields remain after their diagnostic readers/history were deleted. The entry survives only to transfer an elapsed time into a list nobody reads. `build_s` at :150 is unused. This is explicitly retained by design B8, so the implementation is faithful but carries unnecessary bookkeeping.

Fix: remove the unused totals, entry/timing collection and discarded elapsed-time local as a consistent trim. Retain `consecutive_rank_loss`, `max_consecutive_rank_loss`, `basis` and `last_refresh_epoch`: they implement the actual rank policy. Do not remove the column scaling or its Gram-diagonal calculation; those affect the numerical solve.

### F10. Dead simulator result fields survived the diagnostic removal

`Clean/model_augmentation/fit_systems/closed_loop.py:117`, `:134`, `:225` | mainline `MA/fit_systems/closed_loop.py:375`, `:416`, `:702`; design:514 to :525 | **low**.

`WindowControllerIndex.last_counts` is written but its old reader, `ClosedLoopSimulator.last_window_counts`, was removed. `last_per_record` is also assigned but has no retained reader. Counts and per-record errors are still needed as local values; their persistent copies are not.

Fix: remove these three state assignments. Keep `last_per_channel`, `last_sim_rms` and `val_sigma_e`, which the history reads.

### F11. Unused arguments remain at internal call boundaries

`Clean/model_augmentation/fit_systems/lbfgs_polish.py:156`; `Clean/model_augmentation/fit_systems/interconnect.py:343`, `:375`, `:414`; `Clean/model_augmentation/fit_systems/blocks.py:180`, `:223`; `Clean/gantry_dynamic/training.py:253` | mainline `MA/fit_systems/lbfgs_polish.py:464`, `MA/fit_systems/interconnect.py:594`, `:651`, `:809`, `MA/fit_systems/blocks.py:814`, `:903`, `GD/training.py:827`; design:385, :564, :605 | **low**.

`_make_closure.params` lost its only use with the gradient-None fill. `Dtype_DataLoader.seed` is ignored; its actual fixed seed is zero. `terms`/`sterms` no longer capture anything. `train.norm` is not read. The fit/init_model `auto_fit_norm` arguments and fit `optimizer_kwargs` are inert on the trimmed initialized-only path. Some are explicitly retained by the design/report because callers still pass them, which explains equivalence but does not make them necessary.

Fix: remove internal-only inert parameters together with their call-site arguments. Preserve externally required deepSI-compatible signatures where an inherited caller actually uses the interface. Do not change the fixed loader RNG or remove overwritten random draws.

### F12. Removed add-to wiring still has an argument, stored field and parser branch

`Clean/model_augmentation/fit_systems/interconnect.py:278`, `:282`, `:324`, `:330`, `:336` | mainline `MA/fit_systems/interconnect.py:457`, `:461`, `:532`, `:537`, `:543`; design:548 to :555 | **low**.

The add-to implementation was removed from `init_connection_matrices`, but the public plumbing and parser aliases remain. No retained construction uses them. Passing that mode would now silently produce no implemented connection behavior. It is outside the spec, so this does not invalidate any specified model, but the leftover surface is unnecessary and misleading.

Fix: remove the add-to parameter, conversion, field and parser alternative together. Keep concatenation and additive behavior and their summation order.

### F13. Affine-only correction retains tangent-space branches

`Clean/gantry_dynamic/obc_gantry.py:123`, `:138`, `:151` | mainline `GD/obc_gantry.py:219`, `:259`, `:307`; design:404 to :410 | **low**.

`GantryOBCCorrection` is only constructed with affine space, yet its direction selection, buffer-size choice and final correction selection still branch on tangent space. Other tangent pieces were already removed. The separate basis builder's tangent option is an evaluation interface and should remain.

Fix: simplify only the correction module's constant affine branches while keeping its `space` attribute for evaluation. Preserve the basis builder's explicit space argument and both basis modes.

### F14. Two OBC coefficient paths duplicate the same solve

`Clean/model_augmentation/fit_systems/obc.py:192`, `:201` | mainline `MA/fit_systems/obc.py:512`, `:558`; design:639 to :641 | **low**.

After the diagnostics were removed, `prediction_coefficient` repeats the reference-field call, device move, solve and cast of `coefficient`, differing only by no-grad context. There is no longer a distinct prediction-solving algorithm to maintain.

Fix: perform the same coefficient helper call under `torch.no_grad()` for prediction. Keep autograd enabled in the training call and preserve the device transfer and dtype conversion order.

### F15. Constant S2 rollback and fixed-dtype alternatives remain

`Clean/model_augmentation/fit_systems/ps2_opening.py:226`; `Clean/gantry_dynamic/config.py:185`, `:197`, `:201` | mainline `MA/fit_systems/ps2_opening.py:251`, `GD/config.py:543`, `:555`, `:559`; design:302 to :307 and :688 | **low**.

D-274 initializes `state` to a snapshot and never assigns None afterwards, so the `state is not None` branch is dead. `fs_new` is fixed at 4000 and `use_f64` at True, so the copied None/float32 alternatives in the derived properties have no spec caller. These are small design-conforming remnants.

Fix: restore the S2 snapshot unconditionally and return the fixed derived rate/dtypes directly if a further minimality pass permits it. Do not alter S2's variance convention or initialization draws.

## Harness findings that can conceal a failing implementation

These do not overturn the accepted local results; they limit what future PASS messages certify.

### H1. Comparisons do not establish a complete layer-4 campaign

Affected implementation: `Clean/gantry_dynamic/config.py:135`, `Clean/gantry_dynamic/training.py:132` and the evaluation interfaces in the file table below | `V/compare_l4.py:36`, `:59`, `:70`; design:1140 to :1144 | **medium**.

The comparator discovers only files present on the mainline side and iterates whatever checks its summaries contain. It does not require the nine runs, three splits, required layer-4b interfaces, or a nonempty evaluation check. Empty dictionaries satisfy the nested `all` calls. Both sides can therefore compare a subset, or even fabricated empty summaries with no arrays, and print PASS. The normal batch script requests the full run, but the comparator does not prove completion of that request.

Fix: assert the expected run/split/interface sets, both sides' complete file manifests, and all required positive finite validation-reference checks before comparing values. Cite a partial result as partial rather than giving it the full campaign PASS label.

### H2. The fingerprint gate never rejects changed clean source

Affected implementation: all 24 files, for example `Clean/gantry_dynamic/__init__.py:10` and `Clean/gantry_dynamic/training.py:253` | `V/compare.py:35`, `V/hcommon.py:154`; design:1102 to :1114 | **medium**.

The gate recomputes mainline/data hashes but never compares current clean hashes with the hashes of the clean process that produced its output. A later broken clean edit can be compared against stale passing clean output without rejection. The clean source manifest also excludes the import-origin guard and the empty package initializers, although those affect self-containment. Saving hashes is useful provenance, but it is not freshness enforcement unless checked.

Fix: recompute and compare the clean output's source hashes against the current full 24-file tree before accepting a comparison. Compare the current clean tree with the captured clean tree, not with the golden's intentionally different implementation.

## 1. Fidelity to the amended design

The 24-file layout, package origin guard, roots, preset table, default CUDA launch, CPU/smoke overrides, config adapter and checkpoint key contracts match the design. The adapter rejects non-spec fixed settings and accepts the mainline schema. D-274's K1 filter and initial S2 snapshot are present. Ordinary-run OBC polish compares with selected `bestfit` and retains the rebuilt model after rejection. D-275 amendment 3's optimizer fallback is present at `Clean/gantry_dynamic/training.py:59`, so the design's older fresh-moments prose is superseded and is not an implementation defect. Evaluation scripts were not part of the requested change.

The main caveats are F7, a required-label deviation, and design-conforming issues F1 to F6 and F16. F8 to F15 are smaller opportunities beyond the design's explicit copy decisions. The implementation should not be blamed for retaining statements the design specifically tells it to retain; nevertheless those statements count when answering whether the result is minimal.

### Every reported deviation

- 1, deleting now-dead locals: Judgment: Justified; Reason: Every listed removal loses its reader with a specified trim. It does not reorder arithmetic or RNG. The retained inert arguments/headers are faithful but still covered by F11/F15.
- 2, imports following retained names and resolving star imports: Judgment: Justified; Reason: The same retained objects are explicitly imported; removed names have no retained reader. This improves static checking without changing construction order.
- 3, fit signature pruning: Judgment: Justified; Reason: Removed parameters belong to removed features. The remaining inert compatibility arguments can be trimmed with their callers, subject to F11.
- 4, comment/docstring pruning: Judgment: Not fully justified; Reason: Wording is outside this review. Marker placement and THEORY/HEURISTIC presence are in scope: F7 contradicts the blanket label-survival claim. Removing labels attached only to dropped alternatives is justified.
- 5, removing the empty polish print suffix and its placeholder: Judgment: Justified; Reason: The resulting output has the same content; no training state changes.
- 6, using an existing base-environment pyflakes: Judgment: Justified; Reason: Static analysis does not import the pipeline; no dependency or environment mutation was needed. I independently checked syntax, not the reported pyflakes invocation.
- 7(a), redirecting golden save_dir and excluding post-run diagnostics: Judgment: Justified; Reason: These happen outside, or redirect, artifact/report production; the copied training/selection/polish path still runs.
- 7(b), redirecting evaluation cache: Judgment: Justified; Reason: It isolates verification outputs from thesis results without changing the evaluated tensors.
- 7(c), hashing large arrays: Judgment: Justified; Reason: Hashes preserve a bitwise identity check without storing every array. Shape and hash are checked; this is not a numerical tolerance.
- 7(d), mainline smoke environment only for PS settings and explicit budget replacement: Judgment: Justified; Reason: Both sides receive the intended short PS spec and matched budget. This must not be interpreted as production PS plateau coverage.
- 7(e), pinning CPU least-squares to gelsd in the loader process: Judgment: Justified with a scope qualification; Reason: It controls a documented source of nondeterministic last bits on both sides. It verifies the common pinned evaluation path, not literal default-driver output. The evaluation self-check also runs with the pin, so it does not independently prove default-driver equality. The report already discloses this and the small default-driver difference.

### Markers

All five empty `model_augmentation` initializers are intentionally empty, as specified; they are not new method implementations requiring invented markers. Retained added-file markers and added-class/function decorators are present where mainline has them. The 25 planned clean-trim markers in Jan definitions are present, and the periodic last-save statement has its own CHANGED marker. I found no missing required origin or trim marker. F7 is the label-preservation exception. No comment wording is judged here.

## 2. Fidelity to the thesis

The baseline matrices, ten-combination map, bounded signed mass difference, fixed length, RK4 with endogenous Y at each stage, normalized parallel correction and unaugmented output match the method. The encoder has the intended 30 past samples, no current measurement, random added-state linear map and fitted physical affine map. The physical fit uses the simulated closed-loop histories, centered/scaled float64 SVD and the specified readiness checks. Weighted window loss, record/axis population-standard-deviation normalization and complete-record validation match the mathematical definitions.

The OBC reference stack uses measured positions and fourth-order reconstructed velocities, zero added states and successor-valid samples. The float64 scaled SVD, differentiable coefficient solve, detached epoch expansion point, routed subtraction and fixed-basis polish implement the intended projection. The hand tangent propagates through every RK4 stage. PS S1 updates only the decoder and added-state encoder outputs, and S2 freezes encoder pairs and fits the added-state transition. The initial candidate is now eligible for S2 selection.

Two small current prose inconsistencies are independent of the settled rulings:

* **T1, low:** `Clean/model_augmentation/fit_systems/closed_loop.py:237` and `Clean/gantry_dynamic/data.py:140` use population standard deviation, as mainline `MA/fit_systems/closed_loop.py:715` and `GD/data.py:233` do. S3:892 currently describes the denominator as RMS servo error, whereas S5:321 and the displayed metric define standard deviation. Consequence: an implementation following that sentence literally would weight nonzero-mean servo errors differently. Fix the sentence to standard deviation; keep the agreed code and D-256/D-262 convention.
* **T2, low:** `Clean/model_augmentation/fit_systems/ps2_opening.py:151` constructs the shifted history ending just before `k+1`, so it includes measurements through `k`. Mainline `MA/fit_systems/ps2_opening.py:177` does the same. S3:1281 says the shifted window uses data up to `k_i+1`. Consequence: that sentence suggests an extra current sample that the encoder equations and implementation exclude. Fix it to data through `k_i`, preserving the stated encoder convention.

Remaining disagreements and representational qualifications, listed explicitly to close question 2:

1. **F3:** polish-only can compare with the rebuilt score rather than the selected score. This is the remaining operational exception to D-274's thesis-aligned acceptance rule.
2. **T1:** the current identification prose says RMS servo error while the actual denominator and metric definition use population standard deviation.
3. **T2:** the S2 shifted-history prose includes one sample too many; the equations and code use past samples only.
4. **Accepted by D-274, no new change requested:** S2 uses `T.var(0)`, so its scalar cost is `(B-1)/B` times the displayed variance fraction (`Clean/model_augmentation/fit_systems/ps2_opening.py:202`, S3:1264). This is the settled convention, not a new code defect.
5. **Equivalent affine coordinates, no method disagreement:** the implementation stacks `[J, fbar]` at `Clean/gantry_dynamic/obc_gantry.py:177`, while S4:191 and :247 write `[J, fbar-J*beta_lin]`. Their spans and subtracted corrections agree. Saved first-ten coefficients require the conversion `a_gamma = a_code + alpha*beta_lin`; do not compare them literally without conversion. This is the existing inventory qualification.
6. **Existing evaluation-only disagreement, outside Clean training:** the negation evaluator uses the stored expansion point for OBC, while S4:390 writes current beta (`Thesis-writeup/Code/Evaluation/eval_negation.py:71`, `Clean/gantry_dynamic/obc_gantry.py:122`). It remains unchanged under amendment 4; no evaluation edit is proposed in this task.

The encoder net's input column order is `[u,y]`, whereas the displayed history stacks `[y,u]`; this is a parameter-column permutation, not a different function class or use of different samples. Float32 materialization of nominal constants, P and selected buffers before float64 promotion, scale stabilizers, gauge reconstruction, auxiliary Adam defaults, batch shuffling, tie selection, rank-failure policy and polish probes are implementation details already identified in the inventory. They remain unstated or only partly stated in the thesis, but do not establish a different mathematical method. No new finding is raised against the settled Horner, filter, 100-percent, scheduling-variable or K1 decisions. The appendix now explicitly records the noise-free point-sampling qualification.

## 3. Minimality

F8 to F15 identify the concrete remaining unnecessary code. The largest removable clusters are OBC diagnostic bookkeeping, the internal graph-check helper, dead simulator fields and inert internal interface arguments. They are small compared with the reduction already achieved; removing them should be a controlled trim with the existing equivalence evidence retained as the baseline.

Several apparently redundant pieces are necessary here: nominal float32 buffers and Lambda_combo preserve the component checkpoint schema; overwritten initialization draws preserve the RNG stream; the optimizer property preserves dump semantics; the controller registry and basis-builder space argument support evaluation; the eager context and compiled fallback support polish-only; the column-scale calculation affects SVD conditioning. Empty package initializers preserve package resolution. None should be deleted solely because a CPU execution count is zero.

## 4. Robustness outside local harness coverage

- CUDA moves and removed self-heals: Reading verdict: Faithful. `Interconnect._apply` at :43 moves its non-buffer connection matrices, while registered ANN/output/correction buffers follow module moves. Lazy initialization places matrices on the current module device/dtype. The controller bank rehomes at rollout entry. I found no new missing device move.
- Compiled training: Reading verdict: Faithful. The pair is built in `Clean/gantry_dynamic/controller.py:196`; the simulator selects it only for CUDA with gradients, marks a cudagraph step and clones returned tensors at `closed_loop.py:198` to :209. Per-objective physical/OBC tensors remain explicit arguments. A production CUDA compile success cannot be certified by reading.
- Post-fit versus polish-only compilation: Reading verdict: Faithful. Simulator serialization drops the compiled pair at `closed_loop.py:179`; fit reloads that eager simulator. A fresh polish-only build retains the compiled pair. Both paths remain reachable as intended.
- Compiled polish try, eager fallback, second probe: Reading verdict: Faithful to mainline `lbfgs_polish.py:657` to :698. The eager context restores the pair in finally. Existing validation initializes lazy wiring before probes on the normal polish-only path. F3's fix must preserve that initialization.
- Rejected polish and step/probe exceptions: Reading verdict: Snapshot/restore and OBC resync are copied. No basis rebuild occurs in rollback. See F5 for exceptions outside those boundaries.
- CUDA OOM: Reading verdict: The copied step handler restores parameters and resyncs; probe failure can reach eager fallback and abort. General allocation/terminal-scoring recovery is incomplete, as in mainline, F5. No claim that the server OOMs have been resolved.
- OBC rank loss: Reading verdict: Faithful. `obc.py:160` retains the previous valid basis after one loss, aborts if no valid basis exists, aborts on the second consecutive loss and resets the consecutive counter after success. Device transfer and refresh-epoch updates survive. A rejected candidate does not replace vbar used for correction.
- NaN training loss: Reading verdict: The mainline break remains at `interconnect.py:502`; best reload and resync remain. This checks NaN, not every nonfinite value, exactly as mainline does.
- PS plateau, screening and fallback: Reading verdict: Copied, including retaining S1 encoder changes on fallback, discarding the decoder, and initial-candidate S2 restore. Production thresholds are not established by the smoke-budget cases.
- Own-code deepSI resume: Reading verdict: Optimizer fallback and moment merge are copied, and schema/dtype/coordinate checks remain. It is a weights-and-moments restart with the progress/PS limitations of F2/F4, not full process-state restoration. Mainline dumps are outside the agreed clean resume contract; cross-code component pairs remain supported.
- Periodic last dump: Reading verdict: Correct placement at `interconnect.py:435`, immediately after best-checkpoint handling and before train mode. The save wrapper temporarily stashes lifecycle/reference/phase state and restores it in finally. It does not sync basis, draw RNG or advance Adam. Writing the entire dump directly is inherited and is not an atomic-file transaction; interruption during a save can leave an unreadable newest file.
- CLI and roots: Reading verdict: Default R1 lowpass CUDA, CPU/smoke flags, mutually exclusive rerun kinds and source restrictions match the design. Data/output roots are supplied directly and are not restored from old machine paths. Output root affects fresh family folders; explicit reruns deliberately keep the supplied run folder. F1/F6 are the lifecycle risks.
- Server mainline polish-only: Reading verdict: The component pair and config adapter support it, as intended. The copied full config's fixed values still need the pending server validation. F3 is relevant to this operation; successful pair loading alone does not establish the right acceptance reference.

## Verdict for every file

Here F = design fidelity, M = minimality and R = robustness by reading. A pass on R is scoped to reading, not a claim that pending server tests ran. Low-priority further trims do not change a file's faithful-copy verdict.

- `Clean/gantry_interconnect_dynamic.py`: F: Pass; M: Appropriate orchestration; R and findings: Needs lifecycle fixes F1/F6. Rerun/archive implementation otherwise matches D6.
- `Clean/gantry_dynamic/__init__.py`: F: Pass; M: Minimal; R and findings: Origin guard correctly detects mixed clean/mainline packages.
- `Clean/gantry_dynamic/config.py`: F: Pass; M: Small fixed-setting remnants F15; R and findings: Adapter and roots match; no new server-only failure found by reading.
- `Clean/gantry_dynamic/data.py`: F: Label exception F7; executable trim passes; M: Appropriate retained loader/evaluation interface; R and findings: K1 filter, normalization and data guards faithful. T1 concerns thesis prose.
- `Clean/gantry_dynamic/controller.py`: F: Pass; M: Appropriate evaluation generality; R and findings: Compiled pair, registry and controller bank faithful. GPU execution pending.
- `Clean/gantry_dynamic/model.py`: F: Pass; M: Construction and LS checks justified; R and findings: CPU construction, CUDA guard, prefit device finally and gelsd fit faithful.
- `Clean/gantry_dynamic/obc_gantry.py`: F: Pass; M: F13; R and findings: Reference alignment, caching, device changes and tangent subtraction faithful. Affine coordinate qualification listed in question 2.
- `Clean/gantry_dynamic/training.py`: F: Pass, including amendment 3; M: F11; R and findings: F2/F3/F4/F5. Architecture/dtype checks and optimizer moment restore survive.
- `Clean/gantry_dynamic/closed_loop_report.py`: F: Pass; M: Internal recheck F8; R and findings: Both evaluation helpers retained; deepcopy and interventions faithful.
- `Clean/model_augmentation/__init__.py`: F: Pass; M: Minimal, empty; R and findings: No eager benchmark imports.
- `Clean/model_augmentation/fit_systems/__init__.py`: F: Pass; M: Minimal, empty; R and findings: Package path preserved.
- `Clean/model_augmentation/fit_systems/interconnect.py`: F: Pass; M: F8/F11/F12; R and findings: Move/compile/checkpoint seams faithful; F4 limits continuation state. Periodic save correctly marked and placed.
- `Clean/model_augmentation/fit_systems/blocks.py`: F: Pass; M: F8/F11; R and findings: RK4, analytic derivative/tangent and admissible-coordinate formulas faithful; checkpoint buffers retained.
- `Clean/model_augmentation/fit_systems/closed_loop.py`: F: Pass; M: F10; R and findings: Compiled/eager selection and free-run grouping faithful. External window-alignment checks justified.
- `Clean/model_augmentation/fit_systems/lbfgs_polish.py`: F: Pass; M: Unused closure argument F11; R and findings: F5; copied compiled/fallback/rejection behavior otherwise faithful.
- `Clean/model_augmentation/fit_systems/obc.py`: F: Pass; M: F9/F14; R and findings: Rank policy and differentiable scaled-SVD solve faithful.
- `Clean/model_augmentation/fit_systems/pre_encoder.py`: F: Pass; M: Appropriate; R and findings: Initialization draw order, zeroing, history reshape and checkpoint names faithful.
- `Clean/model_augmentation/fit_systems/ps2_opening.py`: F: Pass; M: Constant guard F15; R and findings: Production branches copied; D-274 candidate restored. T2 is a prose disagreement.
- `Clean/model_augmentation/systems/__init__.py`: F: Pass; M: Minimal, empty; R and findings: Package path preserved.
- `Clean/model_augmentation/systems/gantry_ss.py`: F: Pass; M: Required constants/builders only; R and findings: Float32 nominal construction is intentional; no runtime environment overrides remain.
- `Clean/model_augmentation/utils/__init__.py`: F: Pass; M: Minimal, empty; R and findings: Package path preserved.
- `Clean/model_augmentation/utils/utils.py`: F: Pass; M: Graph-check helper F8; R and findings: Tensor-conversion and connection matrices retain original numeric behavior.
- `Clean/model_augmentation/utils/torch_nets.py`: F: Pass; M: Appropriate; R and findings: Live added-state and zero physical-output initialization, including draws, faithful.
- `Clean/model_augmentation/utils/deepSI_corrections.py`: F: Pass; M: Appropriate compatibility subset; R and findings: Exact history/future slicing and normalization retained; no copied alternative encoder path.

No source fix is made by this review. The pending full server layers and GPU smokes remain pending; H1/H2 should be considered when interpreting their eventual PASS reports.
