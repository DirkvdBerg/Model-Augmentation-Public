# Simplification proposals for Clean

2026-10-21. Proposal-only review of the current working tree, following
`tasks/handoffs/2026-10-21-clean-code-optimise-codex.md`. No source changes or numerical runs.

**Estimated saving: 93 nonblank code lines across P01 to P29**, with every existing class and file
name retained. These are conservative implementation estimates, not a measured patch. Comments,
docstrings and blank-line tightening earn no saving. P19 is conditional on preserving initialization
draws exactly; excluding it gives 91 lines. Clarity-only proposals below add no counted saving.
The separate choice-3-breaking alternative is excluded, as are any savings from deleting its
compatibility classes. The estimates include helper/decorator/import overhead and do not count the
same deletion twice.

The five largest gains:

| Rank | Proposal | Estimated code lines saved |
|-|-|-|
| 1 | P01: replace the physics matrix cache bridge with direct argument passing | 14 |
| 2 | P02: share signal allocation and ordered graph evaluation | 8 |
| 3 | P03: remove unused OBC report fields and their calculations | 8 |
| 4 | P04: express constant configuration properties as class attributes | 6 |
| 5 | P05: remove the empty block-initialization call chain | 6 |

## Scope, constraints and evidence

Read the implementation report, design sections 0, 1, 3, 5 and 6.1, the Codex judge including
Minimality, and D-275, D-278 and D-279. Reviewed Part A locally and Parts B/C through two read-only
review agents, as the handoff permits. Read relevant evaluation/harness consumers to check interfaces.
Did not read the independent Claude optimisation report. Only this requested report is written.

Constraint references below mean:

* **C1:** identical numerical statements/order/values, dtype materialization and random draws.
* **C2:** component keys/formats, config schema/adapter, CLI, rerun rules and evaluation imports.
* **C3:** origin/change markers and formula labels remain meaningful beside retained code.
* **C4:** design choice 3, preserving Jan's class names and files.
* **C5:** F1/F4/F5/F6/F16 robustness remains in substance.

All P proposals preserve C4. Unless an entry names a narrower interface, C2 is unchanged. None removes
the existing run-folder refusal, deferred archival, unfinished-opening refusal, cumulative epoch
count, atomic dump replacement or periodic last dump. None proposes comment wording. Removing a
whole unused definition also removes its attached documentation; that is not a separate saving.
New helpers in Jan's files need the appropriate added marker. Retained labels move with their formulas.

Safety arguments below are static arguments, not new bitwise test results. The implementation report's
existing passes describe the present tree. A future implementation must regenerate clean outputs with
the fingerprint gate and run the listed layers. L1 = construction/forward, L2 = loss/gradients,
L3 = short training/artifacts, L3b = rejected polish, L3c = reruns, L4/4b = evaluation/loading,
6.4 = compiled GPU training and polish smokes. GPU results cannot be established by source reading.

## Ranked proposals preserving choice 3

### P01. Remove the temporary physics-matrix cache bridge: 14 lines, medium risk

`Clean/model_augmentation/fit_systems/blocks.py:163-169,328,381-401`;
`Clean/gantry_dynamic/obc_gantry.py:153-157`.

Have the trainable `nonlinear_function` select/build local matrices, slice the input, and call `_rk4`
directly. Remove the unused parent implementation, `_mats`, `_cur` initialization, the cache-only
pickle hooks and the float64-copy cache reset. Keep `pass_mats` and the pure transition/tangent APIs.
This removes indirection and serialization machinery for a value already supplied explicitly.

C1: build/select matrices before the input slices, matching the current trainable path; copy all
RK4 and derivative statements unchanged. C2: retain every class, buffer and parameter registration,
including nominal float32 buffers. Existing dumps may contain an unused `_cur` attribute. This proposal
specializes the retained hierarchy to its actual instantiated reduced block; no Clean/evaluation
caller instantiates the parent gantry class on its own. C3: keep the applicable D-192/D-195 markers
on the direct path. C5: checkpoint transaction boundaries remain. Confirm inherited nn.Module pickle
handling, loading pre-change clean dumps, deepcopy and compiled graphs before approval of an implementation.
Verify L1/2, L3/b/c, L4b and 6.4.

### P02. Share signal allocation and graph evaluation: 8 lines, medium risk

`Clean/model_augmentation/fit_systems/interconnect.py:80-99,201-212`.

Extract a helper taking inputs, the ordered output indices, and optional per-pass matrices. Both
`forward` and `output_only` use it; their reshape and OBC handling remain in their respective callers.
C1: copy the zero allocations, matrix multiplies, in-place additions and dependency traversal exactly.
The forward order and output-only dependency cone remain distinct arguments. Preserve the existing
no-pass block call when no matrices apply. No identity matmul becomes a slice, and no sum is reassociated.
C2: public signatures/results and buffer keys stay. C3: mark the new helper and preserve change markers.
C5 unaffected. Numerical equivalence of the eager statements is clear; compiler equivalence is conditional
on 6.4, since helper extraction can change graph capture. Verify L1/2, L3, L4/4b and 6.4.

### P03. Finish trimming unused OBC reporting: 8 lines, low risk

`Clean/model_augmentation/fit_systems/obc.py:19-24,36,63-67`.

Remove `RankReport.n_rows`, `singular_values` and `cond`, their unused calculations, and the duplicated
`OBCBasis.n_rows/n_cols` attributes. Construct the report from `n_cols`, `rank` and `tol`.
C1: Gram diagonal, column scaling, SVD, cutoff, rank truncation and coefficient solve stay unchanged;
only unread reporting values disappear. C2: no Clean/evaluation/harness consumers of those fields were
found. **Keep `OBCBasis.rank`**, which the layer-1 capture reads, and keep report rank/tolerance/column
count for rank-loss handling. C3 file marker stays; C5 rank-loss readiness policy stays. Verify OBC L1/2
and L3. Do not fold Gram/scaling removal into this proposal.

### P04. Replace constant-property wrappers: 6 lines, low risk

`Clean/gantry_dynamic/config.py:183-185,195-201`.

Represent `fs_new_hz`, `dtype_np` and `dtype_pt` as typed ClassVar attributes with the existing fixed
values. Three decorators, definitions and returns become three assignments. C1: the same integer and
dtype objects reach every caller; no data or random operation changes. C2: attribute names remain,
dataclass instance fields stay unchanged, and config.json values/keys and `from_config_json` remain.
This relies on the fixed RunConfig rather than supporting an undocumented subclass overriding `fs_new`.
C3 file marker stays; C5 unaffected. Verify config round trip, L1 and L4b.

### P05. Delete empty initialization hooks: 6 lines, low risk

`Clean/model_augmentation/fit_systems/blocks.py:23-24`;
`Clean/model_augmentation/fit_systems/interconnect.py:57-59,396`.

Delete `Block.init_block`, `Interconnect.init_model` and the call to the latter from SSE initialization.
Every retained block inherits the same empty hook. C1: this removes only empty tensor allocation and
no-op calls, with no RNG or value mutation. C2: actual SSE network/device/optimizer initialization stays;
none of these hooks is an evaluation import. C3 class trim markers remain. C5 unaffected. Verify static
callers, L1/2 and L3c.

### P06. Restrict internal signal-name parsing: 5 lines, low risk

`Clean/model_augmentation/fit_systems/interconnect.py:305-313`.

Map the four names used by model construction directly to their existing indices; delete numbered-name
parsing. C1: `x/xp` still map to 0 and `u/y` to 1 before connection-matrix construction. C2: narrows
unused generic wiring names, not the declared evaluation interface or CLI. Retain the method name.
C3 trim marker stays; C5 unaffected. Verify construction callers and L1/2.

### P07. Restrict internal connection-mode parsing: 4 lines, low risk

`Clean/model_augmentation/fit_systems/interconnect.py:315-322`.

Replace case-folding and synonym lists with the exact mapping used by callers: `concat`,
`concatenation`, `additive`. C1: the stored canonical modes and matrix-build order are identical.
C2: only unused generic aliases/case variants disappear. C3 trim marker stays; C5 unaffected.
Verify callers and L1/2. P06/P07 deliberately narrow generic construction utilities to the spec path.

### P08. Inline the single remaining rollout segment: 4 lines, low to medium risk

`Clean/model_augmentation/fit_systems/closed_loop.py:58-72,95-96`.

Move `_rollout_segment` into its sole caller, `closed_loop_rollout`, initializing local state from
`x0`. Segmentation is no longer used. C1: bank relocation, gathering, initial controller state, timestep
order, both model-call branches and output stack remain exactly ordered. C2: retain the public rollout
signature and result tuple. C3 retain the file marker and formula labels beside the loop. C5 unaffected.
Verify L1/2, L3, L4 and 6.4.

### P09. One definition of transient checkpoint attributes: 4 lines, low risk

`Clean/model_augmentation/fit_systems/interconnect.py:620-625,639-641,650-653`.

Use one tuple containing `obc`, `obc_reference`, `training_phase` for save stashing, load preservation
and serialization filtering. C1: no numerical work changes. C2: retain the exact attribute set and order,
save pop/finally-restore behavior, and load's restoration of only non-None live objects. Do not use one
generic stash operation for these different semantics. C3 keep the phase change marker. C5: atomic save,
opening progress fields and periodic saves stay. Verify L3/b/c, including unfinished-opening refusal.

### P10. Merge lifecycle refresh branches: 3 lines, low risk

`Clean/model_augmentation/fit_systems/obc.py:171-179`.

Use one body when the basis is absent or the epoch policy requires refresh. C1: preserve short circuiting
and exactly one `vbar_fn` evaluation and refresh in the same cases. C2 return values stay. C3 unchanged;
C5 consecutive-rank-loss handling stays. Verify OBC L1/2 and L3.

### P11. One restore-and-resync helper: 3 lines, medium risk

`Clean/model_augmentation/fit_systems/lbfgs_polish.py:57-60,292-293,321-322,359-360,374-375`.

Give the private restore helper the fit system and let it copy parameters then resync, replacing four
identical call pairs. C1: same copy order followed immediately by the same resync; resync remains outside
the helper's no-grad context, matching current scope. C2 outcomes and messages stay. C3 retain recovery
markers. C5: keep the broad F5 try boundary, retry/probe order, release of failed frames, cache clearing
and distinct abort metadata. This is a small helper consolidation, not a recovery rewrite. Verify L3/b/c,
injected exceptions at batch construction/final scoring, and GPU recovery. Existing CPU passes alone
cannot establish OOM safety.

### P12. Remove the unread physical-prior switch: 3 lines, low risk

`Clean/model_augmentation/fit_systems/blocks.py:307-308,324`;
`Clean/gantry_dynamic/model.py:50-52`.

Remove `flag_loss_reg` from the constructor, instance and call. No retained code reads it. C1: no
numerical statements or draws change. C2: keep `_RMSE_baseline`, its argument and `Lambda_combo`
materialization exactly, and retain PARAM_PRIOR/PARAM_RMSE_BASELINE in config.json. C3 the marker on the
removed argument leaves with that code. C5 unaffected. Verify L1 keys/values, L3c and L4b.

### P13. Inherit the forwarding constructor: 2 lines, low risk

`Clean/model_augmentation/fit_systems/blocks.py:89-90`.

Delete the constructor of `Discrete_Nonlinear_Function_Block`, which only forwards to `Block`.
C1: identical constructor receives identical arguments, no draws. C2: class and inheritance remain.
C3 retain class trim marker; C5 unaffected. Verify imports and L1/2.

### P14. Merge prediction refresh branches: 2 lines, low risk

`Clean/model_augmentation/fit_systems/interconnect.py:587-590`.

Refresh once when explicit refresh is requested or the basis is absent. C1: same condition, refresh
count, arguments and coefficient sequencing. C2/C3/C5 unchanged. Verify OBC L1/2 and L3/b/c.

### P15. Collapse simulator guards on the sole supported path: 2 lines, low risk

`Clean/model_augmentation/fit_systems/interconnect.py:402-403,415-416`.

Call the simulator and its validation method directly. The pipeline attaches it before either method
runs. C1: identical calls; keep validation probes afterward. C2: narrows unsupported simulator-free
usage; retain both methods, signatures and validation-measure forwarding for PS2/harness callers.
C3 trim markers stay; C5 polish-only initialization stays. Verify L1/2, PS2 L3 and L3c.

### P16. Remove aliases for fixed ANN factories: 2 lines, low risk

`Clean/gantry_dynamic/model.py:65-74`.

Pass `torch.nn.Tanh` and `live_g_feed_forward_nn` directly rather than binding `_act` and `_ann_net`.
C1: identical callable objects and construction timing, including n_a=0. C2 module registration and
checkpoint keys stay. C3/C5 unchanged. Verify L1 construction and RNG state.

### P17. Remove training-history aliases: 2 lines, low risk

`Clean/gantry_dynamic/training.py:318-322,352-354`.

Create the ValHistory as `history`, append that object, and restore the original `prev` directly.
Remove the second name for `prev` and the read-back through the just-created probe tuple. C1: same
object identities, callback order and finally restoration. C2 history arrays/events stay. C3 unchanged;
C5 best reload and opening cleanup stay. Verify L3 and L3c history arrays.

### P18. Delete unused network dimension attributes: 2 lines, low risk

`Clean/model_augmentation/utils/torch_nets.py:15-16`.

Remove `self.n_in` and `self.n_out` from the live ANN. C1 no effect on construction/draws; C2 these
plain attributes have no Clean/evaluation/harness readers and are not state_dict entries. Old dumps
can retain harmless extra attributes. C3 added marker stays; C5 unaffected. Verify L1 and L3c.

### P19. Share the MLP construction factory: 2 lines, medium risk, conditional

`Clean/model_augmentation/utils/torch_nets.py:17-27`;
`Clean/model_augmentation/fit_systems/pre_encoder.py:58-65`.

Extract their identical linear/activation stack and final-bias zeroing into an added factory returning
the Sequential and final layer. The main gain is one definition of creation order; line saving is small
after helper/import overhead and could vanish with wider formatting.
C1: construct input, hidden and output Linear layers in precisely the current order and default float32,
then Sequential, then bias zeroing, then caller-specific weight zeroing. Do not move the factory earlier
than encoder map draws or optimize away n_a=0 draws. C2: assign to the same `self.net` attributes with no
extra module wrapper, retaining encoder and nested ANN keys and registration order. C3 add the helper
marker, retain existing labels. C5 unaffected. The static recipe preserves draws, but acceptance remains
conditional on all L1 RNG captures, L2, PS2/OBC L3 and 6.4. Do not promote this to an implemented
bitwise guarantee without those results.

### P20. Use the known encoder final layer in PS2: 2 lines, medium risk

`Clean/model_augmentation/fit_systems/ps2_opening.py:125-128`.

Replace optional-net handling and a module walk with the deployed encoder's `enc.net[-1]`.
C1: retain auxiliary parameter order: Wa_u, Wa_y, final weight, final bias, after head parameters.
No new initialization or draw. C2: specializes PS2Opening to its sole Clean encoder; evaluation does
not import it. Generic encoders without this structure cease to be supported. C3 unchanged; C5 opening
progress/refusal and S2 candidate restoration stay. Verify PS2 L3 and L3c.

### P21. Drop unused optimizer group overrides: 1 line, low risk

`Clean/model_augmentation/fit_systems/interconnect.py:384,388-389`.

Remove `parameters_optimizer_kwargs`, unused by the sole caller, and build fresh group dictionaries
from `parameters_with_names.values()` in the same order. C1: preserve dictionary copies and parameter
order; do not hand mutable source entries directly to Adam. C2 optimizer property/moments, optimizer
kwargs and scheduler initialization stay. C3/C5 unchanged. Verify installed deepSI property semantics,
L2, L3 Adam state and L3c moments before implementation acceptance.

### P22. Return the weighted loss directly: 1 line, low risk

`Clean/model_augmentation/fit_systems/interconnect.py:613-618`.

Remove the immediate assignment/return of `L`. C1: use the exact existing expression and operations.
C2 unchanged; C3 formula label stays adjacent; C5 unaffected. Verify L2.

### P23. Return per-pass matrices directly: 1 line, low risk

`Clean/model_augmentation/fit_systems/blocks.py:385-388`.

Remove the immediate local assignment/return. C1 identical nested calls and returned object;
C2 `pass_mats` and tuple format stay; C3/C5 unchanged. Verify L1/2. Independent of P01.

### P24. Flatten tensor-conversion control flow: 1 line, low risk

`Clean/model_augmentation/utils/utils.py:13-20`.

Use an identity check for None and dedent the conversion after the returning Tensor guard.
C1: retain `torch.Tensor(input)`, preserving float32 materialization; no replacement by as_tensor.
C2: retain accepted values and current error translation. C3 added marker stays; C5 unaffected.
Verify L1/2. Exception-policy changes are outside this proposal.

### P25. Flatten normalization type handling: 1 line, low risk

`Clean/model_augmentation/utils/deepSI_corrections.py:52-59`.

Reject unsupported types first, then run normalization without the final else. Remove the redundant
subclass isinstance alternative. C1: same assertion and subtraction/division order for supported records.
C2: retain identity-transform behavior, signature and external-data error text. C3 file marker stays;
C5 unaffected. Verify L1 normalized arrays/windows and imports.

### P26. Inline the single-use physics kwargs dictionary: 1 line, low risk

`Clean/gantry_dynamic/model.py:46-58`.

Put the `_common` entries at the only constructor call. This mainly makes construction readable in
one place. C1: identical values and constructor order; dictionary values are locals/fixed config
attributes, so moving their lookup past combination construction adds no arithmetic or draws.
Do not move nominal/combination calculations or the dtype cast. C2/C3/C5 unchanged. Verify L1.
The estimate is marginal to P12 and does not count its deleted keyword again.

### P27. Remove overwritten polish-outcome initialization: 1 line, low risk

`Clean/gantry_dynamic/training.py:281,372-373`.

Delete the initial None assignment: the unconditional polish call assigns the variable before any
read. C1 no numerical work changes; C2 outcomes and conditions after the call stay. C3/C5 unchanged.
Verify static flow and L3/b/c.

### P28. Collapse the fixed successor branch: 1 line, low risk

`Clean/gantry_dynamic/obc_gantry.py:76-79`.

Use the current successor-valid index bound directly and append next states unconditionally.
RunConfig fixes `obc_ref_successor=True`, including configs reconstructed by the adapter. C1: identical
sample indices, reconstructed velocities and next-state values. C2: preserve the reference fields,
their dtypes and config key; only a non-spec fake config disabling successors loses support. C3 keep
relevant labels; C5 unaffected. Verify reference captures in L1 and overlap evaluation in L4b.

### P29. Use the known physics block position at OBC attachment: 1 line, low risk

`Clean/gantry_dynamic/obc_gantry.py:201-202`;
construction evidence `Clean/gantry_dynamic/model.py:57-63`.

Use connected_blocks[0] rather than copying the collection and searching it. C1: same module object;
C2: the component schema already fixes physics/output/ANN order, and only build_model calls this
attachment helper. Generic differently ordered graphs lose support here. C3/C5 unchanged.
Verify OBC L1/2 and L4b.

## Additional clarity proposals, zero counted saving

* **Q01, low risk:** `Clean/model_augmentation/fit_systems/interconnect.py:467,530`.
  Remove unread `val_counter` fields from the tuple assignments. C1: unused integer bookkeeping only;
  C2/C3/C4/C5 unchanged. Static flow plus L3.
* **Q02, low risk:** `Clean/model_augmentation/fit_systems/interconnect.py:609`.
  Call `tangent_pair` directly instead of falling back to theta. The sole constructed correction has
  this method. C1 selected call unchanged; C2 narrows unused correction types; C3/C4/C5 preserved.
  OBC L1/2 and 6.4.
* **Q03, low risk:** `Clean/gantry_dynamic/obc_gantry.py:107-113,207-209`.
  Remove unused `nu` constructor argument/attribute. C1 no arithmetic/draw change; C2 no reader or
  state_dict entry, old dumps may retain it; C3/C4/C5 preserved. L1 and L3c.
* **Q04, low risk:** `Clean/model_augmentation/fit_systems/lbfgs_polish.py:39-60,222-223,242`.
  Return identity-deduplicated parameters directly instead of retaining unused group names in tuples.
  Rename the private helper accordingly. C1 preserve traversal and first-identity order; C2 parameter
  order is critical to LBFGS and must remain. C3/C4/C5 preserved; integrate with P11 once. L2, L3/b
  and explicit parameter-order comparison.
* **Q05, low risk:** `Clean/model_augmentation/fit_systems/ps2_opening.py:93`.
  Simplify the screen condition to a screen-event-or-end condition combined with the unchanged
  threshold comparison. Boolean absorption preserves the outcome, including nonfinite calibration
  values; operands are pure scalars. C1/C2 status and phase results unchanged; C3/C4/C5 preserved.
  PS2 L3, including end and screen fallback branches.
* **Q06, low risk:** `Clean/gantry_dynamic/data.py:128-129`.
  Unpack the known third array as `r` directly rather than a one-element remainder list.
  C1 same arrays and np.asarray operations; C2 loader interface unchanged; C3/C4/C5 preserved. L1.
* **Q07, low risk:** `Clean/model_augmentation/fit_systems/interconnect.py:384`.
  Remove unused `nu` and `ny` initializer arguments. Values come from sys_data at line 387.
  C1 initialization unchanged; C2 unused internal keyword compatibility narrows; C3/C4/C5 preserved.
  Keep scheduler initialization and optimizer kwargs. Static callers and L1.

## Separate alternative: breaks choice 3

**X01: merge SSE_Interconnect and SSE_Interconnect_Composed**, at
`Clean/model_augmentation/fit_systems/interconnect.py:362-394,565-578`.
Estimated structural saving **4 to 8 lines before compatibility overhead**, medium to high risk.
This breaks choice 3: Jan's base class no longer maps cleanly to its original. Most methods still
have to exist, so the size benefit is modest.

C1: preserve deepSI parent initialization and parameter iteration. C2: keep the composed public class
name, generated dump basename and optimizer property; pre-change own-code dumps still reference both
class paths, so compatibility may consume all savings. C3: the inherited/added boundary becomes harder
to mark. C5: preserve all stash, opening-progress and atomic-save logic. L1/2, L3/b/c and loading
pre-change dumps are required. Do not count this on top of P proposals without recalculating overlap.
Recommendation: retain the existing classes unless Dirk values the flatter structure over provenance.

## Reviewed and retained

The 24 files were all read: the entry point; all eight gantry_dynamic files including its initializer;
model_augmentation's initializer; all eight fit_systems files including its initializer; both systems
files; and all four utils files. Counts include four intentionally empty model_augmentation initializers.

The numerical controller construction and systems/gantry_ss formulas have no compelling safe shared
implementation: the controller uses NumPy constants while the model intentionally materializes torch
float32 constants. The prefit, normalization and reference-set inversions likewise have intentional
dtype differences. Do not merge them based on mathematical similarity.

Retain these despite apparent redundancy:

* Nominal buffers, Lambda_combo, coordinate metadata, out_gate and all overwritten/n_a=0 draws.
* Connection matrices and their ordered matmuls/additions; primal and tangent RK4/derivative paths.
* Reference record/sample indices, names and successor states: evaluation/harness read them.
* Controller registry, gain handling, external-data guards, basis-builder space argument and timing
  return; evaluation has broader needs than training.
* The eager context, compiled retry/probes, scalar closure conversion and parameter ordering.
* The two seed points, fixed-loader RNG, validation-history object and optimizer property.
* The fixed_System_data constructor: inheriting the upstream constructor would alter x handling.
  Its apparently unused history-window keyword arguments may be passed by inherited deepSI list code.
* Separate source lookup and archive steps: combining them risks undoing F6's load-before-move fix.
* Empty package initializers and the package-origin guard.

No proposal silently repairs the existing three-dimensional-state issue in Interconnect.forward:
its shape flag is assigned only on the two-dimensional input path. Replacing that control flow would
also change unsupported-input behavior and should be reviewed separately, not counted as cleanup.

The report contains no implementation patch. Savings and safety must be reassessed against the merged
approved list before any source edit, especially P01/P02/P19 and the choice-3 alternative.

Completion checks: 24 Python files accounted for; all 24 source SHA256 hashes unchanged across the
review; 29 counted proposals; no prohibited dash sequences in this report. No training, simulation,
source import or git write operation was performed.
