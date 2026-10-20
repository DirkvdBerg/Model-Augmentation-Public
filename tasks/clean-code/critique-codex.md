# Independent critique of the clean-code design

2026-10-20. Static review against the current working tree. Only this file was created. No training, simulation, import of the pipeline, git writes, or evaluation edits were performed. The independent Claude critique was not read.

References below use design = `tasks/clean-code/design.md`, GD = `scripts/gantry/gantry_dynamic/`, MA = `model_augmentation/`, EV = `Thesis-writeup/Code/Evaluation/`, EP = `scripts/gantry/gantry_interconnect_dynamic.py`. Line references describe code read during this review; code is paraphrased rather than quoted.

The design is not ready to implement literally. Several copy/trim pairs leave dangling names or incompatible calls, and the historical-checkpoint evaluation plan does not work with the new configuration schema. The four handoff decisions also require coordinated amendments.

## Findings, ranked by consequence

### 1. High: the evaluation can silently select mainline instead of Clean

**Design:** 115 to 118, 783 to 788; layer 4 at 872.

**Source:** `EV/eval_greybox.py:105` imports `inspect_checkpoint` before its pipeline imports. `scripts/gantry/thesis-results/inspect_checkpoint.py:30` to `31` unconditionally puts the repository root and `scripts/gantry` at the front of the search path. Its `clamped` helper subsequently imports `Static_ANN_Block` locally at line 87. `EV/eval_common.py:19` to `21` is therefore not the only import-path writer.

**Consequence:** changing only eval_common as proposed does not isolate the clean implementation. In a fresh eval_greybox process, importing inspect_checkpoint can put mainline ahead of Clean before gantry_dynamic or model_augmentation is imported. Layer 4 can compare mainline against itself and falsely certify equivalence. With partially imported packages, it can also mix implementations and class identities.

**Fix:** include inspect_checkpoint's import side effects in the migration plan. Give the evaluation one authoritative package-root selection and make helpers respect it. At harness startup and after helper imports, assert that both package families' resolved module files are under the intended root. Retaining the helper directory on sys.path alone is insufficient.

### 2. High: copied constructors and imports refer to definitions explicitly dropped

**Design:** 310 to 311, 421, 441, 584; rule 1 at 29 to 32.

**Source:** `MA/fit_systems/blocks.py:8` imports zero_init_feed_forward_nn, and `Static_ANN_Block.__init__` uses it as its default at line 36. `MA/fit_systems/interconnect.py:18` imports three parameterized block classes that the mapping drops. Its constructor at 628 to 634 uses modified_encoder_net as a default, although design:440 drops that class. `GD/model.py:11`, `17` and `18` import discarded networks, gantry_linearization and normalize_linear_ss_matrices. `GD/training.py:20` to `21` imports train_model and WindowErrorStats, both removed or merged by the design.

**Consequence:** a literal copy-and-trim fails at import with ImportError or NameError, before any preset runs. Passing the live network explicitly does not avoid evaluation of a Python default when the class is defined.

**Fix:** specify import cleanup and signature changes as explicit mapping edits. Make the retained Static_ANN_Block default the live network, or require its network argument; remove the unused SSE encoder defaults and assignments as well as their arguments. Remove the dead imports. Do not restore entire discarded implementations merely to satisfy defaults that no kept caller needs.

### 3. High: RankReport is reduced but its copied producer still supplies deleted fields

**Design:** 490, 494.

**Source:** `MA/fit_systems/obc.py:124` to `137` defines twelve RankReport fields. `OBCBasis.from_matrix` at 222 to 233 constructs it with n_rows, rtol, col_norms, col_corr, build_seconds, svd_seconds and extra, in addition to the five fields the design retains.

**Consequence:** the first OBC basis build fails with an unexpected-keyword TypeError. This affects R1, R2 and R13 before their first training update.

**Fix:** either retain the dataclass fields, or mark from_matrix as trim and explicitly remove the corresponding constructor arguments and diagnostic-only calculations. Preserve the Gram diagonal, column scales, SVD and rank arithmetic exactly; the Gram diagonal also serves column equilibration and cannot all be discarded as report preparation.

### 4. High: deleting the debugging attribute conflicts with several copy rows

**Design:** 400, 405, 408 to 411.

**Source:** `MA/fit_systems/interconnect.py:39` creates debugging. The copied init_forward reads it at 224, 248, 255 and 265. The copied add_block and connect_signals read it at 455 and 470. The trimmed init_connection_matrices also retains accesses at 355, 383, 432 and 439 outside the specified 375 to 378 print removal.

**Consequence:** deleting the attribute as instructed produces AttributeError during model construction, usually in the first add_block call. Removing only the one listed print block does not fix it.

**Fix:** retain debugging=False, or explicitly trim every read and the corresponding constructor argument/call in build_model. Update the affected copy rows to trim. The latter is smaller but must be an enumerated change under rule 1.

### 5. High: copied config properties require a missing constant, and the encoder call has the same problem

**Design:** 173 to 174, 242, 521, 649 to 657, 662 to 664.

**Source:** `GD/config.py:555` and `559` read use_f64. It is absent from the proposed fields and ClassVar list. `GD/model.py:259` passes cfg.wa_encoder_init in the retained data_lsq constructor call; that attribute is also absent. `MA/fit_systems/pre_encoder.py:351` defaults that option to legacy_kaiming, while the selected branch at 428 to 435 is xavier_uniform_gain1.

**Consequence:** literal copies raise AttributeError while loading/building. Omitting the encoder keyword to resolve the error selects Kaiming instead, changing the added-state initialization, RNG consumption and every subsequent PS2/training result.

**Fix:** declare use_f64=True if the dtype properties remain copies, or mark both properties as rewrites returning float64. Declare wa_encoder_init='xavier_uniform_gain1', or explicitly replace/remove that keyword together with specialization of the encoder constructor. Record the exact call-site change; a future grep is not a substitute for this design decision.

### 6. High: the historical config round trip cannot construct the new RunConfig

**Design:** 629 to 635, 743 to 745, 785 to 786, 872, 897 to 898.

**Source:** `EV/eval_greybox.py:35` to `51` constructs options by intersecting dataclass field names with upper-case metadata keys. Mainline `GD/config.py:679` to `692`, `722` and the remainder of config_json_dict write MODE, OBC, NX_ANN and COMBO_INIT_DETUNE, not ARM, START, N_A and NOISE. The PS2 key does exist.

**Consequence:** dropping nf_override and na_nb_override is enough only for newly written clean configs. All nine historical layer-4 runs lack four mandatory new constructor arguments. Their evaluation raises TypeError before loading a checkpoint.

**Fix:** define a historical-metadata adapter: arm from OBC, start from COMBO_INIT_DETUNE, n_a from NX_ANN, and noise from a validated MODE mapping. Reject unsupported historical settings rather than silently guessing. Test both historical and new config schemas. The planned evaluation edits must include this adapter, or the harness must supply an explicitly documented converted config without altering source run artifacts.

### 7. High: whole mainline checkpoints need classes removed from the clean import namespace

**Design:** 283, 287 to 291, 446, 756, 824 to 828.

**Source:** `GD/training.py:105` unpickles the whole deepSI dictionary before extracting hfn and encoder at 117. `_install_nf_val_probe` at 675 to 678 attaches an _NfProbe; its constructor at 363 and 375 to 380 stores a system reference and WindowErrorStats instances. `MA/fit_systems/interconnect.py:902` to `908` saves improving checkpoints with those probes installed. The save stash at 1505 to 1512 excludes training_phase and OBC/traj objects, not validation_probes or loss_stats.

**Consequence:** the same module paths and component state_dict names do not suffice to unpickle a mainline _best.pth under Clean. The dropped _NfProbe and WindowErrorStats qualified names are required by pickle even if the evaluation never uses their methods. Preliminary evaluation, mainline-to-clean resume and checkpoint_status's own whole-file load can fail before the component extractor runs. Clean's own new checkpoints do not have this historical-class problem.

**Fix:** distinguish component-pair compatibility from whole-pickle compatibility. Either retain narrowly scoped serialization compatibility definitions, or convert historical whole checkpoints to component pairs in an isolated mainline process and define an explicit metadata/status reader for them. The latter also requires changing the preliminary-checkpoint evaluation plan; retaining resolve_checkpoint unchanged does not achieve it.

### 8. High: resume reinstatement must cover orchestration, polish semantics and metadata together

**Design:** 147, 286, 293 to 300, 390, 473 to 482, 654, 694, 753 to 755, 947 to 961.

**Source:** `EP:399` to `400` resolves resumed hp, and 444 skips encoder prefit for a resume. `GD/training.py:295` to `315` restores Adam moments or reports their absence; 851 to 898 loads the checkpoint, computes remaining epochs and suppresses a new PS2 opening; 904 to 908 preserves the epoch count for polish-only starts; 975 to 1007 drops optimizer state when appropriate and records real resume provenance. `GD/training.py:788` deliberately excludes polish-only starts from reuse of the saved bestfit. `MA/fit_systems/lbfgs_polish.py:638` to `686` warms lazy state, tries compilation when available and falls back to eager. The simulator is freshly compiled on this entry path, unlike the post-fit case.

**Consequence:** restoring only resolve_resume_hp and load_optimizer_moments still leaves the requested behavior broken. A resume can redo the expensive encoder fit, replay PS2, claim 200 completed epochs for a polish-only run, save misleading empty Adam state, or compare against a saved score where mainline remeasures it. Eager-only specialization is no longer generally equivalent once polish-only entry is retained.

**Fix:** replace choice 12 and propagate the change through every listed row. Add resume_checkpoint and start_phase as real arguments/fields, skip prefit on resume, preserve mainline phase gating and provenance, and restore the necessary compiled/eager polish machinery. Add separate mainline-versus-clean tests for Adam resume from a component pair, an interrupted-run checkpoint in the supported format, and polish-only from pre_lbfgs. Compare moments, PS2 attachment, before/after scores, epoch metadata and optimizer omission. Compare each resumed implementation against mainline resume; mainline does not promise equality to uninterrupted training.

### 9. High: the folder and root arguments have not been propagated into the design

**Design:** 58 to 67, 103 to 106, 115 to 118, 178, 185, 215, 683 to 695, 704 to 706, 783 to 788, 912 to 914.

**Source:** `GD/config.py:25` to `26` derives REPO_ROOT from the current source location; 603 to 606 derives save_dir from it. `GD/data.py:143` uses the dataset root selected by traj_dir. `EP:386` derives the run directory through save_dir. `EV/eval_common.py:19` to `21` supplies package roots independently.

**Consequence:** the old ROOT/../../.. rule is wrong once ROOT means <repo>/Clean. Fixed DATA_ROOT and OUT_ROOT also do not implement the authorized data/output-root arguments. A literal location-only replacement can read or write outside the intended repository tree or keep the implementation tied to its previous nesting depth.

**Fix:** set the tree under <repo>/Clean; define repo defaults from Clean's actual parent; add data_root and output_root to main/config and CLI, with the specified defaults; derive traj_dir and save_dir from those arguments. Serialize the resolved roots so evaluation can reconstruct the same data selection, with an explicit override for moved datasets. Update every import-path example and harness launcher. No new numerical algorithm is needed.

### 10. Medium: the GPU compilation acceptance test disables compilation itself

**Design:** 689 to 691 versus 887 to 893.

**Source:** `GD/controller.py:329` to `341` creates compiled callables and prints the required COMPILED line only when compile_mode is not None. `MA/fit_systems/closed_loop.py:509` stores that pair.

**Consequence:** the prescribed smoke word sets compile_mode=None, so the required COMPILED line can never appear. The test provides no evidence for removing device self-heals from the compiled route, despite choice 14 relying on that evidence.

**Fix:** define a GPU harness budget that retains reduce-overhead compilation while shortening data/updates and PS2. Keep the existing eager CLI smoke behavior if desired. Run both implementations with that same explicit compiled budget; assert the compiled branch was actually invoked, not merely that wrappers were constructed.

### 11. Medium: the four-layer plan does not exercise several claimed evaluation interfaces

**Design:** 727 to 742, 770 to 779, 868 to 872.

**Source:** `EV/eval_overlap.py:46` to `53` calls make_basis_builder with space; 106 to 140 uses reference tuples, reconstructed successors and sensitivity matrices. `EV/eval_negation.py:76` to `103` uses deployed correction and record grouping. `EV/eval_greybox.py:70` to `86` sets frozen-Y attributes. `scripts/gantry/thesis-results/inspect_checkpoint.py:94` to `125` differentiates the loaded model to obtain added-state poles.

**Consequence:** matching greybox predictions does not validate overlap/bias, negation, frozen-Y behavior, or the retained poles helper. In particular design:267 lists a signature without space, whereas 741 requires it. An implementer following the mapping signature can pass all four proposed layers and still break overlap and negation with TypeError.

**Fix:** explicitly retain the space keyword in the mapping, specialized to affine as appropriate. Add small interface comparisons on loaded checkpoints for deployed_basis, overlap/bias, negation, added-state poles and frozen-Y predictions. These are verification tasks, not changes to the method or a request to implement diagnostics inside Clean.

### 12. Medium: the validation-score acceptance is a printed diagnostic, not an assertion

**Design:** 872 claims both evaluations must pass the existing relative 1e-3 check.

**Source:** `EV/eval_greybox.py:196` to `199` computes relative discrepancy and prints OK or FAILED. Unlike the report comparison at 187 to 190, it does not assert or raise. It writes val_score and ref_score into the prediction cache at 204.

**Consequence:** a subprocess can return success and produce caches while failing the stated checkpoint-score criterion. Treating process success as harness acceptance misses that failure.

**Fix:** have the harness independently assert the finite validation-score comparison from the returned/cache values. Also assert a finite positive reference score, rather than relying on text matching alone.

### 13. Medium: D-274 golden checkpoint selection needs matching metadata and report handling

**Design:** 872, 897 to 905.

**Source:** `GD/training.py:935` to `937` stores the selected Adam bestfit beside pre_lbfgs weights, while 946 to 962 computes the final score from the rebuilt model or accepted polish. `EV/eval_common.py:115` to `119` selects the unsuffixed component pair and reads its bestfit. `EV/eval_greybox.py:178` to `190` checks any existing final run report against the loaded predictions.

**Consequence:** where the D-274 decision changes which final model is kept, merely naming pre_lbfgs as the layer-4 reference is insufficient. That pair's metadata contains the selected score, not necessarily the rebuilt score, and the existing report describes the original final model. The standard evaluator still loads the unsuffixed pair. Comparing the new model with the old report can reject an intended change or select the wrong reference.

**Fix:** specify a golden staging directory and a complete selection procedure per affected row: choose weights, recompute the saved model's reference score through mainline, write consistent final metadata, and regenerate or omit the obsolete report there. Preserve the original server artifacts. Distinguish intentional D-274 re-decision from clean-versus-current-mainline equivalence.

### 14. Low: source-only golden fingerprints do not pin the numerical input

**Design:** 860 to 862.

**Source:** `GD/data.py:79` to `90` obtains membership/order from MANIFEST.csv; 210 loads each record; `GD/obc_gantry.py:100` loads records again for the basis reference set. `GD/model.py:379` to `405` groups and concatenates prefit inputs by the resulting record lengths/order.

**Consequence:** identical copied-source hashes can accompany changed manifests or MAT files. The harness would call an old golden current even though window ordering, normalization, LS rank and OBC basis inputs changed.

**Fix:** fingerprint both manifests and the actual record files used by each golden, plus the resolved configuration and data root. Check those fingerprints when reusing goldens. No training rerun is requested by this critique.

## Required changes for Dirk's four decisions

1. **Clean folder:** apply finding 9 throughout sections 1, 3, 5, 7 and the harness. Replace the old nesting-depth calculation, not just the displayed directory name.
2. **Data and output roots as arguments:** apply finding 9 to config, CLI, main, traj_dir, save_dir and reconstruction metadata. Keep the two agreed defaults.
3. **CLI with no-argument R1 lowpass:** design:683 to 684 and choice 4 currently require both positional arguments. Make their defaults explicit, and verify an empty argument list resolves to the R1 axis tuple in design:670 with noise lowpass. Keep CPU and smoke overrides and add the root/resume arguments. Evidence for the preset's intended method is `EP:342` to `372` and `EV/eval_common.py:40` to `42`.
4. **Resume including polish-only:** apply finding 8, including choice 16 and the equivalence plan. Keep mainline's phase-specific behavior rather than assuming every polish follows fit.

## Checks that did not produce a finding

- **Truth fields and six reference-set fields:** the proposed retained fields cover the current overlap/negation readers. `GD/obc_gantry.py:131` to `154` reconstructs the operative state from y and its finite difference, not recorded x_logical or absorber truth. `EV/eval_overlap.py:121` to `140` uses x_phys_next_rec or the retained tuple/index fallback; `EV/eval_negation.py:93` to `103` uses Z_step, record_ix and record_names. Keep reconstructed successors and their alignment. There is no reason here to restore discarded truth fields.
- **ValHistory:** record/hist_arrays carry the selection and final event data in `GD/training.py:593` to `625` and 939 to 962. Removing the nf diagnostics does not itself change bestfit. Preserve the external history reference across the best-checkpoint restore. Finding 7 concerns historical pickle names, not the new history arithmetic.
- **Device self-heals:** the stated scoped argument is supported by `MA/fit_systems/interconnect.py:77` to `91`, 267 to 277 and the registered out_gate buffer at `MA/fit_systems/blocks.py:67`. No definite device mismatch was found for normally constructed clean models. The compiled GPU test must first be made executable as in finding 10; static review is not GPU verification.
- **RNG:** the specified live network, Wb, Wa and encoder network order agrees with `GD/model.py:143` to `151`, 251 to 261 and `MA/fit_systems/pre_encoder.py:376` to `377`, 432 to 435, 446 to 459. PS2's lazy global head draws remain at `MA/fit_systems/ps2_opening.py:148` to `150`. Fix finding 5 without changing this order. No additional global draw from the proposed diagnostic removal was identified.
- **D-274:** the current code keeps the S2 start candidate at `MA/fit_systems/ps2_opening.py:233` to `235`, K1 manifest filter at `GD/data.py:89` to `90`, and selected-score polish comparison with rebuilt rejection model at `GD/training.py:784` to `796`, 946 to 962. The design's normal Adam path reflects these changes.

## Mapping coverage and limits

All seven retained fit_systems files were spot-checked: blocks (findings 2 and 5), closed_loop (findings 8 and 10), interconnect (findings 2, 4 and 7), lbfgs_polish (finding 8), obc (finding 3), pre_encoder (finding 5), ps2_opening (current D-274 verified). GD/model and GD/training were checked through construction, prefit, checkpoint selection, history and polish orchestration. The imports and top-level definitions of the seven fit_systems files dropped whole were checked; no additional required training branch was found there. Removing their incoming imports remains necessary under finding 2.

No runtime numeric equivalence is claimed. Clean does not exist yet, and the task forbids training/simulation. The proposed repairs need import checks first, then the agreed four-layer harness with resume and interface cases added. No reason was found to re-decide the other agreed method choices.
