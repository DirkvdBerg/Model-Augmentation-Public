# Final-run implementation review, 2026-10-16

Reviewable proposal for `tasks/handoffs/2026-10-16-final-run-clean-implementation.md`,
section 1 step 3. Two independent read-only analyses received identical briefs and
finished without seeing each other's output. Both agree on the core changes below.
Only this planning section has been added by the parent. No training source,
dataset, thesis text or run table has been edited; no jobs submitted or git writes made.

- [x] Complete two independent code analyses, covering all fixes and 17 revnotes.
- [x] Audit OBC duration, updates, epoch refresh and completed stages.
- [x] Reconcile implementation proposal and source conflicts.
- [x] Obtain the handoff's single approval of the proposal below (Dirk 2026-10-16: merged proposal
  approved; code changes wait for his go).
- [ ] Log approved clarifications before their corresponding implementation.
- [ ] Implement encoder, metrics, PS2, OBC, polish, BB, evaluation and provenance.
- [ ] Pass short CPU smoke runs for OBC, U and BB, plus affected row types.
- [ ] Prepare new run-table rows and exact server job list; Dirk submits and commits.

Abbreviations: GD = `scripts/gantry/gantry_dynamic/`; FS =
`model_augmentation/fit_systems/`; EV = `Thesis-writeup/Code/Evaluation/`;
BB = `scripts/gantry/blackbox-cl-trainable/`; EP =
`scripts/gantry/gantry_interconnect_dynamic.py`.
Locations describe current code inspected by the analysts, not completed fixes.

## Cross-reference of fixes and launch checks

| Item | Current location/behavior | A/B comparison and recommended change | Decision/status | Risk |
|---|---|---|---|---|
| 1 | GD/model.py:207 uses nominal linearization despite detuned start at :92. | Agree: satisfy through item 10 using actual start block; do not build another analytical initializer for final runs. | D-251 supersedes old remedy; proposed | High |
| 2 | FS/lbfgs_polish.py:94/:121 meter gate includes truth combination error from GD/training.py:652. | Agree: validation-only acceptance, finite-value checks and diagnostic meters retained. Thesis param_loss already zero with prior disabled, FS/blocks.py:1528. | D-255; proposed | Medium |
| 3, launch check | anti-alias/measure_aliasing_lowpass.py:40 and its log use correct data; filtered leak 0.22 to 0.50 nm. | Agree: completed; retain D-232 filter, no data/generator change. | Already correct | Low |
| 4, launch check | GD/data.py:199 and EV/eval_blackbox.py:82 use explicit paths, not stale pair field. | Agree: harmless for these paths; twin resolver also uses explicit noisy/free roots. | Already harmless; no data edit | Low |
| 5 | FS/ps2_opening.py:173/:183 supplies/scores M samples starting at k. | Agree: simulate M+1, discard first residual; target k+1 through k+M. Keep decoder size and sampling bound; preserve S2 input k. | Listed fix; decision entry before implementation | Medium |
| 6 | EP:346 defaults zero; GD/model.py:133 already supports live. | Agree: new final rows explicitly select MLP/live and save initializer metadata; preserve historical toggle. | D-248; proposed configuration | Low |
| 7 | FS/pre_encoder.py:478 restores means before both Wb and Wa. | Agree: item 10 final encoder uses centered histories with no offsets or separate physical bias; nonlinear physical output bias carries constants. | D-251 amendments; proposed | Medium |
| 8 | GD/model.py:39, EP:334 use 29 past plus current; FS/pre_encoder.py:367 allocates na+1. | Agree: exactly 30 past samples, zero right extensions, actual-length constructor. Avoid accidental 31-sample map; audit every window/scoring consumer. | D-250; proposed | High |
| 9 | GD/model.py:217 recomputes scales while GD/data.py:337 stores epsilon-adjusted scales. | Agree: item 10 supersedes final analytical path; prefit histories/targets use stored normalization. Audit no remaining final caller recomputes scales. | D-251; proposed supersession | Low |
| 10 | GD/model.py:204, FS/pre_encoder.py:399/:445 use analytical Wb and zero nonlinear outputs; no physical prefit. | Agree: random physical encoder prefit on closed-loop start-baseline simulated yhat/ucl/state. Fit Wb, shared hidden layers, physical output rows/bias; retain fitted values. Wa stays Xavier; added nonlinear rows stay zero. No LS warm start or oracle gate. | D-251 option A/amendment 2; proposed | High |
| 11 | GD/obc_gantry.py:129/:133/:135 keeps last stencil tuple; GD/obc_diagnostics.py:235 uses simulator successor. | Agree: k=2 through N-4, qd[k-2]; store reconstructed q[k+1]/qd[k-1] successor separately from simulator successor. Same full stack for coefficients/discrepancy. | D-252; proposed | High |
| 12 | FS/interconnect.py:1148 rebuilds basis; GD/training.py:746 passes stale bestfit to polish. | Agree: OBC remeasures rebuilt model before polish; U may reuse valid score. Label selected and rebuilt scores separately. | D-253; proposed | Medium |
| 13 | FS/closed_loop.py:678 discards per-axis scorer result; GD/evaluation.py:248 saves scalar history. | Agree: retain 6-by-3 RMS/denominators and update/epoch/event identity at every validation; persist incrementally outside checkpoint rewind. Identify selected entry; last chronological training score need not equal final model. | D-256; proposed | High |
| 14 | FS/closed_loop.py:664/:689 accepts sim-RMS and length-weights scalar records. | Agree: named NRMS selector returns equal mean of 18 ratios from common k0; reference shares trim/grid; retain old metric as diagnostic. Denominator conflict below. | D-256; proposed | High |
| 15 | FS/interconnect.py:1418 unweighted normalized MSE; FS/closed_loop.py:622 carries controller metadata only. | Agree: fixed record-axis weights through existing named-extra-array seam, separately from shared controller index. Weight is output std/servo denominator before squaring, identical in Adam/polish. | D-256; proposed | High |
| 16 | FS/lbfgs_polish.py:62/:639 absolute 1e-12 tolerance assumes old tiny objective. | Agree: short pilot measures weighted opening loss/gradient infinity norm and termination. B also flags tol_change. Calibrate from evidence, not invented constant. | Listed fix; log changed calibration | Medium |
| 17 | GD/training.py:748 requests sim-RMS in polish. | Agree: same NRMS selector as Adam, after OBC rebuild, validation-only acceptance. | D-253/255/256; proposed | Low |
| 18 | BB/train_bb_clmap.py:140/:146 native unweighted loss/sim-RMS; grid :33 selects old metric. | Agree: declared BB subclass for matched weighted windows, mean-of-ratios selection and history. Undo BB's own output normalization. No monkey patches. | D-256; proposed | High |
| 19, training | EP:333 selects tangent; GD/obc_gantry.py:242/:324/:387 already implements affine rank-11 protection. | Agree: reuse [J,Fbase] path, same span as [Phi,Gamma]. A accepts equivalent coordinates; B emphasizes exact coefficient conversion if exporting literal Gamma coordinates. Label internal basis explicitly. | D-257; proposed | Medium |
| 19, evaluation | EV/eval_negation.py:64/:67 and eval_overlap.py:67 assume J times ten-vector; full next-state cosine. | Agree: deployed correction API handles 11 coefficients; held-out cosine uses Fbase-current state. Freeze training coefficient, never refit on held-out tuples. | D-257; proposed | High |

## Additional required coverage

| Item | Location | A/B comparison and recommendation | Decision/status |
|---|---|---|---|
| SHA every config | GD/config.py:587/:609/:627; GD/evaluation.py:260; BB/train_bb_clmap.py:166; run_thesis.sbatch:102 | Agree: helper exists but fallback not exported, embedded/BB config lacks SHA. Resolve once, export submission fallback, require usable SHA for final jobs, save before fitting and in results. Identify dirty source separately; bare SHA is insufficient. | Handoff; log enforcement decision |
| Controller-rate check | Setup/controller_rate.py:44/:47; GD/data.py:148/:165; GD/controller.py:104 | Agree: comparison mixes point-time error/controller output with forward interval mean. Separate timing/filter/Tustin effects before selecting remedy; preserve D-087 force area. Parent calculation below. | D-087/099/141/232; diagnosis before remedy |
| NRMS reporting | EV/make_results.py:40/:100; eval_blackbox.py:86 | Agree: add per-record-axis NRMS, metres and floor; normalize before aggregation, separate multisine/motion-only. Use approved shared denominator. | D-256 amendment, D-259; proposed |
| Predicted first-order bias | EV/eval_overlap.py:83/:89; FS/blocks.py:1397 | Agree: float64 Phi(beta*) pseudoinverse applied to Xi on reconstructed-successor stack; map coordinate bias through combination Jacobian, including bounded m_diff. Compare local prediction to observed change, not a guaranteed nonlinear optimum. | D-252/259, thesis recovery equation; proposed |
| Multisine twins | GD/data.py:79/:137; EV/eval_greybox.py:87; eval_common.py:22 | Agree: explicit evaluation-only resolver for seven noisy/free pairs, stored controller metadata, existing training normalization. Never add twins to train/val. | D-258; proposed |
| k0 and new artifacts | EV/eval_common.py:25; eval_greybox.py:98; eval_blackbox.py:72 | Agree: derive history start from model/config, use common scoring interval; distinguish old 29/current-inclusive and final 30/past-only models, update run mapping/cache identities. | D-250; proposed |
| R8 final retrain | BB/runners/thesis_r8_grid.conf:35-42; run_bb_search.sh:43/:61 | Approved: run the full 11-configuration grid with matched loss/selector directly at the 80000-update cap on the A100 partitions (lanai, molokai); the selected run is R8, no separate retrain. Reason: in the 2026-10-11 search only index 3, the winner, had its best validation at the 20000 cap (`bb_search_88757_3.out`); selection at a truncated budget favours fast learners. Cost: 20000 updates take about 1.1 h on blade6 (`_3.out`), so about 4.5 h per point; blade1 is about 5x slower. Existing plateau criterion marks nonconvergence. | D-256/259; new entry for the budget change |
| BB NaN breaks | bb_search_88757_6/8/10.out | New finding: three of eleven points broke on a NaN training loss (index 6 at update 11461, 8 at 3199, 10 at 19021, all on blade1); deepSI reloads the best checkpoint, so their reported scores are pre-NaN bests, not converged results. Report them as failures in the search table (conf :30-31 requires it); the FIX-18 subclass keeps the NaN break and best-checkpoint reload. | Report only |
| m_diff scale | GD/training.py:432; Code/Figures/res_obc_combinations.py:34 | Agree: already mean actuator mass; reuse for new bias table. | D-259; correct |
| Single seed | EP:321/:328 | B lists separately; A agrees via D-259. Seed 1 with paired detuning, no unsupported spread threshold. | D-259; supported |
| PS2 S2 input | FS/ps2_opening.py:220/:226/:232 | Agree: recorded input k for target encoder state k+1 is correct; preserve during S1 shift. | Checked-no-change |
| Epoch refresh | FS/interconnect.py:1406; FS/obc.py:501; restore hook :1332 | Agree: coefficient every objective, basis every epoch; actual historical logs confirm all boundaries. | D-254; correct |

## All 17 revnotes mapped

| Revnote | A/B | Change/classification |
|---|---|---|
| III, 03_augmentation.tex:494 | Agree | FIX-8, GD/model.py:39, past-only history. |
| III :515 | Agree | FIX-7/10, FS/pre_encoder.py:478, centered history. |
| III :590 | Agree | FIX-1/10, GD/model.py:207, actual start physics. |
| III :594 | Agree | FIX-10, FS/pre_encoder.py:445, whole physical encoder prefit. |
| III :863 | Agree | FIX-13 to 18, weighted loss/selection/polish. |
| III :868 | Agree | No separate method change; verify baseline equivalence from same physical initial state, not equality of estimated and true states. |
| III :904 | Agree | FIX-10, retain fitted physical nonlinear rows. |
| III :909 | Agree | FIX-6, live g_aug. |
| III :983 | Agree | Editorial decoder-symbol collision, no training/evaluation change. |
| III :1003 | Agree | FIX-5, S1 target k+1. |
| IV, 04_obc.tex:210 | Agree | FIX-19, offset-inclusive protection. |
| IV :239 | Agree | FIX-11, successor tuples. |
| IV :339 | Agree | FIX-12, post-rebuild polish score. |
| IV :411 | Agree | Editorial vector-symbol choice, no training/evaluation change. |
| V, 05_setup.tex:316 | Agree | R8 matched search/final retrain preparation. |
| V :356 | Agree | EV/make_results.py:40/:100, NRMS reporting. |
| V :384 | Agree | EV/eval_overlap.py:83, first-order bias. |

Editorial classifications are proposed for approval, not already approved exceptions.
Thesis edits remain out of scope, as the handoff explicitly requires.

## Why the OBC jobs took less than a day

Logs share prefix `Thesis-writeup/Results/11-10-2026/thesis-results-2/`.
Line numbers here count LF-delimited lines; tqdm carriage returns give different
apparent line counts in some readers.

| Run | Start/end CEST, log references | Entire job | Adam interval | Polish |
|---|---|---|---|---|
| R1, job 88762, thesis-obc_88755_201.out | Oct 8 17:55:56 to Oct 9 13:04:39, :8/:2194 | 19 h 08 min 43 s | 16 h 22 min 17 s, :1994 | Accepted, 12 iterations/27 closures/1253 s, :2014/:2015 |
| R2, job 88755, thesis-obc_88755_202.out | Oct 8 17:55:55 to Oct 9 08:59:05, :8/:2200 | 15 h 03 min 10 s | 13 h 07 min 16 s, :2000 | Accepted, 12 iterations/27 closures/909 s, :2020/:2021 |

Both headers at :5 and :11 to :15 identify A100 80 GB PCIe GPUs and EPYC 9334
hosts, blade5/lanai and blade6/molokai. Both :105 report 85572 windows, batch 512,
167 updates/epoch: 200 * 167 = 33400. Logs finish all 33400; final summaries
R1 :2018 and R2 :2024 report 200 epochs. Configs under
`thesis_taf_lowpass_noise_linear_map/88762` and `88755` record EPOCHS=200,
N_ITS=null, NF=400, stride=10, epoch refresh.

Each log contains all 200 unique refresh epochs 0 through 199 plus one restoration
rebuild, 201 refresh records total. Final epoch refresh is R1 :1986, R2 :1992.
PS2 completes at :149 in both. No missing epochs, skipped opening/polish or sparse
refresh explains the short duration. Polish's 100 iterations were a cap; FS/lbfgs_polish.py:733
stops when PyTorch terminates an inner block. Historical output does not identify
which internal termination condition fired at 12. FIX-16 remains necessary.

Matched U/PS2 R4 (`thesis-u_88756_204.out`) ran the same 33400 updates on blade5
with the same host/GPU and a 6 h 42 min 40 s Adam interval. OBC intervals were
about 1.96 to 2.44 times this U interval. The old 48+ hour estimate is in
submit.sh:55 and D-242; PS2-METHOD.md:502 warns specifically about blade1/oahu.
Conclusion (hardware, established): matched old-node timings exist in
`scripts/gantry/meeting/meeting-01-10-2026/server/thesis-results/thesis_86892_*.out`
(same OBC arm, 200 x 167 updates). Runs 41 and 43 on blade1/oahu (RTX 2080 Ti) ran
at about 3.4 s/it and stopped at 32214 and 31319 updates after about 45 h; run 45 on
blade5/lanai (A100) ran at 1.74 s/it and finished in 16.1 h. The 48 h estimate came
from the oahu runs. Keep 72 h on hawaii, lanai, molokai; expect 13 to 20 h on A100
and about 30 to 35 h on hawaii (Quadro RTX 6000, run 49 at about 3.2 s/it).

## Controller-rate intervention

A short read-only calculation in the project Conda environment used existing
controller_rate/scipy helpers on TR-S1_r1. No source/data was changed. It excludes
first 0.5 s and last 100 low-rate samples. Relative RMS, X1/X2/Y:

| Force compared to K4 on filtered retained-grid servo error | X1 | X2 | Y |
|---|---|---|---|
| Raw forward five-sample mean, current comparison | 0.10840546 | 0.10915398 | 0.16722416 |
| Raw point sample | 0.00487848 | 0.00495952 | 0.00970427 |
| Filtered forward mean | 0.10840524 | 0.10915388 | 0.16722414 |
| Filtered point sample | 0.00486705 | 0.00495407 | 0.00970374 |
| Filtered centered five-sample mean | 0.00767485 | 0.00779763 | 0.01483264 |

Forward mean is centered 2/20000 = 0.1 ms after retained y/error. Removing that
timing difference reduces the discrepancy dramatically on this record; filtering
alone does not. This supports a dominant timing contribution here, not attributing
the full original number to Tustin warping. It is not an all-record corrected median
and does not justify point-sampling the training force: D-087/D-099 intentionally
preserve interval impulse, and point sampling previously caused drift.

Approved interpretation (diagnostic fix only): the point sample shows that K1 at
4 kHz reproduces the controller (0.5 to 1 %); the 10.8 % of the forward mean is the
force variation within one 0.25 ms hold, the inherent cost of closing the loop at
4 kHz in simulation, not an error in the D-087 reduction. Its output-level effect is
bounded by the oracle row of the closed-loop report (GD/evaluation.py:3-6): 4.0e-7 m
validation, 1.5e-7 m test, against 1.6e-6 and 1.4e-6 m for the model. Change only the
comparison in Setup/controller_rate.py; no preprocessing or controller-reduction change.

## Concrete choices (approved 2026-10-16, with the changes to 4 and 5 below)

1. Use population std(e) consistently for weights, selection, polish and reporting,
   following the explicit 2026-10-16 D-256 amendment/final thesis. Record a clarification:
   primary FINAL-RUN-FIXES still says RMS(e). Retain both denominator arrays to measure
   train/val differences. Reject zero/nonfinite denominators explicitly.
2. Start each baseline pretraining simulation at transformed first measured positions,
   zero velocities and zero controller state, verifying training-record initial rest/transient.
   Fit random physical encoder exactly as D-251 option A. Determine/log a deterministic
   bounded budget from a short simulated-state fit pilot; no truth-based gate or LS start.
3. Reuse affine OBC with explicitly labeled [J,Fbase] internal coordinates and exact
   Gamma-coordinate conversion where exported. Verify identical deployed correction.
4. Run the full 11-configuration BB grid under the matched objective at the 80000-update
   cap on A100 partitions; the selected run is R8. Report NaN breaks as failures and
   nonconvergence at the cap (see "BB NaN breaks" and "R8 final retrain" rows).
5. Controller rate: fix the comparison in Setup/controller_rate.py only; preserve the
   force-area reduction; no preprocessing change.
6. Approve the no-change/editorial classifications above. Recommend validation records
   for appendix design-choice comparisons, which the source explicitly leaves pending.
   Do not start excluded new detuned n_a=0 pair, 99% band edge or T3 low-frequency work.

## Minimal implementation sequence and proof

1. Record approved clarifications; add reference/denominator/provenance plumbing in GD.
   One metric definition serves both model families, retaining labeled old diagnostics.
2. Add final encoder class to existing FS/pre_encoder.py and explicit pretraining after
   controller setup, before joint optimizer creation. Preserve load-only/historical
   checkpoint construction. Common deterministic pretraining preserves plain/PS draw equality.
3. Reuse named window metadata for record-axis weights in Adam and polish. Add named
   selector, durable per-validation history, validation-only acceptance and rebuilt OBC opening score.
4. Apply PS2 S1 shift, successor alignment and affine selection, preserving lifecycle
   and frozen held-out coefficients. Add explicit coefficient conversion where required.
5. Implement declared BB hooks, then evaluation k0/artifacts/twin roots/caches, NRMS,
   first-order bias and increment negation. Keep twins evaluation-only.
6. Fix the controller-rate comparison and run the weighted-polish calibration; prepare new IDs
   with live initialization and usable provenance. Preserve historical rows; no submission/git writes.

Verification gates: unit weights preserve old loss; nonunit batch matches independent
calculation; shared-controller records retain distinct weights; polish chunked gradients
agree; mean of 18 ratios equals offline score; history survives best-checkpoint restore;
encoder simulated-state error decreases with Wa unchanged/added nonlinear rows zero;
30-sample history excludes current y; same-state initial augmented physical output equals
baseline; tuple count loses one per record with retained Phi unchanged and reconstructed
successors aligned; rank-11 projection is orthogonal to both Phi and Gamma; equivalent
coefficient bases give same correction; worsening truth meter cannot veto better validation;
rejected polish restores state; all seven twin pairs resolve without training inclusion;
synthetic local bias solve/physical-coordinate conversion recover known perturbations.

Finally run short CPU OBC+PS2, U+PS2, U plain, n_a=0 and BB launches using existing
smoke settings, only a few updates and short polish. Save/reload fitted encoder and
11-coefficient correction. Every config records usable SHA, approved metric, encoder
definition and actual initialization. CPU smoke does not validate GPU compilation;
state that launch limitation. Pre-register full production budgets before launch.

Framework functions/classes use @added; modified blocks use CHANGED markers.
No new framework module is necessary. Numerical formulas/constants carry matching
THEORY or HEURISTIC labels. No monkey patches, data regeneration, thesis editing,
server submission or git writes are included.
