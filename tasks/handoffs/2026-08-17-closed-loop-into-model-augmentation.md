# Handoff: implement the closed-loop controller training path inside `model_augmentation/`
**From**: session of 2026-08-17 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

Execute the migration specified in
`scripts/gantry/closed-loop-controller/PLAN-move-to-model-augmentation.md`. That document is the
specification and it is complete: it fixes the design (four seams in `interconnect.py`, one new
`model_augmentation/fit_systems/closed_loop.py`, composition instead of a new fit-system subclass),
the migration order (steps 0 to 8 in section 5), the acceptance criteria (section 5.1, three
reference sets A/B/C with data-derived tolerances), the gates that must survive (section 7), and
the attribution labels required in the code (section 6). Work through section 5 in order. Where the
plan and the current code disagree, the plan wins and the code changes. The end state is: no
monkey patch anywhere in the training or validation path, exactly one closed-loop rollout
implementation, `param_loss` and the orthogonality penalty inherited rather than copied, and the
existing MATLAB gates still passing.

## 2. Out of scope

- **The write-up.** `scripts/gantry/closed-loop-controller/documentation/controller-implementation.tex`
  and `controller-slide.tex` are done and compiled. Do not edit them.
- **`Cfb`'s design.** `ruleOfThumb.m` lives in the read-only FP model. `RECORD_Y_OP`, `y_op_for`,
  `build_cfb_at` and `controller_ss` stay in `scripts/gantry/`; the framework never learns what
  `Y_op` or `ruleOfThumb` are (plan rule 5).
- **`multiple_shooting.py` removal.** It is marked CANDIDATE FOR REMOVAL in
  `gantry_dynamic/model.py` and that is a separate decision. Step 4 only routes its inner loop
  through `self.simulate()`; do not delete it, do not enable `n_seg > 1`.
- **`zero_the_ann`.** DECIDED to keep as a monkey patch (plan 3.6d). It is gate-only and fails
  visibly. Do not convert it.
- **The equivalence and precision questions.** All settled, see section 4. Do not re-run them.
- **`docs/decisions.md:45` and `docs/gantry-augmentation-problem-log.md:739`.** Their 20 kHz `D_c`
  value is correct in context; leave both.

## 3. Where things stand

Branch `Augmentation`, last commit `d4582cf` "Add closed-loop-controller diagnostics and
implementation". Tree is dirty across many directories; the ones that matter here are
`scripts/gantry/closed-loop-controller/` (four new scripts and the plan, all uncommitted) and
`model_augmentation/fit_systems/` (pre-existing modifications, not from this session). No run is
in flight. Nothing from this session is committed.

New this session, all uncommitted:
- `PLAN-move-to-model-augmentation.md` (the specification)
- `cl_direct_vs_residual.py`, `cl_precision_gradient.py`, `cl_precision_validation.py`
- `documentation/controller-implementation.{tex,pdf}`, `documentation/controller-slide.{tex,pdf}`
- one docstring edit in `cl_controller.py` (the `D_c` correction)

## 4. Established and verified

**The residual form is exact for the nonlinear model.** `cl_direct_vs_residual.py`: direct form
(driven by `r_sim`, `f_sim`) versus residual form (driven by `u_total`, `y`), same nonlinear
augmented model with the ANN active. Float64, 100 closed-loop steps: **2.62e-14 m**. The same
comparison in float32: 1.79e-03 m, i.e. the gap scales with machine epsilon, so it is not an
algebra error. The one-step gap was `2.400e-02` N identical at ANN gains 0, 0.01 and 0.1, i.e.
model-independent. The long-horizon gap tracks a deliberate one-eps perturbation of `x0` step for
step. Supervisors' objection ("you cannot subtract because the plant is LPV and the ANN is
nonlinear") does not hold: at the subtraction step `y_d` and `y_m` are signals and `f` never
appears.

**The condition that IS real**: both loops must apply the same operator. They do, because `Cfb` is
frozen at the record's `Y_op`, which is exogenous. A controller scheduled on the model's own state
would break the subtraction while still being linear.

**`delta` is the 20 kHz to 4 kHz gap, and it is benign for training.** `u_total` differs from
`Cfb(r - y_d) + f_ms` at the training rate by 10 % of `u_fb` on V1 and 1.6 to 2.7 % on T10. This is
the same thing as the operator mismatch above. It perturbs the path, not the optimum: the
correction `Cfb(y_d - y_m)` vanishes as `y_m -> y_d`, so the training fixed point is independent of
the controller discretisation. It does affect the validation free run, where `y_m != y_d`
persistently.

**Training precision: float32 is fine.** `cl_precision_gradient.py` on
`server-results/deep-SI-checkpoints/FitSys_ClosedLoop_Go1qTA_best.pth`, batch of 16 windows at
`nf = 400`: `cos(g32, g64) = 0.999997769`, gradient norms agree to 0.03 %, against a batch-to-batch
cosine of **-0.218**. The precision disagreement is 1.8e-06 of the scatter SGD already tolerates.
Closed loop is 9x more precision-sensitive than open loop (`1-cos` 2.2e-6 versus 2.5e-7), the `D_c`
amplification showing up as predicted and nowhere near mattering.

**Validation precision: float32 too.** `cl_precision_validation.py`, both checkpoints, four
validation records, full-record closed-loop free runs. Ranking does not flip. Selection scalar
shifts by at most **7.6e-11 m** against a checkpoint gap of **1.39e-09 m**, i.e. 5.5 %.
`Go1qTA_best` scores 1.393335e-06 m in float32.

**The loop steps at 4 kHz and `D_c` was wrong in one place.** `ControllerBank` is built with
`cfg.ts_new` everywhere in the training path. Through the real builder, `controller_ss(0.0, ts)`:
`diag(D_c) = [2.844, 2.914, 1.509]e6` N/m at 20 kHz and `[8.055, 8.253, 4.275]e6` at 4 kHz. Cause:
`D_c = kappa_j * Cnorm(2/ts)`, and Tustin sends `z = inf` to the finite `s = 2/ts`, which lands at
different points on `Cnorm`'s roll-off (0.1307 versus 0.3701, ratio 2.832). Fixed in
`cl_controller.py`'s docstring this session.

**Feasibility of the four seams.** All call sites are inside `SSE_Interconnect.fit`, which is
already a `# CHANGED:` replacement of deepSI's: `make_training_data` at `interconnect.py:568`,
`loss` at `:623`, `cal_validation_error` at `:518`/`:522`. `make_training_data` is defined on
deepSI's `SS_encoder_general` and `cal_validation_error` on deepSI's `System_fittable`, so both are
plain overrides in our file. **The installed deepSI package does not need editing.**

**The output's dependency cone is one block.** Measured on the built model:
`out 1 (y) <- [3]`, `out 3 (Linear_Output_Block) <- [0, 1]`. So `y` costs one
`Linear_Output_Block` forward and needs neither the ANN nor the state block. Also measured:
`sig3 <- u` is a (9,3) matrix with unit entries, so `u` really is wired into the output path and
`D_d = 0` holds because the block's coefficients are zero, **not** because the graph forbids
feedthrough. `check_no_feedthrough` is therefore not redundant and must be kept.

**Two banks are not needed.** `cl_step6_run.py:134-136` builds `bank_tr` and `bank_va` only because
`rec_ix` is a per-list position, so index 0 means `T1` in one context and `V1` in the other. `Cfb`
is per trajectory; train versus validation is not an axis. One bank over all 22 records, indexed
globally, replaces both.

**Kessels, verified against the PDF** (`literature/augmentation/kessels2025_ai-control.pdf`, PDF
page = thesis page + 26). He **does** use truncated-window training: Figure 5.3 p155, `V_T` at
(5.12) p156 with `C = n_TW*T := (N-T+1-n_o)*T`, encoder constraint (5.13a) p156. His Remark 5.4
p157 reconstructs the feedback state from measured `y`, the reference and the controller, and
initialises it **per window to a nonzero value**, because his (5.13d) filters `r_bar - y_hat` which
exists before the window. Ours filters `y_d - y_m`, which does not. So `xc = 0` is a definition,
not Remark 5.4, and the code must say so as a contrast (plan section 6). `cl_controller.py`'s
docstring is the correct version; the earlier draft framing ("identical assumption, different
algebra") is wrong.

**Two production monkey patches on `cal_validation_error`, not one.**
`gantry_dynamic/training.py:181` `_install_nf_val_probe` patches it on every run, with a
`__reduce__` no-op to survive pickling; `cl_validation.py:169` installs the closed-loop validator.
`cl_validation.py:118` documents the collision.

## 5. Assumed but not verified

- **That the `simulate()` extraction can be bit-identical.** The base `loss()` means per-timestep
  `mse_loss` values while the seam takes one `mse_loss` over the stacked prediction, and `fit()`
  applies `sqrt_train` downstream. It should be identical because every timestep has the same
  element count. Not demonstrated. This is migration step 2's assert and the largest execution risk.
- **`concurrent_val`.** `cl_sanity.py` records it MUST be False for the closed-loop path. Whether
  that is fundamental or an artefact of a patched method not surviving the process boundary is
  unknown. Settled by trying `concurrent_val=True` with the seam in place.
- **The `augment_training_data` signature.** Invented API. The fifth-array mechanism is proven, the
  method shape is not. Settled by reading deepSI's `to_hist_future_data` return and `fit`'s
  `self.loss(*train_batch)` call.
- **Where the runtime goes.** Section 3.8's table is priors, not measurements. Migration step 0 is a
  profile precisely so nothing gets optimised on a guess.
- **That `Static_ANN_Block.output_scale` would be a clean off-switch.** Sketched, never written.

## 6. Tried and failed

- Version 1 of the equivalence test compared 20000-step rollouts -> agreed to 4.6e-09 m with the
  ANN off but disagreed at 7e-04 m with it on, comparable to the model error -> a single
  end-of-rollout number cannot separate "the algebra is wrong" from "the algebra is exact and two
  floating-point programs diverge exponentially over 20000 steps" -> stopped mid-run and rewrote
  with a one-step test, a growth curve against an eps-perturbation control, and a float32/float64
  comparison. Only the last is decisive.
- Version 1 also carried an ANN-activity probe that printed `0.000e+00` at every gain -> a broken
  forward hook -> dropped it; the model error moving 3.9e-06 to 1.0e-01 already proves the ANN was
  active.
- First `cl_precision_gradient.py` run died on `operands could not be broadcast together with
  shapes (48000,3) (3,1)` -> the pipeline's `Norm` dataclass stores `(nu, 1)` columns -> normalise
  from `fs.norm` (deepSI's own object), which is what the encoder and `hfn` were built with. Mixing
  the two silently rescales the loop rather than erroring.
- Deriving the controller from a window's own measured `Y` to delete the `rec_ix` plumbing -> wrong,
  and silently so on the `ysweep` records -> the machine froze `Cfb` at the record's nominal
  `Y_op`, so a per-window derivation applies a different operator than the machine used, breaking
  the condition the residual identity depends on. Recorded as rejected in plan 3.5.
- Tagging `System_data` with a record name so identity travels with the data -> `fit()` calls
  `norm.transform(train_sys_data)`, which returns new objects, so a plain attribute is dropped
  mid-pipeline and surfaces as a silent `None` -> rejected in plan 3.6; register `(name, sys_data)`
  with the simulator instead and assert by content.
- Building the closed loop as `SSE_Interconnect_ClosedLoop(SSE_Interconnect_OrthLoss)` -> couples
  two unrelated concerns and adds a fourth link to a chain where order is load-bearing -> rejected
  in favour of composition (plan 3.2). User declined this explicitly.

## 7. Achieved

**Specification, complete and reviewed against the code:**
`scripts/gantry/closed-loop-controller/PLAN-move-to-model-augmentation.md`, 9 sections. Feasibility
checked against `interconnect.py`, `gantry_dynamic/` and the installed deepSI.

**Measurements, all implemented and validated** (see section 4 for the numbers):
- `cl_direct_vs_residual.py` (486 s) -> the residual form is exact for the nonlinear model
- `cl_precision_gradient.py` (21 s) -> float32 training is fine
- `cl_precision_validation.py` (1642 s) -> float32 validation is fine, ranking does not flip

**Write-up, implemented and compiled:** `documentation/controller-implementation.{tex,pdf}` (2
pages, four steps: given controller, interconnection, residual form, separate block) and
`documentation/controller-slide.{tex,pdf}` (one 16:9 frame).

**Correction applied:** `cl_controller.py` docstring now carries the 4 kHz `D_c` with its rate and
an explanation of the rate dependence.

## 8. The open question

Nothing blocked. The specification is write-ready and the three remaining unknowns (section 5) are
all checks that resolve inside their own migration step rather than decisions that change the
design.

One thing worth surfacing rather than deciding silently: the migration invalidates
`FitSys_ClosedLoop_Go1qTA_{best,last}.pth`, because they pickle a class created at runtime by
`attach()` which stops existing. DECIDED: no shim, retrain (plan section 8). Every number already
extracted from those checkpoints survives in `server-results/step6_result_76573.json` and in this
folder's scripts.

## 9. Next action

Migration step 0: profile one training step on the current code, so section 3.8's priors are
replaced by measurements before anything is optimised. Then step 1, the reference sets. Run:

```
cd scripts/gantry/closed-loop-controller
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output -n GraduationProject \
  python -u cl_step6_run.py            # or a cProfile wrapper over one loss()+backward()
```

Rationale: step 0 is cheap and it decides whether the block-diagonal `Cfb` storage and the einsum
fusion in plan 3.8 are worth writing at all. Doing it first prevents optimising a few percent while
the 400 sequential `hfn` calls dominate.

## 10. Acceptance criterion

Per plan 5.1, three references, MATLAB first because it is the only external ground truth:

- **A, MATLAB.** `test_controller_exact.py` L1 coefficients `<= 1e-11` (9.6e-12 already achieved at
  20 kHz), `verify_controller.py` at 4.5e-09 relative, `verify_cfb_against_records.py`,
  `p1_equivalence.py`. **Plus the new one**: export `c2d(kappa*Cnorm, 1/4000, 'tustin')` from
  `export_controller.m` and run L1 at the training rate. Nothing currently checks the 4 kHz
  controller against MATLAB at all.
- **B, R2 regression.** Closed-loop loss relative `<= 1e-3` (float32/float64 differ by 3.1e-4),
  gradient `1 - cos <= 1e-5` (float32/float64 give 2.2e-6), validation selection scalar `<= 1e-10` m
  (float32/float64 shift is 7.6e-11 m, checkpoint gap 1.39e-09 m).
- **C, R1 seam no-op.** Bit-identical with `simulator = None`.

Plus greppable: no `__class__ = type`, no `cal_validation_error =` assignment, exactly one
`closed_loop_rollout` definition. End-to-end: a retrain reproducing 36.3 % improvement within a few
percent, as a sanity band not a parity assert.

## 11. Read these first

1. `scripts/gantry/closed-loop-controller/PLAN-move-to-model-augmentation.md` — the specification;
   everything else is context for it.
2. `model_augmentation/fit_systems/interconnect.py`, `SSE_Interconnect.loss` and `fit` — where the
   four seams go.
3. `scripts/gantry/closed-loop-controller/cl_controller.py` — the rollout, `ControllerBank`, and the
   `xc = 0` reasoning that must be carried across verbatim.
4. `scripts/gantry/closed-loop-controller/cl_fitsys.py` — what is being deleted, and why its
   `attach()` docstring exists.
5. `scripts/gantry/closed-loop-controller/cl_plant.py` — `identify_output_map` and the three checks
   that become tests of `output_only`.

## 12. Do not

- Do not edit the installed deepSI package. Every seam is an override in `interconnect.py`.
- Do not create a fit-system class at runtime, patch `cal_validation_error`, or add a link to
  `ParamLoss -> OrthLoss -> MultipleShooting`.
- Do not write a checkpoint compatibility shim.
- Do not derive `Cfb` from a window's measured `Y` (section 6).
- Do not tag `System_data` with a name (section 6).
- Do not re-run the equivalence, gradient-precision or validation-precision experiments; they are
  settled in section 4.
- Do not touch `kamtin-fp-model/`, the two `documentation/*.tex` files, or the D-140 `D_c` entries
  in `docs/decisions.md` and the problem log.

## 13. Operational

Env `GraduationProject`. Long runs stream per the live-output convention. Checkpoint for reference
measurements: `server-results/deep-SI-checkpoints/FitSys_ClosedLoop_Go1qTA_best.pth`, loaded
state-only into a freshly built system after `CLF.attach(...)`, per `cl_plot_step6.py`. Reference
run metadata: `server-results/step6_result_76573.json` (`lr = 1e-7`, 12 epochs, val 2.187e-06 ->
1.393e-06). Data normalisation for any new script: `fs.norm` (deepSI's), not the `Norm` dataclass,
see section 6. Runtimes measured this session: gradient test 21 s, equivalence test 486 s,
validation-precision test 1642 s. The 4 kHz controller export needs MATLAB.

## 14. Delegation

None for the next action. Step 0 is a targeted profile and steps 1 to 8 are specified file by file
in the plan; the context-holding session is better placed than a subagent. If a wide search becomes
necessary (for example finding every caller of `zero_the_ann` or `identify_output_map` across
`scripts/`), one Explore subagent, medium breadth.
