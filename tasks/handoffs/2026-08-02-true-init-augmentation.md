# Handoff: can the augmentation ANN learn the absorber when the initial condition is exact?
**From**: session of 2026-08-02 | **Branch**: Augmentation | **Effort suggested**: xhigh
**Mode**: autonomous. The user is away and will not answer questions. Read section 15 first.

## 1. Task

Build a new folder `scripts/gantry/true-init-augmentation/` that runs the augmentation training
with every initialisation confound removed, and answer one question: **can the ANN learn the
absorber at all?** Two changes against the current pipeline. First, give the baseline the truth's
full static mass distribution at `delta_a = 0`, so the centre-of-gravity mismatch stops polluting
the X and Theta rows. Second, replace the SUBNET encoder with the truth's **six physical states
only**, using analytically exact velocities rather than finite differences; the model's augmented
rows 6-7 start at zero as they do today, and must **not** be seeded (section 8). Then, in
this order: (i) verify per window that the target the model is asked to fit is correct on **all
six physical states**, not only Y; (ii) train the ANN and report whether it learns. Every training
run to date carried the CoG mismatch, the encoder-init error and the missing augmented dynamics
simultaneously, so no existing run can attribute a failure. This one can.

## 2. Out of scope

- **Any encoder work.** The point of this experiment is that the encoder is gone. Do not
  implement, initialise, or diagnose `W^a` / `W^b`. `scripts/gantry/encoder_initialisation/` and
  `scripts/gantry/encoder-augmentation/` are other threads.
- **Implementing a learnable `A_aug` on the augmented partition.** It is the leading candidate
  (section 8) and this experiment is what decides whether it is needed. Do not build it first.
- **Re-running the literature.** Sections 10 and 12 of
  `scripts/gantry/coulomb-offset/IMPLEMENTATION-LOG.md` hold two sweeps from 2026-08-01/02.
- **Do not modify** `model_augmentation/` (mark any addition per the tracking rule if genuinely
  unavoidable), `kamtin-fp-model/` (read-only), or anything in
  `scripts/gantry/coulomb-offset/` (that thread is closed and its log is the evidence base).
- **Do not regenerate or overwrite any dataset.** New dataset means new folder.

## 3. Where things stand

Branch `Augmentation`, last commit `cf7bef2` "Create multiple shooting inside interconnect
library". Tree is dirty across `scripts/gantry/`, `docs/`, `Matlab-scripts/` and `tasks/`; none of
it is mid-edit and none of it blocks this task. No run is in flight.

New this session, both off by default and both safe to ignore for this task:
`scripts/gantry/gantry_dynamic/rezero_gate.py` (behind `ANN_REZERO_GATE`, refuted, section 6) and
the accumulated-defect term in `model_augmentation/fit_systems/multiple_shooting.py` (behind
`defect_acc_weight`, default `0.0`).

## 4. Established and verified

Evidence pointers are to `scripts/gantry/coulomb-offset/IMPLEMENTATION-LOG.md` (`LOG`) unless
stated. Every number below was measured, not inferred.

- **From the exact initial condition the model is correct.** Free-run residual on Y has mean
  `-1.43e-09` m against a `5.91e-09` m floor, on all record classes. The offset the project has
  been chasing is an initialisation artefact, not a model error. `LOG` F1, section 10.2.
- **The per-window DC is caused by the missing absorber state.** Correlation `-1.000` with
  `vDelta_a(0)`, slope matching `-(ma/mh) . nf/2` to 3 %. `LOG` F3.
- **It is variance, not bias.** Zero-mean across windows, HAC Newey-West `|t| < 1.3`, six
  axis/record combinations, frictionless and Karnopp. `LOG` F4. This is why no mean-penalty helps.
- **Scale of the corruption**: per-window Y scatter `1.045e-04` m under encoder-style seeding
  against `2.979e-08` m for a free run from the exact IC, i.e. `3507x`. `LOG` section 10.2,
  `diag_continuity_limit.py`.
- **The CoG term is real, small, and does NOT remove the offset.** Truth X-Theta coupling is
  `B12 - mh*Y - ma*L0 - ma*delta_a` against the baseline's `B12 - mh*Y`; static difference
  `ma*L0 = 0.1010 kg*m`, Theta inertia difference `ma*(2*Y*L0 + L0^2) = 0.0303 kg*m^2` at
  `Y = 0.10`. The `L0` terms enter X and Theta only, **never Y**. Correcting it measured
  X `-4.1270e-06 -> -4.0378e-06` (2.2 % better) and Y `-3.3142e-05 -> -3.5987e-05` (8.6 % **worse**).
  `LOG` F4 in section 4. **Correct it as confound removal, not as a fix, and do not expect it to
  move the offset.**
- **The current seeding uses finite-difference velocities.** `gantry_interconnect_dynamic.py:154-157`
  seeds `x0_phys=data.val_x_logical[K0]` and labels it "True-x0 (oracle)", but the velocity rows of
  `x_logical` are finite differences. Using the analytic exact IC instead dropped frictionless X
  from `1.06e-06` to `5.37e-10`. This is the "velocity fix" the task refers to.
- **The augmented states cannot persist.** `model.py:132` connects `ann_block` to `xp` on
  `route_ix` and `model.py:135` connects `phy_block` on `PHY_IX = 0..5`; nothing else writes rows
  6-7. With the ANN at zero the propagated `x_aug` at segment end is exactly `0.000000e+00`
  (`verify_ms_gradient.py` gate G6). The augmented partition is overwritten every step.
- **The framework says it should have its own `A`.** Györök et al., `arXiv:2604.11421`, held at
  `literature/augmentation/Data-driven augmentation of first-principles models under
  constraint-free well-posedness and stability guarantees.pdf`, p3 defines `dim x_a > 0` as
  *dynamic augmentation* for "unmodeled dynamic components, such as actuator dynamics and flexible
  modes", and p9 parameterises it contracting: `A = alpha_bar . sigma_A . A_bar`,
  `sigma_A = sigmoid(alpha) in (0,1)`. `LOG` section 12.

## 5. Assumed but not verified

- **That F1's zero-mean result survives per-window re-seeding from the EXACT physical IC.** F1 was
  measured on a continuous free run, and the `2.979e-08` per-window figure comes from chunking that
  free run after the fact (`diag_continuity_limit.py` config B). Nobody has measured per-window
  means where **each window is re-seeded** from the exact 6-state IC. That is the difference
  between config A and config B and it is precisely what this experiment does, so treat the clean
  target as the hypothesis under test, not as an established fact. Task item (i) settles it.
- **That the ANN has the capacity to represent the absorber correction at all.** Never tested in
  isolation. This experiment is the test.
- **That `route_ix` should stay `(0,1,2,3,4,5,6,7)`.** The X/Y routing constraint is a hard project
  constraint (D-103): the ANN must route to X and Y, never Theta-only. Keep it unless a measurement
  forces otherwise, and say so if it does.
- **That the per-window target is clean on states other than Y.** The whole zero-mean result
  (`LOG` F4) was measured on X, Theta and Y positions. The velocity rows were not checked. Task
  item (i) exists because of this.

## 6. Tried and failed

Each of these is closed by measurement. Re-running any of them is wasted time.

- **Per-window zero-mean penalty** -> cannot help -> the per-window DC is already zero-mean
  (HAC `|t| < 1.3`, six combinations), so the penalty prices a quantity whose mean is zero and only
  adds variance -> `LOG` F4, `zeromean_pin.py`.
- **Multiple shooting / continuity defects** -> per-window scatter went to `0.6-0.7x`, i.e. worse
  -> on a `K = 0` axis a node-0 error produces a *growing* displacement, so continuity lets it run
  the whole chain instead of `nf` steps; the `3507x` ceiling required an exact node 0, and the
  exact node 0 was doing all the work -> `diag_wrong_node0.py`, `diag_continuity_limit.py`,
  `LOG` F8. Implementation itself is verified correct (E1-E7 forward, G1-G7 backward).
- **ReZero / Fixup zero-gate over a live branch** -> `6x` worse at step 1, `19x` worse at step 5 on
  `|g W^a|/|g W^b|` -> the plain zero-init final layer already recovers gradient at optimizer step
  1 (`dL/dW_final = <dL/dw, h>` is non-zero even at `W_final = 0`), so the dead zone is a one-step
  transient and the gate merely throttles the branch through one scalar -> `verify_rezero_gate.py`,
  `simulations/gantry_subnet/diagnostics/rezero_gate.json`, `LOG` section 11. **This refutes
  D-130's "structural dead zone" reading; D-130 has not yet been amended.**
- **Defect term on the augmented rows** -> live gradient, degenerate target -> with `x_aug`
  propagating to exactly zero the augmented defect is `enc_aug(node_j) - 0`, whose minimiser is
  `enc_aug = 0` -> `verify_ms_gradient.py` G3 + G6.
- **Hard identity path on rows 6-7 (proposed 2026-08-01, withdrawn same day)** -> wrong object ->
  the absorber is a *damped* oscillator so `|lambda| < 1`; an integrator is not what it needs, and
  the Györök parameterisation guarantees `||A||_2 < 1` which excludes it -> `LOG` section 12.3.

## 7. Achieved

Implemented and validated:
- Karnopp stick-slip truth model and matched Python baseline; truth replay floor `8.75e-09` m
  (was `2.29e-06` with hard `sign`). `Matlab-scripts/Augmentation-coulomb/`,
  `scripts/gantry/coulomb-offset/plant_coulomb.py`.
- Full diagnosis of the offset as an initialisation phenomenon, with the mechanism, the scale and
  the correlation. `LOG` sections 4 and 10.
- Multiple shooting verified correct forward and backward, then shown not to help here.

Implemented, not validated: `rezero_gate.py` (refuted), `defect_acc_weight` (never trained with).

## 8. The open question

**If the ANN fails to learn from an exact physical initial condition, is it persistence or
capacity?**

Settled before this handoff, so do not reopen it: **seed the six physical states only.** Seeding
the model's rows 6-7 with the truth's `[delta_a, vDelta_a]` was considered and rejected for two
independent reasons. Rows 6-7 are the model's own latent coordinates and the ANN is free to choose
any representation for them, so equating them with physical absorber coordinates assumes an
alignment nothing enforces. And at initialisation the ANN is zero, so G6 applies immediately:
anything seeded into rows 6-7 is overwritten to `0.000000e+00` at step 1 and the seed does nothing
beyond sample 0. The arm is ill-posed, not merely weak.

That leaves one genuine question, to be answered by the results rather than in advance. With an
exact physical IC the residual is a zero-mean oscillation (F1) and the target should be clean, so
if the ANN still fails to learn, the cause is one of:

| Candidate | Discriminating evidence |
|-|-|
| **Persistence.** Rows 6-7 are rebuilt from scratch each step (G6), so the model cannot represent a second-order oscillator no matter how good the init. | Failure with a clean per-window target from item (i). Fix is the learnable contracting `A_aug` of section 4's last bullet. |
| **Coordinate pinning.** Rows 6-7 carry no meaning because nothing ties them to anything. | An auxiliary term supervising rows 6-7 against the truth's `[delta_a, vDelta_a]` changes the outcome. This is the well-posed version of the rejected seed-8 idea, and it is sub-question (c) of the 2026-08-02 sweep, which returned four whole-of-arXiv zeros and is probably a vocabulary miss rather than a gap. |
| **Capacity.** The ANN simply cannot represent the correction. | Both of the above implemented and it still fails. |

Do not implement either fix in this task. Report which candidate the results implicate.

## 9. Next action

Create `scripts/gantry/true-init-augmentation/` and write the **per-window target check** before
any training: re-seed each window from the exact 6-state physical truth IC at a grid of window
starts, roll the baseline
(with corrected CoG) forward `nf = 400` samples, and report per-window mean and scatter for all six
physical states against the truth. This is task item (i), it needs no training, and it is what
tells you the target is trustworthy before you spend a run on it. Reuse the harness in
`scripts/gantry/coulomb-offset/diag_wrong_node0.py` (window chunking, per-window means) and the
plant in `plant_coulomb.py`; read, do not import across, that closed folder.

## 10. Acceptance criterion

Per-window DC scatter on **every** physical state at or below the free-run-from-exact-IC level,
which is the model-plus-discretisation floor for this setup: Y `2.979e-08` m, X `9.147e-08` m,
Theta `3.730e-09` rad (measured, `diag_continuity_limit.py` config B). Against the encoder-seeded
`1.045e-04` m on Y that is the `3507x` gap this experiment is meant to close by construction.
Velocity-row thresholds are not yet measured; derive them the same way, from a free run at the
exact IC, and state them before using them.

If any state's scatter lands materially above its free-run figure, do **not** train on it and do
**not** stop: a corrupted target is itself a result. Establish whether the cause is a defect in the
new code (fix it and re-run) or a genuine property of per-window re-seeding (characterise it, since
that would falsify the section 5 assumption and is the most valuable thing this task could
produce). Only then decide whether the training arm is worth running, and record the decision.

For the training arm, the criterion is comparative, not absolute: does val free-run sim-RMS with
the ANN on beat the ANN-off baseline on the same seeds. Per the run-discipline rule this needs a
row in `docs/gantry-augmentation-problem-log.md` section 12 stating the hypothesis before launch.

## 11. Read these first

1. `scripts/gantry/coulomb-offset/IMPLEMENTATION-LOG.md` sections 4, 10, 11, 12: the whole
   evidence base for sections 4 and 6 above; section 12 has the `A_aug` finding.
2. `scripts/gantry/gantry_interconnect_dynamic.py`: the entry point this folder mirrors; lines
   154-165 are the true-x0 versus encoder-init baselines you are replacing.
3. `scripts/gantry/gantry_dynamic/model.py:96-140`: the interconnection, where CoG lives and
   where rows 6-7 are (not) written.
4. `scripts/gantry/coulomb-offset/diag_wrong_node0.py`: the per-window harness to reuse.
5. `docs/decisions.md` D-103 (X/Y routing constraint), D-117/D-118 (passivity and Lipschitz routes
   already explored), D-130 (the dead-zone claim now refuted by section 6).

## 12. Do not

- Do not retry any item in section 6.
- Do not write to `kamtin-fp-model/`, `kamtin-data/Data Telica/` (blocked in `.claudeignore`), or
  `scripts/gantry/coulomb-offset/`.
- Do not enable `ANN_REZERO_GATE` or `defect_acc_weight` for this task.
- Do not treat the CoG correction as a fix for the offset; F4 measured that it is not.
- Do not seed from `data.val_x_logical[K0]` and call it exact; its velocity rows are finite
  differences (section 4).
- Do not seed the model's augmented rows 6-7 with the truth's `[delta_a, vDelta_a]`; the reasons
  are in section 8 and the decision is settled.
- Do not amend D-130. Nobody is available to approve it. Draft the proposed amendment in the new
  folder's log instead and leave `docs/decisions.md` untouched on that entry.

## 13. Operational

Conda env `GraduationProject`. Per the live-output rule, launch anything longer than a few seconds
as:

```
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output -n GraduationProject \
    python -u scripts/gantry/true-init-augmentation/<script>.py
```

in the background, and tell the user the `.output` path. `conda run python -c` cannot take a
multi-line argument; write to the session scratchpad and run the file. Diagnostic JSON convention
in this project is `simulations/gantry_subnet/diagnostics/<name>.json`; figures go in the new
folder's own `figures/`. The Karnopp dataset lives under `augmentation_coulomb_karnopp/`; the
hard-`sign` dataset it replaced is stale and must not be used.

The per-window check runs in minutes on cached traces. A training arm at `nf = 400`, stride 100 is
hours; get the check green first.

## 14. Delegation

None for the main line: every file needed is named above and the work is targeted. Ceiling of one
Explore subagent, and only if a genuine hunt opens up across the MATLAB data-generation scripts.
Running autonomously is not a reason to delegate more; it is a reason to delegate less, because a
subagent's findings arrive without the context that would let you judge them and there is nobody
to arbitrate.

## 15. Autonomous operation

**The user is away and will not answer anything. Do not ask questions, do not wait for approval,
and do not stop early to check in.** This section is the standing authorisation for that, and it
overrides the "check in with user" step of the project task-flow rule and any instinct to confirm
direction. The rules it does **not** override: the read-only paths in section 12, the
no-new-datasets rule, the run-discipline rule (a hypothesis row before every training launch), and
the code-quote verification rule.

**First action.** Write the task list to `tasks/todo.md` before touching code, then execute it
without pausing for confirmation. Keep it updated as you go; it is the record of what you did.

**Decision rules, so that nothing blocks.** Where this handoff leaves a choice, take the default
below, record the choice and its reason in the log, and continue. Never idle on an unanswered
question.

| Situation | Do this |
|-|-|
| A section 5 assumption turns out false | That is a finding, not a blocker. Record it, re-plan around it, continue. Falsifying the clean-target assumption is a better outcome than a green run. |
| Per-window check fails | Section 10, second paragraph. Diagnose, do not stop. |
| A design choice is genuinely 50/50 | Pick the one that is cheaper to reverse, state the alternative in the log, continue. |
| Something looks like it needs a decision only the user can make | It almost certainly does not. If it truly does (scope change, deleting data, anything outward-facing), skip that branch, do everything else, and put it in the report. |
| A run fails or a result is ambiguous | Two diagnostic attempts, then move to the next task-list item and record the state. Do not loop. |
| You finish everything | Use the backlog below. Do not invent scope. |

**Backlog, in order, if the task completes with time left.** Stop at the end of the task list if
none applies.
1. The 12 s free-run arm, if only the windowed metric was covered.
2. The second record class, for error bars on item (i).
3. Draft (do not apply) the D-130 amendment, per section 12.
4. Nothing else. Do **not** start `A_aug`, coordinate-pinning supervision, or any section 2 item.

**Reporting.** Keep a running `scripts/gantry/true-init-augmentation/IMPLEMENTATION-LOG.md`,
modelled on `scripts/gantry/coulomb-offset/IMPLEMENTATION-LOG.md`, written as you work rather than
at the end. Open it with a section titled `## 0. Read this first` holding, in this order: what was
run, the numbers against the section 10 criterion, which section 8 candidate the results implicate,
what was assumed, and what is still open. Report everything you found, including partial,
inconclusive and negative results; do not filter to what seems important. A negative result with a
mechanism is the most valuable thing this task can return.

Example of the wanted tone for that section, showing a failed check reported plainly:

> Per-window re-seeding from the exact 6-state IC gives Y scatter `X.XXe-XX` m against the
> `2.979e-08` m free-run floor, a factor `N`. This falsifies the section 5 assumption: the clean
> target does not survive per-window re-seeding. Mechanism: <...>. Consequence: the training arm
> was not run, because <...>.

**Commits.** Commit working increments on the `Augmentation` branch as you go, so nothing is lost.
Do not push, do not merge, do not open a PR.
