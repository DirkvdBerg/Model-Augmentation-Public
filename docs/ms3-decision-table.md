# MS3 decision table: what each outcome means and what gets tested next

**Written 2026-07-26, BEFORE arms B and C produced numbers**, so the reading cannot be
fitted afterwards. Companion to the MS3 row in `docs/gantry-augmentation-problem-log.md`
section 12 and to D-126 / D-127 in `docs/decisions.md`.

## 0. The gates (user's terms, 2026-07-26)

Scored on val sim-RMS **against the epoch-0 value**, not against any physical target.

| gate | test | rank |
|---|---|---|
| **G1** | `max_epochs(simRMS) <= ~1.05x` epoch-0 -- training did NOT degrade the init | **PRIMARY** |
| **G2** | `min_epochs(simRMS) < 1.0x` epoch-0 -- some epoch BEAT the init | secondary |

G1 outranks G2. The present failure is that training is **net destructive**
(`8.015e-05 -> 1.916e-03`, 24x, best checkpoint never leaves epoch 0), so an arm that
learns nothing but merely HOLDS the init already passes the thing that is broken.

## 1. The arms

| arm | config | isolates |
|---|---|---|
| **A** | n_seg=1, nf=400, defect OFF | the current pipeline, on clean `pysynth` data |
| **B** | n_seg=3, nf=1200, defect OFF | window LENGTH alone (the move refuted by SLURM 71013) |
| **C** | n_seg=3 x 400, defect ON, `scale = 1/e_roll` (MS1) | the continuity term |

B is the arm that makes C interpretable: without it, any C-vs-A difference could be
attributed to the longer window rather than to the defect.

## 2. The hypothesis being tested

Training is destructive because the windowed loss does not price the low-frequency
component that the free run integrates. The defect prices it, because with the defects
included the objective equals the **full-record** objective regardless of segment length
(Ribeiro et al., Automatica 121:109158, 2020, Thm 2 / Cor 3). Predicted: **A fails G1,
B fails G1, C passes G1.**

Pre-declared limitation: n_seg=3 spans 0.3 s against a 12 s deliverable. `T* = 0.141 s`
(`drift-visual/f09`), so 0.3 s is past drift onset but far short of the deliverable, and
only a PARTIAL effect is expected.

## 3. The branches, pre-committed

| # | observation | conclusion | next test |
|---|---|---|---|
| **1** | A fails G1, B fails G1, **C passes G1** | **Hypothesis confirmed.** The destructive term is objective decoupling, and it is not window length (B controls that) | **Scale `n_seg`: 3 -> 5 -> 10** at fixed `nf_seg=400` and check the G1 margin improves MONOTONICALLY. That is what separates "the term works" from "it helped once". Then, and only then, ask G2 |
| **2** | A fails G1, **B passes G1** | Window length alone fixed it, on clean data. My framing was wrong and 71013's refutation does not survive the D-126 data change | Rerun A and B **on the OLD MATLAB data** with identical settings. If B fails there and passes here, the interaction is with the solver mismatch, not with the horizon |
| **3** | C fails G1 **and `last_defect_rms` FELL** | **Falsifier. Hypothesis dead.** The constraint was genuinely enforced and training still wrecked the init, so the destructive term is something else | Run A on **`pysynth_baseline`** (absorber OFF, model == data exactly, encoder live). If training degrades the init even with NOTHING to learn, the damage is estimator-manufactured and independent of the residual -- which points at the optimiser/encoder interaction, not the objective |
| **4** | C fails G1 **and `last_defect_rms` did NOT fall** | **Inconclusive, not falsified.** A non-squared exact penalty needs a fixed weight above an unknown `rho*` | Re-run C at `defect_weight` 10x then 100x. If the defect still will not fall, the scale from MS1 is wrong, not the idea |
| **5** | **A passes G1** | **The failure does not reproduce on clean data at all.** The destructive training was the ode45-vs-RK4 data mismatch (measured at 2.4x to 3.3x the target signal, D-126), and both the defect work and most of the drift campaign were aimed at a data artefact | Highest-value branch. Rerun A on the **OLD MATLAB data**, identical settings, as the direct control. If it fails there and passes here, D-126 is the finding and the whole objective story is secondary |
| **6** | any arm returns no finite `Loss_val` | infrastructure, not physics | debug before drawing any conclusion; do NOT report a verdict from a NaN arm |

## 3b. WHAT ACTUALLY HAPPENED (2026-07-26, added after the run)

**Branch 5 fired and was then REFUTED by its own control.** Arm A passed both gates on
`pysynth` (G1 1.000, G2 0.595) AND on the original MATLAB data (G1 1.000, G2 0.857). The
failure is absent from both, so it is not the data.

The table above was missing the branch that actually occurred, and it is the one every
future run of this shape must check FIRST:

| # | observation | conclusion | next test |
|---|---|---|---|
| **0** | **A passes G1 on BOTH datasets** | **The run is INVALID: the control does not reproduce the failure.** No treatment arm can be interpreted. Do not run B or C | Make the control FAIL first, then re-run the table |

Diagnosed cause: sizing the run to fit a foreground call cut the optimizer step count
~250x (**~21 Adam steps** here versus **~5250** in production). At 21 steps at `lr = 1e-7`
the iterate is still in the initial descent; the documented degradation is a later-time
phenomenon.

**Rule this adds:** a pre-registered decision table MUST carry an explicit
"control does not reproduce the failure" branch, and the control arm must be run and
checked BEFORE any treatment arm. Cost-driven downsizing changes the step count, and the
step count is the axis the failure lives on.

## 4. Ranking, if more than one branch fires

Branch **5 outranks everything**: if the failure does not reproduce on clean data, no
conclusion about the objective is safe. Branch **3 outranks 1**: a falsifier is worth more
than a confirmation. Branch **4 blocks any verdict on C** -- an under-weighted arm is not
evidence either way.

## 5. Known limits that travel with any MS3 result

* 1 validation record (V1), not 4, so the selector is noisier than production. Both gates
  are ratios against that same record's epoch-0 value, so the comparison is valid, but the
  absolute numbers are not comparable to the 4-record production figures.
* 4 training records, not 14; stride 400, not 10; batch 64, not 256; 3 epochs, not 20.
  This sizes the run to fit a foreground call. It is a DIRECTION test, not a production
  result, and no number from it belongs in the thesis.
* 1 seed. Below the project's 3-seed floor. Any surviving branch needs seeds 1 and 2
  before it is graded above SINGLE.
* `e_roll` (the defect scale) was measured at INITIALISATION, so it is anchored to the
  untrained model and becomes stale as training proceeds.
