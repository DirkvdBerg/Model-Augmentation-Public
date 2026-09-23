# Handoff: build a self-contained real-Telica augmentation setup whose baseline, controller and learnability are proven before any server training
**From**: session of 2026-09-22 | **Branch**: Augmentation | **Effort suggested**: xhigh (controller re-derivation, friction port, and designing a learnability metric are demanding derivation work over many files)

## 0. Standing authorisation for this session (read first)
The user has authorised this session to run **unattended overnight, without asking for
feedback**, inside the scope below. For this session only, this replaces the CLAUDE.md rules
"answer before code", "no new files unless asked", and "check in with user": decide, log the
decision, and continue. Every other CLAUDE.md rule stays in force (no em-dashes, THEORY/HEURISTIC
labels, code-quote verification, live-output run convention, data access policy, `kamtin-fp-model/`
read only).
- Create and edit files **only inside `telica-real/`** at the repo root. Nothing outside it is
  created, edited, moved or deleted. Do not commit; the user commits.
- Decisions go to `telica-real/DECISIONS.md` (numbered `TR-001`, ...), logged BEFORE implementing.
  Runs go to `telica-real/RUNS.md` with the hypothesis written BEFORE launch (the run-discipline
  rule, applied locally). The folder is self-contained, so these replace `docs/decisions.md` and
  the problem-log run table for this work.
- When a choice arises that section 4 does not settle, pick the option that keeps the physical
  model interpretable, log it as a TR entry with the alternative you rejected, and continue.
- Internet access is allowed. Literature searches go through the `deep-research` skill (D-121);
  the open Chrome browser (claude-in-chrome) may be used for pages it cannot fetch.

## 1. Task
Build `telica-real/`, a self-contained folder that takes the project from simulation to the real
Telica gantry, and bring it to the point where a server augmentation run is justified. Work
through the gates in section 9 in order. Each gate ends with a PASS or FAIL verdict and its
numbers in `telica-real/REPORT.md`. On a FAIL, diagnose the mechanism and fix it, at most three
documented attempts per gate. Then either PASS, or stop the night with the FAIL and its
mechanism written down. The end state: one gantry (BHL) with a physical baseline using **Telica
parameters where available and Garcia 2013 as the fallback**, **Karnopp Coulomb friction in the
baseline**, **no hidden MSD/absorber**, the **real Telica feedback controller** re-derived and
implemented in the training form, a **learnability metric** that says whether the augmentation
can learn the missing dynamics from this data, and an **adapted augmentation pipeline** that
passes a local smoke test and is ready for the server. No augmentation training epochs run
locally.

## 2. Out of scope
- **Augmentation training locally.** At most 2 optimizer updates, as a smoke test (G6). Full runs
  crash this PC; that is what the server is for. Do not submit server jobs either: prepare the
  runner script, the user launches it.
- **Editing anything outside `telica-real/`**: `model_augmentation/`, `scripts/`,
  `lpv_lfr_baseline/`, `Matlab-scripts/`, `docs/`, `tasks/`, `CLAUDE.md`. Copy what you need.
- **The right gantry (BHR).** One gantry only, BHL, rows LX1, LX2, LY. Every existing Telica
  module already uses BHL.
- **Orthogonal projection / OBC on real data.** The next step after this setup exists, not part
  of it.
- **Thesis text, slides, meeting material.**
- **Installing packages into the `GraduationProject` env** (including `psutil`). Use what exists;
  the watchdog is PowerShell.

## 3. Where things stand
- Branch `Augmentation`, last commit `a97a577`. The tree is dirty in many directories (the user's
  ongoing work). Copy the **working-tree** version of each file and record in
  `telica-real/vendor/VENDORED.md`: source path, commit `a97a577`, "working tree, dirty", date.
- No run in flight. `telica-real/` does not exist yet.
- **Machine limits (measured 2026-09-22):** 15.8 GB RAM; **C: has 6.3 GB free of 472 GB**, and the
  repo lives in OneDrive on C:. Windows logged "shadow copy storage could not grow" on C: at
  2026-09-22 03:19, so the disk runs full at night. Past overnight sessions ended with every
  window closed; the cause could not be established (the event logs do not reach back to those
  nights), and no system crash or memory-exhaustion event was logged in the window that exists.
  Treat disk space as the first-order risk, RAM second.

## 4. Established and verified
- **Telica data layout, split and signals:** `docs/kamtin-telica-schema.md` (train/val/test
  folders encode the operating point `xpos_*_ypos_*`; `iter0` = feedforward off, an exact
  controller I/O pair; `MF230` = feedback current [A]). Records are 20 kHz native (D-073) and
  short: 8196 samples = 0.41 s (`scripts/gantry/real-data-verification/runs/telica_residual_spectrum.json`).
- **Loader:** `scripts/gantry/real-data-verification/telica_loader.py` (reads `Telica 1.mat` for
  Kt_X 109, Kt_Y 77.6 N/A, fs 20 kHz; D-112).
- **Controller files** (NOT in the blocked folder): `kamtin-data/dFeedbackControllersTelica.mat`
  (6x6 diagonal zpk, Ts 5e-5 s, rows LX1 LX2 LY RX1 RX2 RY, input error [m], output current [A],
  integrators by design) and `kamtin-data/dFeedbackControllersTelica_ba.mat` (bit-exact num/den
  export by `real-data-verification/export_controllers_ba.m`).
- **Controller replay on iter0** (`real-data-verification/telica_controller.py` header, D-073,
  D-074): controller fed the MEASURED error reproduces the logged `MF230` in shape, corr
  0.96-0.97, but the amplitude is off per axis: time-domain LS scale X1 1.16, X2 1.33, Y 3.55;
  in the coherent band 0.74 / 1.08 / 1.23. Not a frame error (Y is P-invariant and worst).
- **Coulomb in the LPV-LFR baseline, smooth form** (D-116, `real-data-verification/coulomb_lfr.py`,
  `lfr_param_block_coulomb.py`, `COULOMB_HANDOFF.md`): `u_eff = u - F_c` at BOTH u-sites,
  `F_c = P (cc .* tanh(P' qdot / v0))`; verified against the direct EOM (3.6e-15) and the MATLAB
  EOM (2.8e-18 m). Initial cc = datasheet static friction 43 / 43 / 49 N (X1 / X2 / Y).
- **Friction law used in the simulation (the one the user wants in the baseline):**
  `Matlab-scripts/Augmentation-coulomb/gantrySystemExtendedCoulomb.m` (set-valued Karnopp,
  stage-frame P projection, Delassus stuck-set solve, active-set breakaway, `V_BRK`),
  `coulomb_v_brk.m`; rationale D-138, D-204, D-209. **Trust the `.m` file and D-209 over
  `Matlab-scripts/Augmentation-coulomb/README.md`**, whose band section (`V_EPS_MARGIN = 9`) and
  gate table are pre-D-209 and stale.
- **Telica parameters** (D-112): datasheet `literature/gantry/telica-xyz-0750-0800-data.pdf`
  gives X-moving 91.0 kg, Y-moving 19 kg (incl. 1.7 kg Z stage), viscous "dynamic friction
  maximal" 2x136 / 98 N/(m/s), static friction 2x43 / 49 N. Not on the datasheet: kb, cb, J, d,
  Lb (Garcia/Kamtin values in `model_augmentation/systems/gantry_ss.py:37-56`).
- **Previous real-data recovery, no friction** (run 70821, `COULOMB_HANDOFF.md`): open-loop NRMSE
  about 50 % (X), 38-65 % (Y); viscous terms inflated 6-7x past the datasheet; closed-loop sim
  diverged (to about 4e5 A) because the real controller sat on the wrong plant.
- **Error spectrum peaks on Telica** (`runs/telica_residual_spectrum.json`, test record
  xpos -135 ypos -120): 586 Hz, 957 Hz, 2393 Hz, 3057 Hz above the floor. The last two sit above
  the 2 kHz Nyquist of the pipeline's default 4 kHz training rate.
- **Simulation pipeline assumptions that break on real data** (read 2026-09-22, entry
  `scripts/gantry/gantry_interconnect_dynamic.py`, package `scripts/gantry/gantry_dynamic/`):
  1. `gantry_dynamic/controller.py` builds a per-record ruleOfThumb controller from Garcia
     parameters and redesigns it at every rate. It is not the Telica controller. Only the
     residual form `u_plant = u_data + Cfb (y_data - y_model)` carries over.
  2. Physical parameters are hard-coded twice: `gantry_ss.py` and `gantry_dynamic/controller.py`.
  3. `data.py::compute_normalization` differences positions for the velocity scale; its own
     comment calls this unusable with measurement noise (dTheta inflated about 190x at SNR 60).
     Its preferred fix: freeze the scales from a calibration record.
  4. `data.py::load_traj` point-samples `y[::D]`, exact only because simulated y has no content
     above 2 kHz. Real y does.
  5. Ground-truth-only diagnostics (`load_mat_aug`, true-x0 baselines in `baselines.py`,
     `_oracle_nrms` and aug-state plots in `evaluation.py`, `state_recovery_diagnostic`) need
     `x_logical` / `delta_a`, which real data lacks; `load_datasets` fails without them.
  6. File lists `TRAIN_FILES` / `VAL_FILES` / `TEST_FILES`, `RECORD_Y_OP` and the `.mat` format
     (`u_total`, `y`) are simulation-specific.
  7. ANN settings (band [140, 230] Hz, `nx_ann = 8`, routing, lr; D-188, D-189) were sized for
     the simulated 212 Hz absorber, which does not exist on Telica.

## 5. Assumed but not verified
- **The force scale.** D-112 multiplies logged current by `_S_FORCE = [3.469, 3.469, 3.202]`
  (HEURISTIC) to reconcile datasheet masses with the data; a factor 2 is documented (a forcer
  pair per logged channel), the remaining ~sqrt(3) is an open question to Kamtin. Mass and force
  scale are not separately identifiable from this data. Present every consistent interpretation;
  settle nothing by assumption.
- **The controller amplitude mismatch** (1.16 / 1.33 / 3.55) is attributed to an unmodelled
  decoupling or gain path. Not established. G3 settles it.
- **The Coulomb recovery run** (problem log row "Telica Coulomb recovery", D-116) was pending
  launch in the log; no outcome is recorded. Do not assume it ran.
- **Whether the Telica controller is purely linear.** The zpk is LTI; the logged path may
  contain saturation, feedforward leakage or a decoupling matrix. G3 checks this on iter0.
- **The large real DC force residual** (-157.5 N X, -83.7 N Y, problem log MS12, "C7") was
  measured against a baseline whose force convention may predate D-112. Re-measure, do not cite.
- **Karnopp `V_BRK` at the pipeline's step size.** D-209 chose it for fixed-step ode4 at 20 kHz;
  it does not transfer to another rate unchanged.

## 6. Tried and failed
- Real-data parameter recovery without friction (70821) -> open-loop NRMSE ~50 %, cg/cy 6-7x
  datasheet, masses -30 % -> the optimizer inflates viscous damping to imitate missing dry
  friction, and masses drift along the degenerate mass/force-scale direction -> `COULOMB_HANDOFF.md`.
- Closed-loop simulation with the real controller on the 70821 plant -> diverged (to about 4e5 A)
  -> real controller on a wrong plant is unstable; closed-loop NRMSE 300-1200 % -> `diag_70821_feedback.py`.
- Fixed Garcia cc (16 N) on real data -> marginal gain on a 564 %-viscous baseline -> cc must be
  recovered jointly, not fixed from a different machine -> `diag_phase3_coulomb_realdata.py`.
- Zero-mean penalty on the ANN output -> ruled out: real friction IS a DC-like force, suppressing
  DC forbids learning it; the simulation cannot validate any DC handling -> problem log MS12.
- Controller with Garcia ruleOfThumb in the loop for real data -> not tried, and do not: it is a
  design rule for a simulated plant, not the machine's controller (section 4, item 1).

## 7. Achieved
- Implemented and validated (simulation / format level): smooth Coulomb LPV-LFR block and its
  MATLAB cross-check (D-116); Telica controller as a steppable filter with scipy self-test
  (`telica_controller.py`); Telica loader with Kt from `Telica 1.mat` (D-112).
- Implemented, validated only in shape: controller replay on iter0 (corr 0.96-0.97, amplitude
  open).
- Not achieved: a baseline that is stable in closed loop with the real controller on real data;
  the Karnopp law in Python; any learnability measure on real data.

## 8. The open question
Is the Telica-parameter baseline with friction good enough, in closed loop with the real
controller, that what remains in the residual is dynamics the augmentation could learn rather
than a wrong force scale, a wrong controller path, or noise? Candidate answers: (a) yes, residual
well above the noise floor and structured; (b) no, the residual is dominated by a scale or
controller-path error that belongs in the baseline, not the ANN; (c) no, the residual is at the
noise floor and there is nothing to learn. G4 and G5 decide between them.

## 9. Next action: the gates, in order
Gate G0 first. Each gate: pre-register its thresholds as a TR entry BEFORE computing, then
compute, then write the verdict to `REPORT.md`.

**G0 Infrastructure.** Create the folder:
```
telica-real/
  README.md          structure + how to run each gate
  DECISIONS.md  RUNS.md  REPORT.md
  vendor/            copies: model_augmentation/, gantry_dynamic/, the lpv_lfr_baseline parts
                     that are imported, the real-data-verification modules used; VENDORED.md
  tools/watchdog.ps1 launch wrapper (below)
  params/  friction/  controller/  baseline/  learnability/  pipeline/
  outputs/           run artefacts only
```
Imports resolve inside `telica-real/` only (one `sys.path` root at `telica-real/vendor`). Data is
read from its original location through the loader, never copied.
`tools/watchdog.ps1`: launches a command, samples every 5 s the process tree's working set, CPU,
system available RAM and C: free space into `outputs/<run>/resources.csv`, and **kills the tree**
when available RAM < 2.5 GB or C: free < 2.0 GB, printing a line starting `WATCHDOG KILL`. It
prints `WATCHDOG ALERT` at 4 GB RAM / 3 GB disk. Every script in every gate runs through it,
with the live-output convention, `torch.set_num_threads(4)`, and `OMP_NUM_THREADS=4`. Watch the
job with the Monitor tool on `WATCHDOG` lines. Before each run, check C: free space; if under
3 GB, delete this folder's own `outputs/` artefacts that no later gate needs, never anything
else. Keep at most one checkpoint per run; figures as PNG only.
PASS: a 30 s dummy load under the watchdog produces the CSV and a forced low threshold kills it.

**G1 Parameters.** `params/telica_params.py` plus `params/PARAMS.md`: one row per parameter,
value, unit, source (datasheet page / `Telica 1.mat` / Garcia 2013 section / D-112 heuristic),
and whether it is identifiable (only kb1+kb2, cb1+cb2, Jb+Jh as sums). Rule: Telica where
available, else Garcia. The force scale is a separate, labelled row with both interpretations.
PASS: every parameter used by the baseline, the friction law and the controller has a row, and
both old hard-coded sets (`gantry_ss.py`, `gantry_dynamic/controller.py`) are replaced in the
vendored copies by imports from `params/`.

**G2 Friction.** Port `gantrySystemExtendedCoulomb.m` to batched torch in `friction/`, as a
stage-frame force at the u-sites of the baseline (the D-116 placement), no hidden MSD. Keep the
D-116 tanh form as the differentiable surrogate for anything that needs gradients through cc,
and record which form each later gate uses. `V_BRK` re-derived for the step size actually used,
with the D-209 reasoning. PASS: cc = 0 reproduces the frictionless baseline bit-for-bit; the
Karnopp port matches a MATLAB reference of `gantrySystemExtendedCoulomb.m` on a short
trajectory (if MATLAB is callable from this session; otherwise against the direct-EOM check
style of D-116), within a pre-registered tolerance.

**G3 Controller.** Re-derive the Telica controller path for the training form, starting from the
zpk file, not from the ruleOfThumb. Settle: sign and units (error [m] -> current [A] -> force via
Kt and the force scale); the rate (keep 20 kHz, or discretise another way, justified by the 2.4 /
3.1 kHz peaks); whether the logged path is linear (iter0 check for saturation, decoupling matrix,
feedforward leakage); and the 1.16 / 1.33 / 3.55 amplitude mismatch, with its mechanism. Then
implement it as the vendored pipeline's `ControllerBank` in the residual form. PASS: (i) replay
on all iter0 records reproduces `MF230` with per-axis gain inside a pre-registered, noise-derived
interval, or the mismatch has a named mechanism that is modelled; (ii) with `y_model = y_data`
the residual-form loop returns `u_data` exactly.

**G4 Baseline adequacy.** Telica parameters + friction + controller, closed-loop replay on every
train, val and test record. Report servo-error sim vs measured per axis, NRMSE, residual PSD
against the noise floor (standstill / high-frequency error floor from the data), recovered
parameters (joint recovery of the identifiable combinations and cc is allowed here; it is
baseline identification, not augmentation training) and their distance to the datasheet, and
the frictionless and 70821 variants as references. PASS: stable on all records; better than
70821 on held-out records; parameters within physical bounds; thresholds pre-registered and
data-derived.

**G5 Learnability.** Design, pre-register and compute a metric that predicts whether the
augmentation can learn the remaining dynamics from this data without training epochs. You decide
the metric; a literature pass through `deep-research` is warranted. Starting points: residual
predictability (share of the baseline's closed-loop residual that a cheap linear dynamic model
fitted on train explains on val, against the noise floor), residual coherence with the inputs per
band, and the gradient signal-to-noise of the augmentation parameters at initialisation over a
few batches (forward and backward only, no optimizer step). The sibling tools are
`scripts/gantry/closed-loop-controller/cl_headroom.py`, `cl_residual_spectrum.py` and
`real-data-verification/telica_residual_spectrum.py`. Output: a verdict "server run warranted:
yes / no" with the numbers, and the ANN settings (nx, band, routing) derived from the residual
rather than from the simulation.

**G6 Pipeline ready.** Adapt the vendored augmentation pipeline for real data: all seven items of
section 4's list, the frozen normalisation from a calibration record, anti-alias filtering before
decimation (or no decimation, per G3), output-only evaluation. Smoke test: at most 2 optimizer
updates, batch 8, one short window, under the watchdog; record peak RAM. Write
`pipeline/runners/server_run.sh` (not submitted) with a memory estimate for the server batch.
PASS: the smoke test runs end to end and its peak RAM is recorded.

## 10. Acceptance criterion
Done when `telica-real/REPORT.md` has a verdict for every gate G0-G6, each PASS with its numbers,
thresholds pre-registered in `DECISIONS.md` and derived from data (noise floor), never from the
model. The alternative ending is also done: a gate marked FAIL after three documented fix attempts,
with the failure mechanism stated and the next step for the user named. `REPORT.md` opens with a
morning summary in this form:

```
G3 Controller: PASS. Replay gain LX1 0.98 / LX2 1.02 / LY 1.04 (interval 0.93-1.07, TR-011).
  Y mismatch was the 2x forcer pair counted twice in Kt (TR-012).
G4 Baseline: FAIL after 3 attempts. Closed loop unstable on val xpos_-210_ypos-40 at 957 Hz.
  Mechanism: ... Next for the user: ...
Resources: peak RAM 6.1 GB (G6 smoke), C: min free 3.4 GB, 0 watchdog kills.
```

## 11. Read these first
1. `scripts/gantry/real-data-verification/COULOMB_HANDOFF.md`: state of real-data friction,
   controller findings, 70821 numbers.
2. `docs/kamtin-telica-schema.md`: signals, split, controller files.
3. `scripts/gantry/real-data-verification/telica_controller.py` and `telica_loader.py`: what
   exists for G3 and data access.
4. `Matlab-scripts/Augmentation-coulomb/gantrySystemExtendedCoulomb.m` with D-209 in
   `docs/decisions.md`: the friction law for G2.
5. `scripts/gantry/gantry_interconnect_dynamic.py` and `scripts/gantry/gantry_dynamic/`: the
   pipeline G6 adapts.

## 12. Do not
- Read, grep, cat or head anything in `kamtin-data/Data Telica/`. Access it only through a Python
  loader run with `conda run`, and print only summaries (shapes, statistics, metrics), never raw
  arrays.
- Touch `kamtin-data/Telica.mat` at all (user rule). `Telica 1.mat` (with a space) is the allowed one.
- Copy raw Telica data into `telica-real/`.
- Use the Garcia ruleOfThumb controller for real data.
- Add a hidden MSD / absorber to the baseline.
- Apply a zero-mean penalty to the ANN output (section 6).
- Run more than 2 optimizer updates of augmentation training, or any Workflow fan-out.
- Run anything without the watchdog.
- Judge the augmentation by latent states; judge it at the output.

## 13. Operational
- Env: `conda run -n GraduationProject`. No `psutil`; resources via PowerShell (`Get-Process`,
  `Get-CimInstance Win32_OperatingSystem`, `Get-PSDrive C`).
- Launch pattern: `powershell -File telica-real/tools/watchdog.ps1 -Run "<conda run --no-capture-output -n GraduationProject python -u <script>>" -Out telica-real/outputs/<run>` with `run_in_background`, `PYTHONIOENCODING=utf-8`, `PYTHONUNBUFFERED=1`.
- GPU: CUDA may exist on this PC; smoke tests stay on CPU unless G6 records why not.
- Expected: G0-G3 a few hours of wall time; no single script should exceed 30 min locally.
- Before starting, the user sets Windows sleep to Never (plugged in) and starts in auto mode.

## 14. Delegation
At most 2 subagents in total (user's session-limit rule): one `deep-research` run for G5, and
optionally one Explore agent if G0's dependency sweep for the vendored copies is wide. No
Workflow tool.
