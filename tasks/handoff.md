# Session Handoff — open blockers only

## >>> AUTONOMOUS SESSION: start at `tasks/auto-session-handoff.md` <<<
For the self-running/auto-mode session (identify problem -> propose -> score against the 5 requirements ->
self-reflect on run outputs -> adjust): read `tasks/auto-session-handoff.md` FIRST. It has the mission, the
hard guardrails, the converged diagnosis, the R1-R5 scorecard, the runnable task queue (TASK 1 = SGD
estimator test, runnable now), the self-reflection protocol, and the STOP/ASK conditions. The block below is
the diagnostic backstory it draws on.

## >>> START HERE: SESSION END STATE (2026-07-24) — read this block, then the 3 files it points to <<<

### The issue in plain terms
We augment the physics model with a neural net (ANN). In long free-run (open-loop) simulation the
augmented model DRIFTS away from truth on the POSITION states (X, Y), which are marginal integrators
(discrete pole ~ 1). That drift gates the deliverable. This session found WHY and where the cure is.

### What is now ESTABLISHED (with evidence)
1. **The model is correct.** Baseline reproduces its own data to ~e-07 with the TRUE initial state
   (per-channel X1/X2 ~e-08, Y ~e-07). The ~e-05 free-run floor is the SUBNET ENCODER-init error
   (horizon-independent), NOT a model bug. Files: `baseline-null/README.md`, `floor_horizon.{npz,png}`.
2. **The drift is an ESTIMATOR/OPTIMIZER artifact, not learning.** In a NULL test (perfect model,
   noiseless data, true init, nothing to learn: windowed loss ~1e-12) training STILL drifts. The
   one-step lr sweep is decisive: one optimizer step displaces the ANN by exactly 3.48*lr and the
   free-run drift is PROPORTIONAL to lr (-> the 3.79e-7 floor at lr=0). So it is step-size DISPLACEMENT
   amplified by the integrator, OPTIMIZER-AGNOSTIC (Adam vs SGD only changes the effective lr). You
   cannot fix it by optimizer choice or lowering lr (can't train at lr->0). Files: `baseline-null/`
   (`run_baseline_null.py`, `lr_sweep.py`, `lr_sweep.png`).
3. **The CURE is STRUCTURAL** — a do-no-harm / DC-output pin / orthogonal-projection constraint that
   stops the ANN injecting on the integrator channel. NOT a contraction/Lipschitz cap (D-118): the
   literature is explicit that strict contraction/REN/Lipschitz PULLS THE MARGINAL POLE STRICTLY INSIDE
   (destroys the genuine pole-1 integrator). Pole-1-preserving cures: conservative port-Hamiltonian
   (R=0), state-AND-output do-no-harm regularization (W-PGNN, Liu-Toth-Schoukens = the supervisors'
   own work), DC/output pins, and the thesis's own orthogonal-projection (Gyorok). Full cited report
   (deep-research 108 agents + a diagnostics agent, verified): `baseline-null/diagnostics-literature.md`.
4. **Problem 1 vs Problem 2 likely UNIFY as ANN DC/displacement on the integrator chain.** Problem 1 =
   velocity-level DC -> LINEAR Y drift. Problem 2 (the "Y destabilization") was called STRUCTURAL /
   exponential (pole>1) by test_efolding, but this session's D4/D8 diagnostics REFINE it toward a
   MARGINAL / QUADRATIC displacement (the ANN FORCING the double-integrator, pole ~ 1 not strictly >1;
   baseline ANN-off is BOUNDED, so no real baseline instability -- the frozen 1.00016 pole was a
   defective-eigenvalue numerical artifact). Files: `ARTBP/growth_aug_vs_base.py` +
   `growth_aug_vs_base_*.{npz,png}`, `ARTBP/pole_horizon_diag.py`.

### OPEN QUESTIONS / NEXT STEPS (what a fresh session should do)
- **A. Confirm Problem 2 is marginal/quadratic vs exponential** (quad-vs-exp was near-tied for 1 of 3
  records, single checkpoint): apply test_efolding's growing-prefix rate/order-stabilization ANN-on vs
  ANN-off, and repeat on a 2nd checkpoint (`data/72659/ckpt_poly6_seed0_ep8_h1600_b256_best.pt`).
- **B. BUILD AND TEST THE CURE (the real deliverable work, not yet started):** a DC/output pin or the
  orthogonal-projection / do-no-harm constraint on the integrator (X,Y) channel, then re-measure the
  free-run drift. This is the payoff; everything above is diagnosis pointing here.
- Optional: does an unbiased-gradient method (ARTBP / multiple shooting) at CONSTANT lr help at all?
  (Open research question -- theory says unbiasedness restores convergence only under DECREASING lr, so
  likely NO -> reinforces "go structural".)

### The 3 files to read (in order) for a fresh session
1. `scripts/gantry/baseline-null/README.md` -- the null experiment + all the drift numbers.
2. `scripts/gantry/baseline-null/diagnostics-literature.md` -- cited cause + cure landscape (which
   cures keep pole=1; D-118 is the wrong knob).
3. `docs/gantry-augmentation-problem-log.md` §12 (the BASELINE-NULL row) -- the run-table record.

### Meta note for the fresh session (lessons this session, in tasks/lessons.md)
Several over-reads were corrected here by TESTING not asserting: "systematic march" (was a transient),
"over-fitting the discretization floor" (windowed loss went UP = displacement), "finite-diff artifact"
(needed an eps-sweep), and "structural exponential Problem 2" (longer baseline-controlled fit says
quadratic/marginal). New lessons: `diagnose-drift-with-lr-sweep-not-optimizer-swap`,
`test-dont-label-a-surprising-number`, `rule-out-bug-with-the-right-reference`,
`null-magnitude-vs-real-effect`, `verify-cross-source-match-claims`,
`set-experiment-flags-explicitly-not-inherited`.

---


## CURRENT ACTIVE THREAD (2026-07-23): Problem 2 (Y destabilization) — is it HORIZON, STRUCTURAL, or LPV SELF-SCHEDULING (exposure bias)?

### RESULT (2026-07-23, both tests RUN on a real checkpoint): STRUCTURAL. Not exposure bias, not horizon.
Checkpoint used: `scripts/gantry/ARTBP/data/ckpt_poly6_seed0_ep10_h3200_b256.pt` (cluster run 72644,
poly6 H_max=3200 b256 ep10; best-trained ep 6, val sim-RMS 2.93e-3; best-ep=0 i.e. never beat the
1.846e-4 init; drift Y full/off 47.9x per its .out `server-output/h3200_72644.out`). Provenance VERIFIED:
ckpt epoch=6, val_sim=0.0029318833 match the .out exactly + file mtime = job finish 12:17 (the "did not
overwrite" copy left the correct file; nothing with that name pre-existed). A second valid poly6 ckpt
exists, `data/72659/ckpt_poly6_seed0_ep8_h1600_b256_best.pt` (H_max=1600, val_sim 1.97e-3, NO .out copied
so its drift is unconfirmed locally) -- h3200 was chosen because its 47.9x Y drift is confirmed.
  - **test_self_scheduling.py: ALL 5 conditions = 1.00x control on all 3 standstills (T1/T3/T5).**
    Teacher-forcing the true Y into physics M(Y) (B1), the ANN input (B2), both (B3), or freezing
    scheduling changes the Y drift by NOTHING -> NOT the LPV self-scheduling loop, NOT the ANN Y-input
    feedback. Verdict key: all~1 -> structural/horizon. Data: `data/self_scheduling_ckpt_poly6_...h3200_b256.npz`.
  - **test_efolding.py: the Y-error is a slow UNFORCED GROWING OSCILLATION** (mode ~0.085 Hz, period ~12s),
    not a monotone drift. RE-RUN at the FULL 12s record (TEF_HSWEEP=12000,24000,36000,47000) gives a clean
    verdict on all 3 records: EXPONENTIAL fits better (R2 0.56-0.81), pole/step ~1.00006-1.00009 > 1,
    tau ~11.5-16.7k steps = 2.9-4.2s = 29-42x the 0.1s/400-step window, AND `r stabilises: YES` (r converges
    once horizon > ~9s = ~1 period; the 8s run had NOT converged -> earlier "sign-flipping r" was just <1
    period). Cleanest discriminator = env-ratio climbs monotonically (T3 5->37->70->100x) and NEVER saturates,
    whereas the ZERO-INIT baseline SATURATES at ~11x -> the trained ANN ADDED the instability.
  - **u-check (standstill non-exciting, `scratchpad/ucheck.py` logic):** u is a multisine at 130-180 Hz,
    ac-rms ~36-45 N, with **0.00% power below 1 Hz** and 0.000% at the ~0.09 Hz Y mode -> the growing sub-0.2Hz
    Y oscillation is UNFORCED -> on an unexcited standstill a GROWING amplitude requires |lambda|>1 (structural),
    independent of the curve fit. (Also: common u-response cancels in err = yhat - ytrue, so growth = model's
    own transient growing.) CAVEAT: mode period ~12s ~ record length -> only ~1 e-fold observed, so tau is
    order-of-period (seconds), not a precise number; a longer free-run would be needed to pin tau/|lambda|.
  - **PROOF FIGURES built** (`make_problem2_figures.py`, from saved npz + u): 
    `figures/problem2_fig1_self_scheduling_...png` (top: 4 conditions coincide; bottom: |cond-control| ratio
    3-7e-4 below the drift -> rules out c) and `figures/problem2_fig2_structural_...png` (A raw growing
    oscillation to -2/-3mm; B u-PSD all >1Hz, 0% at the mode -> unforced; C trained envelope grows to 1-2mm
    vs baseline saturates at 0.1-0.3mm -> confirms b, rules out a). npz: `data/{efolding,self_scheduling}_...h3200_b256.npz`.
  - **COMBINED VERDICT: Problem 2 is STRUCTURAL** -- an intrinsic, UNFORCED growing oscillatory Y mode
    (|lambda|>1, ~0.085 Hz) in the learned dynamics, independent of the Y-scheduling path, timescale seconds
    (29-42x the training window). Cure = the D-117 passivity/stability-constraint route, NOT horizon /
    multiple shooting / state-consistency reg. NEXT: (1) confirm on a 2nd checkpoint (72659 h1600); (2) a
    longer free-run (repeat/extend u) to pin tau/|lambda| beyond ~1 e-fold; (3) log a D-number for this result.

**The question.** ARTBP removes Problem 1 (the dY DC). Problem 2 — the free-run Y destabilization
that gates the deliverable (val sim-RMS floors ~10x above the 1.84e-4 init baseline; H_max=6400/1.6s
reduced it but was unstable + impractical to train) — is NOT fixed by ARTBP. Three candidate causes:
(a) training HORIZON too short (0.1s/400 steps << the instability timescale); (b) STRUCTURAL —
the ANN learned a wrong-sign/anti-damping term (pole strictly >1), needs a stability constraint;
(c) LPV SELF-SCHEDULING exposure bias — the model schedules M(Y) on its OWN drifting simulated Y
(a self-reinforcing loop), fixable WITHOUT a huge horizon. This session built the decisive test for (c)
and the literature basis for all three.

**BUILT + plumbing-VERIFIED (do NOT rebuild): `scripts/gantry/ARTBP/test_self_scheduling.py`.**
Teacher-forces the scheduling variable in a free-run of a TRAINED checkpoint, 5 conditions differing
only in which Y-path sees the TRUE (data) Y vs the simulated one:
  - control (deployed self-scheduling) / B1_sched_true (physics M(Y) <- true Y) /
    B2_ann_true (ANN's Y input <- true Y) / B3_both_true / frozen_sched (M(Y) at a constant, standstill cross-check).
  - Reports per-axis free-run RMS + Y last-quarter RMS ratio vs control. VERDICT read:
    B1/B3 small -> self-scheduling loop (exposure bias); B2 small -> ANN Y-feedback; all ~1 -> structural/horizon.
  - KEY implementation subtlety: the physics block BOTH schedules on and integrates Y (blocks.py
    Gantry_State_Block.deriv, LPV Y_op=None, `Y = x2[:,2]`). A plain input override would teacher-force
    the Y STATE, defeating the free-run. So B1 uses a MIRROR of deriv where only the M(Y) rational + Delta
    read the injected Y while the integrating x2 stays simulated (Y position still drifts). B2 is a simple
    `ann.forward` z[:,2] override (ANN only reads Y). Both via runtime patches (interconnect calls
    block.forward/deriv directly). A faithfulness self-check asserts mirror-deriv == blocks.py deriv on the
    no-override path (PASS). Smoke-tested on zero-init: all 5 conditions coincide at 1.00x (correct, no drift).
  - Y position = state index 2 (NOT dY=idx5, which was the DC study). Scheduling reads Y position.

**RUN IT (needs a real checkpoint — none exists yet):**
  `TSS_CKPT=<path/ckpt_poly6_..._best.pt> PYTHONIOENCODING=utf-8 conda run --no-capture-output -n GraduationProject python -u scripts/gantry/ARTBP/test_self_scheduling.py`
  Env: TSS_RECORDS (default T1/T3/T5 standstills), TSS_H (default 8000 = 2s free-run).
  CHECKPOINT SOURCE (user chose 2026-07-23): the CLUSTER demo. `runners/run_demo.sh` (poly6, saves
  `ckpt_poly6_..._best.pt` with {ann, encoder} state_dicts) -> copy the .pt back -> point TSS_CKPT at it.
  (Local training was declined.) The checkpoint must have the Y-drift, i.e. a converged-ish poly6 run.

**DEEP-RESEARCH DONE (2026-07-23), full result at:**
`C:\Users\20203253\AppData\Local\Temp\claude\...\237df9f7-...\tasks\w2cw1y5b8.output` (JSON; 109 agents,
22/25 claims confirmed). It INDEPENDENTLY prescribes this test (its open-Q #2: "a controlled ablation
scheduling on TRUE vs SIMULATED Y separates exposure-bias from a structural sign error"). Key citable findings:
  - Horizon-vs-structural TEST = state-transition contractivity constant L_h vs 1 (Ribeiro et al.,
    Automatica 121:109158, 2020): non-contractive (L_h>=1, pole~1) -> sim-error loss smoothness GROWS with
    horizon N (poly at L_h=1, exp for L_h>1) -> why 6400-step training was unstable. Structural, not tuning.
  - Proper-horizon rule: TBPTT bias ~ rho^K/(1-rho) (Aicher-Foti-Fox, UAI 2019); rho->1 for a near-integrator
    -> bound DIVERGES -> a long horizon is quantitatively the WRONG cure for the Y mode.
  - Cures without a huge horizon: multiple shooting (Ribeiro 2020; Iakovlev ICLR 2023) makes sub-interval a
    design knob; and for the LPV self-scheduling drift specifically, a STATE-CONSISTENCY REGULARIZER on a
    fixed multi-step loss (Sertbas-Kumbasar, arXiv:2510.24757, 2025) — model schedules on its own drifting
    state = exactly ours. (Its Schur-stability-by-construction claim was REFUTED 0-3; only the regularizer holds.)
  - Caveat: exp loss-growth needs pole STRICTLY >1; at exactly 1 it's polynomial -> measure the free-run
    Y-error E-FOLDING TIME vs the 400-step window to pin marginal-vs-structural (research open-Q #1).

**BUILT + plumbing-VERIFIED (2026-07-23): `scripts/gantry/ARTBP/test_efolding.py`** -- the e-folding
companion. Free-runs the deployed self-scheduling control ONCE per standstill at the longest horizon,
then fits the Y-error ENVELOPE (windowed RMS over successive 400-step windows) to competing growth laws
on GROWING PREFIXES (a prefix of the 8s run IS the 2s run -- causal free-run), so it reports whether the
growth rate r / order p STABILISE across the horizon sweep (default 8000/16000/24000/32000 = 2/4/6/8s).
  - VERDICT read: EXPONENTIAL fit wins + pole>1 + a STABILISING r/tau -> STRUCTURAL (needs a stability
    constraint; a longer horizon is the WRONG cure). POLYNOMIAL fit wins + pole~1 -> MARGINAL (multiple
    shooting / state-consistency reg suffices). Reports tau in STEPS vs the 400-step training window.
  - Envelope method = drift-by-position-envelope lesson (a bounded resonator trips a raw slope test).
    exp-vs-poly discriminator = # THEORY Ribeiro et al. Automatica 121:109158 (2020); tau=1/r = e-fold def.
  - SMOKE (zero-init, 213s, exit 0): the baseline is NOT flat -- the K=0 free-integrator on Y (pole
    exactly 1) drifts from encoder-init mismatch. Correctly read as MARGINAL/benign: poly wins (R2 0.84
    vs exp 0.39), pole/step ~1.00003, envelope-ratio SATURATES (~11x plateau by 4s), r FALLS with
    horizon (0.96->0.12, never stabilises = the polynomial signature). This is the BASELINE NULL to beat.
  - RUN IT (same checkpoint dependency as the self-scheduling test -- needs the cluster poly6 ckpt):
    `TEF_CKPT=<path/ckpt_poly6_..._best.pt> PYTHONIOENCODING=utf-8 conda run --no-capture-output -n GraduationProject python -u scripts/gantry/ARTBP/test_efolding.py`
    Env: TEF_RECORDS (default T1/T3/T5), TEF_HSWEEP (default 8000,16000,24000,32000), TEF_WWIN (default 400).
    Outputs: `data/efolding_<stem>.npz` + `figures/efolding_<stem>.png` (log-y envelope + both fits per record).

**Prior context this session (still valid):** ARTBP Phase B DONE + reframed (D-120), Phase D 5-seed grid
DONE (poly6 wins), gate-2 built — details in the section below.

---

## PRIOR THREAD (2026-07-22): ARTBP verification — Phase B DONE + plan REFRAMED (D-120)
**START HERE**: `scripts/gantry/ARTBP/README.md` §0 (the Phase B result + reframe banner; it
supersedes the original §1-§5 plan). Then D-120 in `docs/decisions.md` and the ARTBP Phase B/B0 row
in `docs/gantry-augmentation-problem-log.md` §12.

Phase B result (built + run this session; do NOT re-run to re-derive):
- `ARTBP/ground_truth.py`: the raw DC-direction GRADIENT does NOT converge on the z=1 dY axis —
  variance explodes ~H^3, mean unresolvable (SE ~= mean, |t|<1.3 at every H). So there is **no
  `true_grad(T)` ground truth** and the "~1/nf bias gap" framing is void. Cause: `g = -kappa(H)*c*(H)`,
  kappa explodes (~H^3.8) while c* -> 0.
- `ARTBP/instrument_select.py` (Phase B0): the loss-landscape instrument (`c*(H)`, `kappa(H)`, parabola
  fit + window bootstrap, FAITHFUL ann-route injection = patched `ann.forward` col 5) is validated:
  Test1 kappa rel-SE <0.05% vs raw-grad ~42% (~1000x better); Test2 recovers planted +4e-6/+1e-6/+3e-7
  to 99.8-100.1%; Test3 the loss optimum `c*(H)` is small, convention-dependent (+1.7e-6 ann-route vs
  -2.44e-6 v11 state-row) and -> 0, NEVER the trained -4.5e-6 DC.
- **Mechanism (reframes the drive):** the DC is Adam walking a near-flat, weakly-constrained direction
  (small kappa at nf=400), NOT a loss optimum; kappa(H) growing is the restoring curvature ARTBP
  supplies via rare long rollouts. Reconciles v3b (systematic gradient) with v11/SGD (flat/Adam
  artifact); explains v12 (ARTBP collapses the DC). The v12 intervention result still stands.

Phase D (variance grid) BUILT + cluster-ready (D-090 row logged; user runs on the cluster):
- `ARTBP/train_artbp.py` (fixed/geom/poly4/poly6, per-step DC + dLoss/dbias probe + held-out nf-RMS;
  `--task_idx` for the SLURM array). Runners: `ARTBP/runners/run_phase_d_base.sh` (single job, all 20
  combos ~4 h) and `run_phase_d_grid.sh` (array 0-19). Collect: `runners/collect_phase_d.py`.
  Import fix applied: `train_artbp.py` inserts REPO on sys.path + runners export PYTHONPATH (repo root)
  so `model_augmentation` resolves on the cluster.
- **5-seed cluster grid DONE (2026-07-22): ALL 4 CRITERIA PASS, poly6 (alpha=6) WINS.** fixed DC
  -4.12e-6 (frac<0 1.00, var 2.05e-7); geom -4.03e-7 (var 10.2); poly4 -2.58e-7 (var 4.80, 2.1x<geom);
  poly6 -1.21e-7 (var 2.22, 4.6x<geom, smallest DC). All ARTBP scattered-sign, in B0's c*(1600) band,
  fit +2-2.7%. **CORRECTION vs the 1-seed probe (which showed 24-47x): true variance reduction is ~2-5x**
  -- the 1-seed poly numbers were optimistic quiet draws; 5-seed means higher with sd~=mean (heavy-tailed),
  which is why the 5 seeds were needed. poly6-vs-poly4 is ~1sigma (both beat geom); poly6 preferred.
  Collect: `runners/collect_phase_d.py` (datDir defaults to the absolute repo path -> runs from $HOME).
  CAVEAT: nf-RMS ~1.2e-3 is BASELINE-level (lr=1e-7/1 epoch) so the fit gate is weak -> gate-2 tests the
  CONVERGED (20-epoch production) fit.
- **GATE-2 (converged-fit) BUILT + ready**, `runners/run_gate2.sh` (TA_EPOCHS=20, TA_DRIFT=1,
  fixed+geom+poly6, seed 0; production lr=1e-7 already matches). `train_artbp.py` extended with per-epoch
  nf-RMS trajectory + best-epoch (the D-114 test: does ARTBP move best-epoch off 0?) and a free-run Y/X
  drift eval (full-ANN vs ANN-off tail-RMS on standstills, the v5/v6 metric). Drift-path smoke-tested.
  Tests DC persistence + fit convergence + overshoot tripwire + stability + Y drift at production HPs.

SECOND drift component (Y destabilization, v5/v6/D-117) -- ARTBP feasibility PROBED, `ARTBP/feedback_instrument.py`:
- kappa_g(H) along a Y anti-damping gain grows ~H^3.7 (315 -> 1.85e8 to H=12800), IDENTICAL to the DC's
  kappa(H); the destabilizing response turns on at H~400, WITHIN the ARTBP cap (1600) -> ARTBP should
  suppress the Y anti-damping feedback by the SAME mechanism/cap as the DC (my a-priori "needs seconds-cap"
  was refuted). CAVEAT: synthetic canonical gain on the baseline, NOT the trained ANN's real v5/v6 direction.
- **DECISIVE next test:** ARTBP-train, then measure the long-horizon Y free-run drift (v5/v6 full/off
  tail-RMS ~2 s) vs the fixed control -> settles estimator (ARTBP) vs structural (D-117 passivity) for
  the Y destabilization. Data/fig: `ARTBP/data/b0fb_feedback.npz`, `figures/b0fb_feedback.png`.

Reusable facts: `demo_common.build_pipeline`; state order [X,Theta,Y,dX,dTheta,dY,delta_a,vdelta_a],
dY idx 5, ann_route_ix=(0..7) so ANN output col j -> state j; injection must patch `ann.forward`
(interconnect calls block.forward directly, nn.Module hooks do not fire); K>=2 in any truncated rollout.
Data/figs: `ARTBP/data/{b_bias_gap,b0_instrument_select,b0fb_feedback}.npz`,
`ARTBP/figures/{b_bias_gap,b0_landscapes,b0_instrument,b0fb_feedback}.png`.

Meeting pack (offset investigation, complete): `scripts/gantry/gantry-zero-mean/meeting-2026-07-21/
meeting.html` (E0 dY headline through E8 ARTBP + E6b + framing table). Built by `make_meeting_pack.py`.

---

**Background for the drift / non-zero-mean investigation** (superseded by the above for day-to-day):
`scripts/gantry/gantry-zero-mean/README.md` (2026-07-15) is the self-contained picture: context,
run-71167 checkpoint provenance + the `_best` trap, key measured numbers, glossary, what is
established vs demoted, the Jan meeting notes, and the V1-V6 verification plan.

**Trimmed**: 2026-07-15 (full 2026-07-13 content archived to
`archive/sessions/2026-07-15-handoff-layer2-prebuild.md`; prior sessions in `archive/sessions/`).

## Recently closed
- **G-A (does the physics carry a DC the baseline lacks?) CLOSED on the physics side, 2026-07-17**,
  via `v1f_dc_excitation_openloop.m` (open-loop, same input to both plants, sustained offset + 150 Hz
  tone; see `scripts/gantry/gantry-zero-mean/README.md` §V1f). Both DC mechanisms measured:
  static-gain DC = 0 (M drops out at qddot=0); the delta_a^2 rectification DC = 3.1e-10 rad
  (confirmed by amplitude^2 scaling: `<delta_a^2>=(1.67e-5)^2=2.8e-10`); largest DC anywhere ~1e-7
  (the L0/mass-split static asymmetry). All 5+ orders below the ANN's DC. Verdict: the ANN's DC is
  NOT physics-justified; source is the estimator/training. **Live gates are now G-B and G-C**
  (README §7-8: V2 normalization/init audit, then V3 birth-of-the-DC probe — Jan's core ask).
  Lessons added: `verify-nonlinear-mechanism-fully`, `test-zero-mean-properly` clause (5).

## Open blockers

1. **Supervisor gate (THE decision): is empirical R4 acceptable as the deliverable?** For-all-weights
   no-drift is proven incompatible with full expressivity (`all-five-construction-spec.md` §4); Route B =
   demonstrated, not guaranteed. The drift-visual deck (`scripts/gantry/drift-visual/`, regenerated
   2026-07-15 from run 71167's rescued `_last` checkpoint, D-114/D-115) is the meeting material answering
   Jan's mail (not energy: f08; not zero-mean: f04; amplifier: K=0 axes).
2. **Build Layer 2**: the DIRECT measured joint-DC pin with near-DC frequency selectivity
   (`docs/data-silent-regularization-concept.md` §7 as corrected by d16; safety d12/d14v2, stationarity
   d15, Fisher-SVD refuted d16). Validate per concept §9 (12 s envelope ~1, absorber band untouched,
   injected-friction discriminator — needs a NEW friction-injected MATLAB dataset).
   **New caveat (2026-07-15)**: run 71167's DC has a state-dependent component (drift-visual f05: DC
   removal collapses only 2.6x on Y, not 133x as the old checkpoint) → re-verify d15-style pin
   stationarity on `gantry_drift_71167_last` before the build; iterative re-aiming is the documented
   contingency.
3. **Open question**: what was cluster run 71168 (finished Jul 14 23:37, ~4.6 h fit)? If it was a
   zero-mean-pin run, its `_last` checkpoint feeds `demo7_g9_intervention` as the intervention figure.
4. **f02 unit question**: measured V1/V3 Y trajectory sits ~0.10 m while captions say "Y = +10 mm";
   check `Matlab-scripts/Augmentation/data/` generator (`gtd_build_records`) to settle Yp10's unit.

## Standing constraints (unchanged, enforced)
- Theta-only routing never the deliverable (D-103); fixes act on loss/estimator, not routing.
- Velocity/accel-domain loss = LAST RESORT (supervisor-gated).
- No compute-cost adjectives without a measured basis; ask the user what is runnable.
- Read any run log's printed Configuration block FIRST (deployed copies lag local edits).
- Every gantry training script prints per-epoch `[nf-probe] train/val nf-RMS` (D-102).
- `conda run -n GraduationProject`, `PYTHONIOENCODING=utf-8` every PowerShell call.

## Where everything is documented (do NOT re-derive)
- Diagnosis d1–d16: `docs/drift-diagnosis-status.md` (§3b chain, §10 index).
- Layer-2 concept + limits: `docs/data-silent-regularization-concept.md` + `-limits.md`.
- Deck provenance + figure specs: `scripts/gantry/drift-visual/README.md`, D-114/D-115 in
  `docs/decisions.md`; run table `docs/gantry-augmentation-problem-log.md` §12.
