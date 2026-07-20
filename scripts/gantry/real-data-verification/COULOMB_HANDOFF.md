# Coulomb Friction — Session Handoff (2026-07-16/17)

Status snapshot so a new session can continue. Full plan:
`Matlab-scripts/coulomb-friction/PLAN.md`. Decision: `docs/decisions.md` **D-116**.
Run-table row: `docs/gantry-augmentation-problem-log.md` §12 (D-090).

## Goal (one paragraph)

Real-data parameter recovery on the Telica gantry (run **70821**, LPV-LFR baseline,
no Coulomb) plateaus at ~50% open-loop NRMSE and cannot train it out. Diagnostic:
the viscous friction coefficients recover to **6-7x their datasheet maximum**
(cg1 136->841, cy 98->664), i.e. the optimizer inflates viscous damping to fake a
**Coulomb (dry) friction** the model lacks. Fix: add trainable Coulomb friction to
the LPV-LFR baseline, recover cc1/cc2/ccy from the Telica data, and check that it
lowers open-loop NRMSE and lets cg/cy relax toward the datasheet.

## Hard rules honored (do not break)

- **All Coulomb code lives in `scripts/gantry/real-data-verification/`.** It OVERRIDES
  the baseline via monkey-patches. **`lpv_lfr_baseline/` core is NEVER edited.**
- Coulomb is added in **LPV-LFR format**: `u_eff = u - F_c` at the force node, where
  `F_c = P (cc .* tanh(P' qdot / v0))`. `u_eff` replaces `u` at **BOTH** places `u`
  is used in the forward (`fnet` AND the `[x,w,u]@G` concat) — miss the 2nd site and
  friction vanishes at Y=0. **G, K, C, and the M(Y)^-1 loop are unchanged.**

## What is DONE and VERIFIED

| File (real-data-verification/) | What | Verification |
|---|---|---|
| `coulomb_lfr.py` | Coulomb LPV-LFR forward + RK4 + `simulate_coulomb`; `direct_xdot` reference; `sign`/`tanh` switch; **compiled** like baseline rk4 | `lfr`==`direct` incl **Y=0** (3.6e-15); `cc=0`==baseline bit-for-bit (0.0); tanh equivalence (Check 1b); fnet-only trap shown |
| `lfr_param_block_coulomb.py` | `ParameterizedLFRBlockCoulomb` (baseline block + trainable `log_cc`); `cc_init=(43,43,49)`, `v0=1e-3`; `coulomb_cc()`, `cc_table()` | gradient check: grad flows to `log_cc` AND `log_params` |
| `run_telica_param_recovery.py` | **`USE_COULOMB` switch** (top, currently `True`). Patches `tr.ParameterizedLFRBlock`, `tr._build_sim_params` (stashes gradient-connected cc), `tr.simulate`. Windowed-val + open/closed-loop eval all use Coulomb. Best-epoch cc restore. Per-run output folder. Saves `coulomb_cc_<run_id>.pt` + prints `cc_table()`. | smoke test: friction active, grad->cc + base params, `SMOKE: ALL PASS` |
| `verify_vs_matlab.py` | Cross-check the exact LPV-LFR Cramer sim vs the supervisor's **direct MATLAB EOM** (`gantrySystem`+Coulomb, `M\`) | **max diff 2.8e-18 m** over 6000-step RK4 with active friction |

MATLAB references in `Matlab-scripts/coulomb-friction/`: `gantrySystemCoulomb.m`
(base-agnostic `base(u-Fc,x)` wrapper), `export_matlab_coulomb_ref.m` (writes
`matlab_coulomb_ref.mat` for `verify_vs_matlab.py`), `run_coulomb_validation.m`,
`make_coulomb_model.m` (Simscape helper — see NOTE below), `PLAN.md`.

## Key parameter/design choices

- **cc init = Telica datasheet static friction (maximal): X 2x43 N, Y 49 N** ->
  `CC_INIT=(43.0, 43.0, 49.0)` (literature/gantry/telica-xyz-0750-0800-data.pdf; the
  same datasheet's dynamic-friction row 2x136/98 is the viscous cg/cy init, D-112).
  Supersedes Garcia 2013's 16.8/18.35/11.6 (different gantry). It is STATIC/breakaway
  (>= kinetic) and maximal, so an **upper-bound init** for the trainable kinetic cc.
- **`v0=1e-3` m/s** tanh transition width (HEURISTIC). Hard `sign` only for the MATLAB
  cross-check; `tanh` for training (Makkar & Dixon 2005, differentiable BPTT).
- **Per-run output folder**: `simulations/pr_telica_split/<run_id>/` (`run_id` =
  `$SLURM_JOB_ID` or a local timestamp). NOTE: the precompute cache lives there too,
  so it **rebuilds once per run** (cache is run-independent; to share, point precompute
  at the parent — not done, unmeasured cost).

## HOW TO RUN THE RECOVERY

`USE_COULOMB=True` already. Launch `run_telica_param_recovery.py` on the cluster as
before (prints `[run_telica] COULOMB FRICTION ON ...` at startup). Outputs land in
`simulations/pr_telica_split/<SLURM_JOB_ID>/`: the trajectory overlays (measured vs
model), residuals, feedback-current plots, `eval_data_*.pt`, checkpoints, main `.pt`,
and `coulomb_cc_<run_id>.pt`. Set `USE_COULOMB=False` to reproduce the 70821 baseline.

## PENDING / NEXT

1. **Launch the recovery run** and fill in the D-090 outcome. Compare vs 70821:
   - open-loop NRMSE (init-state free-run) on held-out val/test LOWER than 70821?
   - viscous cg1/cg2/cy RELAX toward datasheet (undo the 6-7x)?
   - recovered cc positive, order tens of N (init 43/43/49)?
2. **Ablation** (Phase 4): hard `sign` vs `tanh` at 2 `v0` values — does smoothing
   matter, and pick `v0` from data.
3. **Phase 3 result (context):** fixed-Garcia-cc open-loop check was inconclusive
   (Coulomb helped only marginally at 16 N on a 564% datasheet-viscous baseline) —
   deprioritized because cc must be jointly recovered. `diag_phase3_coulomb_realdata.py`.

## CONTROLLER SIDE-INVESTIGATION (orthogonal to Coulomb; do not block on it)

70821's **closed-loop** NRMSE was WORSE than open-loop (300-1200%). Findings
(`diag_70821_feedback.py`, `telica_controller.py` self-test):
- CL sim **diverges** (orange -> ±400,000 A) = **plant-model instability** (real-tuned
  controller on the known-wrong 6x-viscous plant). This is why CL >> OL.
- Controller input-output check (green=controller(measured error) vs black=logged
  current, **no plant model**): dynamics correct (**corr ~0.97**) but per-axis
  amplitude off: **LS scale (logged/replayed) X1 1.16, X2 1.33, Y 3.55** (documented
  in `telica_controller.py` docstring, D-073). Controller coeffs come from
  `dFeedbackControllersTelica_ba.mat` (NOT Telica 1.mat; Telica 1.mat only gives Kt).
- This gap is a **controller-path** issue (per-axis gain / unmodeled decoupling), NOT
  the plant and NOT Telica 1.mat values. It is **NOT** a logical-vs-stage frame error:
  Y is invariant under the P transform yet has the biggest error (3.55x), and corr
  stays ~0.97 (no X1<->X2 cross-coupling). So the cause is per-axis **gain/units**,
  worst on Y. **Open, optional test:** apply the controller in the logical frame and
  see if LS scales move to 1 (prediction: Y stays 3.55 -> confirms gain, not frame).

## GOTCHAS / CORRECTIONS MADE THIS SESSION (see tasks/lessons.md)

- `gantry_2025a.slx` has **orphaned** Coulomb blocks in its XML that are NOT in the
  loaded model (`find_system` returns none) — so there is no Simscape Coulomb oracle;
  the MATLAB cross-check uses the analytic `gantrySystem`+Coulomb EOM instead.
- MATLAB is OFF the critical path; the format is verified in Python (vs `direct` and
  vs the MATLAB EOM).
- `rk4_step_coulomb` must be `torch.compile`d like the baseline (was eager, fixed).

## 70821 BASELINE NUMBERS (comparison target)

Open-loop NRMSE: X1/X2 ~50%, Y 38-65% (POOR >30% on all). Recovered params (identifiable):
kb_sum 3975->53345 (+1242%), cb_sum 18->93, cg1 136->841, cg2 136->784, cy 98->664,
masses shrink ~30%. Checkpoint: `simulations/server-output/lfr_param_recovery_telica_split_22traj_e4000_70821.pt`.
