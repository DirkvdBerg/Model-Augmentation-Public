# Diagnostic Plan: GNS random-walk noise injection WITH our encoder

**Date**: 2026-07-11. **Status**: plan, unbuilt (awaiting go-ahead to implement). **Question**: does GNS-style
random-walk noise injection reduce the X/Y free-run drift in OUR encoder-based SUBNET pipeline, WITHOUT
damping the marginal pole, while keeping the absorber? **Background/provenance**:
`docs/rollout-stability-literature.md` (GNS mechanism + the GNS-fit section), `docs/ml-for-control-search-
sweep.md` (Direction 1, timescale + two-component caveats), `docs/open-loop-solution-decision.md` (D-107).
This is a self-contained SIM diagnostic (no friction), like the existing `dC_`/`dB_` diagnostics.

## 1. What we are testing and why
Rollout methods target the DISTRIBUTION-SHIFT half of the drift (the model never sees its own accumulated
error because the encoder RE-INITIALIZES the state clean every 0.1 s window). GNS injects random-walk noise
to synthesize the drifted state artificially. KEY RISK (`rollout-stability-literature.md`): "correct-back"
training may induce an effective RESTORING/DAMPING action -> DAMP the marginal pole (fails requirement 3).
This diagnostic decides: does GNS help (no-drift, absorber kept, pole preserved, trains) or trade drift for
damping?

## 2. Hypotheses (falsifiable) -- what each measures and its falsification criterion
| # | Hypothesis | Metric | PASS | FALSIFIED |
|---|---|---|---|---|
| **H1** | GNS reduces free-run drift | X/Y position-ENVELOPE growth: RMS\|q\| 4th-quarter / 3rd-quarter over 12 s | ratio ~1.0 | ratio >~1.2 (still drifting) |
| **H2** | Absorber kept | 130-180 Hz band RMS on Y (free-run) | unchanged vs unconstrained | drops (coupling suppressed) |
| **H3** | **Marginal pole preserved (LOAD-BEARING)** | linearized discrete X/Y rigid-body eigenvalue \|lambda\| | \|lambda\| = 1 (pole at origin) | \|lambda\| < 1 (damped) |
| **H4** | It trains | per-epoch train AND val nf-RMS | both decrease | neither decreases |

**H3 is the point of the diagnostic.** H1+H2 alone (drift gone, absorber kept) would look like success, but if
H3 fails (\|lambda\|<1) GNS merely damped the free axis into a strictly-stable one -- the same wall
dissipativity hits (`dissipativity-limits.md` B3). Report all four; a "win" requires H1,H2,H4 pass AND H3
pass.

## 3. State indices and injection targets (verify against the interconnect before coding)
Logical state (verify in `model.py` / `drift_common`): `[X=0, Theta=1, Y=2, Xdot=3, Thetadot=4, Ydot=5,
delta_a=6, vdelta_a=7]`. K=0 free axes: **X (pos 0, vel 3), Y (pos 2, vel 5)**. Y-POSITION (row 2) is the LPV
SCHEDULING variable -- `M(Y)` depends on it. **HARD RULE: never perturb row 2 (Y-position)**; corrupting the
scheduling variable while keeping a clean target would teach the model to UNDO Y-scheduling.

## 4. The three arms
| Arm | What | Injection target |
|---|---|---|
| **Control** | unconstrained ANN, correct lr, same data/seed | none |
| **GNS-A** | literal GNS: random-walk noise on velocity | rows 3 (Xdot), 5 (Ydot) during the nf-window rollout |
| **GNS-B** | long-horizon position-drift exposure (X only) | row 0 (X-position) offset on the encoder x0, scaled to free-run drift at a random horizon; NEVER row 2 (Y) |

**Why both, and the honest asymmetry:** our drift lives in POSITION (free integrator), but velocity is damped
-> A (velocity noise) reaches position drift only via integration, which our 0.1 s window barely provides
(timescale gap); B offsets position directly and reaches the regime. BUT B cannot touch Y-position
(scheduling), so **X gets both A and B; Y-drift robustness is testable ONLY via A (velocity injection on row
5).** This asymmetry is itself a finding: the LPV scheduling on Y limits the position-exposure trick to X.

## 5. Injection mechanics (details)
- **GNS-A:** random-walk `n_k = sum_{i<=k} eps_i`, `eps_i ~ N(0, sigma_A^2)`, added to the PROPAGATED
  Xdot/Ydot during the rollout (NOT the encoder position-window input -- velocities are encoder-reconstructed
  by differentiating positions, so position-window noise is amplified; [[trace-state-reconstruction]]).
  Target stays the clean truth trajectory.
- **GNS-B:** offset the encoder's x0 on row 0 (X-position) by a random draw scaled to the MEASURED free-run
  X-drift at a random horizon tau in [0, 12 s] (simulates "the model is at its free-run state at time tau").
  Optionally also offset Xdot (row 3). Target clean. Row 2 (Y) untouched.
- **sigma DATA-DERIVED (not a knob):** sigma_A from the model's own one-step error scale (GNS recipe) or the
  measured accumulated-error scale; GNS-B magnitude from the measured free-run drift-vs-horizon curve (d7,
  already computed). Do NOT hand-tune; flag any constant as HEURISTIC per CLAUDE.md if it cannot be derived.

## 6. The eigen-check (H3) -- how
Linearize the TRAINED augmented model about an operating point (and/or along the trajectory): extract the
discrete state-transition Jacobian `A = d x_{k+1} / d x_k`, compute eigenvalues, identify the X and Y
rigid-body (integrator) modes, report \|lambda\|. Pole-at-origin (continuous) = \|lambda\|=1 (discrete). If
GNS pushed \|lambda\|<1 the free axis was damped -> req-3 fail. Do this for all three arms (control should
show \|lambda\|~1; GNS arms tested against it).

## 7. Metrics discipline (per lessons)
- Drift = position-ENVELOPE growth (RMS\|q\| window ratio), NOT a slope/velocity proxy (a bounded oscillation
  trips slope; [[detect-drift-by-envelope]]).
- Absorber = 130-180 Hz band RMS (the coupling we must keep).
- Per-epoch train AND val nf-RMS PRINTED live each epoch (mandatory; [[carry-monitoring-prefs]]).
- All numbers labeled RMS [m] with the noise floor sigma_n for reference.

## 8. Staging (isolation-first; cheap before pipeline)
- **Stage 0 -- unit test (ms):** the random-walk generator produces correct accumulated statistics; injects
  ONLY on the intended rows (0/3/5, never 2); target unchanged. No training.
- **Stage 1 -- short training, 3 arms:** correct post-D-101 lr (69399 ran at lr=1e-3, invalid -- do NOT reuse
  its conclusion). Small nf, few epochs. Print train+val nf-RMS/epoch. Then free-run 12 s + eigen-check per
  arm. Compute H1-H4.
- **CONTROL-HEALTH GATE (mandatory before concluding):** the CONTROL arm must at least TRAIN IN-WINDOW
  (nf-RMS decreases, learns the absorber) even though it drifts on free-run. If the control does not learn
  in-window, the setup is broken and the A-vs-B-vs-control comparison is INVALID (the 550x-vs-23x incident,
  [[control-must-be-healthy]]). Fix lr/steps until the control is in-window-healthy, THEN compare.

## 9. Scope and honest limits (built in, not hidden)
- **Current sim has NO friction.** This tests H1 no-drift, H2 absorber, H3 marginal-preservation, H4
  trainability ONLY. It does NOT test friction capture (that needs the injected-friction sim, D-D2). H3 is
  the new thing this buys over prior diagnostics.
- **Distribution-shift half only.** If the drift is dominated by the IDENTIFIABILITY half (unexcited DC,
  §5m), GNS may FAIL H1 even with correct injection reaching the right scale -- and that is an INFORMATIVE
  negative (would say: rollout methods insufficient; need excitation / data-silent). Report it as such.
- **Sim only**; real closed-loop Telica is a separate later step.
- **GNS-B is our adaptation**, not literal GNS; GNS-A is the proven recipe. Keep them distinguishable in the
  results.

## 10. Decision gates (what each outcome means)
- **H1,H2,H4 pass AND H3 pass (|lambda|=1):** GNS works for our drift without damping -> promote to the D-107
  first step; then D-D2 (injected friction) for friction capture.
- **H1 passes BUT H3 fails (|lambda|<1):** GNS traded drift for DAMPING -> req-3 fail; rollout-correct-back
  is NOT marginal-preserving -> reject as the fix, record the failure, fall back to excitation/data-silent.
- **H1 fails (still drifts):** injection did not reach the drift regime OR the identifiability half dominates
  -> check injection scale vs the measured drift; if scale is right and it still drifts, that is evidence the
  identifiability half dominates (excitation/data-silent needed, not rollout).
- **A vs B:** if B (X-position exposure) beats A (velocity) on X-drift, confirms the position-exposure /
  timescale argument; if A suffices, simpler and it also covers Y. Informs which to carry forward.

## 11. Outputs
Data (JSON/CSV) + falsifiable plots to `simulations/gantry_subnet/diagnostics/` (NOT under `scripts/`). Each
plot poses the test (prediction vs measurement, quantified deviation), does not assert the conclusion in the
title ([[falsifiable-plots]]). One decision entry in `docs/decisions.md` after the run.

## 12. Related documents
- `docs/rollout-stability-literature.md` -- GNS mechanism + fit + req-3 risk (the basis for this plan).
- `docs/ml-for-control-search-sweep.md` -- Direction 1 (timescale + two-component caveats).
- `docs/open-loop-solution-decision.md` (D-107) -- the open-loop, solve-not-hide direction this serves.
- `docs/drift-diagnosis-status.md` -- diagnosis (d6 DC, d7 timescale), §5g phases, §0 index.
