# Session Handoff — IMPLEMENT the passivity-constrained augmentation

**Written**: 2026-07-10. _Prior (diagnosis-phase) handoff archived to `archive/sessions/2026-07-10-handoff-diagnosis.md`._

## Mission for the new session
**Start IMPLEMENTING** the chosen solution, following the staged plan of approach in
`docs/drift-diagnosis-status.md` §5g. The diagnosis is DONE; do not re-diagnose.

## Read FIRST (everything is here)
- **`docs/drift-diagnosis-status.md` — the master document.** Read it in full. Key sections:
  §1–4 diagnosis (drift = ANN energy-injecting DC on X/Y); §5 solution ranking; §5b chosen construction;
  §5c Coulomb-vs-integrator; §5d grey-box alternative; **§5e acceptance checklist**; **§5e.1 D-D1/D-D2
  plan**; **§5e.2 cross-coupling HARD CONSTRAINTS (MIMO + integral)**; **§5f PASSIVITY not pure
  dissipativity (CRITICAL)**; **§5g PLAN OF APPROACH (start here)**; **§5h friction injection (LuGre)**;
  §9 literature.
- `CLAUDE.md`, `tasks/lessons.md` (active constraints, esp. the new rules on not over-claiming).

## State of play (what is DONE / VALIDATED)
- **Diagnosis complete & measured.** Drift = the ANN's *energy-injecting* DC force on the K=0 (X/Y)
  free integrators, invisible to the 0.1 s loss. Input/IC/encoder/lr/nf all CLEARED (d1–d7, s2, s3, dA,
  dB). Optuna 69399: 4/5 trials epoch 0, best 18%, nf-RMS flat → not a tuning problem.
- **Bounded-integral "integrator bound" VALIDATED (D-C).** `bounded_integral_block.py` +
  `dC_boundedintegral_train.py`: trains (nf-RMS AND full-12s sim-RMS improve, beats epoch 0), no drift
  (full-12s Y sim-RMS 2.05e-4 vs 1.60e-2 drifted, 78×, flat slope), keeps absorber. This de-risks the
  "constrain the ANN output and train it end-to-end" mechanism. BUT bounded-integral forbids ALL DC →
  cannot represent friction → it is the SIM solution / drift-half, NOT the real-data solution.

## The chosen solution (what to build)
A **PASSIVE (dissipativity-constrained) ANN on X/Y**: constrain the learned X/Y force to be passive —
there is a stored-energy function `V(x) ≥ 0` with `dV/dt ≤ F·v` (i.e. `∫ F·v ≥ −V(0)`). Plain: the ANN
may **STORE and RETURN energy** (springs, masses, augmented states — so it CAN model the absorber and
real passive dynamics), but may **not CREATE net energy** (so it can't drift). Grounded in
Negative-Imaginary free-body theory (Mabrok 2014).

### THREE non-negotiable formulation rules (§5f, §5e.2)
1. **PASSIVITY (with storage V), NOT pure dissipativity `∫F·v ≤ 0`.** Pure dissipativity forbids energy
   storage → cannot represent the absorber. Use passivity-with-storage.
2. **MIMO (whole X/Y output jointly), NOT per-axis.** Per-axis forbids legitimate cross-axis energy
   transfer (the coupling) → can't learn it.
3. **INTEGRAL (storage-based), NOT pointwise.** Pointwise `F·v ≤ 0` forbids the oscillatory absorber
   coupling (transiently injects).
Scope the constraint to **X and Y only**; leave Θ and the absorber rows UNCONSTRAINED (springs → no
drift). Keep X/Y routing (D-103).

## START HERE — §5g Phase 1, then Phase 2 (D-D1)
- **Phase 1: build the passive block + unit-verify in ISOLATION.** Passive-with-storage, MIMO on X/Y.
  Recommended parametrization: neural port-Hamiltonian-style (`H ≥ 0` storage, `R ≥ 0` dissipation, `J`
  skew → passive by construction). Unit test standalone (mirror how `bounded_integral_block.py` was
  unit-tested): (a) energy audit `∫F·v ≥ −V(0)` for random inputs; (b) can produce a STORED-energy
  (spring-like) response, not only damping; (c) gradients flow.
- **Phase 2: D-D1 (self-contained diagnostic, no main-file edits).** Copy the pattern of
  `dC_boundedintegral_train.py` (class-level monkeypatch, per-rollout reset via `fit_sys.loss`). Verify:
  trains (nf-RMS AND sim-RMS), no drift (flat slope), **KEEPS THE ABSORBER (130–180 Hz band)** ← the
  make-or-break check, and energy audit holds. **If it kills the absorber → the formulation is wrong
  (per-axis/pointwise/pure-dissipative) → rework Phase 1.**

## The two MAKE-OR-BREAK checks
1. **D-D1 (Phase 2): does the passive block still learn the absorber?** If no → wrong formulation.
2. **D-D2 (Phase 4): does passive beat bounded-integral on injected LuGre friction?** If no → fall back
   to grey-box (Coulomb-in-`f_base` + bounded-integral, §5d).

## Later phases (see §5g)
P3 inject-friction sim (**LuGre** — nonlinear, dynamic, PROVABLY PASSIVE → fair test; add via a bristle
state per axis, pattern in `Matlab-scripts/Augmentation/additional_state_lagrangian.m`; §5h has refs).
P4 D-D2. P5 framework integration (build_model flag, encoder init of storage states, well-posedness,
joint estimation). P6 real Telica data. PT theory proofs (parallel; the contribution).

## Assets to REUSE
- `scripts/gantry/gantry_dynamic/bounded_integral_block.py` — block pattern (unit-verified).
- `scripts/gantry/diagnostics-drift/dC_boundedintegral_train.py` — self-contained train-diagnostic
  pattern (build_model → monkeypatch block → train → eval drift+absorber). `make_drift_checkpoint.py` —
  fast training setup. `drift_common.py` — loader, truth EOM, baseline EOM, P-helpers. d1–d7, dA, dB, s2, s3.
- Truth model: `Matlab-scripts/Augmentation/gantrySystemExtended.m` + `additional_state_lagrangian.m`.
- Entry/training: `scripts/gantry/gantry_interconnect_dynamic.py`, `gantry_dynamic/{model,training,data}.py`.
- Sim data: `data/gantry/matlab/trajectory/augmentation/` (V1 standstill has the absorber; T9/T10/T11
  aprbs excite X/Y). Outputs → `simulations/gantry_subnet/diagnostics/`.

## ENVIRONMENT GOTCHAS (important — cost real time last session)
- **`conda run -n GraduationProject ...` is currently CRASHING** (conda launcher bug). Run the env
  python DIRECTLY: `"/c/Users/20203253/AppData/Local/anaconda3/envs/GraduationProject/python.exe"` and
  set `PYTHONIOENCODING=utf-8` (Windows console cp1252 can't print unicode like Δ/∫/·). Keep script
  output ASCII to be safe.
- The Git-Bash `fatal error - add_item ... errno 1` fork error is TRANSIENT — just re-run the command.
- `conda run python -c` cannot take a newline string — write a scratch file (scratchpad dir).
- Every gantry training/search script MUST print `[nf-probe] train nf-RMS / val nf-RMS` per epoch, and
  report BOTH nf-RMS AND full-12s free-run sim-RMS (the deliverable metric).

## Out of scope for now
Noise/SNR; real Telica data (P6); the formal theory proofs (PT, parallel). Do NOT add Coulomb to the
current sim (no friction there → model–data mismatch); friction only enters the injected-friction sim (P3).
