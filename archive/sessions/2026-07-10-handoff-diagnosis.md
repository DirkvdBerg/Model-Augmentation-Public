# ARCHIVED 2026-07-10 — diagnosis-phase handoff (mission COMPLETE)

This was the "why can't X+Θ+Y learn" diagnosis brief. That diagnosis is now DONE and fully captured in
`docs/drift-diagnosis-status.md`. Conclusion: drift = the ANN's energy-injecting DC force on the K=0
(X/Y) free integrators, invisible to the 0.1 s loss; input/IC/encoder/lr/nf all cleared. Kept for
provenance only. See the current `tasks/handoff.md` for the implementation brief.

---

# Session Handoff — Augmentation free-run mismatch: data-integrity / encoder-init diagnosis

**Written**: 2026-07-09. _Prior handoff (D-068/D-071 era) archived to `archive/sessions/2026-07-09-handoff-prev.md`._

## Mission for the new session
Find out **why the X+Θ+Y augmentation cannot learn**. Decide between two hypotheses **before any more training or hyperparameter search**:
- **(A) Data / encoder-init / storage inconsistency** — the free-run starts from the *wrong state* (especially X/Y position), so on the K=0 integrators it can never recover and it *looks like* drift.
- **(B) Genuine K=0 free-integrator drift** — the physics is fine, the drift is structural.

The user recalls a **slide showing measured/simulated `x_logical` vs the encoder reconstruction already mismatching before 0.1 s**. Chasing that is the point of this session. If (A), there is a data bug that made *every* augmentation attempt fail and it must be fixed first. If (B), proceed to the structural fix.

## Read first (hard constraints)
- `CLAUDE.md`, `docs/control-reasoning.md`.
- `tasks/lessons.md` — especially: **X/Y routing is a hard requirement, never propose Θ-only** (D-103); **every gantry training/search script must print `[nf-probe] train nf-RMS / val nf-RMS` per epoch**; a comparison diagnostic proves nothing until its control condition is well-behaved.
- `docs/decisions.md` D-100…D-103 (this session).
- `docs/gantry-augmentation-problem-log.md` (failure modes + run table).

## System, in one paragraph
Physics-based LPV-LFR baseline (Garcia-Herreros dual-gantry, Y-scheduled inertia) + a learned dynamic parallel ANN (Hoekstra LFR framework, SUBNET encoder). Truth = baseline + a hidden mass-spring-damper absorber (fa = 150 Hz, `delta_a std ≈ 2.2e-5 m` — tiny). Augmentation-mode excitation is a narrowband multisine [130,180] Hz. Logical state `x_logical = [X, Θ, Y, Ẋ, Θ̇, Ẏ]` (idx 0..5) + absorber `[δ_a, δ̇_a]` (idx 6,7). **X and Y are K=0 double integrators → any nonzero-mean correction integrates into unbounded drift.** Θ (kb1+kb2) and the absorber have restoring forces.

## Symptom (job 69399: X+Θ+Y, lr=1.3e-7, nf=1600, full data)
`[nf-probe] train nf-RMS` flat ~1.67e-4, `val nf-RMS` flat ~1.2e-4, **val sim-RMS rises** (init 8e-5 → 3–5e-4), **best checkpoint = epoch 0**, training loss barely moves.

## (Superseded — see docs/drift-diagnosis-status.md for the full completed diagnosis and the passivity solution.)
