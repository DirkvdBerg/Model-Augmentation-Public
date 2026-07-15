# The Dissipative Augmentation Block — Constructive Spec (heuristic-free)

**Date**: 2026-07-10. **Purpose**: convert the recursive diagnostic sprawl into a linear build. This is the
EXACT block to construct, its guarantees mapped to the §5 four requirements, and a strict separation of
**PROVEN / ASSUMED / EMPIRICAL**. No tuned constants. Companion: `docs/drift-diagnosis-status.md` (diagnosis,
§5g plan), `docs/passivity-augmentation-literature.md` (§G/§H provenance). Decisions: D-103/104/105/106.

---

## 1. Scope (what the block is and is not)
- **Acts on the K=0 free-integrator axes X and Y only** (§5e.1). Theta (spring `kb1+kb2`) and the absorber
  (spring `ka`) have restoring forces → bounded → left UNCONSTRAINED (raw ANN output), to preserve expressivity.
- **Reads the FULL state + input `z = [x, u]`** including X/Y position (HARD constraint, D-105/§5k0). The
  guarantee is enforced on the block's OUTPUT, never by amputating the input.
- **Writes a FORCE to the X/Y velocity (acceleration) rows.** It adds NO stiffness/damping to the position
  row → the integrator pole stays exactly at the origin (criterion 3, by inspection).

## 2. The block (exact definitions — this is what to code)
Let `net_θ : R^{n_z} → R^{2}` be the ANN (full-state input). Define the **bounded potential**
```
    φ_θ(z) = tanh( net_θ(z) ) ∈ (−1, 1)^2         # bounded by construction, ‖φ_θ‖∞ ≤ 1
```
**X/Y force output = exact first difference of the bounded potential (Route B, "bounded net impulse"):**
```
    F_xy,k = φ_θ(z_k) − φ_θ(z_{k−1})               # the routed force on the X/Y velocity rows
```
That is the entire criterion-4 mechanism. (Continuous analogue: `F_xy = d/dt φ_θ(z)`; the exact difference
is its discrete, telescoping form.) **No cutoff frequency, no ε, no penalty weight — zero tuned constants.**
(One benign scale `S`: `F_xy = S·(φ_θ(z_k)−φ_θ(z_{k−1}))`, fixed from output normalization like the existing
`Cd_norm`, not a tuned knob. `tanh` saturation caps instantaneous force at `2S`; size `S` from the data.)

**EXPRESSIVITY — what this FORBIDS (precise statement; stronger than "no DC").** Because `φ_θ` is
single-valued, `Σ F = φ_θ(z_N) − φ_θ(z_0)` is a STATE FUNCTION ⇒ **around any closed state-space loop
(`z_N=z_0`) the net impulse is ZERO.** So the block forbids the ANN from producing **any force with a
non-conservative / hysteretic / sustained NET-IMPULSE component**, namely:
(i) sustained DC (gravity, preload, cogging, net-directional friction);
(ii) hysteretic/path-dependent net impulse (Coulomb friction over asymmetric-timing motion:
`∫sign(v)dt ≠ 0` even returning to the same position);
(iii) the small nonlinear `M(Y)` rectification DC (D-A) → becomes a bounded offset, not learned.
It ALLOWS any zero-net-impulse force — oscillatory/resonant coupling, the 150 Hz absorber (validated D-B),
transients, anything `= d/dt[bounded ψ]`. **This is the §5f net-impulse-vs-passivity tradeoff made concrete:
the net-impulse bound buys criterion 4 (bounded position) at the cost of criterion 2 (non-conservative
friction impulse).** Resolutions: (a) friction → `f_base` (§5d) so the ANN residual is the zero-impulse
coupling; (b) add PH storage states (§2 optional layer) so `φ` may be path-dependent in the OBSERVED
coordinates while `Σ F` stays bounded via `H ≥ 0` (a LESS restrictive block); (c) the single NI certificate
(unifies crit-2 and crit-4). Do NOT read the bare difference as "only removes DC" — it removes all
non-conservative net impulse.

**Optional expressivity layer (for the single-certificate version, §5g Phase 1/5, NOT required for the sim
fix):** give any added augmentation states a passive port-Hamiltonian structure `ẋ_a = (J−R)∇H`, `J`skew,
`R = L L^T ⪰ 0`, `H ≥ 0` (built + energy-audited, §5i/§5j). This provides passive energy STORAGE (the
absorber) but — proven §5j — does NOT bound position; it is orthogonal to the criterion-4 mechanism above.

## 3. What each piece guarantees (mapped to the four requirements)
| Req | Mechanism in §2 | Status |
|---|---|---|
| **1 knowledge-free** | exact-difference form holds for ALL weights θ; property of the parametrization, not the data | PROVEN |
| **2 friction-permitting** | oscillatory (zero-net-impulse) residual IS representable; SUSTAINED-DC friction is NOT → goes to `f_base` | PARTIAL — see §5, the one assumption |
| **3 marginal-preserving** | force on velocity row only; no stiffness/damping on position row → pole stays at origin | PROVEN (by inspection) |
| **4 non-drifting** | telescoping net impulse + damped axis (`c>0`) ⇒ bounded position (proof §4) | PROVEN |

## 4. The criterion-4 proof (3 lines, no tuned constant)
Per axis, mass-damper dynamics `m v̇ + c v = F_ext + F_xy`, with `c > 0` (measured: τ_X=1.55 s, τ_Y=1.01 s ⇒
`c = m/τ > 0`). Integrate over `[0,T]`, hold `F_ext` (the stored input) fixed:
```
    c·(q(T) − q(0)) = m·(v(0) − v(T))  +  ∫₀ᵀ F_ext dt  +  ∫₀ᵀ F_xy dt
```
- `∫₀ᵀ F_xy dt = Σ (φ_θ(z_k) − φ_θ(z_{k−1})) = φ_θ(z_N) − φ_θ(z_0)`, **bounded by `2‖φ_θ‖∞ ≤ 2√2` for ALL θ**
  (telescoping).
- `v(T)` bounded (damped axis), `∫F_ext dt` bounded (zero-mean finite record).
- ⇒ `q(T)` bounded, uniformly in θ. **The ANN cannot induce position drift.** ∎
(This is the honest core: it bounds the ANN's CONTRIBUTION to position, holding the baseline/input fixed.
`c>0` is load-bearing — a pure double integrator `c=0` would need the Negative-Imaginary layer instead.)

## 5. The ONE modeling assumption (declared, not hidden)
> **Any NON-CONSERVATIVE net-impulse force on X/Y (sustained DC OR hysteretic/path-dependent impulse) is
> PHYSICS, not learned residual.** Genuine friction/preload belongs in `f_base` (grey-box Coulomb/LuGre,
> §5d); the ANN residual is then the zero-net-impulse coupling, which §2 represents exactly. (This is the
> crit-2 cost of the net-impulse bound — see §2 expressivity note and §5f. If real friction proves too rich
> for `f_base`, escalate to the PH-storage block (§2 optional layer) or the NI certificate.)

This is a **defensible modeling choice, not a theorem** (D-A showed the dominant sim residual is zero-DC;
the small DC is partly external). On the **current sim there is NO friction**, so this assumption is vacuous
and the §2 block is complete as-is. On real data it becomes active (add Coulomb to `f_base`).

## 6. Proof-obligation ledger (PROVEN / ASSUMED / EMPIRICAL — no bluffing)
- **PROVEN**: crit-1 (all-θ), crit-3 (pole at origin), crit-4 (bounded position, §4). Passive PH block is
  passive-by-construction + discrete-passive (§5j audit `max_r ~ 4e-17`).
- **ASSUMED**: the §5 modeling assumption (sustained DC → `f_base`); MIMO net-dissipative cross-coupling on
  X/Y (D-A energy proxy, not proven). The two-channel separability on real data.
- **EMPIRICAL (validated post-hoc, NOT yet in-training)**: bounded-impulse removes drift while keeping the
  130–180 Hz absorber (D-B: Y 2.6e-2→flat, X→2e-6; band RMS unchanged). D-C: the constrained block TRAINS.
- **UNBUILT**: the single semidefinite-storage Negative-Imaginary certificate that would replace the
  two-reason argument (§2 impulse-bound + §2 PH-storage) with one energy/NI theorem. Thesis prize (Phase T).

## 7. Explicitly NOT claimed (honesty guard)
- The block is NOT "passive ⇒ no drift": passivity bounds velocity, not position (§5j). The no-drift
  guarantee is the **net-impulse** argument (§4), a DIFFERENT and stronger structural property for `c>0`.
- The high-pass `fc=30 Hz` used in the D-B DIAGNOSTIC is a proxy with a tuned cutoff; **the block in §2 has
  no cutoff** (exact difference). Do not conflate the diagnostic proxy with the construction.
- Nothing here bounds a genuine EXTERNAL sustained DC (gravity/preload) — that is `f_base`'s job, by §5.

## 8. Linear build sequence (replaces the recursive diagnostics)
1. **B1 — build §2 block** standalone; unit-test: `Σ F_xy` bounded for random θ (confirms §4 numerically).
2. **B2 — D-C on current sim (no friction)**: trains (nf-RMS AND sim-RMS ↓), no drift (flat slope), keeps
   130–180 Hz absorber, energy/impulse audit on the free-run. (Some parts done post-hoc; run in-training.)
3. **B3 — eigen-check**: linearise the trained augmented model; confirm the X/Y pole stays at the origin.
4. **B4 (real-data only)** — add Coulomb/LuGre to `f_base`; confirm §5 assumption holds; re-run B2/B3.
5. **Phase T (parallel)** — derive the single semidefinite-NI certificate to replace the two-reason argument.

**Gate to stop diagnosing:** B1–B3 test THIS spec, not an open-ended "is it safe?" question. If B2 fails
(kills the absorber or still drifts), the fault is localized to §2, not to a new diagnostic.
