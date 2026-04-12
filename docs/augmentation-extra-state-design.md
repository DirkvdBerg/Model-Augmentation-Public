# Augmentation Extra State Design
**Session date**: 2026-04-10  
**Last updated**: 2026-04-10 (revised after literature review and candidate analysis)  
**Status**: Design under revision — primary candidate changed (see Core Problem below)

---

## 0. Core Problem

We have been optimising for two conflicting goals:

1. **Physical motivation**: the extra state should represent a real unmodelled effect in
   the Telica gantry.
2. **Identifiability**: the extra state must be clearly learnable by the augmentation from
   standard closed-loop data.

Every candidate evaluated scores well on one axis but poorly on the other:

| Candidate | Physical motivation | Identifiability |
|---|---|---|
| Dahl friction [z1, z2] | Strong (Lan Jia 2023 confirms position-dependent Fc) | Weak: requires micro-reversal excitation at ±0.001 m/s; Fc ~ 2 N; closed-loop suppresses signal |
| First-order F_Δ lag [z] | Moderate (cable, actuator, or bearing compliance) | Moderate: continuously excited, but quasi-static component is absorbable by baseline |
| Structural DOF [δ, δ̇] | Weak (Garcia: cross-arm bending negligible; granite frame) | Strong: resonance is non-absorbable, always active |
| Yaw-flex mode [φ, φ̇] | Moderate (dual-drive literature mentions guide-bearing resonances) | Strong: resonance at chosen ω_φ is non-absorbable; active during normal tracking |

No candidate is obviously good on both axes simultaneously. That is a real problem, not a
sign that we are missing something.

**Resolution**: for a simulation proof-of-concept, identifiability takes priority over
physical motivation. The benchmark must succeed first; physical realism is secondary.
The **primary recommendation is now [φ, φ̇]** (Section 5.7 / Section 5.9). See Section 5.9
for the full argument.

---

## 1. Goal

**Important clarification: static versus dynamic augmentation**

In Jan's LFR augmentation framework, the word *augmentation* can mean two different things:

- **Static augmentation**: add a learned correction to the baseline state/output equations
  **without increasing the state dimension**. The model keeps the same states as the
  baseline and only changes the functions acting on those states.
- **Dynamic augmentation**: add **additional augmentation states** with their own dynamics.
  The model state is enlarged from the baseline state `x_base` to an augmented state
  `[x_base, x_aug]`, where `x_aug` contains the hidden states needed to represent dynamics
  that are structurally absent from the baseline.

**This document is about dynamic augmentation.**  
We are **not** trying to add only a static correction to the existing 6 baseline states.
We are explicitly focusing on the case where the augmentation introduces **additional
states** that carry missing dynamics, exactly in the same structural sense as Jan's MSD
example where the augmentation adds states beyond the reduced baseline.

**We want to add an extra state to the data-generating model such that:**

1. The extra state is reflected in the data the model generates — meaning the output
   trajectories [X1, X2, Y] carry information about that state that would not be present
   if the state were absent.
2. The 6-state LPV baseline, trained on that data, cannot explain the outputs fully —
   because the state is structurally absent from the baseline by construction.
3. The augmented model (baseline + ANN with extra learned states) is able to discover and
   capture that state — recovering the missing dynamics from the data alone.

**If the augmentation succeeds, it proves that the LFR augmentation framework can learn
dynamics that the baseline is structurally incapable of representing — not just re-tune
parameters, but genuinely extend the model's capabilities.**

**A strong additional requirement** (beyond the three conditions above) is that the extra
state introduces **coupling between axes** or **position-dependent behaviour**. This matters
for two reasons:

- The gantry's primary challenge is cross-axis coupling (X1/X2 → Θ) and its variation
  with payload position Y. These are the effects that limit real-world positioning accuracy
  and that the LPV baseline only partially captures.
- If the extra state couples into the Θ channel with a Y-dependent amplitude, the
  augmentation must learn not just the state dynamics but also their scheduling-variable
  dependence — a richer and more meaningful demonstration of the framework's capability
  than a simple decoupled extra DOF would provide.

**Practical requirement** (added after literature review and candidate analysis): the extra
state must produce a signal that is **clearly visible and identifiable from realistic
closed-loop position data**. A state that requires specialised, pathological excitation
to activate its effect is unsuitable for a demonstration study.

**Primary recommended state** (see Section 5.9 and Core Problem above): a second-order
yaw-flex compliance mode [φ, φ̇] coupled into the Θ channel with Y-dependent amplitude:

```
φ̈ + 2ζ_φ ω_φ φ̇ + ω_φ² φ  =  b(Y) · F_Δ        F_Δ = ½(F_X1 - F_X2)
X1_out = X + (Θ + φ)·Lb/2
X2_out = X - (Θ + φ)·Lb/2
```

**Alternative documented state** (Section 5.8): first-order F_Δ-driven internal variable.
Simpler ODE, one state, but partially absorbable by baseline parameter tuning.

This is a simulation study: because we construct the data-generating model ourselves, the
true extra state trajectory z(t) is known exactly and can be used to verify whether the
augmentation has correctly identified it.

---

## 2. Purpose of This Document

This document captures the full reasoning behind the choice of extra state, including all
candidates evaluated, the literature that informed the decision, and the observability
principle that governs identifiability. The simulation study proceeds in three steps:

1. Build a **data-generating model** (baseline + extra state) and simulate training data.
2. Train the **augmented model** on that data, using the 6-state LPV as the incomplete
   baseline.
3. Verify that the augmented model recovers the missing dynamics — comparing learned states
   against the known ground truth.

This mirrors exactly what Jan did in his MSD benchmark: data from 3-DOF, baseline is 2-DOF,
and the **dynamic augmentation adds states** to recover the missing dynamics of `m3`.

---

## 3. Jan's Approach — The Correct Analogy

**Important clarification on direction**: the extra state goes in the *data-generating model*,
not in the baseline. The baseline stays as the 6-state LPV (García-Herreros rigid model). The
augmented model then adds states on top of the baseline to recover what the baseline misses.

```
Jan:
  Data generator: 3-DOF MSD (6 states)  ->  includes m3 with nonlinear spring
  Baseline:       2-DOF MSD (4 states)  ->  m3 entirely absent
  Augmentation:   adds 2 states         ->  must learn m3 dynamics and its coupling to m2
  Verification:   true x3(t) known      ->  can check augmentation state against truth

Our case:
  Data generator: 7-state model         ->  6-state LPV + first-order internal state z
  Baseline:       6-state LPV           ->  Garcia-Herreros rigid model, constant C and K
  Augmentation:   adds 1 state          ->  must learn z dynamics and its Y-dependent coupling
  Verification:   true z(t) known       ->  can check augmentation state against truth
```

**What a state is (fundamental principle)**:
A state is any quantity that (a) has its own dynamics (its own differential equation), and (b)
cannot be computed from the other states at the current instant alone. It carries independent
*memory* — its value depends on the history of the system, not just the present.

Consequence: even with perfect knowledge of all parameters, if a physical quantity that
satisfies (a) and (b) is absent from the state vector, the model is *structurally* unable to
reproduce that quantity's effect on the dynamics. Parameters alone cannot substitute for a
missing state.

---

## 4. What Our LPV Baseline Already Captures

The 6-state LPV captures:

- **Y-dependent inertia coupling** through M(Y) = M0 + M1·Y + M2·Y²
  - M[0,1] = M[1,0] = (m1-m2)·Lb/2 - mh·Y  (coupling between X-translation and Θ-rotation)
  - M[1,2] = M[2,1] = -mh·d                 (coupling between Θ-rotation and Y-axis)
- **Viscous damping** (constant C matrix): cg1+cg2 on X, cg1-cg2 coupling on Θ, cy on Y
- **Joint stiffness** (constant K matrix): kb1+kb2 on Θ

**What the baseline does NOT capture** (structural gaps, not parameter gaps):

| Missing effect | Type | State needed? |
|---|---|---|
| Y-dependent damping (friction load distribution) | Position-dependent nonlinearity | Yes — dynamic internal state |
| Coriolis/centripetal coupling (Ẏ·Θ̇ terms) | Velocity-dependent nonlinearity | No — same states, nonlinear terms |
| Dynamic lag in force-path (cable, actuator, bearing compliance) | First-order internal dynamics | Yes — one internal state |
| Cross-arm bending mode | Structural flexibility | Yes — bending DOF |
| Base/support structure resonance | Structural flexibility | Yes — [x_b, ẋ_b] |

The baseline has M(Y) but **constant C and K**. Any dynamic effect that adds a pole to the
input-to-output path (beyond the six poles from the rigid-body model) is structurally absent.

---

## 5. Candidates Evaluated

### 5.1 Support Structure Resonance [x_b, ẋ_b]

Garcia (Section 2.4) explicitly identifies a 37.7 Hz die-cast base resonance as unmodelled.

**Rejected for our system**: The Telica datasheet specifies a granite/polymer-concrete frame
(not die-cast aluminium). Granite is approximately 6 times stiffer; its first resonance is well
above 100 Hz and above the control bandwidth. The 37.7 Hz resonance was specific to Garcia's
experimental rig. Additionally, this state has no Y-dependence — it does not enrich the LPV
structure.

**Note on ChatGPT research (second document, Candidate A)**: The second literature analysis
document re-proposes this resonance from Garcia's experimental data with ω_ξ = 2π·37.7 rad/s.
The physical motivation is that vibration at this frequency was visible in X1-X2 residuals of
order ±20-40 µm during aggressive accelerations in Garcia's setup. This is a genuine and
large effect in Garcia's system. However it does not apply to the Telica granite frame. The
candidate remains rejected on physical grounds for our specific system.

### 5.2 Flexible Structural DOF [δ, δ̇]

This category covers structural extra DOFs: cross-arm bending, or a flexible mounting between
the Y-payload and the crossarm.

**Physical mechanism**: treating the Y-payload mh as connected to the crossarm via a flexible
joint (spring kf, damper cf) rather than rigidly bolted. The payload then has a relative
displacement δ with its own ODE:

```
mh·δ̈ = -kf·δ - cf·δ̇ + (coupling to crossarm acceleration)
```

**Why the closest formal analogy to Jan**: extra DOF in data generator, absent from baseline,
Newton's law ODE, coupling via spring force — exactly mirroring m3 in the MSD benchmark.

**Why rejected for the Telica system**:
1. Garcia (Section 2.4) explicitly calls cross-arm bending "negligible in comparison to the
   coupling between actuators" and uses the rigid rod formula for the crossarm.
2. The Telica granite frame is specifically chosen to suppress structural modes above the
   control bandwidth. No structural mode frequencies appear in the datasheet.
3. kf and cf would have to be fabricated with no reference values from any source.
4. A linear [δ, δ̇] DOF does not add Y-dependence to the LPV structure.

**Retained as documented alternative**: if the supervisor prefers a closer formal analogy
to Jan's MSD case, [δ, δ̇] is defensible — but requires accepting that the physical premise
contradicts the available documentation.

### 5.3 Coriolis Coupling (velocity-dependent)

Terms like 2·mh·Ẏ·Θ̇ appear in the full nonlinear EOM but are dropped in the linearisation.

**Not a separate state**: Coriolis terms are nonlinear functions of existing states [Θ̇, Ẏ].
They do not require an extra state variable. A static (non-dynamic) augmentation could add
Coriolis correction without extra states. This is the effect targeted in the second
augmentation step (D-024).

### 5.4 sign(Ẏ) — Supervisor Suggestion

A supervisor suggested adding sign(Ẏ) to capture hysteresis direction. Decision D-025 records
this as a valid scheduling variable for the augmentation.

**Not a state**: sign(Ẏ) is a static (memoryless) nonlinearity with no dynamics.

### 5.5 Y-Position-Dependent Friction States [z₁, z₂] — Evaluated and Rejected

**Physical mechanism**: when the Y-payload sits at position Y along the cross-arm, the normal
load on each X-guide is different, giving Y-dependent Coulomb amplitudes:

```
Fc1(Y) = Fc · (Lb/2 - Y) / Lb
Fc2(Y) = Fc · (Lb/2 + Y) / Lb
```

The Dahl (pre-sliding friction) state for each guide:

```
ż1 = Ẋ1 - (|Ẋ1|/g) · z1      Ẋ1 = Ẋ + (Lb/2)·Θ̇
ż2 = Ẋ2 - (|Ẋ2|/g) · z2      Ẋ2 = Ẋ - (Lb/2)·Θ̇
```

The torque asymmetry on Θ when Y ≠ 0 is the key coupling mechanism.

**Why this was initially attractive**: physical motivation is documented (Garcia lists
Y-dependent friction as a gap; supervisor observes hysteresis; Dahl/LuGre states are the
standard model in the friction literature). The supervisor's sign(dY/dt) observation (D-025)
is precisely the static approximation of what z1, z2 model dynamically.

**Why rejected after literature review** (see Section 12 for sources):

1. **Fc in gantry context is 1.4-2.75 N** (Lan Jia 2023, TU Delft, gantry stage measurements
   at 9 positions). The design document initially assumed Fc = 20 N. At Fc = 2 N and Y = 0.3 m
   on a Lb = 0.725 m crossarm, the asymmetric torque on Θ is approximately:
   `(Fc1 - Fc2) · Lb/2 = Fc · (2Y/Lb) · Lb/2 ≈ 2 · 0.83 · 0.36 ≈ 0.6 N·m`
   This is small relative to the Θ restoring force from kb1+kb2 = 3975 N·m/rad.

2. **Required excitation is pathological**: Lan Jia (2023) uses velocity reversal patterns
   over 4 µm strokes, repeated 50 times, with end speeds of ±0.001 m/s to activate the
   pre-sliding transient. This is far from any realistic closed-loop tracking trajectory.

3. **Closed-loop suppresses the signal**: Lan Jia explicitly reports that closed-loop
   identification of dynamic friction parameters is difficult because the controller
   suppresses exactly the low-speed reversal behaviour needed to observe the pre-sliding
   transient. Open-loop identification is preferred in the friction identification literature.

4. **Pre-sliding transient is very short**: At normal operating speeds (v = 0.1 m/s) and
   g = 5 µm, the pre-sliding transient lasts Δt = 2g/v = 0.1 ms (2 samples at 20 kHz).
   The baseline's viscous terms can absorb the steady-state Coulomb-like portion. Only the
   transient and the Y-dependent asymmetry are genuinely non-absorbable — but both are
   small at realistic Fc values and require special excitation.

**Conclusion**: Physically well-motivated, but the combination of small Fc, pathological
excitation requirements, and closed-loop signal suppression makes [z1, z2] a poor choice
for a demonstration study where identifiability is paramount.

### 5.6 Cable/Dresspack Viscoelastic State [z_c]

**Identified in**: second ChatGPT literature analysis document (Candidate B), citing a TU
Delft thesis that measures "cable slab forces" of order 1-3 N with strong position dependence
on a high-precision motion stage, attributing the variation to stored elastic energy as a
function of bend radius and routing. The ETEL TELICA datasheet explicitly mentions "integrated
cables and tubes".

**Physical mechanism**: a viscoelastic cable/dresspack model (Maxwell/Zener Standard Linear
Solid) introduces one internal stress-relaxation state z_c:

```
ż_c = -(1/τ_c(Y)) z_c + k_c(Y) · ℓ̇(q)
F_cable = k_0(Y) · ℓ(q) + z_c
τ_Θ_cable = r_c(Y) · F_cable
```

where ℓ(q) is the effective cable extension as a function of gantry configuration (dominated
by Y), τ_c(Y) is a relaxation time constant, and r_c(Y) is an effective moment arm from
cable routing asymmetry.

**Why viable**:
- One extra state with a physically interpretable first-order ODE
- The ETEL datasheet confirms cables exist
- Active during normal motion (cable forces present whenever the gantry moves)
- Y-dependent parameters through cable routing geometry
- Couples into Θ via r_c(Y) moment arm

**Why not chosen as primary**:
- The specific claim about cable slab force magnitude and Y-dependence comes from a ChatGPT
  document (reference 16 in that document). This has not been verified against the primary
  TU Delft thesis.
- The coupling into Θ depends on r_c(Y) — the effective moment arm from asymmetric cable
  routing. This is speculative: cable routing geometry is not documented for the Telica system.
- If r_c(Y) is approximately zero (symmetric routing), the Θ coupling vanishes and the state
  only affects X. Without confirmed routing asymmetry, the Y-dependent Θ coupling cannot
  be asserted.

**Status**: physically plausible but unverified for the Telica system. The ODE structure is
essentially the same as the chosen first-order state (Section 5.8), so choosing the more
general formulation (Section 5.8) covers this case without requiring unverified routing
geometry.

### 5.7 Bearing/Guide-Compliance Yaw-Flex Mode [φ, φ̇]

**Identified in**: second ChatGPT literature analysis document (Candidate C), citing an
IEEE/ASME paper that models and experimentally verifies a dominant vibration mode attributed
to bearing/guideway compliance in a linear motor stage, and a dual-drive gantry paper that
explicitly mentions frequency-domain resonances attributed to guide bearings and crossbeam
rotation.

**Physical mechanism**: a compliance DOF in the bearing/guideway connection introduces an
extra rotational mode on top of the rigid Θ:

```
φ̈ + 2ζ_φ ω_φ φ̇ + ω_φ² φ = b_φ(Y) F_Δ + d_φ(Y) Ẏ F_Σ
```

Output mapping: X1_out = X + sin(Θ+φ) Lb/2, X2_out = X - sin(Θ+φ) Lb/2. For small angles:
φ appears as an additive extra rotation in the X1-X2 difference.

**Strengths**:
- Two extra states (matching Jan's 2-state augmentation structure from the MSD benchmark)
- Driven by F_Δ (active during normal closed-loop tracking with yaw corrections)
- Creates a resonance peak — an unmistakable spectral signature the baseline cannot absorb
- Y-dependent coupling b_φ(Y) can enrich the LPV structure
- The dual-drive literature explicitly supports guide-bearing-related resonances in H-type
  gantries

**Previous objections and why they are resolved**:

- *"Narrow-band resonance contributes little to time-domain MSE."* This is addressed by
  choosing ω_φ within the tracking bandwidth (30-50 Hz) with moderate damping (ζ = 0.1-0.2,
  quality factor Q ~ 3-5). A mode at 40 Hz responds to the broadband acceleration content
  of any pick-and-place trajectory; the residual appears as a damped oscillation
  superimposed on every acceleration/deceleration event — clearly visible in X1-X2.
- *"Garcia dismisses structural modes."* This is a simulation benchmark, not a claim about
  the real Telica system. Physical motivation is secondary to making the demonstration work.
- *"Three parameters to fabricate."* This is true for every candidate — parameters are
  always chosen. For this state, ω_φ and ζ_φ are chosen for visibility; b(Y) = b0 + b1·Y
  for a simple Y-dependent coupling.

**Why this is NOW the primary recommendation** (see also Section 0):

The fundamental issue with every other candidate is absorbability or poor excitation:
- Dahl [z1, z2]: quasi-static component absorbed by viscous tuning; transient only visible
  under pathological micro-reversal excitation.
- First-order lag [z]: quasi-static component (z ≈ k·τ·F_Δ) is a pure gain on F_Δ that
  the baseline's B matrix can absorb. The non-absorbable part is a subtle phase shift.
- [φ, φ̇]: a resonance at ω_φ introduces new poles that the baseline's M(Y), C, K
  **structurally cannot produce**. The baseline has exactly three resonance frequencies
  determined by its three mechanical DOFs. A fourth frequency requires extra states, period.
  There is no parameter adjustment that makes a 3-DOF rigid model produce a 4th resonance.

This is the same reason Jan's m3 works: it adds a resonance (or near-resonance) that the
2-DOF baseline cannot absorb regardless of how its parameters are tuned.

**Concrete parameter recommendation**:
- ω_φ = 2π·40 rad/s (40 Hz, well within the 100 Hz tracking bandwidth)
- ζ_φ = 0.15 (Q ≈ 3.3; broad enough to be excited by standard trajectories)
- b(Y) = b0 + b1·Y with b0, b1 chosen so max |φ| ~ 10-20% of max |Θ| during tracking
- Physical label: "yaw-flex compliance mode from bearing or cable compliance"

**Status**: PRIMARY RECOMMENDATION. Matches Jan's 2-state dynamic augmentation structure,
produces a robustly non-absorbable resonance signature, and is active under standard
closed-loop tracking without specialised excitation.

### 5.8 First-Order F_Δ-Driven Internal State [z] — DOCUMENTED ALTERNATIVE

**Identified as**: the "maximally identifiable demonstration state" in the second ChatGPT
literature analysis document, synthesised from Candidates B and D and the general
disturbance-as-state literature.

**State ODE and coupling**:

```
ż = -(1/τ(Y)) z + k(Y) F_Δ        F_Δ = ½(F_X1 - F_X2)
τ_Θ_aug = g(Y) · z
```

where τ(Y) is a Y-dependent time constant, k(Y) scales the input amplitude, and g(Y) is
the Y-dependent gain from z into the Θ channel.

**Physical interpretation**: z represents any internal variable in the F_Δ to Θ signal path
that introduces a dynamic lag not present in the rigid-body model. Physically, this could be:
- Cable/dresspack relaxation (Candidate B structure, Section 5.6)
- First-order actuator force-path dynamics (Candidate D from the second analysis document)
- Bearing compliance introducing a damped internal torque

The ODE structure is identical across all three physical interpretations. The simulation study
demonstrates that the augmentation framework can recover a state of this structure; the
specific physical source is secondary.

**Why chosen**: see Section 6.

---

## 6. Why the First-Order F_Δ State Is the Right Choice

| Criterion | Satisfied? | How |
|---|---|---|
| Adds a genuine dynamic state | Yes | z has its own first-order ODE and carries memory |
| Structurally absent from baseline | Yes | Adds a pole to the F_Δ to Θ path; baseline has no dynamic in this path |
| Observable in [X1, X2, Y] | Yes | z drives Θ (yaw) which is directly the X1-X2 difference |
| Y-dependent coupling | Yes | τ(Y), k(Y), g(Y) all vary with Y |
| Cross-axis coupling | Yes | X-forces (F_Δ) drive z; z affects Θ |
| Continuously excited | Yes | F_Δ is nonzero whenever the yaw controller corrects, i.e. throughout tracking |
| Identifiable from realistic data | Yes | No special excitation needed; normal pick-and-place trajectory suffices |
| Non-absorbable by baseline | Partially | See analysis below |
| Verifiable in simulation | Yes | True z(t) known from data generator |
| Analogy with Jan's MSD | Yes | Extra state with ODE, absent from baseline, couples into primary dynamics |

**Non-absorbability analysis**: at low frequencies (well below 1/(2πτ)), z approaches the
quasi-static value k·τ·F_Δ. This proportional contribution to τ_Θ looks like a gain on
F_Δ in the Θ equation, which the baseline can partially absorb by adjusting the B matrix
(M(Y) entries). However, two aspects are genuinely non-absorbable:

1. **The phase shift near f = 1/(2πτ)**: the lag introduces a frequency-dependent phase
   delay in the F_Δ to Θ response. Constant M(Y), C, K produce no such lag (their response
   is algebraically related to the current state, not to a filtered version of the input).
2. **The Y-dependence of τ(Y) and g(Y)**: the phase shift moves with Y. Constant C and K
   cannot reproduce a gain or phase that varies with Y. This is the same structural gap
   that M(Y) addresses for inertia — our extra state adds the same kind of Y-dependence
   to the dynamic F_Δ to Θ path.

**Why this beats Dahl [z1, z2] on identifiability**:

| Property | Dahl [z1, z2] | First-order z |
|---|---|---|
| Driving signal | Ẋ1, Ẋ2 velocity | F_Δ = differential X-force |
| When active | Only during direction reversals | Whenever F_X1 ≠ F_X2 (all of tracking) |
| Required excitation | Micro-reversals at ±0.001 m/s, 4 µm strokes | Standard point-to-point or diagonal moves |
| Fc in gantry context | 1.4-2.75 N (Lan Jia 2023) | g(Y) is a free parameter chosen for visibility |
| Effect in closed-loop | Suppressed by controller | F_Δ is generated by the controller; z follows it |
| States required | 2 | 1 |

**Connection to existing decisions**:
- D-022 (non-baseline physics in augmentation): z does NOT go in the baseline. It is in
  the data-generating model only. The augmentation must discover it.
- D-024 (resonance first, Coriolis second): this first-order state simulation study is the
  first augmentation demonstration.
- D-025 (hysteresis, sign(dY/dt)): z provides the dynamic version of what D-025 describes
  statically. The Y-dependence of g(Y) is a direct analogue.

---

## 7. Observability Principle — Verified Against Jan's Code

**This section documents the precise meaning of "observable in [X1, X2, Y]" and why it is
a requirement on the DATA GENERATOR, not on the augmented model architecture.**

### 7.1 What "observable" means

The augmentation framework trains by minimising:

```
loss = MSE(y_measured, y_predicted)
```

where y_measured comes from the data generator (which includes z), and y_predicted comes
from the augmented model (baseline + ANN). For the augmentation to **learn z**, y_measured
must carry information about z. If z in the data generator never affects y_measured, then
y_measured would be identical whether z exists or not. No gradient descent procedure can
recover z from data that contains no information about z.

### 7.2 Verification against Jan's code

Inspected `model_augmentation/systems/mass_spring_damper.py`:

```python
def h(self, x, u):
    y = x[0::2]           # all mass positions
    y = y[self.output_ix] # output_ix = [1] -> selects x_m2 only
    return y
```

**m3 is NOT in the output.** Only m2's position is measured. Yet the augmentation
successfully recovers m3's dynamics.

The reason is in `deriv` — m3 drives m2 via the spring force:

```python
Fi_ = self.k[i+1]*di_ + ...   # spring force from m3 on m2
dx[2*i+1] = (-F_i + Fi_ + u[i]) / self.m[i]   # m2 acceleration depends on m3
```

The chain is: `m3 position -> spring force -> m2 acceleration -> m2 position -> output`.
m3 is not the output but its effect propagates through the physics to the output. y_measured
therefore contains the imprint of m3.

### 7.3 Gradient path in the augmented model

In `interconnect.py`, the output block is wired as:

```python
interconnect.connect_signals("x", output_block, "concat",
    selection_matrix(np.array([0, 1, 2, 3]), nxd))  # FP states only
```

The augmentation states [4,5] do NOT directly enter the output block. Mathematically,
d(yhat)/d(z_aug) = 0 through the direct path.

The ANN block sees the full state vector and contributes additively to all of xp:

```python
interconnect.connect_block_signals(ANN_state_block, ["x", "u"], ["xp"])
```

A mathematical gradient path exists from loss to z_aug through the ANN (loss → yhat →
FP states → ANN corrections to FP states → z_aug as ANN input). However, this gradient
only points toward meaningful z-like dynamics if y_measured contains information about z.
Without that signal in the data, the gradient pushes z_aug toward zero rather than toward
the true state trajectory.

### 7.4 Summary

"Observable in [X1, X2, Y]" means: **the extra state in the data-generating model must
physically propagate its effect to the measured position outputs**. This is a requirement
on the data-generator physics. For our chosen first-order state:

```
z -> τ_Θ_aug = g(Y)·z -> Θ̈ (extra torque) -> Θ trajectory -> X1-X2
```

X1 = X + Θ·Lb/2 and X2 = X - Θ·Lb/2 are both measured. The z-induced Θ error appears
in both X1 and X2, most clearly in their difference X1-X2. y_measured carries the imprint
of z throughout the tracking trajectory (not only at special events).

---

## 8. How Adding the State Changes the Model

**Baseline force equation for Θ (what the LPV model predicts)**:

```
τ_Θ = (F1 - F2)·Lb/2 - (cg1-cg2)·Lb/2·Ẋ - (cc1-cc2)·Lb/2·sign(Ẋ)
```

The F_Δ to Θ path is a direct algebraic mapping through M(Y)^{-1}·B. There is no dynamic
element in this path — the response of Θ to F_Δ has no lag.

**Data-generating model force equation with z**:

```
τ_Θ = (F1 - F2)·Lb/2 - (cg1-cg2)·Lb/2·Ẋ + g(Y)·z
```

where z is driven by F_Δ through a first-order lag:

```
ż = -(1/τ(Y)) z + k(Y) · F_Δ
```

The Θ response to F_Δ now has a dynamic component. At frequencies near 1/(2πτ), the
contribution of F_Δ through z introduces a phase-shifted and amplitude-modified torque
on top of the direct baseline contribution. The baseline cannot reproduce this regardless
of M(Y), C, K tuning — it has no mechanism to introduce frequency-dependent behaviour in
the F_Δ to Θ path.

**Extended state vector**:

```
Baseline (6 states):     x  = [X,  Θ,  Y,  Ẋ,  Θ̇,  Ẏ]
Data generator (7 states): x̄ = [X,  Θ,  Y,  Ẋ,  Θ̇,  Ẏ,  z]

Dynamics of extra state:
  ż = -(1/τ(Y)) z + k(Y) · F_Δ     F_Δ = ½(F_X1 - F_X2)
```

The extra state z starts at zero and evolves through the differential X-force history.
In the augmented model, the ANN must learn to maintain a state that behaves like z and
couple it into the Θ equation with Y-dependent amplitude g(Y).

---

## 9. Observability Analysis for the Chosen State

### 9.1 Signal chain

```
F_Δ(t) nonzero -> z evolves as filtered F_Δ -> extra torque g(Y)·z on Θ
               -> Θ trajectory differs from baseline prediction
               -> X1-X2 = Θ·Lb contains the residual throughout the trajectory
```

### 9.2 When is the effect active?

F_Δ = ½(F_X1 - F_X2) is nonzero whenever the two X-actuators apply different forces. In
closed-loop tracking this occurs whenever:
- The yaw controller produces a differential correction (synchronisation errors)
- Diagonal moves with simultaneous X and Y motion (the controller generates differential
  force to maintain yaw stability)
- Any asymmetric disturbance causes a transient F_Δ ≠ 0

Unlike Dahl friction, the effect is not conditional on velocity reversals. It is present
throughout any realistic tracking trajectory.

### 9.3 Parameter design for visibility

| Parameter | Constraint | Reason |
|---|---|---|
| τ(Y) | 5-20 ms range | Phase shift at 8-32 Hz; within gantry tracking bandwidth |
| g(Y) | Chosen so max τ_Θ_aug ~ 5-10% of baseline Θ torques | Visible but not dominant |
| Y-dependence | Monotone: e.g. τ(Y) = τ_0 + τ_1·Y | Simple, interpretable, verifiable |
| k(Y) | Scale F_Δ input appropriately | Controls steady-state z amplitude |

### 9.4 Partial absorbability and mitigation

At frequencies well below 1/(2πτ), z ≈ k·τ·F_Δ (quasi-static), and the contribution
looks like a gain on F_Δ in the Θ equation. The baseline optimizer may inflate B matrix
entries to absorb this quasi-static component. The residual left for the augmentation is
the dynamic part: the phase shift and the Y-dependent amplitude variation.

Mitigation: choose τ(Y) so that the lag frequency falls within the bandwidth of the
training trajectory's F_Δ spectral content. A trajectory with aggressive diagonal moves
(high acceleration, generating large F_Δ transients) at multiple Y-positions provides the
richest training signal.

### 9.5 Comparison to Dahl observability

The Dahl pre-sliding transient requires the velocity to pass through zero slowly
(±0.001 m/s over 4 µm strokes per Lan Jia 2023). The first-order z requires only that
F_Δ be nonzero and vary — which is true during any closed-loop trajectory with yaw
corrections. The training signal from z is broadband and continuous; the training signal
from Dahl is narrow-band and episodic.

---

## 10. Simulation Study Setup

| Component | Description |
|---|---|
| **Data generator** | 7-state model: 6-state LPV + first-order internal state z |
| **Baseline** | 6-state LPV (Garcia-Herreros rigid model, constant C, K) |
| **Augmentation type** | Dynamic augmentation (Jan's framework): augmented state x_hat = [x_baseline; z] |
| **Augmentation adds** | 1 extra state; ANN learns z dynamics (xp[6]) AND z coupling into Θ (xp[4]) |
| **Inputs** | F_X1, F_X2, F_Y (same as baseline) |
| **Outputs** | X1, X2, Y (same as baseline; measured in stage coordinates) |
| **Scheduling variable** | p = Y (unchanged — preserved in augmented model) |
| **Verification** | True z(t) from data generator; compare to augmentation state trajectory |
| **Key metric** | Θ prediction error as function of Y-position and F_Δ magnitude |

**What the augmentation must discover**:
1. That there is one extra state with a first-order ODE driven by F_Δ
2. That this state couples into Θ with a Y-dependent amplitude g(Y)
3. That the time constant τ(Y) itself varies with Y (demonstrating LPV recovery)

**Excitation design for training data**:
- Diagonal moves at multiple Y-positions to generate large, varied F_Δ
- Include segments at different Y-offsets (e.g. Y = -0.3, 0.0, +0.3 m) to make
  the Y-dependence of τ(Y) and g(Y) learnable
- Avoid trajectories that keep F_Δ ≈ 0 throughout (pure X translation, no yaw correction)

---

## 11. Critical Comparison: How Similar Is This to What Jan Did?

| Criterion | Jan (MSD, m3) | Our proposal (first-order z) |
|---|---|---|
| Type of extra state | Macroscopic mechanical DOF | First-order internal dynamic |
| Physical object | Mass block m3 | Internal lag state (cable, actuator, or bearing) |
| Why absent from baseline | 3-DOF simplified to 2-DOF | Rigid-body model has no dynamic in F_Δ to Θ path |
| ODE | Newton's law for m3 | First-order lag: ż = -(1/τ) z + k·F_Δ |
| Coupling mechanism | Nonlinear spring force on m2 | Lagged torque on Θ with Y-dependent gain |
| Nonlinear? | Yes (nonlinear spring k2) | Nonlinear through Y-dependence of τ(Y), g(Y) |
| Y-dependent coupling | No | Yes — τ(Y), g(Y) vary with Y |
| States in data generator | 2 (x3, ẋ3) | 1 (z) |
| Augmentation type | Dynamic (state vector enlarged) | Dynamic (state vector enlarged) |
| States added by augmentation | 2 | 1 (or 2 if second-order escalation needed) |
| Verification method | Compare learned x3(t), ẋ3(t) to truth | Compare learned z(t) to truth |

**Note on Static_ANN_Block**: the ANN block class is stateless (feedforward), but the overall
model is dynamic because the interconnect state x_hat = [x_baseline; z] persists across
time steps. The memory lives in the state vector, not inside the block. This matches Jan's
setup exactly — `dynamic_aug = True`, `nxd = 2*dof` in his interconnect script.

**Is the difference from Jan a problem?** Formally no. The mathematical structure is:
extra state with its own ODE, absent from the baseline, coupling back into the primary
equations. Whether the hidden state is a mass or an internal lag variable, the augmentation
framework faces the same challenge: discover the governing ODE and the coupling. The
difference is in physical interpretation, not in mathematical structure.

**What this study proves**: the LFR augmentation framework can learn Y-dependent,
continuously-active dynamic coupling that the LPV baseline structurally cannot represent —
specifically, a dynamic lag in the differential-force to yaw channel with Y-dependent
parameters. This is the class of effects that limits gantry synchronisation performance
in practice.

---

## 12. Literature and Decision Cross-References

### Primary literature (verified sources)

| Source | Relevance |
|---|---|
| Garcia-Herreros 2013, Section 2.4 | Lists unmodelled phenomena; cross-arm bending called negligible; 37.7 Hz base resonance is die-cast specific |
| Telica ASME-YGNN-08-0750-0800W3 | Granite frame; ball bearings; integrated cables and tubes confirmed |
| Lan Jia 2023, MSc thesis TU Delft (Prodrive) | Gantry stage friction identification; Fc = 1.4-2.75 N, Fs = 1.9-3.2 N, measured at 9 positions; position-dependent; requires open-loop micro-reversal excitation |
| Olsson et al. 1998, EJC review | Standard Dahl/LuGre state form confirmed; ż = v - (σ|v|/Fc)z, F = σz |
| Tanaka et al. 2009, IJAT | Linear ball guideway; Fc = 10 N, Fs = 12 N; pre-sliding NSB at ~10 µm |
| Swevers et al. 2000, IEEE TAC | Normal-force/position scaling of friction confirmed; supports Fc(Y) formulation |
| Hoekstra 2025 (Jan's paper) | Section 4: MSD simulation example — the model analogy for our study |

### ChatGPT analysis documents (unverified synthesis; claims require primary source verification)

| Document | Key claims used in this document |
|---|---|
| "Literatuuronderzoek voor Dahl-frictiestaten..." (first document) | Fc = 1.4-2.75 N in gantry context (attributed to Lan Jia); micro-reversal excitation requirement (4 µm, ±0.001 m/s, 50 repetitions); closed-loop friction identification difficulty |
| "Extra dynamische toestand(en) voor grey-box modelaugmentatie..." (second document) | Four candidate states (A-D) with ODEs and coupling; designed demonstration state template (first-order F_Δ-driven); dual-drive resonances attributed to guide bearings; cable/dresspack as viable one-state candidate |

**Note**: Claims from the ChatGPT documents are used for design motivation only. The
primary sources (Lan Jia, Olsson, Tanaka) should be consulted directly before citing any
of these claims in formal writing.

### Decision cross-references

| Decision | Relevance |
|---|---|
| D-022 | Extra states go in data generator, not baseline |
| D-023 | Validate parameter estimation on synthetic data before augmentation |
| D-024 | Resonance first, Coriolis second — this first-order state study is the first demonstration |
| D-025 | sign(dY/dt) for hysteresis (static) — z is the dynamic version with Y-dependent parameters |
