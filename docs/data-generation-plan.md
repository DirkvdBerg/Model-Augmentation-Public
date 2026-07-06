# Data Generation Plan

**Status:** Plan document. Describes current implementation, identified gaps, and the
improved dataset design for ANN augmentation training.

**Script under discussion:** `Matlab-scripts/Augmentation/data/generate_oscillatory_multisine_data.m`

**Related theory docs:**
- Modal coordinate multisine design: `docs/trajectory-plus-multisine-design.md`
- FIM-based frequency selection: `docs/experiment-design-pipeline.md`
- Fixed-Y FRF pretest: `docs/frozen-y-mimo-frf-pretest.md`

---

## 1. What we currently have

### Trajectory set (14 experiments total)

| ID | Split | Type | Y coverage | X excitation |
|----|-------|------|-----------|--------------|
| T1 | train | osc | Y sweep 0 -> 0.30 m, f=0.3 Hz | none |
| T2 | train | osc | Y sweep 0 -> 0.20 m, f=0.2 Hz | X_sym 1.5 Hz |
| T3 | train | osc | Y fixed at 0 | X_sym 2.0 Hz |
| T4 | train | osc | Y fixed at 0.30 m | X_sym 2.0 Hz |
| T5 | train | osc | Y sweep 0 -> 0.25 m, f=0.3 Hz | X_anti 0.8 Hz |
| T6 | train | osc | Y sweep 0 -> 0.30 m, f=0.7 Hz | X_sym 1.3 Hz |
| T7 | train | osc | Y sweep 0 -> 0.20 m, f=0.4 Hz | X_sym 1.5 Hz + X_anti 0.8 Hz |
| T8 | train | osc | Y sweep 0 -> 0.25 m, f=0.2 Hz | X_sym multi-amplitude |
| T9 | train | p2p | Y sweep 0.3 -> -0.3 m, repeated | none |
| T10 | train | p2p | Y sweep 0.2 -> -0.2 m, repeated | X_sym + X_anti |
| V1 | val | osc | Y sweep 0.25 -> 0.15 m | X_sym 1.8 Hz |
| V2 | val | p2p | Y sweep 0.30 -> 0.10 m | X_sym |
| E1 | test | osc | Y sweep 0.15 -> 0.05 m | X_sym 1.1 Hz + X_anti |
| E2 | test | p2p | Y sweep 0.20 -> 0.00 m | X_anti only |

All oscillatory trajectories use single-frequency sinusoidal position references.

### Multisine design (current)

- Band: narrowband [130, 180] Hz (targets MSD resonance at fa = 150 Hz)
- Generation: random-phase multisine, 200 candidates per split
- Selection criterion: lowest crest factor over candidates
- Per-channel design: independent random phase seeds for F_X1, F_X2, F_Y
- Per-split independence: different seeds for train, val, test

### Hardware amplitude scaling (current)

- Active channel: RMS = 5% of trajectory-only RMS force for that channel
- Inactive channel: floor at 5% of the dominant active channel amplitude (narrowband heuristic)
- Validation: peak force and RMS checked against ETEL hardware limits

---

## 2. Identified gaps

### Gap 1: Two separate identification problems with incompatible frequency requirements

The system has two distinct dynamics targets for the augmentation:

**Structural modes at 2 Hz and 5 Hz.** These are the dominant gantry dynamics: the
common X mode and Y mode. The FP model captures these in its backbone. The augmentation
residual at these frequencies is small but non-zero (coupling effects, Y-dependent
inertia mismatch, friction). The encoder must reconstruct the 6 physical states, which
live primarily in the [1, 10] Hz range.

**MSD resonance/anti-resonance at 157 Hz.** This is the primary target of the ANN
augmentation: the hidden mass-spring-damper on the payload is not in the FP model and
must be learned entirely from data. The narrowband multisine [130, 180] Hz directly
targets this. It is appropriate for MSD identification.

The gap is not that the MSD is unexcited -- the narrowband multisine is correct for that.
The gaps are:

1. **The structural mode range [1, 10] Hz is PE(2) only.** The only signal there comes
from the sinusoidal trajectory references. A single sinusoid is PE(2), which identifies
at most one mode. The 6-state physical system requires the data to be persistently
exciting of order >= 2n = 12 per channel (Ljung 1999, Theorem 13.1) for the encoder
to reconstruct all states reliably.

2. **The mid-range [10, 130] Hz is completely unexcited.** If any model error or
coupling effect exists in this range, the ANN will never see it in training.

3. **Switching naively to broadband [1, 200] Hz would dilute MSD identification.**
A broadband multisine at the same total RMS spreads power across 199 Hz instead of
50 Hz, reducing power spectral density at 157 Hz by roughly a factor of 4. For a
lightly damped MSD (zeta = 0.05, Q approx 10), the resonance response is narrow and
requires concentrated excitation power near 157 Hz. A flat broadband spectrum may not
provide enough SNR at the resonance.

The proposed solution (see Section 3) is therefore not a simple band switch but a
structured combination: broadband excitation for the structural modes and encoder
identifiability, plus retained narrowband power near the MSD resonance.

### Gap 2: Sinusoidal trajectories trace ellipses in state space

Each oscillatory trajectory generates a closed 1D orbit in the 6D state space. The ANN
learns the residual map f(x, u) -> delta_x. For this map to generalize, the training data
must cover the (x, u) manifold relevant to test conditions. Single-frequency sinusoids at
different frequencies give differently-shaped ellipses but all remain 1D manifolds. The
ANN sees no off-ellipse states during training, which are the states that appear during
transients, irregular control, and real operation.

### Gap 3: Multisine designed in stage coordinates mixes physical modes

The current design generates independent F_X1, F_X2, F_Y signals. Any single realization
of F_X1 and F_X2 is a random superposition of the symmetric mode (X1+X2)/2 and the
anti-symmetric mode (X1-X2)/2. In any given experiment, the power split between these
two modes is random: the ANN sees uneven coverage of the two physical modes across the
dataset. Additionally, the yaw amplitude (proportional to |F_X1 - F_X2|) is
uncontrolled and can accidentally approach the hardware limit |X1-X2| <= 6 mm.

### Gap 4: No systematic local LPV coverage

The scheduling variable is Y in [-0.30, +0.30] m. The current trajectories visit different
Y values, but not systematically. T3 and T4 are the only fixed-Y experiments, and both
use a single X_sym sinusoid at 2 Hz. There is no broadband excitation at a controlled
fixed Y. The local LPV identification approach (Ghosh et al. 2018, Automatica) requires
informative excitation at each scheduling operating point. Currently, the Y-coverage is
incidental to the motion design, not a design requirement.

---

## 3. Proposed improvements

### Improvement A: Shaped dual-band multisine

Gap 1 in Section 2 shows that a naive band switch from narrowband [130, 180] Hz to
broadband [1, 200] Hz would solve the [1, 10] Hz PE problem but would dilute MSD
identification. The MSD resonance is narrow (zeta = 0.05, Q approx 10) and requires
concentrated power at 157 Hz. A flat broadband at the same total RMS spreads power
across 199 Hz instead of 50 Hz, reducing spectral density near 157 Hz by a factor of
approximately 4.

Two options:

**Option A1 -- Summed dual-band multisine.** Generate two independent multisines and
sum them before injection:

```
f_broadband  = multisine over [1, 10] Hz  (structural modes, encoder identifiability)
f_narrowband = multisine over [130, 180] Hz  (MSD at 157 Hz, keep current approach)
f_total = f_broadband + f_narrowband
```

Each band has its own amplitude set independently. The broadband amplitude is sized
from the structural-mode trajectory RMS (targets the encoder problem). The narrowband
amplitude is kept at the current level (targets MSD identification). Hardware limits
are checked on f_total after summing.

**Option A2 -- Amplitude-shaped broadband.** Use a single broadband multisine [1, 200]
Hz but with frequency-shaped amplitude: flat in [1, 10] Hz, boosted by a factor of ~4
near [130, 180] Hz to compensate for the wider band. This preserves MSD SNR while
filling the mid-range gap.

**Recommendation:** Option A1 is simpler to reason about and implement. Each band's
amplitude is set from its own physical criterion (structural modes vs. MSD). The total
force budget check catches any hardware limit violation after summing.

Note: the [10, 130] Hz mid-range gap remains unexcited in Option A1. Whether this
matters depends on whether the augmentation residual has significant content in that
range. If the FP model is accurate from 10 to 130 Hz, the gap is benign.

### Improvement B: Multisine design in logical (modal) coordinates

See Section 4 for the full justification. The change:

Instead of generating independent multisines for F_X1, F_X2, F_Y and minimizing crest
factor in stage coordinates, generate three independent multisines in logical (modal)
coordinates:

```
F_sym   -- excites the symmetric X mode (common motion X1 = X2)
F_anti  -- excites the anti-symmetric X mode (yaw, X1 = -X2)
F_Y     -- excites the Y axis
```

Transform to physical actuator forces before simulation:

```
F_X1 = F_sym + F_anti
F_X2 = F_sym - F_anti
F_Y  = F_Y
```

Crest factor minimization is done on the physical forces [F_X1, F_X2, F_Y] after the
transform, following `docs/trajectory-plus-multisine-design.md` Section "MIMO And Mode
Design".

### Improvement C: Fixed-Y local LPV experiments

Add experiments where Y is held at fixed operating points and the only excitation is the
broadband force multisine in logical coordinates. Proposed grid:

```
Y_fixed in {-0.30, -0.15, 0.00, +0.15, +0.30} m
X_sym reference = 0, X_anti reference = 0
```

At each Y position, run three independent mode experiments:
- F_sym only (common X)
- F_anti only (differential X/yaw)
- F_Y only

This gives a clean FRF at each operating point (consistent with the frozen-Y pretest
design in `docs/frozen-y-mimo-frf-pretest.md`) and locally informative ANN training
data at each scheduling point.

These experiments directly implement the local LPV identification approach and can be
justified with Ghosh et al. (2018), "Optimal identification experiment design for LPV
systems using the local approach," Automatica 87:258-266.

### Improvement D: Chirp position reference

Add one or two trajectories where the position reference is a frequency-swept (chirp)
signal instead of a fixed-frequency sinusoid. A log-chirp from 0.1 to 5 Hz over 10 s
sweeps through all gantry structural modes and gives PE of high order in the
position/velocity state space. This complements the force multisine: the multisine
provides frequency richness in force-to-output path; the chirp provides frequency
richness in the reference-to-state path.

---

## 4. Why logical coordinates for the multisine

This section explains why designing the multisine in modal (logical) coordinates is
physically better than designing in stage coordinates.

### The physical modes of the gantry

The gantry has three natural modes, defined by how the mass matrix M(Y) decouples:

| Mode | Physical motion | Actuator direction |
|------|----------------|-------------------|
| Symmetric (common X) | X1 and X2 move together | F_X1 = F_X2 |
| Anti-symmetric (yaw) | X1 and X2 move oppositely | F_X1 = -F_X2 |
| Y axis | Y moves independently | F_Y only |

The P-transform used throughout the FP model maps between stage coordinates [X1, X2, Y]
and logical coordinates [X_sym, X_anti, Y]. The same transform applies to forces.

### Why stage coordinates cause problems

When independent random-phase multisines are generated for F_X1 and F_X2, each actuator
force is a sum of symmetric and anti-symmetric components:

```
F_X1(t) = ms_1(t)   (independent random signal)
F_X2(t) = ms_2(t)   (independent random signal)

Symmetric component:  (ms_1 + ms_2) / 2  -- uncontrolled amplitude
Anti-symmetric component: (ms_1 - ms_2) / 2  -- uncontrolled amplitude
```

In any single realization, the power split between the symmetric and anti-symmetric modes
is random. Two problems follow:

**Problem 1: Uneven mode coverage across the dataset.** If one realization happens to put
most power in the symmetric mode, the ANN training gets little information about the
anti-symmetric dynamics in that experiment. Another realization may flip this. The ANN
sees inconsistent mode emphasis, which means some mode-specific residuals are
underrepresented in the effective gradient signal.

**Problem 2: Uncontrolled yaw amplitude.** The anti-symmetric component (F_X1 - F_X2)/2
drives the yaw mode, which is constrained by the hardware limit |X1 - X2| <= 6 mm
(lim.diff in the MATLAB script). In stage coordinates, the yaw excitation is an
uncontrolled random quantity. Its amplitude is bounded only by the overall crest factor
check, which operates in stage coordinates and cannot separately control the yaw content.
In a worst-case realization, the anti-symmetric component can be much larger than
intended, risking the yaw limit.

### Why logical coordinates solve these problems

Designing in logical coordinates gives direct control over each mode's excitation:

```
F_sym(t)   -- set amplitude explicitly for the symmetric mode
F_anti(t)  -- set amplitude explicitly for the anti-symmetric mode (yaw)
F_Y(t)     -- set amplitude explicitly for the Y mode
```

Then transform:

```
F_X1 = F_sym + F_anti
F_X2 = F_sym - F_anti
```

The amplitude of F_anti is a design choice, not a random variable. It can be set to keep
|X1 - X2| comfortably within the 6 mm hardware limit. The amplitude of F_sym is set
independently to excite the common X mode at the desired level.

**Information matrix conditioning:** When each modal channel is independently excited,
the MIMO information matrix in modal coordinates is closer to diagonal. The identification
of each mode is decoupled: the symmetric mode is identified from (F_sym, X_sym) data,
and the anti-symmetric mode from (F_anti, X_anti) data. In stage coordinates, both modes
appear in both (F_X1, X1) and (F_X2, X2) data simultaneously, coupling the information
matrix and requiring more data to achieve the same parameter variance.

**No ANN coordinate consistency argument.** Verified from `model_augmentation/fit_systems/blocks.py`
line 783: `Gantry_State_Block.deriv()` receives stage forces and applies `P_mat`
internally. The ANN block receives `["x", "u"]` from the interconnect where `u` is the
global normalized input in stage coordinates (F_X1, F_X2, F_Y). The state `x` is in
logical coordinates. So the ANN input is mixed (x_logical, u_stage). Designing the
multisine in logical coordinates does not change which coordinate system the ANN sees.
The mode-decoupling and yaw-control arguments above are the only valid justifications
for the logical coordinate design.

### Note on the coordinate system lessons.md rule

The existing rule in `tasks/lessons.md` states: "coordinate system choice is driven by
data, not model structure." That rule applies to the model representation (which
coordinate system to use for the state-space matrices). It does not apply here. The
argument for logical coordinates in excitation design is not about model structure: it is
about physical mode decoupling and information matrix conditioning, which are properties
of the physical system. The same argument would apply regardless of which coordinate
system is used for the model.

---

## 5. Summary of current vs. proposed dataset

| Property | Current | Proposed |
|----------|---------|---------|
| Frequency band | [130, 180] Hz only | [1, 200] Hz broadband |
| PE order per channel | PE(2) in [1-10 Hz] + PE(~100) in [130-180 Hz] | PE(>>12) from 1 Hz to 200 Hz |
| Multisine coordinate system | Stage (F_X1, F_X2, F_Y) | Logical (F_sym, F_anti, F_Y) |
| Yaw amplitude control | Random (uncontrolled) | Explicit (F_anti amplitude set directly) |
| LPV scheduling coverage | Incidental from trajectory geometry | Systematic: fixed-Y grid + broadband per point |
| State-space coverage | 1D ellipses in 6D state space | Ellipses + broadband trajectory variants (chirp) |
| Train/val/test independence | Different phase seeds per split | Unchanged (keep) |
| Crest factor minimization | 200 candidates, physical coordinates | 200 candidates, physical coordinates after modal transform |

## 6. Implementation priority

1. Switch to broadband [1, 200] Hz (parameter change only, highest return)
2. Redesign multisine in logical coordinates (modify `generate_cached_multisine` call
   and add F_sym/F_anti/F_Y -> F_X1/F_X2/F_Y transform)
3. Add fixed-Y local LPV experiments (new trajectory entries)
4. Add chirp position reference (new trajectory type function)
