# Theory Validation Checklist

**Purpose:** Single reference for every theory claim in the experiment design pipeline.
Organized in two parts:
1. **Pipeline steps** — per-choice breakdown of what needs a source before implementation
2. **Theory reference** — the underlying claims with source locations for you to verify

**Rule:** Every choice marked `HEURISTIC` or `NO SOURCE` must be declared as such to
supervisors. Do not present it as theory-backed. Every choice marked `THEORY` must have
its source verified (equation number confirmed) before it goes into code or thesis.

**Your task:** work through Part 2 and mark `[ ]` → `[x]` once you have verified the
claim against the actual source. Then check Part 1 to confirm all implementation choices
are covered.

**Source files:**
- `docs/experiment-design-pipeline.md`
- `docs/experiment-design-closed-loop.md`
- `docs/diagnostics-theory-basis.md`
- `literature/experiment-design/deep-research/closed-loop-multisine-gpt-research.md`
- `literature/experiment-design/deep-research/augmentation-nonparametric-gpt-research.md`
- `literature/experiment-design/deep-research/samplingrate-and-segmentlength-gpt-research.md`
- `literature/experiment-design/deep-research/segment-length-multisine-preanalysis.md`

---

# Current Assumptions and Required Extensibility

> Read this before implementing anything.

**What we are designing for now:**
- Noiseless simulation (no measurement noise)
- Complete parametric model (no unmodeled dynamics, augmentation network not yet fitted)
- Open-loop BPTT training with replayed plant input `u_total`
- **Active input design: resonance/bandwidth-weighted broadband odd multisine** (Lecture 9 slide 13)

**What must be possible to extend to later (without redesigning from scratch):**
- Hardware with measurement noise → switch to indirect closed-loop FRF estimator
  (`Ĝ = Φ_yd / Φ_ud`) instead of direct `Ĝ = Ŷ/Û`; amplitude weighting may be updated
- Augmentation active → rerun Step 0 with augmented model (new poles, different S(jω));
  switch to broadband uniform per D-050 Phase 2 criterion
- Multiple operating points / scheduling → broadband uniform covers full operating range;
  FIM aggregation over Y deferred to G12

**Consequence for every design choice:** if a choice only works in the noiseless / no-augmentation
case and cannot be extended, it must be flagged explicitly. Do not implement anything that
paints us into a corner.

**On FRF estimation: parametric vs nonparametric**
In noiseless simulation, the analytical FRF `G(jω) = C(jωI−A_c)^{-1}B_c` and a
nonparametric FRF estimated from data give identical results — the model IS the truth.
Computing analytically is therefore preferred (no pre-run simulation needed).
On hardware, a nonparametric FRF requires the **indirect closed-loop estimator**
(`Φ_yd/Φ_ud`) because the direct estimator (`Ŷ/Û`) is biased in closed loop.
The indirect estimator is not needed now but the preanalysis structure must be compatible
with plugging in a measured FRF later in place of the analytical one.

**On reference vs force injection**
This is context-dependent — both answers from the literature are correct, but for
different identification paradigms:
- **Nonparametric / PEM identification**: reference injection is preferred (P&S,
  Gevers) because `|T| ≈ 1` inside bandwidth delivers excitation well.
- **BPTT with u-replay (our method)**: force injection is required. With reference
  injection the controller absorbs the multisine (`u_recorded ≈ u_fb` only), so
  replaying `u_recorded` through our model carries no multisine excitation — gradients
  vanish. This is why our supervisor's recommendation is correct for our specific
  identification method, and why the literature recommendation does not apply directly.
  Documented in D-048 and `docs/ref-injection-openloop-incompatibility.md`.

---

# PART 1 — Pipeline: Per-Step Theory Requirements

> These are the three steps that must be implemented and justified before any simulation
> results are shown to supervisors. Every row is a design choice with its source.
> Choices marked **NO SOURCE** are the dangerous ones — they must be either replaced
> with a theory-backed alternative or explicitly declared as engineering heuristics.

---

## STEP 0 — Pre-analysis: `diagnostics_system.m`

**What it does:** Runs a short broadband probe through the closed-loop simulation to
empirically estimate the sensitivity survival profile `|Ŝ(jω)|²`. Outputs f_low, f_high,
τ_max, and the survival profile used to shape Step 1 amplitude weighting.

> **Why simulation-based (not analytical):** In the current parametric model, empirical
> Ŝ and analytical S are identical — equivalent results. But when the model becomes
> incomplete (augmentation added, or hardware), the nominal analytical S(jω) diverges
> from the true survival profile. Running from simulation data means the same code works
> at every stage without modification.

**Procedure:**
1. Inject a flat broadband probe multisine (all odd harmonics from f_probe_low to f_probe_high,
   equal amplitude) as force injection — **probe signal design is itself a gap: frequency
   range, amplitude, and number of averaging periods must be specified before implementation**
2. Record `f_sim(t)` and `u_total(t)` from the closed-loop simulation
3. Estimate `Ŝ(jω) = FFT(u_total) / FFT(f_sim)` — empirical survival profile
4. Derive f_low = lowest frequency where `|Ŝ(jω)|²` is meaningfully above zero
5. Derive τ_max from the dominant pole of the open-loop plant state matrix A_c
   (A_c is the plant, not closed-loop, matrix; BPTT replays u open-loop through the plant)
6. Use `|Ŝ(jω)|` profile as input to Step 1 amplitude weighting —
   **formula for A_k as a function of |Ŝ| must be specified before implementation (see gap below)**

**Future (augmentation active):** re-run Step 0 with augmented closed-loop — Ŝ reflects
new dynamics automatically. No code change.
**Future (hardware):** replace simulation run with measured `f_sim` and `u_total` from
the real system. Same estimation formula. No code change.

| Choice | Type | Source to verify | Flag |
|--------|------|-----------------|------|
| `Ŝ(jω) = FFT(u_total) / FFT(f_sim)` as empirical survival | THEORY | Feedback algebra: with force injection and no multisine in reference r, at each injected frequency ω_k: U_total = S×F_sim exactly (derivation: Y = G×S×F_sim, U_fb = −C×Y = −CGS×F_sim, U_total = U_fb + F_sim = F_sim(1−CGS) = S×F_sim). D-048; any feedback control textbook, e.g. Skogestad & Postlethwaite (2005) Ch.2. **Note: u_fb DOES contain the multisine — the ratio is S because u_fb + f_sim = S×f_sim, not because u_fb = 0** | **Verify derivation. Single-period FFT exact in noiseless sim; need period averaging on hardware** |
| f_low from lowest frequency where `\|Ŝ\|²` meaningful | HEURISTIC | No universal threshold — engineering choice. Declare threshold value explicitly to supervisors. **Check for general range guidance:** Lecture 9 (`literature/experiment-design/System-identification/Lecture 9.pdf`) slides on bandwidth selection; Lecture 13 on experiment design bounds | **NO SOURCE for threshold — must declare value and basis. Look in lectures first.** |
| f_high from controller bandwidth | Engineering constraint | Controller design spec — no theory needed | OK |
| `τ_max` from dominant pole of open-loop plant matrix A_c | THEORY | BPTT replays u open-loop through the plant model; memory is governed by open-loop plant poles. τ_max = −1/Re(λ_max(A_c)) where A_c is the continuous-time plant state matrix | **Verify A_c is plant matrix not closed-loop; confirm τ_max definition** |
| Probe signal design (frequency range, amplitude, periods) | **GAP** | **No design specification exists for the Step 0 probe signal.** Must specify: f_probe_low (use known physical lower bound), f_probe_high (use controller bandwidth), amplitude (within actuator limits), N_avg periods. **Check for general guidance:** Lecture 3 (`Lecture 3.pdf`) on periodic signal measurement; Lecture 9 slides on averaging/SNR; P&S Ch.2 §2.5 on number of averages for FRF convergence (DOI: 10.1002/9781118287422) | **Must specify before implementation. Look in Lecture 3, 9 and P&S Ch.2 first.** |
| A_k = g(|Ŝ(jω_k)|) amplitude weighting formula | **GAP** | **No formula exists translating the |Ŝ| profile into actual amplitude values A_k.** Options: A_k ∝ |Ŝ(jω_k)|; A_k ∝ 1/|Ŝ|; proximity to resonance. **Check for general guidance:** Lecture 9 slide 13 and 27 on power allocation rules; Lecture 13 on input spectrum shaping. If no formula found, declare engineering choice | **Must specify before implementation. Look in Lecture 9, 13 first.** |
| `fs_new ≥ 10 × f_osc_min` | HEURISTIC from lecture | 5SMB0 Lecture 9, slides 10–12 (TU/e internal). Not confirmed at page level in Ljung (1999) or P&S (2012) | **Verify slide number. Declare as lecture heuristic if no book source** |
| Integer decimation: `D = round(fs_orig / fs_new)` | THEORY | Standard decimation — any DSP textbook | OK |
| FIM-driven band selection and amplitude shaping | DEFERRED | Not currently used — see Gap G12 | — |

**Supervisor statement for f_low threshold:**
> "f_low is set to the lowest frequency at which the empirical sensitivity estimate
> `|Ŝ(jω)|²` exceeds [declared threshold]. This threshold has no universal literature
> value and is declared as an engineering design choice."

**Skogestad & Postlethwaite (2005):** "Multivariable Feedback Design," 2nd ed., Wiley.
ISBN: 978-0-470-01168-3. Widely available as PDF; DOI of 1st ed: 10.1002/0470012978.

---

## STEP 1 — Multisine design: `design_multisine.m`

**What it does:** Constructs the multisine excitation signal from the f_low, f_high, fs_new
output of Step 0. Every parameter below must be justified before the signal is generated.

### Frequency line selection

> **Active strategy (resonance/bandwidth-weighted broadband):** all odd harmonics from f_low
> to f_high, with amplitude biased toward resonances and system bandwidth. No FIM-driven
> line filtering. FIM-based selection deferred to G12.

| Choice | Type | Source to verify | Flag |
|--------|------|-----------------|------|
| `f_k = k × Δf` (integer multiples of fundamental) | THEORY | P&S (2012), Ch.2 §2.2.3–2.2.5, eqs. (2-11), (2-16) — integer periods give exact DFT | **Verify eq. numbers** |
| `Δf = 1 / T_p` (fundamental = inverse of period) | THEORY | Same source — leakage-free condition | **Verify eq. numbers** |
| **Period length T_p:** must satisfy `T_p ≥ 1/f_low` so that the lowest desired frequency is a line | THEORY | Direct consequence of `f_k = k/T_p` — lowest line is `1/T_p` | Verify and document |
| Full odd-harmonic coverage from f_low to f_high | THEORY | Broadband informativity: Ljung (1999) Ch.13 p.423–424 — closed-loop data informative when input spectrum is nonzero over relevant band; PE condition (Lecture 6 sl.17–20) requires F ≥ 7 lines spread across band | **Verify Ljung §13 page; Lecture 9 slide 13 is about amplitude allocation, not coverage — do not cite for this row** |
| **PE condition: F ≥ 7 positive sinusoids** (PE order = 2F ≥ 14 = n_params) | THEORY | 5SMB0 Lecture 6 slides 17–20: "nonzero spectrum at n points → PE order n; single sine → PE order 2"; Lecture 9 slide 22: "PE(u) = 2 × harmonics" | **Verify slides — current impl min 7 bins is sufficient** |

### Amplitude

> **Active strategy — Phase 1 (parametric only):** amplitude biased toward resonances and
> system bandwidth using |Ŝ(jω)| profile from Step 0. Declared HEURISTIC — variance
> motivation in source is PEM/noise-based, not BPTT-specific.
> **Phase 2 (augmentation active):** switch to flat amplitude spectrum (D-050).
> FIM-optimal shaping deferred to G12.

| Choice | Type | Source to verify | Flag |
|--------|------|-----------------|------|
| Principle: concentrate amplitude toward resonances and system bandwidth | HEURISTIC | 5SMB0 Lecture 9 slide 13 (TU/e internal): "allocate power to relevant frequency ranges, e.g. resonance frequency and bandwidth"; Lecture 9 slide 27: "colored noise/filtering can assign more power to important frequency ranges". Also check Lecture 13 for any quantitative guidance on power allocation | **HEURISTIC — PEM/noise justification, not BPTT-specific. Declare to supervisors. Verify slide numbers against `Lecture 9.pdf` and `Lecture 13.pdf`** |
| Formula A_k = g(\|Ŝ(jω_k)\|) — how to translate profile into amplitudes | **GAP** | **No formula specified.** Check Lecture 9 slides 13, 27 and Lecture 13 for any quantitative rule (e.g. proportional to |Ŝ|, power of |Ŝ|, dB-based). If none found: declare engineering choice and specify value | **Must specify before implementation. Check `Lecture 9.pdf` and `Lecture 13.pdf` first.** |
| Phase 2 (augmentation): flat amplitude spectrum | THEORY | Ljung (1999) Ch.13 — informative inputs over full operating range | **Verify §number** |
| Normalize RMS to ETEL actuator limits | Engineering constraint | Hardware spec — no theory needed | OK |
| Clipping / amplitude sweep to max passing limits | Engineering constraint | Hardware spec | OK |
| FIM-optimal amplitude shaping `A_k ∝ 1/\|S\|` or convex `p_k` optimization | DEFERRED | No primary source for our exact setup — see Gap G12 | — |

**Supervisor statement for amplitude shaping:**
> "We concentrate multisine amplitude toward resonances and system bandwidth using the
> empirical |Ŝ(jω)| profile from Step 0, following the input design guidance in
> 5SMB0 Lecture 9 (slide 13). The specific amplitude formula A_k = g(|Ŝ(jω_k)|) is
> an engineering design choice with no primary source. The variance-based justification
> in the lecture assumes PEM with a noise model; for our noiseless BPTT setup the same
> principle applies qualitatively but has no formal proof.
> FIM-optimal line-power allocation (Gevers et al. 2011) is recorded as future work (G12)."

### Phase and harmonic structure

| Choice | Type | Source to verify | Flag |
|--------|------|-----------------|------|
| Schroeder phases: `φ_k = −k(k−1)π / F` | THEORY | Schroeder (1970), IEEE Trans. IT, 16(1):85–89, DOI 10.1109/TIT.1970.1054411 | **Verify primary source** |
| Odd harmonics only | THEORY | P&S (2012), Ch.4 §4.3.2 and Appendix 4.A, p.147 — odd-only enables nonlinearity detection via even output lines | **Verify §4.3.2** |
| Different seed per MIMO channel (phase offset) | HEURISTIC | Motivated by MIMO decorrelation need, but **insufficient per literature** | **NO SOURCE for MIMO decorrelation via phase alone** |
| Correct MIMO approach: disjoint frequency line sets per channel | THEORY | Pintelon, Vandersteen, Schoukens, Rolain (2011) "Fast FRF measurement of multivariable systems" | **Must verify — current implementation does not do this** |

**MIMO declared limitation:** The current script uses a phase seed offset per channel.
Per Pintelon et al. (2011), correct MIMO identification requires disjoint frequency line
sets per channel OR n_u separate experiments. Phase offset alone is insufficient.
**This is accepted as a known limitation for the simulation phase.** In simulation,
the impact is reduced because we are doing BPTT parameter recovery (not FRF estimation)
and channels share physical coupling. Must be resolved before hardware experiments.
Declare explicitly to supervisors.

### Injection point

| Choice | Type | Source to verify | Flag |
|--------|------|-----------------|------|
| Force injection after controller (not reference) | THEORY | D-048, `docs/ref-injection-openloop-incompatibility.md` | OK — well documented |
| cy identified from reference trajectories (T1/T6), NOT from force injection | THEORY | Inside bandwidth `\|S\| ≈ 0` → force suppressed; reference via `T ≈ 1` → reaches plant | OK — confirm in thesis framing |

**Why force injection, not reference — full argument for supervisors:**
Reference injection is the literature-preferred choice for nonparametric FRF estimation
and PEM (P&S, Gevers et al.) because `|T| ≈ 1` inside bandwidth delivers excitation
well to the plant output. However, our identification method is **BPTT with u-replay**:
we record `u_total = u_fb + f_sim` and replay it through our parametric model.
With reference injection, the controller tracks r_ms, so `u_recorded ≈ u_fb` only —
the multisine does not appear in the recorded plant input. Replaying `u_recorded` through
our model carries no multisine, gradients vanish, and the parameters become unidentifiable.
Force injection puts f_sim directly into `u_total`, so it survives the replay.
The attenuation by `|S|` inside bandwidth is the cost — partially compensated by
concentrating amplitude toward the system bandwidth (resonance-weighted heuristic, Lecture 9 slide 13).
This argument is specific to our u-replay identification method and does not contradict
the literature; the two recommendations apply to different paradigms.

**Future (hardware / noise):** force injection remains the correct choice for BPTT
u-replay. If identification switches to PEM or indirect FRF estimation, reference
injection should be reconsidered.

---

## STEP 2 — Segment analysis for BPTT

**What it does:** Determines the sample rate and segment length used for BPTT training.
These choices directly affect which parameters can be identified and whether gradients
are biased.

### Sample rate

| Choice | Type | Source to verify | Flag |
|--------|------|-----------------|------|
| `fs_new ≥ 10 × f_osc_min` | THEORY | 5SMB0 Lecture 9, slides 10–12 | **Verify — and note: confirmed in lecture, NOT verified in Ljung/P&S at page level** |
| Anti-aliasing via `scipy.signal.decimate` before downsampling | THEORY | 5SMB0 Lecture 9 pre-processing; 4CM00 `lecture_digital-filters.pdf` slides 30–35 | **Verify attenuation spec** |
| Pole clustering argument (do not use 20 kHz): discrete poles cluster near z=1 | THEORY | Ljung (1999) — somewhere in Ch.2 or Ch.8; also samplingrate GPT research | **Verify chapter/page** |

### Segment length

| Choice | Type | Source to verify | Flag |
|--------|------|-----------------|------|
| Segment is integer multiple of `T_p` | THEORY | P&S (2012) Ch.2 §2.2.3; Lecture 3 (5SMB0) — integer periods → exact DFT, no leakage | **Verify** |
| **`10 × τ_max` rule from Lecture 9** | **WRONG CONTEXT** | Lecture 9 slide 9 rule is for **FRF estimation** (transient must settle within segment for spectral accuracy). For BPTT the transient IS the signal. **Do not apply this rule.** | **Must not use** |
| Segment long enough to cover dominant BPTT memory: `T ≳ 3–5 × τ_max` | THEORY | Aicher, Foti, Fox (2019) UAI — Theorem 1: bias decays as `β^(K−τ)/(1−β)`, geometrically past memory length. Beintema, Schoukens, Tóth (2023), Automatica 156 — "few times the largest characteristic time scale" | **Verify both papers** |
| Spectral resolution constraint: `T ≥ 1/f_lowest_param` | Engineering inference | If a parameter's information is at frequency f, a segment shorter than 1/f cannot resolve it. Follows from `Δf = 1/T`. No direct BPTT primary source — declare inference. | Declare inference |
| **N_periods = 3 (current)** | HEURISTIC | **No primary source.** 3 × T_p at f_osc_min = 3 × 0.2 s = 0.6 s. This is shorter than τ_max = 1.57 s, which means it is shorter than one memory length. Insufficient by the Aicher et al. criterion. | **NO SOURCE — current value likely too short** |
| Single segment length for all 14 parameters | HEURISTIC | Different parameters have information at different time scales (K: low freq, C: mid, M: high). One-size fits all has no theoretical basis. | Declare limitation |

**Supervisor statement for segment length:**
> "Segment length is set to N_periods × T_p, where T_p is the multisine period. N_periods
> is chosen as the smallest integer such that the segment exceeds the dominant system
> memory length (≈ 3–5 × τ_max per Aicher et al. 2019 and Beintema et al. 2023). The
> integer-period alignment prevents leakage (Pintelon & Schoukens 2012, Ch.2). The 10 ×
> τ_max rule from Lecture 9 is not applied here as it is derived for FRF estimation, not
> gradient-based identification."

**Implication for current design:**
- τ_max = 1.57 s → memory length ≈ 3 × τ_max ≈ 4.7 s minimum
- Current segment = 0.607 s < τ_max → gradients for slow parameters (cy, cb_sum) are
  likely biased. This must be declared or the segment length increased.

---

# PART 2 — Theory Reference (15 Claims to Verify)

> Verify each source below. Mark `[x]` when you have confirmed the equation number
> and context match. These verified claims become the `% THEORY:` labels in code.

---

## 1. Output-Error / Simulation-Error Identification

### Claim
Our training objective `min_θ ||q_sim(x0, u_recorded, θ) - q_recorded||²` is called
**output-error (OE) identification** / simulation-error minimization. Under correct model
structure and open-loop data, the estimator is consistent (converges to true θ).
Under misspecification it converges to a **pseudo-true parameter θ\***, not the true θ.

### Sources to verify
| Book | Where | What to check | Find it |
|------|-------|---------------|---------|
| Ljung (1999) | Ch. 8, Theorem 8.4, pp. 259–260, eqs. (8.45)–(8.50) | Consistency of OE under correct spec | Lennart Ljung, "System Identification: Theory for the User," 2nd ed., Prentice Hall, 1999. ISBN 978-0-13-656695-3. Search Google Scholar: "Ljung 1999 system identification" |
| Ljung (1999) | Ch. 8, eqs. (8.71a)–(8.71b), pp. 266–267 | Pseudo-true parameter under misspecification | Same book |
| Pintelon & Schoukens (2012) | §9.9.1, pp. 305–307, eqs. (9-43)–(9-47) | NLS/OE formulation | Rik Pintelon & Johan Schoukens, "System Identification: A Frequency Domain Approach," 2nd ed., Wiley-IEEE, 2012. DOI: 10.1002/9781118287422 |

### GPT confidence: HIGH — equation numbers traceable in GPT research
### Status: `[ ]` Verified

---

## 2. Closed-Loop: Direct vs. Indirect Method

### Claim
Replaying `u_recorded` through the model as if it were an open-loop input is the
**direct method** (Ljung Ch. 13), NOT the indirect method. The indirect method uses the
reference r and a known controller to identify the closed-loop transfer, then inverts it.

### Sources to verify
| Book | Where | What to check | Find it |
|------|-------|---------------|---------|
| Ljung (1999) | §13.5, pp. 435–437, eqs. (13.57)–(13.60) | Indirect method definition and back-calculation | Same book as Claim 1 |
| Ljung (1999) | §13, eqs. (13.53a)–(13.53b), p. 433 | Bias expression for direct CL identification | Same book |

### GPT confidence: HIGH — direct vs indirect distinction clearly resolved
### Status: `[ ]` Verified

---

## 3. Closed-Loop Fisher Information Matrix (FIM)

### Claim
For **force/plant-input injection** (multisine added after controller output), the FIM is:

```
FIM(θ) ∝ ∫ |∂G/∂θ_i|² × |S(jω)|² × Φ_f(ω) / Φ_v(ω) dω
```

where S(jω) = 1/(1 + G(jω)C(jω)) is the sensitivity function. This is a reduction of the
general closed-loop FIM formula under the output-error / H=1 assumption.

The full general formula is:
```
M̄_θ = (1/2πλ₀) ∫ F(e^jω, θ₀) Φ_χ₀(ω) F*(e^jω, θ₀) dω
```

### Sources to verify
| Paper | Where | What to check |
|-------|-------|---------------|
| Gevers, Bombois, Hildebrand, Solari (2011) | §5, eqs. (5.1)–(5.10), pp. 207–208 | Full closed-loop FIM formula |
| Gevers et al. (2011) | eq. (1.2) | Correct-spec asymptotic normality around θ₀ |
| Gevers et al. (2011) | eq. (1.3) | Pseudo-true parameter under misspecification |

**DOI:** 10.4310/CIS.2011.v11.n3.a1  
**Journal:** Communications in Information and Systems, 11(3):197–224, 2011

> **Note:** The reduced scalar form `|∂G/∂θ|² |S|² Φ_f` is a derivation from the
> general formula + standard feedback algebra, NOT a directly quoted single equation
> from Gevers et al. Verify that the derivation steps are sound.

### GPT confidence: HIGH for full formula; MEDIUM for the reduced form (derived, not quoted)
### Status: `[ ]` Verified

---

## 4. OED Criteria (D-optimal, A-optimal, E-optimal)

### Claim
Standard optimality criteria minimize functions of `P_θ = FIM⁻¹`:
- **D-optimal**: minimize `det(P_θ)` → maximizes volume of information ellipsoid
- **A-optimal**: minimize `tr(P_θ)` → minimizes average parameter variance
- **E-optimal**: minimize `λ_max(P_θ)` → minimizes worst-case variance

FIM is linear/affine in the line powers `p_k = A_k²`, making D/A/E-optimal design a
convex optimization problem over spectrum amplitudes.

### Sources to verify
| Paper | Where | What to check |
|-------|-------|---------------|
| Gevers et al. (2011) | §2, §6, eq. (2.1) and around (6.6)/(5.14), pp. 203–210 | D/A/E definitions and LMI framework |
| De Cock, Gevers, Schoukens (2016) | abstract | Convex D-optimal input design for nonlinear systems |

**De Cock et al. DOI:** 10.1016/j.automatica.2016.04.052  
**Journal:** Automatica, 73:88–100, 2016

### GPT confidence: HIGH for criteria definitions; MEDIUM for De Cock et al. (abstract only)
### Status: `[ ]` Verified

---

## 5. Persistent Excitation (PE) Condition

### Claim
A sum of F real sinusoids is PE of order 2F (each sinusoid contributes 2 spectral lines).
For n_params = 14: required PE order ≥ 14, so 2F ≥ 14 → **F ≥ 7 distinct positive-frequency
sinusoids is the minimum**. The current implementation minimum of 7 bins is sufficient.

Note: PE is a necessary richness condition only — it does not prove identifiability of
the specific 14 ETEL physical parameters under closed-loop force injection (see G2).

### Sources
| Source | Where | What it says | Find it |
|--------|-------|--------------|---------|
| 5SMB0 Lecture 6 | slides 17–20 | "nonzero spectrum at n frequency points → PE of order n; single sine → PE order 2" | TU/e internal. Local: `literature/experiment-design/System-identification/Lecture 6.pdf` |
| 5SMB0 Lecture 9 | slide 22 | "multisine: PE(u) = 2 × harmonics" | TU/e internal. Local: `literature/experiment-design/System-identification/Lecture 9.pdf` |
| Gevers et al. (2011) | §6 around eq. (6.6)/(5.14) | D-optimal upper bound on sinusoid count | DOI: 10.4310/CIS.2011.v11.n3.a1 — **Verify eq. numbers** |

### GPT confidence: HIGH — corrected and confirmed from lecture primary sources
### Status: `[ ]` Verify slide numbers against PDFs

---

## 6. Leakage-Free Multisine Condition

### Claim
Leakage is eliminated when:
1. An integer number of periods is measured
2. Multisine frequencies are integer multiples of a fundamental: `f_k = k × Δf`
3. Period length: `T_p = 1/Δf`

### Sources to verify
| Source | Where | What to check | Find it |
|--------|-------|---------------|---------|
| 5SMB0 Lecture 3 | Periodic measurement material | "Only then x(t) is exactly periodic: spectrum is exact" | TU/e internal. Local: `literature/experiment-design/System-identification/Lecture 3.pdf` |
| Pintelon & Schoukens (2012) | Ch.2 §2.2.3–2.2.5, eqs. (2-11), (2-16) | Integer periods → zero leakage | DOI: 10.1002/9781118287422 |

### GPT confidence: HIGH — multiple independent sources confirm this
### Status: `[ ]` Verified

---

## 7. Amplitude Shaping Rule *(DEFERRED — not the active design; see G12)*

> **Not currently used.** Active design is resonance/bandwidth-weighted heuristic (Lecture 9
> slide 13). This claim documents the FIM-optimal shaping for future reference.

### Claim
For **force/plant-input injection**, amplitude shaping compensates for controller suppression:
```
A_k ∝ 1 / |S(j·2π·f_k)|
```
For **reference injection**, flat amplitude spectrum suffices because `|T| ≈ 1` in-band.

### Sources to verify
| Source | Where | What to check |
|--------|-------|---------------|
| Landau (2001) | §"Plant model identification in closed loop", around Fig. 4, p. 54 | "effective input = excitation filtered by S" |
| Pintelon & Schoukens (2012) | §9 or relevant section | Amplitude shaping for closed-loop multisine |

**Landau DOI:** 10.1016/S0967-0661(00)00082-4  
**Journal:** Control Engineering Practice, 9(1):51–65, 2001

> **WARNING from GPT research:** `A_k ∝ 1/|S|` was NOT found as an explicitly stated
> formula in any verified primary source. It is a well-motivated heuristic derived from
> Landau's qualitative description, but not a directly quoted design rule. The correct
> approach per Gevers et al. is to jointly optimize line powers `p_k = A_k²` directly
> via the FIM criterion. Flag this as **HEURISTIC** unless you find an explicit source.

### GPT confidence: LOW — heuristic, no primary source quotes this exact formula
### Status: `[ ]` Verified / confirmed as HEURISTIC

---

## 8. Schroeder Phases (Crest Factor Minimization)

### Claim
Schroeder phase assignment minimizes crest factor:
```
φ_k = -k(k-1)π / F
```
Gives approximately CF ≈ 1.58 (often stated but not confirmed from primary source).

### Sources to verify
| Source | Where | What to check |
|--------|-------|---------------|
| Schroeder (1970) | IEEE Transactions on Information Theory, 16(1):85–89, DOI 10.1109/TIT.1970.1054411 | Original formula |
| Ojarand & Min (2017) | Elektronika ir Elektrotechnika, 23(2):59–62, DOI 10.5755/j01.eie.23.2.18001, §I.A, eqs. (2)–(3), p. 59 | General form and equal-power specialization |

> **Note from GPT research:** The CF ≈ 1.58 claim was NOT verified from the Schroeder
> (1970) primary text. Ojarand & Min explicitly state this gives "acceptable but not the
> best CF". Better modern alternatives exist (Van der Ouderaa 1988, Guillaume 1991,
> Retzler 2022).

### GPT confidence: HIGH for formula; LOW for CF ≈ 1.58 numerical claim
### Status: `[ ]` Verified

---

## 9. MIMO Multisine — Channel Decorrelation

### Claim
For MIMO identification (3 channels), random phase offset per channel is NOT sufficient
to guarantee identifiability. Channels must use non-overlapping frequency line sets
(zippered/disjoint spectra) OR run n_u separate experiments with uncorrelated input sets.

### Sources to verify
| Paper | Where | What to check | Find it |
|-------|-------|---------------|---------|
| Pintelon, Vandersteen, Schoukens, Rolain (2011) | "Fast FRF measurement of multivariable systems using periodic excitations" | No common excited frequencies between input pairs | Search Google Scholar: "Pintelon Vandersteen Schoukens Rolain fast multivariable FRF 2011". Likely IEEE Trans. Instrum. Meas. or Mech. Syst. Signal Process. — **DOI not confirmed, must find** |

### GPT confidence: MEDIUM — claim is from GPT research, paper not fully read
### Status: `[ ]` Verified — **find DOI before citing in thesis**

---

## 10. Sampling Rate Rule (10× system bandwidth)

### Claim
The recommended sampling rate is:
```
fs_new ≥ 10 × f_system_bandwidth    (lower bound)
fs_new ≤ 30 × f_system_bandwidth    (upper bound — avoid pole clustering)
```
where bandwidth = 2π × f_osc_min (physics-derived, NOT signal content f_99).

### Sources to verify
| Source | Where | What to check | Find it |
|--------|-------|---------------|---------|
| 5SMB0 Lecture 9 | Slides 10–12 | "10ωb ≤ ωs ≤ 30ωb" — exact statement | TU/e internal. Local: `literature/experiment-design/System-identification/Lecture 9.pdf` |
| Ljung (1999) | Ch. 2 or Ch. 8 — page not confirmed | Pole clustering near unity when oversampled | ISBN 978-0-13-656695-3. Search index for "sampling" or "pole clustering" |
| Pintelon & Schoukens (2012) | Relevant section — not confirmed | Model band vs. excitation band | DOI: 10.1002/9781118287422 |

> **WARNING from GPT research:** The "10× rule" as a canonical equation was NOT found
> in verifiable page-level previews of Ljung (1999) or Pintelon & Schoukens (2012).
> It could only be confirmed from the 5SMB0 lecture slides. Label as **engineering
> recommendation** from lecture notes unless you find it in the books.

> **González et al. (2024)** confirms that parametric estimators stay consistent even
> with out-of-band aliasing — the 10× rule is NOT a hard consistency threshold for PEM.
> DOI: 10.1109/LCSYS.2024.3487501

### GPT confidence: MEDIUM — confirmed in lecture notes; not verified in primary books
### Status: `[ ]` Verified / confirmed as HEURISTIC from lecture

---

## 11. Anti-Aliasing Before Decimation

### Claim
An anti-aliasing filter must be applied before any downsampling.
`scipy.signal.decimate` applies a Chebyshev Type I filter automatically.
Required attenuation: ≥40 dB, preferably ≥60 dB at fs_new/2.

### Sources to verify
| Source | Where | What to check | Find it |
|--------|-------|---------------|---------|
| 5SMB0 Lecture 9 | Pre-processing steps slide | "Apply anti-aliasing filter before any downsampling" | TU/e internal. Local: `literature/experiment-design/System-identification/Lecture 9.pdf` |
| 4CM00 `lecture_digital-filters.pdf` | Slides 30–35 | ≥40 dB attenuation requirement | TU/e internal. Local: `literature/experiment-design/4CM00 Control engineering/lecture_digital-filters.pdf` |

### GPT confidence: HIGH — standard signal processing, well established
### Status: `[ ]` Verified

---

## 12. Segment Length for BPTT

### Claim
Segment length should be an integer multiple of the multisine period:
```
segment_len = N_periods × T_p × fs_new    (samples)
```
N_periods chosen by inspection (small integer, 1–3). This ensures each segment sees
the same excitation pattern and avoids leakage.

### Sources to verify
| Source | Where | What to check |
|--------|-------|---------------|
| 5SMB0 Lecture 3 | Integer period measurement | Leakage-free condition |

> **WARNING from GPT research:** No primary source was found that derives this as an
> **optimal** segment length rule for BPTT/simulation-error identification. The
> integer-period rule is from spectral estimation (FRF), not gradient-based identification.
> The τ_max rule from Lecture 9 slide 9 ("N ≥ 10 × τ_set,95") applies to FRF estimation,
> not BPTT — see `docs/experiment-design-pipeline.md` §4 for the explicit warning.
> The `N_periods` choice is a **HEURISTIC**.

### GPT confidence: LOW — no primary source for BPTT-specific segment length rule
### Status: `[ ]` Confirmed as HEURISTIC

---

## 13. BPTT Truncation Bias

### Claim
BPTT gradient bias decreases exponentially with truncation length K once K exceeds
the effective memory length τ:
```
||E[ĝ_K(θ)] - g(θ)|| ≤ M × E[φ_τ] × β^(K-τ) / (1-β)
```
For system identification: truncation length should be "a few times the largest
characteristic time constant."

### Sources to verify
| Paper | Where | What to check | Find it |
|-------|-------|---------------|---------|
| Aicher, Foti, Fox (2019) | UAI 2019, Assumption (A-1) eq. (7); Theorem 1 eq. (9), p. 3; Theorem 2 p. 4 | Bias bound formula | arXiv: 1905.07473. Conference proceedings: auai.org/uai2019/. Full title: "Adaptively Truncating Backpropagation Through Time to Control Gradient Bias" |
| Beintema, Schoukens, Tóth (2023) | Automatica 156, art. 111210, §3.1, §3.4 | "few times largest time scale" guideline | DOI: 10.1016/j.automatica.2023.111182 (verify). Journal: Automatica, Vol.156, 2023. Search: "Beintema Schoukens Toth nonlinear state space encoder 2023" |

### GPT confidence: HIGH for Aicher et al.; MEDIUM for Beintema et al. (HTML parse)
### Status: `[ ]` Verified

---

## 14. LPV Experiment Design — Local Approach *(DEFERRED — FIM-based, see G12)*

> **Not currently used.** Active design is resonance-weighted broadband (D-050).
> This claim is retained for reference when FIM-based design is revisited (G12).

### Claim
For LPV systems, optimal experiment design uses the **local approach**: run separate
LTI experiments at frozen scheduling values (Y = Y₁, Y₂, ...). The inverse covariance
of the LPV model is a sum of local contributions. Joint optimization over operating
points and input spectra is possible and preferable to worst-case min_Y(v(ω)).

### Sources to verify
| Paper | Where | What to check | Find it |
|-------|-------|---------------|---------|
| Ghosh, Bombois, Huillery, Scorletti, Mercère (2018) | Automatica 87:258–266 | Local approach for LPV OED; joint design | DOI: 10.1016/j.automatica.2017.10.013 |
| Khalate, Bombois, Tóth, Babuška (2009) | IFAC 2009 | Earlier local approach paper | DOI: 10.3182/20090706-3-FR-2004.00027 |

### GPT confidence: MEDIUM — abstract/metadata level only for Ghosh et al.
### Status: DEFERRED — verify when G12 is activated

---

## 15. LPV Consistency — BPTT / Simulation-Error

### Claim
There are no known primary theorems proving consistency of full-trajectory
simulation-error identification for LPV systems with incomplete model structure.
Available formal results are for LPV-ARX/prediction-error approaches only.

### Sources to verify
| Source | Where | What to check | Find it |
|--------|-------|---------------|---------|
| Tóth, Heuberger, Van den Hof (2012) | Book chapter, Def. 2.3–2.4, Theorem 2.1–2.2 | LPV-PEM consistency (prediction-error, not simulation-error) | Roland Tóth, "Modeling and Identification of Linear Parameter-Varying Systems," Springer, 2010. DOI: 10.1007/978-3-642-13812-6. Or: Tóth et al. in LPV Systems book chapter — search "Toth Heuberger Van den Hof LPV consistency 2012" |

> **Conclusion:** This is an **open gap in the literature**. Our approach has no formal
> consistency guarantee. Acknowledge in thesis write-up.

### GPT confidence: HIGH — absence of result is well-supported
### Status: `[ ]` Acknowledged as open gap

---

## Summary Table

| # | Claim | GPT Confidence | Type | Status |
|---|-------|---------------|------|--------|
| 1 | OE identification / pseudo-true | HIGH | THEORY | `[ ]` |
| 2 | Direct vs indirect CL method | HIGH | THEORY | `[ ]` |
| 3 | Closed-loop FIM formula | HIGH (full) / MEDIUM (reduced) | THEORY + derivation | `[ ]` |
| 4 | D/A/E-optimal criteria | HIGH | THEORY | `[ ]` |
| 5 | PE condition (F ≥ 7, i.e. 2F ≥ 14) | HIGH | THEORY — Lecture 6 sl.17–20, Lecture 9 sl.22 | `[ ]` verify slides |
| 6 | Leakage-free condition | HIGH | THEORY | `[ ]` |
| 7 | Amplitude shaping A_k ∝ 1/\|S\| | — | **DEFERRED → G12** | — |
| 8 | Schroeder phases | HIGH (formula) / LOW (CF=1.58) | THEORY + unconfirmed claim | `[ ]` |
| 9 | MIMO disjoint spectra | MEDIUM | THEORY | `[ ]` |
| 10 | 10× sampling rate rule | MEDIUM (lecture only) | HEURISTIC from lecture | `[ ]` |
| 11 | Anti-aliasing before decimation | HIGH | THEORY | `[ ]` |
| 12 | Segment length = N × T_p | LOW | **HEURISTIC** | `[ ]` |
| 13 | BPTT truncation bias bound | HIGH | THEORY | `[ ]` |
| 14 | LPV local approach OED | — | **DEFERRED → G12** | — |
| 15 | LPV BPTT consistency | HIGH (absence) | Open gap | `[ ]` |

---

# PART 3 — Gaps: Noted, Not Current Priority

> These are known theoretical gaps or verification methods that are NOT needed before
> the current implementation. They are recorded here so they are not forgotten. Revisit
> when the three pipeline steps are complete and results need deeper justification.

---

### G1. Gauss-Newton Gramian vs stochastic FIM

The stochastic FIM (Gevers et al. 2011) applies to PEM/ML with a noise model. Our BPTT
loss is a deterministic simulation error. The correct local curvature for our objective
is the **Gauss-Newton Gramian of output sensitivities**:
```
M̂ = (1/N) Σ_t (∂q_sim/∂θ)ᵀ (∂q_sim/∂θ)
```
Using FIM language for our objective is a category mismatch. For current purposes we use
FIM as a design proxy; for thesis write-up this distinction must be stated.

**Source:** Ljung (1999) Ch.7 eq. (7-79); `augmentation-nonparametric-gpt-research.md`

---

### G2. Data informativity (beyond PE line count)

The PE condition (F ≥ n_params) is necessary but not sufficient. The correct condition
for consistent identification is **data informativity**: the spectral matrix Φ_z(ω) > 0
for all ω. A multisine can satisfy PE yet the closed-loop data still not be informative
for specific parameters (e.g. cy under strong feedback suppression at low frequencies).

**Source:** Lyon closed-loop lecture slides (cited in `closed-loop-gpt-research.md`):
"Rather than r being persistently exciting, it is sufficient to require that the data set
is informative with respect to M."

---

### G3. Realized / empirical FIM as post-hoc verification

After running trajectories: compute the realized FIM from actual sensitivity trajectories
and compare to the designed FIM. Three scalars: `log det M̂`, `λ_min(M̂)`, `κ(M̂)`.
This is the correct post-hoc check that the data actually carries the expected information.

**Source:** Ljung (1999) Ch.13 eqs. (13.22)–(13.26); Ch.16.3 pp. 497–498

---

### G4. Model-error model for augmentation design

After fitting: model the residual as `e(t) = G_e(q)u(t) + ξ_t` (Ljung Ch.16 §16.6
eq. 16.66). The Bode plot of G_e shows which frequency bands the model fails to capture.
This is the correct way to size and target the augmentation network.

**Source:** Ljung (1999) Ch.16 §16.6, eqs. (16.52), (16.54)–(16.55), (16.58), (16.66),
pp. 511–515.

---

### G5. Multi-rate training strategy

Different parameters have information at different frequencies:
- K (stiffness): quasi-static / low frequency
- C (damping): intermediate (resonance vicinity)
- M (inertia): high frequency (ω²M dominates)

A single sample rate and segment length for all 14 parameters is architecturally
suboptimal. A multi-rate approach (fast rate for inertia, slow rate for damping) is more
efficient. No primary source gives the exact multi-rate rule — it is an inference from
sampling theory and practical identifiability.

**Source:** `samplingrate-and-segmentlength-gpt-research.md` §"Parameter-dependent time scales"

---

### G6. Periodicity verification — correct method

The correct published method to verify that data has reached periodic steady state is:
collect multiple periods, compute mean and variance over periods (not RMS difference
between consecutive periods), and check that inter-harmonic / non-excited DFT bins are
at the noise floor.

**Source:** P&S (2012) Ch.2 §2.5.1–2.5.2, eqs. (2-33)–(2-40); Ch.7 §7.3.2, p.250

---

### G7. Nonlinearity detection via odd harmonics (F-test)

Odd-only multisine → even output bins should be zero if system is linear. The published
test is the **F-test** on non-excited harmonics (P&S eq. 4-33), not a dB threshold.
Relevant for diagnosing whether the LTI model assumption holds in the data.

**Source:** P&S (2012) Ch.4 §4.3.2 and Appendix 4.A, p.147, eqs. (4-33)–(4-46)

---

### G8. Survival verification — line-by-line BLA variance

After simulation, verify that each excited frequency line actually carries information.
The published measure is line-by-line BLA variance from neighboring non-excited harmonics
(P&S eqs. 4-34 to 4-37), not a raw output/input power ratio. No universal threshold
exists — the 10 dB bias-neglect condition is not a survival criterion.

**Source:** P&S (2012) Ch.4 §4.3.1–4.3.2, eqs. (4-25)–(4-37)

---

### G9. Augmentation identifiability / confounding

When the augmentation network is sufficiently expressive, the prediction loss can be
minimized for many values of θ_physics — regularization, not data, decides the physical
parameter estimate (Takeishi & Kalousis 2019, Proposition 1). Relevant when the
augmentation network is added to the model.

**Source:** Takeishi & Kalousis (2023), AISTATS/PMLR 206, §2.3, Proposition 1;
Giampiccolo et al. (2024), npj Systems Biology, DOI 10.1038/s41540-024-00460-3

---

### G10. Operating-point stationarity within segment (ΔY bound)

No primary source gives a universal bound on ΔY within a BPTT segment. The correct
check is: ensure ΔY within a segment is small enough that the FRF/BLA does not change
beyond measurement uncertainty. Declare as engineering choice if used.

---

### G11. MIMO coherence bias in closed loop

Ordinary u-y coherence in closed loop is biased (Evers et al. 2020, eqs. 24–28). The
0.9 threshold has no primary source. For MIMO, use multiple coherence per output and
partial coherence per input-output pair. The correction belongs in the estimator, not
in a post-hoc coherence adjustment.

**Source:** Evers, Voorhoeve, Oomen (2020); de Vlugt et al. (2003),
J. Neuroscience Methods 122, eqs. (18)–(21)

---

### G12. FIM-driven frequency selection and amplitude shaping (future parametric design)

The FIM-based experiment design approach was deferred in favour of broadband uniform
(D-050). When parametric parameter recovery is complete and the identification is to be
made more efficient, the following approach should be revisited:

**Frequency band selection:** restrict excited lines to bands where
`v(ω) = Σ_i |∂G/∂θ_i|² × |S(jω)|² / Φ_v(ω)` exceeds a threshold. Avoids wasting
signal on frequencies that carry no gradient for any of the 14 parameters.
- Source: Gevers, Bombois, Hildebrand, Solari (2011), §5, eqs. (5.1)–(5.10)
- Gap: threshold value has no primary source — must declare as heuristic
- Gap: Gevers derives for stochastic PEM; our BPTT loss requires G1 (GN Gramian) correction

**Amplitude shaping:** `A_k ∝ 1/|S(j·2πf_k)|` compensates for sensitivity attenuation
inside controller bandwidth, so plant-input SNR is approximately uniform.
- Qualitative motivation: Landau (2001) §around Fig. 4 p.54
- No primary source gives this as a design rule — declare heuristic
- Formally correct alternative: convex optimization over line powers `p_k = A_k²`
  (D-optimal / A-optimal / E-optimal); source: De Cock et al. (2016), Automatica 73:88–100

**Aggregate over operating points:** `min_Y v(ω)` (worst-case) or joint optimization.
- Source: Ghosh et al. (2018) Automatica prefer joint optimization over worst-case

**When to activate:** after broadband training confirms the 14 parameters are recoverable,
and before hardware experiments where signal budget is limited.
