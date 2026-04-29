# System Identification — Experiment Design Notes
_Synthesized from Lectures 0–13 (5SMB0). Applied to dual-gantry LPV parameter recovery._
_Generated: 2026-04-17_

---

## How to Use This Document

This file is a **working synthesis**, not a slide-by-slide transcript. It keeps the parts of the 5SMB0 lectures that matter for designing identification experiments for the dual-gantry LPV parameter-recovery problem.

Use it in two ways:

1. **Design guide:** Sections 1-9 translate the lecture material into concrete choices for our setup.
2. **Lookup table:** The tables below tell you where to go in the original lecture PDFs when you want the source derivation or slide context.

The current synthesis covers the experiment-design-relevant material from the slides, but it intentionally does not include every lecture topic. For example, generic predictor derivations, full ARMAX/OE/BJ algebra, subspace-realization details, and general validation theory are only included where they affect experiment design.

---

## Lecture Lookup Table

| Question | Look in | Main idea | Used here |
|---|---:|---|---|
| What makes data informative? | Lecture 6, slides 12-15 | Data must distinguish models in the chosen model set; for open loop this is linked to input spectrum. | Section 1 |
| What is persistent excitation? | Lecture 6, slides 17-21 | PE is a rank/non-singularity condition on input autocorrelation; white noise is PE of any order, one sine is PE order 2. | Section 1 |
| How many frequencies do I need? | Lecture 6, slide 19; Lecture 9, slide 22 | Nonzero input spectrum at enough frequency points; multisine PE order is twice the number of excited lines. | Sections 1-2 |
| How does experiment design influence variance? | Lecture 7, slides 14-19 | Parameter covariance depends on input power/SNR and experimental conditions. | Sections 6-8 |
| Where is the Fisher information matrix used? | Lecture 7, ML slides near the end | ML/PEM covariance is asymptotically the inverse Fisher information matrix. | Section 10 |
| Where is the Jacobian used? | Lecture 12, transfer-function identification slides around the Gauss-Newton derivation | Frequency-domain nonlinear least-squares uses a Jacobian of the complex residuals; covariance is built from that Jacobian. | Section 10 |
| How should I choose experiment length? | Lecture 9, slide 9 | Minimum `N >= 10*n_theta`; better: long relative to slowest settling time. | Section 4 |
| How should I choose sampling frequency? | Lecture 9, slides 10-12 | Sample fast enough for dynamics/noise, but decimate before parametric ID when oversampled. | Section 4 |
| Which input signals are discussed? | Lecture 9, slides 15-29 | RBS, PRBS, multisine, swept sine, colored noise, staircase/interleaved multisine. | Section 2 |
| Why multisine? | Lecture 3 and Lecture 9, slides 22-24 | Periodic, exact frequency-line control, multiple transient-free periods, controllable crest factor. | Section 2 |
| How to handle MIMO excitation? | Lecture 9, slides 46-52 | Require positive-definite MIMO input spectrum; use zippered or orthogonal multisines. | Section 2 |
| How to handle closed-loop identification? | Lecture 11, plus Lecture 6 background | Feedback correlates input and disturbances; external reference excitation can restore consistency. | Section 3 |
| Why check plant input, not only injected reference? | Lecture 10-11 closed-loop material | Controller can attenuate excitation at some frequencies, causing loss of excitation. | Section 3 |
| What if model uncertainty is too high? | Lecture 8, slides 50-56 | Redesign the experiment: increase input power or add spectral power near uncertain/resonant regions. | Sections 6-8 |
| Why use odd multisines? | Lecture 13, slides 28-40 and 59-61 | Odd-only lines help detect/separate even and odd nonlinear distortion. | Sections 2 and 9 |
| What is BLA and why does it matter? | Lecture 13, slides 43-58 | Nonlinear systems produce an input-dependent best linear approximation and nonlinear distortion floor. | Sections 1 and 9 |

---

## Method Lookup: Designing Experiments Beforehand

| Method | What it chooses | What it needs | Strength | Limitation |
|---|---|---|---|---|
| PE/rank check | Enough independent frequency content | Model order or number of parameters | Simple necessary sanity check | Does not tell which frequencies are best |
| Input spectrum shaping | Where to put input power over frequency | Frequency range of interest, rough dynamics | Directly targets bandwidth/resonances | Usually heuristic unless optimized |
| Multisine design | Frequency lines, amplitudes, phases, periods | Sampling rate, duration, amplitude limits | Clean FRF/BLA estimates and no leakage when periodic | Needs careful MIMO design |
| PRBS/RBS/white noise | Broad excitation under amplitude/power constraints | Bandwidth and actuator limits | Good generic excitation | Less targeted; can waste power |
| Orthogonal MIMO multisine | Independent MIMO input directions | `n_u` repeated experiments or zippered grid | Well-conditioned FRF matrix estimate | More experiments |
| Covariance/FIM design | Input spectrum or trajectory minimizing parameter covariance | Nominal model and noise assumptions | Formal optimal experiment design | Needs a model beforehand; local if nonlinear |
| Jacobian/sensitivity design | Trajectory maximizing parameter-output sensitivities | Differentiable simulation model | Directly relevant to parameter recovery | Can miss structural non-identifiability |
| BLA/nonlinear distortion analysis | Whether linear approximation is trustworthy | Repeated random-phase multisines | Separates noise from nonlinear distortion | Does not by itself recover nonlinear parameters |

For our current project, the most useful workflow is:

```text
Lecture rules -> generate safe candidate excitations
model sensitivities/Jacobian -> score parameter recoverability
FIM/covariance -> check whether all parameter directions are excited
redesign input spectrum/trajectory -> repeat
```

---

## Our Setup (context for all notes below)

- **Plant:** dual-gantry, 3 inputs `[F_X1, F_X2, F_Y]` → 3 outputs `[X1, X2, Y]` (3×3 MIMO)
- **Model:** quasi-LPV with scheduling variable Y; M(Y) varies continuously
- **Identification goal:** recover 13 physical parameters from simulated closed-loop data
- **Control loop:** feedback controller `Cfb` (designed at `Y_op = Y_initial` per trajectory)
- **Excitation:** multisine injected at feedforward slot `f` after `Cfb`, directly at actuator force level
- **Data:** 6 trajectories at different Y operating points; each includes nominal motion + multisine

```
r ──► [Cfb] ──► (+) ──► [Plant G(Y)] ──► y = [X1, X2, Y]
              ▲   ▲
              │   └── f_multisine   ← identification excitation goes here
              └─────────────────────── feedback
```

---

## 1. Fundamental Concepts

### Consistency and identifiability (Lectures 4–6)

Parameter estimate `θ_N → θ*` as `N → ∞`. This is only the true parameter `θ_0` if:
1. The system is in the model set (`S ∈ M`) — the model structure can represent the true system
2. The data is **informative** with respect to the model structure
3. The model is **identifiable** at `θ_0`

**Identifiability:** model is locally identifiable at `θ_1` if no other parameter in a neighbourhood gives the same transfer function at almost all frequencies.

**Data informativity condition:** for open-loop, having `Φ_u(ω) > 0` at enough frequencies is sufficient. For MIMO: `Φ_u(ω) ≻ 0` (positive definite matrix) — inputs must not be linearly correlated at identification frequencies.

### Persistence of excitation (Lectures 4–5)

A signal `u` is persistently exciting (PE) of order `n` if its autocorrelation matrix `R_u^n` is non-singular. This means:
- White noise: PE of infinite order
- Multisine with `m` distinct frequencies: PE of order `2m`
- Single sine: PE of order 2 only

**Rule:** PE order must be ≥ the number of parameters to identify. For our 13-parameter model, the multisine must have ≥ 7 distinct frequency lines per channel.

### Best Linear Approximation (BLA) for nonlinear systems (Lecture 13)

Since our gantry is nonlinear (LPV, friction), the identified model is strictly the **best linear approximation**:

```
G_bla(ω) = argmin E_u,v { |Y(n) - G(ω)U(n)|² }
```

Key properties:
- **Input-dependent**: different multisine realisations (different random phases) give slightly different G_bla
- **Does not converge with N**: the stochastic nonlinear contribution `y_s` has O(N⁰) variance — it cannot be averaged away by collecting more data
- Output decomposes as: `y = y_bla + y_s + v` where `y_s` is the nonlinear distortion and `v` is measurement noise

**Implication for us:** the 6 trajectories give 6 BLA estimates at different Y operating points. The parameter recovery fits a physics model to these BLAs. Any unmodeled nonlinearity (Coulomb friction, Coriolis) appears as `y_s` and inflates the residuals — it cannot be reduced by longer experiments, only by better model structure.

---

## 2. Multisine Design

### Definition and structure (Lectures 3, 9, 12)

```
u(k) = Σ_{n=1}^{F} α_n · sin(ω_n·k + φ_n)
```

- Deterministic, periodic with period `T_p = N·T_s` samples
- Energy concentrated at chosen frequency lines only
- PE order = `2F` (F distinct frequency lines)
- MATLAB: `idinput(N, 'sine', [f_low/fs, f_high/fs], [-amp, amp])`

### Phase selection and crest factor (Lectures 3, 9)

The crest factor CF = ‖u‖_∞ / ‖u‖_2 measures peak-to-RMS ratio. High CF means the signal hits large peak forces, risking actuator saturation or exciting nonlinearities.

| Phase choice | Crest factor |
|---|---|
| Linear: φ_n = −τω_n | 22.34 |
| Random: φ_n ~ U(0, 2π) | ~3.08 |
| **Schroeder:** φ_n = −n(n−1)π/F | **1.58** |

**Use Schroeder phases** to minimise peak actuator force for the same RMS power. This is critical for filtering against ETEL hardware limits.

### Frequency range for our system (Lectures 9, 13)

- Controller bandwidth: `fbw = 100 Hz`
- Plant resonances: rotational mode around `kb1+kb2` (stiffness); translational at low frequency
- Recommended range: **1–200 Hz** (covers controller bandwidth + margin; well below Nyquist at 10 kHz)
- Frequency resolution: `Δf = fs/N` — choose N to get resolution ≤ 1 Hz
- **Avoid controller zeros** where possible — the closed-loop may attenuate the multisine at these frequencies

### MIMO orthogonal multisine (Lecture 9) — IMPORTANT

For a 3×3 MIMO system, a single multisine on all inputs simultaneously does **not** give an invertible input matrix. Two approaches:

**Option A — Zippered multisine (one experiment):**
- Interlace excited frequency grids across inputs (F_X1 gets even lines, F_X2 odd lines, F_Y every third, etc.)
- Lower frequency resolution, relies on ideal spectrum; FRFs estimated on different frequencies
- Simple to implement

**Option B — Orthogonal multisine (3 experiments, RECOMMENDED):**
- Run 3 separate experiments with orthogonal input patterns
- Example for 2×2 (generalises to 3×3):
  - Experiment 1: `[U1, U2]`
  - Experiment 2: `[U1, −U2]`
  - Stack: `Ĝ = [Y1, Y2] · [U1 U1; U2 -U2]^{-1}`
- Input matrix is orthogonal → well-conditioned inverse guaranteed
- **For 3×3: requires 3 experiments**, each with a different sign pattern on the 3 multisines

**What this means for our 6 trajectories:** each trajectory operating point requires 3 experiments (3 multisine sign patterns) to fully recover the 3×3 G matrix. Total: 18 simulations.

### Odd harmonics design (Lecture 12–13)

If even-order nonlinearities are suspected (e.g., u² terms from Coriolis or quadratic friction):
- Excite **odd harmonics only**: `f_0, 3f_0, 5f_0, ...`
- Even harmonics (2f_0, 4f_0, ...) remain empty in the spectrum
- Even-order nonlinear distortions appear at even harmonics → detectable separately from the model output at odd harmonics
- This allows quantification of nonlinear distortion level without changing model structure

For our gantry: Coriolis terms (Ẏ·Ẋ) are second-order → use odd-harmonics multisine to detect their contribution.

### Amplitude selection (supervisor's approach)

1. In simulation, vary multisine RMS amplitude freely (no hardware risk)
2. Simulate and check actual `q1` response
3. Filter: keep amplitudes where `q1` derivatives stay within ETEL limits:
   - Position: ±375 mm (X1, X2), ±400 mm (Y), ≤100 mm differential
   - Velocity: ≤ 2 m/s
   - Acceleration: ≤ 50 m/s²
4. Use maximum passing amplitude → maximum excitation within hardware constraints

**Rule of thumb from lectures:** keep multisine amplitude ≤ 20–30% of maximum actuator force to stay in linear regime.

---

## 3. Closed-Loop Identification

### The fundamental problem (Lectures 6, 11)

In closed-loop, the plant input `u` and disturbance `v` are **correlated** because feedback adjusts `u` to compensate for `v`. This violates the standard PEM consistency assumption, causing biased estimates if not handled.

### Reference signal method — directly applicable to our setup (Lecture 3)

Since our multisine is injected at the feedforward slot `f`, it acts as an external reference signal `r_ff` that is **independent** of the disturbance `v`. The FRF can be estimated by projecting output and input onto this reference:

```
Ĝ(ω_n) = S_YR(n) / S_UR(n)
         = Σ_p Y^[p](n) · R̄_ff^[p](n)  /  Σ_p U^[p](n) · R̄_ff^[p](n)
```

This eliminates the bias from the feedback controller automatically. The key requirement: `f_multisine` must be **truly independent** of the feedback signal (do not compute it from the output).

### Residuals test interpretation in closed-loop (Lecture 7)

| Residual behaviour | Interpretation |
|---|---|
| R̂_eu(τ) ≠ 0 for τ < 0 | **Expected in closed-loop** — due to feedback. NOT a model error. |
| R̂_eu(τ) ≠ 0 for τ ≥ 0 | Model structure inadequate — increase model order |
| R̂_e(τ) ≠ 0 for τ ≠ 0 | Noise model inadequate |

**Do not misinterpret the τ < 0 cross-correlation as a model failure** — it is the signature of feedback and is always present.

### Loss of excitation (Lectures 10–11)

If the controller `Cfb` has high gain at certain frequencies, it will attenuate the multisine excitation reaching the plant at those frequencies. This causes **loss of excitation** → those frequencies become unidentifiable.

**Check:** verify that the multisine amplitude at the **plant input** (not just the feedforward reference) is sufficient. With `u = Cfb·(r−q1) + f_multisine`, the plant sees both. If `Cfb` has very high gain, `Cfb·(r−q1)` dominates and `f_multisine` is relatively small.

---

## 4. Data Collection and Pre-processing

### Data length (Lecture 9)

Minimum rule: `N ≥ 10 · n_θ` (number of parameters)
Better rule: `N ≥ 10 · τ_set,95` (10× settling time of slowest mode)

For each trajectory: collect **≥ 2 periods** of multisine (discard first as transient, use second onward).
For reliable BLA: collect **10–15 transient-free periods** per operating point.

### Sampling frequency (Lecture 9)

```
10·ω_b ≤ ω_s ≤ 30·ω_b
```

Our system: `fs = 20 kHz`, bandwidth `fbw = 100 Hz → ω_b ≈ 628 rad/s`.
`10·ω_b = 6280 rad/s → fs_min ≈ 1 kHz`. Our 20 kHz is well above minimum.

Consider **decimation** before parametric identification — working at 20 kHz with a 200 Hz model is wasteful and numerically poor. Decimate to e.g. 2–4 kHz.

### Transient removal (Lectures 3, 12)

- The system has a transient response when the multisine switches on
- **Remove the first full period** before any spectral analysis
- Faster systems need shorter transient removal; slower settling → more periods to discard
- Lecture 12 standard: use 10 transient-free periods for stable FRF estimates

### Leakage — critical for multisine (Lecture 3)

Leakage occurs when the data window does not contain an **integer number of multisine periods**. Energy from each frequency bin spreads across all others, corrupting FRF estimates.

**Fix:** ensure `N = integer · T_p` exactly. For periodic multisines with `fs/f_0 = integer`, this is automatic. Design multisine frequencies as **integer multiples of the frequency resolution** `Δf = fs/N`.

### Pre-processing steps (Lecture 9)

1. Remove outliers / sensor spikes (`filloutliers` in MATLAB)
2. Remove DC offset / trends (`detrend`)
3. Apply anti-aliasing filter before any downsampling
4. Optional: pre-filter `L(q)` to shape noise dynamics before parametric ID

---

## 5. Model Structure Choice

### Independent G and H parametrisation (Lectures 5–6)

| Structure | G and H independent? | Consistent if S ∉ M but G_o ∈ G? |
|---|---|---|
| ARX | No | Only if u ⊥ e |
| ARMAX | No | No |
| **OE** | **Yes** | **Yes** |
| **BJ** | **Yes** | **Yes** |
| FIR | Yes (H=1) | Yes |

**Recommendation:** use OE or BJ structure. With independent G/H parametrisation, getting the noise model wrong does not bias the plant model estimate — critical since we do not know the true noise model.

For MIMO: state-space models are preferred over polynomial MIMO models to avoid identifiability issues (A, B matrices not necessarily co-prime in MIMO polynomial structures).

### Bias from model mismatch (Lecture 6)

If `S ∉ M` (true system outside model class), parameter estimates are biased. The bias does not decrease with more data. Example from lectures: ARX fit to ARMAX data gives `a ≈ −0.8` instead of true `−0.9`.

**For us:** if the LPV model cannot represent the true plant at a given operating point (e.g., unmodeled Coriolis), all 13 parameters will be biased. This is why trajectory diversity matters — different biases at different operating points expose the model mismatch.

---

## 6. Validation

### FRF confidence bounds (Lecture 7)

```
√cov{G(e^jω, θ̂_N)} < 0.1 · |G(e^jω, θ̂_N)|   for all ω in [0, ω_b]
```

This is the rule-of-thumb for control applications: model uncertainty should be < 10% of model magnitude within the control bandwidth.

### Variance from averaging (Lecture 3)

Averaging over `P` periods reduces FRF variance by `1/P`. Target P = 10–15 for practical accuracy.

**Warning:** the nonlinear distortion `y_s` does NOT decrease with P or N (O(N⁰)). If residuals plateau after sufficient averaging, the residual floor is nonlinear distortion — not noise. Longer experiments will not help; a better model is needed.

### SNR requirement (Lecture 2)

Rule of thumb: SNR > 10 dB required for reliable identification.
```
SNR = 10·log10(P_ỹ / P_v)   [dB]
```

Verify at the output `[X1, X2, Y]` for each trajectory and each multisine amplitude level.

---

## 7. Pitfalls Summary

| # | Pitfall | Risk | Mitigation |
|---|---|---|---|
| 1 | **Actuator saturation** from high crest factor | Nonlinear response → biased BLA | Use Schroeder phases; keep amplitude ≤ 20–30% max force |
| 2 | **Leakage** from non-integer periods | FRF estimates corrupted across all frequencies | Design frequencies as integer multiples of Δf = fs/N |
| 3 | **Transient not removed** | Systematic bias especially at resonances | Discard ≥ 1 full period; use 10+ transient-free periods |
| 4 | **PE order too low** | Parameters not identifiable | Use ≥ 7 frequency lines per input; check Φ_u(ω) ≻ 0 |
| 5 | **MIMO inputs correlated** | Singular input matrix → no FRF estimate | Use orthogonal multisine (3 experiments for 3×3) |
| 6 | **Closed-loop residual misread** | τ < 0 cross-correlation misread as model error | Expected in CL; not a model failure |
| 7 | **Loss of excitation** | Controller attenuates multisine at some frequencies | Verify plant input amplitude, not just feedforward amplitude |
| 8 | **Model structure mismatch** | Biased parameters that don't improve with more data | Start with OE/BJ; validate residuals per trajectory |
| 9 | **Nonlinear distortion y_s** | Residuals plateau → cannot be averaged away | Quantify y_s; use odd-harmonics multisine to detect source |
| 10 | **Even-order nonlinearity aliasing** | u² terms contaminate excited frequency bins | Use odd-harmonics-only multisine |
| 11 | **Data too short** | High parameter variance | N ≥ 10·τ_set,95; collect 10–15 periods per trajectory |
| 12 | **Sampling too fast or too slow** | Numerical issues or aliasing | Target fs ≈ 10–30 × bandwidth; decimate before parametric ID |
| 13 | **No noise in simulation** | Overly optimistic parameter recovery | Add realistic position measurement noise to q1 before evaluating |
| 14 | **Input-dependent BLA** | Different multisine seeds → different G_bla | Average across 3–5 realisations per operating point |

---

## 8. What This Means Concretely for Our Scripts

### Immediate changes needed in `export_lpv_multi_traj.m`

1. **Add multisine at `f`:**
   ```matlab
   % Replace: f = zeros(size(r));
   % With (per trajectory, after Cfb/G design):
   f = zeros(size(r));
   for ch = 1:3
       f(:,ch) = multisine_schroeder(size(r,1), fs, f_low, f_high, amp_rms);
   end
   % Make each channel independent (different phase seeds)
   ```

2. **Make multisine frequencies integer multiples of Δf = fs/N** (no leakage)

3. **Collect ≥ 2 periods** — discard first as transient, use rest for identification

4. **Amplitude sweep + ETEL filter:** run at multiple amplitudes; filter on actual `q1` velocity and acceleration (not reference)

5. **For full MIMO identification:** run 3 experiments per trajectory with orthogonal sign patterns on `[F_X1, F_X2, F_Y]`

### Validation checks to add

- Whiteness test on residuals (autocorrelation of prediction error)
- Cross-correlation between residuals and input (expect nonzero for τ < 0 — that is fine)
- SNR check per channel per trajectory
- FRF confidence bounds: verify √cov{G} < 0.1·|G| in control bandwidth

### No noise yet — to do

Current simulation is noise-free. Add realistic encoder noise (~1–10 nm RMS at 20 kHz) to `q1` before reporting parameter recovery results. Without noise, results are overly optimistic and will not reflect real-hardware performance.

---

## 9. Linear vs. Nonlinear Caveat

These lectures cover **linear** system identification. Our plant is nonlinear (LPV + friction + Coriolis). The key transfer:

| Linear concept | Applies to us as... |
|---|---|
| FRF at operating point | Linearised LPV model at fixed Y (each trajectory) |
| Parameter consistency | Physical parameter recovery when model structure matches |
| BLA | What our LPV fit actually estimates when nonlinearities present |
| PE condition | Multisine must excite all 13 parameter sensitivities |
| Closed-loop bias | Reduced by reference signal method; not eliminated |
| y_s (nonlinear distortion) | Coriolis, friction terms not in baseline — will inflate residuals |

The nonlinear parts (Coriolis, Coulomb friction) appear as `y_s` and set a floor on the achievable residual. This floor is what the **augmentation** phase is designed to model.

---

## 10. Jacobian, Fisher Information, and OED Connection

The system-identification lectures mostly teach experiment design through **data informativity**, **persistent excitation**, and **input spectrum design**. They do also contain the ingredients used by optimal experiment design:

| Object | Where in the lectures | Meaning |
|---|---|---|
| Parameter covariance `P_theta / N` | Lecture 7, slides 14-19 | Quantifies how uncertain the estimated parameters are. Larger input power and lower noise reduce this covariance. |
| Fisher information matrix `J_N` | Lecture 7, ML/CRLB slides near the end | For ML/PEM, asymptotically `cov(theta_hat) = J_N^-1`. This is the formal link between experiment design and parameter uncertainty. |
| Residual Jacobian | Lecture 12, frequency-domain fitting | Nonlinear least-squares fitting uses a Jacobian of the residuals with respect to parameters. Parameter covariance can be approximated from this Jacobian. |
| Input spectrum `Phi_u(omega)` | Lecture 6 and Lecture 9 | Determines which model differences can be seen in the data and how much variance the estimate has. |

For our differentiable LPV/LFR simulator, the same logic becomes:

```text
u(t) = [F_X1(t), F_X2(t), F_Y(t)]
simulate model with known/nominal parameters
J = d vec([X1(t), X2(t), Y(t)]) / d log_params
F = J^T R^-1 J
cov(theta_hat) approx F^-1
```

Then we can evaluate a candidate trajectory by scalar criteria such as:

```text
D-optimal: maximize log det(F + lambda I)
A-optimal: minimize trace((F + lambda I)^-1)
E-optimal: maximize lambda_min(F + lambda I)
conditioning: minimize cond(F + lambda I)
```

This is the formal version of the lecture guidance:

```text
PE condition:
    make sure enough independent parameter directions are visible

FIM/Jacobian design:
    quantify exactly which parameter directions are visible for this model
```

The practical recommendation is to use both:

1. Use Lecture 9 rules to generate safe, physically meaningful input families: multisine, PRBS, colored noise, swept sine, staircase, orthogonal MIMO multisine.
2. Use the simulator Jacobian/FIM to rank or optimize those candidates for recovery of the 13 physical parameters.
3. Inspect the smallest eigenvectors of `F` to see which parameter combinations remain hidden.
4. Add or reshape trajectories to target those weak directions.

Important caveat: if two physical parameters enter the model only through the same combination, such as a pure sum, no trajectory can separate them. The FIM will reveal this as a structural rank deficiency, but experiment design cannot fix it without extra measurements or a different parametrization.
