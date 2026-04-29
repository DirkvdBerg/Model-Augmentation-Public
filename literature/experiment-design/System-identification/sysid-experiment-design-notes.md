# System Identification — Experiment Design Notes
_Synthesized from Lectures 0–13 (5SMB0). Applied to dual-gantry LPV parameter recovery._
_Generated: 2026-04-17; corrected against `Matlab-scripts/export_param_recovery.m` on 2026-04-29._

---

## How to Use This Document

This file is a **working synthesis**, not a slide-by-slide transcript. It keeps the parts of the 5SMB0 lectures that matter for designing identification experiments for the dual-gantry LPV parameter-recovery problem.

Important scope note: the current MATLAB workflow is **not** a classical standalone closed-loop FRF experiment. It is a closed-loop trajectory-following simulation with an additional force-level multisine perturbation. Therefore, some lecture rules transfer directly (periodicity, leakage, PE sanity checks, odd lines, crest factor), while other entries below are engineering choices for safe parameter-recovery data generation rather than direct slide rules.

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
| How to handle closed-loop identification? | Lecture 11, plus Lecture 6 background | Feedback correlates plant input and disturbances in real closed-loop data; independent external excitation or IV/reference methods may be needed for unbiased FRF/PEM estimates. | Section 3 |
| Why check plant input, not only injected signal? | Lecture 10-11 closed-loop material plus actuator constraints | The signal that matters for identification and safety is the total plant/actuator input, not just the generated multisine. | Section 3 |
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
- **Reference:** smooth third-order position trajectories, not lecture-style white-noise/PRBS/reference-only experiments
- **Excitation:** optional multisine injected at feedforward slot `f` after `Cfb`, directly at actuator force level
- **Data:** 8 trajectories at different Y operating points; each can be exported with or without multisine

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

### Persistence of excitation (Lecture 6)

A signal `u` is persistently exciting (PE) of order `n` if its autocorrelation matrix `R_u^n` is non-singular. This means:
- White noise: PE of infinite order
- Multisine with `m` distinct frequencies: PE of order `2m`
- Single sine: PE of order 2 only

**Rule:** PE order should be at least the number of parameters to identify. For a scalar multisine with `F` distinct frequency lines, `2F >= 13` gives the practical guard `F >= 7`.

For `export_param_recovery.m`, this is only a **minimum spectral richness sanity check**. It does not prove that the 13 physical parameters are identifiable, because identifiability also depends on the trajectory, operating point, input direction, and parameter sensitivities of the LPV model.

### Best Linear Approximation (BLA) for nonlinear systems (Lecture 13)

This lecture concept is relevant for **real hardware** and for the **Simscape secondary path**, but it should not be over-applied to the current primary parameter-recovery target.

`export_param_recovery.m` exports `q1` as the primary signal. `q1` is generated by the simplified continuous-time quasi-LPV model:

```text
M(Y) qdd + C qdot + K q = u
```

with no Coulomb friction, no Coriolis/centripetal terms, and a linearised stage-coordinate mapping for the rotation. For a given scheduling trajectory `Y(t)`, this is a linear time-varying/LPV system. Because `Y` is itself a state, the model is self-scheduled quasi-LPV rather than LTI, but it is not the full nonlinear multibody model.

For a genuinely nonlinear system, the identified model is strictly the **best linear approximation**:

```
G_bla(ω) = argmin E_u,v { |Y(n) - G(ω)U(n)|² }
```

Key properties:
- **Input-dependent**: different multisine realisations (different random phases) give slightly different G_bla
- **Does not converge with N**: the stochastic nonlinear contribution `y_s` has O(N⁰) variance — it cannot be averaged away by collecting more data
- Output decomposes as: `y = y_bla + y_s + v` where `y_s` is the nonlinear distortion and `v` is measurement noise

**Implication for us:** BLA language is mainly appropriate when analysing Simscape or future hardware data. For current `q1`-based parameter recovery, the cleaner statement is: the 8 trajectories give multiple operating regimes for fitting the simplified quasi-LPV physics model. The gap between `q1` and `q_simscape` is where omitted nonlinear physics such as Coriolis/centripetal terms would appear.

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
- Current implementation ranges: common X mode **1–100 Hz**, differential X mode **1–20 Hz**, Y mode **1–20 Hz**
- Frequency resolution: `Δf = fs/N` — choose N to get resolution ≤ 1 Hz
- If injecting through the position reference, avoid frequencies that the controller suppresses; with the current force-feedforward injection, check the actual force and `q1` response instead

### MIMO multisine design (Lecture 9, plus project-specific choice)

For a classical 3x3 FRF estimate, the input spectrum must be full-rank. That usually means zippered frequency grids or repeated orthogonal sign-pattern experiments. In that setting, a single experiment with fully correlated actuator signals is not enough to invert the full MIMO frequency response.

That is **not exactly what `export_param_recovery.m` is doing**. The script is generating training data for BPTT-based physical parameter recovery, not estimating a standalone 3x3 FRF at each operating point. It therefore uses physically meaningful force modes:

- `common`: `F_X1 = F_X2`, translational X excitation
- `diff`: `F_X1 = -F_X2`, rotational excitation
- `y`: `F_Y`, Y-axis excitation

These mode-shaped multisines are appropriate as simultaneous parameter-recovery perturbations, especially because the nominal reference trajectory is also moving. They should not be described as strict orthogonal MIMO FRF experiments. For future hardware FRF estimation, strict zippered or orthogonal multisines remain the cleaner lecture-consistent design.

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

### External excitation in our setup

The multisine in `export_param_recovery.m` is an **external force perturbation**, independent of the feedback signal. This is good experiment design: it prevents the identification excitation from being generated by the measured output.

However, the script does not currently implement the lecture's full closed-loop FRF/reference projection workflow. It exports simulated total forces and positions for parameter recovery. If we later estimate hardware FRFs in closed loop, the external multisine can be used as the independent reference signal:

```
Ĝ(ω_n) = S_YR(n) / S_UR(n)
         = Σ_p Y^[p](n) · R̄_ff^[p](n)  /  Σ_p U^[p](n) · R̄_ff^[p](n)
```

This projection can reduce closed-loop bias in an FRF setting, provided the reference is independent of the disturbance and the measured plant input is used correctly. It should not be claimed as an automatic guarantee for the current BPTT parameter-recovery pipeline.

### Residuals test interpretation in closed-loop (Lecture 7)

| Residual behaviour | Interpretation |
|---|---|
| R̂_eu(τ) ≠ 0 for τ < 0 | **Expected in closed-loop** — due to feedback. NOT a model error. |
| R̂_eu(τ) ≠ 0 for τ ≥ 0 | Model structure inadequate — increase model order |
| R̂_e(τ) ≠ 0 for τ ≠ 0 | Noise model inadequate |

**Do not misinterpret the τ < 0 cross-correlation as a model failure** — it is the signature of feedback and is always present.

### Loss of excitation (Lectures 10–11)

If the excitation is injected through the position reference, the controller can attenuate it before it reaches the plant. In the current script the multisine is injected after `Cfb` as force feedforward, so it is not filtered by the position controller in the same way.

The relevant check is therefore not "does the reference contain enough excitation?", but:

```text
u_total = u_feedback + f_multisine
```

`validate_forces` and `summarize_forces` check this total actuator command against ETEL peak and RMS limits. For hardware, actual motor-current-derived force would be better than commanded force if amplifier dynamics or saturation matter.

### Lecture-verifiable closed-loop multisine design

To make the multisine part of the trajectory defensible from the closed-loop system-identification lectures, separate two use cases:

| Use case | What we may claim | What we must not claim |
|---|---|---|
| Moving parameter-recovery trajectory | The multisine improves spectral richness and parameter sensitivity during BPTT training | A clean closed-loop FRF/BLA estimate at one operating point |
| Constant-operating-point ID window | A lecture-style closed-loop multisine experiment, suitable for FRF/coherence checks | Full LPV parameter recovery by itself |

The lecture-clean version should satisfy this checklist:

1. **External excitation independent of feedback/noise**  
   Inject the multisine as a known signal `f_multisine` that is not computed from `q1`, `q`, or tracking error. In `export_param_recovery.m`, this is already true because `f` is generated before simulation and added at the force feedforward input after `Cfb`.

2. **Measure/log the plant input actually used for identification**  
   The plant input is not only `f_multisine`; it is:

   ```text
   u_total = u_feedback + f_multisine
   ```

   For current exports, `u_feedback` is `u_q1` and `f_multisine` is `f_sim`, so `u_total` can be reconstructed. For hardware, log motor-current-derived force if possible.

3. **Use the external signal as the reference/instrument for closed-loop FRF checks**  
   If estimating an FRF from closed-loop data, use the external multisine as the independent reference:

   ```text
   G_hat(omega_k) = S_yf(omega_k) / S_uf(omega_k)
   ```

   where `f` is the injected multisine, `u` is the measured total plant input, and `y` is the measured output. This is the closed-loop reference/instrument idea; it is separate from the BPTT training loss.

4. **Keep the analysis window periodic and approximately stationary**  
   For a lecture-style FRF/coherence check, use a hold segment at a fixed operating point, or a repeated periodic reference. A nonperiodic point-to-point move plus multisine is useful for parameter recovery but is not a clean stationary FRF experiment.

5. **Use integer periods and discard transient periods**  
   `pad_to_multisine_periods` gives integer 1 s periods. For FRF checks, analyse only complete periods after the initial transient. The current script guarantees at least two periods, but a stronger FRF run should collect more periods, e.g. 10-15 transient-free periods.

6. **Check MIMO input rank/coherence on the analysed window**  
   For full 3x3 FRF estimation, the input spectrum must be full-rank. The current `common`/`diff`/`y` modes are useful physical perturbations for parameter recovery, but strict FRF estimation should use orthogonal or zippered multisines, or separate repeated experiments.

7. **Keep amplitude below actuator and response limits**  
   The amplitude sweep, `validate_response`, and `validate_forces` implement this engineering constraint. In lecture terms, this protects against saturation and nonlinear distortion from excessive input amplitude.

Practical consequence for this project: keep the current moving multisine trajectories for gradient-based parameter recovery, but add one or more optional **ID-hold trajectories/windows** if we want lecture-verifiable closed-loop FRF/coherence plots.

### Choice-to-slide map

Use this as the slide-check map for justifying `export_param_recovery.m`. The exact wording should be verified against the PDFs; entries marked "exact slide TBD" are lecture locations that still need the PDF opened to pin down the page number.

| Implementation choice | Script location / variable | Lecture source to verify | Justification to claim |
|---|---|---|---|
| Use multisine excitation | `generate_multisine` | Lecture 9, slides 15-29; Lecture 9, slides 22-24 | Multisines are a standard experiment-design input with controlled spectral content and crest factor. |
| Use periodic records | `multisine_schroeder_periodic` | Lecture 3 multisine/periodic measurement material, exact slide TBD; Lecture 9, slides 22-24 | Periodic excitation enables clean spectral analysis after transients. |
| Use integer DFT-bin frequencies | `f0 = fs / N_period`, `freqs = k * f0` | Lecture 3 leakage/frequency-resolution material, exact slide TBD | Excited frequencies should lie on DFT bins to avoid leakage. |
| Choose 1 s period, `df = 1 Hz` | `N_period = round(fs)` | Lecture 3 frequency-resolution material, exact slide TBD | Frequency resolution is `df = fs/N`; the 1 s period is our engineering choice giving simple 1 Hz line spacing. |
| Pad to integer periods | `pad_to_multisine_periods` | Lecture 3 leakage material, exact slide TBD | The analysed record should contain an integer number of periods for leakage-free periodic excitation. |

was here!!

| Require at least two periods | `max(2, ceil(N / N_period))` | Lecture 9, slide 9 | Collect multiple periods so transient and useful data can be separated. |
| Discard transient periods for FRF checks | downstream analysis, not currently in export | Lecture 9, slide 9; Lecture 3 periodic measurement material, exact slide TBD | First period(s) may contain transient response and should not be used for steady-state FRF/coherence checks. |
| Use odd-only harmonics | `k = k(mod(k, 2) == 1)` | Lecture 13, slides 28-40 and 59-61 | Odd-only excitation leaves even lines available for detecting even-order nonlinear distortion. |
| Use Schroeder phases | `phi = -idx .* (idx - 1) * pi / F` | Lecture 9, slides 22-24; Lecture 3 multisine phase material, exact slide TBD | Schroeder phases reduce crest factor, allowing more RMS excitation before peak limits. |
| Per-channel RMS normalisation | channel loop after mode summation | Lecture 9, slides 22-24 | Compare amplitudes using RMS and control the input power delivered per actuator channel. |
| PE line-count guard `F >= 7` | `if F < 7` | Lecture 6, slides 17-21; Lecture 6, slide 19; Lecture 9, slide 22 | A multisine with `F` lines has PE order `2F`; `F >= 7` gives `2F >= 14 > 13` as a minimum richness check. |
| Treat PE as a sanity check, not proof | notes + future FIM/Jacobian check | Lecture 6, slides 12-21; Lecture 7, slides 14-19 | Informativity depends on the model set and experiment; covariance/sensitivity checks are needed for parameter directions. |
| Inject excitation externally in closed loop | `f` feedforward input after `Cfb` | Lecture 11 closed-loop identification material, exact slide TBD | In closed loop, excitation used for identification should be independent of feedback/noise. |
| Log/check total plant input | `u_q1`, `f_sim`, `force_report` | Lecture 11 closed-loop identification material, exact slide TBD | Closed-loop identification should reason about the plant input actually applied, not only the generated reference/excitation. |
| Use external signal as reference/instrument for FRF checks | proposed diagnostic: `S_yf / S_uf` | Lecture 11 closed-loop identification material, exact slide TBD | An independent external signal can be used to form closed-loop FRF/reference estimates. |
| Validate actual response, not reference only | `validate_response(q1, ...)` | Lecture 9, slides 22-24; Lecture 11 closed-loop material, exact slide TBD | The controller and plant shape the actual motion; safety and excitation checks must use measured/simulated output. |
| Sweep amplitude and choose max passing level | `amp_rms_grid` loop | Lecture 9, slides 22-24; Lecture 8, slides 50-56 | Increase input power to improve information/SNR, while staying within response and actuator constraints. |
| Check actuator peak/RMS force limits | `validate_forces`, `summarize_forces` | Lecture 9, slides 22-24 | Input amplitude must remain below saturation/actuator limits to avoid invalid nonlinear/saturated data. |
| Use physical MIMO modes `common`, `diff`, `y` | `ms_modes`, `mode_band` | Lecture 9, slides 46-52 | MIMO excitation must provide distinguishable input directions; our adaptation uses physically meaningful modal directions for parameter recovery. |
| Use strict orthogonal/zippered MIMO only for FRF diagnostics | proposed ID-hold runs | Lecture 9, slides 46-52 | Full MIMO FRF estimation requires a full-rank input spectrum; physical modes are not the same as strict FRF orthogonality. |
| Use fixed/hold windows for lecture-clean FRF checks | proposed ID-hold trajectories/windows | Lecture 3 periodic measurement material, exact slide TBD; Lecture 9, slide 9 | A stationary periodic window is needed for clean FRF/coherence interpretation; moving trajectories are parameter-recovery data, not pure FRF experiments. |

---

## 4. Data Collection and Pre-processing

### Data length (Lecture 9)

Minimum rule: `N ≥ 10 · n_θ` (number of parameters)
Better rule: `N ≥ 10 · τ_set,95` (10× settling time of slowest mode)

For each trajectory: collect **≥ 2 periods** of multisine. `export_param_recovery.m` enforces the integer-period and minimum-two-period record length, but transient exclusion is a downstream segmentation/analysis decision.
For reliable standalone BLA/FRF estimation, collect **10–15 transient-free periods** per operating point.

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
| 4 | **PE order too low** | Weak spectral richness; possible hidden parameter directions | Use ≥ 7 frequency lines as a minimum guard, then check FIM/Jacobian rank |
| 5 | **MIMO inputs correlated** | Singular input matrix for classical FRF estimation | Use orthogonal/zippered multisine for FRF tests; current BPTT export uses physical modes |
| 6 | **Closed-loop residual misread** | τ < 0 cross-correlation misread as model error | Expected in CL; not a model failure |
| 7 | **Loss of excitation** | Generated excitation may not create useful plant motion | Verify total actuator input and actual `q1` response |
| 8 | **Model structure mismatch** | Biased parameters that don't improve with more data | Start with OE/BJ; validate residuals per trajectory |
| 9 | **Nonlinear distortion y_s** | Residuals plateau → cannot be averaged away | Quantify y_s; use odd-harmonics multisine to detect source |
| 10 | **Even-order nonlinearity aliasing** | u² terms contaminate excited frequency bins | Use odd-harmonics-only multisine |
| 11 | **Data too short** | High parameter variance | N ≥ 10·τ_set,95; collect 10–15 periods per trajectory |
| 12 | **Sampling too fast or too slow** | Numerical issues or aliasing | Target fs ≈ 10–30 × bandwidth; decimate before parametric ID |
| 13 | **No noise in simulation** | Overly optimistic parameter recovery | Add realistic position measurement noise to q1 before evaluating |
| 14 | **Input-dependent BLA** | Different multisine seeds → different G_bla | Average across 3–5 realisations per operating point |

---

## 8. What This Means Concretely for Our Scripts

### Current implementation in `export_param_recovery.m`

| Component | Status | Interpretation |
|---|---|---|
| `generate_multisine` | Implemented | Builds force-level perturbations in physical modes (`common`, `diff`, `y`) |
| Odd-only harmonic lines | Implemented | Lecture-consistent nonlinear distortion diagnostic idea |
| Schroeder phases | Implemented | Lecture-consistent low crest-factor phase choice |
| 1 s period / `df = 1 Hz` | Implemented | Engineering choice that gives simple harmonic lines on DFT bins |
| `pad_to_multisine_periods` | Implemented | Pads final hold so the multisine record contains integer periods |
| `F < 7` guard | Implemented | Minimum PE sanity check, not an identifiability proof |
| Per-channel RMS normalisation | Implemented | Practical force scaling after mode summation |
| Amplitude sweep | Implemented | Safety/search heuristic: choose largest passing RMS amplitude |
| `validate_response(q1, ...)` | Implemented | Checks actual simulated response, not the reference trajectory |
| `validate_forces` / `summarize_forces` | Implemented | Checks total commanded force against ETEL peak/RMS limits |
| Strict orthogonal MIMO FRF experiment | Not implemented | Not required for current BPTT parameter-recovery data; still useful for future FRF hardware tests |
| Closed-loop reference projection FRF estimator | Not implemented | Relevant if estimating FRFs from hardware data, not part of current export script |

### Validation checks still useful downstream

- SNR check per channel per trajectory if noise is added
- Coherence/FRF confidence bounds if doing frequency-domain FRF analysis
- Residual checks for fitted models, interpreted carefully in closed loop
- FIM/Jacobian conditioning to verify that the trajectories excite all parameter directions

### No noise yet — to do

Current simulation is noise-free. Add realistic encoder noise (~1–10 nm RMS at 20 kHz) to `q1` before reporting parameter recovery results. Without noise, results are overly optimistic and will not reflect real-hardware performance.

---

## 9. Linear, LPV, and Nonlinear Caveat

These lectures mostly cover **linear** system identification. Our current primary data target, `q1`, is a simplified quasi-LPV model: linear for a fixed scheduling trajectory `Y(t)`, self-scheduled because `Y` is a state, and intentionally missing Coulomb friction and Coriolis/centripetal terms. The Simscape path and real hardware are the genuinely nonlinear targets.

| Linear concept | Applies to us as... |
|---|---|
| FRF at operating point | Linearised LPV model at fixed Y (each trajectory) |
| Parameter consistency | Physical parameter recovery when model structure matches |
| BLA | Relevant for Simscape/hardware nonlinear analysis, not the main interpretation of `q1` training |
| PE condition | Multisine must excite all 13 parameter sensitivities |
| Closed-loop bias | Relevant for noisy hardware FRF/PEM; current simulation exports commanded forces and positions for BPTT |
| y_s (nonlinear distortion) | Coriolis, friction, and geometry effects when comparing against Simscape/hardware |

When training against `q1`, the target is the same simplified quasi-LPV physics that the baseline is meant to represent. When comparing against `q_simscape` or hardware, omitted nonlinear effects appear as a residual floor. That residual is what the **augmentation** phase is designed to model.

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
