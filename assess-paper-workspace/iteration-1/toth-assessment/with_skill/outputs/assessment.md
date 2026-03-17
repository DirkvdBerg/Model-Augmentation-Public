# Paper Assessment: Tóth et al. (2010) — "On The Discretization of LPV State-Space Representations"

**Paper**: R. Tóth, P. S. C. Heuberger, P. M. J. Van den Hof. "On The Discretization of LPV
State-Space Representations." DRAFT, August 16, 2010.

**Spec**: `docs/lpv-discretization.md`

**Assessed against**: LPV discretization of the gantry FP model for use in validation simulation
(Use case 1) and the PyTorch training loop (Use case 2).

---

## Part 1 — Question-by-question assessment

---

**Q1: Frozen-at-sampling-instant accuracy — What is the order of the approximation error in ts when Y is frozen at Y[k] during [k·ts, (k+1)·ts]? Is it O(ts²) or better? Does it depend on the rate of change of Y?**

Verdict: YES — fully addressed, with quantified order and rate-of-change dependence.

Evidence: Section III-B (Complete method, eq. 9a), Section IV-A (Local Truncation Error analysis, Table I, eqs. 22–24), Section III (ZOH Assumption 1, eq. 3b), Section IV-D (Switching effects).

Key results:

1. The "frozen-at-sampling-instant" approach is precisely what the paper calls the **Complete method** (eq. 9a): A_d(p_d(k)) = exp(A_c(p_c(kT_d)) · T_d). Under ZOH Assumption 1, p_c(t) := p_d(k) for t in [kT_d, (k+1)T_d], so the scheduling variable is held constant during each interval. The paper states explicitly (p. 7): "The complete method theoretically provides errorless discretization in terms of the ZOH setting." This means the complete method has **zero local truncation error** — the approximation is exact for the frozen scheduling problem.

2. The error in question comes from the **mismatch between the ZOH assumption and reality** (i.e., the scheduling variable Y actually changes within the interval). This is the "switching effect" analyzed in Section IV-D (eqs. 50–51, pp. 19–20). The paper shows that when p_c changes within a sampling interval, additional terms involving p_d(k-1) and u_d(k-1) appear in the true solution (eq. 50) that are not captured by the ZOH model. The paper concludes (p. 20): "neglecting the switching effects introduces discretization errors in the LPV case which can be even more significant if T_d is decreased." However, it also states: "it is true that the discontinuous phenomena which are described by (50) never happen in reality. One reason is that usually p_c is not actuated by ZOH and it changes smoothly and relatively slowly with respect to the actual dynamics of the plant."

3. For the **approximative methods** (Rectangular, Polynomial, Trapezoidal), the local unit truncation (LUT) error orders are given explicitly in Table I (p. 13):
   - Rectangular method: ε_k = (T_d/2) · x_c^(2)(τ) — **O(T_d), first-order consistent**
   - n-th polynomial method: ε_k = T_d^(n+1)/((n+1)!) · x_c^(n+1)(τ) — **O(T_d^n), n-th order consistent**
   - Trapezoidal method: ε_k = (1/12) T_d^2 · x_c^(3)(τ) — **O(T_d^2), second-order consistent**
   - Adams-Bashforth (3-step): ε_k = (3/8) T_d^3 · x_c^(4)(τ) — **O(T_d^3)**

4. Rate-of-change dependence: Yes. From eq. 23 (p. 12): x_c^(2)(τ) = (∂f/∂x_c)·x_c_dot + (∂f/∂u_c)·u_c_dot + (∂f/∂p_c)·p_c_dot. Under ZOH (Assumptions 1–2), u_c_dot = p_c_dot = 0 inside each interval, so the error reduces to ‖ẍ_c(τ)‖ ≤ max ‖A_c²(p)x + A_c(p)B_c(p)u‖ — the sensitivity constant M^(1). The rate of change of Y within the interval (p_c_dot) only contributes if the ZOH assumption on the scheduling is violated. The paper's conclusion: at sufficiently high sampling rates (T_d below the N-stability radius), these effects are negligible.

**Practical answer for our case**: At fs = 16 kHz (T_d = 62.5 µs) with ΔY ≤ 0.125 mm/sample, the ZOH assumption on the scheduling is well-satisfied (Y changes very slowly relative to the dynamics). The complete method (scipy `cont2discrete` at each step) is theoretically exact for the frozen problem and introduces no LUT error. The only real error source is the slowly-varying scheduling mismatch, which the paper shows to be negligible for slow-varying p_c.

---

**Q2: Polynomial expansion of A_d(Y) — Does the paper provide a method to compute A_d(Y) = A0_d + Y·A1_d + Y²·A2_d + ... analytically from A_c(Y)? If so, how many terms are needed at ts = 62.5 µs?**

Verdict: PARTIAL — the polynomial method is provided (eq. 12 / Section III-C-2) and gives a systematic expansion, but the paper does not prescribe how many terms are needed for a specific system or sampling period. That requires system-specific computation.

Evidence: Section III-C-2 "Polynomial (Hanselmann) method" (pp. 8–9, eq. 12), Table I (p. 13), Table II (p. 21), Section IV-C (eq. 48 for sampling bound).

Key result: The **n-th order polynomial discretization** gives:

```
A_d(p_d(k)) = I + sum_{l=1}^{n} (T_d^l / l!) · A_c^l(p_c(kT_d))
B_d(p_d(k)) = T_d · (I + sum_{l=1}^{n-1} (T_d^l / (l+1)!) · A_c^l(p_c(kT_d))) · B_c(p_c(kT_d))
```

This is exactly the Taylor expansion of the matrix exponential (eq. 12). If A_c(Y) has polynomial dependence on Y (which ours does — rational, not polynomial, but see Q3), then A_c^l(Y) is also a function of Y with increasing degree. The result is A_d(Y) as a polynomial in Y, enabling Option A.

The paper does not prescribe the required n for a given (system, T_d) pair. Instead it provides the sampling upper-bound formula (Table I, third row): T_hat_d = n^(1/(n+1)) · sqrt((ε_max · M_x^max · (n+1)!) / (100 · M^(n))). This requires computing the n-th order sensitivity constant M^(n) = max ‖A_c^(n+1)(p)x + A_c^n(p)B_c(p)u‖. To determine the required n for our system at T_d = 62.5 µs, one would need to evaluate M^(n) for successive n until the bound is satisfied. The paper does not do this for any specific system — it is left to the user.

For the numerical example in Section VII, the 2nd-order polynomial method achieves η_hat_max = 0.06% at T_d = 10^(-4) (10 kHz), comparable to the trapezoidal method's 0.19% — suggesting n=2 is generally sufficient at high sampling rates.

---

**Q3: Non-affine handling — A_c(Y) is rational (not polynomial) in Y. Does Tóth's method apply directly, or does it assume an affine (A0 + ρ·A1) structure?**

Verdict: PARTIAL — the paper's framework applies to any analytic matrix function of the scheduling variable (including rational functions), but the polynomial method generates a closed-form polynomial in p only when A_c(p) itself is polynomial in p. For rational A_c(Y), the polynomial method produces rational (not polynomial) A_d(Y), and the paper is silent on how to handle this.

Evidence: Section II (Definition 1, p. 3), Section III-B (eq. 9a), Section III-C-2 (eq. 12), Section V (Table II, "preservation of affine dependence" row, p. 21).

Key results:

1. Definition 1 (p. 3) states the system matrices are "analytic matrix functions on P." The paper makes no global assumption of affine or polynomial dependence for the continuous-time matrices. The methods are formally valid for any analytic A_c(p).

2. However, the polynomial method's practical utility depends on A_c(p) being polynomial in p. The paper explicitly notes (Section III-C, p. 8): "Identification and control-synthesis procedures are often based on the assumption of linear, polynomial, or rational (static) dependence on p_c, and hence it is required to develop approximative discretization methods that try to achieve good representation of the original behavior, but with a low complexity of the coefficient dependence."

3. Table II (p. 21) shows "preservation of affine dependence" is a property only of the Rectangular and Adams-Bashforth methods (marked "+"), while the polynomial method is marked "-". This means: if A_c(p) is affine in p, then A_d(p) from the polynomial method is generally NOT affine (it becomes polynomial of degree n). For rational A_c(p), A_d(p) from the polynomial method becomes rational of increasing complexity.

4. For the complete method (eq. 9a), A_d(p) = exp(A_c(p)·T_d), which for rational A_c(p) is transcendental in p — heavy nonlinear dependence, as the paper notes.

**Practical implication for our system**: Since M(Y)^(-1) is rational in Y, A_c(Y) is rational in Y. The polynomial discretization method gives A_d(Y) as a truncated series of powers of A_c(Y), which is rational (not polynomial) in Y. To get a polynomial A_d(Y) (Option A in the spec), the paper does not directly help. One approach consistent with the paper's framework: apply the polynomial expansion to an affine approximation of A_c(Y) (i.e., linearize M(Y)^(-1) around a nominal Y_0). The paper does not discuss this strategy.

---

**Q4: B_d(Y) treatment — Does B_d(Y) require the same polynomial expansion, or is the frozen-at-sampling-instant approach sufficient for B even if A needs a correction?**

Verdict: YES — the paper treats A_d and B_d symmetrically; both are derived from the same polynomial expansion (Section III-C-2, eq. 12). No asymmetric treatment is proposed or justified.

Evidence: Section III-C-2 (Polynomial method conversion table, p. 9), Section III-B (Complete method, eq. 9a), Section III-C-1 (Rectangular method table, p. 8).

Key result: In all discretization methods reviewed by the paper, B_d(p) is derived from the same approximation level as A_d(p). The polynomial method gives:

```
B_d(p_d(k)) = T_d · (I + sum_{l=1}^{n-1} (T_d^l / (l+1)!) · A_c^l(p_d(k))) · B_c(p_d(k))
```

The rectangular (n=1) case simplifies to B_d = T_d · B_c(p). The paper provides no result that would justify using a higher-order approximation for A_d while using the simpler frozen formula for B_d. Since B_c(Y) = [0; M(Y)^(-1)] shares the same rational Y-dependence as A_c(Y), the same approximation considerations apply to both.

One practical observation: at the rectangular level (n=1), B_d(Y) = T_d · B_c(Y), which preserves exactly the same rational Y-dependence as the continuous B_c(Y). This is the simplest choice and is differentiable in Y as long as M(Y) is invertible. For the training loop (Use case 2), this means B_d could be computed analytically from the physics without any Taylor expansion.

---

**Q5: Quasi-LPV considerations — Y is a state, not an exogenous signal. Does Tóth address quasi-LPV systems, and does it change the discretization approach?**

Verdict: PARTIAL — the paper acknowledges quasi-LPV systems (p. 5) but does not analyze them in detail; its formal results apply to the general-LPV (exogenous scheduling) case and are stated to be applicable to quasi-LPV by the same logic, with a caveat about the ZOH assumption on p_c.

Evidence: Section III-A (pp. 5–6, Assumption 1 discussion), Section IV-D (Switching effects, pp. 19–20).

Key result: The paper explicitly names quasi-LPV at p. 5 (ZOH setting discussion): "in the LPV framework, this setting, i.e. Assumption 1, is criticized as, in terms of the use of LPV models, p_c is often considered to be a measurable external/environmental effect (general-LPV) or some function of the states, inputs, or outputs of the system S (quasi-LPV). Therefore, in reality it is possibly not fully influenced by the digitally controlled actuators of the plant which contain the ZOH."

The paper's response to this is that the ZOH assumption on p_c is adopted as a mathematical convenience (to enable a DT description), not as a physical claim. For quasi-LPV systems where p_c = x_c (or a function of it), the paper notes (p. 20): "usually p_c is not actuated by ZOH and it changes smoothly and relatively slowly with respect to the actual dynamics of the plant" — which is exactly our case (Y-position changes at ≤ 0.125 mm/sample relative to the gantry dynamics).

The paper does NOT provide modified discretization formulas specifically for quasi-LPV. The same frozen-at-sampling-instant / complete / polynomial methods apply. The quasi-LPV structure means p_c_dot is not zero within the sampling interval (unlike the ZOH assumption), but for slowly-varying p_c this error is shown to be negligible (the M^(n) sensitivity constants bound it, and the p_c_dot term drops out under ZOH Assumption 2, eq. 24).

**Practical conclusion**: For our system at 16 kHz with Y-rate ≤ 2 m/s (0.125 mm/sample), the quasi-LPV nature of the scheduling does not change the recommended discretization approach. The frozen-at-sampling-instant assumption is justified.

---

## Part 2 — Decision summary

The Tóth et al. (2010) paper directly justifies **Use case 1 (validation simulation)** using the complete method (frozen-at-sampling-instant via `scipy.signal.cont2discrete` at each step): this method is theoretically errorless within the ZOH setting (Section III-B), and for slowly-varying scheduling such as our Y-position at 16 kHz, the switching-effect error is negligible (Section IV-D). For **Use case 2 (training loop)**, the paper's polynomial discretization (Section III-C-2, eq. 12) is the closest match to Option A (`A_d(Y) = A0_d + Y·A1_d + ...`), but it does not directly yield a polynomial A_d(Y) when A_c(Y) is rational in Y — it yields a rational function of increasing complexity. The paper is silent on the strategy of approximating M(Y)^(-1) before discretization, which would be required to make Option A tractable. Option D (hardcode A_d(Y) symbolically using the rectangular method A_d(Y) = I + T_d·A_c(Y), B_d(Y) = T_d·B_c(Y)) is also supported: Table II confirms the rectangular method preserves affine/polynomial dependence on p, and at T_d = 62.5 µs a linear (n=1) polynomial approximation of exp(A_c(Y)·T_d) is likely sufficient given the numerical example results in Section VII. The remaining open question is whether a rational approximation of M(Y)^(-1) — necessary for any closed-form Option A or D — can be evaluated and differentiated efficiently in PyTorch; this is not addressed by Tóth and requires a separate analysis. Recommended path: use the complete method for Use case 1 (no changes to `gantry_discrete_ss`), and implement Option D with the rectangular approximation A_d(Y) = I + T_d·A_c(Y) for Use case 2, evaluating M(Y)^(-1) analytically from the known physics — this mirrors the MSD block pattern, is torch-differentiable, and is justified by the paper's stability and error analysis at high sampling rates.
