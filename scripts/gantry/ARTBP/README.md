# ARTBP for the gantry augmentation: implementation and verification strategy

Status: Phase A done; Phase B done + the plan REFRAMED (D-120, 2026-07-22). Created 2026-07-22.

Reference: Corentin Tallec and Yann Ollivier, "Unbiasing Truncated Backpropagation
Through Time," 2017 (arXiv:1705.08209). Local copy:
`literature/stability-training/Unbiasing Truncated Backpropagation Through Time.pdf`.
Equation numbers below refer to that paper.

---

## 0. Phase B result and reframe (D-120, read first)

Phase B was built and run (`ground_truth.py`, then `instrument_select.py` = Phase B0). It overturned
the original plan below, so read this before §1-§5.

- The raw DC-direction GRADIENT does NOT converge on the z=1 (marginal) dY axis: its variance
  explodes ~H^3 while its mean is unresolvable (SE ~= mean, |t|<1.3 at every horizon). So there is
  **no convergent `true_grad(T)`** to serve as ground truth, and the pre-registered "~1/nf bias gap"
  framing (old §4) is void. The gradient blows up only because `g = -kappa(H) * c*(H)` (identity held
  to 3 sig figs): the curvature kappa explodes (~H^3.8) while the loss-optimum c* -> 0.
- The **loss-landscape** instrument (`c*(H)` = loss-optimal constant, `kappa(H)` = curvature, parabola
  fit + window bootstrap) is validated as the right, bounded instrument: kappa rel-SE <0.05% vs the
  gradient's ~42% (~1000x better); it recovers planted offsets +4e-6/+1e-6/+3e-7 to 99.8-100.1%.
- But the trained -4.5e-6 DC is **NOT the loss optimum**: `c*(H)` is small, convention-dependent
  (+1.7e-6 faithful ann-route, -2.44e-6 v11 state-row) and -> 0 with horizon. The DC is **Adam walking
  a near-flat, weakly-constrained direction** (small kappa at the training horizon); `kappa(H)` growing
  is the restoring curvature ARTBP supplies via rare long rollouts. This reconciles v3b (systematic
  gradient) with v11/SGD (flat direction / Adam artifact) and explains v12 (ARTBP collapses the DC).

Reframe (supersedes §4-§5): (1) adopt `kappa(H)` / `c*(H)` as the bounded MECHANISM instrument
(Phase B deliverable, in hand); (2) verify the FIX with a TRAINING-DYNAMICS instrument (the ANN's DC
trajectory under {fixed, ARTBP-geometric, ARTBP-poly-tail}, extending v12), NOT a gradient-unbiasedness
test. The paper's variance bound needs geometrically-decaying memory (Sec. 4); the z=1 axis has none,
so the poly-tail (Eq. 14) is necessary not optional, and variance is a live risk to measure. Data:
`data/{b_bias_gap,b0_instrument_select}.npz`; figures `figures/b0_{landscapes,instrument}.png`.

The sections below are the ORIGINAL Phase-A plan, kept for provenance; where they conflict with §0,
§0 wins.

---

## 1. Goal and scope

The learned augmentation develops a systematic constant (DC) on the K=0 dY row that
integrates into free-run drift. It was proven by intervention (run log v12, mechanism-proof
ARTBP vs fixed control) that this DC is driven by the **truncated-BPTT bias**: training on a
short fixed window (nf=400) gives a biased gradient that rewards a constant which helps
in-window but drifts over the full run. Longer FIXED windows do not fix it (DC ~ 1/nf, refuted
up to nf=3200). An UNBIASED gradient at the same average cost removes it.

This folder develops ARTBP properly and **verifies** it, replacing the quick diagnostic in
`gantry-zero-mean/v12_artbp.py`. Two questions to settle with numbers:

1. Is our windowed estimator actually **unbiased** for the long-horizon gradient?
2. Which truncation distribution gives the lowest **variance** at the same cost (our geometric
   vs the paper's recommended poly-tail, Eq. 14)?

Scope boundaries (what we are NOT claiming): this is not the literal streaming ARTBP (Eq. 11)
unless Phase D shows we need it; unbiasedness holds only up to the truncation cap; the goal here
is the DC-direction gradient (the phenomenon), not a full retrain of the deliverable.

---

## 2. Formula inventory and reuse decision

Reuse rule (`theory-context-must-match`): reuse a formula only if the assumptions it was derived
under hold in OUR setting. The decisive difference: the paper is **streaming** (one long sequence,
hidden state carried across subsequences, gradient flows across subsequence boundaries); ours is
**independent free-run windows** (each rollout starts fresh from an encoder init, no carried state).

| Paper eq. | What it is | Verdict for us | One-line justification |
|-----------|-----------|----------------|------------------------|
| Eq. 2 | total loss `L_T = Σ ℓ_t` | reuse (target) | our deployment objective is the full free-run loss over the horizon |
| Eq. 7-8 | exact BPTT gradient + backprop recursion | reuse (ground truth) | full BPTT over a feasible horizon T is our exact reference gradient |
| Eq. 9 | fixed truncated BPTT (length L) | reuse (control) | this is our `fixed nf` baseline, the biased comparator |
| Eq. 10 | `c_t = P(truncate at t)` | reuse (definition) | the truncation-probability schedule; general |
| **Prop. 1 / Eq. 13** | unbiasedness of random-truncation + compensation | **reuse the principle** | its only assumption (states independent of the truncation draw) holds for us |
| **Eq. 11** | per-step `1/(1-c_t)` compensation INSIDE backprop | **adapt / optional** | derived to restore gradient flow across subsequence boundaries in streaming; our independent windows have no such boundaries, so our loss-term reweighting is already unbiased. Implement as a variant only to compare variance. |
| **Eq. 14** | poly-tail `c_t = (α-1)/((α-2)L0 + δt)`, α=4 or 6 | **reuse (variance fix)** | derived purely to make compensation factors grow polynomially (finite variance for α>3) instead of exponentially; setting-agnostic. This is the main thing to adopt. |
| Eq. 15-16 | influence-balancing test system | skip (build own) | their validation task, not the algorithm; our system replaces it (their result is the conceptual template, see note below) |
| Eq. 17-24 | proof of Prop. 1 | cite only | background for the unbiasedness claim |

Note (validation template): the paper's influence-balancing experiment (Sec. 6.1, Fig. 3) is our
problem in miniature: "a parameter has a positive short term influence, but a negative long term one
that surpasses the short term effect", truncated BPTT gets the gradient sign wrong and diverges even
above the intrinsic timescale, lower lr does not help, ARTBP converges. Cite this as external
confirmation of the mechanism.

Our estimator vs the paper's, stated honestly:
- Paper: reweights the gradient FLOW per step (Eq. 11), needed for streaming cross-boundary flow.
- Ours: importance-reweights the LOSS TERMS, `w_k = 1/P(K>=k)`, then full BPTT within one window.
  Unbiased in our setting by the exchange argument `E[grad L_hat] = grad E[L_hat] = grad L_full`
  (valid because the horizon draw K is independent of the parameters and `E[L_hat] = L_full`).
- The paper's warning that "global reweighting between subsequences does not give unbiasedness" is
  about their streaming setting; it does not apply to our independent windows. Phase D confirms this
  empirically rather than by assertion.

---

## 3. Design: one configurable estimator

A single module `artbp.py` exposing a gradient estimator with two independent axes:

- `distribution`: `geometric` (constant c = 1/L0, our current) | `poly_tail` (Eq. 14, parameter α)
- `reweighting`: `loss_term` (ours) | `eq11` (paper's per-step compensation)

Shared settings: mean truncation length `L0` (matched to the `fixed nf` control for equal cost),
horizon cap `H_max`, and `K >= 2` (a length-1 window has only the detached init output, no gradient).

Everything else (model, encoder init, data windows, DC probe) reuses `demo_common.build_pipeline`
so the estimator is tested on the exact pipeline object.

---

## 4. Verification plan

All measurements on the **DC-direction gradient** first (the scalar that determines whether the
phenomenon forms); extend to the full parameter gradient only if the scalar result is ambiguous.

- Ground truth: `true_grad(T)` = exact full BPTT over T steps (no truncation), for the largest T
  where full BPTT is feasible on our model (sized in Phase B). T is set equal to the ARTBP cap
  `H_max`, so ARTBP is unbiased for exactly this horizon.
- Biased reference: `fixed_grad(nf)` = the standard truncated-window gradient. Plotting it against
  `true_grad(T)` directly visualizes the truncation bias (gap ~ 1/nf).
- For each ARTBP variant: draw N independent samples of the estimator, measure
  - bias = `mean(estimate) - true_grad(T)` (with its standard error),
  - variance = sample variance of the estimates.

### Pre-registered success criteria (write before running)

1. Unbiasedness: `|mean(estimate) - true_grad| < 2 * SE` for every correct variant (fixed is NOT
   expected to pass; that is the point).
2. Variance: report `Var(poly_tail)/Var(geometric)`; the paper predicts finite vs exponential, so a
   clear reduction is expected. The winning variant is the unbiased one with the lowest variance.
3. Application (Phase E): trained DC inside the +/-3e-7 clean-noise band; nf-RMS no worse than the
   fixed control; DC-trajectory variance across seeds lower than v12.

Falsification: if the poly-tail does NOT reduce variance vs geometric, or if our loss-term
reweighting shows a measurable bias, the reuse decision in Section 2 is wrong and gets revised.

---

## 5. Phases, steps, deliverables

Each phase de-risks the next; do not start a phase before its predecessor's deliverable exists.

**Phase A - strategy (this document).** Deliverable: this README with the filled formula table and
pre-registered criteria. DONE on write.

**Phase B - ground truth + biased reference.** `ground_truth.py`:
- `true_grad(T)`: exact full-BPTT DC-direction gradient over T steps; confirm feasible/stable at the
  chosen T (memory/compute check, pick T).
- `fixed_grad(nf)`: the biased truncated-window gradient.
- Figure: `fixed_grad(nf)` vs `true_grad(T)` across nf, showing the 1/nf bias gap.
Deliverable: the two functions + the bias-visualization figure.

**Phase C - ARTBP estimator.** `artbp.py` with the two axes (Section 3). Unit self-check: with
`distribution=geometric, reweighting=loss_term` it reproduces `v12_artbp.py`'s behavior (sanity).
Deliverable: `artbp_grad(config)` returning one stochastic estimate.

**Phase D - bias/variance verification (decisive).** `verify_bias_variance.py`:
- Run N samples per variant {geometric, poly_tail(α=4), poly_tail(α=6)} x {loss_term, [eq11]}.
- Bias/variance table vs `true_grad`; figure.
- Verdict against the pre-registered criteria: which variant is unbiased + lowest variance.
Deliverable: bias/variance table + figure + a one-paragraph verdict appended here.

**Phase E - application.** `train_artbp.py`:
- Train the augmentation with the winning variant; compare to the v12 fixed control and v12 ARTBP.
- Confirm DC collapse + lower DC-trajectory variance + preserved nf-RMS.
Deliverable: training run(s) + comparison figure; a D-090 run-log row before launch.

---

## 6. Conventions

- Any numerical formula/threshold carries an inline `# THEORY:` (paper eq.) or `# HEURISTIC:` label.
- Live-output convention for any run > a few seconds (unbuffered background, read the .output).
- Log non-trivial choices to `docs/decisions.md`; every training run gets a D-090 row before launch.
- No em-dashes anywhere (prose, code comments, figure text).
- Reuse `demo_common.build_pipeline`; do not touch `kamtin-fp-model/`.
