# Structural Identifiability of the Dual-Gantry LPV Model

## 1. Model structure

The equations of motion for the dual-gantry system in logical coordinates are

$$M(Y)\ddot{q} + C\dot{q} + Kq = Bu$$

where $q \in \mathbb{R}^3$ are the logical-coordinate positions, $u \in \mathbb{R}^3$ are the
logical-coordinate forces, and the mass matrix depends on the payload position $Y$ (the LPV
scheduling variable).

The full physical parameter vector is

$$\theta = [k_{b1},\ k_{b2},\ c_{g1},\ c_{g2},\ c_y,\ c_{b1},\ c_{b2},\ m_h,\ m_1,\ m_2,\ m_b,\ J_b,\ J_h,\ d] \in \mathbb{R}^{14}$$

The cross-arm length $L_b = 0.725\ \text{m}$ is fixed geometry (enters the coordinate transform
$P$ only) and is not trainable.

---

## 2. The observable map

$M(Y)$ is polynomial in $Y$, so its matrix coefficients are observable at different powers of
$Y$:

$$M(Y) = M_0 + M_1 Y + M_2 Y^2$$

The data can therefore determine the independent entries of
$\{M_0,\ M_1,\ M_2,\ K,\ C\}$ separately.
Listing every non-zero independent entry gives the observable map
$f: \mathbb{R}^{14} \to \mathbb{R}^{10}$:

| Matrix | Entry | Expression in $\theta$ |
|--------|-------|------------------------|
| $M_0$  | [0,0] | $m_1 + m_2 + m_b + m_h$ |
| $M_0$  | [0,1] | $(m_1 - m_2)\,L_b/2$ |
| $M_0$  | [1,1] | $J_\text{sum} + (m_1+m_2)\,L_b^2/4 + m_h d^2$ |
| $M_0$  | [1,2] | $-m_h\,d$ |
| $M_0$  | [2,2] | $m_h$ |
| $K$    | [1,1] | $k_{b,\text{sum}} = k_{b1}+k_{b2}$ |
| $C$    | [0,0] | $c_{g1}+c_{g2}$ |
| $C$    | [0,1] | $(c_{g1}-c_{g2})\,L_b/2$ |
| $C$    | [1,1] | $c_{b,\text{sum}} + (c_{g1}+c_{g2})\,L_b^2/4$ |
| $C$    | [2,2] | $c_y$ |

---

## 3. Jacobian rank

The Jacobian $\partial f/\partial\theta \in \mathbb{R}^{10\times14}$ has a near-block-triangular
structure. Ordering $\theta$ as
$[k_{b1}, k_{b2}, c_{g1}, c_{g2}, c_y, c_{b1}, c_{b2}, m_h, m_1, m_2, m_b, J_b, J_h, d]$,
the 10 rows are linearly independent by inspection, giving

$$\operatorname{rank}\!\left(\frac{\partial f}{\partial\theta}\right) = 10$$

The null space therefore has dimension $14 - 10 = 4$.

---

## 4. The four null directions

Solving $(\partial f/\partial\theta)\,v = 0$ yields four independent non-identifiable directions.

| Direction | Perturbation | Physical meaning | Treatment |
|-----------|-------------|-----------------|-----------|
| $n_1$ | $\Delta k_{b1} = +\varepsilon,\ \Delta k_{b2} = -\varepsilon$ | Stiffness split | Regularization — physical system has symmetric joints ($k_{b1}=k_{b2}$ by design); prior is physically motivated |
| $n_2$ | $\Delta c_{b1} = +\varepsilon,\ \Delta c_{b2} = -\varepsilon$ | Damping split | Same argument as $n_1$ |
| $n_3$ | $\Delta J_b = +\varepsilon,\ \Delta J_h = -\varepsilon$ | Inertia split | Regularization on log-ratio; symmetric design justification |
| $n_4$ | $\Delta m_1 = \Delta m_2 = +\varepsilon,\ \Delta m_b = -2\varepsilon,\ \Delta J_\text{sum} = -(L_b^2/2)\,\varepsilon$ | Mass–inertia coupling | **Reparameterize** — no physical symmetry argument; regularization would identify the prior, not the physics |

For $n_1$–$n_3$, regularization is acceptable because the prior encodes **known physical
symmetry** of the mechanical design (equal joints), not an arbitrary guess.
For $n_4$, no such symmetry exists: shifting mass equally between $m_1$, $m_2$ and $m_b$
while compensating $J_\text{sum}$ has no physical motivation. Regularizing $n_4$ would make
the recovered values depend on the regularization weight, not on the measured data.

**The correct treatment for $n_4$ is reparameterization** — replace the five parameters
$\{m_1, m_2, m_b, J_b, J_h\}$ with the three identifiable combinations
$\{m_\text{total},\ m_\text{diff},\ J_\text{eff}\}$ (see Section 6).

---

## 5. Experimental verification of $n_4$

Training results confirm the theoretical null direction exactly.
Starting from 10% detuned initial values, the optimizer converges to:

| Parameter | True | Learned | $\Delta$ |
|-----------|------|---------|----------|
| $m_1$ | 10.200 | 10.452 | $+0.252$ |
| $m_2$ | 10.700 | 10.952 | $+0.252$ |
| $m_b$ | 22.800 | 22.296 | $-0.504$ |
| $J_\text{sum}$ | 1.050 | 0.984 | $-0.066$ |

Setting $\varepsilon = 0.252$ and checking $n_4$:

$$\Delta m_b = -2\varepsilon = -0.504 \quad \checkmark$$
$$\Delta J_\text{sum} = -\frac{L_b^2}{2}\,\varepsilon = -\frac{0.725^2}{2}\times 0.252 = -0.066 \quad \checkmark$$

All other identifiable quantities ($m_h$, $m_\text{diff} = m_1 - m_2$, etc.) are recovered to
within numerical precision. The optimizer found a valid point on the flat manifold; $\varepsilon$
is not a hyperparameter but the magnitude of drift along $n_4$ determined by the optimizer
trajectory from the detuned initialization.

---

## 6. The 10 identifiable combinations

| # | Quantity | Expression | From |
|---|---------|------------|------|
| 1 | $m_h$ | $m_h$ | $M_0[2,2]$ |
| 2 | $d$ | $d$ | $M_0[1,2]\,/\,(-m_h)$ |
| 3 | $m_\text{diff}$ | $m_1 - m_2$ | $M_0[0,1]$ |
| 4 | $m_\text{total}$ | $m_1 + m_2 + m_b$ | $M_0[0,0]$ |
| 5 | $J_\text{eff}$ | $J_\text{sum} + (m_1+m_2)\,L_b^2/4$ | $M_0[1,1]$ |
| 6 | $k_{b,\text{sum}}$ | $k_{b1}+k_{b2}$ | $K[1,1]$ |
| 7 | $c_{g1}$ | $c_{g1}$ | $C[0,0]$ and $C[0,1]$ combined |
| 8 | $c_{g2}$ | $c_{g2}$ | $C[0,0]$ and $C[0,1]$ combined |
| 9 | $c_{b,\text{sum}}$ | $c_{b1}+c_{b2}$ | $C[1,1]$ |
| 10 | $c_y$ | $c_y$ | $C[2,2]$ |

Note that $c_{g1}$ and $c_{g2}$ **are individually identifiable**: the damping matrix $C$
encodes both their sum (via $C[0,0]$) and their difference scaled by $L_b$ (via $C[0,1]$),
which together uniquely determine both scalars.

---

## 7. Reparameterization

Replace $\{m_1, m_2, m_b, J_b, J_h\}$ (5 parameters) with
$\{m_\text{total}, m_\text{diff}, J_\text{eff}\}$ (3 parameters).
This eliminates $n_4$ from the parameter space entirely.

The physical matrices are then reconstructed as:
$$m_1 = \tfrac{1}{2}(m_\text{total} - m_b^* + m_\text{diff}), \quad m_2 = \tfrac{1}{2}(m_\text{total} - m_b^* - m_\text{diff})$$

where $m_b^*$ must be fixed (e.g. from design specifications), or alternatively,
$m_\text{total}$ and $m_\text{diff}$ are used directly as the matrix entries:

$$M_0[0,0] = m_\text{total} + m_h, \qquad M_0[0,1] = \frac{m_\text{diff}\,L_b}{2}, \qquad M_0[1,1] = J_\text{eff} + m_h d^2$$

This yields a well-posed, minimal 10-parameter model in which every free parameter is
independently determined by the data.
