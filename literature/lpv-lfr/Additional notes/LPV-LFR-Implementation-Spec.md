# LPV-LFR Implementation Specification
## Gantry Baseline — True LFR-First Forward Pass

This document derives the exact formulas required to implement the gantry baseline model as a
genuine LPV-LFR interconnection, not a collapsed LPV-SS model.
The audit criterion is stated at the end.

---

## 1. The Model Object: $(G, \Delta(Y))$

The LPV-LFR model is defined by two explicit objects.

### 1.1 Constant interconnection matrix $G$

$$G = \begin{bmatrix} A_x & B_w & B_u \\ C_z & D_{zw} & D_{zu} \\ C_y & D_{yw} & D_{yu} \end{bmatrix} \in \mathbb{R}^{15 \times 15}$$

All submatrices are **constant** (no dependence on $Y$):

$$A_x = \begin{bmatrix} 0 & I_3 \\ -M_0^{-1}K & -M_0^{-1}C \end{bmatrix} \in \mathbb{R}^{6\times6}$$

$$B_w = \begin{bmatrix} 0 & 0 \\ -M_0^{-1}M_1 & -M_0^{-1}M_2 \end{bmatrix} \in \mathbb{R}^{6\times6}$$

$$B_u = \begin{bmatrix} 0 \\ M_0^{-1} \end{bmatrix} \in \mathbb{R}^{6\times3}$$

$$C_z = \begin{bmatrix} -M_0^{-1}K & -M_0^{-1}C \\ 0 & 0 \end{bmatrix} \in \mathbb{R}^{6\times6}, \qquad D_{zw} = \begin{bmatrix} -M_0^{-1}M_1 & -M_0^{-1}M_2 \\ I_3 & 0 \end{bmatrix} \in \mathbb{R}^{6\times6}, \qquad D_{zu} = \begin{bmatrix} M_0^{-1} \\ 0 \end{bmatrix} \in \mathbb{R}^{6\times3}$$

$$C_y = \begin{bmatrix} I_3 & 0 \end{bmatrix} \in \mathbb{R}^{3\times6}, \qquad D_{yw} = 0 \in \mathbb{R}^{3\times6}, \qquad D_{yu} = 0 \in \mathbb{R}^{3\times3}$$

These are precomputed once from physical parameters and stored as fixed tensors.

### 1.2 Scheduling block $\Delta(Y)$

$$\Delta(Y) = Y I_6 \in \mathbb{R}^{6\times6}$$

This is the explicit scheduling object. It is constructed at each forward pass from the
current scheduling variable $Y$, **not absorbed into any matrix product**.

---

## 2. LFR Signal Flow (Mandatory Ordering)

The forward pass must follow this exact sequence:

$$\underbrace{x,\, u,\, Y}_{\text{inputs}} \;\xrightarrow{\;\text{Step 1}\;}\; \Delta(Y) \;\xrightarrow{\;\text{Step 2}\;}\; \text{rhs} = C_z x + D_{zu} u \;\xrightarrow{\;\text{Step 3}\;}\; z = L(Y)^{-1}\,\text{rhs} \;\xrightarrow{\;\text{Step 4}\;}\; w = \Delta(Y)z \;\xrightarrow{\;\text{Step 5}\;}\; \dot{x} = A_x x + B_w w + B_u u$$

Steps 3–5 must happen **in this order**. $\dot{x}$ must be driven through $B_w w$, never directly from the solved acceleration $a$.

---

## 3. Loop Matrix and Its Analytical Inverse

### 3.1 Loop matrix

$$L(Y) := I_6 - D_{zw}\,\Delta(Y) = \begin{bmatrix} I_3 + Y M_0^{-1}M_1 & Y M_0^{-1}M_2 \\ -Y I_3 & I_3 \end{bmatrix}$$

### 3.2 Right-hand side

Since $D_{yu} = 0$ and $D_{yw} = 0$, the loop RHS evaluates to:

$$\text{rhs} = C_z x + D_{zu} u = \begin{bmatrix} M_0^{-1} f_\text{net} \\ 0 \end{bmatrix} \in \mathbb{R}^6$$

where the net force is:

$$f_\text{net} = [-K,\,-C]\,x + u \in \mathbb{R}^3$$

### 3.3 Analytical loop solution

The analytical inverse $L(Y)^{-1}$ applied to this specific RHS gives:

$$z = L(Y)^{-1}\,\text{rhs} = \begin{bmatrix} M(Y)^{-1} f_\text{net} \\ Y\,M(Y)^{-1} f_\text{net} \end{bmatrix} \in \mathbb{R}^6$$

$$w = \Delta(Y)\,z = Y\,z = \begin{bmatrix} Y\,M(Y)^{-1} f_\text{net} \\ Y^2 M(Y)^{-1} f_\text{net} \end{bmatrix} \in \mathbb{R}^6$$

where $M(Y)^{-1}$ is the rational function derived in `LPV-LFR-Rational-rewrite.md`.

---

## 4. Polynomial Expression for $M(Y)^{-1}$

### 4.1 Scalar shorthand constants (precomputed once)

$$\alpha = m_1 + m_2 + m_b + m_h$$

$$\beta = \frac{(m_1 - m_2)L_b}{2}$$

$$\gamma = J_b + J_h + \frac{(m_1+m_2)L_b^2}{4} \qquad \textbf{(without } m_h d^2\textbf{)}$$

### 4.2 Denominator polynomial in $Y$ (scalar)

$$d(Y) = m_h \!\left(\alpha\gamma - \beta^2 + 2\beta m_h Y + m_h(\alpha - m_h)Y^2\right)$$

This is always strictly positive for physically admissible parameters (proved via $M(Y) \succ 0$).

### 4.3 Adjugate polynomial coefficient matrices (precomputed once)

$$N_0 = \begin{bmatrix} m_h\gamma & -\beta m_h & -\beta d\, m_h \\ -\beta m_h & \alpha m_h & \alpha d\, m_h \\ -\beta d\, m_h & \alpha d\, m_h & \alpha(\gamma + m_h d^2) - \beta^2 \end{bmatrix}$$

$$N_1 = \begin{bmatrix} 0 & m_h^2 & d\, m_h^2 \\ m_h^2 & 0 & 0 \\ d\, m_h^2 & 0 & 2\beta m_h \end{bmatrix}$$

$$N_2 = \begin{bmatrix} m_h^2 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & \alpha m_h - m_h^2 \end{bmatrix}$$

### 4.4 Rational inverse

$$M(Y)^{-1} = \frac{N(Y)}{d(Y)}, \qquad N(Y) = N_0 + Y N_1 + Y^2 N_2$$

### 4.5 Acceleration vector

$$a := M(Y)^{-1} f_\text{net} = \frac{N(Y)\, f_\text{net}}{d(Y)} \in \mathbb{R}^3$$

This is the upper half of $z$; the lower half is $Y a$.

---

## 5. Complete Forward Pass (PyTorch Pseudocode)

```python
# ================================================================
# PRECOMPUTED ONCE — all constant, stored as tensors
# ================================================================
# Physical parameter scalars
alpha = m1 + m2 + mb + mh
beta  = (m1 - m2) * Lb / 2
gamma = Jb + Jh + (m1 + m2) * Lb**2 / 4   # WITHOUT mh*d^2

# G submatrices (6x6, 6x6, 6x3, 6x6, 6x6, 6x3, 3x6)
M0inv = torch.linalg.inv(M0)
Ax  = torch.cat([torch.cat([Z33, I3],       dim=1),
                 torch.cat([-M0inv@K, -M0inv@C], dim=1)], dim=0)
Bw  = torch.cat([torch.cat([Z33, Z33],          dim=1),
                 torch.cat([-M0inv@M1, -M0inv@M2], dim=1)], dim=0)
Bu  = torch.cat([Z33, M0inv], dim=0)
Cz  = torch.cat([torch.cat([-M0inv@K, -M0inv@C], dim=1),
                 torch.cat([Z33, Z33],            dim=1)], dim=0)
Dzw = torch.cat([torch.cat([-M0inv@M1, -M0inv@M2], dim=1),
                 torch.cat([I3, Z33],              dim=1)], dim=0)
Dzu = torch.cat([M0inv, Z33], dim=0)
Cy  = torch.cat([I3, Z33], dim=1)

# Adjugate coefficient matrices (3x3)
N0 = torch.tensor([[mh*gamma,         -beta*mh,            -beta*d*mh         ],
                   [-beta*mh,          alpha*mh,             alpha*d*mh        ],
                   [-beta*d*mh,        alpha*d*mh,  alpha*(gamma+mh*d**2)-beta**2]])
N1 = torch.tensor([[0,        mh**2,     d*mh**2],
                   [mh**2,    0,         0      ],
                   [d*mh**2,  0,         2*beta*mh]])
N2 = torch.tensor([[mh**2, 0,  0               ],
                   [0,     0,  0               ],
                   [0,     0,  alpha*mh-mh**2  ]])

# ================================================================
# FORWARD PASS  —  called at each time step
# ================================================================
def lfr_forward(x, u, Y, Ax, Bw, Bu, Cz, Dzw, Dzu, Cy,
                N0, N1, N2, K, C, alpha, beta, gamma, mh, d_param):
    """
    Inputs:
        x  : (batch, 6)  state  [q; qdot]  in logical coordinates
        u  : (batch, 3)  input  f_ell
        Y  : (batch,)    scheduling variable

    Outputs: xdot, z, w, y
    """
    # Step 1 — Explicit scheduling block  Delta(Y) = Y * I6
    #   Not materialised as a full matrix here; applied as scalar multiplication
    #   but conceptually it is the explicit Delta block of the model.

    # Step 2 — RHS of loop equation:  C_z x + D_zu u  = [M0inv f_net; 0]
    q    = x[:, :3]
    qdot = x[:, 3:]
    f_net = -(q @ K.T) - (qdot @ C.T) + u              # (batch, 3)
    # rhs_upper = M0inv @ f_net  — embedded inside z computation below

    # Step 3 — Solve loop analytically: z = L(Y)^{-1} rhs
    #   Denominator polynomial d(Y)
    d_Y = mh * (alpha*gamma - beta**2
                + 2*beta*mh * Y
                + mh*(alpha - mh) * Y**2)               # (batch,)

    #   Adjugate polynomial N(Y) = N0 + Y*N1 + Y^2*N2
    Ye  = Y[:, None, None]                               # (batch, 1, 1)
    N_Y = N0 + Ye * N1 + Ye**2 * N2                     # (batch, 3, 3)

    #   a = M(Y)^{-1} f_net  =  N(Y) f_net / d(Y)
    a = (N_Y @ f_net.unsqueeze(-1)).squeeze(-1) / d_Y[:, None]  # (batch, 3)

    #   z = [a; Y*a]   — upper half from analytical loop, lower from Delta chain
    z = torch.cat([a, Y[:, None] * a], dim=-1)          # (batch, 6)

    # Step 4 — w = Delta(Y) z = Y * z
    w = Y[:, None] * z                                   # (batch, 6)

    # Step 5 — xdot = A_x x + B_w w + B_u u   (THROUGH G, not directly from a)
    xdot = (x @ Ax.T) + (w @ Bw.T) + (u @ Bu.T)        # (batch, 6)

    # Step 6 — y = C_y x
    y = x @ Cy.T                                         # (batch, 3)

    return xdot, z, w, y
```

---

## 6. LFR-First Audit Checklist

The following tests distinguish a genuine LFR-first implementation from a secretly collapsed one.

### Structural audit (code-level)

| Test | LFR-first (PASS) | Collapsed (FAIL) |
|------|-----------------|-----------------|
| Are `Ax`, `Bw`, `Bu`, `Cz`, `Dzw`, `Dzu`, `Cy` stored as explicit tensors? | Yes | No — never constructed |
| Is `Delta(Y) = Y*I6` an explicit object or step? | Yes | No |
| Is `z` computed **before** `xdot`? | Yes | No — `xdot` computed from `a` first |
| Is `xdot` computed as `Ax@x + Bw@w + Bu@u`? | Yes | No — computed as `cat([qdot, a])` |
| Do `z` and `w` feed into `xdot` through `Bw`? | Yes | No — `w` is reconstructed after |

### The single decisive test

In the collapsed implementation:
```python
xdot = torch.cat([x[:, 3:], a], dim=-1)   # w plays no role in xdot
```

In the LFR implementation:
```python
xdot = (x @ Ax.T) + (w @ Bw.T) + (u @ Bu.T)   # w actively drives xdot
```

Both produce numerically identical `xdot`, but **only the second one is a genuine evaluation
of the LFR interconnection**. The test is whether `w` is causally upstream of `xdot` in the
computation graph.

### Signal flow audit (verify with autograd)

```python
# If the implementation is genuinely LFR-first, a gradient injected into w
# must propagate through Bw into xdot.
# Check: d(xdot) / d(w) == Bw   (not zero)
```

### Numerical equivalence

The LFR-first pass must produce results that match the collapsed pass to within floating-point
tolerance for all valid $(x, u, Y)$:

```python
assert torch.allclose(xdot_lfr, xdot_collapsed, atol=1e-10)
assert torch.allclose(z_lfr,    z_collapsed,    atol=1e-10)
assert torch.allclose(w_lfr,    w_collapsed,    atol=1e-10)
```

Numerical equivalence is **necessary but not sufficient** for LFR-first structure.
The structural tests above are the actual criterion.

---

## 7. What This Enables vs. the Current Code

| Capability | Current (collapsed) | This spec (LFR-first) |
|---|---|---|
| $G$ available for synthesis | No | Yes |
| $\Delta(Y)$ explicit | No | Yes |
| Augmentation by enlarging $G, \Delta$ | Cannot — no interface | Yes — append rows/cols |
| Coupling structure (single repeated $Y$) visible to synthesis | No | Yes |
| Numerically efficient (no 6×6 solve) | Yes (3×3) | Yes (polynomial evaluation) |
| Autograd through $Y$ | Yes | Yes |
