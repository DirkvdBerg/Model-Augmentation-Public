# FP Model Structure Reference

Static reference for `kamtin-fp-model/`. Describes what each file contains and how files relate to each other. Source files are immutable — this document must be updated if they change.

---

## File Map

```
kamtin-fp-model/
├── 02 EulerLagrange/
│   ├── get_eom.m              ← entry point: defines Lagrangian, calls EulerLagrange()
│   ├── EulerLagrange.m        ← symbolic EOM derivation engine (generic, not gantry-specific)
│   ├── Gantry_sys_temp.m      ← auto-generated output: full nonlinear ODE (with Coriolis)
│   └── Gantry_sys.m           ← (same as _temp, earlier version)
│
└── 03 Simulink gantry/
    ├── main.m                 ← entry point: parameters, state-space, simulation, plots
    ├── gantry_2025a.slx       ← Simulink model: compares three model variants
    ├── functions/
    │   ├── gantrySystem.m          ← simplified EOM (no Coriolis): f(u,x,params) → dxdt
    │   ├── gantrySystemCoriolisCentripetal.m  ← full nonlinear EOM: f(u,x,params) → dxdt
    │   ├── getss.m                 ← builds continuous-time SS object from M, C, K
    │   ├── ruleOfThumb.m           ← feedback controller design utility
    │   └── thirdOrderSetpointETEL.m ← setpoint trajectory generator
    └── slprj/                 ← Simulink build cache (ignore)
```

---

## How the Files Relate

```
get_eom.m
  → defines symbolic Lagrangian T, V, D
  → calls EulerLagrange(L, q, Q_i, Q_e, D, par, 'm', 'Gantry_sys_temp')
       → writes Gantry_sys_temp.m  (auto-generated, do not edit)

main.m
  → defines numerical parameters (mb, mh, m1, m2, Jb, Jh, ...)
  → constructs M(Y), C, K matrices directly
  → calls getss(n, M, C, K)  →  returns sys (MATLAB ss object)
  → calls c2d(sys, ts, 'zoh')  →  discrete-time SS at fixed Y
  → calls sim('gantry_2025a')  →  runs Simulink model

gantry_2025a.slx
  → subsystem "Single H-gantry"    uses Simscape physical model
  → subsystem "MATLAB Function1"   wraps gantrySystem.m        (no Coriolis)
  → subsystem "MATLAB Function2"   wraps gantrySystemCoriolisCentripetal.m  (full EOM)
  → LTI System blocks              use discrete SS from getss via main.m workspace
```

---

## `02 EulerLagrange/get_eom.m`

Defines the Lagrangian symbolically and triggers code generation.

**Generalised coordinates:** `q = {X, Theta, Y, dX, dTheta, dY}`

**Lagrangian:**
```matlab
T = 0.5*((m1+m2+mb+mh)*dX^2 + ...
    (Jb+Jh+mh*(d^2+Y^2)+(m1+m2)*cos(Theta)^2*Lb^2/4)*dTheta^2) + ...
    0.5*mh*(dY^2 - 2*d*dY*dTheta - 2*dX*(cos(Theta)*Y*dTheta + sin(Theta)*(dY-d*dTheta))) + ...
    (m1-m2)*dX*dTheta*cos(Theta)*Lb/2

V = 0.5*(kb1+kb2)*Theta^2

D = 0.5*((cg1+cg2)*dX^2 + 2*(cg1-cg2)*dX*dTheta*cos(Theta)*Lb/2 + ...
    ((cb1+cb2)+(cg1+cg2)*cos(Theta)^2*Lb^2/4)*dTheta^2 + cy*dY^2)
```

**External forces:** `Q_e = {Qe1, Qe2, Qe3}` (forces/torques on X, Theta, Y)

**Output:** generates `Gantry_sys_temp.m`

---

## `02 EulerLagrange/EulerLagrange.m`

Generic Euler-Lagrange engine (MathWorks, 2015–2016). Not gantry-specific.

**Signature:** `VF = EulerLagrange(L, X, Q_i, Q_e, R, par, 'm', filename)`

Applies `d/dt(∂L/∂q̇) - ∂L/∂q + ∂R/∂q̇ = Q_i + Q_e` for each generalised coordinate, converts to vector field via `odeToVectorField`, writes result as a MATLAB function file.

---

## `02 EulerLagrange/Gantry_sys_temp.m` (auto-generated)

**Signature:** `dxdt = Gantry_sys_temp(t, X, Qe1, Qe2, Qe3, m1, m2, mb, mh, Jb, Jh, d, Lb, kb1, kb2, cg1, cg2, cb1, cb2, cy)`

Full nonlinear ODE including Coriolis and centripetal terms. Auto-generated — do not read for model insight, look at `get_eom.m` instead.

**State index mapping inside this file:**
```
X(1) = X,      X(2) = Theta,   X(3) = Y
X(4) = dX,     X(5) = dTheta,  X(6) = dY
```

**Output:** `dxdt = [dX; dTheta; dY; ddX; ddTheta; ddY]`

---

## `03 Simulink gantry/functions/gantrySystem.m`

**Signature:** `dxdt = gantrySystem(u, x, m1, m2, mb, mh, Lb, Jb, Jh, d, cg1, cg2, cb1, cb2, cy, kb1, kb2)`

Simplified EOM — ignores Coriolis and centripetal terms. Constructs M(Y), C, K and evaluates `dxdt = A*x + B*u`.

**Important: dead C matrix on line 23**
```matlab
C = [eye(3), zeros(3)]   % BUG: this line should be commented out
```
This line redefines `C` (silently overwriting the damping matrix) but is never used — the function only returns `dxdt`, not an output `y = C*x`. The coordinate transform is handled externally in Simulink by Gain4 (= P.'). This line should be `% C = [eye(3), zeros(3)]`.

The correct output C matrix for stage coordinates is `P.' * [eye(3), zeros(3)]` — identical to G's C — but this is applied by the Gain blocks in the Simulink model, not inside this function.

**Mass Matrix M(Y)** — Y = x(3):
```matlab
M = [ m1+m2+mb+mh,              (m1-m2)*Lb/2 - mh*Y,               0   ]
    [ (m1-m2)*Lb/2 - mh*Y,   Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y^2,  -mh*d ]
    [ 0,                         -mh*d,                              mh  ]
```

**Damping Matrix C** — constant:
```matlab
C = [ cg1+cg2,              (cg1-cg2)*Lb/2,               0  ]
    [ (cg1-cg2)*Lb/2,   cb1+cb2+(cg1+cg2)*Lb^2/4,         0  ]
    [ 0,                         0,                        cy ]
```

**Stiffness Matrix K** — constant:
```matlab
K = [ 0,   0,        0 ]
    [ 0,   kb1+kb2,  0 ]
    [ 0,   0,        0 ]
```

**State-space matrices (inline, for integration):**
```matlab
A = [ zeros(3),  eye(3);
     -M\K,      -M\C  ]
B = [ zeros(3); pinv(M) ]
```

**Coordinate system note:** `gantrySystem.m` uses logical-coordinate dynamics (M, C, K are in logical coordinates). The internal states integrated in Simulink are logical [X, Θ, Y, dX, dΘ, dY]. The Simulink model handles coordinate transforms externally (Gain3 = P on input, Gain4 = P.' on output).

---

## `03 Simulink gantry/functions/gantrySystemCoriolisCentripetal.m`

**Signature:** `dxdt = gantrySystemCoriolisCentripetal(u, x, m1, m2, mb, mh, Jb, Jh, d, Lb, kb1, kb2, cg1, cg2, cb1, cb2, cy)`

Full nonlinear EOM (same result as `Gantry_sys_temp.m`, repackaged for Simulink). The long `et1...et18` expressions are the symbolic expansion of the full Euler-Lagrange equations including Coriolis and centripetal terms.

**Same state index mapping as `Gantry_sys_temp.m`.**

---

## `03 Simulink gantry/functions/getss.m`

**Signature:** `[sys, A, B, C, D] = getss(n, M, C, K)`

Builds a MATLAB `ss` object from M, C, K (at a fixed, caller-supplied M).

```matlab
A = [ zeros(n),  eye(n);
      -M\K,     -M\C  ]
B = [ zeros(n); eye(3)/M ]
C = [ eye(3), zeros(n) ]    % outputs = positions only
D = zeros(3)
```

Called from `main.m` as `getss(3, M, C, K)` where M is evaluated at a fixed Y.

---

## `03 Simulink gantry/main.m`

Entry point for analysis and simulation.

**What it does (in order):**
1. Defines all numerical parameters
2. Evaluates M(Y) at `Y = 0.3` m (fixed operating point)
3. Computes mode shapes via `eigs(K, M, n, 'smallestabs')`
4. Calls `getss(3, M, C, K)` → `sys` (continuous-time SS in logical coordinates)
5. Constructs coordinate transformations P (logical ↔ stage) and Psi (stage ↔ modal)
6. Discretises: `G = c2d(StageCoordinatesSystem, ts, 'zoh')` at `fs = 16 kHz`
7. Runs Simulink model `gantry_2025a` for comparison
8. Runs `lsim` on discrete model and plots residuals

**Coordinate transformation matrix P** (logical → stage):
```matlab
P = [ 1,     1,    0 ]
    [ Lb/2, -Lb/2, 0 ]
    [ 0,     0,    1 ]
T = pinv(P')
```
Stage system: `sys_stage = P' * sys_logical * P`

**Numerical parameters:**
```
mb=22.8 kg, mh=10.1 kg, m1=10.2 kg, m2=10.7 kg
Jb=1.0 kg·m², Jh=0.05 kg·m²
cg1=14.5, cg2=20.3 N·s/m  (X viscous friction)
cy=10 N·s/m  (Y viscous friction)
cb1=cb2=9 N·m·s/rad  (elastic joint friction)
kb1=kb2=1987.5 N·m/rad  (elastic joint stiffness)
Lb=0.725 m, d=0.1 m
cc1=16.8, cc2=18.35 N  (Coulomb friction, not in SS model)
ccy=11.6 N  (Y Coulomb friction, not in SS model)
```

---

## `gantry_2025a.slx` — Block Structure (extracted)

### Root level

| Block | Type | Role |
|-------|------|------|
| Single H-gantry | SubSystem | Simscape physical model (ground truth) |
| MATLAB Function1 | SubSystem | Wraps `gantrySystem.m` (no Coriolis) + explicit Integrator |
| MATLAB Function2 | SubSystem | Wraps `gantrySystemCoriolisCentripetal.m` + explicit Integrator |
| LTI System / LTI System1 / LTI System2 | Reference | Discrete SS from `getss` workspace variable |
| Gain1 (SID=161), Gain3 (SID=148) | Gain | Matrix `P` — converts stage forces → logical forces (input side) |
| Gain2 (SID=162), Gain4 (SID=149) | Gain | Matrix `P.'` — converts logical positions → stage positions (output side) |
| Selector1 (SID=154), Selector2 (SID=172) | Selector | Selects indices [1,2,3] from 6-state integrator → logical positions [X,Θ,Y] |
| Integrator / Integrator1 | Integrator | Integrates `dxdt`; 6 states in logical coordinates |
| Reference / Reference1 / Reference2 | FromWorkspace | Reference trajectory `[t, r]` |
| Feedforward / Feedforward1 / Feedforward2 | FromWorkspace | Feedforward force `[t, f]` |
| To Workspace `q` (SID=59) | ToWorkspace | Simscape output — stage positions [X1, X2, Y] |
| To Workspace `q1` (SID=141) | ToWorkspace | Simplified EOM output — stage positions [X1, X2, Y] |
| To Workspace `q2` (SID=175) | ToWorkspace | Full EOM output — stage positions [X1, X2, Y] |
| Error / Error1 / Error2 | ToWorkspace | Tracking errors `e`, `e1`, `e2` |
| Sum (×6) | Sum | Feedback error and force summation |

**Signal flow for MATLAB Function paths (q1, q2):**
```
Stage forces [F_X1, F_X2, F_Y]
  → Gain3/Gain1 (×P) → logical forces [F_X, F_Θ, F_Y]
  → MATLAB Function (gantrySystem / gantrySystemCoriolisCentripetal)
  → dxdt → Integrator → logical states [X, Θ, Y, dX, dΘ, dY]
  → Selector[1,2,3] → logical positions [X, Θ, Y]
  → Gain4/Gain2 (×P.') → stage positions [X1, X2, Y]
  → To Workspace q1 / q2
```

**Signal flow for Simscape path (q):**
```
Stage forces [F_X1, F_X2, F_Y]
  → Single H-gantry (Simscape, nonlinear, includes Coulomb friction)
  → physical outports x1, x2, y (directly stage positions)
  → Mux → To Workspace q
```

---

### Workspace output variables — coordinate systems

All four output signals are in **stage coordinates [X1, X2, Y]** and are directly comparable:

| Variable | Source | Linearized? | Coriolis? | Coulomb friction? |
|----------|--------|-------------|-----------|-------------------|
| `q` | Simscape physical model | No (nonlinear) | Yes | Yes |
| `q1` | `gantrySystem.m` + P.' | Yes | No | No |
| `q2` | `gantrySystemCoriolisCentripetal.m` + P.' | Partial | Yes | No |
| `q3` | `lsim(G, ...)` in `main.m` post-processing | Yes | No | No |

`q3` is NOT a Simulink workspace variable — it is computed in `main.m` after the simulation by calling `lsim` on `G = c2d(StageCoordinatesSystem, ts, 'zoh')`. G's C matrix = `P.' * [I, 0]` maps logical states to stage positions, consistent with q1/q2.

The Simscape model (q) is the ground truth for the nonlinear system. The linearization gap between q3 and q (4–16 µm at Y=0.3 m, from Step 1 validation) is what the augmentation must learn to close.

### Subsystem: Single H-gantry (system_47)

Simscape multibody model. Inputs: `Fx1, Fx2, Fy`. Outputs: `x1, x2, y`.

| Block | Type | Role |
|-------|------|------|
| X1, X2 planar | Reference | Simscape planar joint for X actuators |
| X1 mass, X2 mass | Reference | Actuator masses |
| Y payload | Reference | Payload mass on Y-axis |
| Y-beam | Reference | Y-axis beam body |
| Prismatic Joint | Reference | Y-axis prismatic joint |
| Rigid Transform ×4 | Reference | Frame connections |
| World Frame | Reference | Inertial reference frame |
| Mechanism Configuration | Reference | Simscape solver settings |
| Solver Configuration | Reference | Simulink-Simscape interface |
| Sign ×3 | Signum | `sign(velocity)` for Coulomb friction |
| Gain (cc1, cc2, ccy) | Gain | Coulomb friction magnitudes |
| Sum ×3 | Sum | Applied force = input force − Coulomb friction |
| PS-Simulink Converter ×6 | Reference | Physical signal → Simulink |
| Simulink-PS Converter ×3 | Reference | Simulink → physical signal |

### Subsystem: MATLAB Function1 and Function2 (system_88, system_165)

Same structure for both. Wraps the respective MATLAB ODE function as an S-Function.

| Block | Type | Role |
|-------|------|------|
| u (Inport) | Inport | Force input |
| x (Inport) | Inport | Current state |
| SFunction | S-Function | Calls `gantrySystem` or `gantrySystemCoriolisCentripetal` |
| Demux | Demux | Splits S-Function outputs |
| dxdt (Outport) | Outport | State derivative output |
| Terminator | Terminator | Unused output port |
