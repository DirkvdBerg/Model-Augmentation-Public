# Decisions: orthogonal addition (OA-xxx)

Logged BEFORE implementing. Handoff: `tasks/handoffs/2026-09-23-orthogonal-addition.md`.
The clock times in the headers are estimates made during the session and drift by up to about
an hour from real time; real times are in each run's `outputs/<run>/resources.csv`. The
authoritative sequence is each entry's position relative to the runs it cites. OA-004 sits last
only because later entries were inserted above it; it was written before any screen ran.

## OA-001 Infrastructure and vendoring (2026-09-23 22:15)
- Folder layout as handoff G0. The measurement path is vendored into `vendor/` and imported as
  `vendor.<module>`; the production `scripts/gantry` is NOT put on `sys.path`, so the vendored
  `gantry_dynamic` cannot be shadowed. `model_augmentation.fit_systems.blocks` and
  `model_augmentation.systems.gantry_ss` are imported from production, read only: the
  "production OBC Jacobian" is by definition that block's `transition_from_free`.
- Vendored edits, each marked `# CHANGED (vendor, OA-001)`: repo-root depth in `config.py`;
  `TRAIN/VAL/TEST_FILES` pinned to the 14/4/4 T/V/E records (the Telica TP/VP/EP entries were
  appended to production after the 0.2046 measurement and do not exist in these datasets);
  absent `delta_a`/`vdelta_a` read as zeros (baseline-truth records carry no absorber state);
  `obc_gantry` imports the vendored `obc`.
- `measure/measure_delta.py`: dataset via `OA_MODE`, output to `outputs/measure_<tag>.json`,
  `m_diff` also reported on its own scale, run-84033 comparison only on the original dataset.
- HEAD is `1ec57c1`, not the handoff's `a97a577` (an ancestor); all vendored sources are clean
  at HEAD (`git status` empty for them). Recorded in `vendor/VENDORED.md`.
- Every run: `tools/watchdog.ps1` (copy of `telica-real/tools/watchdog.ps1`, thresholds
  unchanged: kill at RAM < 2.5 GB or C: < 2.0 GB).

## OA-002 G0 pass criterion (pre-registered 22:15)
PASS if the vendored measurement gives `rho*` (J at theta*, Delta* stencil) = 0.199 +- 0.002 at
stride 97 (quick check) and 0.2046 +- 0.002 at stride 1 on `augmentation_ma50_b140-230_a6_z03`.
Source of both numbers: `baseline-and-added-dynamics.md` sect. 4.

## OA-003 Truth-plant architecture (22:50)
- The addition goes into a copy of the BASELINE truth `gantrySystem.m` (the `q1` loop of
  `kamtin-fp-model/03 Simulink gantry/gantry_2025a.slx`), as `matlab/gantrySystemOA.m`. The
  existing hidden absorber is REMOVED; the constructed addition replaces it.
  Rejected: a copy of `gantrySystemExtended.m` with the addition on top of the existing absorber.
  Then `Delta*` = absorber (rho* 0.2) + addition, and the null control (addition off) would still
  carry the absorber, so it could not measure the floor. `gantrySystemExtended` at `ma = 0` is
  singular (cubic README), so "absorber off" is not available there either. The handoff's
  "addition off reproduces gantrySystemExtended.m bit-identically" is therefore implemented as
  "addition off reproduces the baseline truth `gantrySystem.m` bit-identically" (Class A), and the
  Class B check compares the model copy with the original `gantry_2025a` `q1` loop.
- One general EOM, all terms default zero: stateless `M_a(Y) = Ma0 + Y Ma1`,
  `K_a(Y) = Ka0 + Y Ka1 + Y^2 Ka2`, `C_a = Ca0`, and `N_SLOTS = 4` lossless absorbers with
  `b_i(Y) = b0_i + Y b1_i`, mass-conserving (absorber masses are part of the baseline rigid mass,
  as `mh_rigid + ma = mh` in the existing truth). Derived EOM:
  `(M - SUM m_i b_i b_i^T + M_a) qdd = u - (C + C_a) qd - (K + K_a) q + SUM m_i w_i^2 d_i b_i`,
  `dd_i'' = -b_i^T qdd - w_i^2 d_i`. Frozen-`b` (no Coriolis), the convention `gantrySystem`
  already uses for `M(Y)`. Unused slots have `m_i = 0`, `b_i = 0`: their states are exactly zero.
- Generator: `gtd_config('augmentation', false, 0.50)` (non-MSD `q1` branch) with the production
  dataset's knobs re-applied: band `[140, 230]` Hz, amplitude x6 on `A_sym`, `A_Y`, `A_anti`
  (derived). Same records, seeds (`100*track_id + k`, track_id from TRACK only), controller
  (`gtd_build_plant` never sees `ma`), references. Records carry `delta_a = vdelta_a = 0`
  (loader compatibility; the absorber states are not logged by the `q1` loop).

## OA-005 Prune the two parallel plants from the model copy (22:30, after R-004 was killed)
- R-004: compiling the original `gantry_2025a` (Simscape Multibody plant `q` + two ODE plants)
  took the MATLAB tree to 4.76 GB and the watchdog killed it at 1.72 GB available (15.8 GB
  machine, about 6 GB free with the other sessions and editors open). The kill threshold stays.
- `gantry_oa.slx` therefore COMMENTS OUT the Simscape loop (`q`) and the Coriolis loop (`q2`):
  every block of those two loops, so no port is left dangling. They share no signal with the
  `q1` loop, and `gtd_run_simulation` reads only `q1` on the non-MSD branch.
- Consequence, stated not hidden: the solver is ode45 (variable step, as in the original), and
  its step control no longer sees the removed plants' states, so `q1` may differ from the
  original 3-plant model at the solver-tolerance level. It is common to the null control and the
  addition dataset (both from `gantry_oa`), so it cannot enter the G4 comparison.
- Class B B1 is therefore: `gantry_oa` (addition off, 14 states) vs `gantry_oa_ref` (same pruning,
  ORIGINAL chart `gantrySystem`, 6 states), must be bit-identical. Rejected: lowering the kill
  threshold (handoff sect. 13 fixes it) and `-nojvm` (Simulink needs the JVM).

## OA-006 Legitimacy rule: what the addition may contain (22:45)
- A stateless term may only occupy a STRUCTURAL ZERO of the baseline: an (entry, Y-power) slot of
  `M(Y)`, `C`, `K` that is zero for EVERY parameter value (`design/columns.py`, `BASELINE_SLOTS`).
  Otherwise part of the "addition" is a parameter change in disguise: the truth's physical
  parameter would be the shifted one, and "exact recovery" of the nominal value would be a
  relabelling, not a recovery. This is what "these parameters CANNOT represent what we add" means
  operationally. Rejected: the unrestricted `(M_a, K_a)` class of the previous route B member,
  whose `K22` (the `kb_sum` slot) and `M11, M12, M22, M33` entries are representable.
- Dynamic terms (absorbers) are legitimate as a whole: they carry states no parameter has.
- Physically named stateless terms (sympy-checked in `derive/`): `k_x` X spring at the beam
  centre (`K11`), `k_y` Y spring at the payload's rotation-neutral point (`K33`), `k_xp` X spring
  at the payload (`K = k b b^T`, `b = [1, -Y, 0]`: slots `K11`, `K12 Y`, `K22 Y^2`), `gamma` skew
  of the Y guide (`M13 = mh gamma`, `M22 += -2 mh d gamma Y`, first order in gamma).
- Lossless: no damping slot is used (handoff sect. 4: a dissipative addition is never orthogonal
  to the damping family), absorbers are undamped.

## OA-007 Route C search design (23:00)
- 4 absorbers (8 states = `nx_ann`) + the 4 physical mount terms. Linear unknowns: 4 masses +
  4 mount coefficients = 8 against 10 conditions, so the geometry (frequency, direction, lever per
  absorber, 12 numbers) must make the 10 x 8 condition matrix rank <= 7: 3 scalar conditions
  (codimension of rank-7 10 x 8 matrices), with the masses positive. Solved as: for a geometry,
  the best linear member is the generalised eigenvector of `(Q, Gram)` (min `rho*`), and the
  geometry minimises that `rho*` plus a penalty on negative masses, multi-start Powell at a
  strided reference set (strided `rho*` is fine for screens, handoff sect. 12).
- Frequency ranges searched in two families: in-band (140 to 230 Hz) and out-of-band
  (20 to 130 and 240 to 600 Hz). R-005 B3 measured that an in-band absorber changes the closed-loop
  motion at O(1) (4.8e-05 m against 1.7e-05 m q rms), so a first-order screen of an in-band design
  is expected to transfer badly to regenerated data; out-of-band designs are preferred when both
  screen below 1e-3. HEURISTIC split, from that one measurement.
- Time box 3 h wall from the first route C run (handoff G2).

## OA-008 G4 band (pre-registered 23:05, before the null control is measured)
All numbers at stride 1, `J` at `theta*`, `Delta*` stencil variant, `measure/measure_delta.py`.
- `b_null`: predicted bias `J^+ Delta*` per parameter on the null control (the pipeline's own
  floor: ZOH, integration, stencil and float32 together). `b_add`: the same on the addition
  dataset. Units: percent of the combo scale for nine parameters, `m_diff` ALSO on its own scale
  (`|m1 - m2|` = 0.5 kg).
- `B` = RMS over the ten of `b_null` (combo scale): the floor's own combo-err.
- **PASS** iff (i) `max_j |b_add,j - b_null,j| <= B` over the nine non-`m_diff` parameters, (ii)
  `|b_add,m_diff - b_null,m_diff| <= B` with both in percent of `|m1 - m2|` (21x stricter than
  the combo scale), (iii) `||Delta*_add|| >= 8.49` and `>= 5 ||Delta*_null||` (OA-004).
  In words: the addition moves no parameter's predicted estimate by more than the numerical
  floor alone biases the whole vector. HEURISTIC: the floor-RMS as band is the data-derived scale
  the handoff asks for; per-parameter floor values are not used as bands because a parameter the
  floor happens not to bias would get a zero-width band.
- Reported beside it, not deciding: the same with `J` at the detuned `theta_0` (R4, the operative
  condition in training), and `rho*` of the addition dataset.
- Iteration on FAIL (handoff sect. 5): re-solve the addition's linear coefficients on the
  regenerated addition dataset's own tuples, regenerate, re-measure; at most 3 attempts per
  route, then the next route.

## OA-009 Route B member selection: least actuator force per unit of learnable discrepancy (23:40)
- R-009 found `null(A)` of dimension 17 for the 27 legitimate slots; the "largest physical share"
  member had a 21 degree skew and 7.6e4 N/m^3 stiffness curvature, i.e. the share criterion does
  not control magnitudes. Replaced by: the member of `null(A)` that maximises
  `||Delta||^2 / SUM_k ||P^-1 phi[k]||^2`, the learnable discrepancy per unit of STAGE actuator
  force (N on X1, X2, Y). Generalised eigenproblem on the null-space coordinates.
- Why this criterion: the addition must stay inside the actuator budget (records already peak at
  1727 N against the 2000 N limit, R-006) and a small force perturbs the closed loop least, which
  is what makes the first-order screen transfer to regenerated data (handoff sect. 5).
- Rejected: largest physical share (magnitudes uncontrolled), minimum coefficient norm (mixes
  N/m, kg and rad units arbitrarily).

## OA-010 Numeric check tolerances for DERIVATION.md (pre-registered 23:00)
- N1 (Eq. 4, continuous vs discrete coefficient): every condition row within 5 % of its largest
  entry. HEURISTIC: Eq. (4) is a leading-order statement; the prior work measured the sensitivity
  itself at 1.55e-03 relative (T2) but in-band content has `omega Ts` up to 0.36, so a few percent
  is the expected O(Ts) gap. N1 shows the continuous formula is the right EXPLICIT form; the
  design itself always uses the discrete coefficients (handoff sect. 6).
- N2 (each of the ten conditions on the design data): `|SUM_c A_jc a_c| <= 1e-8 SUM_c |A_jc a_c|`
  (a solved linear system: round-off level).
- N4 (fast operator = production step): relative 1e-6 (prior verification 7.7e-09).

## OA-011 Route C reformulated: one physical absorber + an exactly solved compensator (23:15)
- R-010 (OA-007, 4 absorbers + 4 physical mount terms, 3 Powell starts, 800 evaluations each)
  reached screen rho* 6.15e-04 but the member is not physical: absorber 3 mass -0.005 kg, mount
  `k_x` -2270 N/m and `k_xp` -488 N/m (negative springs), absorbers 9.25 kg = 92 % of the payload,
  skew 3.5 degrees. And its screen bias already breaks the G4 band: cg1 -0.65 %, cg2 -0.76 % against
  `B` = 0.267 %, because the cg1/cg2 columns are nearly collinear and amplify any residual. So a
  NUMERICALLY exact member is required, which a nonlinear geometry search does not deliver.
- New route C: ONE lossless absorber with fixed physical parameters, the production absorber made
  lossless and centred: `m = 5.05 kg` (ma_frac 0.50), `f = 150 Hz` (so `k = m (2 pi 150)^2` as in
  `gtd_config`), along Y at the payload, `b = [0, -d, 1]` (`L0 = 0`, the handoff's prerequisite that
  removes the `m_diff` coupling). Plus a stateless COMPENSATOR from the 27 legitimate slots
  (OA-006), solved EXACTLY from the ten linear conditions with the least actuator force (OA-009
  metric). 10 equations, 27 unknowns: always solvable, no geometry search. States: 2 (<= 8).
- The absorber is the physically meaningful subset ("an added MSD"); the compensator is the
  derived part that makes the whole addition orthogonal. Its share of `||Delta||` is reported.
- Closed loop (R-005 B3: an in-band absorber changes the motion at O(1)): the compensator is
  solved on data that ALREADY contains the absorber, not on the null control. Sequence: generate
  the absorber-only training records, measure their `Delta*`, solve the compensator against the
  MEASURED `Delta*` with target `Phi^T (Delta* + Delta_comp) = Phi^T Phi b_null` (the addition's
  own bias zero, the floor's bias as in the null control), regenerate all 22 records with absorber +
  compensator, measure (G4). The compensator is stateless and small against the controller, so its
  own closed-loop effect is second order; iterate per OA-008 if not.
- The screen (G2) for this route is the same solve on the null control with the absorber's force
  from the fast operator (first order).
- Rejected: continuing the OA-007 geometry search (time box, and it cannot give round-off
  exactness), and a damped absorber (lossless requirement, OA-006).

## OA-012 Compensator library: stiffness-only preferred (pre-registered 23:30)
- Two libraries are solved on the absorber-only data: `KM` (all 27 legitimate slots) and `K` (the
  17 stiffness slots). Stiffness terms act at low frequency, where the 100 Hz loop is stiff
  (R-005 B2: 1000 N/m moved q by 1.1e-09 m), so a K-only compensator barely changes the regenerated
  trajectory and the one-shot solve should transfer; inertial slots act in-band, above the loop
  bandwidth, and change the motion by `M_a / M`.
- Rule: take `K` if its compensator stage-force rms (largest axis) is at most 2x that of `KM`,
  otherwise `KM`. HEURISTIC factor.

## OA-013 Route C attempt 1 member: K-only, least force (23:35, after R-015)
- R-015 solved the compensator on the absorber data (measured `Delta*`, floor bias kept) to
  round-off with every variant; all predict `b_add = b_null` exactly on that data. Costs:
  least force KM: comp. force rms (249, 185, 133) N, `||Delta_comp||` 175 (99.6 % of the addition);
  least force K: (416, 410, 362) N, 36 (91.5 %); least `||Delta_comp||` KM: (504, 488, 173) N, 20
  (78.6 %); least `||Delta_comp||` K: (485, 549, 304) N, 29 (87.9 %).
- OA-012's rule picks K (416 / 249 = 1.67 <= 2). The least-`||Delta_comp||` objective is rejected:
  it lowers the compensator's share only to 79 to 88 % while raising its force.
- Finding recorded as it stands: an in-band absorber's projection is inertia-like (mh +12.9 %,
  m_total +8.7 %, R-014), and structural-zero terms are by construction far from `range(Phi)`
  (single-column rho 0.04 to 0.3, R-009), so cancelling `||P Delta_abs||` = 1.86 takes a
  compensator several times the absorber. The MSD cannot be the dominant part of an orthogonal
  addition built this way; it can be an exactly orthogonal PART of one.

## OA-014 Second-order-aware design through the recorded input (00:05)
- R-017 identified the mechanism by which regenerated closed-loop data break a first-order
  design: below the loop bandwidth the controller cancels a low-frequency addition force, so the
  record keeps x and carries `u_new = u - phi`. The step is affine in `u`, so
  `Delta*_new = Delta*_null + G phi` exactly (what the design assumed), but `Phi` is evaluated at
  `(x, u - phi)`, second order in the addition, never seen by a null-control design.
- `design/design_u_aware.py` solves the G4 target `Phi'^+ (Delta*_null + G phi) = b_null` with
  `Phi' = Phi(x, u - phi(kappa))` (quadratic in kappa) by iteration: re-evaluate `Phi'`, least-force
  correction, repeat to round-off. Assumption, stated: x unchanged (stiff loop; R-005 B2 measured
  1e-9 m for a 1000 N/m spring). It is checked by the regenerated dataset, not assumed away.
- Used for route B if attempt 1 fails, and it is the principled form of the handoff sect. 5
  "re-solve on the regenerated data": the re-solve is done where the regenerated data will be.

## OA-015 Size held fixed in the u-aware iteration (00:20)
- R-020: iteration 0 of `design_u_aware.py` predicted R-019's measured failure from the null
  tuples alone (max shift 11.71 % predicted, 10.93 % measured; m_diff own -12.75 % vs -12.79 %):
  the OA-014 mechanism is the cause, quantitatively. The iteration converged (residual 8.9e-05) but
  the least-force steps removed most of the addition (`||Delta||` 12.7 -> 0.95 < 8.49).
- Fix: each iteration takes the affine solution set of the conditions linearised at the current
  `Phi'`, keeps the null-space part closest (force metric) to the current member, and rescales it
  to `||Delta_add|| = 12.735` (1.5 x the OA-004 minimum, as before). Rejected: accepting the
  collapsed member (fails OA-004), or a smaller target (the minimum is pre-registered).

## OA-016 Solve the exact quadratic system instead of iterating (00:30)
- R-020b: the projected fixed-point iteration at full size does not converge (residual 45 to 127
  over 8 iterations, force growing to 69 N rms): the second-order term is not a small perturbation
  at `||Delta|| = 12.7`.
- But its structure is exact. The production step is affine in `u` (N4, 7.7e-09 in the prior
  work), so `Phi(x, u - phi(z)) = Phi_0 - SUM_s z_s Phi_s` with `Phi_s = Phi(x, u) - Phi(x, u - phi_s)`
  EXACTLY, and the ten G4-target conditions are the quadratic system
  `F_j(z) = c0_j + SUM_t L_jt z_t - SUM_st z_s z_t (Phi_s^T Delta_t)_j = 0` (the floor-bias term
  `Phi'^T Phi' b_null` kept to first order in z; its z^2 part is z^2 |b_null|, negligible).
- `design/design_quadratic.py`: builds `Phi_s` for the 23 slots one at a time (stride 1, 23
  Jacobian builds), accumulates the coefficient tensors, then solves `F(z) = 0`,
  `||SUM z_t Delta_t|| = 12.735` with least force (SLSQP from the OA-009 member), and verifies the
  solution with a fresh shifted-input Jacobian (the u-aware iteration-0 check).
- Model assumption kept visible: the loop cancels the addition's force completely (`u_new = u -
  phi`), which R-020 validated for this member (predicted 11.71 / -12.75 % vs measured 10.93 /
  -12.79 %: 7 % error of the effect). The residual of that assumption is what attempt 3 re-solves
  on the regenerated data.

## OA-017 Route B attempt 3: the quadratic solve on the stiffness slots only (01:05)
- R-023 (attempt 2) failed worse than attempt 1 and its `Delta*` did not transfer (14.05 measured
  vs 12.735 designed). Its member leans on inertial slots at the kg level (M13 1.37 kg, M13 Y
  -5.6 kg/m, M33 Y +4.9 kg/m, M11 Y -5.1 kg/m): in-band forces the loop does NOT cancel, so the
  OA-014 model (`x` unchanged, `u -> u - phi`) is wrong exactly where the solver used it.
- Attempt 3: `design_quadratic.py` with the library restricted to the 17 stiffness slots
  (`OA_LIB=K`). Stiffness forces follow `q`, which is dominated by the low-frequency reference
  motion that the loop cancels; this is the regime in which the model predicted attempt 1 within
  7 %. Same size (12.735), same least-force objective. Also: `design_quadratic` gains
  `OA_DELTA_MODE` (correction around a regenerated dataset) for later use.
- Rejected: correcting attempt 2 on its own data (its residual is 16 %, so the correction would be
  as large as the addition and meet the same model error).

## OA-018 Route B closed as FAIL; route C attempt 2 with an in-band compensator (01:15)
- Route B, three documented attempts: R-019 (linear design, 10.9 % shift), R-023 (quadratic,
  KM library, 16.3 %), R-024 (quadratic, K library: no solution at size 12.735, SLSQP stalls at
  |F| 24, predicted 9.5 %). Mechanism: the loop cancels a low-frequency addition force, so the
  recorded input carries it and the mass-family regressor `dM_j M^-1 (u - C qd - K q)` contains
  `-dM_j M^-1 phi`. For PSD `dM_j` (m_total, J_eff, mh) the resulting term in `Phi^T Delta` is a
  sign-definite quadratic form in `phi`. A spring `k` makes the linear term scale as
  `(k/M) SUM qd^2` and the quadratic one as `(k/M)^2 SUM q^2`; they balance at `k ~ M w_eff^2`
  (w_eff = rms velocity / rms position, about 3 rad/s on X, so about 400 N/m), and the size
  minimum needs several thousand. Above that the quadratic term wins and the conditions have no
  solution. Low-frequency stateless additions of learnable size cannot be orthogonal on
  closed-loop data. That is the route B verdict.
- The same analysis says what CAN work: a force the loop does not cancel (above its 100 Hz
  bandwidth) leaves the recorded `u` unchanged, so the sensitivity does not see it, and only the
  small change of `x` enters at second order. The lossless absorber is such a force (150 Hz); route C
  attempt 1 (R-017) failed on its STIFFNESS compensator.
- Route C attempt 2: absorber + an INERTIAL compensator (forces follow `qdd`, which is dominated
  by the 140 to 230 Hz multisine). The legitimate inertial slots up to `Y^1` are only 6 (< 10
  conditions), so the truth EOM gains `oa_Ma2` (`M_a` up to `Y^2`: 11 inertial slots, M11 Y, Y^2;
  M12 Y^2; M13 1, Y, Y^2; M22 Y; M23 Y, Y^2; M33 Y, Y^2). Class A and B are re-run on the rebuilt
  model; the null control stays valid because addition-off is bit-identical to the baseline in both
  builds (A1/B1). Compensator solved on the absorber-only data (measured `Delta*`, R-015 procedure,
  `OA_LIB=M`), least force.
- Physical reading of the inertial slots: Y-dependent inertia (a cable carrier whose moving mass
  grows with the axis position is the textbook case) and a skew-type X-Y inertial coupling.

## OA-019 Route C attempt 3: tune the absorber itself, measured in the loop (01:30)
- R-026: the exact inertial-only compensator on the absorber data is not physical (M11 Y
  4357 kg/m, 3 kN rms): 11 nearly collinear inertial slots against 10 conditions. Route C attempt 2
  fails at the design stage.
- Correction to OA-018's wording: "sign-definite" holds for a PSD `dM_j` only if the weighting
  `M^-1 S_v^-2 M^-1` commutes with it; in general the quadratic term is merely dominant. The
  empirical statement is R-024: no stiffness-only solution was found.
- Attempt 3 uses the one lever that needs no compensator: the absorber's own tuning. Its reaction
  is an apparent mass `m_app = m w^2 / (w_a^2 - w^2)` along `b`, positive below resonance and
  negative above, so an absorber tuned INSIDE the 140 to 230 Hz band can average its inertia-like
  projection out (the sign change noted in the prior write-up, sect. 8.4). Its force is in-band,
  so the loop does not move it into `u` (no OA-014 term). Scan `f_a` = 170, 185, 200, 215 Hz
  (5.05 kg, Y at the payload, lossless), 14 training records each, measured in the loop; the
  measurement path gains `OA_TRAIN_ONLY` (val/test lists skipped; the measurement uses the 14
  training records only). Selection: smallest predicted bias (combo RMS and m_diff own), then a
  small inertial compensator on that data if needed (least force), then the full 22-record dataset.

## OA-020 Size/orthogonality trade-off (handoff sect. 5 asks for it) (02:20)
- Routes A, B, C have their verdicts (A: size; B and C: three attempts each). R-028 closed the
  last stateless idea (inertial-only member: +-17 kg of Y-dependent payload inertia).
- The handoff asks, when regenerated data miss the floor, to "report the size/orthogonality
  trade-off". `design/tradeoff.py` computes it with the u-aware prediction validated in R-020
  (11.71 % predicted vs 10.93 % measured at full size): the predicted G4 shift of a design as a
  function of its size `||Delta||`, for (a) the linear route B member (R-009b) scaled, and (b) the
  quadratic stiffness-only solve (R-024 machinery) at sizes where it has a solution.
- Pre-registered use: the largest size whose predicted max shift is <= B/3 = 0.089 % (a 3x margin
  for the model's measured 7 % error of the effect plus what it omits) is regenerated ONCE and
  measured with `g4_verdict.py`. If that size is below 8.49 it is a DEMONSTRATION that the
  construction transfers to regenerated data where the second-order term is small, not a G4 PASS
  (criterion iii fails by construction), and it is reported as such. If it is >= 8.49, the
  dataset is scored against all three criteria, flagged as a design found outside the three-
  attempt budget.

## OA-021 Continuation instead of SLSQP for the quadratic conditions (02:35)
- R-024 and R-030: SLSQP from the attempt-1 member stalls ("rank-deficient equality constraint
  subproblem") and does not even reach the requested size (R-030 asked 9.0, ended at 12.49): it
  starts far from the constraint set.
- `design/continuation.py`: `c0 = Phi_0^T (Delta*_null - Phi_0 b_null) = 0` exactly, so at small
  size the exact solution is a member of `null(L_t)` (OA-009 choice inside it); the branch is
  followed upward in size by least squares from the previous, scaled solution. Where `|F|` stops
  reaching round-off the branch ends: the largest size this library admits under the OA-014
  model. Verification with a fresh Jacobian at 8.49 and 9.0.
- Use (OA-020 rule): if a size >= 8.49 is on the branch with verified shift <= B/3, that addition is
  regenerated once and scored with all three G4 criteria, flagged as found outside the three-
  attempt budget; otherwise the branch end is the reported trade-off.

## OA-022 N1 failed on two rows; the explicit form becomes the exact RK4 polynomial (03:05)
- R-033: N1 (continuous leading-order Eq. 4 vs the production discrete coefficients, 5 % per
  row, OA-010) FAILS for cb_sum (43 %) and mh (5.6 %); the other eight rows are within 3.6 %.
  The yaw-damping sensitivity is small at leading order (yaw velocity is small) while the
  O(Ts^2) RK4 term (via the in-band yaw acceleration) is not. The FAIL stays on record.
- The explicit form of the ten conditions in DERIVATION.md becomes the EXACT frozen-Y RK4 form:
  `Phi_j = S^-1 d/dtheta_j [Phi_d(A) x + Gamma_d(A) B u]`, `Delta_phi = S^-1 Gamma_d(A) B_f phi`,
  `Phi_d = I + Ts A + (Ts A)^2/2 + (Ts A)^3/6 + (Ts A)^4/24`,
  `Gamma_d = Ts (I + Ts A/2 + (Ts A)^2/6 + (Ts A)^3/24)` (RK4 of a linear system with a held
  input is exactly its 4th-order Taylor polynomial), with `A(Y)`, `B(Y)` from Eq. (1) at the
  sample's `Y`. Check N1b: this closed form against the production Jacobian, per condition row,
  pre-registered tolerance 1e-3 of the row's largest entry (what is left is `Y` moving inside the
  step, O(Ts Yd)). Eq. (4) (continuous) is kept as a remark with its measured N1 accuracy.

## OA-023 N1b failed on two rows; the explicit form is the LPV RK4 map itself (03:15)
- R-033b: the exact frozen-Y RK4 polynomial matches the production coefficients to <= 7.7e-04 on
  eight rows; cy (5.2e-03) and cb_sum (1.5e-03) miss the pre-registered 1e-3. The residual is `Y`
  moving inside the step (largest on the Y-damping row, where Yd is the regressor). The FAIL stays
  on record; (4) and (4') are kept in DERIVATION.md as approximations WITH their measured errors,
  not as verified equations.
- The explicit definition of the sensitivity in the ten conditions is the LPV RK4 map (Eq. 4''):
  `x+ = x + Ts/6 (k1 + 2 k2 + 2 k3 + k4)`, `k_i = F(x_i, u)`, `F = [qd; M(Y_i)^-1 (u - C qd - K q)]`,
  `x_2 = x + Ts k1/2`, `x_3 = x + Ts k2/2`, `x_4 = x + Ts k3`, with `M` re-evaluated at every stage's
  `Y_i`. Check N1c: an INDEPENDENT numpy implementation of (4''), differentiated by central
  differences in each combination (h = 1e-4 relative), contracted with the production `Delta_c`,
  against the production coefficients, per condition row, pre-registered tolerance 1e-5 of the
  row's largest entry. It tests that the production Jacobian is the derivative of the map written
  in the document, which is what "explicit form" has to mean for a discrete-time model.

## OA-004 Size minimum and screen threshold (pre-registered 22:50)
- `||Delta*||` minimum: at least 0.5 x the current absorber's `||Delta*||` on the same reference
  set definition (stride 1, stencil variant, 14 T records): 0.5 x 16.98 = **8.49**, and at least
  **5 x** the null-control dataset's own `||Delta*_null||` (resolvable against the regenerated floor).
  HEURISTIC: half the discrepancy the project has been trying to learn keeps it the same order as
  that target; 5x the floor keeps it resolvable (the current absorber sits at 9.7x the stencil floor).
- G2 screen PASS (handoff): screen `rho* < 1e-3` and screen `||Delta*|| >= 8.49`. Screen = first
  order (fast operator) on trajectories without the addition in the loop.
