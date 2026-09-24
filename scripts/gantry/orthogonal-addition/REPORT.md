# Report: an addition orthogonal to the gantry baseline, tested on regenerated closed-loop data

Thresholds are pre-registered in `DECISIONS.md` (OA-xxx); every run is in `RUNS.md` (R-xxx) with
its hypothesis written before launch. The derivation, with a machine check per equation, is
`DERIVATION.md`.

## Summary
```
Additions the ten parameters cannot represent exist and are derived in closed form: the null
space of an explicit 10 x n matrix (DERIVATION sect. 4, 6). On data WITHOUT the addition a
stateless member is orthogonal to round-off (rho* 2.2e-15, all ten conditions at <= 6.3e-14).
On closed-loop data generated WITH it, the loop moves the addition's low-frequency force into
the recorded input, where the baseline sensitivity sees it a second time: the condition becomes
quadratic (DERIVATION sect. 7), measured 10.93 % vs predicted 11.71 % on the worst parameter.
Result: G4 FAIL after three attempts on each of routes B and C, route A fails on size. Orthogonality
on regenerated data holds only up to ||Delta*|| ~ 1 (demonstrated: 0.066 % max shift at 1.09,
band 0.267 %), a sixteenth of the absorber the project has been learning.
```

## G0 Infrastructure: PASS
Vendored measurement reproduces `augmentation_ma50_b140-230_a6_z03` (OA-002): `rho*` 0.198551 at
stride 97 (target 0.199 +- 0.002), **0.204623 at stride 1** (target 0.2046 +- 0.002); combo-err
4.956 % (doc 4.96), cond 766.40, stencil floor 8.7494e-04, floor bias 0.742 % (R-001, R-002).
Folder, vendor (`vendor/VENDORED.md`), watchdog on every run.

## G1 Formulas: PASS
`DERIVATION.md`: baseline EOM (1) [D1], sensitivity per family (2), (3) [D2, D3], the exact
one-step map (4'') [N1c: independent LPV RK4 vs production, all ten rows <= 1.5e-08], Eq. (16)
[D10] and Condition 4 (17) [N2], the admissible class as `null(A)` with `A_jc = Phi_j^T G phi_c`
(5) [D9, N2], legitimacy of every slot [D8], the physical terms (6) to (10) and the absorber's
apparent mass (14) [D5, D6, D7, D11], counting and Györök level per route (sect. 5), and the TEN
CONDITIONS written out in the addition's coefficients for the route B member (sect. 6), each
checked separately on the design data [N2: relative residual 7.3e-16 to 6.3e-14]. The closed-loop
form (11) to (13) [N4 4.5e-07, N5 2.8e-08, N6 3.8e-13]. Symbolic D1 to D11 all OK (R-011d);
numeric N1c (<= 1.5e-08), N2, N4, N5 (R-033c) and N6 (R-033d) OK.
Recorded against my own pre-registration: the continuous leading-order form (4) misses its 5 %
tolerance on cb_sum (43 %) and mh (5.6 %), the frozen-Y RK4 polynomial misses 1e-3 on cy and cb_sum
(R-033, R-033b, OA-022/023); both stay in the document as approximations with measured error,
and the conditions are stated with the exact map (4'').

## G2 Construction: PASS (route B)
```
Route C (4 absorbers + physical mount, geometry search, OA-007): screen rho* 6.15e-04 < 1e-3 but
  not physical (a negative absorber mass, -2270 N/m mount spring, absorbers 92 % of the payload),
  and its screen bias already exceeds the G4 band on cg1/cg2 (R-010).
Route C (production absorber made lossless and centred + exact compensator, OA-011): exact on its
  own data (residual 1.4e-12), but the compensator carries 91 to 99.6 % of ||Delta|| and 250 to
  3000 N rms (R-015, R-026).
Route A (parity): Eq. (16) exact on sign-symmetric data, but only standstill records mirror and
  there every even term is second order in the 5.4 um vibration: 5.0e-04 of the size minimum (R-008).
Route B (27 structural-zero slots): rank 10, admissible class of dimension 13 (23 columns);
  OA-009 member rho* 2.2e-15, ||Delta|| 12.74 >= 8.49, 13 N rms stage force (R-009b).  PASS
```

## G3 Truth plant: PASS
`matlab/gantrySystemOA.m` (copy of `gantrySystem.m` + addition: stateless `M_a(Y)` to `Y^2`,
`K_a(Y)` to `Y^2`, `C_a`, four lossless absorber slots; OA-003, OA-018). Class A
(`check_oa_noop`, R-004, R-025): addition off bit-identical to `gantrySystem.m` (0.000e+00 over
200 states), absorber EOM equals its Lagrangian form (1.4e-14), stateless terms enter as
`-(M_a qdd + C_a qd + K_a q)` (4.9e-15). Class B (`check_oa_reaches_plant`, R-005, R-025): addition
off bit-identical through Simulink to the pruned original-chart model (0.000e+00 m on T3 and T7),
an X spring scales linearly (1.9999 / 2.0001), an absorber reaches the plant. The Simscape and
Coriolis plants are commented out of the model copy (OA-005: their compile does not fit in RAM;
they share no signal with the `q1` loop). Datasets (22 records each, same records, seeds,
references, band [140 230] Hz, x6 amplitude and controller as the production dataset; limiter
never fired): null control `augmentation_oa_null_b140-230_a6` (R-006, force peaks 63 to 1727 N)
and addition `augmentation_oa_demo_b140-230_a6` (R-032), the two kept (handoff sect. 13). The
learnable-size addition `augmentation_oa_b_b140-230_a6` (R-018, peaks within 1 % of the null
control) and the other attempts were generated, measured and then removed (R-036); each is
regenerated from its addition file (README). Per-record force peaks are in `outputs/R-006.log`,
`R-013`, `R-016`, `R-018`, `R-022`, `R-027`, `R-032` logs. Two exceedances of the pre-check's
limits, which cannot see the addition, are recorded: R-016 up to 2494 N, R-022 E3 Y 1482 N.

## G4 Proof on regenerated data: FAIL (three attempts per route, mechanism identified)
Floor (null control, R-007): `||Delta*_null||` 0.280, **B = 0.267 %** combo-err, m_diff -0.053 %
own scale. Band (OA-008): every parameter's shift `|b_add - b_null| <= 0.267 %`, m_diff on its own
scale too, and `||Delta*_add|| >= 8.49`.
```
Route B attempt 1 (linear design, R-019): ||Delta*|| 12.736 (designed 12.735: the discrepancy
  transfers exactly) but cb_sum +10.93 %, mh +3.34 %, m_total +3.21 %, m_diff -12.8 % own scale:
  FAIL. Mechanism (OA-014): the loop cancels the addition's low-frequency force, so the RECORDED
  input carries it and the sensitivity, formed at qdd = M^-1 (u - C qd - K q), moves. Predicted from
  the null tuples alone: 11.71 % / -12.75 % (R-020).
Route B attempt 2 (exact quadratic system, all slots, R-021/R-023): |F| 2.4e-15 and predicted
  0.005 %, measured 16.3 %, m_diff +96 %: the solver used kg-level inertial slots whose in-band
  force the loop does NOT cancel, where the model is wrong. FAIL.
Route B attempt 3 (quadratic, stiffness slots, where the model holds, R-024/R-030/R-031): no
  solution at ||Delta|| >= 8.49; the exact branch ends between ||Delta|| 1 and 2. FAIL.
Route C attempt 1 (absorber + stiffness compensator, R-017): mh +126 %, rho* 0.52. FAIL.
Route C attempt 2 (absorber + inertial compensator, R-026): exact but 3 kN rms, not built. FAIL.
Route C attempt 3 (absorber tuning 150 to 215 Hz, R-027): the inertia-like projection never changes
  sign (mh +12.9 to +25.4 %). FAIL.
Absorber alone, the non-orthogonal contrast (R-014): combo-err 5.30 %, mh +12.9 %, m_diff -9.6 %.
```
**Size/orthogonality trade-off** (handoff sect. 5; OA-020, R-029): for the linear design the
predicted shift is exactly quadratic in size, 0.069 % at `||Delta*||` 1, 1.11 % at 4, 5.07 % at
8.49, 11.71 % at 12.7. **Demonstration** (R-032, pre-registered as not a PASS): the same member at
`||Delta*||` 1.09, regenerated: max shift 0.066 % (predicted 0.076), m_diff -0.070 % own scale
(predicted -0.088), both inside B; criterion (iii) fails (1.09 < 8.49, and < 5 x floor 1.40).
So the construction transfers to regenerated closed-loop data exactly where the second-order term
is small, and the learnable-size requirement is what breaks it.

## G5 Capturability and server prep: PASS
**States.** Route B addition: 0 states (static); route C: 2 per absorber. Both within `nx_ann = 8`.

**Acyclicity (Hoekstra et al. 2025, Condition 1 and Theorem 1).** In the truth, `qdd` solves
`(M - m b b^T + M_a) qdd = u - C qd - (K + K_a) q + m w^2 d b` at each instant, so the truth's
one-step map is a STATIC function of `(x_tilde, x_bar, u)`, and so is `Delta = f_truth - f_base`.
A parallel augmentation `f_aug(x_tilde, x_bar, u)`, `g_aug(x_tilde, x_bar, u)` realises it with the
graph `inputs -> {f_base, phi_aug} -> sum -> next state`, topological order inputs, f_base,
phi_aug, sum: acyclic, Condition 1 holds. The earlier objection ("`M_a qdd` is an algebraic loop")
applies only if the augmentation is fed the BASELINE block's acceleration while writing into it;
the addition is defined with the truth's `qdd`, itself solved from `(x, u)`.

**Prediction for the server runs** (`runners/run_oa_arm.sh`, `runners/oa_entry.py`, prepared,
not submitted; the dataset folder must be copied to the cluster):
- `OA_MODE=augmentation_oa_demo_b140-230_a6`: the `obc` arm converges to the null floor,
  combo-err about 0.29 % (the measured `J^+ Delta*`, R-032), m_diff within 0.12 % of its own
  scale; `noproj` has no such guarantee.
- `OA_MODE=augmentation_oa_b_b140-230_a6` (learnable size): the `obc` arm converges to the
  BIASED point `theta* + J^+ Delta*` measured in R-019 (cb_sum +11.8 %, mh +3.4 %, m_total +3.3 %,
  kb_sum +0.9 %, m_diff -12.8 % own scale), not to `theta*`: a direct training test of the
  closed-loop mechanism. `noproj`: no uniqueness, so no prediction beyond "not the same point".
  (This dataset is removed locally per the handoff's two-dataset rule and is regenerated with
  `generate_oa_data('NAME','augmentation_oa_b_b140-230_a6','ADDITION','design/addition_route_b_R-009b_null_s1.mat')`, 7 min.)

**Smoke test: PASS** (R-034, under the watchdog, two optimizer updates in total). `runners/oa_entry.py`
(the production entry, vendored package, `OA_MODE`, `OA_SMOKE`) on
`augmentation_oa_demo_b140-230_a6`: `OBC_ARM=obc` built the model, attached OBC to the 671944
reference tuples (rank 10, cond 928.95 at the detuned start), wrapped the closed-loop controller
bank, did ONE update (`sqrt loss 7.627e-05`) and returned; `OBC_ARM=noproj` the same, ONE update.
Smoke settings, stated: stride 100 and validation records truncated to 4000 samples (plumbing
only); deepSI's checkpoint directory redirected into `outputs/localappdata`. An earlier attempt was
stopped inside the pre-training validation with zero updates.

## What to do with this
The thesis claim this supports: OBC's parameter recovery is exact exactly when the added dynamics
are orthogonal ON THE DATA OBC SEES, and closed-loop data add a second-order term through the
recorded input that is not visible in any analysis done on data without the addition. The one
experiment that settles it in training is the pair in the prediction above (`obc` on the
demonstration dataset to the floor, `obc` on the learnable-size dataset to the measured biased
point); both runners are prepared.
