# Audit of the first-campaign data design

## Verdict

- **Fit with a named short list of changes within the budget:** the proposal has broad training coverage and clean record-level splits, but it is not fit for generation yet because I4 is categorically outside training despite being called interpolation, R1 lacks the positive standstill pair needed for a two-sided position conclusion, and R7 has only one changed-controller condition while its decision rule is still unsettled.

## 1. Scope and standard used

- This audit treats section 7 of `DATA-DESIGN.md` as the proposed campaign and sections 2, 5 and 6 as its rules.
- The fixed user decisions in section 4 of the handoff are not reopened.
- The latest supervisor decisions override the older result wording where they conflict: R1 is baseline LPV versus baseline LTI without augmentation or black box, R3 is a BLA check, R5 exists only if a truly orthogonal addition exists, and slide numbering makes controller transfer R7.
- `RESULTS-DESIGN.md` still contains older R1, R5, R6 and R8 criteria; any criterion changed in `DATA-DESIGN.md` must be frozen in one authoritative result document before generation.
- No data, code or numerical simulation was run for this audit.

## 2. Support for R1 to R7

| Result | Records supplied | Pre-stated decision rule | Can the records decide it? |
|-|-|-|-|
| R1, position dependence | TR-S1 to S5 as fitted reference; I2/E1 standstill; I3/E2 moving | Baseline LPV versus baseline LTI, no augmentation or black box; compare end rise relative to interior with the S6 superiority rule | **Marginal.** E1 gives only the negative standstill end. E2 visits both moving ends, but one moving record is not a second standstill pair per side. The older R1 criterion still compares augmented arms, so the authoritative criterion is not frozen. |
| R2, added-state order | Order selected on VA-S1, VA-Y1, VA-P1, VA-L1, VA-T1 and VA-T2; curve reported on I1 to I6 for T_AF and T_F | `n_a = 2` better than 0; 4 and 8 equivalent to 2; same knee on T_AF and T_F; ablation worse than trained | **Marginal.** The split avoids test selection and the six tests cover several regimes. T_F is still `(?)`, validation has no yaw exposure, and the six validation definitions do not span the point-to-point level and jerk-time range used in training. |
| R3, plant-level BLA | TF-1 to TF-3 at Y = 0.15, 0.225 and 0.35, two phase realisations each; TF-4 at -0.35 optional | Resonance and anti-resonance within frequency resolution; magnitude error inside the nonparametric FRF standard deviation at all three Y values | **Marginal.** The trained, between-point and positive extrapolation locations are present. Period count and hence resolution are unset, and two phase realisations alone give a weak variance estimate. Negative-Y asymmetry is not part of the mandatory set. |
| R4, negation and uniqueness | TR records for parameter estimates; I1 to I6 for accuracy; five starts per arm | F-test on parameter-spread ratios; Aug-OBC accuracy equivalent to Aug-U; spread-matched penalty worse | **Yes, conditional on the planned five independent starts.** The records are adequate for the parameter and accuracy parts. The validation omission of yaw weakens penalty/checkpoint selection but does not make the final criterion impossible. |
| R5, recovery condition | T_0 and proposed T_OA on the training set, validation set and noise-free training twins | Latest rule: show R5 only for a truly orthogonal addition; older rule: predicted bias per combination across several discrepancies, including a nonzero case | **No as written today.** T_OA and its orthogonality are unresolved, and the older multi-discrepancy criterion is not supported by only T_0 and T_OA. If R5 is formally narrowed to exact recovery on T_0 and a verified T_OA, the allocated records are sufficient. |
| R6, generalisation versus black box | I1 to I6 paired with E1 to E4 and E6; I4 for cross-coupling | Aug-OBC competitive at rung 1, better than BB-NL on standard and beyond-standard rungs, better than LTI throughout, and better than B-LPV and BB-NL on the cross-coupling group | **No for the full criterion.** I4 is an untrained axis-combination class and is therefore range-E under section 6, not interpolation. Only X-only motion is present although the criterion says a cross-coupling group of X-only and Y-only moves. One record per extrapolation axis supports case studies, not an operating-condition population claim. |
| R7, controller transfer | I3 under K1 and E5 under K2-g+; one controller-specific noise twin | `DATA-DESIGN.md` proposes absolute error and predicted change; `RESULTS-DESIGN.md` still specifies an NRMS ratio and more than one K2 | **No for the broad claim; marginal for one named case.** One +20 percent controller on one route can decide only whether transfer works for that case. Independent noise in I3 and E5 contaminates a direct change score near the floor. The metric is still `(?)`. |
| Sufficiency check | Training against validation and I1 | Similar training/held-out error; I1 is the same-setting test that selects nothing | **Marginal.** I1 is a valid unbiased same-setting check. Aggregate training versus validation mixes different record proportions and validation omits yaw and most point-to-point extremes, so it is not a record-matched sufficiency estimate. |

### Statistical sufficiency check

- Five starts per arm do provide a model-start distribution, but they do not create five independent test operating conditions.
- With five starts in each of two arms and equal start standard deviation `s`, the S6 Welch threshold is approximately `t_0.975,8 s sqrt(2/5) = 1.46 s`; effects smaller than about 1.46 start standard deviations will not pass even before data-realisation uncertainty is added.
- Under the same approximation, the 90 percent equivalence half-width is `t_0.95,8 s sqrt(2/5) = 1.18 s`; the chosen `delta` must exceed this or equivalence is structurally unlikely.
- For one controller-transfer ratio over five starts, a 90 percent interval has half-width approximately `t_0.95,4 s_r/sqrt(5) = 0.95 s_r`; one E5 record cannot establish transfer across controllers, trajectories or noise realisations.
- The three data realisations are described as separate from start spread, but no result rule says how their uncertainty enters Welch, equivalence or the F test. This must be fixed before runs.

## 3. Findings, most severe first

1. **[major] I4 is not an interpolation record under the design's own categorical rule.**
   - Where: sections 4, 5.7, 6 and 7.3
   - Why: training contains simultaneous XY moves, sweeps, Lissajous paths and yaw, but no X-only point-to-point or ILC-shape motion. Section 4 explicitly makes axis combination an extrapolation kind and section 6 makes every untrained categorical value E. Calling I4 `I (range and class)` contradicts those rules and leaves only five valid interpolation records.
   - Fix: put guaranteed X-only and Y-only segments into existing TR-P4 and held-out counterparts into VA-P1, retaining yaw segments in TR-P4 and the 18/6 counts; then let I4 contain both axis-isolated routes with new setpoints. Otherwise relabel I4 as categorical E and replace an E slot with a genuine I record.

2. **[major] R1 lacks the symmetric standstill evidence implied by its claim.**
   - Where: I2/E1, I3/E2 and section 7.5's statement that R1 shows two test points per side
   - Why: I2/E1 is only at negative Y. E2 reaches both moving ends inside one trajectory, so positive and negative motion share one route and one noise realisation. There is no positive standstill interpolation/extrapolation pair, and the design itself leaves the need for one open in Q4.
   - Fix: replace the lower-priority velocity pair I6/E4 by standstill records at `+0.225/+0.39 m`. R1 then has negative and positive standstill pairs plus the moving I3/E2 pair, with 6 I, 6 E and 12 rollouts unchanged.

3. **[major] The pre-stated result criteria are not yet one coherent contract.**
   - Where: `RESULTS-DESIGN.md` R1, R5, R6 and R8 versus `DATA-DESIGN.md` sections 5.11, 7.7 and 7.8
   - Why: R1 changes from augmented LPV/LTI to unaugmented baselines, R5 changes from several discrepancy truths to conditional T_0/T_OA, and R7 changes from an NRMS ratio over K2 controllers to absolute error and predicted change for one K2. A data table cannot be declared sufficient against mutually different pass/fail rules.
   - Fix: freeze one criterion per R1 to R7 before generation. No record count changes are needed, but the broad R7 and old R5 claims must either be narrowed or receive additional conditions by replacing other test/control records.

4. **[major] R7 has one controller and one route, so it cannot support a general controller-transfer result.**
   - Where: E5 against I3; all other controller records are deferred
   - Why: a +20 percent gain change explores one direction only and one multisine-free route only. Five model starts quantify optimisation sensitivity, not variation over controllers. I3 and E5 also have independent noise by the fixed rule, so `e_K2 - e_K1` contains a noise difference as well as a controller difference.
   - Fix: for the present budget, title and conclude R7 as transfer to `K2-g+ on the I3 route`, use absolute error against E5's controller-specific floor, and do not claim invariance across controllers. A broad claim requires replacing another E record by K2-g- or a second controller and using the same deterministic reference definition.

5. **[major] The proposed matched pairs are not fully specified as matched records.**
   - Where: I3/E2, I5/E3, I6/E4 and I3/E6
   - Why: I3 has no frozen Y lattice, direction sequence or setpoint list; E2 says only that I3 is extended to +/-0.39. I5/E3 fix stroke and T3 while acceleration changes, which necessarily changes jerk and normally timing or velocity. I6/E4 give strokes and velocity caps without the exact capped-profile construction. E6 also changes acceleration, jerk and velocity relative to I3 and is correctly a stress record, not a one-axis match.
   - Fix: publish deterministic reference definitions and a realised-value difference table for every I/E pair. Require every non-target range label to remain I, call unavoidable dependent changes covariates, and stop describing E6 as differing only in one extrapolated axis.

6. **[major] Four class-level noise twins do not justify every per-record floor invoked by the criteria.**
   - Where: sections 2, 5.1, 5.5, 5.7, 5.10 and manifest 7.9
   - Why: the noise response changes with Y, motion regime and controller; the design itself expects up to about 30 percent X-floor variation over training Y. One standstill, one multisine move, one no-multisine move and one E5 twin cannot be called each record's own floor. Cross-coupling I4 specifically requires a local floor, and E1 at -0.39 is outside the calibrated Y range.
   - Fix: name the four exact source records for the twins and restrict floor-relative criteria to them. Recommended allocation is E1 or its retained end record, I1, I4 and E5. For all other records report absolute errors without claiming a record-specific floor, unless an extra truth simulation is bought by removing an equal-cost control or FRF simulation.

7. **[major] Validation does not cover the variables on which model order and black-box size can depend.**
   - Where: VA-S1 to VA-T2
   - Why: validation has no yaw excitation, one point-to-point record only at 50 percent and T3 30 ms, and no axis-isolated route. Training includes yaw, point-to-point levels 25, 50 and 75 percent, and T3 values 12, 25, 30 and 36 ms. Hyperparameter search can therefore choose a model that is best on a materially narrower mixture than the training and reported test tasks.
   - Fix: retain six records but add a small yaw overlay to VA-L1 and divide VA-P1 into deterministic held-out segments spanning the trained point-to-point levels, jerk times and X-only/Y-only cases. Weight validation by record family rather than by raw sample count.

8. **[moderate] The density threshold is still ambiguous for unseen record types.**
   - Where: section 6, `one threshold for every test type: the largest p over all validation records`
   - Why: this can mean one global threshold or a different threshold per type, but validation has no long-stroke, changed-controller or axis-isolated type. The choice can change I3 to I6 from dense to sparse after the fact.
   - Fix: freeze one global validation threshold, or freeze an analogue map before generation. A global maximum is reproducible but conservative; report that limitation.

9. **[moderate] I3's range coverage is sound, but its record-level interpolation claim is stronger than the evidence.**
   - Where: I3 versus TR-T1 to T4 and VA-T1/T2
   - Why: 60 percent acceleration is exactly halfway between the trained 45 and 75 percent levels, and the 40/80 mm sine family and no-multisine class are trained. Its lattice, dwell and direction sequence are unstated, so independence from the validation ILC lattice cannot be checked.
   - Fix: freeze I3 on a lattice not used by VA-T1/T2, or state explicitly that the operating points may be validation-seen and that only the route and kinematics are held out.

10. **[moderate] The R6 cross-coupling criterion asks for a group that the first campaign does not contain.**
   - Where: `RESULTS-DESIGN.md` R6 and DATA-DESIGN sections 5.7 and 7.3
   - Why: the criterion names X-only and Y-only moves; I4 contains X-only motion only. The design's hand estimates give different mechanisms, about 24 N m from X motion and 30 N m from Y motion, so one is not a replicate of the other.
   - Fix: alternate X-only and Y-only moves within I4 and score the two segments separately. This preserves the record count and gives both directions the same noise realisation and controller.

11. **[moderate] R3's uncertainty criterion is not operational until the periodic design is complete.**
   - Where: section 7.6
   - Why: the number of periods is `(?)`, while frequency resolution and FRF variance are part of the pass/fail criterion. Two phase realisations give only one degree of freedom if phase-to-phase variation alone supplies the standard deviation.
   - Fix: freeze period count and the variance estimator before generation. Keep TF-1 to TF-3 if the claim is limited to one trained, one between-point and one positive extrapolation Y; make TF-4 mandatory only if negative-Y asymmetry is claimed.

12. **[moderate] R5 is conditional but its optional 42-simulation block is counted as if committed.**
   - Where: sections 7.7 and 7.9
   - Why: T_OA is `(?)`, R5 is omitted if it is not truly orthogonal, yet its 42 simulations are in the roughly 208 total. This obscures the committed versus conditional budget.
   - Fix: publish `166 committed + 42 conditional T_OA` rather than one approximate total, and do not generate T_OA until its orthogonality gate passes.

13. **[moderate] Unique test phases protect independence but weaken the I2/E1 causal contrast.**
   - Where: fixed own-phase rule and the standstill Y pair
   - Why: changing Y and multisine phase together means the raw rise is not a pure one-factor comparison. All arms see the same record, so within-record arm differences remain fair, but between-record end rises include phase variability that one phase per test cannot estimate.
   - Fix: use a difference-in-differences between LPV and LTI arms within each record, not a raw E-minus-I error alone, and describe phase as a nuisance factor. Do not violate the fixed unique-phase decision.

14. **[moderate] The first-campaign manifest is arithmetically clear but the replicate policy is not.**
   - Where: sections 2 and 7.9
   - Why: 54 + 18 + 6 + 6 + 6 + 4 = 94 for T_AF and 42 + 42 + 30 = 114 controls, total 208. Control validation appears to use one realisation although main validation uses three, and the statistical rules do not say whether three data realisations are pooled or reported as three separate experiments.
   - Fix: add columns for truth, record realisation, model start and noise/phase twin to the manifest; state the unit of replication for each result.

15. **[minor] Section 7.5 overstates the R1 point count.**
   - Where: `R1 therefore shows two test points per side, not a curve`
   - Why: the current first campaign has one negative standstill endpoint and one moving record that visits two ends. That is not two independently generated points per side.
   - Fix: correct the statement or adopt the positive standstill replacement in finding 2.

16. **[minor] Several open generation gates remain prerequisites rather than audit evidence.**
   - Where: B_tr, force-noise recheck, K1/K2 stability, post-truth limits, final ASMPT profiles, Y origin and seed implementation
   - Why: the paper design names the gates correctly, but a record cannot be called feasible or final until they pass.
   - Fix: mark the campaign `design-approved, generation-blocked` after the data-design fixes, then lift the block only when the named gates are recorded.

## 4. Smallest repair set within 18/6/12

1. Replace I6/E4, the velocity pair, by `I2+` at Y = +0.225 and `E1+` at Y = +0.39; retain the current negative standstill pair and moving Y pair.
2. Put guaranteed X-only, Y-only and yaw segments into TR-P4 without adding a record; put held-out X-only and Y-only segments into VA-P1; let I4 alternate X-only and Y-only segments.
3. Add yaw to VA-L1 and make VA-P1 cover held-out 25, 50 and 75 percent point-to-point segments and the trained T3 range inside one 12 s record.
4. Freeze exact lattices, directions, dwells, setpoint lists and realised-value acceptance checks for I3/E2, I5/E3 and every retained pair.
5. Freeze one density threshold rule and name the four records that own the four available floor twins.
6. Narrow R7 to K2-g+ on I3 and choose one metric before generation; do not claim transfer over a controller family from E5 alone.
7. Freeze the narrowed R1 and R5 criteria in the result design; omit the T_OA block if the orthogonality gate fails.

- This repair keeps 18 training records, 6 validation records, 6 interpolation records, 6 extrapolation records, 12 s per record and 12 test rollouts per trained model.
- It trades the velocity extrapolation case for the evidence needed by the core R1 claim; velocity remains explicitly deferred.

## 5. Independence audit

### Training reference

- Standstill Y points: `{-0.30, -0.15, 0, +0.15, +0.30}`.
- Continuous moving Y coverage: sweeps and random point-to-point moves cover `[-0.30,+0.30]`; Lissajous records cover continuous subranges.
- ILC-shape lattices: residues `0`, `0.02` and `0.06 mod 0.08 m`; trained levels 45 and 75 percent; sine shape; no multisine.
- Point-to-point family: simultaneous XY random setpoints, 25, 50 and 75 percent, T3 of 36, 30 and 12 ms; one 50 percent yaw record at T3 25 ms; all with multisine.
- Controller: K1 only.

### Validation against training and test

| Record | Against training | Against validation peers | Against test | Independence verdict |
|-|-|-|-|-|
| VA-S1, Y +0.10 | New standstill Y; training motion passes +0.10 | Distinct from other validation references | No test standstill at +0.10 | Independent record, but only one standstill validation point |
| VA-Y1, centre +0.05, range -0.20 to +0.30 | Same sweep class and 0.5 Hz as TR-Y2; new centre and amplitude | Overlaps VA-P1 and VA-L1 in Y | Test routes cross the same continuous Y range | Valid distribution-level holdout, not a held-out Y region |
| VA-P1, full-range random moves | Same 50 percent and T3 30 ms definition as TR-P2; new setpoint draw and phases | Shares broad Y/X domain with all moving validation | I1 uses the same settings with another draw | No record leakage if tuple seeds are unique; selection sees I1's distribution by design |
| VA-L1, centre -0.05 | Same X frequency/amplitude as TR-L2, changed Y centre/amplitude and no yaw | Overlaps VA-Y1 continuously | No exact test Lissajous record | Independent waveform, but validation never tests yaw |
| VA-T1 and VA-T2, residues +/-0.04 mod 0.08 | Held-out from training residues; same sine/no-multisine family; levels 75 and 45 percent are trained | The two records occupy the same Y lattice and differ mainly in level and direction details | I4 at Y -0.20 lies exactly on this lattice; I3/I5 lattice is unspecified | No exact-record leakage, but checkpoint selection sees the held-out lattice later used by I4 and possibly I3/I5 |

### Test interpolation and extrapolation

| Record | Y and lattice relation | Family relation | Controller relation | Split verdict |
|-|-|-|-|-|
| I1 | Full trained Y/X range | Same definition as TR-P2 and VA-P1, new setpoints, phases and noise | K1 | Clean same-distribution test; appropriate for sufficiency |
| I2 | Y -0.225 lies between standstill points; training motions pass it | Standstill family trained; not a copied record | K1 | Clean record-level interpolation; operating point is not globally unseen |
| I3 | Inside +/-0.20; exact lattice unset | Sine, 40/80 mm and no-multisine family trained at 45/75 percent; 60 percent is bracketed | K1 | Range-I, but independence from VA-T lattices cannot be checked until fixed |
| I4 | Y -0.20 is on the VA-T lattice and inside continuous training motion | X-only route absent from training and validation | K1 | Categorical E under the current rule; not valid in the I set without the repair |
| I5 | Inside +/-0.20; exact route unset | S-curve shape trained and no-multisine class trained separately, but their combination is new | K1 | Marginal range-I; density label must carry the unseen combination |
| I6 | Y -0.28 to +0.28, X -0.25 to +0.25 | Long-stroke sine/no-multisine combination not trained; scalar ranges trained | K1 | Range-I but likely density-sparse; proposed for replacement by positive standstill I |
| E1 | Y -0.39 is 0.09 m beyond the training edge | Same standstill family as I2, different mandatory phase | K1 | Clean range-E in Y; phase is a nuisance factor |
| E2 | Visits +/-0.39 | Intended to reuse I3 family, but the expanded route is not specified | K1 | Range-E in Y; pair cannot be certified exact yet |
| E3 | Interior Y | Intended I5 S-curve with acceleration 30/50; jerk rises but stays inside training | K1 | Range-E in acceleration only if all realised non-target axes pass the I checks |
| E4 | Same long-stroke family as I6 | Velocity 1.95 versus 1.5 m/s; exact capped profile unset | K1 | Range-E in velocity if realised acceleration and jerk remain inside; proposed for replacement |
| E5 | Same deterministic I3 reference definition | Same route, new controller and independent noise | K2-g+ only | Proper categorical controller E; both records are test, so reuse of the reference definition is not selection leakage |
| E6 | Same nominal strokes and sine family as I3 | Exact measured move; acceleration, jerk and velocity all change, acceleration and jerk exceed training | K1 | Legitimate declared stress record, not a one-factor pair |

- No validation or test record is an exact generated copy of a training record if the new tuple seed rule is implemented.
- Sharing continuous Y locations across training, validation and test is intentional for interpolation and is not leakage by itself.
- Sharing the VA-T lattice with I4 means that I4 does not test a held-out operating point; it can still test a held-out route, but that route is categorical E until trained.
- E5 reusing I3's deterministic reference definition is desirable matching, not leakage, because neither record selects a checkpoint.

## 6. Range and predicted density labels

| Record | Range label from section 6 | Predicted density label | Basis |
|-|-|-|-|
| I1 | I on all axes | Dense | Same record definition as TR-P2 with a new draw; validation contains the same distribution |
| I2 | I on Y, class, band and amplitude | Dense or borderline `(?)` | Y is between standstill points and crossed by moving training records, but zero-velocity samples at that exact Y may be less dense than continuous-motion samples |
| I3 | I on Y, X, velocity, acceleration, jerk, distance and excitation class | Dense `(?)` | 60 percent is halfway between trained 45 and 75 percent sine/no-multisine records; exact lattice can change the result |
| I4 | **E in axis combination**, I on scalar ranges | Sparse | X-only motion is absent from training and creates a new joint state/force combination |
| I5 | I on separately enumerated axes and classes | Sparse or borderline `(?)` | S-curve is trained with multisine and no-multisine is trained with sine, but the joint combination is new |
| I6 | I on scalar ranges and trained sine/no-multisine class | Sparse `(?)` | Long strokes visit trained marginal ranges but new joint position/velocity sequences |
| E1 | E in Y | Sparse | Y is 0.09 m beyond the training edge, `0.09/0.40 = 0.225` in the normalised Y coordinate |
| E2 | E in Y | Sparse | All samples beyond `|Y| = 0.30` are outside the training support |
| E3 | E in acceleration | Dense `(?)` | Expected disagreement pre-stated by the design: acceleration is new but force remains inside the high A_prod training-force range |
| E4 | E in velocity | Sparse | `1.95/1.50 = 1.30` beyond the trained velocity maximum, which is itself a network input |
| E5 | E in controller | Dense or borderline `(?)` | Controller is categorically new, but physical state and force inputs may remain inside the K1 cloud |
| E6 | E in acceleration and jerk; stress beyond rated X acceleration | Dense or borderline `(?)` | X acceleration is `39.3/22.5 = 1.75` of the training maximum and X jerk is `3090/1875 = 1.65`; force magnitude is nevertheless predicted inside the A_prod training envelope |

- The density entries are predictions, not labels assigned in advance; the frozen kNN rule must produce the final labels after generation and before test errors are inspected.
- If the repair in section 4 is adopted, the new positive standstill I/E records are predicted dense/borderline and sparse respectively, matching I2/E1.

## 7. Hand estimates used by this audit

- **Training fractions:** `22.5/30 = 37.5/50 = 0.75`; E3 is `30/22.5 = 50/37.5 = 1.333` times the trained acceleration maximum.
- **I3 bracket:** `(0.60 - 0.45)/(0.75 - 0.45) = 0.50`, so its acceleration level is centered between the two trained ILC-shape levels.
- **Y extrapolation:** `0.39/0.30 = 1.30`; E1 is 90 mm beyond the training edge and 10 mm inside the stroke.
- **Velocity extrapolation:** `1.95/1.50 = 1.30`; arithmetic re-derived, exact realised profile not re-derived.
- **Measured-profile jerk ratios:** X `3090/1875 = 1.65`; Y `3430/3125 = 1.10`; the design is correct to call E6 acceleration-and-jerk E.
- **True low-frequency coupling at Y = -0.20:** `M12 = -0.68625 - 10.1(-0.20) = 1.33375 kg m`; at X acceleration 18 m/s2 the torque is `1.33375 x 18 = 24.0 N m`.
- **Y-only yaw torque:** `mh d aY = 10.1 x 0.10 x 30 = 30.3 N m`; the two cross-coupling directions are therefore not duplicates.
- **LPV residual from absorber offset:** `ma L0 = 5.05 x 0.10 = 0.505 kg m`; yaw-inertia residual at Y = 0.39 is `1.01 x 0.39 + 0.0505 = 0.444 kg m2`; re-derived from the stated truth.
- **Absorber pole:** `150 sqrt(1 + 5.05/5.05) = 150 sqrt(2) = 212.13 Hz`.
- **Training rail-force screen:** `32.0 x 22.5 x 1.37 = 986 N`; adding the stated 500 to 750 N multisine peak gives about 1486 to 1736 N. This is only a mixed screening estimate, not a truth-limit result.
- **E3 rail-force screen:** `32.0 x 30 x 1.37 = 1315 N`; E6 gives `32.0 x 39.3 x 1.37 = 1723 N`. Both require the binding post-simulation truth check.
- **Welch and equivalence power:** re-derived in section 2 using five starts and equal variance; actual unequal-variance degrees of freedom were not re-derived.
- **Rigid-baseline crossover 74 to 108 Hz, truth Y crossover near 83 Hz, and K2-g+ near 120 Hz:** not re-derived; retained only as screening statements pending the mandatory truth-loop gate.
- **Closed-loop peak 266 Hz and 233 to 299 Hz half-power band:** not re-derived; taken from the cited excitation validation.
- **Differential rail error of order micrometres, standstill noise response outside Y = 0, multisine crest factors, and every density label:** not re-derived and must not be used as a safety or pass/fail value before generation.
- **Manifest arithmetic:** T_AF `54 + 18 + 6 + 6 + 6 + 4 = 94`; controls `42 + 42 + 30 = 114`; total `94 + 114 = 208`.

## 8. Earlier CRITIC.md points

| Point | Status in the first campaign | Audit disposition |
|-|-|-|
| C1, two labels disagree | Resolved | Range and density are both reported and disagreements are predeclared |
| C2, twin realisations set tau | Resolved | All realisations of the same reference are left out |
| C3, scoring mode absent | Resolved | Closed-loop scoring from reference and injection is explicit |
| C4, no test floor or phase spread | **Open** | Only four class-level twins remain and second test phases are deferred |
| C5, jerk represented by T3 | Resolved | Realised peak jerk is the axis; E6 has both E labels |
| C6, 0.1 mm record is multi-axis | Resolved outside campaign | Those records are deferred and described correctly |
| C7, absorber omitted from controller analysis | Partly resolved | Truth-loop gate is specified but not yet passed |
| C8, +/-20 percent arithmetic | Resolved | +20 percent is now the core K2 case |
| C9, payload shifts absorber | Resolved outside campaign | Payload test is deferred and band split is stated |
| C10, LPV baseline not exact | Resolved | Affine residual is explicit and arithmetically sound |
| C11, validation subset and biased sufficiency | **Open in a new form** | I1 fixes the selection bias, but validation omits yaw and point-to-point extremes |
| C12, R7 NRMS denominator | **Open** | Replacement metric is proposed but still `(?)` |
| C13, no-multisine class absent | Resolved | ILC-shape records without multisine are in training and validation |
| C14, unmatched E anchors | **Open for first campaign details** | Partners are named but exact reference constructions are incomplete |
| C15, acceleration force distance weak | Open but nonfatal | E3 is expected density-dense; this limits the strength of an input-extrapolation interpretation |
| C16, generator limits omitted | Partly resolved | Checks are specified but not implemented or passed |
| C17, R2 selected on test | Resolved | Validation chooses order; test reports the curve |
| C18, records unmapped | Resolved for first campaign | Every retained record has a named use |
| C19, friction regime ignored loop | Resolved | Loop and stick-band dependence are stated and session-1 evidence is used |
| C20, mismatched numbers | Resolved | Current endpoint and growth numbers are consistent |
| C21, truth notation differs from generator | Resolved on paper | Damping override and A_prod are explicit; implementation remains gated |
| C22, T_AF10 described as mass only | Resolved | T_AF10 is dropped |
| C23, no label tolerance | Resolved | Realised quantities and 1 percent tolerance are specified |
| C24, wrong short-move generator assumption | Resolved outside campaign | FIR-chain behaviour is stated |
| C25, yaw absent from density | Resolved | Differential rail displacement and rate are included |
| C26, criteria lack failure | Resolved on paper | Each retained axis has a ranking, equivalence or stress interpretation |
| C27, cross-coupling near floor | **Open** | Only X-only is present and no quantitative local floor is assigned |
| C28, endpoint fallback | Resolved by fixed decision | Full Telica stroke is the declared synthetic benchmark domain |

## 9. Earlier REVIEW.md points

| Point | Status in the first campaign | Audit disposition |
|-|-|-|
| R1, acceleration maximum misdefined | Resolved | Datasheet 30/50 anchors the ladder; measured move is a stress record |
| R2, excitation result not incorporated | **Open** | A_prod is incorporated, but B_tr and the force-noise recheck remain `(?)` |
| R3, density undefined for unseen types | **Open** | Current threshold sentence remains ambiguous |
| R4, tolerance undefined | Resolved | One percent realised-value tolerance is stated |
| R5, phases reused across truths | Resolved | Every truth record receives its own phase |
| R6, K1 stability not shown | **Open gate** | Gate is mandatory but not executed |
| R7, Y outside Garcia geometry | Resolved by fixed decision | It is explicitly a synthetic Telica-stroke benchmark |
| R8, unmatched anchors/random routes | **Open for retained pairs** | Exact routes and realised-value checks are still missing |
| R9, force checked on nominal plant | Resolved in design | Binding post-simulation check is on T_AF, pending implementation |
| R10, campaign count understated | Resolved | Manifest totals about 208, though conditional scope should be split |
| R11, no-multisine patch weakens design | Resolved | Existing ILC-shape records provide the class |
| R12, profiles/controllers not frozen | **Open** | ASMPT list and K2 set await external inputs |
| R13, payload intervention vague | Open outside campaign | Payload record is deferred and pending ASMPT confirmation |
| R14, seed and setpoint rules affect coverage | Resolved on paper | Tuple seed and log strata are specified, not implemented |
| R15, jerk arithmetic inconsistent | Resolved | X and Y jerk are separate and E6 values are explicit |
| R16, cross-coupling used baseline | Resolved | Current 24 and 30 N m estimates use T_AF |
| R17, X invariance predicts BB failure | Resolved outside campaign | Equivalence per arm is now the rule |
| R18, combined-cycle bound unsupported | Resolved outside campaign | It is a deferred stress score only |
| R19, controller IDs inconsistent | Resolved | E5 and I3 are used consistently |
| R20, validation wording ambiguous | Resolved | Definitions are shared; phases, noise and setpoints are independent |
| R21, ILC range given physical status | Resolved | It is labelled historical campaign coverage only |

## 10. Checked and found sound

- The benchmark truth is consistently baseline physics plus Coulomb rail friction plus the payload absorber, never absorber alone.
- Training contains all fixed record types, including multisine-free ILC-shape profiles at 45 and 75 percent.
- Every ordinary training, validation and test record uses one K1 designed at Y = 0; controller variation occurs only in E5.
- Training Y range is +/-0.30 m and every Y extrapolation lies inside the fixed +/-0.40 m stroke.
- I3's 60 percent acceleration is bracketed by trained ILC-shape levels rather than being a hidden range extrapolation.
- I1 is a proper same-definition, independently realised test and does not select checkpoints.
- R2 order is selected on validation and only reported on test.
- Validation records are independently generated and are not exact copies of training records.
- Both profile shapes and the no-multisine excitation class occur in training.
- E1, E2, E3, E4 and E5 each have a plausible single range or categorical axis beyond training; E6 is openly declared multi-axis.
- The range rule uses realised quantities and a numerical tolerance; the density rule excludes same-reference realisations and avoids the convex hull.
- Moving E records are to be scored on out-of-range segments as well as whole records.
- The fixed rated-acceleration ladder, measured over-rated stress record and full-stroke modelling assumption are clearly distinguished.
- The post-simulation truth check is correctly made binding over the rigid-baseline pre-check.
- The generator work list identifies fixed K1, per-record kinematics, axis-isolated routes, deterministic setpoints, collision-free seeds, acceleration exceptions and controller gates.
- The 18 training, 6 validation and 12 rollout counts respect the fixed per-run budget.
- FRF references are outside the learned-model rollout count and target one trained, one interpolated and one extrapolated Y.
- T_F is a conceptually sound R2 control because Coulomb friction is static in velocity and should not require added dynamic states.
- T_0 is a sound zero-discrepancy control for R5.
- Real Telica records are not mixed into training or reported as main-result evidence.

## 11. Final generation decision

- **Do not generate the first campaign from the current table.**
- Approve it for generation after the seven repairs in section 4 are reflected in one frozen manifest and one authoritative R1 to R7 criterion list.
- The remaining physics and implementation gates in finding 16 are then generation gates, not reasons to redesign the 18/6/12 split.
