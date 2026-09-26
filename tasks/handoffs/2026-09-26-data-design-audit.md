# Handoff: audit the first-campaign dataset design (training, validation, test) and decide whether it is a good dataset design for the thesis results
**From**: session of 2026-09-26 | **Branch**: Augmentation | **Effort suggested**: high (a judgement-heavy audit on paper with hand arithmetic; no computation)

## 1. Task
Audit the dataset proposal in `Thesis-writeup/Documentation/DATA-DESIGN.md`, section 7 ("Sets (first campaign)"), as an independent examiner in system identification, precision motion control and data-driven modelling. Section 7 defines:
- training: 18 records;
- validation: 6 records;
- test: 6 interpolation and 6 extrapolation records;
- the deferred list;
- the FRF references;
- the control truths;
- the manifest.

Decide whether it is a good dataset design for the thesis results R1 to R7. Judge it on six things:
- **Coverage:** does training cover what the test interpolates?
- **Independence of the sets:** leakage between training, validation and test.
- **Validation adequacy:** checkpoint, added-state and black-box hyperparameter selection.
- **Result support:** can each result's pre-stated criterion be decided from the records it gets?
- **I and E clarity:** is each extrapolation record really one axis beyond training, against its interpolation partner?
- **Fixed decisions and budget:** does it respect the user's fixed decisions (section 4) and the budget?

Where it falls short, propose the smallest changes that fix it within the fixed decisions and the budget. Write the audit to `scripts/gantry/data-set-design/DATA-AUDIT.md`.

## 2. Out of scope
- Editing `DATA-DESIGN.md`, `NOISE-INJECTION.md`, `RESULTS-DESIGN.md`, `CRITIC.md` or `REVIEW.md`. The audit is input to the user's decision.
- Re-opening any user decision of section 4. You may show that one of them makes a result undecidable, and say which, but not propose reversing it.
- Growing the per-run load: 18 training and 6 validation records of 12 s, 12 test rollouts per trained model. A proposed change that adds records must say what it removes.
- Generating data, training, running MATLAB or Python, or any numerical check. The audit is on paper.
- The multisine band, which is session 1's; the noise model, which is `NOISE-INJECTION.md`'s; the deferred records' own designs (section 5 of the design).
- Real Telica data; reading `kamtin-data/`.

## 3. Where things stand
- Branch `Augmentation`, last commit `745d1e0`. The tree is dirty with unrelated work: do not commit or clean.
- `Thesis-writeup/Documentation/DATA-DESIGN.md` (about 540 lines) is the current proposal; section 7 is authoritative for the first campaign.
- History:
  - first critique: `scripts/gantry/data-set-design/CRITIC.md`;
  - independent review of an earlier, larger version: `scripts/gantry/data-set-design/REVIEW.md`;
  - how each was handled: sections 11 to 13 of the design.
- No run in flight. Nothing has been generated.

## 4. Established and verified
**User decisions (fixed; audit the implementation, not the choice):**
- **True system:** physics baseline + Coulomb rail friction + payload absorber (ma 0.50 mh, damping 0.03, pole 212 Hz, closed-loop peak 266 Hz). Never the absorber alone.
- **Record types:**
  - standstill multisine;
  - Y sweep + multisine;
  - point-to-point moves (Jasper's ETEL S-curve generator) + multisine;
  - Lissajous + multisine;
  - ASMPT / ILC-shape profiles without multisine.
- **ILC-shape profiles** are in training and validation, as in the current dataset (D-206). They run at 45 % and 75 % of the datasheet maximum.
- **Phases:** every record with a multisine has its own phase realisation. The noise-only twins reuse their source record's phases on purpose (floor).
- **One controller K1** for every training, validation and test record: the rule-of-thumb 100 Hz controller, designed once at Y = 0. The generator currently designs one per record; the design explains why that changes (5.11). Controller variation appears only in the R7 test records.
- **Acceleration anchor** is the datasheet maximum: X 30, Y 50 m/s² (`literature/gantry/telica-xyz-0750-0800-data.pdf` p. 2; the p. 4 typical cycle runs at it).
  - Training goes up to 75 % of it.
  - The measured ILC move (X 39.3, Y 50.7 m/s², D-210) is a test stress record beyond the maximum.
- **Y:**
  - The operating range is the full stroke, ±0.40 m: the datasheet p. 4 cycle picks at Y = -0.40.
  - Training covers ±0.30.
  - The simulator is stated as Garcia's parameter set run over the Telica stroke. The earlier ±0.36 m geometry cut was rejected.
- **Validation** lies inside the training range but is not a copy of training, as in the current dataset (V1 to V4, VP1, VP2).
- **Record length:** every record in every set is 12 s at 20 kHz, downsampled to 4 kHz. There are no shorter records for one set.
- **Per-run size** equals the current dataset: 18 training and 6 validation records.
- **Test set:**
  - 6 interpolation and 6 extrapolation records;
  - each extrapolation record has one interpolation partner differing only in the extrapolated axis;
  - everything else is deferred and generated only if time allows (supervisor: start with the basics).
- **Noise:** a force disturbance on the motor force inside the loop, calibrated to the measured Telica standstill error (9.8 / 10.4 / 6.3 nm rms), per `Thesis-writeup/Documentation/NOISE-INJECTION.md`. The level matches the measurement, the spectrum does not.
- The 10 % parameter detuning is a joint-estimation model start, not a data property. Settling is dropped. Real Telica data are out of the main results.
- **Supervisor comments on results that bind the data:**
  - R1 is LPV vs LTI without augmentation and without black box;
  - the black box appears in the final result only, with its size chosen by hyperparameter search;
  - R3 is a best linear approximation (an FRF is linear, friction won't show in it);
  - R5 appears only if a truly orthogonal addition exists;
  - the coverage map is optional.

**Facts read in the author session:**
- **Generator** (`Matlab-scripts/Augmentation/data/`):
  - `gtd_build_plant.m` designs the controller per record at Y_op;
  - `gtd_enforce_limits.m` checks force, position, velocity and yaw on the rigid baseline's linear loop, not on the truth;
  - `gtd_make_reference.m` draws point-to-point setpoints uniformly;
  - `thirdOrderSetpointETEL` is Biagiotti and Melchiorri's FIR chain;
  - record seeds are `100 x track_id + k`.
- **Current dataset** (`gtd_build_records.m`, `Matlab-scripts/Augmentation-telica-profiles/gtd_build_records_telica.m`, merged in D-208): T1 to T14 and TP1 to TP4 train; V1 to V4 and VP1, VP2 validate; E1 to E4 and EP1 test. No current test record extrapolates in Y. E3's acceleration lies below the TP records in training.
- **Session 1** (`EXCITATION-VALIDATION.md`, in `scripts/gantry/excitation-closed-loop/` and `Thesis-writeup/Documentation/`):
  - every record class excites all ten parameter combinations above an encoder-noise floor;
  - no parameter band is needed;
  - the proposed band is 140 to 390 Hz at the production multisine level (240 N symmetric, 87 N m anti, 180 N Y);
  - at one sixth of that level the rails stick.

## 5. Assumed but not verified
- **Every hand estimate in the design:**
  - rail effective mass and crossovers under one K1;
  - force margins;
  - the true system's M(Y) growth beyond ±0.30;
  - jerk values;
  - cross-coupling torques.

  Each now says which model it uses; recompute any you rely on.
- **That K1 designed at Y = 0 is stable on the true system out to ±0.39.** Not checked: it is a generation gate in the design.
- **The band and verdicts.** Session 1 computed its floor with encoder noise, so under force noise the floor moves below 200 Hz. The band and the joint-estimation verdict still need that recheck.
- **Open inputs:** the ASMPT profile set (Jasper and Dragan), which controller changes are realistic (Quinten), where Y = 0 sits relative to the beam centre, the 0.75 training fraction, the orthogonal-addition truth and the friction-only truth. All are marked (?) in the design.
- **That 6 interpolation and 6 extrapolation records**, with 5 starts per arm and 3 data realisations, give enough statistical power for the Welch and equivalence rules of `RESULTS-DESIGN.md` S6. Not argued anywhere yet.

## 6. Tried and failed (do not re-propose without a new argument)
- **Validation as a record-for-record mirror of training** -> rejected by the user -> the user wants validation inside the range but not exact copies, as in the current data.
- **Shorter records for validation only** -> rejected -> the user does not want mixed record lengths.
- **20 validation records, and a test set of about 120 records plus about 160 twins** -> rejected as too much -> per-run and evaluation cost; the budget is now 18 / 6 / 12.
- **ILC profiles test-only, patched by a no-multisine training move** -> wrong -> the current dataset already has multisine-free ILC records in training and validation for that reason (D-206).
- **A ±0.36 m Y cut from Garcia's rail geometry** -> rejected -> the datasheet cycle operates to -0.40, and Garcia's own table would already exclude ±0.30.
- **Acceleration anchored to the measured ILC move** -> replaced by the datasheet maximum (user).
- **Force and crossover estimates on the rigid baseline presented as truth values** -> corrected -> the design now labels them. The generator's pre-check is baseline-only, so the design adds a binding post-simulation check on the truth.

## 7. Achieved
- `DATA-DESIGN.md` has, on paper:
  - machine values with sources;
  - principles;
  - the operating-point and extrapolation kinds enumerated;
  - axis rows with prediction, falsifier and criterion;
  - a two-label rule (range and density);
  - the first-campaign sets;
  - the deferred list;
  - the manifest (about 208 simulations);
  - feasibility;
  - a 12-item implementation list.
- Not validated by any computation.

## 8. The open question
Is the first-campaign design a good dataset design for these results? Candidate answers: fit for generation; fit with a named short list of changes within the budget; not fit, with the reason.

What decides it:
- whether each result R1 to R7 can reach a verdict on its pre-stated criterion from the records it gets;
- whether any interpolation or extrapolation claim fails its own labelling rule;
- whether the sets leak into each other or are unrepresentative for selection.

## 9. Next action
Read the files of section 11 in order, then write `DATA-AUDIT.md`. Form your own findings from the design and its sources before reading `CRITIC.md` and `REVIEW.md`. Those reviewed an earlier, larger version, so check which of their points still apply to the first campaign. Examples of the checks expected, not the scope:
- **I3's coverage.** Does training cover the interpolation claim of I3? It is an ASMPT sine at 60 % of max, where training has ILC-shape at 45 % and 75 %.
- **E1's density label.** Will the density label call E1 at -0.39 sparse, given moves in training stop no further than ±0.30?
- **R1 with one pair per side.** Can R1 decide LPV vs LTI from one I / E standstill pair per side plus one moving pair?
- **Validation for black-box selection.** Is a 6-record validation set, with one standstill point, adequate to select a black-box size by hyperparameter search?
- **Statistical power.** Is a single controller record (E5) against a single partner enough for R7's criterion?

Format: headers plus one-line bullets, open points marked (?), no em-dashes. One finding, shown on an unrelated system so nothing can be copied:
```
2. **[major] The only extrapolation record for pump speed shares its operating point with a validation record.**
   - Where: test E3 and validation VA-2 (both at 1450 rpm)
   - Why: validation selects the checkpoint, so E3 is no longer unseen by the selection loop
   - Fix: move E3 to 1520 rpm (still inside the stroke), no change in record count
```

## 10. Acceptance criterion
`DATA-AUDIT.md` is done when it has all of the following:
- **Verdict:** one line from section 8, with its three strongest reasons.
- **A support table for R1 to R7:** the records each result gets, its pre-stated criterion, and whether that criterion can be decided (yes, marginal, no, with the reason). Include the sufficiency check.
- **Findings,** most severe first, marked [major], [moderate] or [minor], each with where, why (with arithmetic where a number is involved) and a concrete fix that respects section 4 and the budget. Report everything you find; do not filter.
- **An independence check:** every Y point, lattice, controller and setpoint family in validation and test, set against training and against each other.
- **A label check:** for each of the 12 test records, the range label and your prediction of the density label.
- **Hand estimates:** every one the audit relies on, re-derived or marked "not re-derived".
- **Earlier points:** the points of `CRITIC.md` and `REVIEW.md` that still apply to the first campaign, each marked resolved or open.
- **A "Checked and found sound" list.**

## 11. Read these first
1. `Thesis-writeup/Documentation/DATA-DESIGN.md`: section 7 (the object), then sections 2, 5 and 6 (principles, axis rows, labelling rule).
2. `scripts/gantry/thesis-results-design/RESULTS-DESIGN.md` section 4 and S6: the results and their decision rules. Its truth and S1 are superseded; the slides' numbering applies (R7 is controller transfer).
3. `Thesis-writeup/Documentation/NOISE-INJECTION.md`: the noise model the data carry.
4. `Thesis-writeup/Documentation/EXCITATION-VALIDATION.md` sections 11 to 13: what each record class excites.
5. `Matlab-scripts/Augmentation/data/gtd_build_records.m` and `Matlab-scripts/Augmentation-telica-profiles/gtd_build_records_telica.m`: the current dataset the proposal replaces.

After your own pass: `scripts/gantry/data-set-design/CRITIC.md` and `REVIEW.md`. For field conventions on splits, coverage and held-out operating points: `scripts/gantry/thesis-results-design/LITERATURE.md` sections 2 (c) and 7.

## 12. Do not
- Edit any file other than `DATA-AUDIT.md`.
- Propose anything section 6 lists as rejected without a new argument that the user has not heard.
- Propose changes that grow the per-run load or the test set beyond 12 rollout records without naming what they replace.
- Read `kamtin-data/`.
- Run code, MATLAB or scripts.

## 13. Operational
- No computation; no scripts.
- Output: `scripts/gantry/data-set-design/DATA-AUDIT.md`, the one new file this handoff authorises.
- Quoting code with line numbers follows the code-quote verification rule.

## 14. Delegation
- No subagents. The audit's value is one reader holding the whole design; read the held PDFs inline.
- A `deep-research` run only if a load-bearing claim about field practice (for example validation-set size for model selection) is disputed and not covered by `LITERATURE.md`. At most one, with the FRAME step.
