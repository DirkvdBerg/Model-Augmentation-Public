# Literature Search Conclusion: the augmentation gap is confirmed (incl. by a 2025 survey)

**Date**: 2026-07-11. **Purpose**: record the CONCLUSION of the exhaustive ML-for-control literature search
for a method satisfying our augmentation requirements. Bottom line: **no published method satisfies all five
requirements; the gap is confirmed by an authoritative 2025 survey; property-preserving LPV/time-varying
identification is explicitly OPEN.** This is a thesis-positive (citable) negative result. Detail lives in
`docs/ml-for-control-search-sweep.md` (Directions 1-8), `docs/augmentation-literature-verdict.md` (requirement
table), `docs/dissipativity-limits.md`, `docs/rollout-stability-literature.md`. Quotes transcribed from
on-disk PDF text layers; re-verify character-exact before thesis use.

## The five requirements (the target no published method meets)
1. **Knowledge-free** — guarantee holds without knowing the true residual/dynamics (unknown real system).
2. **Friction-permitting / full expressivity** — represents ANY dissipative state-dependent residual
   (Coulomb/cogging); no restriction on the learnable dynamics class.
3. **Marginal-preserving** — keeps the zero-stiffness free-integrator (X/Y) pole at the origin; must NOT damp.
4. **Non-drifting** — bounded free-run position on X/Y (open-loop metric; closed-loop hides — D-107).
5. **Scheduling-integrity (the Y conflict)** — Y is SIMULTANEOUSLY a K=0 free-integrator (drifts) AND the LPV
   self-scheduling variable (`M(Y)` reads the drifting `x[2]`; CONFIRMED code-read). Drift DETUNES `M(Y)` (a
   feedback X lacks). A method must not corrupt Y-scheduling nor damp the Y pole while de-drifting it.

## What was searched (exhaustive, primary-read where load-bearing)
- **Dissipativity / passivity / NI family** (PRIMARY-READ: DiLaR-PINN 2604.18277, RENs 2104.05942, cyclo
  2003.10143, EID 1709.06986, Mabrok 1305.1079, nonlinear-NI 2011.14610, Casimir 2112.03339, Krasovskii
  1907.07420): each fails >=1 requirement (`dissipativity-limits.md`). Marginal-storage theory EXISTS (cyclo/
  EID) but does not bound POSITION.
- **Structural / stability-by-design** (RENs, Schur-LPV 2510.24757): contraction damps the marginal pole
  (fails R3).
- **Rollout stability / exposure bias** (PRIMARY-READ: pushforward 2202.03376, GNS 2002.09405, unrolled-
  training 2402.12971, transient-amplification 2605.08856): the right CATEGORY (solve-not-hide, expressivity-
  preserving) but EMPIRICAL only, and timescale/identifiability caveats (`rollout-stability-literature.md`).
- **Estimation / bias / IV** (PRIMARY-READ: Piga-Bemporad closed-loop-LPV bias-correction; Kuang-Lin IV
  2511.09024; Kuntz-Rawlings 2406.03760): linear / linear-in-parameters -> for the REAL-DATA baseline fit,
  not the nonlinear ANN.
- **LPV + ML** (PRIMARY-READ: Verhoek 2204.04060 = our framework, has CONSISTENCY not no-drift;
  Sertbas-Kumbasar 2510.24757 state-consistency regularizer usable, Schur not).
- **Hybrid-model identifiability** (PRIMARY-READ: Loman-Baker 2510.14140, Hotvedt 2010.13416): parametric/
  functional identifiability framing; orthogonal projection independently reinvented (npj 2024, UNVERIFIED).
- **Corrupted-scheduling / EIV-LPV** (R5-driven): a NAMED problem (TU/e flagship Automatica 2015, PAYWALLED)
  BUT it is NOISE-at-identification, ours is DRIFT-at-inference-self-scheduling -> not directly solved.
- **Symmetry / conservation, free-floating base**: conservative or control, not our forced-dissipative
  forward model. No match.

## The authoritative confirmation  [PRIMARY-READ]
**S. Sivaranjani, Y. Shi, N. Atanasov, T. Duong, J. Feng, T. Martin, Y. Xu, V. Gupta, F. Allgower,
"Control-Oriented System Identification: Classical, Learning, and Physics-Informed Approaches", 2025
(arXiv:2512.06315).** Comprehensive survey by leading control groups.
- (Abstract) ML system-ID's "utility in control applications is limited by their ability to provide provable
  guarantees on control-relevant properties." Surveys the SAME property families we triaged ("dissipativity,
  monotonicity, energy conservation, and symmetry-preserving structures").
- (§4.1) "some control-relevant properties such as stability and passivity can be directly embedded through
  parameterization ... However, capturing more complex physics-informed or control-relevant properties
  through identifiable parameterizations REMAINS AN OPEN CHALLENGE."
- (§7.3) "control-informed system identification for [switched and time-varying] systems is an important
  direction for future work." Cites our supervisors' group (Verhoek LPV) as the LPV state of the art -> the
  LPV physics-preserving ID line is ACTIVE and OPEN.
- **No method in the survey matches the five-requirement combination.**

## Conclusion (write this down)
1. **No published method satisfies all five requirements.** Confirmed by exhaustive multi-community search
   AND a 2025 authoritative survey. Structural-guarantee methods sacrifice expressivity (R2) or the marginal
   mode (R3); expressive methods give no structural guarantee (the impossibility: for-all-weights no-drift
   XOR full expressivity); none is native to a free-integrator + drifting-self-scheduling (R5) forward model.
2. **The gap is genuine and CITABLE** to the Sivaranjani et al. 2025 survey (property-preserving time-varying/
   LPV identification = open) -> the contribution sits in an explicitly-open area.
3. **The contribution, precisely:** a LEARNED, forward-augmentation, LPV (self-scheduled) model that is
   marginal-preserving (R3), friction-permitting (R2), non-drifting (R4, EMPIRICAL — the honest ceiling), and
   scheduling-integrity-preserving on the drifting Y (R5) — built on reusable pieces (Verhoek LPV-SUBNET
   framework; Gyorok orthogonal projection; cyclo/EID marginal-storage language; rollout-stability
   conditioning) with the ASSEMBLY + the R4/R5 handling as the novel part.
4. **Search saturation reached.** Further BROAD search is not warranted. Remaining literature work is
   TARGETED: (a) verify + quote the paywalled corrupted-scheduling flagship (Automatica 2015) for R5;
   (b) primary-read Verhoek LPV consistency to localize the contribution. Then the value is in BUILDING
   (D-107 empirical layers) and FRAMING against the survey.

## Provenance
- PRIMARY-READ this thread: ~20 papers (listed above + in the sweep/verdict/limits docs). Quotes from PDF
  text layers; re-verify character-exact before thesis use.
- UNVERIFIED / PAYWALLED (do NOT quote until verified): npj-2024 HNODE decorrelation; Automatica-2015
  corrupted-scheduling LPV flagship.
- Negative result status: firm (multi-angle + authoritative-survey convergence), but a negative is always
  "none FOUND"; keep it as "confirmed open per the 2025 survey", not "proven nonexistent".
