# Review Brief: independent, adversarial verification of the passivity-augmentation literature claims

**For a fresh reviewer (Fable) session. 2026-07-10.** Your job is INDEPENDENT VERIFICATION and RED-TEAM,
NOT review-for-agreement. Assume the compiling author (a prior session) may have over-claimed; your value
is in catching wrong quotes, wrong logic, and a missed counterexample.

## The sources are ALL on disk now (you can verify, do not take quotes on trust)
- New online papers downloaded to `literature/passivity-augmentation/` (filenames start with the arXiv ID).
- Framework papers already on disk: `literature/augmentation/hoekstra2025_lfr-augmentation-ejc.pdf`,
  `literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf`,
  `literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization...pdf`,
  `literature/augmentation/Data-driven augmentation ... stability guarantees.pdf`.
- NOT localizable: Lan Jia (2023) TU Delft MSc gantry friction numbers -- flag as unverifiable here.

## Read these two docs (the claims under test)
1. `docs/passivity-augmentation-literature.md` -- the catalog. Every quote is provenance-tagged. Items
   tagged `[extract - verify]` / `[online-*]` were obtained by automated extraction and are the PRIORITY to
   check now that the PDFs are on disk.
2. `docs/drift-diagnosis-status.md` sections 5i, 5j, 5k0, 5k, 5L (+ its §5L addenda) -- the design synthesis
   and the central gap claim.

## To reduce anchoring
Form your own reading of the PDFs FIRST for the two flagship claims below, THEN compare to the catalog.

## TASK 1 -- VERIFY (open the local PDF for each; confirm / correct the quote and the interpretation)
Priority, load-bearing:
- **DiLaR-PINN, `2604.18277`**: (a) Is the residual really parameterized as `(S - K) grad V` with `S`
  skew and `K` PSD, giving `grad V^T r <= 0` for ALL parameters? Quote the exact equation/proposition
  numbers. (b) **The load-bearing claim: does its stability result (reported as "Proposition 3") REQUIRE an
  ISS baseline** (`grad V^T f_phys <= -alpha3 + sigma`)? Confirm the exact assumption and whether it
  excludes a free-integrator (non-ISS) baseline. (c) Does the residual act only on latent/unmeasured states?
- **RENs, `2104.05942`**: is contraction enforced w.r.t. a STRICTLY positive-definite metric (`P > 0`),
  with incremental passivity enforced JOINTLY with contraction (not as a marginal alternative)? Is there
  ANY `P >= 0` / marginal / integrator-allowing variant? Quote the theorem.
Secondary:
- **Mabrok 2014, `1305.1079`**: confirm it treats poles at the origin (free body); note whether the
  conditions are purely LTI/transfer-function (no nonlinear or semidefinite-storage version). Confirm the
  attribution (Mabrok et al., not Lanzon-Petersen, for this arXiv ID).
- **NINODE, `2504.19497`**: confirm it is a CONTROLLER (stabilizes/damps an NI plant), not a forward-model
  augmentation, and whether it re-imposes a strict DC-gain / excludes the marginal case.
- **passive LuGre PINN, `2504.12441`**: confirm nonzero-at-rest friction via a LuGre bristle latent state,
  and whether passivity is BY CONSTRUCTION or only fit (the catalog claims only-fit).
- Spot-check any `[disk]` quotes in §A of the catalog against the framework PDFs if time permits.

## TASK 2 -- RED-TEAM the central gap claim
`drift-diagnosis-status.md` §5L asserts (paraphrased): *no published method provides a LEARNED DISSIPATIVE/
PASSIVE FORWARD augmentation whose stability guarantee PRESERVES a marginally-stable (free-integrator /
pole-at-origin / non-ISS) baseline mode; every candidate either assumes ISS/attractor, or enforces
contraction, or is a controller that damps the plant.*
- Try to FALSIFY it: find one published method (search freely) that gives a learned dissipative FORWARD
  model/augmentation which keeps a pole exactly at the origin (does not damp it) with a bounded-POSITION
  guarantee. Check `2011.13492`, `2410.00976`, `2011.14610` on disk, and search beyond.
- Also check the logical step in §5j: "passivity/dissipativity bounds VELOCITY (L2), not POSITION on a free
  integrator; position can grow O(sqrt(T))." Is that correct? Is the claim that our X/Y axes are
  mass-DAMPERS (finite tau, not pure double integrators) consistent with a bounded-position argument?

## Report format
For each item: CONFIRMED / CORRECTED / REFUTED, with the exact source location (PDF filename + page/eq/prop)
and a one-line reason. End with a verdict on the central gap claim (holds / holds-with-caveats / refuted by
<citation>) and any wrong quotes found in the catalog.
