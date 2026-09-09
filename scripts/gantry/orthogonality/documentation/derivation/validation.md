# Document validation — 2026-09-09

## Newly executed

Both commands ran in the `GraduationProject` conda environment, on the development
machine, after inspecting the implementation. Counts below describe numerical
implementation checks on small constructed examples, not gantry efficacy measurements.

| Command, relative to repository root | Result | Scope |
|---|---|---|
| `python scripts/gantry/orthogonality/testbed/test_trajectory_projection.py` | 32 passed, 0 failed; exit 0 | Shared geometry, actual compressed nuisance function, serialization, exact chunked gradients and diagnostics |
| `python scripts/gantry/orthogonality/testbed/test_penalty_hook.py` | 12 passed, 0 failed; exit 0 | Selective parameter gradients, MSE normalization, chunk parity and closure ordering |

The local LaTeX build used pdfLaTeX and BibTeX from MiKTeX. Initial sandboxed startup
could not access MiKTeX's configuration; execution with the required access succeeded.
An initial font-expansion failure was corrected by explicitly selecting Latin Modern.
Cross-reference and bibliography passes were then run. Final log checks and PDF inspection
are recorded in `inspection-manifest.json`.

## Source inspection

- Györök 2025: local arXiv v2; state-space/encoder text, Eqs. (9)–(19), offset column,
  frozen/update discussion. Published bibliographic metadata checked against PMLR.
- Györök 2026: local arXiv v3; Condition 4 and Theorem 7 on PDF p. 4.
  Journal metadata checked against the publisher/TU/e records. Final journal theorem
  numbering was not substituted for the inspected version.
- Kon 2023: local arXiv v2; finite-time SVD and fixed physical basis, Eqs. (15)–(23).
- Bolderman: local arXiv v2 (2023 version of the work catalogued as 2024); the parameter
  anchor, Remark 3.1, Eq. (38), and separate neural regularization L-curve discussion.
- Kessels 2025 thesis: title record and printed pp. 156–157, Eqs. (5.12)–(5.13d).

## Not newly executed

MATLAB, the full SymPy reference suite, the gantry geometry ladder, the one-update/resume
experiment, the latent initialization probe, compiled/GPU runs, noisy/perturbed runs and
efficacy comparisons were not rerun for this writing task. Their status is taken from
explicitly identified existing records, never from test-script availability alone.

The hand proofs in the document establish identities under their written assumptions.
Passing the numerical checks is supporting implementation evidence, not their logical proof.

## Critique revision — 2026-09-09

The earlier 44 production-core/hook checks were not repeated for this prose/mathematics-only
revision. Five new exact SymPy checks passed in `GraduationProject`:

1. For `J=(y-theta-a)**2`, `V=beta*a**2`, the selective gradient vanishes at `theta=y,a=0`.
2. The selective field Jacobian determinant equals `4*beta`, hence is nonsingular for positive beta.
3. With `S=[[1,0],[0,1],[1,1]]` and `Delta_perp=b*[-1,-1,1]`, `S.T*Delta_perp=0`.
4. For `Delta=S*c+Delta_perp`, `S.pinv()*Delta=c` exactly.
5. With `P=S*(S.T*S).inv()*S.T`, `(I-P)*Delta=Delta_perp` exactly.

These checks concern the explicitly specified affine examples, not a new FP structural proof,
a nonlinear gantry experiment or a completed beta sweep. The complete MATLAB/SymPy reference
suites were not rerun. The Golub–Pereyra publisher abstract and metadata were inspected at
https://epubs.siam.org/doi/10.1137/0710036; no uninspected full-text theorem is attributed to it.

The revised PDF was built with pdfLaTeX, BibTeX and two further pdfLaTeX passes. The final
artifact hashes and revision validation are recorded in `inspection-manifest.json`.
