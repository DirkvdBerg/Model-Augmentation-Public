# Trajectory orthogonality derivation

Start with `trajectory-orthogonality.pdf`. The editable sources are
`trajectory-orthogonality.tex` and `trajectory-orthogonality.bib`.

## Overleaf

Upload `overleaf.zip` as a new project, or upload the `.tex` and `.bib` files together.
Select `trajectory-orthogonality.tex` as the main document and pdfLaTeX as the compiler.
Only standard TeX packages are used. The repository paths printed in the appendix are
references for readers; compilation does not access those paths or require the repository.

Local build, from this directory:

```text
pdflatex -interaction=nonstopmode -halt-on-error trajectory-orthogonality.tex
bibtex trajectory-orthogonality
pdflatex -interaction=nonstopmode -halt-on-error trajectory-orthogonality.tex
pdflatex -interaction=nonstopmode -halt-on-error trajectory-orthogonality.tex
```

## Scope

This is a thesis derivation and inspection snapshot dated 2026-09-09. It is not another
live implementation plan. The existing plan, `gaps.md`, and verification report retain
their roles. No production code or existing evidence records were changed for this document.

The narrative derives the original stacked projection, the zero-latent-slice limitation,
the closed-loop predictor, physical/latent encoder ownership, the compressed nuisance
basis, the local least-squares displacement result, the selective update, exact chunking,
and diagnostics. It explicitly separates value orthogonality from tangent separation and
parameter recovery.

Appendix A records discrepancies and qualifications found in the current code, notably
model-precision Jacobians versus the plan's float64 preference, split storage of nuisance
spectra, cache provenance/metadata limitations, and the need to check parameter anchoring
in each actual run. Appendix B identifies completed historical evidence and open questions.

`inspection-manifest.json` records source hashes because the repository was not a clean
committed snapshot. It is an inspection fingerprint, not a complete training reproducibility
manifest. `validation.md` records checks actually performed while preparing this document.

Source page/equation references identify the specific local PDF versions inspected.
The bibliography deliberately distinguishes those preprints from journal/proceedings
publication metadata. No blanket parameter-recovery theorem is imported from the sources.

## Critique revision — 2026-09-09

The revised document adds an exact example of aligned true discrepancy being assigned to
physical parameters, a general affine counterpart, local Taylor-remainder diagnostics,
the target-based relative drift bound, explicit encoder/off-model interpretation, and
the Golub–Pereyra background citation. It distinguishes physical-unit invariance from
arbitrary parameter-coordinate invariance and describes a controlled-discrepancy experiment
without claiming it has been run. The PDF and Overleaf bundle were rebuilt together.
