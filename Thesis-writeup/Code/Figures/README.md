# Section III diagrams

The active closed-loop figure is owned entirely by `Thesis-writeup/`:

- `closed_loop_same_reference.tikz.tex`: editable shared-reference loop diagram.
- `closed_loop_same_reference.tex`: standalone compilation wrapper.
- `augmentation_diagrams.py`: compiles in a temporary directory and publishes
  `Writing/figures/closed_loop_same_reference.pdf`.

Run `python Thesis-writeup/Code/Figures/augmentation_diagrams.py` from the repository
root. The publisher also works from other directories. It needs `pdflatex` on PATH.
The paper includes the generated PDF; all inputs remain inside `Writing/` when
that directory is shared separately.

The common-reference diagram adapts the historical meeting diagram at
`scripts/gantry/meeting/meeting-31-08-2026/slides/closed-loop-same-reference-v1.tex`.
That file is provenance only, not a build dependency. Both loops visibly share
one reference and one feedforward. Signals are time-indexed; the model's state
register remains in the architecture figure. The residual reduction, reference
cancellation, update timing and initial-state assumptions are explained in Section III,
without a separate residual panel or delay block in this diagram.

The author-edited architecture is currently maintained directly in
`Writing/figures/tex/augmentation_structure.tex`. The publisher deliberately does
not overwrite it from the older `augmentation_structure.tex` in this directory.
`augmentation_horizons.tex` and `augmentation_closed_loop.tex` are inactive drafts.
