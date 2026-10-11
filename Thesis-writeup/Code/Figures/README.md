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

# Results figures and tables (2026-10-11 runs)

Agreed set with Dirk on 2026-10-11 (D-247). Each output comes from one script, which reads one CSV:

| Output (`Writing/`) | Script | CSV (`Results/`) | Answers |
|-|-|-|-|
| `figures/res_lpv_lti.pdf` | `res_lpv_lti.py` | `res_lpv_lti.csv` | Aspect 1: LPV vs LTI baseline (X axes) |
| `tables/res_structure.tex` | `res_structure.py` | `res_structure.csv` | Aspect 2 (+ black box): test RMS per axis, I1 to I6 |
| `figures/res_ps_spectrum.pdf` | `res_ps_spectrum.py` | `res_ps_spectrum.csv` | Aspect 2: Y simulation-error spectrum, plain vs PS |
| `tables/res_obc_combinations.tex` | `res_obc_combinations.py` | `res_obc_combinations.csv`, `res_obc_overlap.csv` | Aspect 3: ten combinations, overlaps, learned pole |
| `figures/res_generalisation.pdf` | `res_generalisation.py` | `res_generalisation.csv` | Aspect 4: I/E pairs, baseline vs augmented vs black box |
| `figures/res_time_trace.pdf` | `res_time_trace.py` | `res_time_trace.csv` | I1 time trace, Y axis |

Regenerate a figure: `python Thesis-writeup/Code/Figures/<stem>.py` (seconds, no rollout).
Regenerate the CSVs: `python Thesis-writeup/Code/Evaluation/make_results.py` (seconds). It reads
the runs' own `gantry_results_<id>.npz` in `Results/11-10-2026/` and the saved predictions in
`Results/thesis-eval/preds_{R3,R5,BB}.npz`. Only if those caches are lost, rerun (about 4 min each):
`eval_blackbox.py`, `eval_greybox.py R5:base_true,clamped`, `eval_greybox.py R3`,
`eval_greybox.py --lti R5`, `eval_overlap.py R4 R1 R5 R2` (writes `res_obc_overlap.csv`).
Run ids: R1 88762, R2 88755, R3 88763, R4 88764, R5 88767, R6 88768, R7 88756, black box 88757
index 3 (`SS_encoder_general_hf_5rVESQ`).
