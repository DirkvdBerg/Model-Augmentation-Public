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

# Results figures and tables, final set (2026-10-24)

Agreed with Dirk on 2026-10-24 (results preview, `Thesis-writeup/Preview/`). All from the final CSVs:
`THESIS_RESULTS_DIR="$PWD/Thesis-writeup/Results/24-10-2026/CSV" conda run -n GraduationProject python Thesis-writeup/Code/Figures/<stem>.py`.
No metres, no absolute plant gain (confidentiality); frequencies in Hz; no noise-floor rows.

| Output (`Writing/`) | Script | Source | Float |
|-|-|-|-|
| `tables/res_structure.tex` (+ `_train/_val/_test`) | `res_structure.py` | `res_errors.csv` | Table A, all structures, all 11 test records (`tab:res_structure`); `_val` = `tab:app_structure` |
| `figures/res_time_trace.pdf` | `res_time_trace.py` | `res_time_trace.csv` | I1, Y, normalised by sigma_e (`fig:res_time_trace`) |
| `figures/res_val_curves.pdf` | `res_val_curves.py` | run logs `Results/18-10-2026/thesis-results-4` | validation curves R3, R5, R6, R7 (`fig:res_val_curves`) |
| `figures/res_frf.pdf` | `res_frf.py` | `res_frf.csv` | F_Y to Y at Y = 0, relative dB (`fig:res_frf`) |
| `tables/res_obc_combinations.tex` | `res_obc_combinations.py` | `res_obc_combinations.csv` | Table C, ten combinations, both starts (`tab:res_obc`) |
| `tables/res_obc_constraint.tex` | `res_obc_constraint.py` | `res_obc_overlap.csv`, `res_obc_accuracy.csv` | Table D, overlaps and accuracy (`tab:res_obc_constraint`) |
| `tables/res_generalisation.tex` | `res_generalisation.py` | `res_generalisation.csv` | Table E, I against E (`tab:res_generalisation`) |
| `tables/res_all_splits.tex` | `res_all_splits.py` | `res_errors.csv` | appendix, all splits (`tab:app_splits`) |
| `figures/res_frf_full.pdf`, `res_lpv_lti.pdf`, `res_ps_spectrum.pdf` | same stems | same stems | appendix |
| `tables/res_obc_na0.tex`, `res_bb_selection.tex` | same stems | same stems | appendix |

# Results figures and tables (2026-10-11 runs, superseded by the final set above)

Agreed set with Dirk on 2026-10-11 (D-247). Each output comes from one script, which reads one CSV:

| Output (`Writing/`) | Script | CSV (`Results/`) | Answers |
|-|-|-|-|
| `figures/res_lpv_lti.pdf` | `res_lpv_lti.py` | `res_lpv_lti.csv` | Aspect 1: LPV vs LTI baseline (X axes) |
| `tables/res_structure.tex` | `res_structure.py` | `res_structure.csv` | Aspect 2 (+ black box): test RMS per axis, I1 to I6 |
| `figures/res_ps_spectrum.pdf` | `res_ps_spectrum.py` | `res_ps_spectrum.csv` | Aspect 2: Y simulation-error spectrum, plain vs PS |
| `tables/res_obc_combinations.tex` | `res_obc_combinations.py` | `res_obc_combinations.csv`, `res_obc_overlap.csv` | Aspect 3: ten combinations, overlaps, learned pole |
| `figures/res_generalisation.pdf` | `res_generalisation.py` | `res_generalisation.csv` | Aspect 4: I/E pairs, detuned baseline vs augmented vs black box |
| `figures/res_time_trace.pdf` | `res_time_trace.py` | `res_time_trace.csv` | I1 time trace, Y axis |

Regenerate a figure: `python Thesis-writeup/Code/Figures/<stem>.py` (seconds, no rollout).
Regenerate the CSVs: `python Thesis-writeup/Code/Evaluation/make_results.py` (seconds). It reads
the runs' own `gantry_results_<id>.npz` in `Results/11-10-2026/` and the saved predictions in
`Results/thesis-eval/preds_{R3,R5,BB}.npz`. Only if those caches are lost, rerun (about 4 min each):
`eval_blackbox.py`, `eval_greybox.py R5:base_true,clamped`, `eval_greybox.py R3`,
`eval_greybox.py --lti R5`, `eval_overlap.py R4 R1 R5 R2` (writes `res_obc_overlap.csv`).
Run ids: R1 88762, R2 88755, R3 88763, R4 88764, R5 88767, R6 88768, R7 88756, black box 88757
index 3 (`SS_encoder_general_hf_5rVESQ`).
