"""Appendix table: the main-text models on the training, validation and test records in one table
(D-269: every result for every split). Per split one block: the baseline at the true parameters, the
augmented model unconstrained (R4) and with OBC (R1, the final model), and the black box. No physical-part
rows (Dirk 2026-10-24). NRMS_e per axis in %, mean over the split's
records with and without the training multisine (D-256); test over all 11 test records, as Table A. No
noise-floor row: Section V states the floor through the SNR (Dirk 2026-10-24).

Reads Results/res_errors.csv (Code/Evaluation/make_results.py), writes Writing/tables/res_all_splits.tex.
"""
from _fig_common import AXES, AXTEX, errors, mean_over, save_table
from res_structure import records

STEM = "res_all_splits"
SPLITS = [("train", "Training records"), ("val", "Validation records"), ("test", "Test records")]
ROWS = [(r"$\mathcal{M}_{\mathrm{LPV}}$", "R5", "base_true"),
        (r"$\mathcal{M}_{\mathrm{U}}$", "R4", "model"),
        (r"$\mathcal{M}_{\mathrm{OBC}}$", "R1", "model"),
        (r"$\mathcal{M}_{\mathrm{BB}}$", "R8", "bb")]


def main():
    e, ms = errors()
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l r r r r r r}", r"\toprule",
             r" & \multicolumn{3}{c}{Multisine} & \multicolumn{3}{c}{No multisine}\\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"$\mathrm{NRMS}_e$ (\%) & " + " & ".join("$%s$" % AXTEX[a] for _ in range(2) for a in AXES) + r"\\"]
    for split, name in SPLITS:
        r_ms, r_no = records(e, ms, split)
        lines += [r"\midrule", r"\multicolumn{7}{l}{\emph{%s} (%d with, %d without multisine)}\\" % (name, len(r_ms), len(r_no))]
        for lbl, run, model in ROWS:
            cells = [r"$%.1f$" % v for recs in (r_ms, r_no) for v in mean_over(e, split, run, model, recs)]
            lines.append(" & ".join([lbl] + cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    save_table("\n".join(lines), STEM)


if __name__ == "__main__":
    main()
