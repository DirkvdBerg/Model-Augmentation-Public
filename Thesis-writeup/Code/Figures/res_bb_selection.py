"""Appendix table: the black-box search (R8 grid, D-264). One row per configuration: state count, network
width, learning rate, number of parameters, best validation NRMS_e (mean over the validation records
and axes, in %), the update of that best checkpoint, and the run status (failed = non-finite training
loss, recorded with its update). The selected configuration (lowest NRMS_e; within 2 % the fewer
parameters, eval_blackbox.select_bb) is marked.

Reads Results/res_bb_selection.csv (Code/Evaluation/eval_blackbox.py), writes
Writing/tables/res_bb_selection.tex.
"""
import csv
import math

from _fig_common import RESULTS_DIR, save_table

STEM = "res_bb_selection"


def main():
    with open(RESULTS_DIR / f"{STEM}.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    # footnotesize, 3 pt and a wrapping note: at 4 pt in body size the table was 22 pt wider
    # than a column (Dirk 2026-10-25).
    lines = [r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{r r r r r r r l}", r"\toprule",
             r"\# & $n_x$ & Width & LR & Param. & $\mathrm{NRMS}_e$ (\%) & Best update & Status\\", r"\midrule"]
    for r in rows:
        npar = float(r["n_params"])
        m, e = r["lr"].split("e")                                   # '3e-03' -> 3, -3
        cells = [r["index"], r["nx"], r["width"], r"$%s\cdot10^{%d}$" % (m, int(e)),
                 "" if math.isnan(npar) else "%d" % npar, r"$%.1f$" % (100 * float(r["best_val_nrms_e"])),
                 r["best_update"], r["status"]]
        if r["selected"] == "1":
            cells = [r"\textbf{%s}" % c if c else c for c in cells]
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\multicolumn{8}{p{0.9\columnwidth}}{Bold: selected on validation (lowest "
              r"$\mathrm{NRMS}_e$; within 2\,\% the fewer parameters).}\\", r"\end{tabular}", ""]
    save_table("\n".join(lines), STEM)


if __name__ == "__main__":
    main()
