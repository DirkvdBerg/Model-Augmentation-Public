"""Appendix table: the main-text models on the training, validation and test records in one table
(D-269: every result for every split; Dirk 2026-10-18). Per split one block: the baseline at the true
parameters, the augmented model unconstrained (R4) and with OBC (R1, the final model), each with its
physical part alone (learned block off, learned parameters), the black box and the noise floor.
NRMS_e per axis in %, mean over the split's records, with and without the training multisine
(D-256); test over I1 to I6 as in the main text.

Reads Results/res_structure.csv and res_obc_accuracy.csv (Code/Evaluation/make_results.py), writes
Writing/tables/res_all_splits.tex. Rows of a run that has not finished carry a dagger.
"""
import csv

from _fig_common import RESULTS_DIR, nrms_pct, preliminary, save_table

STEM = "res_all_splits"
SPLITS = [("train", "Training records"), ("val", "Validation records"), ("test", "Test records I1 to I6")]
GROUPS = ("multisine", "no multisine")
AX = ("X1", "X2", "Y")
# (label, source, key): source 'structure' = res_structure.csv label, 'physics' = res_obc_accuracy.csv run
ROWS = [(r"Baseline, true $\theta$", "structure", "Baseline, true parameters"),
        ("Augmented, unconstrained", "structure", "PS start, $n_a=2$, detuned start"),
        (r"\quad physical part", "physics", "R4"),
        ("Augmented, OBC (final)", "structure", "Final model (OBC), detuned start"),
        (r"\quad physical part", "physics", "R1"),
        ("Black-box model", "structure", "Black box"),
        ("Noise floor", "structure", "Noise floor")]


def read(name):
    with open(RESULTS_DIR / name, newline="") as fh:
        return list(csv.DictReader(fh))


def cell(v):
    v = nrms_pct(v)
    return (r"$%.1f$" if v >= 0.1 else r"$%.2f$") % v             # the floor is about 0.05 %


def main():
    st = {(r["split"], r["group"], r["label"]): r for r in read("res_structure.csv")}
    ph = {(r["stack"], r["group"], r["run"]): r for r in read("res_obc_accuracy.csv") if r["row"] == "physics"}
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l r r r r r r}", r"\toprule",
             r" & \multicolumn{3}{c}{Multisine} & \multicolumn{3}{c}{No multisine}\\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"$\mathrm{NRMS}_e$ (\%) & $X_1$ & $X_2$ & $Y$ & $X_1$ & $X_2$ & $Y$\\"]
    labels = []
    for split, name in SPLITS:
        lines += [r"\midrule", r"\multicolumn{7}{l}{\emph{%s}}\\" % name]
        for lbl, src, key in ROWS:
            rs = [st.get((split, g, key)) if src == "structure" else ph.get((split, g, key)) for g in GROUPS]
            if all(r is None for r in rs):
                continue
            status = next(r["status"] for r in rs if r is not None)
            labels.append(status)
            dag = r"$^\dagger$" if status.startswith("preliminary") else ""
            cells = [cell(r["nrms_e_%s" % a]) if r is not None else "" for r in rs for a in AX]
            lines.append(" & ".join([lbl + dag] + cells) + r"\\")
    lines.append(r"\bottomrule")
    for lab in preliminary(labels):
        lines.append(r"\multicolumn{7}{l}{\footnotesize $^\dagger$ %s}\\" % lab)
    lines += [r"\end{tabular}", ""]
    save_table("\n".join(lines), STEM)


if __name__ == "__main__":
    main()
