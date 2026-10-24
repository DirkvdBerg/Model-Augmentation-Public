"""Table D (Aspect 2): whether the orthogonality constraint holds and what it costs, unconstrained (R4)
against OBC (R1), both from the detuned start: the network overlap rho(F_aug) on the training and the
test records, the truth overlap rho(Xi) (eq:obc_overlap), and the NRMS_e per axis of the full model and
of the physical part alone (learned block off) on all test records.

Reads Results/res_obc_overlap.csv (Code/Evaluation/eval_overlap.py) and res_obc_accuracy.csv
(Code/Evaluation/make_results.py), writes Writing/tables/res_obc_constraint.tex.
"""
from _fig_common import AXES, AXTEX, read_csv, save_table

STEM = "res_obc_constraint"
RUNS = ["R4", "R1"]


def main():
    ov = {(r["run"], r["stack"]): r for r in read_csv("res_obc_overlap")}
    acc = {(r["run"], r["stack"], r["group"], r["row"]): r for r in read_csv("res_obc_accuracy")}
    lines = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{l r r}", r"\toprule",
             r" & $\mathcal{M}_{\mathrm{U}}$ & $\mathcal{M}_{\mathrm{OBC}}$\\", r"\midrule"]
    for stack, name in (("train", "training"), ("test", "test")):
        lines.append(" & ".join([r"$\rho(F_{\mathrm{aug}})$, %s records" % name]
                                + [r"$%.2f$" % float(ov[(r, stack)]["network_overlap"]) for r in RUNS]) + r"\\")
    lines.append(" & ".join([r"$\rho(\Xi)$, test records"]
                            + [r"$%.2f$" % float(ov[(r, "test")]["truth_overlap"]) for r in RUNS]) + r"\\")
    lines += [r"\midrule", r"\multicolumn{3}{l}{$\mathrm{NRMS}_e$ (\%), test records}\\"]
    for row, name in (("model", "Full model"), ("physics", "Physical part")):
        for a in AXES:
            lines.append(" & ".join([r"%s, $%s$" % (name, AXTEX[a])]
                                    + [r"$%.1f$" % (100 * float(acc[(r, "test", "all", row)]["nrms_e_%s" % a]))
                                       for r in RUNS]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    save_table("\n".join(lines), STEM)


if __name__ == "__main__":
    main()
