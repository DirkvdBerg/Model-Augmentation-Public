"""Table D (Aspect 2): whether the orthogonality constraint holds and what it costs, unconstrained (R4)
against OBC (R1), both from the detuned start: the network overlap rho(F_aug) and the truth overlap
rho(Xi) (eq:obc_overlap) on the training, validation and test records (validation added, Dirk 2026-10-25),
and the NRMS_e per axis of the full model over the records of Table II, with and without a multisine
(no physical-part rows, Dirk 2026-10-24).

Reads Results/res_obc_overlap.csv (Code/Evaluation/eval_overlap.py) and res_obc_accuracy.csv
(Code/Evaluation/make_results.py), writes Writing/tables/res_obc_constraint.tex.
"""
from _fig_common import AXES, AXTEX, errors, read_csv, save_table
from res_structure import mean_pairs, records

STEM = "res_obc_constraint"
RUNS = ["R4", "R1"]


def main():
    ov = {(r["run"], r["stack"]): r for r in read_csv("res_obc_overlap")}
    acc = {(r["run"], r["stack"], r["group"], r["row"]): r for r in read_csv("res_obc_accuracy")}
    lines = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{l r r}", r"\toprule",
             r" & $\mathcal{M}_{\mathrm{U}}$ & $\mathcal{M}_{\mathrm{OBC}}$\\", r"\midrule"]
    # the test overlaps use I1 to I6 only (eval_overlap.py)
    splits = (("train", "training"), ("val", "validation"), ("test", "interpolation test"))
    for stack, name in splits:
        lines.append(" & ".join([r"$\rho(F_{\mathrm{aug}})$, %s records" % name]
                                + [r"$%.2f$" % float(ov[(r, stack)]["network_overlap"]) for r in RUNS]) + r"\\")
    # the truth overlap depends on the data only, so it is printed once across both columns
    for stack, name in splits:
        xi = {"%.2f" % float(ov[(r, stack)]["truth_overlap"]) for r in RUNS}
        assert len(xi) == 1, xi
        lines.append(r"$\rho(\Xi)$, %s records & \multicolumn{2}{c}{$%s$}\\" % (name, xi.pop()))
    # NRMS_e over the same records as Table II: 10 with a multisine, 8 without (Dirk 2026-10-24)
    e, ms = errors()
    r_ms, r_no = records(e, ms, "test")
    groups = [("with a multisine", [("test", x) for x in r_ms] + [("twins", x) for x in records(e, ms, "twins")[0]]),
              ("without", [("test", x) for x in r_no])]
    for name, recs in groups:
        lines += [r"\midrule", r"\multicolumn{3}{l}{$\mathrm{NRMS}_e$ (\%%), %d records %s}\\" % (len(recs), name)]
        mean = {r: mean_pairs(e, r, "model", recs) for r in RUNS}
        for k, a in enumerate(AXES):
            lines.append(" & ".join([r"$%s$" % AXTEX[a]] + [r"$%.1f$" % mean[r][k] for r in RUNS]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    save_table("\n".join(lines), STEM)


if __name__ == "__main__":
    main()
