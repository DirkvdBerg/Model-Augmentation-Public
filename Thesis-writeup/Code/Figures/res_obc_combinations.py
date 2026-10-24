"""Table C (Aspect 2): relative error of the ten identifiable combinations after training, unconstrained
(R4 detuned start, R5 true start) and OBC (R1 detuned start, R2 true start), with the detuned initial
value, the norm to the truth and the norm of the difference between the two starts of one model
(uniqueness, Section I definition). A table, not a plot (Dirk 2026-10-24). Overlaps and accuracy are
Table D (res_obc_constraint.py).

Reads Results/res_obc_combinations.csv (Code/Evaluation/make_results.py), writes
Writing/tables/res_obc_combinations.tex.
"""
from collections import defaultdict

from _fig_common import AXES as AX, AXTEX, read_csv, save_table

STEM = "res_obc_combinations"
SYMBOL = {"kb_sum": r"$k_{b,\Sigma}$", "cg1": r"$c_{g1}$", "cg2": r"$c_{g2}$", "cy": r"$c_y$",
          "cb_sum": r"$c_{b,\Sigma}$", "mh": r"$m_h$", "m_total": r"$m_\Sigma$",
          "m_diff": r"$m_\Delta$", "J_eff": r"$J_{\mathrm{eff}}$", "d": r"$d$"}
GROUPS = [["kb_sum"], ["cg1", "cg2", "cy", "cb_sum"], ["mh", "m_total", "m_diff", "J_eff"], ["d"]]
COLS = ["R4", "R5", "R1", "R2"]          # U detuned, U true, OBC detuned, OBC true
M_ACT_MEAN = 0.5 * (10.2 + 10.7)   # (m1 + m2)/2 [kg], nominal values of Writing/sections/99_appendix.tex tab:nominal_parameters


def rel_errors():
    """Relative errors in % of the learned combinations and of the detuned start (D-259 scaling)."""
    rel, start = defaultdict(dict), {}
    for r in read_csv(STEM):
        c, true = r["combination"], float(r["true"])
        # D-259: m_Delta = m1 - m2 (-0.5 kg) is scaled by the mean actuator mass (m1 + m2)/2, the others by |true|
        scale = M_ACT_MEAN if c == "m_diff" else abs(true)
        rel[c][r["run"]] = 100 * (float(r["learned"]) - true) / scale
        if r["start"] == "detuned":
            start[c] = 100 * (float(r["start_value"]) - true) / scale
    return rel, start


def main():
    rel, start = rel_errors()
    norm = lambda vals: sum(v ** 2 for v in vals) ** 0.5                          # noqa: E731
    fmt = lambda v: r"$%+.1f$" % v                                                # noqa: E731
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l r r r r r}", r"\toprule",
             r" & & \multicolumn{2}{c}{$\mathcal{M}_{\mathrm{U}}$} & \multicolumn{2}{c}{$\mathcal{M}_{\mathrm{OBC}}$}\\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Rel.\ error (\%) & Detuned start & Detuned & True & Detuned & True\\"]   # was "Initial" (Dirk 2026-10-24)
    for g in GROUPS:
        lines.append(r"\midrule")
        for c in g:
            name = SYMBOL[c] + (r"$^*$" if c == "m_diff" else "")
            lines.append(" & ".join([name, fmt(start[c])] + [fmt(rel[c][r]) for r in COLS]) + r"\\")
    lines += [r"\midrule",
              " & ".join(["Distance to truth", r"$%.1f$" % norm(start.values())]
                         + [r"$%.1f$" % norm(rel[c][r] for c in SYMBOL) for r in COLS]) + r"\\",
              " & ".join(["Distance between starts", ""]
                         + [r"\multicolumn{2}{c}{$%.1f$}" % norm(rel[c][a] - rel[c][b] for c in SYMBOL)
                            for a, b in (("R4", "R5"), ("R1", "R2"))]) + r"\\",
              r"\bottomrule",
              # two rows: one long row widened the last column (Dirk 2026-10-24)
              r"\multicolumn{6}{l}{\footnotesize $^*$ relative to the mean actuator mass,}\\",
              r"\multicolumn{6}{l}{\footnotesize because its true value is close to zero.}\\",
              r"\end{tabular}", ""]
    save_table("\n".join(lines), STEM)


if __name__ == "__main__":
    main()
