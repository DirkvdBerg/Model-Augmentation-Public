"""Step 5 (THESIS-RESULTS): relative error of the ten combinations, U vs OBC, two starts.

Reads Results/res_obc_combinations.csv (Code/Evaluation/make_results.py), writes
Writing/tables/res_obc_combinations.tex (a table* fragment body for \\input).
The overlap rows of D-246 are added once Code/Evaluation computes them.
"""
import csv
from collections import defaultdict

import numpy as np

from _fig_common import RESULTS_DIR, save_table

STEM = "res_obc_combinations"
SYMBOL = {"kb_sum": r"$k_{b,\Sigma}$", "cg1": r"$c_{g1}$", "cg2": r"$c_{g2}$", "cy": r"$c_y$",
          "cb_sum": r"$c_{b,\Sigma}$", "mh": r"$m_h$", "m_total": r"$m_\Sigma$",
          "m_diff": r"$m_\Delta$", "J_eff": r"$J_{\mathrm{eff}}$", "d": r"$d$"}
COLS = [("R4", "U"), ("R1", "OBC"), ("R5", "U"), ("R2", "OBC")]


def sci(x):
    e = int(np.floor(np.log10(abs(x))))
    return r"$%.2f\cdot10^{%d}$" % (x / 10 ** e, e)


def main():
    with open(RESULTS_DIR / f"{STEM}.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rel = defaultdict(dict)
    rms = {}
    for r in rows:
        rel[r["combination"]][r["run"]] = 100 * (float(r["learned"]) / float(r["true"]) - 1)
        if r["start"] == "detuned":
            rel[r["combination"]]["start"] = 100 * (float(r["start_value"]) / float(r["true"]) - 1)
        rms[r["run"]] = np.mean([float(r[f"test_rms_{a}"]) for a in ("X1", "X2", "Y")])

    lines = [r"\setlength{\tabcolsep}{4.5pt}", r"\begin{tabular}{l r r r r r}", r"\toprule",
             r" & \multicolumn{3}{c}{Detuned start} & \multicolumn{2}{c}{Correct start}\\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
             r"Rel. error (\%) & Start & U & OBC & U & OBC\\", r"\midrule"]
    fmt = lambda v: r"$%+.1f$" % v                                            # noqa: E731
    for c in SYMBOL:
        lines.append(" & ".join([SYMBOL[c], fmt(rel[c]["start"])] + [fmt(rel[c][run]) for run, _ in COLS]) + r"\\")
    lines.append(r"\midrule")
    norm = {k: np.sqrt(sum(rel[c][k] ** 2 for c in SYMBOL)) for k in ["start"] + [run for run, _ in COLS]}
    lines.append(" & ".join([r"Norm", r"$%.1f$" % norm["start"]] + [r"$%.1f$" % norm[run] for run, _ in COLS]) + r"\\")
    lines.append(" & ".join([r"Test RMS ($\times10^{-6}$ m)", ""]
                            + [r"$%.2f$" % (1e6 * rms[run]) for run, _ in COLS]) + r"\\")
    # D-246 inner products and the learned added-state mode (Code/Evaluation/eval_overlap.py)
    with open(RESULTS_DIR / "res_obc_overlap.csv", newline="") as fh:
        ov = {r["run"]: r for r in csv.DictReader(fh)}
    lines.append(r"\midrule")
    lines.append(" & ".join([r"Learned-term overlap", ""] + [r"$%.2f$" % float(ov[run]["network_overlap"])
                                                        for run, _ in COLS]) + r"\\")
    tru = {round(float(ov[run]["truth_overlap"]), 6) for run, _ in COLS}
    assert len(tru) == 1                       # a property of the data and the truth, not of a run
    lines.append(r"Unmodelled-dyn. overlap & &\multicolumn{4}{c}{$%.2f$}\\" % tru.pop())
    lines.append(" & ".join([r"Learned pole (Hz)", ""] + [r"$%.0f$" % float(ov[run]["mode_hz"])
                                                          for run, _ in COLS]) + r"\\")
    lines.append(" & ".join([r"Learned damping $\zeta$", ""] + [r"$%.3f$" % float(ov[run]["mode_zeta"])
                                                                for run, _ in COLS]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    save_table("\n".join(lines), STEM)


if __name__ == "__main__":
    main()
