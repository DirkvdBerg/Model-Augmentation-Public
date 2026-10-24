"""Appendix table: parameter recovery without added states (n_a = 0, detuned start, Jasper point 5),
unconstrained (R14) against OBC (R13): relative error of the ten combinations and their norm, the
full-model NRMS_e per axis, and the network and truth overlaps. Same layout and
scaling as Table C (res_obc_combinations.py); no learned mode (no added states).

Reads Results/res_obc_combinations.csv, res_obc_accuracy.csv (Code/Evaluation/make_results.py) and
res_obc_overlap.csv (Code/Evaluation/eval_overlap.py); writes Writing/tables/res_obc_na0_<stack>.tex
and res_obc_na0.tex (= test, as Table C).
"""
import csv
from collections import defaultdict

from _fig_common import RESULTS_DIR, nrms_pct, preliminary, save_table
from res_obc_combinations import AX, AXTEX, M_ACT_MEAN, SYMBOL

STEM = "res_obc_na0"
MAIN_STACK = "test"
RUNS = [("R14", r"$\mathcal{M}_{\mathrm{U}}$"), ("R13", r"$\mathcal{M}_{\mathrm{OBC}}$")]   # headers as in Table C


def read(name):
    with open(RESULTS_DIR / name, newline="") as fh:
        return list(csv.DictReader(fh))


def table(stack, rel, start, status, acc, ov):
    fmt = lambda v: r"$%+.1f$" % v                                                   # noqa: E731
    dag = {r: r"$^\dagger$" if status[r].startswith("preliminary") else "" for r, _ in RUNS}
    lines = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{l r r r}", r"\toprule",
             " & Initial & " +" & ".join(n + dag[r] for r, n in RUNS) + r"\\", r"\midrule",
             r"\multicolumn{4}{l}{Relative error (\%)}\\"]
    for c in SYMBOL:
        lines.append(" & ".join([SYMBOL[c], fmt(start[c])] + [fmt(rel[c][r]) for r, _ in RUNS]) + r"\\")
    norm = lambda k: sum((start[c] if k == "start" else rel[c][k]) ** 2 for c in SYMBOL) ** 0.5   # noqa: E731
    lines.append(" & ".join(["Norm", r"$%.1f$" % norm("start")] + [r"$%.1f$" % norm(r) for r, _ in RUNS]) + r"\\")
    lines += [r"\midrule", r"\multicolumn{4}{l}{$\mathrm{NRMS}_e$ (\%%), %s records}\\" % stack]
    # full model only; no physical-part rows (Dirk 2026-10-24)
    for a in AX:
        lines.append(" & ".join([r"$%s$" % AXTEX[a], ""]
                                + [r"$%.1f$" % nrms_pct(acc[(r, "model")]["nrms_e_%s" % a]) for r, _ in RUNS]) + r"\\")
    lines += [r"\midrule",
              " & ".join([r"Network overlap $\rho(F_{\mathrm{aug}})$", ""]
                         + [r"$%.2f$" % float(ov[r]["network_overlap"]) for r, _ in RUNS]) + r"\\",
              " & ".join([r"Truth overlap $\rho(\Xi)$", ""]
                         + [r"$%.2f$" % float(ov[r]["truth_overlap"]) for r, _ in RUNS]) + r"\\",
              r"\bottomrule"]
    for lab in preliminary(status[r] for r, _ in RUNS):
        lines.append(r"\multicolumn{4}{l}{\footnotesize $^\dagger$ %s}\\" % lab)
    lines += [r"\end{tabular}", ""]
    return "\n".join(lines)


def main():
    rel, start, status = defaultdict(dict), {}, {}
    for r in read("res_obc_combinations.csv"):
        if r["run"] not in dict(RUNS):
            continue
        c, true = r["combination"], float(r["true"])
        # D-259: m_Delta scaled by the mean actuator mass, the others by |true|
        scale = M_ACT_MEAN if c == "m_diff" else abs(true)
        rel[c][r["run"]] = 100 * (float(r["learned"]) - true) / scale
        start[c] = 100 * (float(r["start_value"]) - true) / scale       # both runs start detuned, identically
        status[r["run"]] = r["status"]
    acc_rows, ov_rows = read("res_obc_accuracy.csv"), read("res_obc_overlap.csv")
    for stack in ("train", "val", "test"):
        acc = {(r["run"], r["row"]): r for r in acc_rows if r["stack"] == stack and r.get("group", "all") == "all"}
        ov = {r["run"]: r for r in ov_rows if r["stack"] == stack}
        body = "%% stack: %s\n" % stack + table(stack, rel, start, status, acc, ov)
        save_table(body, "%s_%s" % (STEM, stack))
        if stack == MAIN_STACK:
            save_table(body, STEM)


if __name__ == "__main__":
    main()
