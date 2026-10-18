"""Table C (Aspect 3, THESIS-RESULTS presentation 2026-10-16): relative error of the ten
combinations, U vs OBC, detuned and correct start, with the PREDICTED first-order bias of
eq:obc_recovery beside the observed error of each OBC run, the full-model and physical-part-alone
NRMS_e per axis (Gyorok 2025 Table 1), the network and truth overlaps and the learned mode.
One table per tuple stack (D-268): training (where theta_aux is formed: the prediction), validation
and test I1 to I6. The combination errors themselves do not depend on the stack. Agreement between
predicted and observed is not judged here (criterion open, D-268).

Reads Results/res_obc_combinations.csv, res_obc_accuracy.csv (Code/Evaluation/make_results.py),
res_obc_overlap.csv and res_obc_bias.csv (Code/Evaluation/eval_overlap.py); writes
Writing/tables/res_obc_combinations_<stack>.tex and res_obc_combinations.tex (= test, the name the
paper inputs), with the NRMS_e as the mean over the stack's records, and
res_obc_combinations_<stack>_ms.tex (appendix) with the NRMS_e rows split into records with and
without the training multisine (D-256, Dirk 2026-10-18). Columns of a run that has not finished are marked with a dagger and its label is
stated below the table.
"""
import csv
from collections import defaultdict

from _fig_common import RESULTS_DIR, nrms_pct, preliminary, save_table

STEM = "res_obc_combinations"
MAIN_STACK = "test"
SYMBOL = {"kb_sum": r"$k_{b,\Sigma}$", "cg1": r"$c_{g1}$", "cg2": r"$c_{g2}$", "cy": r"$c_y$",
          "cb_sum": r"$c_{b,\Sigma}$", "mh": r"$m_h$", "m_total": r"$m_\Sigma$",
          "m_diff": r"$m_\Delta$", "J_eff": r"$J_{\mathrm{eff}}$", "d": r"$d$"}
# (run, kind): kind 'obs' = observed error of the run, 'pred' = predicted first-order bias (its stack)
COLS = [("R4", "obs"), ("R1", "obs"), ("R1", "pred"), ("R5", "obs"), ("R2", "obs"), ("R2", "pred")]
M_ACT_MEAN = 0.5 * (10.2 + 10.7)   # (m1 + m2)/2 [kg], nominal values of Writing/sections/99_appendix.tex tab:nominal_parameters
AX = ("X1", "X2", "Y")
AXTEX = {"X1": "X_1", "X2": "X_2", "Y": "Y"}


def read(name):
    with open(RESULTS_DIR / name, newline="") as fh:
        return list(csv.DictReader(fh))


def table(stack, rel, start, status, acc, ov, pred, groups=("all",)):
    runs = list(dict.fromkeys(r for r, _ in COLS))
    dag = {r: r"$^\dagger$" if status[r].startswith("preliminary") else "" for r in runs}
    fmt = lambda v: "" if v is None else r"$%+.1f$" % v                             # noqa: E731
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l r r r r r r r}", r"\toprule",
             r" & \multicolumn{4}{c}{Detuned start} & \multicolumn{3}{c}{Correct start}\\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
             r" & Start & Unconstr.%s & OBC%s & OBC pred. & Unconstr.%s & OBC%s & OBC pred.\\"
             % (dag["R4"], dag["R1"], dag["R5"], dag["R2"]),
             r"\midrule", r"\multicolumn{8}{l}{Relative error (\%)}\\"]
    cell = lambda c, run, kind: rel[c].get(run) if kind == "obs" else pred.get((run, c))   # noqa: E731
    for c in SYMBOL:
        lines.append(" & ".join([SYMBOL[c], fmt(start.get(c))] + [fmt(cell(c, r, k)) for r, k in COLS]) + r"\\")
    norm = lambda vals: None if any(v is None for v in vals) else sum(v ** 2 for v in vals) ** 0.5   # noqa: E731
    nr = lambda v: "" if v is None else r"$%.1f$" % v                                # noqa: E731
    lines.append(" & ".join([r"Norm", nr(norm([start.get(c) for c in SYMBOL]))]
                            + [nr(norm([cell(c, r, k) for c in SYMBOL])) for r, k in COLS]) + r"\\")
    for grp in groups:
        what = "%s records" % stack if grp == "all" else "%s records, %s" % (stack, grp)
        lines += [r"\midrule", r"\multicolumn{8}{l}{$\mathrm{NRMS}_e$ (\%%), %s}\\" % what]
        for row, name in (("model", "Full model"), ("physics", "Physical part")):
            for a in AX:
                lines.append(" & ".join([r"%s, $%s$" % (name, AXTEX[a]), ""]
                                        + [r"$%.1f$" % nrms_pct(acc[(r, grp, row)]["nrms_e_%s" % a]) if k == "obs"
                                           else "" for r, k in COLS]) + r"\\")
    lines += [r"\midrule"]
    lines.append(" & ".join([r"Network overlap $\rho(F_{\mathrm{aug}})$", ""]
                            + [r"$%.2f$" % float(ov[r]["network_overlap"]) if k == "obs" else "" for r, k in COLS]) + r"\\")
    tru = {round(float(ov[r]["truth_overlap"]), 6) for r in runs}
    if len(tru) == 1:                       # a property of the data and the truth, not of a run
        lines.append(r"Truth overlap $\rho(\Xi)$ & & \multicolumn{6}{c}{$%.2f$}\\" % tru.pop())
    else:                                   # the tuples come from each run's encoder: say so if they differ
        lines.append(" & ".join([r"Truth overlap $\rho(\Xi)$", ""]
                                + [r"$%.2f$" % float(ov[r]["truth_overlap"]) if k == "obs" else "" for r, k in COLS]) + r"\\")
    lines.append(" & ".join([r"Learned pole (Hz)", ""] + [r"$%.0f$" % float(ov[r]["mode_hz"]) if k == "obs" else ""
                                                          for r, k in COLS]) + r"\\")
    lines.append(" & ".join([r"Learned damping $\zeta$", ""] + [r"$%.3f$" % float(ov[r]["mode_zeta"]) if k == "obs"
                                                                else "" for r, k in COLS]) + r"\\")
    lines += [r"\bottomrule"]
    for lab in preliminary(status[r] for r in runs):
        who = ", ".join(r for r in runs if status[r] == lab)
        lines.append(r"\multicolumn{8}{l}{\footnotesize $^\dagger$ %s: %s}\\" % (who, lab))
    lines += [r"\end{tabular}", ""]
    return "\n".join(lines)


def main():
    rel, start, status = defaultdict(dict), {}, {}
    for r in read(f"{STEM}.csv"):
        c, true = r["combination"], float(r["true"])
        # D-259: m_Delta = m1 - m2 (-0.5 kg) is scaled by the mean actuator mass (m1 + m2)/2, the others by |true|
        scale = M_ACT_MEAN if c == "m_diff" else abs(true)
        rel[c][r["run"]] = 100 * (float(r["learned"]) - true) / scale
        status[r["run"]] = r["status"]
        if r["start"] == "detuned":
            start[c] = 100 * (float(r["start_value"]) - true) / scale
    acc_rows, ov_rows, bias_rows = read("res_obc_accuracy.csv"), read("res_obc_overlap.csv"), read("res_obc_bias.csv")
    for stack in ("train", "val", "test"):
        acc = {(r["run"], r["group"], r["row"]): r for r in acc_rows if r["stack"] == stack}
        ov = {r["run"]: r for r in ov_rows if r["stack"] == stack}
        pred = {(r["run"], r["combination"]): 100 * float(r["predicted_rel"]) for r in bias_rows if r["stack"] == stack}
        body = table(stack, rel, start, status, acc, ov, pred)
        save_table("%% stack: %s\n" % stack + body, "%s_%s" % (STEM, stack))
        if stack == MAIN_STACK:
            save_table("%% stack: %s\n" % stack + body, STEM)
        grps = [g for g in ("multisine", "no multisine") if any(k[1] == g for k in acc)]
        save_table("%% stack: %s, multisine split (appendix)\n" % stack
                   + table(stack, rel, start, status, acc, ov, pred, grps), "%s_%s_ms" % (STEM, stack))


if __name__ == "__main__":
    main()
