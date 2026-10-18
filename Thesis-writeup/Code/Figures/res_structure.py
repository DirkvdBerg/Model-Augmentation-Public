"""Table I (Aspect 2, with the black box of Aspect 4): NRMS_e per axis in % (the primary metric,
05_setup eq:setup_rms) and, in a second table, the RMS in m, each the mean over the records of one
split and record group: with and without the training multisine reported separately (D-256), on
test over I1 to I6 only. One table per split (D-269); the validation table decides the design
choices and is also written as `res_structure.tex`, the name the paper inputs.

Reads Results/res_structure.csv (Code/Evaluation/make_results.py), writes
Writing/tables/res_structure_<split>.tex (NRMS_e), res_structure_<split>_m.tex (m) and
res_structure.tex (= validation, NRMS_e). Rows from a run that has not finished are marked with a
dagger and its label is stated below the table.
"""
import csv

import numpy as np

from _fig_common import RESULTS_DIR, nrms_pct, preliminary, save_table

STEM = "res_structure"
MAIN_SPLIT = "val"
SPLIT_NAME = {"train": "training", "val": "validation", "test": "test (I1 to I6)", "twins": "multisine twins"}
SHORT = {"Baseline, true parameters": r"Baseline, true $\theta$",
         "Baseline, detuned start": r"Baseline, detuned $\theta$",
         "Static parallel, $n_a=0$": r"Static parallel, $n_a{=}0$",
         "Static parallel, $n_a=0$, detuned start": r"Static parallel, $n_a{=}0$, detuned $\theta$",
         "Static parallel (OBC), $n_a=0$, detuned start": r"Static parallel (OBC), $n_a{=}0$, detuned $\theta$",
         "Plain start, $n_a=2$": r"Plain training, $n_a{=}2$",
         "PS start, $n_a=2$": r"PS init., $n_a{=}2$",
         "PS start, $n_a=2$, physics part only": r"PS init., baseline part only",
         "PS start, $n_a=8$": r"PS init., $n_a{=}8$",
         "PS start, $n_a=2$, detuned start": r"PS init., detuned $\theta$",
         "Final model (OBC), detuned start": r"Augmented (OBC), detuned $\theta$",
         "Black box": "Black-box model"}
AX = ("X1", "X2", "Y")
AXTEX = {"X1": "$X_1$", "X2": "$X_2$", "Y": "$Y$"}


def sci(x):
    e = int(np.floor(np.log10(abs(x))))
    return r"$%.2f\times10^{%d}$" % (x / 10 ** e, e)


def table(rows, metric):
    groups = list(dict.fromkeys(r["group"] for r in rows))
    labels = list(dict.fromkeys(r["label"] for r in rows))
    val = {(r["label"], r["group"]): r for r in rows}
    n = len(AX) * len(groups)
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l %s}" % " ".join(["r"] * n), r"\toprule",
             " & " + " & ".join(r"\multicolumn{3}{c}{%s}" % g.capitalize() for g in groups) + r"\\",
             "".join(r"\cmidrule(lr){%d-%d}" % (2 + 3 * k, 4 + 3 * k) for k in range(len(groups))),
             (r"$\mathrm{NRMS}_e$ (\%)" if metric == "nrms_e" else "RMS (m)") + " & "
             + " & ".join(AXTEX[a] for _ in groups for a in AX) + r"\\", r"\midrule"]
    for lbl in labels:
        if lbl in ("Black box", "Noise floor", "Final model (OBC), detuned start") and not lines[-1].endswith(r"\midrule"):
            lines.append(r"\midrule")
        any_row = next(val[(lbl, g)] for g in groups if (lbl, g) in val)
        dag = r"$^\dagger$" if any_row["status"].startswith("preliminary") else ""
        cells = []
        for g in groups:
            r = val.get((lbl, g))
            for a in AX:
                if r is None:
                    cells.append("")
                elif metric == "nrms_e":
                    v = nrms_pct(r["nrms_e_%s" % a])
                    cells.append((r"$%.1f$" if v >= 0.1 else r"$%.2f$") % v)    # the floor is about 0.05 %
                else:
                    cells.append(sci(float(r["rms_%s" % a])))
        lines.append(" & ".join([SHORT.get(lbl, lbl) + dag] + cells) + r"\\")
    lines += [r"\bottomrule"]
    for lab in preliminary(r["status"] for r in rows):
        lines.append(r"\multicolumn{%d}{l}{\footnotesize $^\dagger$ %s}\\" % (n + 1, lab))
    lines += [r"\end{tabular}", ""]
    return "\n".join(lines)


def main():
    with open(RESULTS_DIR / f"{STEM}.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for split in dict.fromkeys(r["split"] for r in rows):
        sub = [r for r in rows if r["split"] == split]
        save_table("%% %s records\n" % SPLIT_NAME.get(split, split) + table(sub, "nrms_e"), "%s_%s" % (STEM, split))
        save_table("%% %s records\n" % SPLIT_NAME.get(split, split) + table(sub, "rms"), "%s_%s_m" % (STEM, split))
        if split == MAIN_SPLIT:
            save_table("%% %s records\n" % SPLIT_NAME[split] + table(sub, "nrms_e"), STEM)


if __name__ == "__main__":
    main()
