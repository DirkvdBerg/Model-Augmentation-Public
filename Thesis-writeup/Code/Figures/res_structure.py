"""Table I (Aspect 2, with the black box of Aspect 4): RMS and NRMS_e per axis, mean over the
validation records (D-269: validation decides the design choice).

Reads Results/res_structure.csv (Code/Evaluation/make_results.py), writes
Writing/tables/res_structure.tex (tabular body for \\input).
"""
import csv

import numpy as np

from _fig_common import RESULTS_DIR, save_table

STEM = "res_structure"


def sci(x):
    e = int(np.floor(np.log10(abs(x))))
    return r"$%.2f\cdot10^{%d}$" % (x / 10 ** e, e)


def main():
    with open(RESULTS_DIR / f"{STEM}.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    short = {"Baseline, true parameters": r"Baseline, true $\theta$",
             "Baseline, detuned start": r"Baseline, detuned $\theta$",
             "Static parallel, $n_a=0$": r"Static parallel, $n_a{=}0$",
             "Plain start, $n_a=2$": r"Plain training, $n_a{=}2$",
             "PS start, $n_a=2$": r"PS init., $n_a{=}2$",
             "PS start, $n_a=2$, physics part only": r"PS init., baseline part only",
             "PS start, $n_a=8$": r"PS init., $n_a{=}8$",
             "PS start, $n_a=2$, detuned start": r"PS init., detuned $\theta$",
             "Final model (OBC), detuned start": r"Augmented (OBC), detuned $\theta$",
             "Black box": "Black-box model"}
    lines = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{l r r r}", r"\toprule",
             r"Model & $X_1$ (m) & $X_2$ (m) & $Y$ (m)\\", r"\midrule"]
    for r in rows:
        if r["label"] in ("Black box", "Noise floor", "Final model (OBC), detuned start"):
            if not lines[-1].endswith(r"\midrule"):
                lines.append(r"\midrule")
        lines.append(" & ".join([short.get(r["label"], r["label"])]
                                + [sci(float(r[f"rms_{a}"])) for a in ("X1", "X2", "Y")]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    save_table("\n".join(lines), STEM)


if __name__ == "__main__":
    main()
