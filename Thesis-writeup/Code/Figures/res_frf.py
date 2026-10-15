"""Result B: open-loop frequency response Y -> Y of the final model, the true system and the baseline.

Reads Results/res_frf.csv (Code/Evaluation/eval_frf.py), writes Writing/figures/res_frf.pdf.
Bode magnitude (dB re 1 m/N) from the stage force F_Y to the stage position Y, linearised at rest, averaged over the seven
positions Y = -0.39 to 0.39 m (training range and extrapolation; the model is LPV): geometric mean of
|G| over Y per frequency. Shaded: range of the final model over Y. The full 3 x 3 response is
res_frf_full (appendix).
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

from _fig_common import COL_WIDTH, COLOR, RESULTS_DIR, save, use_paper_style

STEM = "res_frf"


def load(stem, cols):
    with open(RESULTS_DIR / f"{stem}.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    Ys = sorted({float(r["Y"]) for r in rows})
    f = np.array([float(r["f_Hz"]) for r in rows if float(r["Y"]) == Ys[0]])
    G = {c: np.array([[float(r[c]) for r in rows if float(r["Y"]) == Y] for Y in Ys]) for c in cols}
    return f, Ys, G


def gmean(a):
    # HEURISTIC: geometric mean over Y, so the average sits on the log axis the figure uses
    return np.exp(np.mean(np.log(a), axis=0))


def db(a):
    # THEORY: Bode magnitude in dB, 20 log10(|G| / 1 m/N)
    return 20 * np.log10(a)


TICKS = [20, 50, 100, 200, 500]


def freq_axis(ax):
    ax.set_xticks(TICKS, [str(v) for v in TICKS])



NAMES = {"base": r"Baseline $\mathcal{M}_{\mathrm{LPV}}$", "truth": "System",
         "model": r"Final model $\mathcal{M}_{\mathrm{OBC}}$"}


def main():
    use_paper_style()
    f, Ys, G = load(STEM, ["model_Y_Y", "truth_Y_Y", "base_Y_Y"])
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.0))
    ax.fill_between(f, db(G["model_Y_Y"].min(0)), db(G["model_Y_Y"].max(0)), color=COLOR["ps"], alpha=0.2, lw=0)
    ax.semilogx(f, db(gmean(G["base_Y_Y"])), color=COLOR["baseline"], lw=1.0, label=NAMES["base"])
    ax.semilogx(f, db(gmean(G["truth_Y_Y"])), color="k", lw=1.2, label=NAMES["truth"])
    ax.semilogx(f, db(gmean(G["model_Y_Y"])), color=COLOR["ps"], lw=1.3, ls="--", label=NAMES["model"])
    ax.set_xlim(f[0], f[-1])
    ax.set_xlabel("Frequency (Hz)")
    freq_axis(ax)
    ax.set_ylabel("Magnitude (dB re 1 m/N)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=1.8,
              borderaxespad=0.1)
    save(fig, STEM)


if __name__ == "__main__":
    main()
