"""Appendix: full 3 x 3 open-loop frequency response of the final model, the true system and the baseline.

Reads Results/res_frf_full.csv (Code/Evaluation/eval_frf.py), writes Writing/figures/res_frf_full.pdf.
Bode magnitude (dB re 1 m/N); row j = output (stage position), column l = input (stage force), linearised at rest,
geometric mean over the seven positions Y = -0.39 to 0.39 m; shaded: range of the final model over Y.
"""
import matplotlib.pyplot as plt
import numpy as np

from _fig_common import COLOR, FULL_WIDTH, save, use_paper_style
from res_frf import NAMES, db, freq_axis, gmean, load

STEM = "res_frf_full"
AXES = ("X1", "X2", "Y")
LAB = {"X1": "X_1", "X2": "X_2", "Y": "Y"}


def main():
    use_paper_style()
    cols = ["%s_%s_%s" % (m, a, b) for m in ("model", "truth", "base") for a in AXES for b in AXES]
    f, Ys, G = load(STEM, cols)
    fig, axs = plt.subplots(3, 3, figsize=(FULL_WIDTH, 4.6), sharex=True)
    for j, a in enumerate(AXES):
        for l, b in enumerate(AXES):
            ax = axs[j, l]
            m, t, s = G["model_%s_%s" % (a, b)], G["truth_%s_%s" % (a, b)], G["base_%s_%s" % (a, b)]
            ax.fill_between(f, db(m.min(0)), db(m.max(0)), color=COLOR["ps"], alpha=0.2, lw=0)
            ax.semilogx(f, db(gmean(s)), color=COLOR["baseline"], lw=0.9)
            ax.semilogx(f, db(gmean(t)), color="k", lw=1.1)
            ax.semilogx(f, db(gmean(m)), color=COLOR["ps"], lw=1.2, ls="--")
            ax.set_xlim(f[0], f[-1])
            freq_axis(ax)
            ax.grid(True, which="both", alpha=0.3)
            if j == 0:
                ax.set_title(r"Input $F_{%s}$" % LAB[b], fontsize=8, pad=3)
            if j == 2:
                ax.set_xlabel("Frequency (Hz)")
            if l == 0:
                ax.set_ylabel("Output $%s$\nMagnitude (dB)" % LAB[a])
    handles = [plt.Line2D([], [], color=COLOR["baseline"], lw=0.9, label=NAMES["base"]),
               plt.Line2D([], [], color="k", lw=1.1, label=NAMES["truth"]),
               plt.Line2D([], [], color=COLOR["ps"], lw=1.2, ls="--", label=NAMES["model"])]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, STEM)


if __name__ == "__main__":
    main()
