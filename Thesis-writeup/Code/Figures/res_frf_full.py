"""Appendix: full 3 x 3 frequency response of the final model, the system and the baseline, linearised at
rest. Row = output (stage position), column = input (stage force). Lines at Y = 0, in dB relative to the
system F_Y -> Y at the lowest frequency (one reference for all nine entries, no absolute plant gain);
bands: range over the seven positions Y = -0.39 to 0.39 m of the system (grey) and the final model (blue).

Reads Results/res_frf_full.csv (Code/Evaluation/eval_frf.py), writes Writing/figures/res_frf_full.pdf.
"""
import matplotlib.pyplot as plt
import numpy as np

from _fig_common import AXES, AXTEX, BASE, FULL_WIDTH, MOD, NAME, SYS, save, use_paper_style
from res_frf import freq_axis, load

STEM = "res_frf_full"


def main():
    use_paper_style()
    cols = ["%s_%s_%s" % (m, a, b) for m in ("model", "truth", "base") for a in AXES for b in AXES]
    f, Ys, G = load(STEM, cols)
    i0 = Ys.index(0.0)
    ref = G["truth_Y_Y"][i0, 0]
    db = lambda a: 20 * np.log10(a / ref)                         # noqa: E731
    fig, axs = plt.subplots(3, 3, figsize=(FULL_WIDTH, 4.6), sharex=True)
    for j, a in enumerate(AXES):
        for l, b in enumerate(AXES):
            ax = axs[j, l]
            m, t, s = G["model_%s_%s" % (a, b)], G["truth_%s_%s" % (a, b)], G["base_%s_%s" % (a, b)]
            ax.fill_between(f, db(t.min(0)), db(t.max(0)), color=SYS["color"], alpha=0.6, lw=0)
            ax.fill_between(f, db(m.min(0)), db(m.max(0)), color=MOD["color"], alpha=0.18, lw=0)
            ax.semilogx(f, db(t[i0]), **SYS)
            ax.semilogx(f, db(s[i0]), **BASE)
            ax.semilogx(f, db(m[i0]), **MOD)
            ax.set_xlim(f[0], f[-1])
            freq_axis(ax)
            ax.grid(True, which="both", alpha=0.3)
            if j == 0:
                ax.set_title(r"Input $F_{%s}$" % AXTEX[b], fontsize=8, pad=3)
            if j == 2:
                ax.set_xlabel("Frequency (Hz)")
            if l == 0:
                ax.set_ylabel("Output $%s$\nMagnitude (dB, rel.)" % AXTEX[a])
    hs = [plt.Line2D([], [], label=NAME[k], **s) for k, s in (("sys", SYS), ("base", BASE), ("model", MOD))]
    fig.legend(handles=hs, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02), handlelength=2.2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, STEM)


if __name__ == "__main__":
    main()
