"""Figure (Aspect 1): Bode magnitude from the stage force F_Y to the stage position Y, linearised at rest,
at Y = 0, of the system, the baseline and the final model, in dB relative to the system at the lowest
frequency (no absolute plant gain, confidentiality). Shaded: range of the final model over the seven
positions Y = -0.39 to 0.39 m. The system and the baseline vary by less than 0.3 dB over Y in this
entry, so they get no band (Dirk 2026-10-24). The full 3 x 3 response is res_frf_full (appendix).

Reads Results/res_frf.csv (Code/Evaluation/eval_frf.py), writes Writing/figures/res_frf.pdf.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from _fig_common import BASE, COL_WIDTH, MOD, NAME, SYS, read_csv, save, use_paper_style

STEM = "res_frf"
TICKS = [20, 50, 100, 200, 500]


def load(stem, cols):
    rows = read_csv(stem)
    Ys = sorted({float(r["Y"]) for r in rows})
    f = np.array([float(r["f_Hz"]) for r in rows if float(r["Y"]) == Ys[0]])
    G = {c: np.array([[float(r[c]) for r in rows if float(r["Y"]) == Y] for Y in Ys]) for c in cols}
    return f, Ys, G


def freq_axis(ax):
    ax.set_xticks(TICKS, [str(v) for v in TICKS])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def main():
    use_paper_style()
    f, Ys, G = load(STEM, ["model_Y_Y", "truth_Y_Y", "base_Y_Y"])
    i0 = Ys.index(0.0)
    ref = G["truth_Y_Y"][i0, 0]
    db = lambda a: 20 * np.log10(a / ref)                         # noqa: E731  THEORY: Bode magnitude in dB
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.1))
    ax.fill_between(f, db(G["model_Y_Y"].min(0)), db(G["model_Y_Y"].max(0)), color=MOD["color"], alpha=0.18, lw=0)
    ax.semilogx(f, db(G["truth_Y_Y"][i0]), **SYS)
    ax.semilogx(f, db(G["base_Y_Y"][i0]), **BASE)
    ax.semilogx(f, db(G["model_Y_Y"][i0]), **MOD)
    ax.set_xlim(f[0], f[-1])
    freq_axis(ax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB, rel.)")
    ax.grid(True, which="both", alpha=0.3)
    hs = [plt.Line2D([], [], label=NAME[k], **s) for k, s in (("sys", SYS), ("base", BASE), ("model", MOD))]
    ax.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=2.2,
              borderaxespad=0.1)
    save(fig, STEM)


if __name__ == "__main__":
    main()
