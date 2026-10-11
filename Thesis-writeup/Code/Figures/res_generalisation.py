"""Fig. (Aspect 4): baseline at the detuned start, augmented model (OBC) and black-box model on
the interpolation (I) and extrapolation (E) test records, in two panels.

Reads Results/res_generalisation.csv (Code/Evaluation/make_results.py), writes
Writing/figures/res_generalisation.pdf. Baseline = the detuned start of the final model. Bars:
simulation error RMS averaged over X1, X2, Y; red marks: the noise floor of the same record.
E records are ordered by the property in which they differ from their I partner (Y position,
Y moving, acceleration, velocity, ILC move). E5 (changed controller) is not shown (user 2026-10-11).
"""
import csv

import matplotlib.pyplot as plt

from _fig_common import FULL_WIDTH, COLOR, FLOOR, RESULTS_DIR, save, use_paper_style

STEM = "res_generalisation"
PANELS = [("Interpolation", ["I1", "I2", "I3", "I4", "I5", "I6"]),
          ("Extrapolation", ["E1", "E2", "E3", "E4", "E6"])]
MODELS = [("baseline", r"Baseline model, detuned $\theta$"), ("final", "Augmented model (OBC)"),
          ("bb", "Black-box model")]


def color(m):
    return COLOR["ps" if m == "final" else m]


def main():
    use_paper_style()
    with open(RESULTS_DIR / f"{STEM}.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    val = {(r["record"], r["model"]): float(r["rms_mean"]) for r in rows}

    w = 0.27
    fig, axs = plt.subplots(1, 2, figsize=(FULL_WIDTH, 1.85), sharey=True,
                            gridspec_kw=dict(width_ratios=[6, 5], wspace=0.06))
    for ax, (title, recs) in zip(axs, PANELS):
        for x, rec in enumerate(recs):
            for k, (m, _) in enumerate(MODELS):
                ax.bar(x + (k - 1) * w, val[(rec, m)], w, color=color(m), edgecolor="none", zorder=2)
            ax.plot([x - 1.5 * w, x + 1.5 * w], [val[(rec, "floor")]] * 2, zorder=3, **FLOOR)
        ax.set_yscale("log")
        ax.set_xticks(range(len(recs)), recs)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_xlim(-0.6, len(recs) - 0.4)
        ax.set_title(title, fontsize=8, pad=3)
        ax.grid(True, which="major", axis="y", alpha=0.3, zorder=0)
        ax.grid(False, axis="x")
    axs[0].set_ylabel("Simulation error RMS (m)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color(m), label=lbl) for m, lbl in MODELS] \
        + [plt.Line2D([], [], label="Noise floor", **FLOOR)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False,
               handlelength=1.4, columnspacing=1.2)
    save(fig, STEM)


if __name__ == "__main__":
    main()
