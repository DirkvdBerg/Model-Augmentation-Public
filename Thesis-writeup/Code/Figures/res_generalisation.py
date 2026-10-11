"""Step 6 (THESIS-RESULTS): baseline, final model and black box on the interpolation (I) and
extrapolation (E) test records, grouped by the axis in which each E differs from its I partner.

Reads Results/res_generalisation.csv (Code/Evaluation/make_results.py), writes
Writing/figures/res_generalisation.pdf. Baseline = the detuned start of the final model. Bars: error RMS
averaged over X1, X2, Y; red marks: the
noise floor of the same record. E5 (changed controller) is not shown (user 2026-10-11).
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

from _fig_common import FULL_WIDTH, COLOR, FLOOR, RESULTS_DIR, save, use_paper_style

STEM = "res_generalisation"
PAIRS = [("Y position", "I2", "E1"), ("Y position, moving", "I3", "E2"), ("Acceleration", "I5", "E3"),
         ("Velocity", "I6", "E4"), ("ILC move", "I3", "E6"), ("Cross-coupling", "I4", None)]
MODELS = [("baseline", r"Baseline model, detuned $	heta$"), ("final", "Augmented model (OBC)"), ("bb", "Black-box model")]


def main():
    use_paper_style()
    with open(RESULTS_DIR / f"{STEM}.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    val = {(r["record"], r["model"]): float(r["rms_mean"]) for r in rows}

    w = 0.26                                   # bar width; a record cluster is 3 bars wide
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 1.9))
    x, ticks, labels, groups = 0.0, [], [], []
    for name, i_rec, e_rec in PAIRS:
        start = x
        for rec in (i_rec, e_rec):
            if rec is None:
                continue
            for k, (m, _) in enumerate(MODELS):
                ax.bar(x + (k - 1) * w, val[(rec, m)], w, color=COLOR["ps" if m == "final" else m],
                       edgecolor="none", zorder=2)
            ax.plot([x - 1.5 * w, x + 1.5 * w], [val[(rec, "floor")]] * 2, zorder=3, **FLOOR)
            ticks.append(x)
            labels.append(rec)
            x += 1.0
        groups.append(((start + x - 1.0) / 2, name))
        x += 0.45
    ax.set_yscale("log")
    ax.set_xticks(ticks, labels)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(ticks[0] - 0.6, ticks[-1] + 0.6)
    for xc, name in groups:
        ax.annotate(name, (xc, 0), xycoords=("data", "axes fraction"), xytext=(0, -17),
                    textcoords="offset points", ha="center", va="top", fontsize=7)
    ax.set_ylabel("Simulation error RMS (m)")
    ax.grid(True, which="major", axis="y", alpha=0.3, zorder=0)
    ax.grid(False, axis="x")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR["ps" if m == "final" else m], label=lbl)
               for m, lbl in MODELS] + [plt.Line2D([], [], label="Noise floor", **FLOOR)]
    ax.legend(handles=handles, loc="upper left", ncol=4, frameon=False, handlelength=1.4,
              columnspacing=1.2, bbox_to_anchor=(0, 1.16))
    save(fig, STEM)


if __name__ == "__main__":
    main()
