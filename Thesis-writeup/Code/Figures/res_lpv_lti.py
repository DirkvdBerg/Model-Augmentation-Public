"""Fig. 1 (Aspect 1): LPV baseline against the LTI baseline frozen at Y = 0, both at the true
parameters, on the X axes (the Y axis is identical for both: Y scheduling enters only the X rows).

Reads Results/res_lpv_lti.csv (Code/Evaluation/make_results.py), writes
Writing/figures/res_lpv_lti.pdf. Bars: simulation error RMS averaged over X1 and X2. The noise
floor (about 1e-8 m, in the CSV) is two decades below every bar, so the axis is limited to the
bars and the floor is stated in the caption.
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

from _fig_common import COL_WIDTH, COLOR, RESULTS_DIR, save, use_paper_style

STEM = "res_lpv_lti"
GROUPS = [("Standstill", ["I2", "E1"]), ("Moving", ["I3", "E2"])]
# Y position or range of each record (MANIFEST.csv Y_op and description)
YLAB = {"I2": r"$Y=-0.225$ m", "E1": r"$Y=-0.39$ m", "I3": r"$|Y|\leq0.20$ m",
        "E2": r"$|Y|\leq0.39$ m"}


def main():
    use_paper_style()
    with open(RESULTS_DIR / f"{STEM}.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    val = {(r["record"], r["model"]): np.mean([float(r["rms_X1"]), float(r["rms_X2"])]) for r in rows}

    w = 0.36
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 1.8))
    x, ticks, labels, centres = 0.0, [], [], []
    for name, recs in GROUPS:
        x0 = x
        for rec in recs:
            ax.bar(x - w / 2, val[(rec, "LPV")], w, color=COLOR["baseline"], edgecolor="none", zorder=2)
            ax.bar(x + w / 2, val[(rec, "LTI")], w, color=COLOR["lti"], edgecolor="none", zorder=2)
            ticks.append(x)
            labels.append("%s\n%s" % (rec, YLAB[rec]))
            x += 1.0
        centres.append(((x0 + x - 1.0) / 2, name))
        x += 0.4
    ax.set_yscale("log")
    ax.set_ylim(5e-7, 1.2e-5)
    ax.set_xticks(ticks, labels, fontsize=6.5)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(ticks[0] - 0.6, ticks[-1] + 0.6)
    ax.set_ylabel(r"Sim. error RMS, $X$ (m)")
    ax.grid(True, which="major", axis="y", alpha=0.3, zorder=0)
    ax.grid(False, axis="x")
    for xc, name in centres:
        ax.annotate(name, (xc, 0), xycoords=("data", "axes fraction"), xytext=(0, -24),
                    textcoords="offset points", ha="center", va="top", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR["baseline"], label="LPV baseline"),
               plt.Rectangle((0, 0), 1, 1, color=COLOR["lti"], label=r"LTI baseline ($Y=0$)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              handlelength=1.4, borderaxespad=0.1)
    save(fig, STEM)


if __name__ == "__main__":
    main()
