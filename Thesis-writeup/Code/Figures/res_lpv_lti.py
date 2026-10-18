"""Fig. 1 (Aspect 1): LPV baseline against the LTI baseline frozen at Y = 0, both at the true
parameters, on the X axes (the Y axis is identical for both: Y scheduling enters only the X rows).

Reads Results/res_lpv_lti.csv (Code/Evaluation/make_results.py), writes
Writing/figures/res_lpv_lti.pdf. Bars: NRMS_e in % (the primary metric, 05_setup eq:setup_rms),
one panel per axis X1, X2. The noise floor (NRMS_e columns floor_nrms_e_* in the CSV) is far below
every bar, so the axis is limited to the bars and the floor is stated in the caption.
"""
import csv

import matplotlib.pyplot as plt

from _fig_common import COL_WIDTH, COLOR, RESULTS_DIR, nrms_pct, save, use_paper_style

STEM = "res_lpv_lti"
GROUPS = [("Standstill", ["I2", "E1"]), ("Moving", ["I3", "E2"])]
# Y position or range of each record (MANIFEST.csv Y_op and description)
YLAB = {"I2": r"$Y=-0.225$ m", "E1": r"$Y=-0.39$ m", "I3": r"$|Y|\leq0.20$ m",
        "E2": r"$|Y|\leq0.39$ m"}
AX = (("X1", "X_1"), ("X2", "X_2"))


def main():
    use_paper_style()
    with open(RESULTS_DIR / f"{STEM}.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    val = {(r["record"], r["model"], a): nrms_pct(r["nrms_e_%s" % a]) for r in rows for a, _ in AX}

    w = 0.36
    fig, axs = plt.subplots(len(AX), 1, figsize=(COL_WIDTH, 2.9), sharex=True, gridspec_kw=dict(hspace=0.15))
    for ax, (a, tex) in zip(axs, AX):
        x, ticks, labels, centres = 0.0, [], [], []
        for name, recs in GROUPS:
            x0 = x
            for rec in recs:
                ax.bar(x - w / 2, val[(rec, "LPV", a)], w, color=COLOR["baseline"], edgecolor="none", zorder=2)
                ax.bar(x + w / 2, val[(rec, "LTI", a)], w, color=COLOR["lti"], edgecolor="none", zorder=2)
                ticks.append(x)
                labels.append("%s\n%s" % (rec, YLAB[rec]))
                x += 1.0
            centres.append(((x0 + x - 1.0) / 2, name))
            x += 0.4
        ax.set_yscale("log")
        ax.set_ylabel(r"$\mathrm{NRMS}_{e,%s}$ (%%)" % tex)
        ax.grid(True, which="major", axis="y", alpha=0.3, zorder=0)
        ax.grid(False, axis="x")
        ax.tick_params(axis="x", length=0, pad=2)
    axs[-1].set_xticks(ticks, labels, fontsize=6.5)
    axs[-1].set_xlim(ticks[0] - 0.6, ticks[-1] + 0.6)
    for xc, name in centres:
        axs[-1].annotate(name, (xc, 0), xycoords=("data", "axes fraction"), xytext=(0, -24),
                         textcoords="offset points", ha="center", va="top", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR["baseline"], label="LPV baseline"),
               plt.Rectangle((0, 0), 1, 1, color=COLOR["lti"], label=r"LTI baseline ($Y=0$)")]
    axs[0].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                  handlelength=1.4, borderaxespad=0.1)
    fig.align_ylabels(axs)
    save(fig, STEM)


if __name__ == "__main__":
    main()
