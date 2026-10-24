"""Appendix: LPV baseline against the LTI baseline frozen at Y = 0, both at the true parameters, on the X
axes (Y scheduling enters only the X rows). Bars: NRMS_e in % (05_setup eq:setup_rms), one panel per axis,
linear axis (bar length on a log axis has no meaning). Positions relative to the training range Y_tr
(confidentiality, Dirk 2026-10-24).

Reads Results/res_lpv_lti.csv (Code/Evaluation/make_results.py), writes Writing/figures/res_lpv_lti.pdf.
"""
import matplotlib.pyplot as plt

from _fig_common import AXTEX, BASE, COL_WIDTH, Y_TR, read_csv, save, use_paper_style

STEM = "res_lpv_lti"
GROUPS = [("Standstill", ["I2", "E1"]), ("Moving", ["I3", "E2"])]
SWEEP = {"I3": 0.20, "E2": 0.39}      # |Y| range of the moving records [m] (MANIFEST.csv description)
LTI_GREY = "#a6a6a6"


def main():
    use_paper_style()
    rows = read_csv(STEM)
    val = {(r["record"], r["model"], a): 100 * float(r["nrms_e_%s" % a]) for r in rows for a in ("X1", "X2")}
    yop = {r["record"]: float(r["Y_op"]) for r in rows}
    ylab = {rec: (r"$Y=%.2f\,Y_{\mathrm{tr}}$" % (yop[rec] / Y_TR)) if rec not in SWEEP
            else (r"$|Y|\leq%.2f\,Y_{\mathrm{tr}}$" % (SWEEP[rec] / Y_TR)) for rec in yop}
    w = 0.36
    fig, axs = plt.subplots(2, 1, figsize=(COL_WIDTH, 2.9), sharex=True, gridspec_kw=dict(hspace=0.15))
    for ax, a in zip(axs, ("X1", "X2")):
        x, ticks, labels, centres = 0.0, [], [], []
        for name, recs in GROUPS:
            x0 = x
            for rec in recs:
                ax.bar(x - w / 2, val[(rec, "LPV", a)], w, color=BASE["color"], edgecolor="none", zorder=2)
                ax.bar(x + w / 2, val[(rec, "LTI", a)], w, color=LTI_GREY, edgecolor="none", zorder=2)
                ticks.append(x)
                labels.append("%s\n%s" % (rec, ylab[rec]))
                x += 1.0
            centres.append(((x0 + x - 1.0) / 2, name))
            x += 0.4
        ax.set_ylabel(r"$\mathrm{NRMS}_{e,%s}$ (%%)" % AXTEX[a])
        ax.grid(True, axis="y", alpha=0.3, zorder=0)
        ax.grid(False, axis="x")
        ax.tick_params(axis="x", length=0, pad=2)
    axs[-1].set_xticks(ticks, labels, fontsize=6.5)
    axs[-1].set_xlim(ticks[0] - 0.6, ticks[-1] + 0.6)
    for xc, name in centres:
        axs[-1].annotate(name, (xc, 0), xycoords=("data", "axes fraction"), xytext=(0, -24),
                         textcoords="offset points", ha="center", va="top", fontsize=7)
    hs = [plt.Rectangle((0, 0), 1, 1, color=BASE["color"], label="LPV baseline"),
          plt.Rectangle((0, 0), 1, 1, color=LTI_GREY, label=r"LTI baseline ($Y=0$)")]
    axs[0].legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                  handlelength=1.4, borderaxespad=0.1)
    fig.align_ylabels(axs)
    save(fig, STEM)


if __name__ == "__main__":
    main()
