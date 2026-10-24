"""Appendix app:setup: operating-range coverage of the records (Section V-B).

Reads Results/setup_coverage.csv (Code/Setup/coverage.py), writes Writing/figures/setup_coverage.pdf.
Dot plot, one row per record (training, validation, test I, test E): the Y range of its reference
(bar, a dot for a standstill record), its largest acceleration and its largest velocity (max over X and
Y, % of the machine limit). Shaded: the range the training records cover. Every value is plotted where
it is (no offsets; Dirk 2026-10-18). E5 is left out (not in the thesis).
"""
import csv

import matplotlib.pyplot as plt

from _fig_common import COL_WIDTH, RESULTS_DIR, save, use_paper_style

STEM = "setup_coverage"
COLOR = {"training": "#7f7f7f", "validation": "#1f77b4", "test-I": "#2ca02c", "test-E": "#d62728"}
LABEL = {"training": "training", "validation": "validation", "test-I": "test, I", "test-E": "test, E"}
SHADE = "0.90"
SHOWN = {"E6": "E5"}


def main():
    use_paper_style()
    with open(RESULTS_DIR / "setup_coverage.csv", newline="") as fh:
        raw = [r for r in csv.DictReader(fh) if r["record"] != "E5"]
    rows = [{"record": r["record"], "block": r["block"], "y0": float(r["Ymin_m"]), "y1": float(r["Ymax_m"]),
             "acc": max(float(r["aX_pct"]), float(r["aY_pct"])),
             "vel": max(float(r["vX_pct"]), float(r["vY_pct"]))} for r in raw]
    rows.sort(key=lambda r: list(COLOR).index(r["block"]))            # stable: manifest order within a block
    tr = [r for r in rows if r["block"] == "training"]
    y_tr = max(max(abs(r["y0"]), abs(r["y1"])) for r in tr)
    a_tr, v_tr = max(r["acc"] for r in tr), max(r["vel"] for r in tr)

    n = len(rows)
    fig, (ay, aa, av) = plt.subplots(1, 3, figsize=(COL_WIDTH, 0.095 * n + 0.75), sharey=True,
                                     gridspec_kw=dict(width_ratios=[1.5, 1, 1], wspace=0.08))
    ay.axvspan(-y_tr, y_tr, color=SHADE, lw=0, zorder=0)
    aa.axvspan(0, a_tr, color=SHADE, lw=0, zorder=0)
    av.axvspan(0, v_tr, color=SHADE, lw=0, zorder=0)
    for k, r in enumerate(rows):
        c = COLOR[r["block"]]
        if abs(r["y1"] - r["y0"]) < 1e-6:
            ay.plot([r["y0"]], [k], "o", ms=2.2, color=c, zorder=3)
        else:
            ay.plot([r["y0"], r["y1"]], [k, k], color=c, lw=1.6, solid_capstyle="butt", zorder=3)
        aa.plot([r["acc"]], [k], "o", ms=2.2, color=c, zorder=3)
        av.plot([r["vel"]], [k], "o", ms=2.2, color=c, zorder=3)
    # thin rules between the blocks
    for k in range(1, n):
        if rows[k]["block"] != rows[k - 1]["block"]:
            for ax in (ay, aa, av):
                ax.axhline(k - 0.5, color="0.6", lw=0.4, zorder=1)
    # E6 is shown as E5: the changed-controller record E5 is not in the thesis (Dirk 2026-10-24)
    ay.set_yticks(range(n), [SHOWN.get(r["record"], r["record"]) for r in rows], fontsize=5)
    for t, r in zip(ay.get_yticklabels(), rows):
        t.set_color(COLOR[r["block"]])
    ay.set_ylim(n - 0.4, -0.6)
    ay.set_xlim(-0.45, 0.45)
    aa.set_xlim(-5, 140)
    av.set_xlim(-5, 105)
    aa.set_xticks([0, 50, 100])
    av.set_xticks([0, 50, 100])
    fig.subplots_adjust(top=0.97)                       # legend directly above the panels
    ay.set_xlabel(r"$Y$ range (m)")
    aa.set_xlabel("Acceleration (%)")
    av.set_xlabel("Velocity (%)")
    for ax in (ay, aa, av):
        ax.grid(True, axis="x", alpha=0.3)
        ax.grid(False, axis="y")
        ax.tick_params(axis="y", length=0, pad=1.5)
        ax.tick_params(axis="x", labelsize=6)
    handles = [plt.Line2D([], [], color=COLOR[b], lw=1.6, label=LABEL[b]) for b in COLOR] \
        + [plt.Rectangle((0, 0), 1, 1, color=SHADE, label="training range")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, 0.995), ncol=5, frameon=False,
               fontsize=5.5, handlelength=1.2, columnspacing=0.8)
    save(fig, STEM)


if __name__ == "__main__":
    main()
