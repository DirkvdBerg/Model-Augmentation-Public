"""Appendix app:setup: operating-range coverage of the records (Section V-B).

Reads Results/setup_coverage.csv (Code/Setup/coverage.py), writes Writing/figures/setup_coverage.pdf.
One horizontal segment (a dot for a standstill record) per record: the Y range of its reference against
its largest acceleration (top) and its largest velocity (bottom), max over X and Y, % of the machine
limit. Dashed box: the training range. Test records at the same level are offset by OFFSET for
legibility. E5 is left out (not in the thesis).
"""
import csv
from collections import defaultdict

import matplotlib.pyplot as plt

from _fig_common import COL_WIDTH, RESULTS_DIR, save, use_paper_style

STEM = "setup_coverage"
COLOR = {"training": "#7f7f7f", "validation": "#1f77b4", "test-I": "#2ca02c", "test-E": "#d62728"}
LABEL = {"training": "training", "validation": "validation", "test-I": "test, I", "test-E": "test, E"}
OFFSET = 4.0                            # HEURISTIC: vertical offset [%] between test records at one level


def panel(ax, rows, key, lim):
    tr = [r for r in rows if r["block"] == "training"]
    y_tr = max(max(abs(r["y0"]), abs(r["y1"])) for r in tr)
    ax.add_patch(plt.Rectangle((-y_tr, 0), 2 * y_tr, max(r[key] for r in tr), fill=False, ls="--", lw=0.8, color="k"))
    level = defaultdict(int)
    for r in rows:
        v = r[key]
        if r["block"].startswith("test"):
            n = level[round(v)]
            level[round(v)] += 1
            v += n * OFFSET
        c = COLOR[r["block"]]
        if abs(r["y1"] - r["y0"]) < 1e-6:
            ax.plot([r["y0"]], [v], "o", ms=2.5, color=c)
        else:
            ax.plot([r["y0"], r["y1"]], [v, v], color=c, lw=1.2, solid_capstyle="butt")
        if r["block"].startswith("test"):
            x = r["y1"] if abs(r["y1"] - r["y0"]) > 1e-6 else r["y0"]
            ax.text(x + 0.012, v, r["record"], fontsize=5.5, color=c, va="center")
    ax.set_xlim(-0.45, 0.5)
    ax.set_ylim(-6, lim)
    ax.set_ylabel(key.split("_")[0] + " (%)")


def main():
    use_paper_style()
    with open(RESULTS_DIR / "setup_coverage.csv", newline="") as fh:
        raw = [r for r in csv.DictReader(fh) if r["record"] != "E5"]
    rows = [{"record": r["record"], "block": r["block"], "y0": float(r["Ymin_m"]), "y1": float(r["Ymax_m"]),
             "Acceleration_pct": max(float(r["aX_pct"]), float(r["aY_pct"])),
             "Velocity_pct": max(float(r["vX_pct"]), float(r["vY_pct"]))} for r in raw]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_WIDTH, 3.4), sharex=True)
    panel(ax1, rows, "Acceleration_pct", 145)
    panel(ax2, rows, "Velocity_pct", 115)
    ax2.set_xlabel(r"$Y$ (m)")
    for b in COLOR:
        ax1.plot([], [], color=COLOR[b], lw=1.2, label=LABEL[b])
    fig.legend(loc="upper center", ncol=4, fontsize=6, bbox_to_anchor=(0.55, 0.995), frameon=False)
    fig.align_ylabels((ax1, ax2))
    save(fig, STEM)


if __name__ == "__main__":
    main()
