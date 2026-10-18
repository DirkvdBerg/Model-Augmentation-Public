"""Fig. 4: the prediction on test record I1, Y axis.

Reads Results/res_time_trace.csv (Code/Evaluation/make_results.py), writes
Writing/figures/res_time_trace.pdf.
Top: moving RMS (50 ms) of the prediction error y_hat - y of the baseline and the final model over
the whole record, the zoom window shaded. Bottom: inside the zoom window, the servo error
e = y - r as measured and as predicted by both models. The window is the move with the largest
baseline error, to 50 ms after it stops (chosen from the baseline and the reference only).
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

from _fig_common import COL_WIDTH, COLOR, RESULTS_DIR, mark_preliminary, save, use_paper_style

STEM = "res_time_trace"
FS = 4000.0
WIN = int(0.05 * FS)          # HEURISTIC: 50 ms moving RMS, a display smoothing only
DOT = (0, (0.1, 2.2))       # round dots; the system line shows through the gaps


def moving_rms(e):
    return np.sqrt(np.convolve(e ** 2, np.ones(WIN) / WIN, mode="same"))


def main():
    use_paper_style()
    with open(RESULTS_DIR / f"{STEM}.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    c = lambda k: np.array([float(r[k]) for r in rows])          # noqa: E731
    t, r, y, b, m = c("t_s"), c("r_Y"), c("y_Y"), c("baseline_Y"), c("final_Y")
    z = np.array([int(r_["zoom"]) for r_ in rows], bool)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_WIDTH, 2.8), gridspec_kw=dict(hspace=0.62))
    a1.axvspan(t[z][0], t[z][-1], color="0.80", lw=0, zorder=0)
    a1.semilogy(t, moving_rms(b - y), color=COLOR["baseline"], lw=0.8)
    a1.semilogy(t, moving_rms(m - y), color=COLOR["ps"], lw=0.8)
    a1.set_xlim(t[0], t[-1])
    a1.set_xlabel("Time (s)")
    a1.set_ylabel("Sim. error RMS (m)")
    a1.grid(False, which="minor")

    tz = (t[z] - t[z][0]) * 1e3
    a2.plot(tz, (y - r)[z], color="k", lw=1.0, zorder=2)
    a2.plot(tz, (b - r)[z], color=COLOR["baseline"], lw=0.7, zorder=1)
    a2.plot(tz, (m - r)[z], color=COLOR["ps"], lw=1.3, ls=DOT, dash_capstyle="round", zorder=3)
    a2.set_xlim(tz[0], tz[-1])
    a2.set_xlabel("Time in shaded window (ms)")
    a2.set_ylabel("Servo error (m)")
    a2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    a2.yaxis.get_offset_text().set_fontsize(7)

    handles = [plt.Line2D([], [], color="k", lw=1.0, label="System"),
               plt.Line2D([], [], color=COLOR["baseline"], lw=0.7, label="Baseline model"),
               plt.Line2D([], [], color=COLOR["ps"], lw=1.3, ls=DOT, dash_capstyle="round", label="Augmented model")]
    a1.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              handlelength=1.8, borderaxespad=0.1)
    fig.align_ylabels((a1, a2))
    mark_preliminary(fig, [r_.get("status_final", "") for r_ in rows[:1]], "augmented model (R1)")
    save(fig, STEM)


if __name__ == "__main__":
    main()
