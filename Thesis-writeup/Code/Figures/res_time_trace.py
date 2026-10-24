"""Figure (Aspect 1): the prediction on test record I1, Y axis, normalised by sigma_e,Y of the record
(05_setup eq:setup_rms; no metres, confidentiality, Dirk 2026-10-24).

Reads Results/res_time_trace.csv (Code/Evaluation/make_results.py), writes
Writing/figures/res_time_trace.pdf.
Top: moving NRMS_e,Y (50 ms) of the baseline and the final model over the whole record. Bottom: the
servo error over sigma_e,Y in the zoom window, in the record's own time: the system as a thick light
band, both models as thin lines on top (Dirk 2026-10-24). The window is the move with the largest
baseline error, to 50 ms after it stops (chosen from the baseline and the reference only).
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from _fig_common import BASE, COL_WIDTH, MOD, NAME, SYS, mark_preliminary, read_csv, save, use_paper_style

STEM = "res_time_trace"
FS = 4000.0
WIN = int(0.05 * FS)          # HEURISTIC: 50 ms moving RMS, a display smoothing only


def moving_rms(e):
    return np.sqrt(np.convolve(e ** 2, np.ones(WIN) / WIN, mode="same"))


def main():
    use_paper_style()
    rows = read_csv(STEM)
    c = lambda k: np.array([float(r[k]) for r in rows])          # noqa: E731
    t, r, y, b, m = c("t_s"), c("r_Y"), c("y_Y"), c("baseline_Y"), c("final_Y")
    z = np.array([int(q["zoom"]) for q in rows], bool)
    sig = np.std(y - r)                                          # population sd over the scored samples

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_WIDTH, 2.8), gridspec_kw=dict(hspace=0.62))
    a1.semilogy(t, 100 * moving_rms(b - y) / sig, **BASE)
    a1.semilogy(t, 100 * moving_rms(m - y) / sig, **MOD)
    a1.set_xlim(t[0], t[-1])
    a1.set_xlabel("Time (s)")
    a1.set_ylabel(r"Sim. error (% of $\sigma_e$)")
    a1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: "%g" % v))
    a1.grid(False, which="minor")

    a2.plot(t[z], (100 * (y - r) / sig)[z], zorder=1, **SYS)
    a2.plot(t[z], (100 * (b - r) / sig)[z], zorder=2, **BASE)
    a2.plot(t[z], (100 * (m - r) / sig)[z], zorder=3, **MOD)
    a2.set_xlim(t[z][0], t[z][-1])
    a2.set_xlabel("Time (s)")
    a2.set_ylabel(r"Servo error (% of $\sigma_e$)")

    hs = [plt.Line2D([], [], label=NAME[k], **s) for k, s in (("sys", SYS), ("base", BASE), ("model", MOD))]
    a1.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=2.2,
              borderaxespad=0.1)
    fig.align_ylabels((a1, a2))
    mark_preliminary(fig, [r_.get("status_final", "") for r_ in rows[:1]], "augmented model (R1)")
    save(fig, STEM)
    print("zoom window %.2f to %.2f s" % (t[z][0], t[z][-1]))


if __name__ == "__main__":
    main()
