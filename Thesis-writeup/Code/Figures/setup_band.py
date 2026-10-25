"""Appendix app:setup: the multisine band over the open-loop frequency response (Section VI-B).

Reads Results/setup_band.csv (Code/Setup/band_frf.py), writes Writing/figures/setup_band.pdf.
One panel (Dirk 2026-10-25, in the style of Fig. res_frf): open-loop |G_YY| from F_Y to Y in dB re 1 m/N,
linearised (absorber on, friction off) at the five training positions (lines overlap), of the simulated
system (thick light line) and the rigid baseline (thin dark line). Shaded: the multisine band 106 to 297 Hz.
No hatching, markers or labels.
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

from _fig_common import BASE, COL_WIDTH, NAME, RESULTS_DIR, SYS, save, use_paper_style

STEM = "setup_band"
BAND = (106.0, 297.0)                   # tdg_config.m multisine band (D-219)
TICKS = [20, 50, 100, 200, 500, 1000]


def rows(stem):
    with open(RESULTS_DIR / f"{stem}.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def db(a):
    # THEORY: Bode magnitude in dB, 20 log10 |G| (G in m/N)
    return 20 * np.log10(a)


def main():
    use_paper_style()
    resp = rows("setup_band")
    Ys = sorted({r["Y"] for r in resp})
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.1))
    ax.axvspan(*BAND, color="#1f77b4", alpha=0.10, lw=0)
    for Y in Ys:
        R = [r for r in resp if r["Y"] == Y]
        f = np.array([float(r["f_Hz"]) for r in R])
        ax.semilogx(f, db([float(r["G_true"]) for r in R]), zorder=1, **SYS)
        ax.semilogx(f, db([float(r["G_base"]) for r in R]), zorder=2, **BASE)
    ax.set_xlim(20, 1000)
    ax.set_xticks(TICKS, [str(v) for v in TICKS])
    ax.minorticks_off()
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB re 1 m/N)")
    ax.grid(True, which="major", alpha=0.3)
    hs = [plt.Line2D([], [], label=NAME[k], **s) for k, s in (("sys", SYS), ("base", BASE))]
    ax.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=2.2,
              borderaxespad=0.1)
    save(fig, STEM)


if __name__ == "__main__":
    main()
