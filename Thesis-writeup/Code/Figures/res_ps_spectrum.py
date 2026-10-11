"""Step 3 (THESIS-RESULTS): amplitude spectrum of the Y prediction error, baseline vs plain vs PS.

Reads Results/res_ps_spectrum.csv (Code/Evaluation/make_results.py), writes
Writing/figures/res_ps_spectrum.pdf.
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

from _fig_common import COL_WIDTH, COLOR, FLOOR, RESULTS_DIR, save, use_paper_style

STEM = "res_ps_spectrum"
F_POLE = 212.13          # truth absorber pole, meta.truth.f_pole of every record
MS_BAND = (106.0, 297.0)  # multisine band of I1 and I2, meta.band_Hz


def main():
    use_paper_style()
    with open(RESULTS_DIR / f"{STEM}.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    f = np.array([float(r["f_Hz"]) for r in rows])
    col = lambda k: np.array([float(r[k]) for r in rows])        # noqa: E731
    m = (f >= 5) & (f <= 1500)

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 1.9))
    ax.axvspan(*MS_BAND, color="0.93", lw=0, zorder=0)
    ax.axvline(F_POLE, color="0.45", lw=0.7, ls=":", zorder=1)
    ax.loglog(f[m], col("asd_baseline")[m], color=COLOR["baseline"], label="Baseline model")
    ax.loglog(f[m], col("asd_plain")[m], color=COLOR["plain"], label="Plain training")
    ax.loglog(f[m], col("asd_ps")[m], color=COLOR["ps"], label="PS initialisation")
    ax.loglog(f[m], col("asd_floor")[m], label="Noise floor", **FLOOR)
    ax.set_xlim(f[m][0], f[m][-1])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"Simulation error ASD (m/$\sqrt{\mathrm{Hz}}$)")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(False, which="minor")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, handlelength=1.4,
              columnspacing=0.9, handletextpad=0.4, borderaxespad=0.1)
    save(fig, STEM)


if __name__ == "__main__":
    main()
