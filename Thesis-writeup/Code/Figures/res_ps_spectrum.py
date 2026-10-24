"""Appendix: amplitude spectral density of the Y simulation error, baseline vs plain training vs PS, on the
records with multisine of one split, relative to the baseline's peak in the multisine band (no absolute
units, confidentiality, Dirk 2026-10-24). No noise floor: Section V states it through the SNR.

Reads Results/res_ps_spectrum.csv (Code/Evaluation/make_results.py), one spectrum per split. Writes
Writing/figures/res_ps_spectrum.pdf (validation, where the PS vs plain choice is decided, D-269) and
res_ps_spectrum_<split>.pdf for the other splits.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from _fig_common import BASE, COL_WIDTH, MOD, read_csv, save, use_paper_style

STEM = "res_ps_spectrum"
MAIN_SPLIT = "val"
MS_BAND = (106.0, 297.0)  # multisine band, meta.band_Hz (SetupCrossMax to 297 Hz)
PLAIN = "#ff7f0e"


def figure(rows, stem):
    f = np.array([float(r["f_Hz"]) for r in rows])
    col = lambda k: np.array([float(r[k]) for r in rows])        # noqa: E731
    m = (f >= 5) & (f <= 1500)
    band = (f >= MS_BAND[0]) & (f <= MS_BAND[1])
    ref = col("asd_baseline")[band].max()
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 1.9))
    ax.axvspan(*MS_BAND, color="0.93", lw=0, zorder=0)
    ax.loglog(f[m], col("asd_baseline")[m] / ref, label="Baseline", **BASE)
    ax.loglog(f[m], col("asd_plain")[m] / ref, color=PLAIN, lw=0.9, label="Plain training")
    ax.loglog(f[m], col("asd_ps")[m] / ref, label="PS", **MOD)
    ax.set_xlim(f[m][0], f[m][-1])
    ax.set_xticks([10, 30, 100, 300, 1000], ["10", "30", "100", "300", "1000"])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Error ASD (rel.)")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(False, which="minor")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=1.6,
              borderaxespad=0.1)
    save(fig, stem)
    plt.close(fig)


def main():
    use_paper_style()
    allrows = read_csv(STEM)
    for split in dict.fromkeys(r["split"] for r in allrows):
        figure([r for r in allrows if r["split"] == split], STEM if split == MAIN_SPLIT else "%s_%s" % (STEM, split))


if __name__ == "__main__":
    main()
