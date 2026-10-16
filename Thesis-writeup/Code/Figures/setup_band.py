"""Appendix app:setup: the multisine band over the simulated frequency response (Section V-B).

Reads Results/setup_band.csv and Results/setup_band_features.csv (Code/Setup/band_frf.py) and
Results/setup_band_edge.csv (Code/Setup/band_edge.py), writes Writing/figures/setup_band.pdf.
Stage axis Y, linearised (absorber on, friction off), at the five training positions (lines overlap):
top, open-loop |G_YY| of the simulated system and of the rigid baseline; bottom, complementary
sensitivity |T_YY| with K1. Shaded: the multisine band 106 to 297 Hz; hatched: the range of the K1
crossovers over axes and positions. Markers: anti-resonance, coupled mode, closed-loop peak.
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

from _fig_common import COL_WIDTH, COLOR, RESULTS_DIR, save, use_paper_style

STEM = "setup_band"
BAND = (106.0, 297.0)                   # tdg_config.m multisine band (D-219)


def rows(stem):
    with open(RESULTS_DIR / f"{stem}.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def db(a):
    # THEORY: Bode magnitude in dB, 20 log10 |G|
    return 20 * np.log10(a)


def main():
    use_paper_style()
    resp = rows("setup_band")
    feat = rows("setup_band_features")
    cross = [float(r["Hz"]) for r in rows("setup_band_edge") if r["quantity"] == "crossover"]
    Ys = sorted({r["Y"] for r in resp})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_WIDTH, 2.9), sharex=True)
    for ax in (ax1, ax2):
        ax.axvspan(*BAND, color="#1f77b4", alpha=0.10, lw=0)
        ax.axvspan(min(cross), max(cross), facecolor="none", edgecolor="#7f7f7f", hatch="////", lw=0, alpha=0.5)
    for k, Y in enumerate(Ys):
        R = [r for r in resp if r["Y"] == Y]
        f = np.array([float(r["f_Hz"]) for r in R])
        lab = (k == 0)
        ax1.semilogx(f, db([float(r["G_true"]) for r in R]), color=COLOR["ps"], label="system" if lab else None)
        ax1.semilogx(f, db([float(r["G_base"]) for r in R]), color=COLOR["baseline"], ls="--",
                     label="baseline" if lab else None)
        ax2.semilogx(f, db([float(r["T_true"]) for r in R]), color=COLOR["ps"])
        ax2.semilogx(f, db([float(r["T_base"]) for r in R]), color=COLOR["baseline"], ls="--")
    med = {k: np.median([float(r[k]) for r in feat]) for k in feat[0] if k != "Y"}
    for k in ("anti_resonance_Hz", "coupled_mode_Hz"):
        ax1.axvline(med[k], color="k", lw=0.5, ls=":")
    ax2.axvline(med["closed_loop_peak_Hz"], color="k", lw=0.5, ls=":")
    ax1.text(med["anti_resonance_Hz"], 1.02, "anti-resonance ", transform=ax1.get_xaxis_transform(), ha="right", fontsize=6)
    ax1.text(med["coupled_mode_Hz"], 1.02, " coupled mode", transform=ax1.get_xaxis_transform(), ha="left", fontsize=6)
    ax2.text(med["closed_loop_peak_Hz"], 1.02, "closed-loop peak", transform=ax2.get_xaxis_transform(), ha="center", fontsize=6)
    ax1.set_ylabel(r"$|G_{YY}|$ (dB)")
    ax2.set_ylabel(r"$|T_{YY}|$ (dB)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_xlim(20, 1000)
    ax2.set_ylim(-40, 10)
    ax2.set_xticks([20, 50, 100, 200, 500, 1000])
    ax2.set_xticklabels(["20", "50", "100", "200", "500", "1000"])
    ax1.legend(loc="lower left")
    fig.align_ylabels((ax1, ax2))
    save(fig, STEM)


if __name__ == "__main__":
    main()
