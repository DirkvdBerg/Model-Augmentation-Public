"""Figure (Aspect 1): validation NRMS_e during training for the structures of Table A, all from the true
parameters: n_xbar = 0 (R6), n_xbar = 2 plain (R3), n_xbar = 2 PS (R5), n_xbar = 8 PS (R7).

Source: the training logs of the final runs, not a CSV. Every 1300 updates the log prints
"It <n>, sqrt loss <x>, Val sim-NRMSe <v>"; <v> is the selection score over the whole validation
records (it equals val_nrmse_checkpoint of Results/res_checks.csv at the selected update). Update 0 is
the "Initial Validation sim-NRMSe", the baseline before training. The PS opening lies before the first
logged point. Logs: Results/18-10-2026/thesis-results-4/final-u_89795_*.out (run_id = job id of
res_checks.csv: 89797 R3, 89798 R5, 89799 R6, 89800 R7).

Writes Writing/figures/res_val_curves.pdf.
"""
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from _fig_common import COL_WIDTH, ROOT, save, use_paper_style

STEM = "res_val_curves"
LOG_DIR = ROOT / "Results" / "18-10-2026" / "thesis-results-4"
RUNS = [("89799", r"$n_{\bar x}=0$", "#9467bd"), ("89797", r"$n_{\bar x}=2$, plain", "#ff7f0e"),
        ("89798", r"$n_{\bar x}=2$, PS", "#1f77b4"), ("89800", r"$n_{\bar x}=8$, PS", "#d62728")]


def parse(path):
    s = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    jid = re.search(r"run_id (\d{5})", s).group(1)
    init = float(re.search(r"Initial Validation sim-NRMSe= ([\d.]+)", s).group(1))
    its = re.findall(r"It (\d+), sqrt loss\s+[\d.e-]+, Val sim-NRMSe\s+([\d.e-]+)", s)
    return jid, np.array([0] + [int(i) for i, _ in its]), np.array([init] + [float(v) for _, v in its])


def main():
    use_paper_style()
    curves = {}
    for p in sorted(LOG_DIR.glob("final-u_*.out")):
        jid, x, y = parse(p)
        curves[jid] = (x, y)
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.1))
    for jid, lab, col in RUNS:
        x, y = curves[jid]
        ax.semilogy(x / 1e3, 100 * y, color=col, lw=1.2, marker="o", ms=2.2, label=lab, zorder=3)
    ax.set_xlabel(r"Optimiser update ($\times 10^3$)")
    ax.set_ylabel(r"Validation $\mathrm{NRMS}_e$ (%)")
    ax.set_xlim(-0.5, 34)
    ax.set_ylim(5, 50)
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks([5, 10, 20, 50], ["5", "10", "20", "50"])
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(False, which="minor")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, handlelength=1.5,
              columnspacing=0.9, borderaxespad=0.1)
    save(fig, STEM)


if __name__ == "__main__":
    main()
