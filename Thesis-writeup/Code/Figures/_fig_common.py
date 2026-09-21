"""Shared figure style for the paper.

Convention (one stem, three folders):
    Code/Figures/<stem>.py   reads   Results/<stem>.csv
                             writes  Writing/figures/<stem>.pdf

Every figure in the paper is therefore traceable to exactly one script and one
result file. Scripts never write anywhere else inside Writing/.
"""

from pathlib import Path

import matplotlib as mpl

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[1]                 # Thesis-writeup/
RESULTS_DIR = ROOT / "Results"
DATA_DIR = ROOT / "Data"
FIG_DIR = ROOT / "Writing" / "figures"
TAB_DIR = ROOT / "Writing" / "tables"

# IEEE two-column text widths, in inches.
COL_WIDTH = 3.5
FULL_WIDTH = 7.16


def use_paper_style():
    """Match the IEEEtran body font size so no figure needs rescaling in LaTeX."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
    })


def save(fig, stem):
    """Write the figure as vector PDF into Writing/figures/<stem>.pdf."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{stem}.pdf"
    fig.savefig(out)
    print(f"wrote {out}")
    return out


def save_table(body, stem):
    """Write a tabular fragment into Writing/tables/<stem>.tex for \input."""
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    out = TAB_DIR / f"{stem}.tex"
    out.write_text(body, encoding="utf-8")
    print(f"wrote {out}")
    return out
