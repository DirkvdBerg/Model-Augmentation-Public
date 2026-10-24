"""Shared figure style for the paper.

Convention (one stem, three folders):
    Code/Figures/<stem>.py   reads   Results/<stem>.csv
                             writes  Writing/figures/<stem>.pdf

Every figure in the paper is therefore traceable to exactly one script and one
result file. Scripts never write anywhere else inside Writing/.
"""

import os
from pathlib import Path

import matplotlib as mpl

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[1]                 # Thesis-writeup/
RESULTS_DIR = Path(os.environ.get("THESIS_RESULTS_DIR", ROOT / "Results"))   # e.g. Results/18-10-2026/CSV
DATA_DIR = ROOT / "Data"
FIG_DIR = ROOT / "Writing" / "figures"
TAB_DIR = ROOT / "Writing" / "tables"

# IEEE two-column text widths, in inches.
COL_WIDTH = 3.5
FULL_WIDTH = 7.16

# One colour per model in every results figure.
COLOR = {
    "baseline": "#7f7f7f",     # LPV baseline at the true parameters
    "lti": "#c7c7c7",          # frozen-Y LTI baseline (same family as the baseline, lighter)
    "plain": "#ff7f0e",        # augmented, plain start
    "ps": "#1f77b4",           # augmented, PS start (also the final model)
    "bb": "#2ca02c",           # black box
}
FLOOR = dict(color="#d62728", linestyle="--", linewidth=0.9)   # noise floor

# Line styles of the final results figures (Dirk 2026-10-24): the system is a thick light-grey band, the
# models are thin solid lines on top of it; the baseline dark grey, the final model blue.
SYS = dict(color="#c8c8c8", lw=2.6, solid_capstyle="round")
BASE = dict(color="#4d4d4d", lw=0.8)
MOD = dict(color="#1f77b4", lw=0.9)
NAME = {"sys": "System", "base": r"Baseline $\mathcal{M}_{\mathrm{LPV}}$",
        "model": r"Final $\mathcal{M}_{\mathrm{OBC}}$"}
Y_TR = 0.30      # training range Y_tr [m], Writing/tables/setup_numbers.tex SetupTrainY
AXES = ("X1", "X2", "Y")
AXTEX = {"X1": "X_1", "X2": "X_2", "Y": "Y"}


def read_csv(stem):
    """Rows of Results/<stem>.csv as dicts."""
    import csv
    with open(RESULTS_DIR / f"{stem}.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def errors():
    """Per-record NRMS_e in %: {(split, run, model, record): (X1, X2, Y)} and {record: has multisine},
    from Results/res_errors.csv (Code/Evaluation/make_results.py)."""
    e, ms = {}, {}
    for r in read_csv("res_errors"):
        e[(r["split"], r["run"], r["model"], r["record"])] = tuple(100.0 * float(r["nrms_e_%s" % a]) for a in AXES)
        ms[r["record"]] = r["multisine"] == "1"
    return e, ms


def mean_over(e, split, run, model, recs):
    """Mean NRMS_e per axis over the records `recs` (the per-record mean, as res_structure.csv)."""
    import numpy as np
    return np.mean([e[(split, run, model, rec)] for rec in recs], axis=0)


def use_paper_style():
    """Match the IEEEtran body font size so no figure needs rescaling in LaTeX."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],   # IEEEtran body is Times
        "mathtext.fontset": "stix",
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


def preliminary(statuses):
    """The distinct "preliminary, update <n> of <N>" labels among `statuses` (runs evaluated before
    they finished, Code/Evaluation/eval_common.checkpoint_status)."""
    return sorted({s for s in statuses if s and s.startswith("preliminary")})


def mark_preliminary(fig, statuses, who):
    """Stamp a figure whose numbers come from a run that has not finished; nothing when all final."""
    labs = preliminary(statuses)
    if labs:
        fig.text(0.995, -0.04, "%s: %s" % (who, "; ".join(labs)), ha="right", va="top",   # below the x label
                 fontsize=6, color="#d62728")


def nrms_pct(x):
    """NRMS_e as a fraction -> percent (05_setup eq:setup_rms reports it in %)."""
    return 100.0 * float(x)
