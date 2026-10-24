"""Table A (Aspect 1): NRMS_e per axis in % (05_setup eq:setup_rms) of every structure, the baseline at
the true parameters, the final model and the black box, mean over the records with the training
multisine and over the records without it (D-256), and the noise floor as the last row (Dirk 2026-10-24).

Rows: baseline (R5 base_true), n_xbar = 0 (R6), n_xbar = 2 plain (R3), n_xbar = 2 PS (R5), n_xbar = 8
PS (R7), all from the true parameters, then the final model M_OBC (R1, detuned start) and the black box
(R8).

Reads Results/res_errors.csv (per record, Code/Evaluation/make_results.py), writes
  res_structure.tex          MAIN TEXT: all 11 test records (Dirk 2026-10-24)
  res_structure_<split>.tex  APPENDIX: the same rows on the training, validation and test records
"""
from _fig_common import AXES, AXTEX, errors, mean_over, save_table

STEM = "res_structure"
ROWS = [("R5", "base_true", r"$\mathcal{M}_{\mathrm{LPV}}$"),
        ("R6", "model", r"$n_{\bar x}=0$"),
        ("R3", "model", r"$n_{\bar x}=2$, plain"),
        ("R5", "model", r"$n_{\bar x}=2$, PS"),
        ("R7", "model", r"$n_{\bar x}=8$, PS"),
        ("R1", "model", r"$\mathcal{M}_{\mathrm{OBC}}$")]   # no black box: Aspect 3, Table V (Dirk 2026-10-24)
MIDRULE_BEFORE = ("R1",)


def records(e, ms, split):
    recs = sorted({k[3] for k in e if k[0] == split and k[1] == "R1" and k[2] == "model"})
    return [r for r in recs if ms[r]], [r for r in recs if not ms[r]]


def mean_pairs(e, run, model, pairs):
    """Mean NRMS_e per axis over (split, record) pairs."""
    import numpy as np
    return np.mean([e[(s, run, model, x)] for s, x in pairs], axis=0)


def table(e, ms, split):
    r_ms, r_no = records(e, ms, split)
    # main text: also the motion records simulated with a multisine (Dirk 2026-10-24; never "multisine added")
    # every record with a multisine in one group: on test, also the motion records simulated with one
    with_ms = [(split, x) for x in r_ms]
    if split == "test":
        with_ms += [("twins", x) for x in records(e, ms, "twins")[0]]
    groups = [with_ms, [(split, x) for x in r_no]]
    heads = ["Multisine", "No multisine"]
    n = len(groups)
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l " + "r " * (3 * n) + "}", r"\toprule",
             " & " + " & ".join(r"\multicolumn{3}{c}{%s}" % h for h in heads) + r"\\",
             "".join(r"\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i) for i in range(n)),
             r"$\mathrm{NRMS}_e$ (\%) & " + " & ".join("$%s$" % AXTEX[a] for _ in range(n) for a in AXES) + r"\\",
             r"\midrule"]
    for run, model, name in ROWS:
        if run in MIDRULE_BEFORE:
            lines.append(r"\midrule")
        cells = [r"$%.1f$" % v for recs in groups for v in mean_pairs(e, run, model, recs)]
        lines.append(" & ".join([name] + cells) + r"\\")
    # noise floor (05_setup: NRMS_e with y-hat replaced by y-circ), same records and mean as the model rows;
    # two decimals, since it is below 0.1 % (Dirk 2026-10-24)
    cells = [r"$%.2f$" % v for recs in groups for v in mean_pairs(e, "R8", "floor", recs)]
    lines += [r"\midrule", " & ".join(["Noise floor"] + cells) + r"\\"]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    head = "%% %s records: %d with multisine, %d without\n" % (split, len(groups[0]), len(groups[1]))
    return head + "\n".join(lines)


def main():
    e, ms = errors()
    for split in ("train", "val", "test"):
        body = table(e, ms, split)
        save_table(body, "%s_%s" % (STEM, split))
        if split == "test":
            save_table(body, STEM)


if __name__ == "__main__":
    main()
