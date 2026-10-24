"""Table A (Aspect 1): NRMS_e per axis in % (05_setup eq:setup_rms) of every structure, the baseline at
the true parameters, the final model and the black box, mean over the records with the training
multisine and over the records without it (D-256). No noise-floor row: Section V states the floor
through the SNR (Dirk 2026-10-24).

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
        ("R1", "model", r"$\mathcal{M}_{\mathrm{OBC}}$"),
        ("R8", "bb", r"$\mathcal{M}_{\mathrm{BB}}$")]
MIDRULE_BEFORE = ("R1",)


def records(e, ms, split):
    recs = sorted({k[3] for k in e if k[0] == split and k[1] == "R1" and k[2] == "model"})
    return [r for r in recs if ms[r]], [r for r in recs if not ms[r]]


def table(e, ms, split):
    r_ms, r_no = records(e, ms, split)
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l r r r r r r}", r"\toprule",
             r" & \multicolumn{3}{c}{Multisine} & \multicolumn{3}{c}{No multisine}\\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"$\mathrm{NRMS}_e$ (\%) & " + " & ".join("$%s$" % AXTEX[a] for _ in range(2) for a in AXES) + r"\\",
             r"\midrule"]
    for run, model, name in ROWS:
        if run in MIDRULE_BEFORE:
            lines.append(r"\midrule")
        cells = [r"$%.1f$" % v for recs in (r_ms, r_no) for v in mean_over(e, split, run, model, recs)]
        lines.append(" & ".join([name] + cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    head = "%% %s records: %d with multisine (%s), %d without\n" % (split, len(r_ms), ", ".join(r_ms), len(r_no))
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
