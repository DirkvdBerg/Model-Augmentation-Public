"""Table E (Aspect 3): NRMS_e per axis in % of the baseline at the true parameters, the final model and
the black box, mean over the interpolation (I) and the extrapolation (E) test records, with the training
multisine, without it, and with the training multisine added to the records without it. Replaces the
per-record figure (Dirk 2026-10-24). Each E record has the motion of an I record (E1: I2, E2 and E6: I3,
E3: I5, E4: I6; Appendix tab:setup_records).

Reads Results/res_generalisation.csv (Code/Evaluation/make_results.py), writes
Writing/tables/res_generalisation.tex.
"""
import numpy as np

from _fig_common import AXES, AXTEX, read_csv, save_table

STEM = "res_generalisation"
COLS = [("test", "Multisine", ["I1", "I2"], ["E1"]),
        ("test", "No multisine", ["I3", "I4", "I5", "I6"], ["E2", "E3", "E4", "E6"]),
        ("twins", "Multisine added", ["I3M", "I5M", "I6M"], ["E2M", "E3M", "E4M", "E6M"])]
MODELS = [("baseline_true", r"$\mathcal{M}_{\mathrm{LPV}}$"), ("final", r"$\mathcal{M}_{\mathrm{OBC}}$"),
          ("bb", r"$\mathcal{M}_{\mathrm{BB}}$")]


def main():
    val = {(r["split"], r["record"], r["model"]): r for r in read_csv(STEM) if r["split"] in ("test", "twins")}
    mean = lambda split, recs, m, a: np.mean([100 * float(val[(split, x, m)]["nrms_e_%s" % a]) for x in recs])  # noqa: E731
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l l r r r r r r}", r"\toprule",
             " & & " + " & ".join(r"\multicolumn{2}{c}{%s}" % t for _, t, _, _ in COLS) + r"\\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"$\mathrm{NRMS}_e$ (\%) & & I & E & I & E & I & E\\"]
    for m, name in MODELS:
        lines.append(r"\midrule")
        for k, a in enumerate(AXES):
            cells = [r"$%.1f$" % mean(split, recs, m, a) for split, _, irecs, erecs in COLS for recs in (irecs, erecs)]
            lines.append(" & ".join([name if k == 0 else "", "$%s$" % AXTEX[a]] + cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    save_table("\n".join(lines), STEM)


if __name__ == "__main__":
    main()
