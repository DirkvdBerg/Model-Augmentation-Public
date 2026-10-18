"""Fig. (Aspect 4): NRMS_e per axis (the primary metric, 05_setup eq:setup_rms) of the baseline,
the augmented model (OBC) and the black-box model per record, one row of panels per axis X1, X2, Y.

Reads Results/res_generalisation.csv (Code/Evaluation/make_results.py).
Writes Writing/figures/res_generalisation.pdf: the interpolation (I) and extrapolation (E) test
records and, once evaluated, the multisine twins of the test records (D-258); and
res_generalisation_train.pdf, res_generalisation_val.pdf with the same bars per training and
validation record (D-269). Red marks: the noise floor of the same record. E records are ordered by
the property in which they differ from their I partner (Y position, Y moving, acceleration,
velocity, ILC move). E5 (changed controller) is not scored (user 2026-10-11).
BASELINE: the physics model alone at the TRUE parameters (M_LPV of 05_setup; Dirk 2026-10-18), so
its error is the missing dynamics only; the legend says "true theta" explicitly. The CSV also holds
the detuned start (baseline_detuned), shown in Table I.
"""
import csv

import matplotlib.pyplot as plt

from _fig_common import FULL_WIDTH, COLOR, FLOOR, RESULTS_DIR, mark_preliminary, nrms_pct, save, use_paper_style

STEM = "res_generalisation"
BASELINE = "baseline_true"           # Dirk 2026-10-18; "baseline_detuned" is in the CSV too
TEST_PANELS = [("Interpolation", ["I1", "I2", "I3", "I4", "I5", "I6"]),
               ("Extrapolation", ["E1", "E2", "E3", "E4", "E6"])]
LABEL = {"baseline_detuned": r"Baseline model, detuned $\theta$", "baseline_true": r"Baseline model (true parameters $\theta^\star$)"}
MODELS = [(BASELINE, LABEL[BASELINE]), ("final", "Augmented model (OBC)"), ("bb", "Black-box model")]
AX = ("X1", "X2", "Y")
AXTEX = {"X1": "X_1", "X2": "X_2", "Y": "Y"}


def color(m):
    return COLOR["ps" if m == "final" else "baseline" if m.startswith("baseline") else m]


def figure(rows, panels, stem):
    val = {(r["record"], r["model"], a): nrms_pct(r["nrms_e_%s" % a]) for r in rows for a in AX}
    w = 0.27
    fig, axs = plt.subplots(len(AX), len(panels), figsize=(FULL_WIDTH, 1.25 * len(AX) + 0.3), sharey="row",
                            squeeze=False,
                            gridspec_kw=dict(width_ratios=[len(r) for _, r in panels], wspace=0.06, hspace=0.25))
    for i, a in enumerate(AX):
        for ax, (title, recs) in zip(axs[i], panels):
            for x, rec in enumerate(recs):
                for k, (m, _) in enumerate(MODELS):
                    ax.bar(x + (k - 1) * w, val[(rec, m, a)], w, color=color(m), edgecolor="none", zorder=2)
                ax.plot([x - 1.5 * w, x + 1.5 * w], [val[(rec, "floor", a)]] * 2, zorder=3, **FLOOR)
            ax.set_yscale("log")
            ax.set_xticks(range(len(recs)), recs if i == len(AX) - 1 else [""] * len(recs))
            ax.tick_params(axis="x", length=0, pad=2, labelsize=6 if len(recs) > 12 else 7)
            ax.set_xlim(-0.6, len(recs) - 0.4)
            if i == 0:
                ax.set_title(title, fontsize=8, pad=3)
            ax.grid(True, which="major", axis="y", alpha=0.3, zorder=0)
            ax.grid(False, axis="x")
        axs[i][0].set_ylabel(r"$\mathrm{NRMS}_{e,%s}$ (%%)" % AXTEX[a])
    handles = [plt.Rectangle((0, 0), 1, 1, color=color(m), label=lbl) for m, lbl in MODELS] \
        + [plt.Line2D([], [], label="Noise floor", **FLOOR)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False,
               handlelength=1.4, columnspacing=1.2)
    mark_preliminary(fig, [r["status"] for r in rows if r["model"] == "final"], "augmented model (R1)")
    save(fig, stem)
    plt.close(fig)


def main():
    use_paper_style()
    with open(RESULTS_DIR / f"{STEM}.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    split = lambda s: [r for r in rows if r["split"] == s]                 # noqa: E731
    recs = lambda rs: list(dict.fromkeys(r["record"] for r in rs))         # noqa: E731
    test, twins = split("test"), split("twins")
    panels = TEST_PANELS + ([("Multisine twins", recs(twins))] if twins else [])
    figure(test + twins, panels, STEM)
    for s, title in (("train", "Training records"), ("val", "Validation records")):
        if split(s):
            figure(split(s), [(title, recs(split(s)))], "%s_%s" % (STEM, s))


if __name__ == "__main__":
    main()
