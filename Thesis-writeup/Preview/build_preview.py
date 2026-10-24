"""Preview of the proposed Results floats (Dirk 2026-10-24), for judging before anything enters Writing/.

Reads the final CSVs (Results/24-10-2026/CSV) and the training logs of the final runs
(Results/18-10-2026/thesis-results-4, validation history of R3, R5, R6, R7), writes PNG + PDF figures
and HTML tables into Preview/out/ and one page Preview/index.html. Writing/ is never touched.

Confidentiality (05_setup.tex, Dirk 2026-10-24): no errors in metres, no absolute plant gain, positions
relative to the training range Y_tr, frequencies relative to the controller bandwidth f_b. Every
figure here follows that rule.

Run from the repository root:
    conda run -n GraduationProject python Thesis-writeup/Preview/build_preview.py
"""
import csv
import html
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                    # Thesis-writeup/
CSV_DIR = ROOT / "Results" / "24-10-2026" / "CSV"
LOG_DIR = ROOT / "Results" / "18-10-2026" / "thesis-results-4"
OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "Code" / "Figures"))
import _fig_common as fc                              # noqa: E402

COL, FULL = fc.COL_WIDTH, fc.FULL_WIDTH
F_B = 100.0     # controller bandwidth f_b [Hz], 05_setup.tex ("K_1 has a bandwidth of f_b = 100 Hz")
Y_TR = 0.30     # training range Y_tr [m], Writing/tables/setup_numbers.tex SetupTrainY
F_POLE = 212.13   # absorber pole of the system, Code/Figures/res_ps_spectrum.py
MS_BAND = (106.0, 297.0)   # multisine band, Code/Figures/res_ps_spectrum.py
AXES = ("X1", "X2", "Y")
AXH = {"X1": "X<sub>1</sub>", "X2": "X<sub>2</sub>", "Y": "Y"}
AXT = {"X1": "X_1", "X2": "X_2", "Y": "Y"}

# One style per model in every figure (Dirk 2026-10-24): the system is a thick light-grey band, the
# models are thin solid lines on top; the baseline is dark grey, the final model blue.
SYS = dict(color="#c8c8c8", lw=2.6, solid_capstyle="round")
BASE = dict(color="#4d4d4d", lw=0.8)
MOD = dict(color="#1f77b4", lw=0.9)
C = dict(model="#1f77b4", bb="#2ca02c", plain="#ff7f0e", na0="#9467bd", na8="#d62728", base="#4d4d4d",
         lti="#a6a6a6")
NAME = {"base": r"Baseline $\mathcal{M}_{\mathrm{LPV}}$", "model": r"Final $\mathcal{M}_{\mathrm{OBC}}$",
        "sys": "System"}


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200 if ext == "png" else None)
    plt.close(fig)
    print("wrote", stem)


def read(stem):
    with open(CSV_DIR / f"{stem}.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def pct(x):
    return 100.0 * float(x)


def table_html(head, rows, note=""):
    """Booktabs-like HTML table. head: header rows of (text, colspan). A body row is a group title (str)
    or a list of cells; a cell may be (text, colspan); a first cell starting with "!" draws a rule above."""
    h = ['<table class="bt">']
    for hr in head:
        h.append("<tr>" + "".join(f'<th colspan="{cs}">{t}</th>' for t, cs in hr) + "</tr>")
    for r in rows:
        if isinstance(r, str):
            h.append(f'<tr class="grp"><td colspan="20">{r}</td></tr>')
            continue
        cls = ' class="sep"' if r[0].startswith("!") else ""
        cells = [r[0].lstrip("!")] + list(r[1:])
        h.append(f"<tr{cls}>" + "".join(f'<td colspan="{c[1]}" style="text-align:center">{c[0]}</td>'
                                        if isinstance(c, tuple) else f"<td>{c}</td>" for c in cells) + "</tr>")
    h.append("</table>")
    if note:
        h.append(f'<p class="tnote">{note}</p>')
    return "\n".join(h)


# ---------------------------------------------------------------- per-record errors
def errors():
    """{(split, run, model, record): (nrms X1, X2, Y) in %} and {record: multisine flag}, res_errors.csv."""
    e, ms = {}, {}
    for r in read("res_errors"):
        e[(r["split"], r["run"], r["model"], r["record"])] = tuple(pct(r[f"nrms_e_{a}"]) for a in AXES)
        ms[r["record"]] = r["multisine"] == "1"
    return e, ms


def mean_over(e, split, run, model, recs):
    v = np.array([e[(split, run, model, rec)] for rec in recs])
    return v.mean(axis=0)


# ---------------------------------------------------------------- Aspect 1
TEST_MS = ["I1", "I2", "E1"]
TEST_NOMS = ["I3", "I4", "I5", "I6", "E2", "E3", "E4", "E6"]
STRUCT = [("R5", "base_true", "Baseline &#8499;<sub>LPV</sub>"),
          ("R6", "model", "n<sub>x&#772;</sub> = 0"),
          ("R3", "model", "n<sub>x&#772;</sub> = 2, plain"),
          ("R5", "model", "n<sub>x&#772;</sub> = 2, PS"),
          ("R7", "model", "n<sub>x&#772;</sub> = 8, PS"),
          ("R1", "model", "!Final &#8499;<sub>OBC</sub>"),
          ("R8", "bb", "Black box &#8499;<sub>BB</sub>")]


def table_a(e):
    rows = []
    for run, model, name in STRUCT:
        cells = []
        for recs in (TEST_MS, TEST_NOMS):
            cells += ["%.1f" % v for v in mean_over(e, "test", run, model, recs)]
        rows.append([name] + cells)
    head = [[("", 1), ("With multisine (I1, I2, E1)", 3), ("Without multisine (8 records)", 3)],
            [("NRMS<sub>e</sub> (%)", 1)] + [(AXH[a], 1) for a in AXES] * 2]
    return table_html(head, rows)


def parse_log(path):
    s = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    init = float(re.search(r"Initial Validation sim-NRMSe= ([\d.]+)", s).group(1))
    its = [(int(i), float(v)) for i, _, v in re.findall(r"It (\d+), sqrt loss\s+([\d.e-]+), Val sim-NRMSe\s+([\d.e-]+)", s)]
    jid = re.search(r"run_id (\d{5})", s).group(1)
    return jid, np.array([0] + [i for i, _ in its]), np.array([init] + [v for _, v in its])


def figure_b():
    """Validation NRMS_e during training, structures of Table A, all from the true parameters."""
    want = {"89799": ("na0", r"$n_{\bar x}=0$"), "89797": ("plain", r"$n_{\bar x}=2$, plain"),
            "89798": ("model", r"$n_{\bar x}=2$, PS"), "89800": ("na8", r"$n_{\bar x}=8$, PS")}
    checks = {r["job"]: float(r["val_nrmse_checkpoint"]) for r in read("res_checks")}
    curves = {}
    for p in sorted(LOG_DIR.glob("final-u_*.out")):
        jid, x, y = parse_log(p)
        if jid in want:
            curves[jid] = (x, y)
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    for jid, (key, lab) in want.items():
        x, y = curves[jid]
        ax.semilogy(x / 1e3, 100 * y, color=C[key], lw=1.2, marker="o", ms=2.2, label=lab, zorder=3)
    ax.set_xlabel(r"Optimiser update ($\times 10^3$)")
    ax.set_ylabel(r"Validation $\mathrm{NRMS}_e$ (%)")
    ax.set_xlim(-0.5, 34)
    ax.set_ylim(5, 50)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([5, 10, 20, 50], ["5", "10", "20", "50"])
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(False, which="minor")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, handlelength=1.5,
              columnspacing=0.9, borderaxespad=0.1)
    save(fig, "B_validation_curves")
    return {want[j][1]: (curves[j][1][-1], checks[j]) for j in want}


def time_trace():
    """I1, Y axis, normalised by sigma_e of the record (05_setup eq:setup_rms): top the moving NRMS_e,
    bottom the servo error over sigma_e in the zoom window, in the record's own time."""
    rows = read("res_time_trace")
    c = lambda k: np.array([float(r[k]) for r in rows])          # noqa: E731
    t, r, y, b, m = c("t_s"), c("r_Y"), c("y_Y"), c("baseline_Y"), c("final_Y")
    z = np.array([int(q["zoom"]) for q in rows], bool)
    sig = np.std(y - r)                                          # population sd over the scored samples
    win = int(0.05 * 4000)                                       # HEURISTIC: 50 ms display smoothing
    mr = lambda e: np.sqrt(np.convolve(e ** 2, np.ones(win) / win, mode="same"))   # noqa: E731
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL, 2.8), gridspec_kw=dict(hspace=0.62))
    a1.semilogy(t, 100 * mr(b - y) / sig, **BASE)
    a1.semilogy(t, 100 * mr(m - y) / sig, **MOD)
    a1.set_xlim(t[0], t[-1])
    a1.set_xlabel("Time (s)")
    a1.set_ylabel(r"Moving $\mathrm{NRMS}_{e,Y}$ (%)")
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
    a1.grid(False, which="minor")
    a2.plot(t[z], ((y - r) / sig)[z], zorder=1, **SYS)
    a2.plot(t[z], ((b - r) / sig)[z], zorder=2, **BASE)
    a2.plot(t[z], ((m - r) / sig)[z], zorder=3, **MOD)
    a2.set_xlim(t[z][0], t[z][-1])
    a2.set_xlabel("Time (s)")
    a2.set_ylabel(r"$e_Y/\sigma_{e,Y}$")
    hs = [plt.Line2D([], [], label=NAME["sys"], **SYS), plt.Line2D([], [], label=NAME["base"], **BASE),
          plt.Line2D([], [], label=NAME["model"], **MOD)]
    a1.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=2.2,
              borderaxespad=0.1)
    fig.align_ylabels((a1, a2))
    save(fig, "A_time_trace")
    zr = lambda e: 100 * np.sqrt(np.mean(e[z] ** 2)) / sig       # noqa: E731
    return zr(b - y), zr(m - y), t[z][0], t[z][-1]


def frf_load(stem, cols):
    rows = read(stem)
    Ys = sorted({float(r["Y"]) for r in rows})
    f = np.array([float(r["f_Hz"]) for r in rows if float(r["Y"]) == Ys[0]])
    G = {k: np.array([[float(r[k]) for r in rows if float(r["Y"]) == Y] for Y in Ys]) for k in cols}
    return f, Ys, G


def gmean(a):
    return np.exp(np.mean(np.log(a), axis=0))     # HEURISTIC: geometric mean over Y, as Code/Figures/res_frf.py


def frf():
    """F_Y -> Y at Y = 0, dB relative to the system at the lowest frequency (20 Hz); band: range of the final
    model over Y. The system and the baseline vary by less than 0.3 dB over Y, so they get no band (Dirk 2026-10-24)."""
    f, Ys, G = frf_load("res_frf", ["model_Y_Y", "truth_Y_Y", "base_Y_Y"])
    i0 = Ys.index(0.0)
    ref = G["truth_Y_Y"][i0, 0]
    db = lambda a: 20 * np.log10(a / ref)                         # noqa: E731
    fx = f                                                        # Hz may be shown (05_setup, Dirk 2026-10-24)
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    ax.fill_between(fx, db(G["model_Y_Y"].min(0)), db(G["model_Y_Y"].max(0)), color=MOD["color"], alpha=0.18, lw=0)
    ax.semilogx(fx, db(G["truth_Y_Y"][i0]), **SYS)
    ax.semilogx(fx, db(G["base_Y_Y"][i0]), **BASE)
    ax.semilogx(fx, db(G["model_Y_Y"][i0]), **MOD)
    ax.set_xlim(fx[0], fx[-1])
    ticks = [20, 50, 100, 200, 500]
    ax.set_xticks(ticks, ["%g" % v for v in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB, rel.)")
    ax.grid(True, which="both", alpha=0.3)
    hs = [plt.Line2D([], [], label=NAME["sys"], **SYS), plt.Line2D([], [], label=NAME["base"], **BASE),
          plt.Line2D([], [], label=NAME["model"], **MOD)]
    ax.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=2.2,
              borderaxespad=0.1)
    save(fig, "A_frf")
    modes = [r for r in read("res_frf_modes") if float(r["f_Hz"]) > 100]
    mm = [r for r in modes if r["system"] == "model"]
    tt = [r for r in modes if r["system"] == "truth"][0]
    return (min(float(r["f_Hz"]) for r in mm) / F_B, max(float(r["f_Hz"]) for r in mm) / F_B, float(tt["f_Hz"]) / F_B,
            min(float(r["zeta"]) for r in mm), max(float(r["zeta"]) for r in mm), float(tt["zeta"]))


# ---------------------------------------------------------------- Aspect 2
COMB = ["kb_sum", "cg1", "cg2", "cy", "cb_sum", "mh", "m_total", "m_diff", "J_eff", "d"]
HNAME = {"kb_sum": "k<sub>b,&Sigma;</sub>", "cg1": "c<sub>g1</sub>", "cg2": "c<sub>g2</sub>", "cy": "c<sub>y</sub>",
         "cb_sum": "c<sub>b,&Sigma;</sub>", "mh": "m<sub>h</sub>", "m_total": "m<sub>&Sigma;</sub>",
         "m_diff": "m<sub>&Delta;</sub>", "J_eff": "J<sub>eff</sub>", "d": "d"}
M_ACT_MEAN = 0.5 * (10.2 + 10.7)   # as Code/Figures/res_obc_combinations.py (D-259)


def rel_errors():
    rel, start = {}, {}
    for r in read("res_obc_combinations"):
        c, tr = r["combination"], float(r["true"])
        s = M_ACT_MEAN if c == "m_diff" else abs(tr)
        rel[(r["run"], c)] = 100 * (float(r["learned"]) - tr) / s
        if r["start"] == "detuned":
            start[c] = 100 * (float(r["start_value"]) - tr) / s
    return rel, start


def table_c():
    """The ten combinations from both starts, unconstrained and OBC (Dirk 2026-10-24: a table, not a plot)."""
    rel, start = rel_errors()
    norm = lambda run: np.sqrt(sum(rel[(run, c)] ** 2 for c in COMB))                 # noqa: E731
    diff = lambda a, b: np.sqrt(sum((rel[(a, c)] - rel[(b, c)]) ** 2 for c in COMB))  # noqa: E731
    order = [["kb_sum"], ["cg1", "cg2", "cy", "cb_sum"], ["mh", "m_total", "m_diff", "J_eff"], ["d"]]
    rows = []
    for cs in order:
        for k, c in enumerate(cs):
            rows.append([("!" if k == 0 and c != "kb_sum" else "") + HNAME[c] + ("<sup>*</sup>" if c == "m_diff" else ""),
                         "%+.1f" % start[c]] + ["%+.1f" % rel[(r, c)] for r in ("R4", "R5", "R1", "R2")])
    rows.append(["!Norm to the truth", "%.1f" % np.sqrt(sum(v ** 2 for v in start.values())),
                 "%.1f" % norm("R4"), "%.1f" % norm("R5"), "%.1f" % norm("R1"), "%.1f" % norm("R2")])
    rows.append(["Norm of the difference between the starts", "", ("%.1f" % diff("R4", "R5"), 2), ("%.1f" % diff("R1", "R2"), 2)])
    head = [[("", 1), ("", 1), ("&#8499;<sub>U</sub>", 2), ("&#8499;<sub>OBC</sub>", 2)],
            [("Relative error (%)", 1), ("Detuned initial value", 1), ("detuned start", 1), ("true start", 1),
             ("detuned start", 1), ("true start", 1)]]
    lengths = {c: (abs(rel[("R4", c)] - rel[("R5", c)]), abs(rel[("R1", c)] - rel[("R2", c)])) for c in COMB}
    return table_html(head, rows, note="<sup>*</sup> m<sub>&Delta;</sub> relative to the mean actuator mass (D-259), the others to their true value."), lengths


def table_d():
    ov = {(r["run"], r["stack"]): r for r in read("res_obc_overlap")}
    acc = {(r["run"], r["stack"], r["group"], r["row"]): r for r in read("res_obc_accuracy")}
    trip = lambda run, row: " / ".join("%.1f" % pct(acc[(run, "test", "all", row)][f"nrms_e_{a}"]) for a in AXES)  # noqa: E731
    rows = [
        "Constraint",
        ["Network overlap &rho;(F<sub>aug</sub>), training records", "%.2f" % float(ov[("R4", "train")]["network_overlap"]), "%.2f" % float(ov[("R1", "train")]["network_overlap"])],
        ["Network overlap &rho;(F<sub>aug</sub>), test records", "%.2f" % float(ov[("R4", "test")]["network_overlap"]), "%.2f" % float(ov[("R1", "test")]["network_overlap"])],
        ["Truth overlap &rho;(&Xi;), test records", "%.2f" % float(ov[("R4", "test")]["truth_overlap"]), "%.2f" % float(ov[("R1", "test")]["truth_overlap"])],
        "Accuracy, test records, NRMS<sub>e</sub> (%) X<sub>1</sub> / X<sub>2</sub> / Y",
        ["Full model", trip("R4", "model"), trip("R1", "model")],
        ["Physical part only", trip("R4", "physics"), trip("R1", "physics")],
    ]
    head = [[("", 1), ("&#8499;<sub>U</sub>", 1), ("&#8499;<sub>OBC</sub>", 1)]]
    return table_html(head, rows)


# ---------------------------------------------------------------- Aspect 3
GEN_COLS = [("test", "With multisine", ["I1", "I2"], ["E1"]),
            ("test", "Without multisine", ["I3", "I4", "I5", "I6"], ["E2", "E3", "E4", "E6"]),
            ("twins", "Multisine added", ["I3M", "I5M", "I6M"], ["E2M", "E3M", "E4M", "E6M"])]
GEN_MODELS = [("baseline_true", "Baseline &#8499;<sub>LPV</sub>"), ("final", "Final &#8499;<sub>OBC</sub>"),
              ("bb", "Black box &#8499;<sub>BB</sub>")]


def table_gen():
    """Aspect 3 as a table (Dirk 2026-10-24): per model and axis, the mean NRMS_e over the I and the E records."""
    val = {(r["split"], r["record"], r["model"]): tuple(pct(r[f"nrms_e_{a}"]) for a in AXES)
           for r in read("res_generalisation") if r["split"] in ("test", "twins")}
    rows, gaps = [], {}
    for k, (m, name) in enumerate(GEN_MODELS):
        for i, a in enumerate(AXES):
            cells = []
            for split, _, irecs, erecs in GEN_COLS:
                mi = np.mean([val[(split, rec, m)][i] for rec in irecs])
                me = np.mean([val[(split, rec, m)][i] for rec in erecs])
                cells += ["%.1f" % mi, "%.1f" % me]
                gaps[(m, a, split + _)] = me - mi
            rows.append([("!" if i == 0 and k > 0 else "") + (name if i == 0 else ""), AXH[a]] + cells)
    head = [[("", 2)] + [(t, 2) for _, t, _, _ in GEN_COLS],
            [("NRMS<sub>e</sub> (%)", 1), ("Axis", 1)] + [("I", 1), ("E", 1)] * 3]
    note = ("I: I1, I2 | I3 to I6 | I3, I5, I6. E: E1 | E2, E3, E4, E6 | E2, E3, E4, E6. "
            "Each E record has the motion of an I record (E1: I2, E2 and E6: I3, E3: I5, E4: I6).")
    return table_html(head, rows, note=note), gaps


# ---------------------------------------------------------------- Appendix
def table_splits(e):
    """Training, validation and test, NRMS_e per axis, mean over each set (D-269); no floor row (Section V)."""
    recs = {}
    for (split, run, model, rec) in e:
        if run == "R1" and model == "model":
            recs.setdefault(split, set()).add(rec)
    _, ms = errors()
    lab = [("R5", "base_true", "Baseline &#8499;<sub>LPV</sub>"), ("R4", "model", "&#8499;<sub>U</sub>"),
           ("R4", "physics", "&nbsp;&nbsp;physical part"), ("R1", "model", "&#8499;<sub>OBC</sub>"),
           ("R1", "physics", "&nbsp;&nbsp;physical part"), ("R8", "bb", "&#8499;<sub>BB</sub>")]
    rows = []
    for sp, name in (("train", "Training records"), ("val", "Validation records"), ("test", "Test records")):
        r_ms = sorted(r for r in recs[sp] if ms[r])
        r_no = sorted(r for r in recs[sp] if not ms[r])
        rows.append(f"{name} ({len(r_ms)} with, {len(r_no)} without multisine)")
        for run, model, shown in lab:
            rows.append([shown] + ["%.1f" % v for v in mean_over(e, sp, run, model, r_ms)]
                        + ["%.1f" % v for v in mean_over(e, sp, run, model, r_no)])
    head = [[("", 1), ("With multisine", 3), ("Without multisine", 3)],
            [("NRMS<sub>e</sub> (%)", 1)] + [(AXH[a], 1) for a in AXES] * 2]
    return table_html(head, rows)


def frf_full():
    cols = ["%s_%s_%s" % (m, a, b) for m in ("model", "truth", "base") for a in AXES for b in AXES]
    f, Ys, G = frf_load("res_frf_full", cols)
    ref = gmean(G["truth_Y_Y"])[0]                                # one reference for all nine entries
    db = lambda a: 20 * np.log10(a / ref)                         # noqa: E731
    fx = f                                                        # Hz may be shown (05_setup, Dirk 2026-10-24)
    fig, axs = plt.subplots(3, 3, figsize=(FULL, 4.6), sharex=True)
    for j, a in enumerate(AXES):
        for l, b in enumerate(AXES):
            ax = axs[j, l]
            m, t, s = G["model_%s_%s" % (a, b)], G["truth_%s_%s" % (a, b)], G["base_%s_%s" % (a, b)]
            ax.fill_between(fx, db(t.min(0)), db(t.max(0)), color=SYS["color"], alpha=0.6, lw=0)
            ax.fill_between(fx, db(m.min(0)), db(m.max(0)), color=MOD["color"], alpha=0.18, lw=0)
            ax.semilogx(fx, db(gmean(t)), **SYS)
            ax.semilogx(fx, db(gmean(s)), **BASE)
            ax.semilogx(fx, db(gmean(m)), **MOD)
            ax.set_xlim(fx[0], fx[-1])
            ax.set_xticks([20, 50, 100, 200, 500], ["20", "50", "100", "200", "500"])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.grid(True, which="both", alpha=0.3)
            if j == 0:
                ax.set_title(r"Input $F_{%s}$" % AXT[b], fontsize=8, pad=3)
            if j == 2:
                ax.set_xlabel("Frequency (Hz)")
            if l == 0:
                ax.set_ylabel("Output $%s$\nMagnitude (dB, rel.)" % AXT[a])
    hs = [plt.Line2D([], [], label=NAME["sys"], **SYS), plt.Line2D([], [], label=NAME["base"], **BASE),
          plt.Line2D([], [], label=NAME["model"], **MOD)]
    fig.legend(handles=hs, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02), handlelength=2.2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "App_frf_full")


def lpv_lti():
    rows = read("res_lpv_lti")
    val = {(r["record"], r["model"], a): pct(r[f"nrms_e_{a}"]) for r in rows for a in ("X1", "X2")}
    yop = {r["record"]: r["Y_op"] for r in rows}
    groups = [("Standstill", ["I2", "E1"]), ("Moving", ["I3", "E2"])]
    ylab = {"I2": r"$Y=%.2f\,Y_{\mathrm{tr}}$" % (float(yop["I2"]) / Y_TR),
            "E1": r"$Y=%.2f\,Y_{\mathrm{tr}}$" % (-0.39 / Y_TR),     # E1: |Y| = SetupEY = 0.39 m
            "I3": r"$|Y|\leq%.2f\,Y_{\mathrm{tr}}$" % (0.20 / Y_TR),  # as Code/Figures/res_lpv_lti.py
            "E2": r"$|Y|\leq%.2f\,Y_{\mathrm{tr}}$" % (0.39 / Y_TR)}
    w = 0.36
    fig, axs = plt.subplots(2, 1, figsize=(COL, 2.9), sharex=True, gridspec_kw=dict(hspace=0.15))
    for ax, a in zip(axs, ("X1", "X2")):
        x, ticks, labels, centres = 0.0, [], [], []
        for name, recs in groups:
            x0 = x
            for rec in recs:
                ax.bar(x - w / 2, val[(rec, "LPV", a)], w, color=C["base"], edgecolor="none", zorder=2)
                ax.bar(x + w / 2, val[(rec, "LTI", a)], w, color=C["lti"], edgecolor="none", zorder=2)
                ticks.append(x)
                labels.append("%s\n%s" % (rec, ylab[rec]))
                x += 1.0
            centres.append(((x0 + x - 1.0) / 2, name))
            x += 0.4
        ax.set_ylabel(r"$\mathrm{NRMS}_{e,%s}$ (%%)" % AXT[a])
        ax.grid(True, axis="y", alpha=0.3, zorder=0)
        ax.grid(False, axis="x")
        ax.tick_params(axis="x", length=0, pad=2)
    axs[-1].set_xticks(ticks, labels, fontsize=6.5)
    axs[-1].set_xlim(ticks[0] - 0.6, ticks[-1] + 0.6)
    for xc, name in centres:
        axs[-1].annotate(name, (xc, 0), xycoords=("data", "axes fraction"), xytext=(0, -24),
                         textcoords="offset points", ha="center", va="top", fontsize=7)
    hs = [plt.Rectangle((0, 0), 1, 1, color=C["base"], label="LPV baseline"),
          plt.Rectangle((0, 0), 1, 1, color=C["lti"], label=r"LTI baseline ($Y=0$)")]
    axs[0].legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                  handlelength=1.4, borderaxespad=0.1)
    fig.align_ylabels(axs)
    save(fig, "App_lpv_lti")


def ps_spectrum():
    """Y error ASD on the validation records with multisine, relative to the baseline's peak in the
    multisine band (no absolute units); frequency over f_b; no noise floor (Section V)."""
    rows = [r for r in read("res_ps_spectrum") if r["split"] == "val"]
    f = np.array([float(r["f_Hz"]) for r in rows])
    col = lambda k: np.array([float(r[k]) for r in rows])        # noqa: E731
    m = (f >= 5) & (f <= 1500)
    band = (f >= MS_BAND[0]) & (f <= MS_BAND[1])
    ref = col("asd_baseline")[band].max()
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    ax.axvspan(MS_BAND[0], MS_BAND[1], color="0.93", lw=0, zorder=0)
    ax.loglog(f[m], col("asd_baseline")[m] / ref, label="Baseline", **BASE)
    ax.loglog(f[m], col("asd_plain")[m] / ref, color=C["plain"], lw=0.9, label="Plain training")
    ax.loglog(f[m], col("asd_ps")[m] / ref, label="PS", **MOD)
    ax.set_xlim(f[m][0], f[m][-1])
    ax.set_xticks([10, 30, 100, 300, 1000], ["10", "30", "100", "300", "1000"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Error ASD (rel.)")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(False, which="minor")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=1.6,
              borderaxespad=0.1)
    save(fig, "App_ps_spectrum")


def table_bb():
    rows = [[r["index"], r["nx"], r["width"], r["lr"], "%.1f" % pct(r["best_val_nrms_e"]), r["best_update"],
             r["status"] + (" (selected)" if r["selected"] == "1" else "")] for r in read("res_bb_selection")]
    head = [[("Index", 1), ("n<sub>x</sub>", 1), ("Width", 1), ("Learning rate", 1), ("Best val. NRMS<sub>e</sub> (%)", 1),
             ("At update", 1), ("Status", 1)]]
    return table_html(head, rows)


def table_na0():
    rel, start = rel_errors()
    rows = [[HNAME[c], "%+.1f" % start[c], "%+.1f" % rel[("R14", c)], "%+.1f" % rel[("R13", c)]] for c in COMB]
    nrm = lambda run: np.sqrt(sum(rel[(run, c)] ** 2 for c in COMB))   # noqa: E731
    rows.append(["!Norm", "%.1f" % np.sqrt(sum(v ** 2 for v in start.values())), "%.1f" % nrm("R14"), "%.1f" % nrm("R13")])
    head = [[("Relative error (%)", 1), ("Start", 1), ("Unconstrained", 1), ("OBC", 1)]]
    return table_html(head, rows)


# ---------------------------------------------------------------- page
CSS = """
body{font-family:Georgia,'Times New Roman',serif;max-width:1120px;margin:0 auto;padding:16px;background:#fff;color:#1a1a1a;line-height:1.45}
h1{font-size:22px;margin:8px 0 4px} h2{font-size:18px;margin:34px 0 6px;border-bottom:1px solid #ccc;padding-bottom:4px}
.item{border:1px solid #e3e3e3;border-radius:6px;padding:14px 16px;margin:16px 0;background:#fcfcfb}
.tag{display:inline-block;font:12px/1.6 system-ui,sans-serif;padding:0 8px;border-radius:10px;margin-right:6px;background:#eef}
.new{background:#e3f4e3}.chg{background:#fff1d6}.kept{background:#eee}.cut{background:#f8dddd}
.item h3{font-size:16px;margin:4px 0 8px}
.cap{font-size:14px;margin:8px 0;padding:6px 10px;background:#f4f4f4;border-left:3px solid #999}
.notes{font:14px/1.5 system-ui,sans-serif;margin:8px 0 0} .notes li{margin:3px 0}
.figref{font:14px system-ui,sans-serif;background:#eef4fb;padding:6px 10px;border-radius:4px}
table.bt{border-collapse:collapse;font-size:14px;margin:8px 0;border-top:2px solid #000;border-bottom:2px solid #000}
table.bt th{border-bottom:1px solid #000;padding:3px 10px;font-weight:normal;text-align:center}
table.bt td{padding:2px 10px;text-align:right} table.bt td:first-child{text-align:left}
table.bt tr.grp td{font-style:italic;text-align:left;padding-top:8px}
table.bt tr.sep td{border-top:1px solid #000}
.tnote{font-size:12px;color:#555}
"""


def item(tag, cls, title, body, caption, notes):
    nl = "".join(f"<li>{n}</li>" for n in notes)
    return (f'<div class="item"><span class="tag {cls}">{tag}</span><h3>{title}</h3>{body}'
            f'<div class="cap"><b>Caption:</b> {caption}</div><ul class="notes">{nl}</ul></div>')


def img(stem):
    """Figures are sent as separate images (an embedded page is too large for the side panel)."""
    return f'<p class="figref">Figure file: <b>{stem}.png</b></p>'


def main():
    fc.use_paper_style()
    e, ms = errors()
    tA = table_a(e)
    vb = figure_b()
    nb, nm, t0, t1 = time_trace()
    fmin, fmax, ftrue, zmin, zmax, ztrue = frf()
    tC, lengths = table_c()
    tD = table_d()
    tG, gaps = table_gen()
    frf_full()
    lpv_lti()
    ps_spectrum()
    tS, tBB, tN = table_splits(e), table_bb(), table_na0()

    big = [c for c in COMB if max(lengths[c]) >= 2]          # HEURISTIC: differences under 2 points are not compared
    shorter = [c for c in big if lengths[c][1] < 0.8 * lengths[c][0]]
    longer = [c for c in big if lengths[c][1] > 1.25 * lengths[c][0]]
    equal = [c for c in big if c not in shorter and c not in longer]
    ltxt = "; ".join(f"{HNAME[c]} {lengths[c][0]:.1f} / {lengths[c][1]:.1f}" for c in COMB)
    hl = lambda cs: ", ".join(HNAME[c] for c in cs) or "none"   # noqa: E731
    a_ms = {name: mean_over(e, "test", run, model, TEST_MS) for run, model, name in STRUCT}
    a_no = {name: mean_over(e, "test", run, model, TEST_NOMS) for run, model, name in STRUCT}
    nm_ = lambda s: s.lstrip("!")                                 # noqa: E731
    yline = ", ".join(f"{nm_(n)} {v[2]:.1f}" for n, v in a_ms.items())
    gfin = [f"{lab.lower()} {gaps[('final', 'Y', key)]:+.1f}" for key, lab in (("testWith multisine", "With multisine"),
            ("testWithout multisine", "Without multisine"), ("twinsMultisine added", "Multisine added"))]

    p = ["<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
         f"<title>Results preview</title><style>{CSS}</style></head><body>",
         "<h1>Results preview (final CSVs, 2026-10-24)</h1>",
         "<p>Proposed floats of Section VII, in reading order. Nothing here is in <code>Writing/</code> yet. "
         "All errors are normalised (NRMS<sub>e</sub> or divided by &sigma;<sub>e</sub>), frequencies in Hz, "
         "positions relative to Y<sub>tr</sub>, and no noise floor is shown (Section V states it through the SNR). "
         "Figures are sent as separate images; their file names are given at their place.</p>"]

    p.append("<h2>Aspect 1: structure and training method</h2>")
    p.append(item("Changed: replaces Table I", "chg", "Table A: NRMS<sub>e</sub> of every structure, all test records", tA,
        "NRMS<sub>e</sub> per axis, mean over the test records with the training multisine (I1, I2, E1) and without it (I3 to I6, E2 to E4, E6). "
        "Rows 2 to 5 start from the true parameters, the final model from the detuned start.",
        [f"<b>Claim:</b> only added states with PS capture the missing dynamics. Y with multisine: {yline}.",
         "<b>Criterion:</b> the Y error with multisine reaches the level of the records without multisine. Only the PS rows meet it.",
         "<b>Change:</b> all 11 test records instead of I1 to I6 only, so the multisine mean is over three records instead of two; no noise-floor row.",
         "<b>Critical:</b> without multisine, n=0 and plain are clearly worse than the baseline on Y: a static or badly started network damages what the baseline already has right. Worth one sentence.",
         "<b>Critical:</b> the structure rows start from the true parameters and the final model from the detuned start; the detuned PS run (R4) gives the same conclusion."]))
    p.append(item("Kept (regenerated, restyled)", "kept", "Time trace on I1, Y axis", img("A_time_trace"),
        f"Test record I1, Y axis. Top: moving NRMS<sub>e,Y</sub> (50 ms). Bottom: servo error relative to &sigma;<sub>e,Y</sub> from {t0:.2f} to {t1:.2f} s.",
        [f"<b>Claim:</b> the final model follows the absorber oscillation that the baseline misses. In the window, the error relative to &sigma;<sub>e,Y</sub> is {nb:.0f} % (baseline) against {nm:.0f} % (final).",
         "<b>Changes:</b> normalised by &sigma;<sub>e,Y</sub> of I1 (no metres); no shaded window, the bottom panel is in the record's own time; "
         "system as a thick light band, both models thin solid lines."]))
    p.append(item("New", "new", "Figure B: validation error during training", img("B_validation_curves"),
        "Validation NRMS<sub>e</sub>, mean over the validation records and axes, during training. All four start from the true parameters. "
        "Update 0: the baseline before training.",
        ["<b>Claim:</b> the gap between PS and the other structures is not a matter of training budget: it appears in the first 1300 updates and stays.",
         "<b>Source:</b> the &ldquo;Val sim-NRMSe&rdquo; lines of the final-run logs (every 1300 updates), the selection score over the whole validation records.",
         "<b>Critical:</b> the plain run is still decreasing slowly at the end (29.6 to 25.2 % over 31000 updates); an examiner can ask whether it would converge with more updates.",
         "<b>Open (?):</b> add the black box (its log is on the server) or keep this figure to the grey-box structures."]))
    p.append(item("Changed", "chg", "Frequency response F<sub>Y</sub> to Y", img("A_frf"),
        "Bode magnitude from F<sub>Y</sub> to Y at Y = 0, linearised at rest, relative to the system at 20 Hz. "
        "Shaded: range of the final model over Y. The system and the baseline vary by less than 0.3 dB over Y.",
        [f"<b>Claim:</b> the final model has the absorber mode that the baseline lacks: {100 * fmin:.0f} to {100 * fmax:.0f} Hz against {100 * ftrue:.0f} Hz.",
         f"<b>For the text, not the figure:</b> the learned damping of that mode varies with Y ({zmin:.3f} to {zmax:.3f}) while the system's is constant ({ztrue:.3f}).",
         "<b>Changes:</b> relative dB (no absolute plant gain), frequency in Hz; lines at Y = 0, only the final model's range over Y shaded; the two small mode and damping plots removed."]))

    p.append("<h2>Aspect 2: interpretability</h2>")
    p.append(item("New: main OBC result", "new", "Table C: the ten combinations from both starts", tC,
        "Relative error of the ten identifiable combinations after training, unconstrained and OBC, from the detuned and from the true start.",
        [f"<b>Claim:</b> uniqueness is the difference between the two starts of one model (Section I definition). OBC makes it smaller for {hl(shorter)}, "
         f"about equal for {hl(equal)}, and larger for {hl(longer)}. Per combination, unconstrained / OBC: {ltxt}.",
         "<b>Claim:</b> both models move the dampers far from the truth (up to +47 %); the masses stay within 3 %.",
         "<b>Critical:</b> two starts per model is the minimum to show (non-)uniqueness; Gy&ouml;r&ouml;k et al. use 10."]))
    p.append(item("Changed: replaces Table II", "chg", "Table D: constraint and accuracy", tD,
        "OBC against unconstrained estimation: overlaps of (eq:obc_overlap) and accuracy on the test records.",
        ["<b>Criteria, in order:</b> the constraint holds (overlap near 0), uniqueness (Table C), equal accuracy, and, only in simulation, the distance to the truth (Table C).",
         "<b>Critical:</b> the network overlap of OBC is 0.00 on the training records but 0.24 on the test records: the constraint holds on the data it is imposed on.",
         "<b>Critical:</b> the physical part alone is far off on Y for both models, so a better physical part (Gy&ouml;r&ouml;k 2025's argument) does not separate them here."]))

    p.append("<h2>Aspect 3: generalisation</h2>")
    p.append(item("Changed: replaces the figure", "chg", "Table E: interpolation against extrapolation", tG,
        "NRMS<sub>e</sub> per axis, mean over the interpolation (I) and extrapolation (E) test records, with and without the training multisine, "
        "and with the training multisine added to the records without it.",
        ["<b>Claim:</b> leaving the training range does not raise the error of the final model. E minus I on Y: " + "; ".join(gfin) + " points.",
         "<b>Claim:</b> with the multisine added, the final model stays far below the baseline and the black box on Y.",
         "<b>Critical:</b> the E-I criterion needs a margin; a data-derived one is the spread of the final model over the I records without multisine.",
         "<b>Critical:</b> the black box is above the baseline on every record without multisine; it is a weak reference (not converged, D-264), and the text must say so where it is compared."]))

    p.append("<h2>Appendix</h2>")
    p.append(item("Kept (regenerated)", "kept", "Training, validation and test (app:splits)", tS,
        "NRMS<sub>e</sub> per axis on the training, validation and test records, mean over each set, with and without multisine. Physical part: the trained model with the learned block switched off.",
        ["Required by D-269. Test now over all 11 records (as Table A); no noise-floor row."]))
    p.append(item("Kept (regenerated, normalised)", "kept", "Full 3&times;3 frequency response (app:frf)", img("App_frf_full"),
        "Bode magnitude from the stage forces (columns) to the stage positions (rows), linearised at rest, relative to the system F<sub>Y</sub> to Y at the lowest frequency.",
        ["Same normalisation and styles as the main FRF figure."]))
    p.append(item("Kept (regenerated)", "kept", "LPV against LTI baseline (app:choices)", img("App_lpv_lti"),
        "NRMS<sub>e</sub> on the X axes of the LPV baseline and the LTI baseline at Y = 0, both at the true parameters.",
        ["Positions relative to Y<sub>tr</sub>; linear axis (bars on a log axis mislead)."]))
    p.append(item("Kept, or cut (?)", "kept", "Spectrum of the Y error, plain against PS (app:choices)", img("App_ps_spectrum"),
        "Amplitude spectral density of the Y simulation error on the validation records with multisine, relative to the baseline's peak in the multisine band (shaded).",
        ["<b>Open (?):</b> Table A and Fig. B already show that PS is needed; this figure adds where in frequency the gain is. Keep only if the text uses it."]))
    p.append(item("Kept", "kept", "Combinations without added states (app:choices)", tN,
        "Relative error of the ten combinations without added states, unconstrained and OBC, from the detuned start.",
        ["Here the unconstrained estimate ends closer to the truth. Supports the Discussion, not an aspect."]))
    p.append(item("Kept (regenerated)", "kept", "Black-box search (app:choices)", tBB,
        "Black-box search: the 11 configurations and their best validation NRMS<sub>e</sub>.",
        ["<b>Critical:</b> indices 1 and 8 diverged with a finite loss but are listed &ldquo;completed&rdquo;. The status column should say so."]))
    p.append("</body></html>")
    (HERE / "index.html").write_text("\n".join(p), encoding="utf-8")
    print("wrote", HERE / "index.html")


if __name__ == "__main__":
    main()
