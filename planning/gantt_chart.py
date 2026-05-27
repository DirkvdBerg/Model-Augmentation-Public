"""
Gantt chart for project planning.
X-axis is in weeks relative to a project start date.
Work packages are grouped with curly braces; a vertical "today" line is drawn.

Usage:
    python planning/gantt_chart.py                        # uses sample_data.yaml if present
    python planning/gantt_chart.py -y my_data.yaml
    python planning/gantt_chart.py -y my_data.yaml -o chart.png
"""

import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, timedelta
import numpy as np
import argparse
import sys
import os

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None


# ---------------------------------------------------------------------------
# Project start date — adjust this to your actual start date.
# "Today" is rendered as (today - PROJECT_START).days / 7 on the x-axis.
# ---------------------------------------------------------------------------
PROJECT_START = date(2025, 9, 1)


def weeks_since_start(d=None):
    """Return fractional weeks elapsed from PROJECT_START to d (default: today)."""
    if d is None:
        d = date.today()
    return (d - PROJECT_START).days / 7.0


def create_gantt_chart(data, title="Project Gantt Chart", total_weeks=40):
    df = pd.DataFrame(data)
    df["Duration"] = df["End"] - df["Start"]

    tasks = df[df["Type"] == "Task"].copy()
    milestones = df[df["Type"] == "Milestone"].copy()

    fig, ax = plt.subplots(figsize=(18, 10))

    work_packages = df["Work Package"].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(work_packages)))
    wp_colors = dict(zip(work_packages, colors))

    y_pos = 0
    wp_y_ranges = {}

    for wp in work_packages:
        wp_tasks = tasks[tasks["Work Package"] == wp]
        if wp_tasks.empty:
            continue

        wp_start_y = y_pos
        for _, task in wp_tasks.iterrows():
            ax.barh(
                y_pos,
                task["Duration"],
                left=task["Start"],
                height=0.85,
                color=wp_colors[wp],
                alpha=0.75,
                edgecolor="black",
                linewidth=0.5,
            )
            bar_center_x = task["Start"] + task["Duration"] / 2
            if task["Duration"] >= 1.0:
                ax.text(
                    bar_center_x, y_pos, task["Task"],
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              alpha=0.8, edgecolor="none"),
                )
            else:
                ax.text(
                    task["Start"] + task["Duration"] + 0.2, y_pos, task["Task"],
                    ha="left", va="center", fontsize=8, fontweight="bold",
                    color=wp_colors[wp],
                )
            y_pos += 1

        wp_y_ranges[wp] = (wp_start_y, y_pos - 1)

    # ---- Milestones --------------------------------------------------------
    for _, ms in milestones.iterrows():
        mx = ms["Start"]
        related = []
        if "Related WPs" in ms and pd.notna(ms.get("Related WPs")):
            related = [w.strip() for w in str(ms["Related WPs"]).split(",")]

        mc = []
        for wpid in related:
            for full_wp in wp_colors:
                if wpid in full_wp:
                    mc.append(wp_colors[full_wp])
                    break

        if len(mc) == 1:
            ax.axvline(x=mx, color=mc[0], linewidth=3, alpha=0.8, zorder=10)
        elif len(mc) > 1:
            yb, yt = ax.get_ylim()
            h = yt - yb
            sh = 0.2
            ns = int(h / sh) + 1
            for i in range(ns):
                c = mc[i % len(mc)]
                ys = yb + i * sh
                ye = min(ys + sh, yt)
                if ys < yt:
                    ax.axvline(x=mx,
                               ymin=(ys - yb) / h, ymax=(ye - yb) / h,
                               color=c, linewidth=3, alpha=0.8, zorder=10)
        else:
            ax.axvline(x=mx, color="gray", linewidth=3, alpha=0.8, zorder=10)

    # ---- Milestone labels --------------------------------------------------
    def wrap_text(s, width=20):
        if len(s) <= width:
            return s
        words = s.split()
        lines, cur = [], ""
        for w in words:
            if cur and len(cur + " " + w) > width:
                lines.append(cur); cur = w
            elif cur:
                cur += " " + w
            else:
                cur = w
        if cur:
            lines.append(cur)
        return "\n".join(lines)

    if not milestones.empty:
        ms_sorted = sorted([(r["Start"], r["Task"]) for _, r in milestones.iterrows()])
        for i, (xp, name) in enumerate(ms_sorted):
            row = (i % 3) + 1 if i > 0 and abs(xp - ms_sorted[i-1][0]) < 4 else 0
            ly = -0.5 - row * 1.5
            ax.text(xp, ly, wrap_text(name),
                    ha="center", va="top", fontsize=9, fontweight="bold",
                    color="#333333", zorder=20)

    # ---- This week band --------------------------------------------------------
    current_week_float = weeks_since_start()
    this_week = int(current_week_float)  # 0-based index; week number = this_week + 1
    if 0 <= this_week <= total_weeks:
        ax.axvspan(this_week, min(this_week + 1, total_weeks),
                   color="red", alpha=0.12, zorder=5)
        ax.text(this_week + 0.5, 0, f"Week {this_week + 1}",
                color="red", fontsize=9, fontweight="bold",
                ha="center", va="top", zorder=20)

    # ---- Axes & styling ----------------------------------------------------
    ax.set_yticks([])
    ax.invert_yaxis()
    if y_pos > 0:
        ax.set_ylim(y_pos, -4)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(-4, total_weeks + 1)

    # Week-number ticks — label every 2 weeks to keep it readable
    week_ticks = list(range(0, total_weeks + 1))
    week_labels = [str(w) if w % 2 == 0 else "" for w in week_ticks]
    ax.set_xticks(week_ticks)
    ax.set_xticklabels(week_labels, fontsize=7)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlabel("")

    # ---- Month band (alternating colors) below week ticks ------------------
    # Compute exact fractional week for the 1st of each calendar month
    BAND_COLORS = ["#E8F4FD", "#F0F0F0"]  # alternating soft blue / light gray
    BAND_Y0 = -0.13  # bottom of band in axes-fraction coords
    BAND_Y1 = -0.06  # top of band (just below the tick labels)

    project_end = PROJECT_START + timedelta(weeks=total_weeks)

    # Collect (week_pos, label) for every month boundary within the range
    month_firsts = [(0.0, PROJECT_START.strftime("%b"))]  # project start
    m, y = PROJECT_START.month % 12 + 1, PROJECT_START.year + (1 if PROJECT_START.month == 12 else 0)
    cursor = date(y, m, 1)
    while cursor <= project_end:
        wp = (cursor - PROJECT_START).days / 7.0
        month_firsts.append((wp, cursor.strftime("%b")))
        next_m = cursor.month % 12 + 1
        next_y = cursor.year + (1 if cursor.month == 12 else 0)
        cursor = date(next_y, next_m, 1)
    month_firsts.append((total_weeks, None))  # sentinel

    xform = ax.get_xaxis_transform()  # x: data coords, y: axes fraction
    import matplotlib.patches as mpatches
    for i, (x0, label) in enumerate(month_firsts[:-1]):
        x1 = month_firsts[i + 1][0]
        color = BAND_COLORS[i % 2]
        # Filled rectangle in axes-fraction y space
        rect = mpatches.FancyBboxPatch(
            (x0, BAND_Y0), x1 - x0, BAND_Y1 - BAND_Y0,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor="none",
            transform=xform, clip_on=False, zorder=3,
        )
        ax.add_patch(rect)
        # Month label centered in the band
        ax.text((x0 + x1) / 2, (BAND_Y0 + BAND_Y1) / 2, label,
                ha="center", va="center", fontsize=8,
                color="#444444", transform=xform, zorder=4, clip_on=False)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    # ---- Work-package curly braces -----------------------------------------
    def format_wp(wp_name):
        s = wp_name.replace(": ", "\\\n") if ": " in wp_name else wp_name
        return s

    for wp, (sy, ey) in wp_y_ranges.items():
        c = wp_colors[wp]
        label = format_wp(wp)
        bx = -0.8
        if sy == ey:
            ax.text(-1.2, sy, label, ha="right", va="center",
                    fontweight="bold", fontsize=9, color=c)
        else:
            my = (sy + ey) / 2
            ax.plot([bx, bx + 0.2], [sy - 0.4, sy - 0.4], "k-", linewidth=1.5)
            ax.plot([bx, bx],       [sy - 0.4, ey + 0.4], "k-", linewidth=1.5)
            ax.plot([bx, bx + 0.2], [ey + 0.4, ey + 0.4], "k-", linewidth=1.5)
            ax.text(-1.2, my, label, ha="right", va="center",
                    fontweight="bold", fontsize=9, rotation=20, color=c)

    plt.tight_layout(rect=[0, 0.04, 1, 1])  # reserve bottom space for month band
    return fig, ax


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_from_yaml(path):
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML not installed: pip install PyYAML")
    with open(path) as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "tasks" in data:
        return data["tasks"]
    raise ValueError("YAML must be a list or have a 'tasks' key")


def load_from_csv(path):
    return pd.read_csv(path).to_dict("records")


def load_from_excel(path, sheet=0):
    return pd.read_excel(path, sheet_name=sheet).to_dict("records")


def validate_data(data):
    required = ["Task", "Work Package", "Start", "End", "Type"]
    if not data:
        raise ValueError("No data loaded.")
    missing = set(required) - set(data[0].keys())
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    print(f"Data OK — {len(data)} items loaded.")


# ---------------------------------------------------------------------------
# Sample data (week-based)
# ---------------------------------------------------------------------------
sample_data = [
    {"Task": "Literature review",   "Work Package": "WP1: Preparation", "Start": 1,  "End": 4,  "Type": "Task"},
    {"Task": "Problem definition",  "Work Package": "WP1: Preparation", "Start": 3,  "End": 6,  "Type": "Task"},
    {"Task": "Prep complete",       "Work Package": "WP1: Preparation", "Start": 6,  "End": 6,  "Type": "Milestone", "Related WPs": "WP1"},

    {"Task": "Model derivation",    "Work Package": "WP2: Modelling",   "Start": 6,  "End": 14, "Type": "Task"},
    {"Task": "Verification",        "Work Package": "WP2: Modelling",   "Start": 12, "End": 18, "Type": "Task"},
    {"Task": "Model verified",      "Work Package": "WP2: Modelling",   "Start": 18, "End": 18, "Type": "Milestone", "Related WPs": "WP2"},

    {"Task": "Framework setup",     "Work Package": "WP3: Augmentation","Start": 18, "End": 22, "Type": "Task"},
    {"Task": "Fitting & tuning",    "Work Package": "WP3: Augmentation","Start": 22, "End": 30, "Type": "Task"},
    {"Task": "Augmentation done",   "Work Package": "WP3: Augmentation","Start": 30, "End": 30, "Type": "Milestone", "Related WPs": "WP3"},

    {"Task": "Benchmarking",        "Work Package": "WP4: Evaluation",  "Start": 30, "End": 35, "Type": "Task"},
    {"Task": "Thesis writing",      "Work Package": "WP4: Evaluation",  "Start": 28, "End": 38, "Type": "Task"},
    {"Task": "Submission",          "Work Package": "WP4: Evaluation",  "Start": 38, "End": 38, "Type": "Milestone", "Related WPs": "WP4"},
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate a week-based Gantt chart")
    p.add_argument("-c", "--csv",   help="CSV file path")
    p.add_argument("-x", "--excel", help="Excel file path")
    p.add_argument("-y", "--yaml",  help="YAML file path")
    p.add_argument("-s", "--sheet", default=0, help="Excel sheet name/index")
    p.add_argument("-t", "--title", default="Project Gantt Chart")
    p.add_argument("-o", "--output", help="Save figure to file (e.g. chart.png)")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--weeks", type=int, default=40, help="Total weeks on x-axis")
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = None

    try:
        if args.csv:
            data = load_from_csv(args.csv)
        elif args.excel:
            data = load_from_excel(args.excel, args.sheet)
        elif args.yaml:
            data = load_from_yaml(args.yaml)
        else:
            default_yaml = os.path.join(os.path.dirname(__file__), "sample_data.yaml")
            if YAML_AVAILABLE and os.path.exists(default_yaml):
                data = load_from_yaml(default_yaml)
            else:
                print("No data file specified — using built-in sample data.")
                data = sample_data

        validate_data(data)
        fig, ax = create_gantt_chart(data, title=args.title, total_weeks=args.weeks)

        if args.output:
            plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved to {args.output}")

        if not args.no_display:
            plt.show()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
