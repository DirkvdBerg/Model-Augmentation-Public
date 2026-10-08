"""Paths, palette and slide-size styling shared by every figure of the control-group talk.

The figures are PNG images for 16:9 slides (13.33 x 7.5 in). Each figure is drawn at the size
it is shown on the slide, so a 16 pt font in matplotlib is 16 pt on the projected slide. Colour
is assigned by ROLE and is the same in every figure; swap the hex values here to follow the TU/e
template (palette validated with the dataviz validator: all checks pass, SKY needs direct labels
because its contrast on white is 2.25:1).
"""
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
TALK = HERE.parent                      # control-group-presentation/
FIGDIR = TALK / 'figures'
SERVER = TALK / 'server'
MEETING = TALK.parent                   # scripts/gantry/meeting/
REPO = HERE.parents[4]

# Read-only caches built by earlier sessions (handoff 2026-09-29, section 4). Never rebuilt here.
TRACE_CACHE = MEETING / 'meeting-07-09-2026' / 'code' / 'figure_data_%s.npz'
OBC_JSON = MEETING / 'meeting-18-09-2026' / 'figures' / 'OBC' / 'obc_data.json'
ORTH_JSON = MEETING / 'meeting-21-09-2026' / 'figures' / 'orthogonality_data.json'
THESIS_GANTRY_TEX = REPO / 'Thesis-writeup' / 'Writing' / 'figures' / 'gantry_system_with_absorber.tex'

# Roles. Keep in sync with the TikZ colour definitions in tikz/colors.tex.
BASE = '#D55E00'       # physics model (baseline), vermillion
AUG = '#0072B2'        # augmented model, no projection, blue
PROJ = '#009E73'       # augmented model with orthogonal projection, bluish green
SKY = '#56B4E9'        # the smaller learned block (2 states), always dashed and direct-labelled
INK = '#222222'
MUTED = '#6b6b6b'
GRID = '#e3e3e3'

DPI = 250
UM = 1e6               # metres to micrometres, only to compare with the handoff's µm values


def sci(v, digits=2):
    """'2.06·10⁻⁵' as matplotlib mathtext: errors are always shown in metres, scientific."""
    import math
    e = int(math.floor(math.log10(abs(v))))
    return r'%.*f$\cdot$10$^{%d}$' % (digits, v / 10 ** e, e)


def sci_axis(ax):
    """Metres in scientific notation with a ×10⁻ⁿ factor on the axis."""
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(14)


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold',
        'font.size': 16, 'axes.labelsize': 16, 'legend.fontsize': 15,
        'xtick.labelsize': 14, 'ytick.labelsize': 14,
        'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.linewidth': 1.0,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.axisbelow': True,
        'lines.linewidth': 2.2,
        'legend.frameon': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.dpi': DPI, 'savefig.facecolor': 'white',
    })
    return plt


def save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path)
    print('[fig] %s' % path.relative_to(TALK))
    return path
