"""Shared slide-figure style: large fonts, no titles (the slide title says it)."""
__project_origin__ = "added"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13, 'axes.labelsize': 13, 'legend.fontsize': 11, 'xtick.labelsize': 11,
                     'ytick.labelsize': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e0e0e0', 'grid.linewidth': 0.6, 'lines.linewidth': 2.0})
BAND = '#dbe8f7'
BAD = '#f6d3c2'
BLUE, ORANGE, PURPLE, GREY = '#2a78d6', '#d9581f', '#7b3fb0', '#8a8983'
