"""Shared matplotlib style for figures that go into the two-column thesis (sip-thesis.cls).

Figures are drawn at their final printed size, so a figure included with
``\\includegraphics[width=\\widthnarrow]`` (single column) or ``[width=\\textwidth]``
(both columns) is not rescaled by LaTeX and every font keeps its nominal size.
Body text is 7pt, titles 7.5pt and legends 6pt, all in a light sans-serif weight.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend import Legend
from matplotlib.text import Text

COL_W = 3.34   # in, \widthnarrow = 241.15pt (one column)
TEXT_W = 7.09  # in, \textwidth = 18cm (both columns)
MIN_FONT = 7       # pt, body text (ticks, axis labels, annotations)
TITLE_FONT = 7.5   # pt
LEGEND_FONT = 6    # pt, the only text allowed below MIN_FONT

CAT_COLORS = [
    '#2a78d6',  # slot1 blue
    '#eb6834',  # slot2 orange
    '#1baf7a',  # slot3 aqua
    '#eda100',  # slot4 yellow
    '#e87ba4',  # slot5 magenta
    '#008300',  # slot6 green
    '#4a3aa7',  # slot7 violet
    '#e34948',  # slot8 red
]

BLUE_RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
SEQ_CMAP = LinearSegmentedColormap.from_list('seq_blue', BLUE_RAMP)
DIV_CMAP = LinearSegmentedColormap.from_list('div_blue_red', ['#0d366b', '#f0efec', '#e34948'])

INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
GRID_COLOR = '#e1e0d9'
SPINE_COLOR = '#c3c2b7'

RC = {
    # light sans-serif; Roboto Light is installed here, the rest are fallbacks with a light cut
    'font.family': 'sans-serif',
    'font.sans-serif': ['Roboto', 'Lato', 'Calibri', 'Segoe UI', 'DejaVu Sans'],
    'font.weight': 300,  # Roboto Light ('light' would resolve to 200 and pick Roboto Thin)
    'axes.titleweight': 300,
    'axes.labelweight': 300,
    'figure.titleweight': 300,
    'figure.labelweight': 300,
    'font.size': MIN_FONT,
    'axes.titlesize': TITLE_FONT,
    'axes.labelsize': MIN_FONT,
    'xtick.labelsize': MIN_FONT,
    'ytick.labelsize': MIN_FONT,
    'legend.fontsize': LEGEND_FONT,
    'legend.title_fontsize': LEGEND_FONT,
    'figure.titlesize': TITLE_FONT,
    'figure.labelsize': MIN_FONT,
    'axes.titlepad': 4,
    'axes.labelpad': 2,
    'axes.linewidth': 0.6,
    'axes.edgecolor': SPINE_COLOR,
    'axes.labelcolor': INK_PRIMARY,
    'text.color': INK_PRIMARY,
    'xtick.color': INK_SECONDARY,
    'ytick.color': INK_SECONDARY,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'grid.color': GRID_COLOR,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'patch.linewidth': 0.5,
    'legend.handlelength': 1.5,
    'legend.borderaxespad': 0.3,
    'figure.dpi': 150,  # larger preview in the notebook; does not change the printed size
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}


def apply(seaborn_style=None):
    """Activate the paper style. Pass e.g. ``'whitegrid'`` to also set a seaborn style
    (seaborn's own font scaling is overridden so the sizes above still hold)."""
    if seaborn_style is not None:
        import seaborn as sns
        sns.set_theme(context='paper', style=seaborn_style, rc=RC)
    mpl.rcParams.update(RC)


def clean_axes(ax):
    """Drop top/right spines; tint remaining spines and ticks per the style guide."""
    ax.spines[['top', 'right']].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(SPINE_COLOR)
    ax.tick_params(colors=INK_SECONDARY)
    return ax


def enforce_font_sizes(fig, min_size=MIN_FONT, max_size=None, legend_size=LEGEND_FONT):
    """Clamp every text in ``fig`` (incl. ticks, colorbars, subfigures) to [min_size, max_size]
    and legend text to >= legend_size, and switch regular-weight text to the light weight.
    Needed for third-party plots (MNE, specparam, sklearn) that hard-code their own fonts."""
    legend_texts = {id(t) for leg in fig.findobj(Legend) for t in leg.findobj(Text)}
    for t in fig.findobj(Text):
        lo = legend_size if id(t) in legend_texts else min_size
        size = t.get_fontsize()
        if size < lo:
            t.set_fontsize(lo)
        elif max_size is not None and size > max_size:
            t.set_fontsize(max_size)
        if t.get_fontweight() in ('normal', 'regular', 400):
            t.set_fontweight(300)
    return fig


def subject_id(s):
    """'subject_24' -> '24' (short tick labels)."""
    return str(s).replace('subject_', '')


def session_label(s):
    """'session_1' -> 'Session 1'."""
    return str(s).replace('session_', 'Session ')


apply()
