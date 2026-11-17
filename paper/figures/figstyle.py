"""Matplotlib style of the figures: LaTeX text (Computer Modern), Wong colour-blind-safe palette."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK="#1a1a1a"; GREY="#8c8c8c"; FAINT="#d8d8d8"
ORANGE="#D55E00"; BLUE="#0072B2"; GREEN="#009E73"; PINK="#CC79A7"; YELLOW="#E69F00"; SKY="#56B4E9"

def apply():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
        "font.family":"serif", "font.serif":["Computer Modern Roman"],
        "font.size":8, "axes.labelsize":8, "axes.titlesize":8,
        "xtick.labelsize":7, "ytick.labelsize":7,
        "legend.fontsize":6.8, "legend.frameon":False, "legend.handlelength":1.3,
        "axes.linewidth":0.6, "lines.linewidth":1.3,
        "xtick.direction":"in", "ytick.direction":"in",
        "xtick.major.size":2.6, "ytick.major.size":2.6,
        "xtick.major.width":0.6, "ytick.major.width":0.6,
        "axes.spines.top":False, "axes.spines.right":False,
        "axes.edgecolor":INK, "axes.labelcolor":INK, "text.color":INK,
        "xtick.color":INK, "ytick.color":INK,
        "figure.dpi":160, "savefig.dpi":600, "savefig.pad_inches":0.02,
        "axes.axisbelow":True,
    })
