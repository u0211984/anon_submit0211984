import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2878B5"
ORANGE = "#E07A2D"
DARK = "#30343B"
GRID = "#D8DDE3"
INTRO_COLOR = "#2f6db3"
INTERACTIVITY_COLORS = ("#1f77b4", "#d62728")
CASE_COLORS = {"A100-40GB": DARK, "serial_hbm_160GB": BLUE, "lpddr5x_640GB": ORANGE}
COUNTS = ("hbm_bytes", "l2_bytes", "smem_bytes", "tensor_flops")
COUNT_LABELS = dict(zip(COUNTS, ("HBM bytes", "L2 delivered", "SMEM bytes", "Tensor FLOPs")))
COUNT_COLORS = dict(zip(COUNTS, ("#2f67b1", "#3fb5a6", "#a5d150", "#35102e")))
PHASES = ("train", "prefill", "decode")
PHASE_LABELS = {"train": "Training", "prefill": "Prefill", "decode": "Decode"}
PHASE_COLORS = {"train": "#8e44ad", "prefill": BLUE, "decode": ORANGE}
COUNT_PHASE_COLORS = {"train": "#8e44ad", "prefill": "#e8871e", "decode": "#c0392b"}
PHASE_MARKERS = {"train": "s", "prefill": "o", "decode": "^"}

MODEL_SHORT_NAMES = {
    **{f"SmolLM2-{s}": f"S-{s}" for s in ("135M", "360M", "1.7B")},
    **{f"Llama-{v}-{s}": f"L{v}-{s}" for v, s in
       (("2", "7B"), ("2", "13B"), ("3.2", "1B"), ("3.2", "3B"),
        ("3.1", "8B"), ("3.1", "70B"))},
    **{f"Qwen2.5-{s}": f"Q2.5-{s}" for s in
       ("0.5B", "1.5B", "7B", "14B", "72B")},
    "Llama3-8B": "L3-8B", "Llama3-70B": "L3-70B",
    "Qwen3-30B-A3B": "Q3-30B-A3B",
    "Qwen3-32B": "Q3-32B", "Qwen3-235B-A22B": "Q3-235B-A22B",
    "llama2_13b": "L2-13B", "llama2_7b": "L2-7B", "llama3_8b": "L3-8B",
    "llama32_1b": "L3.2-1B", "llama32_3b": "L3.2-3B",
    "qwen15_7b_base": "Q1.5-7B", "qwen25_7b_base": "Q2.5-7B",
    "qwen72b": "Q2.5-72B",
}

def short_model_names(value):
    import re
    names = sorted(MODEL_SHORT_NAMES, key=len, reverse=True)
    return re.sub(r"(?<![\w])(?:" + "|".join(map(re.escape, names)) + r")(?![\w])",
                  lambda m: MODEL_SHORT_NAMES[m.group()], str(value))

def apply_style(kind="paper"):
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": .65, "axes.edgecolor": "#666A70",
        "grid.color": GRID, "grid.linewidth": .45, "grid.alpha": .8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    if kind == "counts":
        plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8,
            "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5, "axes.linewidth": .6,
            "xtick.major.width": .6, "ytick.major.width": .6,
            "axes.edgecolor": "black", "axes.spines.top": False,
            "axes.spines.right": False})
    elif kind == "training":
        plt.rcParams.update({"xtick.labelsize": 6.5})
    elif kind == "intro":
        plt.rcdefaults()
        plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

def panel_label(ax, letter):
    ax.text(.01, 1.025, f"({letter})", transform=ax.transAxes,
            ha="left", va="bottom", fontweight="bold", fontsize=8)
