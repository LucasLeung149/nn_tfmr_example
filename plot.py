"""Comparison chart: one bar per method, two panels (quiz-40 and validation), seed range as whiskers."""
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:  # experiment.py catches ImportError and skips the plot
    raise ImportError("matplotlib is not installed (pip install matplotlib)") from e

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE = "#2a78d6"  # one measure per panel -> one hue; the bar's position carries the method


def plot_comparison(results, methods, path):
    keys = [k for k, _ in methods]
    names = [n for _, n in methods]
    panels = [("quiz_acc", "Accuracy on the 40 quiz words", "whiskers: min-max over seeds"),
              ("val_bal_acc", "Balanced accuracy on ~280 held-out words", "whiskers: min-max over seeds")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True, facecolor=SURFACE)
    y = np.arange(len(keys))[::-1]
    for ax, (field, title, note) in zip(axes, panels):
        vals = np.array([[r[k][field] for r in results] for k in keys])  # methods x seeds
        mean, lo, hi = vals.mean(1) * 100, vals.min(1) * 100, vals.max(1) * 100
        ax.set_facecolor(SURFACE)
        ax.barh(y, mean, height=0.5, color=BLUE, zorder=2)
        ax.errorbar(mean, y, xerr=[mean - lo, hi - mean], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=3)
        ax.axvline(50, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(50.4, y.max() + 0.62, "chance", color=INK2, fontsize=8, va="bottom")
        for yi, m, h in zip(y, mean, hi):
            ax.text(max(h, m) + 1.2, yi, f"{m:.0f}%", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, 100)
        if ax is axes[0]:
            ax.set_yticks(y, names, fontsize=9, color=INK)  # sharey: label once, on the left panel
        ax.tick_params(axis="y", labelleft=ax is axes[0])
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=22)
        ax.set_xlabel(note, fontsize=8, color=INK2)
        ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=INK2, labelsize=8, length=0)
        ax.tick_params(axis="y", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    fig.suptitle(f"Surname vs pasta: five methods, {len(results)} seeds", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_quiz(rows, path, n=40):
    """rows: [(method name, number correct out of n)] -> one horizontal bar per method, with the chance line."""
    fig, ax = plt.subplots(figsize=(8.5, 0.75 * len(rows) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(rows))[::-1]
    acc = np.array([c / n * 100 for _, c in rows])
    ax.barh(y, acc, height=0.5, color=BLUE, zorder=2)
    ax.axvline(50, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(50.4, y.max() + 0.6, "chance", color=INK2, fontsize=8, va="bottom")
    for yi, (_, c), a in zip(y, rows, acc):
        ax.text(a + 1.2, yi, f"{a:.0f}%  ({c}/{n})", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"accuracy on the {n} quiz words (a 95% interval is about +/-15 points)", fontsize=8, color=INK2)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=INK2, labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
