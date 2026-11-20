"""Reproduce Figure 1 (win rate by label-space size) and Figure 2 (winner grid).

Usage (from the repository root):
    python analysis/make_figures.py

Writes figures/fig_win_rate_by_labelspace.{pdf,png} and figures/fig_winner_grid.{pdf,png},
plus the underlying numbers as CSV.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from make_tables import MODELS, ROOT, coverage, load_results, load_task_info  # noqa: E402

OUT = ROOT / "figures"
K_SHOTS = 8
BINS = [(2, 2, "$K{=}2$"), (3, 5, "$K{=}3$–$5$"), (6, 8, "$K{=}6$–$8$"), (9, 999, r"$K{\geq}9$")]
COLORS = {"Ori.": "0.78", "CC": "#6A8CAF", "DCC": "#7FA87F", "LOOC": "#8B3A3A"}
F1_COLS = {"Ori.": "macro_f1", "CC": "cc_macro_f1", "DCC": "dc_macro_f1", "LOOC": "looc_macro_f1"}
SHORT = {m: m.replace("-Instruct", "-Ins").replace("DeepSeek-R1-Distill-", "DS-") for m in MODELS}

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7,
})


def long_results(info):
    frames = []
    for model, d in load_results("k8").items():
        d = d.copy()
        d["Model"] = model
        frames.append(d)
    res = pd.concat(frames)
    res["tid"] = res.index
    return res.merge(info[["K", "cov"]], left_on="tid", right_index=True)


def figure1(res):
    """Share of task-model pairs where each setting has the best Macro-F1 (ties split)."""
    cols = list(F1_COLS.values())
    is_win = res[cols].eq(res[cols].max(axis=1), axis=0)
    share = is_win.div(is_win.sum(axis=1), axis=0)
    share.columns = list(F1_COLS)

    rows, counts, covs = [], [], []
    for lo, hi, _ in BINS:
        mask = (res.K >= lo) & (res.K <= hi)
        rows.append([100 * share.loc[mask, m].mean() for m in F1_COLS])
        counts.append(res.loc[mask, "tid"].nunique())
        covs.append(100 * res.loc[mask].drop_duplicates("tid")["cov"].mean())
    W = np.array(rows)
    pd.DataFrame(W, columns=list(F1_COLS), index=[b[2] for b in BINS]).assign(
        n_tasks=counts, coverage_pct=covs).to_csv(OUT / "win_rate_by_labelspace.csv")

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    x, w = np.arange(len(BINS)), 0.2
    for i, m in enumerate(F1_COLS):
        bars = ax.bar(x + (i - 1.5) * w, W[:, i], w, label=m, color=COLORS[m], linewidth=0, zorder=3)
        if m == "LOOC":
            for b, v in zip(bars, W[:, i]):
                ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                            xytext=(0, 2), ha="center", fontsize=6, color=COLORS[m])
    ax.axvline(1.5, color="0.45", lw=0.8, ls=":", zorder=2)
    ax.text(1.55, 96, f"$k={K_SHOTS}$ demos rarely\ncover all labels", fontsize=6, color="0.35",
            va="top", ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b[2]}\n$n$={n}\ncov. {c:.0f}%" for b, n, c in zip(BINS, counts, covs)])
    ax.set_ylabel("Win rate (% of task-model pairs)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="0.9", lw=0.5, zorder=0)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15),
              handlelength=1.0, columnspacing=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_win_rate_by_labelspace.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def figure2(res):
    """Best calibration method per model and K group, with margin over the runner-up."""
    methods = ["CC", "DCC", "LOOC"]
    winner = np.empty((len(MODELS), len(BINS)), dtype=object)
    margin = np.zeros((len(MODELS), len(BINS)))
    counts = []
    for j, (lo, hi, _) in enumerate(BINS):
        sub = res[(res.K >= lo) & (res.K <= hi)]
        counts.append(sub.tid.nunique())
        for i, model in enumerate(MODELS):
            d = sub[sub.Model == model]
            means = {m: 100 * d[F1_COLS[m]].mean() for m in methods}
            rank = sorted(means, key=means.get, reverse=True)
            winner[i, j], margin[i, j] = rank[0], means[rank[0]] - means[rank[1]]
    pd.DataFrame(winner, index=MODELS, columns=[b[2] for b in BINS]).to_csv(OUT / "winner_grid.csv")

    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    for i in range(len(MODELS)):
        for j in range(len(BINS)):
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=COLORS[winner[i, j]],
                                       edgecolor="white", linewidth=1.2, zorder=2))
            ax.text(j, i, f"{margin[i, j]:.1f}", ha="center", va="center", fontsize=6, color="white", zorder=3)
    ax.set_xlim(-.5, len(BINS) - .5)
    ax.set_ylim(len(MODELS) - .5, -.5)
    ax.set_xticks(range(len(BINS)))
    ax.set_xticklabels([f"{b[2]}\n$n$={n}" for b, n in zip(BINS, counts)])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([SHORT[m] for m in MODELS])
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[Patch(facecolor=COLORS[m], label=m) for m in methods], frameon=False, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.04), handlelength=1.0, columnspacing=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_winner_grid.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    info = load_task_info()
    rng_info = info.copy()
    rng_info["cov"] = coverage(info, verbose=False)
    res = long_results(rng_info)
    figure1(res)
    figure2(res)
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
