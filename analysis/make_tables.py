"""Reproduce the paper's tables and label-coverage estimates from results/.

Usage (from the repository root):
    python analysis/make_tables.py

Reads results/k8/<model>/task_metrics.csv, results/k4/<model>/task_metrics.csv and
data/classification_tasks/test_tasks_info.csv. No GPU or model access needed.
"""
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TASK_INFO = ROOT / "data" / "classification_tasks" / "test_tasks_info.csv"

MODELS = [
    "Llama-3.2-3B", "Llama-3.2-3B-Instruct",
    "Phi-3.5-mini-instruct", "Phi-3.5-MoE-instruct",
    "Mixtral-8x7B-v0.1", "Mixtral-8x7B-Instruct-v0.1",
    "Qwen2.5-32B", "Qwen2.5-32B-Instruct",
    "DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-32B",
]
SETTINGS = {"Ori.": "", "CC": "cc_", "DCC": "dc_", "LOOC": "looc_"}
METRICS = {"Macro-F1": "macro_f1", "RSD": "rsd", "BiasScore": "bias_score"}
LOWER_IS_BETTER = {"rsd", "bias_score"}
GROUPS = [(2, 2, "K=2"), (3, 5, "3<=K<=5"), (6, 999, "K>=6")]
ALPHA = 0.05
K_DEMOS = 8
N_SIM = 20_000


def load_task_info():
    info = pd.read_csv(TASK_INFO)
    info["dist"] = info["Labels Distribution"].apply(ast.literal_eval)
    info["K"] = info["# Answer Choices"].astype(int)
    info["tid"] = info["Name"].map(task_id)
    return info.set_index("tid")


def task_id(name):
    """'task050_multirc_answerability_classification' -> 'task050'.

    Task names in the results files carry the Super-NI category suffix, while
    test_tasks_info.csv uses the short names, so tasks are matched by ID.
    """
    return re.match(r"(task\d+)", str(name)).group(1)


def load_results(shots="k8"):
    frames = {}
    for model in MODELS:
        path = RESULTS / shots / model / "task_metrics.csv"
        if path.exists():
            d = pd.read_csv(path, index_col=0)
            d.index = d.index.map(task_id)
            frames[model] = d
    return frames


def col(setting, metric):
    return SETTINGS[setting] + metric


def table1(res):
    print("\nTable 1: mean over all 279 tasks (%)")
    header = "".join(f"{s:>7}" for s in SETTINGS)
    print(f"{'Model':30s}" + "".join(f" | {m:^28s}" for m in METRICS))
    print(f"{'':30s}" + (" | " + header) * len(METRICS))
    for model, d in res.items():
        row = f"{model:30s}"
        for metric in METRICS.values():
            row += " | " + "".join(f"{100 * d[col(s, metric)].mean():7.1f}" for s in SETTINGS)
        print(row)


def looc_margin(d, metric):
    """LOOC margin over the better (by mean) of CC and DCC, in points, and Wilcoxon p."""
    lower = metric in LOWER_IS_BETTER
    cc, dc, lo = d[col("CC", metric)], d[col("DCC", metric)], d[col("LOOC", metric)]
    if lower:
        alt = cc if cc.mean() <= dc.mean() else dc
        margin = 100 * (alt.mean() - lo.mean())
    else:
        alt = cc if cc.mean() >= dc.mean() else dc
        margin = 100 * (lo.mean() - alt.mean())
    try:
        p = stats.wilcoxon(lo, alt).pvalue
    except ValueError:  # all differences zero
        p = 1.0
    return margin, p


def table2(res, info):
    print("\nTable 2: LOOC margin over the better of CC and DCC (points); "
          "+ = LOOC better, † = p >= 0.05 (paired Wilcoxon over tasks)")
    groups = [g for _, _, g in GROUPS]
    print(f"{'Model':30s}" + "".join(f" | {m:^24s}" for m in METRICS))
    print(f"{'':30s}" + (" | " + "".join(f"{g:>8s}" for g in groups)) * len(METRICS))
    wins = {(m, g): 0 for m in METRICS.values() for g in groups}
    for model, d in res.items():
        K = info.loc[d.index, "K"]
        row = f"{model:30s}"
        for metric in METRICS.values():
            row += " | "
            for lo, hi, g in GROUPS:
                margin, p = looc_margin(d[(K >= lo) & (K <= hi)], metric)
                wins[(metric, g)] += margin > 0
                row += f"{margin:+7.1f}{'†' if p >= ALPHA else ' '}"
        print(row)
    print(f"{'LOOC better':30s}" + "".join(
        " | " + "".join(f"{str(wins[(m, g)]) + '/' + str(len(res)):>8s}" for g in groups)
        for m in METRICS.values()))


def table3(res8, res4):
    print("\nTable 3: Macro-F1 (%) with k = 8 vs k = 4 demonstrations")
    for model, d4 in res4.items():
        d8 = res8[model]
        print(model)
        for s in SETTINGS:
            a, b = 100 * d8[col(s, "macro_f1")].mean(), 100 * d4[col(s, "macro_f1")].mean()
            print(f"  {s:5s} k=8 {a:5.1f}  k=4 {b:5.1f}  delta {b - a:+.1f}")


def table4(res, info):
    print("\nTable 4: results by label-space size (%); † = LOOC not significantly different "
          "from the better of CC and DCC")
    for metric_name, metric in METRICS.items():
        print(f"\n{metric_name}")
        print(f"{'Model':30s}" + "".join(f" | {g:^30s}" for _, _, g in GROUPS))
        for model, d in res.items():
            K = info.loc[d.index, "K"]
            row = f"{model:30s}"
            for lo, hi, _ in GROUPS:
                sub = d[(K >= lo) & (K <= hi)]
                _, p = looc_margin(sub, metric)
                vals = "".join(f"{100 * sub[col(s, metric)].mean():7.1f}" for s in SETTINGS)
                row += f" | {vals}{'†' if p >= ALPHA else ' '} "
            print(row)


def pair_margin(res, info):
    """Mean LOOC margin over the better of CC and DCC, computed per task-model pair (Section 1)."""
    d = pd.concat(res.values())
    K = info.loc[d.index, "K"].values
    gap = 100 * (d["looc_macro_f1"] - d[["cc_macro_f1", "dc_macro_f1"]].max(axis=1))
    print("\nLOOC Macro-F1 margin over the better of CC and DCC, averaged over task-model pairs")
    for lo, hi, g in [(2, 2, "K=2"), (3, 5, "3<=K<=5"), (6, 8, "6<=K<=8"), (9, 999, "K>=9")]:
        print(f"  {g:9s} {gap[(K >= lo) & (K <= hi)].mean():+.2f} points")


def coverage(info, k=K_DEMOS, n_sim=N_SIM, seed=0, verbose=True):
    """Probability that k demonstrations drawn from the label distribution contain every label."""
    rng = np.random.default_rng(seed)

    def cov(dist):
        p = np.array(list(dist.values()), dtype=float)
        p /= p.sum()
        draws = rng.choice(len(p), size=(n_sim, k), p=p)
        offsets = draws + np.arange(n_sim)[:, None] * len(p)
        counts = np.bincount(offsets.ravel(), minlength=n_sim * len(p)).reshape(n_sim, len(p))
        return (counts > 0).all(axis=1).mean()

    info = info.copy()
    info["cov"] = info["dist"].apply(cov)
    if not verbose:
        return info["cov"]
    print(f"\nLabel coverage with k = {k} ({n_sim:,} draws per task): "
          "share of prompts in which every label appears")
    for lo, hi, g in [(2, 2, "K=2"), (3, 5, "3<=K<=5"), (6, 8, "6<=K<=8"), (9, 999, "K>=9")]:
        sub = info[(info.K >= lo) & (info.K <= hi)]
        print(f"  {g:9s} n={len(sub):3d}  coverage {100 * sub['cov'].mean():5.1f}%")
    return info["cov"]


def main():
    info = load_task_info()
    res8 = load_results("k8")
    res4 = load_results("k4")
    print(f"Tasks: {len(info)} | K=2: {(info.K == 2).sum()}, 3-5: {info.K.between(3, 5).sum()}, "
          f">=6: {(info.K >= 6).sum()} | max K: {info.K.max()}")
    table1(res8)
    table2(res8, info)
    table3(res8, res4)
    table4(res8, info)
    pair_margin(res8, info)
    coverage(info)


if __name__ == "__main__":
    main()
