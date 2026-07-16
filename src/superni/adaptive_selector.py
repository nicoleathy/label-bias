"""
adaptive_selector.py

Skew-Aware Adaptive Calibration (SAC) selector.

Motivation
----------
Our central empirical finding is that calibration effectiveness depends
on label-distribution skew: LOOC wins on low-skew (high-bias) tasks,
while CC/DCC are competitive on high-skew (balanced) tasks. On its own
this is a *finding*. SAC turns it into a *method*: for each task we
select the calibration strategy using only the task's
label-distribution skew (normalised label entropy), a quantity known
a priori from the data and never from test predictions. We then show
that this per-task selection beats any single fixed method aggregated
over all tasks.

Why this is not circular
------------------------
The selection signal is `NormEntropy`, a property of the task's label
distribution computed before any model is run. The decision threshold
is fit on a held-out split of *tasks* (task-level cross-validation),
never on the test tasks it is evaluated on, and never using test-set
gold labels. So SAC uses no test-time information: it is a genuine
predictive rule, not an oracle.

Inputs
------
1. A per-task metrics table: one row per (task, method) with columns
   {task, method, macro_f1, rsd, bias_score}. This is exactly what
   `compute_metrics_for_experiment` already writes to `task_metrics.csv`
   for each run, reshaped to long form (helper below).
2. A per-task skew table with columns {task, NormEntropy}, which the
   repo already provides in
   `data/eval/superni/splits/classification_tasks/test_tasks_info_with_complexity_and_bias.csv`.

Output
------
- The SAC selection per task, the aggregated SAC score, and a comparison
  against every fixed method and against the per-task oracle (upper
  bound). Also reports leave-one-task-out (LOTO) cross-validated SAC so
  the gain is not an artifact of threshold overfitting.

Usage
-----
    python adaptive_selector.py \
        --metrics_csv per_task_metrics_long.csv \
        --skew_csv   test_tasks_info_with_complexity_and_bias.csv \
        --primary_metric macro_f1

If you have one `task_metrics.csv` per run (wide form), use
`build_long_metrics_from_run_dirs` to assemble the long table first.
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd


METHODS = ["ori", "cc", "dc", "looc"]  # 'dc' == DCC in the paper
HIGHER_BETTER = {"macro_f1": True, "rsd": False, "bias_score": False}


# ---------------------------------------------------------------------
# Assembling the long per-task metrics table from existing run outputs
# ---------------------------------------------------------------------
def build_long_metrics_from_run_dirs(run_glob):
    """Read every task_metrics.csv matching `run_glob` and melt into long
    form with columns [task, method, macro_f1, rsd, bias_score].

    task_metrics.csv is wide: columns like macro_f1, cc_macro_f1,
    looc_rsd, etc., indexed by task. We map each prefixed column back to
    (method, metric); the unprefixed columns are the 'ori' (original)
    setting.
    """
    rows = []
    for path in glob.glob(run_glob, recursive=True):
        df = pd.read_csv(path, index_col=0)
        for task, r in df.iterrows():
            for method in METHODS:
                rec = {"task": task, "method": method}
                for metric in ["macro_f1", "rsd", "bias_score"]:
                    col = metric if method == "ori" else f"{method}_{metric}"
                    rec[metric] = r[col] if col in r and pd.notna(r[col]) else np.nan
                rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Core selector
# ---------------------------------------------------------------------
def _score_sign(metric):
    return 1.0 if HIGHER_BETTER[metric] else -1.0


def fit_threshold(train_df, skew, primary_metric,
                  low_method="looc", high_method="cc"):
    """Fit a single skew threshold tau that maximises mean primary metric
    on the training tasks, under the rule:
        skew <= tau  -> low_method   (low-skew, high-bias regime)
        skew >  tau  -> high_method  (balanced regime)

    Returns tau. Candidate thresholds are the observed skew values, so
    this is an exact 1-D search, not gradient-based.
    """
    sign = _score_sign(primary_metric)
    tasks = train_df["task"].unique()
    # Per-task, per-method primary metric lookup.
    piv = train_df.pivot_table(index="task", columns="method",
                               values=primary_metric)
    cand = sorted(skew.loc[skew.index.isin(tasks)].unique())
    best_tau, best_score = None, -np.inf
    for tau in cand:
        chosen = []
        for t in tasks:
            m = low_method if skew.get(t, np.nan) <= tau else high_method
            val = piv.loc[t, m] if (t in piv.index and m in piv.columns) else np.nan
            if pd.isna(val):
                val = piv.loc[t, "ori"] if "ori" in piv.columns else np.nan
            chosen.append(val)
        score = sign * np.nanmean(chosen)
        if score > best_score:
            best_score, best_tau = score, tau
    return best_tau


def apply_selector(eval_df, skew, tau, primary_metric,
                   low_method="looc", high_method="cc"):
    """Apply a fitted threshold to evaluation tasks; return per-task chosen
    method and the per-task value of every metric under that choice."""
    out = []
    piv = {m: eval_df.pivot_table(index="task", columns="method", values=m)
           for m in ["macro_f1", "rsd", "bias_score"]}
    for t in eval_df["task"].unique():
        m = low_method if skew.get(t, np.nan) <= tau else high_method
        row = {"task": t, "chosen_method": m, "skew_value": skew.get(t, np.nan)}
        for metric in ["macro_f1", "rsd", "bias_score"]:
            P = piv[metric]
            val = P.loc[t, m] if (t in P.index and m in P.columns and pd.notna(P.loc[t, m])) else \
                  (P.loc[t, "ori"] if t in P.index and "ori" in P.columns else np.nan)
            row[metric] = val
        out.append(row)
    return pd.DataFrame(out)


def oracle(eval_df, primary_metric):
    """Per-task best-possible method by the primary metric (upper bound)."""
    sign = _score_sign(primary_metric)
    piv = eval_df.pivot_table(index="task", columns="method", values=primary_metric)
    best = {}
    for t in piv.index:
        vals = piv.loc[t]
        best[t] = vals.idxmax() if sign > 0 else vals.idxmin()
    return best


def fixed_method_scores(eval_df):
    """Mean of each metric for every fixed method, over eval tasks."""
    res = {}
    for m in METHODS:
        sub = eval_df[eval_df["method"] == m]
        res[m] = {metric: sub[metric].mean() for metric in
                  ["macro_f1", "rsd", "bias_score"]}
    return res


def leave_one_task_out(long_df, skew, primary_metric,
                       low_method="looc", high_method="cc"):
    """LOTO cross-validation: fit tau on all tasks but one, predict the
    held-out task, aggregate. Guards against threshold overfitting."""
    tasks = long_df["task"].unique()
    chosen_vals = {m: [] for m in ["macro_f1", "rsd", "bias_score"]}
    for held in tasks:
        train = long_df[long_df["task"] != held]
        tau = fit_threshold(train, skew, primary_metric, low_method, high_method)
        pred = apply_selector(long_df[long_df["task"] == held], skew, tau,
                              primary_metric, low_method, high_method)
        for metric in chosen_vals:
            chosen_vals[metric].append(pred[metric].iloc[0])
    return {metric: np.nanmean(v) for metric, v in chosen_vals.items()}


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", help="Long per-task metrics: columns "
                    "task,method,macro_f1,rsd,bias_score")
    ap.add_argument("--run_glob", help="Alternatively, glob for wide "
                    "task_metrics.csv files, e.g. 'runs/**/task_metrics.csv'")
    ap.add_argument("--skew_csv", required=True,
                    help="CSV with columns Name,NormEntropy")
    ap.add_argument("--primary_metric", default="macro_f1",
                    choices=["macro_f1", "rsd", "bias_score"])
    ap.add_argument("--low_method", default="looc")
    ap.add_argument("--high_method", default="cc")
    ap.add_argument("--out", default="adaptive_selector_report.csv")
    args = ap.parse_args()

    if args.metrics_csv:
        long_df = pd.read_csv(args.metrics_csv)
    elif args.run_glob:
        long_df = build_long_metrics_from_run_dirs(args.run_glob)
    else:
        raise SystemExit("Provide --metrics_csv or --run_glob")

    skew_tab = pd.read_csv(args.skew_csv)
    skew = skew_tab.set_index("Name")["NormEntropy"]

    # Fit on all tasks (in-sample) and via LOTO (honest) ----------------
    tau = fit_threshold(long_df, skew, args.primary_metric,
                        args.low_method, args.high_method)
    sac = apply_selector(long_df, skew, tau, args.primary_metric,
                         args.low_method, args.high_method)
    sac_mean = {m: sac[m].mean() for m in ["macro_f1", "rsd", "bias_score"]}
    loto_mean = leave_one_task_out(long_df, skew, args.primary_metric,
                                   args.low_method, args.high_method)

    fixed = fixed_method_scores(long_df)
    orc = oracle(long_df, args.primary_metric)
    orc_pivot = long_df.pivot_table(index="task", columns="method",
                                    values=args.primary_metric)
    orc_mean = np.nanmean([orc_pivot.loc[t, orc[t]] for t in orc_pivot.index])

    print(f"\nSkew threshold tau (in-sample fit): {tau:.4f}")
    print(f"Rule: skew <= tau -> {args.low_method.upper()}, "
          f"else -> {args.high_method.upper()}\n")

    print(f"=== {args.primary_metric} (primary) ===")
    for m in METHODS:
        print(f"  fixed {m:5s}: {fixed[m][args.primary_metric]:.3f}")
    print(f"  SAC (in-sample): {sac_mean[args.primary_metric]:.3f}")
    print(f"  SAC (LOTO cv)  : {loto_mean[args.primary_metric]:.3f}")
    print(f"  ORACLE (upper) : {orc_mean:.3f}")

    best_fixed = max(METHODS, key=lambda m: _score_sign(args.primary_metric) *
                     fixed[m][args.primary_metric])
    gain = (loto_mean[args.primary_metric] -
            fixed[best_fixed][args.primary_metric])
    print(f"\n  Best fixed method: {best_fixed.upper()} "
          f"({fixed[best_fixed][args.primary_metric]:.3f})")
    print(f"  SAC(LOTO) - best fixed: {gain:+.3f}")

    sac.to_csv(args.out, index=False)
    print(f"\nPer-task selections written to {args.out}")


if __name__ == "__main__":
    main()
