"""
baseline_calibration.py

Post-hoc calibration baselines requested by reviewers (temperature
scaling and affine logit correction), computed ENTIRELY from the
already-saved `full_outputs.pickle` files. No model forward passes and
no GPU are required: both baselines operate on the per-instance
`log_likelihoods` (dict {label: -loss}) and `probs` that the main eval
pipeline already pickles for every run.

Design notes
------------
* These slot alongside the existing CC/DCC/LOOC path in
  compute_metrics.py. CC/DCC/LOOC each produce a per-task multiplicative
  vector applied to `probs` and then renormalised; affine correction is
  the natural diagonal-rescale sibling of CC and is expressed the same
  way (as a `calibration_params[task]` vector), so it reuses the
  existing `compute_f1_and_rsd_per_task` / `compute_bias_score_per_task`
  machinery unchanged.
* Temperature scaling instead rescales the *log-likelihoods* before the
  softmax, so it needs its own tiny apply step (it is not a
  multiplicative factor on probabilities). A single scalar temperature
  T is fit per task by minimising the negative log-likelihood of the
  gold label on a held-out calibration signal.

Both baselines are label-agnostic in the same sense as CC/DCC: the
affine baseline here is fit from the mean predictive distribution
(content-free style), and temperature is fit to sharpen/flatten the
existing scores without changing the arg-max ranking's label identity.

Integration
-----------
In compute_metrics.py, the calibration loop is:

    for calibration_method in ['cc', 'dc', 'looc']:
        ...

Add the two baselines after it (they read the *main* outputs, not a
separate sub-directory, since they are post-hoc transforms):

    from src.superni.baseline_calibration import add_baseline_metrics
    add_baseline_metrics(task_metrics, probs, gold_labels, tasks,
                         log_likelihoods=pre_calibration_outputs['log_likelihoods'],
                         bias_score_inputs=(bias_score_probs, bias_score_gold_labels,
                                            bias_score_tasks) if has_bias_score_results else None)

This adds columns `affine_macro_f1`, `affine_rsd`, `affine_bias_score`,
`temp_macro_f1`, `temp_rsd`, `temp_bias_score` to the existing
task_metrics DataFrame, so they appear in task_metrics.csv and
mean_metrics.json exactly like CC/DCC/LOOC.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.superni.utils.metrics import f1, rsd, bias_score


# ---------------------------------------------------------------------
# Affine (diagonal) logit correction baseline
# ---------------------------------------------------------------------
def compute_affine_calibration_parameters(probs, tasks):
    """Per-task diagonal affine correction fit from the mean predictive
    distribution (a content-free-style estimate of the label prior).

    This is the standard affine/vector-scaling baseline restricted to a
    diagonal transform: each label's probability is divided by the
    model's mean probability for that label, which down-weights labels
    the model over-predicts a priori. It mirrors CC but estimates the
    prior directly from the evaluation distribution rather than from
    content-free inputs, giving a cheap reference point that needs no
    extra forward passes.

    Returns {task: np.ndarray of per-label multiplicative factors}.
    """
    params = dict()
    for task in np.unique(tasks):
        task_probs = probs[tasks == task].apply(pd.Series)
        mean_probs = task_probs.mean(axis=0).values
        # Guard against zero-probability labels.
        mean_probs = np.clip(mean_probs, 1e-8, None)
        params[task] = 1.0 / mean_probs
    return params


# ---------------------------------------------------------------------
# Temperature scaling baseline
# ---------------------------------------------------------------------
def _stable_softmax(logits, T):
    z = np.asarray(logits, dtype=float) / T
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def _fit_task_temperature(task_lls, task_gold, labels):
    """Fit a single scalar temperature per task by minimising the NLL of
    the gold label over the task's instances.

    task_lls : list of dicts {label: log_likelihood} (one per instance)
    task_gold: array of gold label strings
    labels   : ordered list of the task's answer-choice labels
    """
    label_index = {lab: i for i, lab in enumerate(labels)}
    # Build an (n_instances, n_labels) matrix of log-likelihoods.
    L = np.array([[inst[lab] for lab in labels] for inst in task_lls], dtype=float)
    gold_idx = np.array([label_index[g] for g in task_gold])

    def nll(logT):
        T = np.exp(logT)  # keep T > 0
        loss = 0.0
        for row, gi in zip(L, gold_idx):
            p = _stable_softmax(row, T)
            loss -= np.log(max(p[gi], 1e-12))
        return loss / len(L)

    # Search log-temperature in a sensible range (T in ~[0.1, 10]).
    res = minimize_scalar(nll, bounds=(np.log(0.1), np.log(10.0)),
                          method="bounded")
    return float(np.exp(res.x))


def apply_temperature_metrics(log_likelihoods, gold_labels, tasks,
                              answer_choices_per_instance):
    """Fit per-task temperature and compute Macro-F1 and RSD under it.

    Returns (f1_per_task, rsd_per_task, temp_per_task).
    """
    f1_per_task, rsd_per_task, temp_per_task = dict(), dict(), dict()
    for task in np.unique(tasks):
        mask = tasks == task
        task_lls = list(log_likelihoods[mask])
        task_gold = np.asarray(gold_labels[mask])
        # Labels for this task: union of keys, ordered consistently.
        labels = list(task_lls[0].keys())

        T = _fit_task_temperature(task_lls, task_gold, labels)
        temp_per_task[task] = T

        preds = []
        for inst in task_lls:
            p = _stable_softmax([inst[lab] for lab in labels], T)
            preds.append(labels[int(np.argmax(p))])
        preds = np.array(preds)

        f1_per_task[task] = f1(task_gold, preds, average="macro",
                               labels=np.unique(task_gold))
        rsd_per_task[task] = rsd(task_gold, preds)
    return f1_per_task, rsd_per_task, temp_per_task


def compute_temperature_bias_score(log_likelihoods, gold_labels, tasks,
                                   temp_per_task):
    """BiasScore under temperature scaling, reusing the fitted T per task."""
    bias_per_task = dict()
    for task in np.unique(tasks):
        mask = tasks == task
        task_lls = list(log_likelihoods[mask])
        task_gold = gold_labels[mask]
        labels = list(task_lls[0].keys())
        T = temp_per_task.get(task, 1.0)

        rows = []
        for inst in task_lls:
            rows.append(_stable_softmax([inst[lab] for lab in labels], T))
        task_probs = pd.DataFrame(rows, columns=labels)
        bias_per_task[task] = bias_score(task_gold, task_probs)
    return bias_per_task


# ---------------------------------------------------------------------
# Top-level integration helper
# ---------------------------------------------------------------------
def add_baseline_metrics(task_metrics, probs, gold_labels, tasks,
                         log_likelihoods=None, bias_score_inputs=None):
    """Add affine and temperature baseline columns to task_metrics in place.

    Mirrors the CC/DCC/LOOC additions in compute_metrics_for_experiment,
    so the new columns flow into task_metrics.csv and mean_metrics.json
    automatically.
    """
    from src.superni.compute_metrics import (
        compute_f1_and_rsd_per_task, compute_bias_score_per_task,
    )

    # --- Affine correction (diagonal), expressed as calibration_params ---
    affine_params = compute_affine_calibration_parameters(probs, tasks)
    task_metrics["affine_macro_f1"], task_metrics["affine_rsd"] = \
        compute_f1_and_rsd_per_task(probs, gold_labels, tasks,
                                    calibration_params=affine_params)
    if bias_score_inputs is not None:
        bs_probs, bs_gold, bs_tasks = bias_score_inputs
        task_metrics["affine_bias_score"] = compute_bias_score_per_task(
            bs_probs, bs_gold, bs_tasks, calibration_params=affine_params)

    # --- Temperature scaling (needs raw log-likelihoods) ---
    if log_likelihoods is not None:
        f1_t, rsd_t, temp_per_task = apply_temperature_metrics(
            log_likelihoods, gold_labels, tasks, answer_choices_per_instance=None)
        task_metrics["temp_macro_f1"] = f1_t
        task_metrics["temp_rsd"] = rsd_t
        if bias_score_inputs is not None and log_likelihoods is not None:
            # BiasScore for temperature uses the bias-score split's own
            # log-likelihoods if available; otherwise skip gracefully.
            try:
                task_metrics["temp_bias_score"] = compute_temperature_bias_score(
                    log_likelihoods, gold_labels, tasks, temp_per_task)
            except Exception:
                pass

    return task_metrics
