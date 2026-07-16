import pandas as pd
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize_scalar
import os
import random
import argparse
import logging
from src.superni.utils.metrics import (
    f1, rsd, bias_score
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def compute_f1_and_rsd_per_task(probs, gold_labels, tasks, calibration_params=None, calibration_do_softmax=False):
    f1_per_task = dict()
    rsd_per_task = dict()
    all_tasks = np.unique(tasks)
    for task in all_tasks:
        task_gold_labels = gold_labels[tasks == task]
        # probs is a Series of dicts, expand it into a DataFrame
        task_probs = probs[tasks == task].apply(pd.Series)
        if calibration_params is not None:
            task_probs *= calibration_params[task]
            if calibration_do_softmax:
                task_probs = softmax(task_probs, axis=1)
            else:
                task_probs = task_probs / task_probs.sum(axis=1).values[:, np.newaxis]
        task_preds = task_probs.idxmax(axis=1).values
        f1_per_task[task] = f1(task_gold_labels, task_preds, average='macro', labels=np.unique(task_gold_labels))
        rsd_per_task[task] = rsd(task_gold_labels, task_preds)

    return f1_per_task, rsd_per_task


def compute_bias_score_per_task(probs, gold_labels, tasks, calibration_params=None, calibration_do_softmax=False):
    bias_score_per_task = dict()
    all_tasks = np.unique(tasks)
    for task in all_tasks:
        task_gold_labels = gold_labels[tasks == task]
        # probs is a Series of dicts, expand it into a DataFrame
        task_probs = probs[tasks == task].apply(pd.Series)
        if calibration_params is not None:
            task_probs *= calibration_params[task]
            if calibration_do_softmax:
                task_probs = softmax(task_probs, axis=1)
            else:
                task_probs = task_probs / task_probs.sum(axis=1).values[:, np.newaxis]
        bias_score_per_task[task] = bias_score(task_gold_labels, task_probs)
    return bias_score_per_task


def compute_task_calibration_parameters(probs, tasks, gold_labels=None):
    task_calibration_params = dict()
    all_tasks = np.unique(tasks)
    for task in all_tasks:
        task_probs = probs[tasks == task].apply(pd.Series)
        if gold_labels is None:
            mean_probs = task_probs.mean(axis=0).values
        else:
            task_gold_labels = gold_labels[tasks == task]
            mean_probs = np.mean([task_probs[task_gold_labels == label].mean(axis=0) for label in np.unique(task_gold_labels)], axis=0)
        task_calibration_params[task] = np.linalg.inv(np.identity(len(mean_probs)) * mean_probs).diagonal()
    return task_calibration_params


# ======================================================================
# Reviewer-requested baselines: affine logit correction + temperature
# scaling. Both are computed post-hoc from the already-saved outputs
# (probs and log_likelihoods), so they need no model forward passes.
# ======================================================================

def compute_affine_calibration_parameters(probs, tasks):
    """Per-task diagonal affine correction fit from the mean predictive
    distribution (a content-free-style label-prior estimate).

    Each label's probability is divided by the model's mean probability
    for that label, down-weighting a-priori over-predicted labels. This
    is the standard diagonal affine/vector-scaling baseline and the
    natural sibling of CC; cost is O(nc) like CC/DCC. Returned in the
    same {task: per-label multiplicative vector} form as the other
    methods, so it reuses the existing metric functions unchanged.
    """
    params = dict()
    for task in np.unique(tasks):
        task_probs = probs[tasks == task].apply(pd.Series)
        mean_probs = task_probs.mean(axis=0).values
        mean_probs = np.clip(mean_probs, 1e-8, None)  # guard zero-prob labels
        params[task] = 1.0 / mean_probs
    return params


def _stable_softmax(logits, T):
    z = np.asarray(logits, dtype=float) / T
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def _fit_task_temperature(task_lls, task_gold, labels):
    """Fit a single scalar temperature per task by minimising the NLL of
    the gold label over the task's instances. T in ~[0.1, 10]."""
    label_index = {lab: i for i, lab in enumerate(labels)}
    L = np.array([[inst[lab] for lab in labels] for inst in task_lls], dtype=float)
    gold_idx = np.array([label_index[g] for g in task_gold])

    def nll(logT):
        T = np.exp(logT)
        loss = 0.0
        for row, gi in zip(L, gold_idx):
            p = _stable_softmax(row, T)
            loss -= np.log(max(p[gi], 1e-12))
        return loss / len(L)

    res = minimize_scalar(nll, bounds=(np.log(0.1), np.log(10.0)), method="bounded")
    return float(np.exp(res.x))


def compute_temperature_metrics(log_likelihoods, gold_labels, tasks):
    """Fit per-task temperature and compute Macro-F1 and RSD under it.

    Temperature scaling rescales the log-likelihoods before the softmax,
    so it is applied here directly rather than as a multiplicative factor
    on probabilities. Being monotonic and label-symmetric, it adjusts
    confidence but cannot correct a systematic label-prior skew the way
    the prior-estimation methods (CC, DCC, LOOC, affine) can.

    Returns (f1_per_task, rsd_per_task, temp_per_task).
    """
    f1_per_task, rsd_per_task, temp_per_task = dict(), dict(), dict()
    for task in np.unique(tasks):
        mask = tasks == task
        task_lls = list(log_likelihoods[mask])
        task_gold = np.asarray(gold_labels[mask])
        labels = list(task_lls[0].keys())

        T = _fit_task_temperature(task_lls, task_gold, labels)
        temp_per_task[task] = T

        preds = []
        for inst in task_lls:
            p = _stable_softmax([inst[lab] for lab in labels], T)
            preds.append(labels[int(np.argmax(p))])
        preds = np.array(preds)

        f1_per_task[task] = f1(task_gold, preds, average="macro", labels=np.unique(task_gold))
        rsd_per_task[task] = rsd(task_gold, preds)
    return f1_per_task, rsd_per_task, temp_per_task


def compute_temperature_bias_score(log_likelihoods, gold_labels, tasks, temp_per_task):
    """BiasScore under temperature scaling, reusing the fitted T per task."""
    bias_per_task = dict()
    for task in np.unique(tasks):
        mask = tasks == task
        task_lls = list(log_likelihoods[mask])
        task_gold = np.asarray(gold_labels[mask])
        labels = list(task_lls[0].keys())
        T = temp_per_task.get(task, 1.0)

        rows = [_stable_softmax([inst[lab] for lab in labels], T) for inst in task_lls]
        task_probs = pd.DataFrame(rows, columns=labels).reset_index(drop=True)
        task_gold = pd.Series(task_gold).reset_index(drop=True)
        bias_per_task[task] = bias_score(task_gold, task_probs)
    return bias_per_task


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/mistralai/Mistral-7B-v0.1/superni/label_bias-def-pos-0/",
        help="The directory for output"
    )
    args = parser.parse_args()

    return args


def compute_metrics_for_experiment(output_dir):

    task_metrics = dict()

    # compute pre-calibration metrics
    pre_calibration_outputs = pd.read_pickle(os.path.join(output_dir, "full_outputs.pickle"))
    probs = pre_calibration_outputs['probs']
    gold_labels = pre_calibration_outputs['gold']
    tasks = pre_calibration_outputs['task']
    log_likelihoods = pre_calibration_outputs['log_likelihoods']
    task_metrics['macro_f1'], task_metrics['rsd'] = compute_f1_and_rsd_per_task(probs, gold_labels, tasks)

    # compute pre-calibration BiasScore
    has_bias_score_results = os.path.exists(os.path.join(output_dir, "bias_score", "full_outputs.pickle"))
    if has_bias_score_results:
        bias_score_outputs = pd.read_pickle(os.path.join(output_dir, "bias_score", "full_outputs.pickle"))
        bias_score_probs = bias_score_outputs['probs']
        bias_score_gold_labels = bias_score_outputs['gold']
        bias_score_tasks = bias_score_outputs['task']
        task_metrics['bias_score'] = compute_bias_score_per_task(bias_score_probs, bias_score_gold_labels, bias_score_tasks)

    # compute post-calibration metrics
    for calibration_method in ['cc', 'dc', 'looc']:
        if os.path.exists(os.path.join(output_dir, calibration_method, "full_outputs.pickle")):
            calibration_outputs = pd.read_pickle(
                os.path.join(output_dir, calibration_method, "full_outputs.pickle"))
            calibration_probs = calibration_outputs['probs']
            calibration_tasks = calibration_outputs['task']

            calibration_gold_labels = None
            if calibration_method in ['looc']:
                # calibration inputs have gold labels that can be used when estimating bias
                calibration_gold_labels = calibration_outputs['gold']

            calibration_params = \
                compute_task_calibration_parameters(calibration_probs, calibration_tasks, gold_labels=calibration_gold_labels)
            task_metrics[f'{calibration_method}_macro_f1'], task_metrics[f'{calibration_method}_rsd'] = \
                compute_f1_and_rsd_per_task(probs, gold_labels, tasks, calibration_params=calibration_params)
            if has_bias_score_results:
                task_metrics[f'{calibration_method}_bias_score'] = \
                    compute_bias_score_per_task(bias_score_probs, bias_score_gold_labels, bias_score_tasks,
                                                calibration_params=calibration_params)

    # --- Reviewer-requested baselines (post-hoc, no re-running needed) ---
    # Affine correction: diagonal rescale by the mean predictive distribution.
    affine_params = compute_affine_calibration_parameters(probs, tasks)
    task_metrics['affine_macro_f1'], task_metrics['affine_rsd'] = \
        compute_f1_and_rsd_per_task(probs, gold_labels, tasks, calibration_params=affine_params)
    if has_bias_score_results:
        affine_bias_params = compute_affine_calibration_parameters(bias_score_probs, bias_score_tasks)
        task_metrics['affine_bias_score'] = \
            compute_bias_score_per_task(bias_score_probs, bias_score_gold_labels, bias_score_tasks,
                                        calibration_params=affine_bias_params)

    # Temperature scaling: per-task scalar T fit on the saved log-likelihoods.
    task_metrics['temp_macro_f1'], task_metrics['temp_rsd'], temp_per_task = \
        compute_temperature_metrics(log_likelihoods, gold_labels, tasks)
    if has_bias_score_results and 'log_likelihoods' in bias_score_outputs:
        try:
            bs_temp_f1, bs_temp_rsd, bs_temp_T = \
                compute_temperature_metrics(bias_score_outputs['log_likelihoods'],
                                            bias_score_gold_labels, bias_score_tasks)
            task_metrics['temp_bias_score'] = \
                compute_temperature_bias_score(bias_score_outputs['log_likelihoods'],
                                               bias_score_gold_labels, bias_score_tasks, bs_temp_T)
        except Exception as e:
            logger.warning(f"Temperature BiasScore skipped: {e}")

    task_metrics = pd.DataFrame(task_metrics)
    metrics = task_metrics.mean()
    task_metrics.to_csv(os.path.join(output_dir, "task_metrics.csv"))
    metrics.to_json(os.path.join(output_dir, "mean_metrics.json"), indent=4)

    # print results
    metric_types = ['macro_f1', 'rsd', 'bias_score']
    metric_names = ['Macro-F1', 'RSD', 'BiasScore']
    for metric, metric_name in zip(metric_types, metric_names):
        logger.info(f"{metric_name}: {np.around(metrics[metric], 3)}")

    for calibration_method, method_name in zip(
            ['cc', 'dc', 'looc', 'affine', 'temp'],
            ['Contextual Calibration', 'Domain Contextual Calibration',
             'Leave-one-out Calibration', 'Affine Correction', 'Temperature Scaling']):
            for metric, metric_name in zip(metric_types, metric_names):
                if f'{calibration_method}_{metric}' in task_metrics:
                    logger.info(f"{method_name} - {metric_name}: {np.around(metrics[f'{calibration_method}_{metric}'], 3)}")


if __name__ == "__main__":

    args = parse_args()
    compute_metrics_for_experiment(args.output_dir)