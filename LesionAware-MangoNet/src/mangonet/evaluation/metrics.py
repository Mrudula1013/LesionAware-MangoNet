"""Manuscript metrics; 15 equal-width calibration confidence bins."""
import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, matthews_corrcoef, cohen_kappa_score,
                             log_loss)


def ece_score(labels, probabilities, bins=15):
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or len(probs) != len(y):
        raise ValueError('labels and probability matrix shapes must match')
    if len(y) == 0:
        return float('nan')
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    calibration = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        selection = (confidence >= lo) & ((confidence <= hi) if hi == 1.0 else (confidence < hi))
        if np.any(selection):
            calibration += np.mean(selection) * abs(correct[selection].mean() - confidence[selection].mean())
    return float(calibration)


def brier_score(labels, probabilities, classes=8):
    probs = np.asarray(probabilities, dtype=np.float64)
    targets = np.eye(classes)[np.asarray(labels, dtype=int)]
    return float(np.mean(np.sum((probs - targets) ** 2, axis=1)))


def classification_metrics(labels, predictions=None, probabilities=None, num_classes=8):
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) == 0:
        raise ValueError('Cannot evaluate zero images')
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        probabilities = np.clip(probabilities, 1e-12, 1.0)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        if predictions is None:
            predictions = probabilities.argmax(axis=1)
    if predictions is None:
        raise ValueError('Predictions or probabilities required')
    pred = np.asarray(predictions, dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, pred, labels=list(range(num_classes)), zero_division=0)
    out = {'n': len(labels), 'accuracy': float(accuracy_score(labels, pred)),
           'macro_precision': float(np.mean(precision)),
           'macro_recall': float(np.mean(recall)), 'macro_f1': float(np.mean(f1)),
           'mcc': float(matthews_corrcoef(labels, pred)),
           'kappa': float(cohen_kappa_score(labels, pred)),
           'per_class_precision': precision.tolist(), 'per_class_recall': recall.tolist(),
           'per_class_f1': f1.tolist(), 'per_class_support': support.astype(int).tolist(),
           'confusion_matrix': confusion_matrix(labels, pred, labels=list(range(num_classes))).tolist()}
    if probabilities is not None:
        out.update({'ece': ece_score(labels, probabilities),
                    'brier': brier_score(labels, probabilities, num_classes),
                    'nll': float(log_loss(labels, probabilities, labels=list(range(num_classes))))})
    return out


def stats_from_seeds(values):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        raise ValueError('No seed results')
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else float('nan')
    ci = float(scipy_stats.t.ppf(0.975, len(arr) - 1) * sd / np.sqrt(len(arr))) if len(arr) > 1 else float('nan')
    return {'n': len(arr), 'mean': float(arr.mean()), 'sd': sd,
            'ci95_lower': float(arr.mean() - ci), 'ci95_upper': float(arr.mean() + ci)}


def paired_significance(comparator, proposed):
    old = np.asarray(comparator, dtype=np.float64)
    new = np.asarray(proposed, dtype=np.float64)
    if old.shape != new.shape or len(old) < 2:
        raise ValueError('Paired tests require >=2 seed-matched results')
    result = scipy_stats.ttest_rel(new, old)
    delta = new - old
    sd = float(np.std(delta, ddof=1))
    return {'paired_t': float(result.statistic), 'p_value': float(result.pvalue),
            'mean_delta': float(delta.mean()), 'effect_size_dz': float(delta.mean() / sd) if sd else None,
            'n_pairs': len(delta)}


def holm_adjust(p_values):
    arr = np.asarray(p_values, dtype=np.float64)
    indices = np.argsort(arr)
    adjusted = np.zeros(len(arr))
    prev = 0.0
    for rank, idx in enumerate(indices):
        adjusted[idx] = min(1.0, max(prev, arr[idx] * (len(arr) - rank)))
        prev = adjusted[idx]
    return adjusted.tolist()
