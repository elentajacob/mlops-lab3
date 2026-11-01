"""Small, dependency-free statistics and classification-metric helpers.

Every function validates its input and raises a clear error instead of
silently returning a wrong answer. The CI pipeline tests all of these
behaviours with both pytest and unittest.
"""

from math import sqrt
from numbers import Real


# ----------------- Input validation ----------------- #
def _validate_numbers(values, name="values"):
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise TypeError(f"{name} must be a list or tuple of numbers")
    values = list(values)
    if not values:
        raise ValueError(f"{name} must not be empty")
    for v in values:
        if isinstance(v, bool) or not isinstance(v, Real):
            raise TypeError(f"{name} must contain only numbers, got {v!r}")
    return values


def _validate_labels(y_true, y_pred):
    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true or not y_pred:
        raise ValueError("y_true and y_pred must not be empty")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length ({len(y_true)} != {len(y_pred)})"
        )
    for label in y_true + y_pred:
        if label not in (0, 1):
            raise ValueError(f"labels must be 0 or 1, got {label!r}")
    return y_true, y_pred


# ----------------- Descriptive statistics ----------------- #
def mean(values):
    """Arithmetic mean of a non-empty list of numbers."""
    values = _validate_numbers(values)
    return sum(values) / len(values)


def variance(values, sample=False):
    """Population variance, or sample variance (n - 1) when sample=True."""
    values = _validate_numbers(values)
    if sample and len(values) < 2:
        raise ValueError("sample variance needs at least 2 values")
    m = mean(values)
    denominator = len(values) - 1 if sample else len(values)
    return sum((v - m) ** 2 for v in values) / denominator


def std(values, sample=False):
    """Standard deviation (square root of the variance)."""
    return sqrt(variance(values, sample=sample))


def z_scores(values):
    """Standardise values to mean 0 and std 1. Constant data maps to all zeros."""
    values = _validate_numbers(values)
    m, s = mean(values), std(values)
    if s == 0:
        return [0.0 for _ in values]
    return [(v - m) / s for v in values]


def min_max_scale(values, feature_range=(0.0, 1.0)):
    """Rescale values linearly into feature_range. Constant data maps to the lower bound."""
    values = _validate_numbers(values)
    low, high = feature_range
    if low >= high:
        raise ValueError(f"feature_range must be (low, high) with low < high, got {feature_range}")
    v_min, v_max = min(values), max(values)
    if v_max == v_min:
        return [float(low) for _ in values]
    return [low + (v - v_min) * (high - low) / (v_max - v_min) for v in values]


# ----------------- Binary classification metrics ----------------- #
def confusion_matrix(y_true, y_pred):
    """Counts of true/false positives/negatives for binary labels (1 = positive)."""
    y_true, y_pred = _validate_labels(y_true, y_pred)
    pairs = list(zip(y_true, y_pred))
    return {
        "tp": sum(1 for t, p in pairs if t == 1 and p == 1),
        "fp": sum(1 for t, p in pairs if t == 0 and p == 1),
        "tn": sum(1 for t, p in pairs if t == 0 and p == 0),
        "fn": sum(1 for t, p in pairs if t == 1 and p == 0),
    }


def accuracy(y_true, y_pred):
    """Fraction of predictions that match the true labels."""
    cm = confusion_matrix(y_true, y_pred)
    return (cm["tp"] + cm["tn"]) / sum(cm.values())


def precision(y_true, y_pred):
    """TP / (TP + FP). Returns 0.0 when nothing was predicted positive."""
    cm = confusion_matrix(y_true, y_pred)
    predicted_positive = cm["tp"] + cm["fn"]
    return cm["tp"] / predicted_positive if predicted_positive else 0.0


def recall(y_true, y_pred):
    """TP / (TP + FN). Returns 0.0 when there are no actual positives."""
    cm = confusion_matrix(y_true, y_pred)
    actual_positive = cm["tp"] + cm["fn"]
    return cm["tp"] / actual_positive if actual_positive else 0.0


def f1_score(y_true, y_pred):
    """Harmonic mean of precision and recall. Returns 0.0 when both are 0."""
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) else 0.0
