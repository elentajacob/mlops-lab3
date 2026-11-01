"""pytest suite: parametrized cases, fixtures, exception checks and approximate comparisons."""

import math

import pytest

from src import ml_metrics


# ----------------- Fixtures ----------------- #
@pytest.fixture
def predictions():
    # 10 binary labels -> TP=4, FP=1, TN=3, FN=2
    y_true = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 1, 0, 0, 1, 0, 0, 0]
    return y_true, y_pred


# ----------------- Descriptive statistics ----------------- #
@pytest.mark.parametrize(
    "values, expected",
    [
        ([1, 2, 3, 4], 2.5),
        ([5], 5.0),
        ([-2, 2], 0.0),
        ([0.1, 0.2, 0.3], 0.2),
        ((10, 20, 30), 20.0),
    ],
)
def test_mean(values, expected):
    assert ml_metrics.mean(values) == pytest.approx(expected)


@pytest.mark.parametrize(
    "values, sample, expected",
    [
        ([2, 4, 4, 4, 5, 5, 7, 9], False, 4.0),
        ([2, 4, 4, 4, 5, 5, 7, 9], True, 32 / 7),
        ([3, 3, 3], False, 0.0),
    ],
)
def test_variance(values, sample, expected):
    assert ml_metrics.variance(values, sample=sample) == pytest.approx(expected)


def test_std_is_sqrt_of_variance():
    values = [2, 4, 4, 4, 5, 5, 7, 9]
    assert ml_metrics.std(values) == pytest.approx(2.0)
    assert ml_metrics.std(values, sample=True) == pytest.approx(math.sqrt(32 / 7))


def test_sample_variance_needs_two_values():
    with pytest.raises(ValueError, match="at least 2"):
        ml_metrics.variance([5], sample=True)


def test_z_scores_have_mean_zero_and_std_one():
    z = ml_metrics.z_scores([10, 20, 30, 40, 50])
    assert ml_metrics.mean(z) == pytest.approx(0.0, abs=1e-12)
    assert ml_metrics.std(z) == pytest.approx(1.0)


def test_z_scores_constant_data_is_all_zeros():
    assert ml_metrics.z_scores([7, 7, 7]) == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "values, feature_range, expected",
    [
        ([0, 5, 10], (0.0, 1.0), [0.0, 0.5, 1.0]),
        ([0, 5, 10], (-1.0, 1.0), [-1.0, 0.0, 1.0]),
        ([4, 4, 4], (0.0, 1.0), [0.0, 0.0, 0.0]),
    ],
)
def test_min_max_scale(values, feature_range, expected):
    assert ml_metrics.min_max_scale(values, feature_range) == pytest.approx(expected)


def test_min_max_scale_rejects_bad_range():
    with pytest.raises(ValueError, match="low < high"):
        ml_metrics.min_max_scale([1, 2, 3], feature_range=(1.0, 0.0))


# ----------------- Input validation ----------------- #
@pytest.mark.parametrize("func", [ml_metrics.mean, ml_metrics.variance, ml_metrics.std, ml_metrics.z_scores])
def test_empty_input_raises(func):
    with pytest.raises(ValueError, match="must not be empty"):
        func([])


@pytest.mark.parametrize("bad", ["123", [1, "2", 3], [1, None], [True, False], 42])
def test_non_numeric_input_raises(bad):
    with pytest.raises(TypeError):
        ml_metrics.mean(bad)


# ----------------- Classification metrics ----------------- #
def test_confusion_matrix(predictions):
    assert ml_metrics.confusion_matrix(*predictions) == {"tp": 4, "fp": 1, "tn": 3, "fn": 2}


def test_metrics_on_known_example(predictions):
    assert ml_metrics.accuracy(*predictions) == pytest.approx(0.7)
    assert ml_metrics.precision(*predictions) == pytest.approx(0.8)
    assert ml_metrics.recall(*predictions) == pytest.approx(4 / 6)
    assert ml_metrics.f1_score(*predictions) == pytest.approx(2 * 0.8 * (4 / 6) / (0.8 + 4 / 6))


@pytest.mark.parametrize(
    "func", [ml_metrics.accuracy, ml_metrics.precision, ml_metrics.recall, ml_metrics.f1_score]
)
def test_perfect_predictions_score_one(func):
    labels = [0, 1, 1, 0, 1]
    assert func(labels, labels) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "y_true, y_pred, func",
    [
        ([0, 0, 1], [0, 0, 0], ml_metrics.precision),  # nothing predicted positive
        ([0, 0, 0], [0, 1, 0], ml_metrics.recall),  # no actual positives
        ([1, 1, 0], [0, 0, 0], ml_metrics.f1_score),  # precision and recall both 0
    ],
)
def test_zero_division_returns_zero(y_true, y_pred, func):
    assert func(y_true, y_pred) == 0.0


@pytest.mark.parametrize(
    "y_true, y_pred, message",
    [
        ([], [], "must not be empty"),
        ([1, 0], [1], "same length"),
        ([1, 2], [1, 0], "must be 0 or 1"),
    ],
)
def test_invalid_labels_raise(y_true, y_pred, message):
    with pytest.raises(ValueError, match=message):
        ml_metrics.accuracy(y_true, y_pred)
