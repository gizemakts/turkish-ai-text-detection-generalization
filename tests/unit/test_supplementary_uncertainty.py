import numpy as np
import pytest

from turkish_ai_detection.experiments.external_cross_corpus.supplementary_uncertainty import (
    metrics,
    sample_unit_indices,
    verify_inner_and_parameters,
)


def _thresholds():
    return {f"{alpha:.2f}": {"threshold": 0.5} for alpha in (0.01, 0.02, 0.03, 0.05)}


@pytest.mark.private_evidence
def test_validation_rechecks_all_realizations_and_sealed_cells():
    report = verify_inner_and_parameters()
    assert report["status"] == "PASS"
    assert report["calibration_rows_checked"] == 30
    assert report["sealed_cells_checked"] == 135
    assert report["thresholds_checked"] == 120


def test_metrics_rejects_missing_class_bootstrap_sample():
    with pytest.raises(RuntimeError, match="both classes"):
        metrics(np.zeros(4, dtype=int), np.full(4, 0.2), _thresholds())


def test_class_stratified_resampling_is_paired_and_reproducible():
    units = [np.array([0, 1]), np.array([2]), np.array([3, 4])]
    unit_labels = np.array([0, 0, 1])
    row_labels = np.array([0, 0, 0, 1, 1])
    left = sample_unit_indices(np.random.default_rng(2604), units, unit_labels, True)
    right = sample_unit_indices(np.random.default_rng(2604), units, unit_labels, True)
    assert np.array_equal(left, right)
    assert set(row_labels[left]) == {0, 1}


def test_unit_clusters_are_not_split_by_resampling():
    units = [np.array([0, 1]), np.array([2, 3])]
    labels = np.array([0, 1, 0, 1])
    sampled = sample_unit_indices(np.random.default_rng(7), units, labels, False)
    sampled_set = set(sampled.tolist())
    for unit in units:
        assert not (sampled_set.intersection(unit.tolist()) and
                    not set(unit.tolist()).issubset(sampled_set))
