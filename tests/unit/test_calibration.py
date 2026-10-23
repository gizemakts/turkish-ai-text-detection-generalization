from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "src"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from turkish_ai_detection.data.study import np_umbrella_k  # noqa: E402
from turkish_ai_detection.evaluation.metrics import calibrated_score, fit_platt  # noqa: E402


def test_synthetic_platt_probabilities_and_np_threshold_ordering():
    """Pure synthetic check: no scientific predictions are read and no files are written."""
    rng = np.random.default_rng(4242)
    n = 1200
    human = rng.normal(-1.2, 1.0, n)
    ai = rng.normal(1.2, 1.0, n)
    scores = np.concatenate([human, ai])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

    calibrator = fit_platt(scores, y)
    probability = calibrated_score(calibrator, scores)

    assert np.isfinite(probability).all()
    assert np.all((probability > 0) & (probability < 1))

    human_probability = np.sort(probability[y == 0])[::-1]
    thresholds: dict[float, float] = {}
    observed_k: dict[float, int] = {}

    for alpha in (0.01, 0.02, 0.03, 0.05):
        k = np_umbrella_k(n, alpha=alpha, delta=0.05)
        threshold = float(human_probability[k])
        observed_k[alpha] = int(k)
        thresholds[alpha] = threshold
        assert int(np.sum(human_probability > threshold)) <= k

    assert observed_k == {0.01: 6, 0.02: 15, 0.03: 26, 0.05: 47}
    assert thresholds[0.01] >= thresholds[0.02] >= thresholds[0.03] >= thresholds[0.05]
