"""Publication runner result checks, without requiring private texts or training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from turkish_ai_detection.data.study import sha256_file
from turkish_ai_detection.experiments.primary.integrity import (
    DATASET_SHA256, FOLD_SHA256, load_active_folds, pair_hash,
    verify_fit_integrity,
)
from turkish_ai_detection.experiments.primary import summarize_statistics, verify_results

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "primary_evaluation.json"
CFG = json.loads(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture
def first_fit(tmp_path):
    folds = load_active_folds(ROOT, CFG)
    pred = tmp_path / "ce" / "seed_42" / "outer_0_predictions.csv"
    pred.parent.mkdir(parents=True)
    pred.write_text("text_id,raw_decision_score\nexample,0.25\n", encoding="utf-8")
    train = set(folds.loc[folds.outer_fold.ne(0), "pair_id"])
    test = set(folds.loc[folds.outer_fold.eq(0), "pair_id"])
    run = {
        "status": "COMPLETE", "experiment": "COMBINED_V2_PRIMARY_ABLATION_V1",
        "objective": "ce", "seed": 42, "outer_fold": 0, "condition": "A",
        "dataset_sha256": DATASET_SHA256, "fold_sha256": FOLD_SHA256,
        "config_sha256": sha256_file(CONFIG),
        "train_pair_count": 1200, "test_pair_count": 300,
        "train_test_pair_overlap": 0,
        "train_pair_sha256": pair_hash(train), "test_pair_sha256": pair_hash(test),
        "outer_test_rows_used_for_training": 0,
        "calibration_used": False, "threshold_used": False,
        "prediction_sha256": sha256_file(pred),
    }
    sidecar = pred.parent / "outer_0_run.json"
    sidecar.write_text(json.dumps(run), encoding="utf-8")
    return folds, pred, sidecar, run


def _verify(first_fit):
    folds, pred, sidecar, run = first_fit
    verify_fit_integrity(
        root=ROOT, cfg=CFG, config_path=CONFIG, folds=folds,
        objective="ce", seed=42, outer=0, sidecar=run, prediction_path=pred,
    )


def test_valid_run_identity_and_prediction_checksum_pass(first_fit):
    _verify(first_fit)


@pytest.mark.parametrize("field, replacement", [
    ("dataset_sha256", "0" * 64),
    ("fold_sha256", "0" * 64),
    ("config_sha256", "0" * 64),
    ("train_pair_sha256", "0" * 64),
    ("test_pair_sha256", "0" * 64),
    ("train_pair_count", 1199),
    ("test_pair_count", 301),
    ("train_test_pair_overlap", 1),
    ("prediction_sha256", "0" * 64),
])
def test_corrupted_sidecar_binding_rejected(first_fit, field, replacement):
    first_fit[3][field] = replacement
    with pytest.raises(RuntimeError, match="mismatch"):
        _verify(first_fit)


def test_prediction_file_changed_after_sidecar_rejected(first_fit):
    first_fit[1].write_text("text_id,raw_decision_score\nexample,0.26\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="prediction SHA-256 mismatch"):
        _verify(first_fit)


def test_direct_statistics_entrypoint_rejects_tampered_predictions(first_fit, monkeypatch):
    _, pred, sidecar, run = first_fit
    pred.write_text("text_id,raw_decision_score\nexample,0.26\n", encoding="utf-8")
    monkeypatch.setattr(summarize_statistics, "OUT", pred.parents[2])
    with pytest.raises(RuntimeError, match="prediction SHA-256 mismatch"):
        summarize_statistics.load_all()


def test_direct_statistics_entrypoint_rejects_tampered_fold_identity(first_fit, monkeypatch):
    _, pred, sidecar, run = first_fit
    run["fold_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(run), encoding="utf-8")
    monkeypatch.setattr(summarize_statistics, "OUT", pred.parents[2])
    with pytest.raises(RuntimeError, match="sidecar fold SHA-256 identity mismatch"):
        summarize_statistics.load_all()


def test_primary_results_verifier_rejects_tampered_dataset_identity(first_fit, monkeypatch):
    _, pred, sidecar, run = first_fit
    run["dataset_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(run), encoding="utf-8")
    monkeypatch.setattr(verify_results, "OUT", pred.parents[2])
    with pytest.raises(RuntimeError, match="sidecar dataset SHA-256 identity mismatch"):
        verify_results.main()
