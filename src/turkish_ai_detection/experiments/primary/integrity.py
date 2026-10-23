# -*- coding: utf-8 -*-
"""Active result/input identity checks shared by primary reporting commands.

These checks concern the *data, fold assignments and generated predictions*;
there is deliberately no historical Python-source checksum gate.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from turkish_ai_detection.data.study import sha256_file

EXPERIMENT = "COMBINED_V2_PRIMARY_ABLATION_V1"
DATASET_SHA256 = "ea5797ac19348957472626d1a23b740de531cd19577f613e1a73894e34d66b44"
FOLD_SHA256 = "16ffae53a5d9027a95e12f3975fd062f17038ef5bf8970a74dd3ef7339d9a100"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def pair_hash(values) -> str:
    """Match the pair identity fingerprint written by the training runner."""
    return hashlib.sha256(
        "\n".join(sorted(map(str, values))).encode("utf-8")
    ).hexdigest()


def load_active_folds(root: Path, cfg: dict) -> pd.DataFrame:
    """Verify and load public, text-free fold assignments, not private texts."""
    require(cfg.get("dataset_sha256") == DATASET_SHA256,
            "primary config dataset SHA-256 identity mismatch")
    require(cfg.get("fold_sha256") == FOLD_SHA256,
            "primary config fold SHA-256 identity mismatch")
    require(cfg.get("seeds") == [42, 137, 314, 2026, 3407],
            "primary config seed plan mismatch")
    require(cfg.get("outer_folds") == [0, 1, 2, 3, 4],
            "primary config outer-fold plan mismatch")
    require(cfg.get("planned_fits") == 50,
            "primary config expected fit count mismatch")
    fold_path = root / "data" / "study" / "nested_folds.csv"
    require(fold_path.is_file(), "missing active, text-free outer-fold assignment file")
    require(sha256_file(fold_path) == FOLD_SHA256,
            "active fold assignment SHA-256 identity mismatch")
    folds = pd.read_csv(fold_path)
    require({"pair_id", "outer_fold"} <= set(folds.columns),
            "active fold schema mismatch")
    require(len(folds) == 1500 and folds.pair_id.is_unique,
            "active fold pair count/IDs mismatch")
    require(set(folds.outer_fold.astype(int)) == {0, 1, 2, 3, 4},
            "active fold coverage mismatch")
    require(all((folds.outer_fold.astype(int) == outer).sum() == 300
                for outer in range(5)), "active fold pair balance mismatch")
    return folds


def verify_fit_integrity(*, root: Path, cfg: dict, config_path: Path,
                         folds: pd.DataFrame, objective: str, seed: int,
                         outer: int, sidecar: dict, prediction_path: Path) -> None:
    """Fail closed before a prediction file is consumed by any statistical path."""
    tag = f"{objective}/seed_{seed}/outer_{outer}"
    require(objective in {"ce", "supcon"}, f"invalid objective: {tag}")
    require(seed in cfg["seeds"] and outer in cfg["outer_folds"],
            f"invalid seed/fold: {tag}")
    require(sidecar.get("status") == "COMPLETE", f"incomplete fit: {tag}")
    require(sidecar.get("experiment") == EXPERIMENT, f"experiment mismatch: {tag}")
    require(sidecar.get("objective") == objective, f"objective mismatch: {tag}")
    require(sidecar.get("seed") == seed, f"seed mismatch: {tag}")
    require(sidecar.get("outer_fold") == outer, f"fold mismatch: {tag}")
    require(sidecar.get("condition") == "A", f"condition mismatch: {tag}")

    require(sidecar.get("dataset_sha256") == DATASET_SHA256,
            f"sidecar dataset SHA-256 identity mismatch: {tag}")
    require(sidecar.get("fold_sha256") == FOLD_SHA256,
            f"sidecar fold SHA-256 identity mismatch: {tag}")
    require(sidecar.get("config_sha256") == sha256_file(config_path),
            f"sidecar config SHA-256 identity mismatch: {tag}")

    test_pairs = set(folds.loc[folds.outer_fold.eq(outer), "pair_id"])
    train_pairs = set(folds.loc[folds.outer_fold.ne(outer), "pair_id"])
    require(sidecar.get("train_pair_count") == 1200,
            f"train pair count mismatch: {tag}")
    require(sidecar.get("test_pair_count") == 300,
            f"test pair count mismatch: {tag}")
    require(sidecar.get("train_test_pair_overlap") == 0,
            f"train/test overlap sidecar mismatch: {tag}")
    require(sidecar.get("train_pair_sha256") == pair_hash(train_pairs),
            f"train pair identity SHA-256 mismatch: {tag}")
    require(sidecar.get("test_pair_sha256") == pair_hash(test_pairs),
            f"test pair identity SHA-256 mismatch: {tag}")

    require(sidecar.get("outer_test_rows_used_for_training") == 0,
            f"outer test training-use mismatch: {tag}")
    require(sidecar.get("calibration_used") is False,
            f"unexpected calibration: {tag}")
    require(sidecar.get("threshold_used") is False,
            f"unexpected thresholding: {tag}")

    require(prediction_path.is_file(), f"missing prediction file: {tag}")
    digest = sidecar.get("prediction_sha256")
    require(isinstance(digest, str) and _DIGEST.fullmatch(digest) is not None,
            f"missing or malformed prediction SHA-256: {tag}")
    require(digest == sha256_file(prediction_path),
            f"prediction SHA-256 mismatch: {tag}")
