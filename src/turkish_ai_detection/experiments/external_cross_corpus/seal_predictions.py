# -*- coding: utf-8 -*-
"""Seal complete label-blind external prediction artifacts before unblinding.

This module does NOT read external labels/subgroups and does NOT compute metrics.

It validates the complete prespecified prediction grid:
    3 primary external corpora
    x 3 methods
    x 3 seeds
    x 5 outer folds
    = 135 prediction cells

For every cell it verifies:
- prediction CSV exists and has exactly the permitted pre-seal columns;
- row count and external_id/text_sha256 identity exactly match the bound primary
  external projection;
- method/seed/fold columns are constant and correct;
- raw scores are finite;
- sidecar exists and binds the prediction hash;
- sidecar explicitly records label-blind/no-performance execution.

Only when ALL 135 cells pass does it atomically write one complete manifest.
Labels/subgroups must not be joined until this manifest exists and is sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHODS = (
    "CE_ONLY_MATCHED",
    "SUPCON",
    "LIT2026_REIMPL_V1_1",
)
SEEDS = (42, 314, 2026)
OUTER_FOLDS = (0, 1, 2, 3, 4)
DATASET_IDS = (
    "renklier2026",
    "tum2024_tr",
    "bostancioglu2026_tr",
)

CONFIG_REL = "configs/external_cross_corpus.json"
FREEZE_REL = (
    "provenance/experiments/external_cross_corpus_v2/PREEXECUTION_FREEZE.json"
)
PRESEAL_ROOT_REL = "artifacts/predictions/external_cross_corpus/preseal"
SEAL_ROOT_REL = "provenance/experiments/external_cross_corpus_v2/prediction_seal"
SEAL_MANIFEST_NAME = "PRESEAL_PREDICTION_MANIFEST.json"

INPUT_COLUMNS = ["external_id", "text", "text_sha256"]
OUTPUT_COLUMNS = [
    "external_id",
    "text_sha256",
    "method",
    "seed",
    "outer_fold",
    "raw_score_or_logit",
]


def find_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / ".gitignore").is_file():
            return candidate
    raise RuntimeError("project root not found via .gitignore")


ROOT = find_root()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def iter_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from iter_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_dicts(value)


def freeze_contains_binding(
    freeze: dict[str, Any],
    path_rel: str,
    sha256: str,
) -> bool:
    return any(
        node.get("path") == path_rel and node.get("sha256") == sha256
        for node in iter_dicts(freeze)
    )


def verify_execution_freeze(config_path: Path) -> dict[str, str]:
    freeze_path = ROOT / FREEZE_REL
    require(
        freeze_path.is_file(),
        "prediction sealing is blocked before PREEXECUTION_FREEZE",
    )
    freeze = read_json(freeze_path)
    cfg = read_json(config_path)
    from turkish_ai_detection.experiments.external_cross_corpus.freeze_v2 import verify_freeze
    verify_freeze(config_path, freeze_path, required_code_path=Path(__file__).resolve())
    require(
        cfg.get("status") == "FROZEN_READY_TO_RUN"
        and cfg.get("execution_permitted") is True,
        "prediction sealing config is not frozen/approved",
    )

    own_path = Path(__file__).resolve()
    require(
        freeze_contains_binding(freeze, rel(own_path), sha256_file(own_path)),
        "PREEXECUTION_FREEZE does not bind this exact seal_predictions.py",
    )
    return {
        "path": rel(freeze_path),
        "sha256": sha256_file(freeze_path),
    }


def verify_config(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    cfg = read_json(config_path)
    evaluation = cfg.get("evaluation")
    require(isinstance(evaluation, dict), "evaluation config missing")

    require(
        evaluation.get("prediction_first_label_later") is True,
        "prediction_first_label_later must be true",
    )
    require(
        evaluation.get("prediction_input_columns_exact") == INPUT_COLUMNS,
        "prediction input contract changed",
    )
    require(
        evaluation.get("labels_and_subgroups_join_only_after_complete_manifest_sealed")
        is True,
        "post-seal-only label join contract changed",
    )

    allowed = evaluation.get("preseal_output_columns_allowed")
    require(isinstance(allowed, list), "preseal allowed-column contract missing")
    require(
        set(OUTPUT_COLUMNS).issubset(set(allowed)),
        "seal output contract is outside allowed pre-seal columns",
    )
    forbidden = set(evaluation.get("preseal_columns_forbidden", []))
    require(
        not forbidden.intersection(OUTPUT_COLUMNS),
        "seal contract intersects forbidden pre-seal columns",
    )

    bindings = cfg.get("external_data_bindings")
    require(isinstance(bindings, dict), "external_data_bindings missing")
    primary = bindings.get("primary")
    require(isinstance(primary, list), "primary external bindings missing")

    by_id: dict[str, dict[str, Any]] = {}
    for item in primary:
        require(isinstance(item, dict), "invalid primary binding")
        dataset_id = item.get("dataset_id")
        require(isinstance(dataset_id, str), "dataset_id missing")
        require(dataset_id not in by_id, f"duplicate dataset_id: {dataset_id}")
        by_id[dataset_id] = item

    require(
        set(by_id) == set(DATASET_IDS),
        f"unexpected primary dataset set: {sorted(by_id)}",
    )
    return cfg, by_id


def expected_projection(binding: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / binding["path"]
    require(path.is_file(), f"primary external source missing: {path}")
    require(
        sha256_file(path) == binding["sha256"],
        f"primary external source hash mismatch: {path}",
    )

    frame = pd.read_csv(path, usecols=["external_id", "text_sha256"])
    require(len(frame) == int(binding["rows"]), f"row count mismatch: {path}")
    require(frame["external_id"].notna().all(), f"null external_id: {path}")
    require(frame["text_sha256"].notna().all(), f"null text_sha256: {path}")
    require(frame["external_id"].astype(str).is_unique, f"duplicate external_id: {path}")

    frame["external_id"] = frame["external_id"].astype(str)
    frame["text_sha256"] = frame["text_sha256"].astype(str).str.lower()
    return frame


def prediction_paths(
    dataset_id: str,
    method: str,
    seed: int,
    outer_fold: int,
) -> tuple[Path, Path]:
    root = (
        ROOT
        / PRESEAL_ROOT_REL
        / dataset_id
        / method
        / f"seed_{seed}"
    )
    return root / f"outer_{outer_fold}.csv", root / f"outer_{outer_fold}.json"


def validate_cell(
    dataset_id: str,
    method: str,
    seed: int,
    outer_fold: int,
    expected: pd.DataFrame,
) -> dict[str, Any]:
    csv_path, sidecar_path = prediction_paths(
        dataset_id,
        method,
        seed,
        outer_fold,
    )

    require(csv_path.is_file(), f"missing prediction CSV: {csv_path}")
    require(sidecar_path.is_file(), f"missing prediction sidecar: {sidecar_path}")

    prediction = pd.read_csv(csv_path)
    require(
        list(prediction.columns) == OUTPUT_COLUMNS,
        f"prediction column contract mismatch: {csv_path}",
    )
    require(
        len(prediction) == len(expected),
        f"prediction row count mismatch: {csv_path}",
    )
    require(
        prediction["external_id"].astype(str).tolist()
        == expected["external_id"].tolist(),
        f"external_id order/identity mismatch: {csv_path}",
    )
    require(
        prediction["text_sha256"].astype(str).str.lower().tolist()
        == expected["text_sha256"].tolist(),
        f"text_sha256 order/identity mismatch: {csv_path}",
    )

    require(
        prediction["method"].astype(str).eq(method).all(),
        f"method column mismatch: {csv_path}",
    )
    require(
        pd.to_numeric(prediction["seed"], errors="raise").astype(int).eq(seed).all(),
        f"seed column mismatch: {csv_path}",
    )
    require(
        pd.to_numeric(
            prediction["outer_fold"],
            errors="raise",
        ).astype(int).eq(outer_fold).all(),
        f"outer_fold column mismatch: {csv_path}",
    )

    raw = pd.to_numeric(
        prediction["raw_score_or_logit"],
        errors="raise",
    ).to_numpy(dtype=float)
    require(np.isfinite(raw).all(), f"non-finite score: {csv_path}")

    forbidden = {
        "text",
        "label",
        "binary_label",
        "generator",
        "domain",
        "scenario",
        "generation_type",
        "subgroup_metadata",
    }
    require(
        not forbidden.intersection(prediction.columns),
        f"forbidden pre-seal column present: {csv_path}",
    )

    prediction_sha = sha256_file(csv_path)
    sidecar = read_json(sidecar_path)

    require(
        sidecar.get("status") == "COMPLETE_LABEL_BLIND_PRESEAL_INFERENCE",
        f"sidecar status mismatch: {sidecar_path}",
    )
    require(sidecar.get("dataset_id") == dataset_id, f"sidecar dataset mismatch: {sidecar_path}")
    require(sidecar.get("method") == method, f"sidecar method mismatch: {sidecar_path}")
    require(int(sidecar.get("seed")) == seed, f"sidecar seed mismatch: {sidecar_path}")
    require(
        int(sidecar.get("outer_fold")) == outer_fold,
        f"sidecar outer_fold mismatch: {sidecar_path}",
    )
    require(int(sidecar.get("rows")) == len(expected), f"sidecar row mismatch: {sidecar_path}")
    require(sidecar.get("external_labels_read") is False, f"label-blind flag failed: {sidecar_path}")
    require(sidecar.get("external_subgroups_read") is False, f"subgroup-blind flag failed: {sidecar_path}")
    require(sidecar.get("performance_metrics_computed") is False, f"performance flag failed: {sidecar_path}")
    require(sidecar.get("external_adaptation") is False, f"adaptation flag failed: {sidecar_path}")
    require(sidecar.get("calibration_applied") is False, f"calibration flag failed: {sidecar_path}")

    pred_record = sidecar.get("prediction")
    require(isinstance(pred_record, dict), f"sidecar prediction record missing: {sidecar_path}")
    require(pred_record.get("path") == rel(csv_path), f"sidecar prediction path mismatch: {sidecar_path}")
    require(pred_record.get("sha256") == prediction_sha, f"sidecar prediction hash mismatch: {sidecar_path}")

    return {
        "dataset_id": dataset_id,
        "method": method,
        "seed": seed,
        "outer_fold": outer_fold,
        "rows": int(len(prediction)),
        "prediction": {
            "path": rel(csv_path),
            "sha256": prediction_sha,
        },
        "sidecar": {
            "path": rel(sidecar_path),
            "sha256": sha256_file(sidecar_path),
        },
    }


def seal(config_path: Path) -> dict[str, Any]:
    _cfg, bindings = verify_config(config_path)
    freeze = verify_execution_freeze(config_path)

    seal_root = ROOT / SEAL_ROOT_REL
    manifest_path = seal_root / SEAL_MANIFEST_NAME
    require(
        not manifest_path.exists(),
        f"refusing to overwrite existing prediction seal: {manifest_path}",
    )

    projections = {
        dataset_id: expected_projection(bindings[dataset_id])
        for dataset_id in DATASET_IDS
    }

    cells: list[dict[str, Any]] = []
    for dataset_id in DATASET_IDS:
        for method in METHODS:
            for seed in SEEDS:
                for outer_fold in OUTER_FOLDS:
                    cells.append(
                        validate_cell(
                            dataset_id,
                            method,
                            seed,
                            outer_fold,
                            projections[dataset_id],
                        )
                    )

    expected_cells = len(DATASET_IDS) * len(METHODS) * len(SEEDS) * len(OUTER_FOLDS)
    require(len(cells) == expected_cells == 135, "complete-grid cardinality mismatch")

    manifest = {
        "schema_version": "external-cross-corpus-preseal-prediction-manifest-v1",
        "status": "SEALED_COMPLETE_LABEL_BLIND_PREDICTION_GRID",
        "created_utc": utc_now(),
        "config": {
            "path": rel(config_path),
            "sha256": sha256_file(config_path),
        },
        "preexecution_freeze": freeze,
        "seal_implementation": {
            "path": rel(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "grid": {
            "dataset_ids": list(DATASET_IDS),
            "methods": list(METHODS),
            "seeds": list(SEEDS),
            "outer_folds": list(OUTER_FOLDS),
            "expected_cells": expected_cells,
            "sealed_cells": len(cells),
        },
        "preseal_guarantees": {
            "external_labels_read_by_inference": False,
            "external_subgroups_read_by_inference": False,
            "performance_metrics_computed_preseal": False,
            "external_adaptation": False,
            "label_or_subgroup_join_permitted_only_after_this_manifest": True,
        },
        "cells": cells,
    }

    atomic_json(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "manifest_path": rel(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "sealed_cells": len(cells),
        "expected_cells": expected_cells,
    }


def validate_only(config_path: Path) -> dict[str, Any]:
    _cfg, bindings = verify_config(config_path)
    expected_cells = len(DATASET_IDS) * len(METHODS) * len(SEEDS) * len(OUTER_FOLDS)

    return {
        "status": "PASS_PREDICTION_SEAL_STATIC_CONTRACT",
        "config_path": rel(config_path),
        "config_sha256": sha256_file(config_path),
        "implementation_path": rel(Path(__file__)),
        "implementation_sha256": sha256_file(Path(__file__)),
        "dataset_ids": list(DATASET_IDS),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "outer_folds": list(OUTER_FOLDS),
        "expected_prediction_cells": expected_cells,
        "bound_primary_rows": {
            dataset_id: int(bindings[dataset_id]["rows"])
            for dataset_id in DATASET_IDS
        },
        "note": (
            "Validation-only mode does not read prediction files, external CSV "
            "content, labels/subgroups, checkpoints, or performance results, "
            "and does not require PREEXECUTION_FREEZE."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / CONFIG_REL,
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_path = config_path.resolve()

    result = validate_only(config_path) if args.validate_only else seal(config_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
