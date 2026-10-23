# -*- coding: utf-8 -*-
"""Read-only LIT2026 retained-checkpoint identity gate for EXTERNAL_CROSS_CORPUS_V2.

This verifier performs NO training, NO inference, NO checkpoint selection, and NO
external-data access. It validates the 15 pre-bound retained LIT2026 cells using
only filesystem identity, sidecar metadata, and retained internal-prediction
identity.

It is intentionally suitable for DRAFT_FOR_REVIEW. Final PREEXECUTION_FREEZE
should later bind this file's SHA-256.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_PROTOCOL_ID = "EXTERNAL_CROSS_CORPUS_V2"
EXPECTED_METHOD = "LIT2026_REIMPL_V1_1"
EXPECTED_SEEDS = (42, 314, 2026)
EXPECTED_OUTER_FOLDS = (0, 1, 2, 3, 4)
EXPECTED_CELL_COUNT = 15
EXPECTED_TEST_ROWS = 600
EXPECTED_TEST_PAIR_COUNT = 300

DEFAULT_BINDING_REL = (
    "provenance/experiments/external_cross_corpus_v2/"
    "LIT2026_CHECKPOINT_BINDING.json"
)


def _root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".gitignore").is_file():
            return candidate
    raise RuntimeError("project root not found")


ROOT = _root()
DEFAULT_BINDING = ROOT / DEFAULT_BINDING_REL


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pair_hash(values) -> str:
    return hashlib.sha256(
        "\n".join(sorted(map(str, values))).encode("utf-8")
    ).hexdigest()


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing JSON file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON: {path}: {exc}")
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def require_file(rel: str, purpose: str) -> Path:
    require(isinstance(rel, str) and rel.strip(), f"{purpose} path is missing")
    path = ROOT / rel
    require(path.is_file(), f"{purpose} missing: {rel}")
    return path


def verify_hash(path: Path, expected: str, purpose: str) -> str:
    require(
        isinstance(expected, str) and len(expected) == 64,
        f"{purpose} expected SHA-256 is invalid",
    )
    observed = sha256_file(path)
    require(
        observed == expected.lower(),
        f"{purpose} SHA-256 mismatch: expected={expected.lower()} observed={observed}",
    )
    return observed


def basename_from_any_path(value: str) -> str:
    # Historical sidecars may contain Windows absolute paths even when verifier
    # runs under a different path parser. Normalize both separators explicitly.
    return value.replace("\\", "/").rstrip("/").split("/")[-1]


def read_prediction_identity(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"text_id", "pair_id"}
        missing = sorted(required - set(fields))
        require(not missing, f"prediction schema missing columns {missing}: {path}")

        rows = 0
        text_ids: list[str] = []
        pair_ids: list[str] = []
        for row in reader:
            rows += 1
            text_ids.append(str(row["text_id"]))
            pair_ids.append(str(row["pair_id"]))

    require(rows == EXPECTED_TEST_ROWS, f"unexpected prediction row count {rows}: {path}")
    require(
        len(set(text_ids)) == EXPECTED_TEST_ROWS,
        f"prediction text_id values are not unique: {path}",
    )
    unique_pairs = set(pair_ids)
    require(
        len(unique_pairs) == EXPECTED_TEST_PAIR_COUNT,
        f"unexpected test pair count {len(unique_pairs)}: {path}",
    )
    return {
        "rows": rows,
        "unique_text_ids": len(set(text_ids)),
        "unique_pair_ids": len(unique_pairs),
        "test_pair_sha256": pair_hash(unique_pairs),
    }


def extract_shared_revision(shared: dict[str, Any]) -> str | None:
    value = shared.get("model_and_tokenizer_revision")
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        for key in ("revision", "model_revision", "tokenizer_revision"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
    return None


def verify_shared_bound_files(
    shared: dict[str, Any],
    *,
    skip_self_binding: bool,
) -> dict[str, Any]:
    """Verify declared path/SHA pairs in shared_inference_contract.

    The verifier's own hash can be skipped during initial draft installation,
    because the binding must be updated only after this file exists. Final
    freeze verification should run without --skip-self-binding.
    """
    checks: dict[str, Any] = {}
    for key, value in sorted(shared.items()):
        if not key.endswith("_sha256"):
            continue
        base = key[: -len("_sha256")]
        path_value = shared.get(base)
        if not isinstance(path_value, str) or not path_value:
            continue

        if base == "current_verifier" and skip_self_binding:
            checks[base] = {
                "status": "SKIPPED_DRAFT_SELF_BINDING",
                "path": path_value,
            }
            continue

        path = require_file(path_value, f"shared binding {base}")
        observed = verify_hash(path, value, f"shared binding {base}")
        checks[base] = {
            "status": "PASS",
            "path": path_value,
            "sha256": observed,
        }
    return checks


def verify_cell(
    cell: dict[str, Any],
    *,
    expected_model: str | None,
    expected_revision: str | None,
) -> dict[str, Any]:
    seed = cell.get("seed")
    outer = cell.get("outer_fold")
    require(seed in EXPECTED_SEEDS, f"invalid bound seed: {seed!r}")
    require(outer in EXPECTED_OUTER_FOLDS, f"invalid bound outer fold: {outer!r}")

    tag = f"seed_{seed}_outer_{outer}"

    model_path = require_file(cell.get("current_model"), f"{tag} model")
    config_path = require_file(cell.get("checkpoint_config"), f"{tag} checkpoint config")
    sidecar_path = require_file(cell.get("run_sidecar"), f"{tag} run sidecar")
    prediction_path = require_file(
        cell.get("retained_internal_prediction"), f"{tag} retained internal prediction"
    )

    model_sha = verify_hash(
        model_path, cell.get("current_model_sha256"), f"{tag} model"
    )
    config_sha = verify_hash(
        config_path, cell.get("checkpoint_config_sha256"), f"{tag} checkpoint config"
    )
    sidecar_sha = verify_hash(
        sidecar_path, cell.get("run_sidecar_sha256"), f"{tag} run sidecar"
    )
    prediction_sha = verify_hash(
        prediction_path,
        cell.get("retained_internal_prediction_sha256"),
        f"{tag} retained internal prediction",
    )

    sidecar = load_json(sidecar_path)
    checkpoint_config = load_json(config_path)

    require(sidecar.get("seed") == seed, f"{tag} sidecar seed mismatch")
    require(sidecar.get("outer_fold") == outer, f"{tag} sidecar outer_fold mismatch")
    require(
        sidecar.get("prediction_sha256") == prediction_sha,
        f"{tag} sidecar prediction_sha256 mismatch",
    )
    require(
        sidecar.get("outer_test_rows_used_for_training") == 0,
        f"{tag} outer test rows were used for training",
    )
    require(
        sidecar.get("outer_test_used_for_model_selection") is False,
        f"{tag} outer test was used for model selection",
    )

    require(
        sidecar.get("test_pair_count") == EXPECTED_TEST_PAIR_COUNT,
        f"{tag} sidecar test_pair_count mismatch",
    )

    historical = cell.get("historical_checkpoint")
    require(
        isinstance(historical, str) and historical,
        f"{tag} historical_checkpoint is missing",
    )
    expected_checkpoint_basename = basename_from_any_path(historical)
    current_checkpoint_basename = model_path.parent.name
    sidecar_best = sidecar.get("best_model_checkpoint")
    require(
        isinstance(sidecar_best, str) and sidecar_best,
        f"{tag} sidecar best_model_checkpoint missing",
    )
    sidecar_checkpoint_basename = basename_from_any_path(sidecar_best)

    require(
        current_checkpoint_basename == expected_checkpoint_basename,
        f"{tag} current checkpoint basename mismatch: "
        f"{current_checkpoint_basename!r} != {expected_checkpoint_basename!r}",
    )
    require(
        sidecar_checkpoint_basename == expected_checkpoint_basename,
        f"{tag} sidecar checkpoint basename mismatch: "
        f"{sidecar_checkpoint_basename!r} != {expected_checkpoint_basename!r}",
    )

    prediction_identity = read_prediction_identity(prediction_path)
    require(
        prediction_identity["test_pair_sha256"] == sidecar.get("test_pair_sha256"),
        f"{tag} retained prediction test-pair identity mismatch",
    )

    model = sidecar.get("model")
    revision = sidecar.get("model_revision")
    require(isinstance(model, str) and model, f"{tag} sidecar model missing")
    require(isinstance(revision, str) and revision, f"{tag} sidecar model_revision missing")

    if expected_model is not None:
        require(model == expected_model, f"{tag} model mismatch: {model!r}")
    if expected_revision is not None:
        require(
            revision == expected_revision,
            f"{tag} model/tokenizer revision mismatch: {revision!r}",
        )

    require(
        checkpoint_config.get("architectures") == ["BertForSequenceClassification"],
        f"{tag} unexpected checkpoint architecture",
    )
    require(
        checkpoint_config.get("model_type") == "bert",
        f"{tag} unexpected checkpoint model_type",
    )

    for key in (
        "dataset_sha256",
        "fold_sha256",
        "config_sha256",
        "runner_sha256",
        "preexecution_freeze_sha256",
        "train_pair_sha256",
        "validation_pair_sha256",
        "test_pair_sha256",
    ):
        value = sidecar.get(key)
        require(
            isinstance(value, str) and len(value) == 64,
            f"{tag} missing/invalid sidecar identity field: {key}",
        )

    return {
        "seed": seed,
        "outer_fold": outer,
        "status": "PASS",
        "checkpoint_basename": expected_checkpoint_basename,
        "model_sha256": model_sha,
        "checkpoint_config_sha256": config_sha,
        "run_sidecar_sha256": sidecar_sha,
        "retained_internal_prediction_sha256": prediction_sha,
        "test_pair_sha256": prediction_identity["test_pair_sha256"],
        "test_rows": prediction_identity["rows"],
        "test_pairs": prediction_identity["unique_pair_ids"],
        "model": model,
        "model_revision": revision,
        "dataset_sha256": sidecar["dataset_sha256"],
        "fold_sha256": sidecar["fold_sha256"],
        "historical_config_sha256": sidecar["config_sha256"],
        "historical_runner_sha256": sidecar["runner_sha256"],
        "preexecution_freeze_sha256": sidecar["preexecution_freeze_sha256"],
    }


def assert_lit2026_identity(
    binding_path: Path = DEFAULT_BINDING,
    *,
    skip_self_binding: bool = False,
) -> dict[str, Any]:
    binding = load_json(binding_path)

    require(
        binding.get("protocol_id") == EXPECTED_PROTOCOL_ID,
        "unexpected protocol_id",
    )
    require(binding.get("method") == EXPECTED_METHOD, "unexpected method")
    require(
        binding.get("checkpoint_count") == EXPECTED_CELL_COUNT,
        "binding checkpoint_count mismatch",
    )
    require(
        binding.get("status") in {"DRAFT_FOR_REVIEW", "FROZEN_READY_TO_RUN"},
        f"unexpected binding status: {binding.get('status')!r}",
    )

    cells = binding.get("bindings")
    require(isinstance(cells, list), "binding 'bindings' must be a list")
    require(len(cells) == EXPECTED_CELL_COUNT, "binding must contain exactly 15 cells")
    require(all(isinstance(cell, dict) for cell in cells), "every binding cell must be an object")

    observed_grid = [(cell.get("seed"), cell.get("outer_fold")) for cell in cells]
    expected_grid = [
        (seed, outer)
        for seed in EXPECTED_SEEDS
        for outer in EXPECTED_OUTER_FOLDS
    ]
    require(
        sorted(observed_grid) == sorted(expected_grid),
        f"bound seed×fold grid mismatch: {observed_grid}",
    )
    require(
        len(set(observed_grid)) == EXPECTED_CELL_COUNT,
        "duplicate seed×fold cells in binding",
    )

    shared = binding.get("shared_inference_contract")
    require(isinstance(shared, dict), "shared_inference_contract missing")

    expected_model = shared.get("model")
    if not isinstance(expected_model, str) or not expected_model:
        expected_model = None
    expected_revision = extract_shared_revision(shared)

    shared_file_checks = verify_shared_bound_files(
        shared,
        skip_self_binding=skip_self_binding,
    )

    reports = [
        verify_cell(
            cell,
            expected_model=expected_model,
            expected_revision=expected_revision,
        )
        for cell in cells
    ]

    # All 15 cells must agree on study-wide identities. This prevents a mixed
    # retained grid even when each individual file hash matches its own entry.
    for key in (
        "model",
        "model_revision",
        "dataset_sha256",
        "fold_sha256",
        "historical_config_sha256",
        "historical_runner_sha256",
        "preexecution_freeze_sha256",
    ):
        values = {row[key] for row in reports}
        require(len(values) == 1, f"cross-cell identity mismatch for {key}: {sorted(values)}")

    return {
        "schema_version": "external-cross-corpus-v2-lit2026-identity-verification-v1",
        "status": "PASS_LIT2026_PRE_EXTERNAL_IDENTITY",
        "created_utc": utc_now(),
        "binding_path": binding_path.relative_to(ROOT).as_posix(),
        "binding_sha256": sha256_file(binding_path),
        "binding_status": binding.get("status"),
        "method": binding.get("method"),
        "checkpoint_count": len(reports),
        "grid": {
            "seeds": list(EXPECTED_SEEDS),
            "outer_folds": list(EXPECTED_OUTER_FOLDS),
            "cells": EXPECTED_CELL_COUNT,
        },
        "shared_bound_file_checks": shared_file_checks,
        "cross_cell_identity": {
            key: reports[0][key]
            for key in (
                "model",
                "model_revision",
                "dataset_sha256",
                "fold_sha256",
                "historical_config_sha256",
                "historical_runner_sha256",
                "preexecution_freeze_sha256",
            )
        },
        "cells": reports,
        "scope": (
            "Read-only structural identity gate. No checkpoint selection, "
            "training, inference, rescoring, or external-data access."
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--write-report", type=Path)
    parser.add_argument(
        "--skip-self-binding",
        action="store_true",
        help=(
            "Draft-installation only: skip current_verifier/current_verifier_sha256 "
            "self-binding until the binding JSON is updated with this file's final hash."
        ),
    )
    args = parser.parse_args()

    report = assert_lit2026_identity(
        args.binding.resolve(),
        skip_self_binding=args.skip_self_binding,
    )
    if args.write_report is not None:
        write_json(args.write_report.resolve(), report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
