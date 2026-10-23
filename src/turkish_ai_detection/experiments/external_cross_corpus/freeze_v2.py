"""Deterministic External Cross-Corpus V2 freeze creation and verification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "external-cross-corpus-v2-preexecution-freeze-v1"
RUNTIME_FIELDS = {"status", "execution_permitted"}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_config_identity(config: dict[str, Any]) -> str:
    projected = json.loads(json.dumps(config))
    for field in RUNTIME_FIELDS:
        projected.pop(field, None)
    return sha256_bytes(canonical_json(projected))


def _iter_path_records(value: Any) -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value["path"], value["sha256"]
        for child in value.values():
            yield from _iter_path_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_path_records(child)


def _required_bindings(root: Path, config: dict[str, Any]) -> list[dict[str, str]]:
    records: dict[str, str] = {}
    for relative, declared in _iter_path_records(config):
        path = Path(relative)
        if path.suffix.lower() == ".csv" and not relative.startswith(("data/study/", "results/primary/")):
            continue
        if declared:
            if not (root / path).is_file():
                raise RuntimeError(f"declared scientific file is missing: {relative}")
            actual = sha256_file(root / path)
            if declared != actual:
                raise RuntimeError(f"declared scientific identity mismatch: {relative}")
            records[relative] = actual
    explicit = [
        "src/turkish_ai_detection/experiments/external_cross_corpus/rerun_fit.py",
        "src/turkish_ai_detection/experiments/external_cross_corpus/inference.py",
        "src/turkish_ai_detection/experiments/external_cross_corpus/seal_predictions.py",
        "src/turkish_ai_detection/experiments/external_cross_corpus/statistics.py",
        "src/turkish_ai_detection/experiments/external_cross_corpus/audit_semantic_equivalence.py",
        "src/turkish_ai_detection/experiments/external_cross_corpus/freeze_v2.py",
        "src/turkish_ai_detection/experiments/primary/run_fit.py",
        "src/turkish_ai_detection/experiments/model_development/supcon/train.py",
    ]
    for relative in explicit:
        path = root / relative
        if path.is_file():
            records[relative] = sha256_file(path)
    return [{"path": path, "sha256": digest} for path, digest in sorted(records.items())]


def build_freeze(contract_path: Path, output_path: Path, *, allow_draft: bool = False) -> dict[str, Any]:
    root = contract_path.resolve().parents[1]
    config = json.loads(contract_path.read_text(encoding="utf-8"))
    if not allow_draft and (
        config.get("status") != "FROZEN_READY_TO_RUN"
        or config.get("execution_permitted") is not True
    ):
        raise RuntimeError("production freeze requires FROZEN_READY_TO_RUN and execution_permitted=true")
    freeze = {
        "schema_version": SCHEMA,
        "status": "FROZEN_READY_TO_RUN",
        "execution_permitted": True,
        "contract": {
            "path": contract_path.resolve().relative_to(root).as_posix(),
            "scientific_identity_sha256": scientific_config_identity(config),
        },
        "bindings": _required_bindings(root, config),
        "hash_policy": {
            "runtime_config_fields_excluded": sorted(RUNTIME_FIELDS),
            "freeze_hash_excluded_from_own_payload": True,
        },
    }
    freeze["freeze_sha256"] = sha256_bytes(canonical_json(freeze))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return freeze


def verify_freeze(
    contract_path: Path,
    freeze_path: Path,
    *,
    require_execution: bool = True,
    required_code_path: Path | None = None,
) -> dict[str, Any]:
    root = contract_path.resolve().parents[1]
    config = json.loads(contract_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != SCHEMA:
        raise RuntimeError("invalid V2 freeze schema")
    expected_hash = dict(freeze)
    recorded_hash = expected_hash.pop("freeze_sha256", None)
    if recorded_hash != sha256_bytes(canonical_json(expected_hash)):
        raise RuntimeError("V2 freeze self-hash mismatch")
    if require_execution and (
        config.get("status") != "FROZEN_READY_TO_RUN"
        or config.get("execution_permitted") is not True
    ):
        raise RuntimeError("execution remains disabled by V2 contract")
    contract = freeze.get("contract", {})
    if contract.get("path") != contract_path.resolve().relative_to(root).as_posix():
        raise RuntimeError("V2 freeze contract path mismatch")
    if contract.get("scientific_identity_sha256") != scientific_config_identity(config):
        raise RuntimeError("V2 scientific config identity mismatch")
    bindings = {
        record.get("path"): record.get("sha256")
        for record in freeze.get("bindings", [])
        if isinstance(record, dict)
    }
    for relative, digest in bindings.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"V2 freeze binding mismatch: {relative}")
    if required_code_path is not None:
        relative = required_code_path.resolve().relative_to(root).as_posix()
        if bindings.get(relative) != sha256_file(required_code_path):
            raise RuntimeError(f"V2 freeze missing code binding: {relative}")
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--allow-draft-candidate", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify_freeze(
            args.contract.resolve(),
            args.freeze.resolve(),
            require_execution=not args.allow_draft_candidate,
        )
    else:
        result = build_freeze(
            args.contract.resolve(),
            args.freeze.resolve(),
            allow_draft=args.allow_draft_candidate,
        )
    print(json.dumps({"status": result["status"], "freeze_sha256": result["freeze_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
