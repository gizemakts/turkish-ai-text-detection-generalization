"""Fail-closed validation for the externally installed Stanza Turkish tokenizer."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from typing import Any


LOCK_FILENAME = "stanza_tr_tokenizer_lock.json"
LOCKED_IDENTITY = {
    "stanza_version": "1.14.0",
    "language": "tr",
    "processor": "tokenize",
    "package": "imst",
    "model_filename": "imst.pt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stanza_lock(lock_path: Path) -> dict[str, Any]:
    if not lock_path.is_file():
        raise RuntimeError(f"Required Stanza resource lock is missing: {lock_path}")
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Stanza resource lock is unreadable: {lock_path}") from exc
    required = {
        "schema_version", "resource_type", "stanza_version", "language",
        "processor", "package", "resources_json_sha256", "model_filename",
        "model_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise RuntimeError("Stanza resource lock has an invalid schema")
    if payload["schema_version"] != 1 or payload["resource_type"] != "stanza_tokenizer_lock":
        raise RuntimeError("Unsupported Stanza resource lock")
    for key, expected in LOCKED_IDENTITY.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"Stanza resource lock has unexpected {key}: "
                f"expected {expected!r}, got {payload.get(key)!r}"
            )
    for key in ("resources_json_sha256", "model_sha256"):
        value = payload[key]
        if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise RuntimeError(f"Stanza resource lock has an invalid {key}")
    return payload


def verify_stanza_resources(lock_path: Path) -> dict[str, Any]:
    """Validate package and exact local bytes before a scientific tokenizer is built."""
    lock = load_stanza_lock(lock_path)
    installed = metadata.version("stanza")
    if installed != lock["stanza_version"]:
        raise RuntimeError(
            f"Stanza version mismatch: expected {lock['stanza_version']}, got {installed}"
        )

    from stanza.resources.common import DEFAULT_MODEL_DIR

    resource_root = Path(DEFAULT_MODEL_DIR)
    resources_json = resource_root / "resources.json"
    model_path = (
        resource_root / lock["language"] / lock["processor"]
        / f"{lock['package']}.pt"
    )
    if not resources_json.is_file():
        raise RuntimeError(f"Locked Stanza resources inventory is missing: {resources_json}")
    if not model_path.is_file():
        raise RuntimeError(f"Locked Stanza tokenizer model is missing: {model_path}")

    resource_hash = sha256_file(resources_json)
    model_hash = sha256_file(model_path)
    if resource_hash != lock["resources_json_sha256"]:
        raise RuntimeError(
            "Stanza resources.json SHA-256 mismatch: "
            f"expected {lock['resources_json_sha256']}, got {resource_hash}"
        )
    if model_hash != lock["model_sha256"]:
        raise RuntimeError(
            "Stanza tokenizer SHA-256 mismatch: "
            f"expected {lock['model_sha256']}, got {model_hash}"
        )

    try:
        resources_payload = json.loads(resources_json.read_text(encoding="utf-8"))
        selected_package = resources_payload[lock["language"]]["default_processors"][lock["processor"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Locked Stanza resources inventory cannot select the Turkish tokenizer") from exc
    if selected_package != lock["package"]:
        raise RuntimeError(
            "Stanza resources inventory selects a different Turkish tokenizer package: "
            f"expected {lock['package']}, got {selected_package}"
        )

    return {
        "schema_version": 1,
        "stanza_version": installed,
        "language": lock["language"],
        "processor": lock["processor"],
        "package": lock["package"],
        "resources_json_sha256": resource_hash,
        "model_filename": model_path.name,
        "model_sha256": model_hash,
        "lock_sha256": sha256_file(lock_path),
    }
