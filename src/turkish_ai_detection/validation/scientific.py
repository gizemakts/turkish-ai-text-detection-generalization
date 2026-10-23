"""Read-only validation of finalized scientific results."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from turkish_ai_detection.paths import PROJECT_ROOT


@dataclass(frozen=True)
class ResultPackage:
    name: str
    directory: Path


PACKAGES = (
    ResultPackage("primary", PROJECT_ROOT / "results" / "primary" / "final_statistics"),
    ResultPackage("literature comparison", PROJECT_ROOT / "results" / "literature_comparison" / "final_statistics"),
    ResultPackage("generator holdout", PROJECT_ROOT / "results" / "generator_holdout" / "final_statistics"),
    ResultPackage("domain shift", PROJECT_ROOT / "results" / "domain_shift" / "final_statistics"),
    ResultPackage("artifact sensitivity", PROJECT_ROOT / "results" / "artifact_sensitivity" / "final_statistics"),
    ResultPackage("calibration", PROJECT_ROOT / "results" / "calibration" / "final_statistics"),
    ResultPackage("near-duplicate sensitivity", PROJECT_ROOT / "results" / "near_duplicate_sensitivity" / "final_statistics"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hash(value: object) -> str:
    return value if isinstance(value, str) else str(value["sha256"])


def validate_result_package(package: ResultPackage) -> list[str]:
    failures: list[str] = []
    manifest_path = package.directory / "final_statistics_manifest.json"
    if not manifest_path.is_file():
        return [f"{package.name}: missing {manifest_path.relative_to(PROJECT_ROOT)}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        failures.append(f"{package.name}: manifest status is not COMPLETE")
    outputs = manifest.get("outputs", {})
    for filename, record in outputs.items():
        # A manifest cannot contain its own final digest. Historical manifests
        # that list themselves use that entry as generation metadata.
        if filename == manifest_path.name:
            continue
        path = package.directory / filename
        if not path.is_file():
            failures.append(f"{package.name}: missing {path.relative_to(PROJECT_ROOT)}")
            continue
        expected = _manifest_hash(record).lower()
        observed = sha256_file(path)
        if observed != expected:
            failures.append(f"{package.name}: SHA-256 mismatch for {path.relative_to(PROJECT_ROOT)}")
        if isinstance(record, dict) and "bytes" in record and path.stat().st_size != int(record["bytes"]):
            failures.append(f"{package.name}: byte-size mismatch for {path.relative_to(PROJECT_ROOT)}")
    return failures


def validate_publication_assets() -> list[str]:
    directory = PROJECT_ROOT / "results" / "publication" / "source_package"
    manifest_path = directory / "PUBLICATION_ASSET_SHA256.csv"
    if not manifest_path.is_file():
        return ["publication: missing publication asset manifest"]
    failures: list[str] = []
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        relative = row.get("relative_path") or row.get("path") or row.get("file")
        if not relative:
            failures.append("publication: asset manifest has a row without a path")
            continue
        path = directory / relative
        if not path.is_file():
            failures.append(f"publication: missing {path.relative_to(PROJECT_ROOT)}")
            continue
        expected = (row.get("sha256") or row.get("SHA-256") or "").lower()
        if sha256_file(path) != expected:
            failures.append(f"publication: SHA-256 mismatch for {path.relative_to(PROJECT_ROOT)}")
        expected_size = row.get("bytes") or row.get("byte_size")
        if expected_size and path.stat().st_size != int(expected_size):
            failures.append(f"publication: byte-size mismatch for {path.relative_to(PROJECT_ROOT)}")
    return failures


def validate_all() -> list[str]:
    failures: list[str] = []
    for package in PACKAGES:
        failures.extend(validate_result_package(package))
    # Publication PDFs and LaTeX belong to the journal submission, not this code archive.
    if (PROJECT_ROOT / "results" / "publication" / "source_package").is_dir():
        failures.extend(validate_publication_assets())
    return failures
