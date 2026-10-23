# -*- coding: utf-8 -*-
"""Pre-performance Turkish AI Detection ↔ external overlap audit.

This is the canonical active overlap auditor for the external cross-corpus study.

It is deliberately model-blind and performance-blind:
- no training
- no model/checkpoint loading
- no inference
- no calibration/threshold use
- no performance metric calculation

It implements the prespecified overlap contract from
configs/external_cross_corpus.json:

1) exact:
   SHA-256 of prepared UTF-8 text bytes.

2) normalized exact:
   turkish_ai_detection.data.study.normalized_duplicate_key
   with the frozen contract Unicode NFC -> collapse whitespace runs to one ASCII
   space -> strip ends, then SHA-256 of normalized UTF-8 text.

3) near duplicate:
   exact Jaccard >= 0.80 on unique Turkish-aware normalized contiguous
   word-level 5-shingle sets, delegating token normalization/shingling to the
   existing frozen near-duplicate builder.

Primary external populations are never modified. Separate overlap-excluded
sensitivity populations/manifests are generated from the union of the three
prespecified overlap rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


PROTOCOL_ID = "EXTERNAL_CROSS_CORPUS_V2"
EXPECTED_CONFIG_STATUS = "DRAFT_FOR_REVIEW"
EXPECTED_PRIMARY_DATASETS = (
    "renklier2026",
    "tum2024_tr",
    "bostancioglu2026_tr",
)
EXPECTED_SHINGLE_SIZE = 5
EXPECTED_JACCARD_THRESHOLD = 0.80

DEFAULT_CONFIG_REL = "configs/external_cross_corpus.json"
DEFAULT_AUDIT_ROOT_REL = (
    "provenance/experiments/external_cross_corpus_v2/overlap_audit"
)
DEFAULT_SENSITIVITY_ROOT_REL = "data/external_prepared/overlap_excluded"

SENSITIVITY_FILENAMES = {
    "renklier2026": "renklier2026_test_overlap_excluded_standardized.csv",
    "tum2024_tr": "tum2024_tr_test_overlap_excluded_standardized.csv",
    "bostancioglu2026_tr": "bostancioglu2026_tr_full_overlap_excluded_standardized.csv",
}


def find_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / ".gitignore").is_file():
            return candidate
    raise RuntimeError("project root not found via .gitignore")


ROOT = find_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turkish_ai_detection.data.study import normalized_duplicate_key  # noqa: E402
from turkish_ai_detection.experiments.near_duplicate_sensitivity.build_clusters import (  # noqa: E402
    shingles,
)


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


def sha256_text_utf8(text: object) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def file_record(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing file: {path}")
    return {
        "path": rel(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


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


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def protocol_normalized_text(text: object) -> str:
    value = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", value).strip()


def normalized_digest(text: object) -> str:
    observed = normalized_duplicate_key(text)
    require(
        isinstance(observed, str),
        "normalized_duplicate_key() must return str",
    )
    expected = protocol_normalized_text(text)
    require(
        observed == expected,
        "normalized_duplicate_key() no longer matches the frozen "
        "NFC + whitespace-collapse + strip contract",
    )
    return hashlib.sha256(observed.encode("utf-8")).hexdigest()


def verify_contract(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = read_json(config_path)
    require(cfg.get("status") == EXPECTED_CONFIG_STATUS, "unexpected config status")

    overlap = cfg.get("turkish_ai_detection_external_overlap_audit")
    require(isinstance(overlap, dict), "turkish_ai_detection_external_overlap_audit missing")

    require(
        overlap.get("status")
        == "PRESPECIFIED_MUST_BE_FROZEN_BEFORE_EXTERNAL_PREDICTION",
        "unexpected overlap audit status",
    )
    require(overlap.get("labels_or_model_scores_used") is False, "overlap must be label/score blind")
    require(overlap.get("primary_population_changed") is False, "primary population must not change")

    require(
        overlap.get("exact") == "SHA-256 of the prepared UTF-8 text bytes",
        "exact overlap contract mismatch",
    )

    normalized = overlap.get("normalized_exact")
    require(isinstance(normalized, dict), "normalized_exact contract missing")
    require(
        normalized.get("implementation")
        == "turkish_ai_detection.data.study.normalized_duplicate_key",
        "normalized exact implementation mismatch",
    )
    require(
        normalized.get("normalization")
        == "Unicode NFC, collapse whitespace runs to one ASCII space, strip ends",
        "normalized exact normalization mismatch",
    )
    require(
        normalized.get("comparison") == "SHA-256 of normalized UTF-8 text",
        "normalized exact comparison mismatch",
    )

    near = overlap.get("near_duplicate_reuse")
    require(isinstance(near, dict), "near_duplicate_reuse missing")
    require(float(near.get("threshold")) == EXPECTED_JACCARD_THRESHOLD, "near threshold mismatch")
    require(near.get("operator") == ">=", "near operator mismatch")
    require(near.get("new_threshold_permitted") is False, "new threshold must be prohibited")
    require(
        near.get("method")
        == "Exact Jaccard on unique Turkish-aware normalized contiguous word 5-shingle sets",
        "near-duplicate method mismatch",
    )

    sensitivity = overlap.get("overlap_excluded_sensitivity")
    require(isinstance(sensitivity, dict), "overlap-excluded sensitivity contract missing")
    require(
        sensitivity.get("exclude_union_of")
        == ["exact", "normalized_exact", "near_duplicate"],
        "sensitivity exclusion union mismatch",
    )
    require(
        sensitivity.get("freeze_exclusion_manifest_before_external_prediction") is True,
        "exclusion manifest must be frozen before external prediction",
    )
    require(
        sensitivity.get("performance_guided_exclusion") is False,
        "performance-guided exclusion must be false",
    )

    builder = ROOT / near["current_migrated_builder"]
    require(builder.is_file(), f"near-duplicate builder missing: {builder}")
    require(
        sha256_file(builder) == near["current_migrated_builder_sha256"],
        "near-duplicate builder hash mismatch",
    )

    method_freeze = ROOT / near["method_freeze"]
    require(method_freeze.is_file(), f"near-duplicate method freeze missing: {method_freeze}")
    require(
        sha256_file(method_freeze) == near["method_freeze_sha256"],
        "near-duplicate method-freeze hash mismatch",
    )

    return cfg, overlap


def external_bindings(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    block = cfg.get("external_data_bindings")
    require(isinstance(block, dict), "external_data_bindings missing")
    primary = block.get("primary")
    require(isinstance(primary, list), "external primary bindings missing")

    by_id: dict[str, dict[str, Any]] = {}
    for item in primary:
        require(isinstance(item, dict), "invalid external primary binding")
        dataset_id = item.get("dataset_id")
        require(isinstance(dataset_id, str), "external dataset_id missing")
        require(dataset_id not in by_id, f"duplicate external dataset binding: {dataset_id}")
        by_id[dataset_id] = item

    require(
        tuple(sorted(by_id)) == tuple(sorted(EXPECTED_PRIMARY_DATASETS)),
        f"unexpected primary external dataset set: {sorted(by_id)}",
    )
    return by_id


def load_internal(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted({"text_id", "text"} - set(frame.columns))
    require(not missing, f"internal dataset missing columns {missing}")
    view = frame.loc[:, ["text_id", "text"]].copy()
    require(view["text_id"].notna().all(), "internal text_id contains null")
    require(view["text_id"].astype(str).is_unique, "internal text_id is not unique")
    require(view["text"].notna().all(), "internal text contains null")
    view["text_id"] = view["text_id"].astype(str)
    view["text"] = view["text"].astype(str)
    require((view["text"].str.len() > 0).all(), "internal text contains empty string")
    return view


def load_external(path: Path, binding: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    require(path.is_file(), f"external dataset missing: {path}")
    require(sha256_file(path) == binding["sha256"], f"external dataset hash mismatch: {path}")

    full = pd.read_csv(path)
    require(len(full) == int(binding["rows"]), f"external row count mismatch: {path}")

    required = {"external_id", "text", "text_sha256"}
    missing = sorted(required - set(full.columns))
    require(not missing, f"external dataset missing columns {missing}: {path}")

    view = full.loc[:, ["external_id", "text", "text_sha256"]].copy()
    require(view["external_id"].notna().all(), f"external_id contains null: {path}")
    require(view["external_id"].astype(str).is_unique, f"external_id is not unique: {path}")
    require(view["text"].notna().all(), f"text contains null: {path}")
    require(view["text_sha256"].notna().all(), f"text_sha256 contains null: {path}")

    view["external_id"] = view["external_id"].astype(str)
    view["text"] = view["text"].astype(str)
    view["text_sha256"] = view["text_sha256"].astype(str).str.lower()
    require((view["text"].str.len() > 0).all(), f"text contains empty string: {path}")

    recomputed = [sha256_text_utf8(value) for value in view["text"]]
    require(
        recomputed == view["text_sha256"].tolist(),
        f"stored text_sha256 does not equal prepared UTF-8 text bytes: {path}",
    )
    return full, view


def make_lookup(values: list[str], ids: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for value, identifier in zip(values, ids):
        out.setdefault(value, []).append(identifier)
    return out


def build_near_edges(internal: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    combined = (
        internal["text"].astype(str).tolist()
        + external["text"].astype(str).tolist()
    )
    vectorizer = CountVectorizer(
        analyzer=lambda doc: shingles(doc, k=EXPECTED_SHINGLE_SIZE),
        binary=True,
        dtype=np.int32,
    )
    x = vectorizer.fit_transform(combined)
    if not sparse.isspmatrix_csr(x):
        x = x.tocsr()

    n_internal = len(internal)
    x_internal = x[:n_internal]
    x_external = x[n_internal:]

    internal_sizes = np.asarray(x_internal.sum(axis=1)).ravel().astype(np.int64)
    external_sizes = np.asarray(x_external.sum(axis=1)).ravel().astype(np.int64)
    intersections = (x_external @ x_internal.T).tocoo()

    internal_ids = internal["text_id"].astype(str).to_numpy()
    external_ids = external["external_id"].astype(str).to_numpy()

    rows: list[dict[str, Any]] = []
    for ext_i, int_i, intersection in zip(
        intersections.row,
        intersections.col,
        intersections.data,
    ):
        intersection = int(intersection)
        if intersection <= 0:
            continue
        union = int(external_sizes[ext_i] + internal_sizes[int_i] - intersection)
        if union <= 0:
            continue
        similarity = float(intersection) / float(union)
        if similarity < EXPECTED_JACCARD_THRESHOLD:
            continue
        rows.append(
            {
                "external_id": external_ids[ext_i],
                "internal_text_id": internal_ids[int_i],
                "intersection_shingles": intersection,
                "union_shingles": union,
                "exact_jaccard": similarity,
            }
        )

    edges = pd.DataFrame(
        rows,
        columns=[
            "external_id",
            "internal_text_id",
            "intersection_shingles",
            "union_shingles",
            "exact_jaccard",
        ],
    )
    if not edges.empty:
        edges = edges.sort_values(
            ["exact_jaccard", "external_id", "internal_text_id"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
    return edges


def audit_corpus(
    dataset_id: str,
    binding: dict[str, Any],
    internal: pd.DataFrame,
    audit_root: Path,
    sensitivity_root: Path,
) -> dict[str, Any]:
    source_path = (ROOT / binding["path"]).resolve()
    full_external, external = load_external(source_path, binding)

    internal_ids = internal["text_id"].astype(str).tolist()
    external_ids = external["external_id"].astype(str).tolist()

    internal_raw = [sha256_text_utf8(value) for value in internal["text"]]
    external_raw = external["text_sha256"].tolist()

    internal_normalized = [normalized_digest(value) for value in internal["text"]]
    external_normalized = [normalized_digest(value) for value in external["text"]]

    raw_lookup = make_lookup(internal_raw, internal_ids)
    normalized_lookup = make_lookup(internal_normalized, internal_ids)

    exact_matches = {
        external_id: sorted(raw_lookup.get(digest, []))
        for external_id, digest in zip(external_ids, external_raw)
    }
    normalized_matches = {
        external_id: sorted(normalized_lookup.get(digest, []))
        for external_id, digest in zip(external_ids, external_normalized)
    }

    near_edges = build_near_edges(internal, external)
    near_matches: dict[str, list[tuple[str, float]]] = {}
    for row in near_edges.itertuples(index=False):
        near_matches.setdefault(str(row.external_id), []).append(
            (str(row.internal_text_id), float(row.exact_jaccard))
        )

    manifest_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []

    for external_id, text_sha256 in zip(external_ids, external_raw):
        exact_ids = exact_matches[external_id]
        norm_ids = normalized_matches[external_id]
        near_pairs = near_matches.get(external_id, [])
        near_ids = sorted({identifier for identifier, _ in near_pairs})
        near_scores = {identifier: score for identifier, score in near_pairs}

        raw_exact = bool(exact_ids)
        normalized_exact = bool(norm_ids)
        near_duplicate = bool(near_ids)
        excluded = raw_exact or normalized_exact or near_duplicate

        reasons: list[str] = []
        if raw_exact:
            reasons.append("RAW_EXACT")
        if normalized_exact:
            reasons.append("NORMALIZED_EXACT")
        if near_duplicate:
            reasons.append("JACCARD_5SHINGLE_GE_0.80")

        manifest_rows.append(
            {
                "external_id": external_id,
                "text_sha256": text_sha256,
                "raw_exact_overlap": raw_exact,
                "normalized_exact_overlap": normalized_exact,
                "near_duplicate_overlap": near_duplicate,
                "max_exact_jaccard": (
                    max((score for _, score in near_pairs))
                    if near_pairs
                    else np.nan
                ),
                "exclude_from_overlap_excluded_sensitivity": excluded,
                "exclusion_reasons": ";".join(reasons),
            }
        )

        all_internal_ids = sorted(set(exact_ids) | set(norm_ids) | set(near_ids))
        for internal_id in all_internal_ids:
            match_rows.append(
                {
                    "external_id": external_id,
                    "internal_text_id": internal_id,
                    "raw_exact_overlap": internal_id in exact_ids,
                    "normalized_exact_overlap": internal_id in norm_ids,
                    "near_duplicate_overlap": internal_id in near_ids,
                    "exact_jaccard": near_scores.get(internal_id, np.nan),
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    matches = pd.DataFrame(
        match_rows,
        columns=[
            "external_id",
            "internal_text_id",
            "raw_exact_overlap",
            "normalized_exact_overlap",
            "near_duplicate_overlap",
            "exact_jaccard",
        ],
    )

    excluded_ids = set(
        manifest.loc[
            manifest["exclude_from_overlap_excluded_sensitivity"],
            "external_id",
        ].astype(str)
    )
    sensitivity = full_external.loc[
        ~full_external["external_id"].astype(str).isin(excluded_ids)
    ].copy()

    manifest_path = audit_root / f"{dataset_id}_exclusion_manifest.csv"
    matches_path = audit_root / f"{dataset_id}_overlap_matches.csv"
    near_edges_path = audit_root / f"{dataset_id}_near_duplicate_edges.csv"
    sensitivity_path = sensitivity_root / SENSITIVITY_FILENAMES[dataset_id]

    for path in (manifest_path, matches_path, near_edges_path, sensitivity_path):
        require(not path.exists(), f"refusing to overwrite existing output: {path}")

    atomic_csv(manifest, manifest_path)
    atomic_csv(matches, matches_path)
    atomic_csv(near_edges, near_edges_path)
    atomic_csv(sensitivity, sensitivity_path)

    return {
        "dataset_id": dataset_id,
        "primary_source": file_record(source_path),
        "primary_rows": int(len(full_external)),
        "decision_columns_used": ["external_id", "text", "text_sha256"],
        "labels_or_subgroups_used_for_overlap_decisions": False,
        "raw_exact_external_rows": int(manifest["raw_exact_overlap"].sum()),
        "normalized_exact_external_rows": int(manifest["normalized_exact_overlap"].sum()),
        "near_duplicate_external_rows": int(manifest["near_duplicate_overlap"].sum()),
        "near_duplicate_edges": int(len(near_edges)),
        "union_excluded_external_rows": int(
            manifest["exclude_from_overlap_excluded_sensitivity"].sum()
        ),
        "overlap_excluded_sensitivity_rows": int(len(sensitivity)),
        "primary_population_modified": False,
        "outputs": {
            "exclusion_manifest": file_record(manifest_path),
            "overlap_matches": file_record(matches_path),
            "near_duplicate_edges": file_record(near_edges_path),
            "overlap_excluded_sensitivity_population": file_record(sensitivity_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / DEFAULT_CONFIG_REL,
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=ROOT / DEFAULT_AUDIT_ROOT_REL,
    )
    parser.add_argument(
        "--sensitivity-root",
        type=Path,
        default=ROOT / DEFAULT_SENSITIVITY_ROOT_REL,
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    audit_root = args.audit_root if args.audit_root.is_absolute() else ROOT / args.audit_root
    sensitivity_root = (
        args.sensitivity_root
        if args.sensitivity_root.is_absolute()
        else ROOT / args.sensitivity_root
    )
    config_path = config_path.resolve()
    audit_root = audit_root.resolve()
    sensitivity_root = sensitivity_root.resolve()

    report_path = audit_root / "OVERLAP_AUDIT.json"
    require(not report_path.exists(), f"refusing to overwrite existing report: {report_path}")

    cfg, overlap_contract = verify_contract(config_path)
    bindings = external_bindings(cfg)

    internal_path = (ROOT / overlap_contract["inputs"]["turkish_ai_detection"]).resolve()
    require(internal_path.is_file(), f"internal reference missing: {internal_path}")
    internal = load_internal(internal_path)

    corpus_records = [
        audit_corpus(
            dataset_id,
            bindings[dataset_id],
            internal,
            audit_root,
            sensitivity_root,
        )
        for dataset_id in EXPECTED_PRIMARY_DATASETS
    ]

    report = {
        "schema_version": "external-cross-corpus-overlap-audit-v1",
        "protocol_id": PROTOCOL_ID,
        "status": "PASS_PREPERFORMANCE_OVERLAP_AUDIT",
        "created_utc": utc_now(),
        "config": file_record(config_path),
        "performance_results_consulted": False,
        "model_predictions_consulted": False,
        "external_labels_or_subgroups_used_for_overlap_decisions": False,
        "primary_external_populations_modified": False,
        "method": {
            "exact": "SHA-256 of prepared UTF-8 text bytes",
            "normalized_exact": {
                "implementation": "turkish_ai_detection.data.study.normalized_duplicate_key",
                "normalization": (
                    "Unicode NFC, collapse whitespace runs to one ASCII space, strip ends"
                ),
                "comparison": "SHA-256 of normalized UTF-8 text",
            },
            "near_duplicate": {
                "method": (
                    "Exact Jaccard on unique Turkish-aware normalized contiguous "
                    "word 5-shingle sets"
                ),
                "threshold": EXPECTED_JACCARD_THRESHOLD,
                "operator": ">=",
                "shingle_size": EXPECTED_SHINGLE_SIZE,
                "builder": overlap_contract["near_duplicate_reuse"][
                    "current_migrated_builder"
                ],
                "builder_sha256": overlap_contract["near_duplicate_reuse"][
                    "current_migrated_builder_sha256"
                ],
                "method_freeze": overlap_contract["near_duplicate_reuse"]["method_freeze"],
                "method_freeze_sha256": overlap_contract["near_duplicate_reuse"][
                    "method_freeze_sha256"
                ],
                "semantic_scope": (
                    "lexical near-duplicate audit; not a semantic-paraphrase detector"
                ),
            },
        },
        "sensitivity_exclusion_policy": (
            "Union of exact, normalized_exact, and near_duplicate; "
            "no performance-guided exclusion."
        ),
        "internal_reference": {
            **file_record(internal_path),
            "rows": int(len(internal)),
        },
        "corpora": corpus_records,
        "scope": (
            "Pre-performance overlap audit and frozen sensitivity-population "
            "construction only. No training, inference, calibration, threshold "
            "selection, or external performance metrics."
        ),
    }

    atomic_json(report_path, report)

    summary = {
        "status": report["status"],
        "report_path": rel(report_path),
        "report_sha256": sha256_file(report_path),
        "corpora": [
            {
                "dataset_id": item["dataset_id"],
                "primary_rows": item["primary_rows"],
                "raw_exact_external_rows": item["raw_exact_external_rows"],
                "normalized_exact_external_rows": item["normalized_exact_external_rows"],
                "near_duplicate_external_rows": item["near_duplicate_external_rows"],
                "union_excluded_external_rows": item["union_excluded_external_rows"],
                "overlap_excluded_sensitivity_rows": item[
                    "overlap_excluded_sensitivity_rows"
                ],
            }
            for item in corpus_records
        ],
        "performance_results_consulted": False,
        "primary_external_populations_modified": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
