# -*- coding: utf-8 -*-
"""Reproducible data infrastructure for the current paired-domain study.

This module deliberately has no write path into the historical ABSTRACT or
sealed-evaluation history. All learned transforms used below are fitted inside
an inner or outer training partition.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


GENERATOR_MAP = {
    "gpt_5-5-high": "GPT",
    "claude-opus-4-8-high": "Claude",
    "gemini_pro-3-1-extended-thinking": "Gemini",
    "qwen_3-7-max": "Qwen",
    "kimi_k-2-6_fast-high": "Kimi",
}
EXPECTED_GENERATORS = tuple(GENERATOR_MAP.values())
EXPECTED_DOMAIN_PAIRS = {"ABSTRACT": 1000, "NEWSPAPER": 500}
EXPECTED_DOMAIN_ROWS = {"ABSTRACT": 2000, "NEWSPAPER": 1000}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Frozen manifests retain their original path keys.  This table changes only
# filesystem resolution; it never rewrites the historical record.
_LEGACY_PREFIX_MIGRATIONS = (
    ("outputs/combined_v2/calibration_v1", "results/calibration"),
    ("outputs/combined_v2/primary_ablation_v1", "results/primary"),
    ("outputs/combined_v2/near_duplicate_sensitivity_v1", "results/near_duplicate_sensitivity"),
    ("provenance/experiments/calibration_v1", "provenance/experiments/calibration"),
    ("provenance/combined_v2/primary_ablation_v1", "provenance/experiments/primary"),
    ("provenance/combined_v2/near_duplicate_sensitivity_v1", "provenance/experiments/near_duplicate_sensitivity"),
)


def resolve_frozen_record_path(
    root: Path | str, recorded_path: str, expected_sha256: str
) -> Path:
    """Resolve an immutable manifest key after the repository migration.

    Active data/result records resolve to their canonical location. Historical
    path keys are preserved in immutable records rather than rewritten.
    """
    project_root = Path(root)
    normalized = recorded_path.replace("\\", "/")
    migrated = normalized
    for old_prefix, new_prefix in _LEGACY_PREFIX_MIGRATIONS:
        if normalized == old_prefix or normalized.startswith(old_prefix + "/"):
            migrated = new_prefix + normalized[len(old_prefix):]
            break
    historical_snapshot = (
        project_root
        / "provenance"
        / "history"
        / "source_snapshots"
        / "pre_refactor"
        / normalized
    )
    candidates = (project_root / normalized, project_root / migrated, historical_snapshot)
    expected_hash = str(expected_sha256).lower()
    checked: list[Path] = []
    for candidate in candidates:
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.is_file() and sha256_file(candidate).lower() == expected_hash:
            return candidate
    rendered = ", ".join(path.as_posix() for path in checked)
    raise RuntimeError(
        f"frozen path/hash cannot be resolved: {recorded_path}; checked {rendered}"
    )


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join(map(str, (seed,) + parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def normalized_duplicate_key(text: object) -> str:
    value = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", value).strip()


def _numeric_suffix(value: object) -> int:
    match = re.search(r"(\d+)$", str(value))
    if not match:
        raise ValueError(f"ID has no numeric suffix: {value!r}")
    return int(match.group(1))


def _word_count(text: object) -> int:
    return len(str(text).split())


def _pair_generators(frame: pd.DataFrame, raw_generator_column: str) -> pd.Series:
    ai = frame.loc[frame["label"].astype(str).str.lower().eq("ai"), ["pair_id", raw_generator_column]]
    if ai["pair_id"].duplicated().any():
        raise ValueError("AI rows do not define one generator per pair")
    unknown = sorted(set(ai[raw_generator_column].dropna()) - set(GENERATOR_MAP))
    if unknown:
        raise ValueError(f"Unknown generator identifiers: {unknown}")
    return ai.set_index("pair_id")[raw_generator_column].map(GENERATOR_MAP)


def build_abstract_rows(path: Path | str) -> pd.DataFrame:
    source = pd.read_csv(path)
    required = {"sample_id", "pair_id", "raw_id", "label", "model", "text"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"ABSTRACT source missing columns: {sorted(missing)}")
    source = source.copy()
    pair_generator = _pair_generators(source, "model")
    pair_number = source["pair_id"].map(_numeric_suffix)
    source["pair_id"] = pair_number.map(lambda n: f"ABS_{n:04d}")
    source["text_id"] = [f"ABS_{n:04d}_{'H' if str(label).lower() == 'human' else 'A'}" for n, label in zip(pair_number, source["label"])]
    old_pair = source["pair_id"].copy()
    # pair_generator is indexed by the old namespace, so map through raw pair numbers.
    generator_by_number = { _numeric_suffix(k): v for k, v in pair_generator.items() }
    source["generator"] = pair_number.map(generator_by_number)
    source["generator_id"] = source["model"]
    source["label"] = source["label"].astype(str).str.upper()
    source["domain"] = "ABSTRACT"
    source["source_year"] = source.get("year")
    source["academic_field"] = source.get("field")
    source["newspaper_source"] = pd.NA
    source["author"] = pd.NA
    source["original_source_id"] = source.get("source_id", source["raw_id"])
    source["word_count"] = source["text"].map(_word_count)
    columns = [
        "text_id", "pair_id", "label", "domain", "generator", "text",
        "source_year", "academic_field", "broad_field", "newspaper_source",
        "author", "word_count", "original_source_id", "generator_id",
        "raw_id", "title", "keywords", "journal", "source_url",
    ]
    del old_pair
    return source.reindex(columns=columns)


def build_newspaper_rows(human_path: Path | str, ai_path: Path | str) -> pd.DataFrame:
    human = pd.read_csv(human_path)
    ai = pd.read_csv(ai_path)
    human_required = {"id", "orijinal_id", "kaynak", "yazar", "tarih", "metin", "kelime_sayisi"}
    ai_required = {"id", "model", "generated_text"}
    if human_required - set(human.columns):
        raise ValueError(f"NEWSPAPER human source missing columns: {sorted(human_required - set(human.columns))}")
    if ai_required - set(ai.columns):
        raise ValueError(f"NEWSPAPER AI source missing columns: {sorted(ai_required - set(ai.columns))}")
    human = human.copy(); ai = ai.copy()
    human["_n"] = human["id"].map(_numeric_suffix)
    ai["_n"] = ai["id"].map(_numeric_suffix)
    if set(human["_n"]) != set(range(1, 501)) or set(ai["_n"]) != set(range(1, 501)):
        raise ValueError("NEWSPAPER IDs must each cover numeric suffixes 1..500 exactly")
    if human["_n"].duplicated().any() or ai["_n"].duplicated().any():
        raise ValueError("NEWSPAPER numeric ID suffix collision")
    pair_meta = human.set_index("_n")
    unknown = sorted(set(ai["model"]) - set(GENERATOR_MAP))
    if unknown:
        raise ValueError(f"Unknown NEWSPAPER generator identifiers: {unknown}")

    common = pd.DataFrame({"_n": range(1, 501)})
    h = common.merge(human, on="_n", validate="one_to_one")
    a = common.merge(ai, on="_n", validate="one_to_one")
    generator_by_number = a.set_index("_n")["model"].map(GENERATOR_MAP)

    def make(part: pd.DataFrame, label: str) -> pd.DataFrame:
        out = pd.DataFrame(index=part.index)
        out["text_id"] = part["_n"].map(lambda n: f"NEWS_{n:04d}_{'H' if label == 'HUMAN' else 'A'}")
        out["pair_id"] = part["_n"].map(lambda n: f"NEWS_{n:04d}")
        out["label"] = label
        out["domain"] = "NEWSPAPER"
        out["generator"] = part["_n"].map(generator_by_number)
        out["text"] = part["metin"] if label == "HUMAN" else part["generated_text"]
        dates = pd.to_datetime(part["tarih"], errors="coerce") if label == "HUMAN" else pd.to_datetime(part["_n"].map(pair_meta["tarih"]), errors="coerce")
        out["source_year"] = dates.dt.year.astype("Int64")
        out["academic_field"] = pd.NA
        out["broad_field"] = pd.NA
        out["newspaper_source"] = part["kaynak"] if label == "HUMAN" else part["_n"].map(pair_meta["kaynak"])
        out["author"] = part["yazar"] if label == "HUMAN" else part["_n"].map(pair_meta["yazar"])
        out["word_count"] = out["text"].map(_word_count)
        out["original_source_id"] = part["orijinal_id"] if label == "HUMAN" else part["_n"].map(pair_meta["orijinal_id"])
        out["generator_id"] = part["_n"].map(a.set_index("_n")["model"])
        out["raw_id"] = part["id"]
        out["title"] = part["baslik"] if label == "HUMAN" else part["_n"].map(pair_meta["baslik"])
        out["keywords"] = part["anahtar_kelimeler"] if label == "HUMAN" else part["_n"].map(pair_meta["anahtar_kelimeler"])
        out["journal"] = pd.NA
        out["source_url"] = part["url"] if label == "HUMAN" else part["_n"].map(pair_meta["url"])
        return out

    return pd.concat([make(h, "HUMAN"), make(a, "AI")], ignore_index=True)


def audit_dataset(data: pd.DataFrame) -> dict:
    checks: dict[str, dict] = {}

    def check(name: str, observed: object, expected: object, passed: bool | None = None) -> None:
        checks[name] = {
            "passed": bool(observed == expected) if passed is None else bool(passed),
            "observed": observed,
            "expected": expected,
        }

    check("row_count", len(data), 3000)
    check("unique_pair_ids", int(data["pair_id"].nunique()), 1500)
    labels = data["label"].value_counts().to_dict()
    check("human_count", int(labels.get("HUMAN", 0)), 1500)
    check("ai_count", int(labels.get("AI", 0)), 1500)
    pair_sizes = data.groupby("pair_id", dropna=False).size()
    check("two_rows_per_pair", int(pair_sizes.eq(2).sum()), 1500, pair_sizes.eq(2).all())
    pair_labels = data.groupby("pair_id", dropna=False)["label"].agg(lambda x: tuple(sorted(x.astype(str))))
    valid_pair_labels = pair_labels.map(lambda value: value == ("AI", "HUMAN"))
    check("one_human_one_ai_per_pair", int(valid_pair_labels.sum()), 1500, valid_pair_labels.all())
    pair_domains = data.groupby("pair_id", dropna=False)["domain"].nunique(dropna=False)
    check("no_cross_domain_pairs", int(pair_domains.eq(1).sum()), 1500, pair_domains.eq(1).all())
    check("no_missing_text", int(data["text"].isna().sum() + data["text"].astype(str).str.strip().eq("").sum()), 0)
    check("unique_text_ids", int(data["text_id"].nunique()), 3000, not data["text_id"].duplicated().any())
    namespace = data.groupby("pair_id")["domain"].nunique()
    check("no_pair_namespace_collision", int((namespace > 1).sum()), 0)
    domain_rows = {str(k): int(v) for k, v in data["domain"].value_counts().sort_index().items()}
    check("domain_row_counts", domain_rows, EXPECTED_DOMAIN_ROWS)
    pair_frame = data.drop_duplicates("pair_id")
    domain_pairs = {str(k): int(v) for k, v in pair_frame["domain"].value_counts().sort_index().items()}
    check("domain_pair_counts", domain_pairs, EXPECTED_DOMAIN_PAIRS)
    actual_generators = {
        domain: {generator: int(count) for generator, count in group["generator"].value_counts().sort_index().items()}
        for domain, group in pair_frame.groupby("domain", sort=True)
    }
    expected_generators = {
        "ABSTRACT": {g: 200 for g in EXPECTED_GENERATORS},
        "NEWSPAPER": {g: 100 for g in EXPECTED_GENERATORS},
    }
    check("pair_generator_counts", actual_generators, expected_generators)
    pair_generator_nunique = data.groupby("pair_id")["generator"].nunique(dropna=False)
    check("shared_pair_generator", int(pair_generator_nunique.eq(1).sum()), 1500, pair_generator_nunique.eq(1).all())

    keys = data["text"].map(normalized_duplicate_key)
    groups = data.assign(_key=keys).groupby("_key", sort=False).filter(lambda x: len(x) > 1)
    duplicate_groups = []
    for key, group in groups.groupby("_key", sort=False):
        duplicate_groups.append({
            "normalized_text_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
            "count": int(len(group)),
            "text_ids": group["text_id"].astype(str).tolist(),
            "pair_ids": group["pair_id"].astype(str).tolist(),
        })
    check("exact_normalized_text_duplicate_groups", len(duplicate_groups), 0)
    return {
        "schema_version": "combined-v2-integrity-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(item["passed"] for item in checks.values()) else "FAIL",
        "normalization": "Unicode NFC, collapse all whitespace, strip; case and punctuation preserved",
        "checks": checks,
        "duplicate_groups": duplicate_groups,
    }


def create_nested_folds(data: pd.DataFrame, seed: int = 42, outer_folds: int = 5, inner_folds: int = 4) -> pd.DataFrame:
    pair = data.drop_duplicates("pair_id")[["pair_id", "domain", "generator"]].copy()
    if len(pair) != 1500 or not pair["pair_id"].is_unique:
        raise ValueError("fold construction requires exactly 1500 unique pairs")
    pair["stratum"] = pair["domain"] + "×" + pair["generator"]
    pair["outer_fold"] = -1
    for stratum, indices in pair.groupby("stratum", sort=True).groups.items():
        ids = sorted(pair.loc[list(indices), "pair_id"].astype(str).tolist())
        rng = np.random.default_rng(stable_seed(seed, "outer", stratum))
        ids = list(rng.permutation(ids))
        for position, pair_id in enumerate(ids):
            pair.loc[pair["pair_id"].eq(pair_id), "outer_fold"] = position % outer_folds
    for outer in range(outer_folds):
        column = f"inner_fold_outer{outer}"
        pair[column] = -1
        training = pair["outer_fold"].ne(outer)
        for stratum, indices in pair.loc[training].groupby("stratum", sort=True).groups.items():
            ids = sorted(pair.loc[list(indices), "pair_id"].astype(str).tolist())
            rng = np.random.default_rng(stable_seed(seed, "inner", outer, stratum))
            ids = list(rng.permutation(ids))
            for position, pair_id in enumerate(ids):
                pair.loc[pair["pair_id"].eq(pair_id), column] = position % inner_folds
    if (pair["outer_fold"] < 0).any():
        raise RuntimeError("unassigned outer fold")
    return pair.sort_values("pair_id", kind="stable").reset_index(drop=True)


def audit_folds(data: pd.DataFrame, folds: pd.DataFrame, outer_folds: int = 5, inner_folds: int = 4) -> dict:
    pair_meta = data.drop_duplicates("pair_id")[["pair_id", "domain", "generator"]]
    joined = folds.merge(pair_meta, on=["pair_id", "domain", "generator"], validate="one_to_one")
    records = []
    for outer in range(outer_folds):
        evaluation = joined[joined["outer_fold"].eq(outer)]
        for (domain, generator), group in evaluation.groupby(["domain", "generator"], sort=True):
            records.append({"level": "outer_evaluation", "outer_fold": outer, "inner_fold": None, "domain": domain, "generator": generator, "pair_count": int(len(group)), "row_count": int(2 * len(group)), "human_count": int(len(group)), "ai_count": int(len(group))})
        column = f"inner_fold_outer{outer}"
        training = joined[joined["outer_fold"].ne(outer)]
        if (training[column] < 0).any() or not joined.loc[joined["outer_fold"].eq(outer), column].eq(-1).all():
            raise ValueError(f"invalid inner assignments for outer fold {outer}")
        for inner in range(inner_folds):
            validation = training[training[column].eq(inner)]
            for (domain, generator), group in validation.groupby(["domain", "generator"], sort=True):
                records.append({"level": "inner_validation", "outer_fold": outer, "inner_fold": inner, "domain": domain, "generator": generator, "pair_count": int(len(group)), "row_count": int(2 * len(group)), "human_count": int(len(group)), "ai_count": int(len(group))})
    distributions = pd.DataFrame(records)
    expected_outer = {"ABSTRACT": 40, "NEWSPAPER": 20}
    expected_inner = {"ABSTRACT": 40, "NEWSPAPER": 20}
    correct = distributions.apply(lambda row: row["pair_count"] == (expected_outer if row["level"] == "outer_evaluation" else expected_inner)[row["domain"]], axis=1).all()
    # Explicit disjointness checks at both levels.
    leakage = []
    for outer in range(outer_folds):
        test = set(folds.loc[folds["outer_fold"].eq(outer), "pair_id"])
        train = set(folds.loc[folds["outer_fold"].ne(outer), "pair_id"])
        if test & train:
            leakage.append({"outer_fold": outer, "type": "outer", "pairs": sorted(test & train)})
        col = f"inner_fold_outer{outer}"
        validation_sets = [set(folds.loc[folds["outer_fold"].ne(outer) & folds[col].eq(inner), "pair_id"]) for inner in range(inner_folds)]
        for left in range(inner_folds):
            for right in range(left + 1, inner_folds):
                if validation_sets[left] & validation_sets[right]:
                    leakage.append({"outer_fold": outer, "type": "inner", "folds": [left, right], "pairs": sorted(validation_sets[left] & validation_sets[right])})
    return {
        "schema_version": "combined-v2-fold-audit-v1",
        "status": "PASS" if correct and not leakage and len(joined) == 1500 else "FAIL",
        "seed": 42,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "stratification": "pair-level domain × generator",
        "all_strata_exactly_balanced": bool(correct),
        "leakage_findings": leakage,
        "distribution_records": records,
    }


def np_umbrella_k(n_human: int, alpha: float = 0.03, delta: float = 0.05) -> int:
    """Largest k with Binomial(n, alpha) lower-tail probability <= delta.

    This generalizes the historical fixed points n=800 -> k=15 and
    n=1000 -> k=20 without using any observed model result.
    """
    from scipy.stats import binom
    valid = [k for k in range(n_human) if float(binom.cdf(k, n_human, alpha)) <= delta]
    if not valid:
        raise ValueError("no valid NP order statistic for requested n/alpha/delta")
    return max(valid)


def _model_dependencies() -> dict:
    return {
        "M1": {"status": "PREPARED_AWAITING_COMBINED_RAW_FEATURES", "definition": "Turkish AI Detection-13; unchanged 13-feature definition", "conditions": ["A", "B"]},
        "M2": {"status": "PREPARED_AWAITING_COMBINED_RAW_FEATURES", "definition": "reduced Turkish AI Detection; unchanged 7-feature definition", "conditions": ["A"]},
        "M3": {"status": "READY", "definition": "unchanged 12 surface rates with log1p and logistic regression", "conditions": ["A", "B"]},
        "M4": {"status": "READY", "definition": "unchanged word(1,2)+char_wb(2,5) TF-IDF logistic baseline", "conditions": ["A", "B"]},
        "M5": {"status": "PREPARED_COMPUTE_HEAVY", "definition": "dbmdz/bert-base-turkish-cased fixed hyperparameters", "conditions": ["A", "B"]},
    }


def write_json(path: Path | str, payload: object) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def hash_tree(paths: list[Path]) -> dict[str, str]:
    return {str(path).replace("\\", "/"): sha256_file(path) for path in sorted(paths) if path.is_file()}


def build_combined_study(config_path: Path | str) -> dict:
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parents[1]
    abstract_path = root / config["sources"]["abstract_base"]
    human_path = Path(os.path.expandvars(config["sources"]["newspaper_human"])).resolve()
    ai_path = Path(os.path.expandvars(config["sources"]["newspaper_ai"])).resolve()
    expected_hashes = config["source_sha256"]
    observed_hashes = {"abstract_base": sha256_file(abstract_path), "newspaper_human": sha256_file(human_path), "newspaper_ai": sha256_file(ai_path)}
    mismatches = {key: {"expected": expected_hashes[key], "observed": value} for key, value in observed_hashes.items() if expected_hashes.get(key) != value}
    if mismatches:
        raise RuntimeError(f"source hash mismatch: {json.dumps(mismatches, indent=2)}")

    data_dir = root / config["outputs"]["data_dir"]
    output_dir = root / config["outputs"]["output_dir"]
    provenance_dir = root / config["outputs"]["provenance_dir"]
    for directory in (data_dir, output_dir, provenance_dir):
        directory.mkdir(parents=True, exist_ok=True)

    combined = pd.concat([build_abstract_rows(abstract_path), build_newspaper_rows(human_path, ai_path)], ignore_index=True)
    combined = combined.sort_values(["pair_id", "label"], kind="stable").reset_index(drop=True)
    audit = audit_dataset(combined)
    audit_path = output_dir / "integrity_audit.json"
    write_json(audit_path, audit)
    if audit["status"] != "PASS":
        raise RuntimeError(f"Combined v2 integrity audit failed; see {audit_path}")

    dataset_path = data_dir / "paired_study.csv"
    combined.to_csv(dataset_path, index=False, encoding="utf-8", lineterminator="\n")
    folds = create_nested_folds(combined, seed=int(config["protocol"]["seed"]))
    folds_path = data_dir / "nested_folds.csv"
    folds.to_csv(folds_path, index=False, encoding="utf-8", lineterminator="\n")
    fold_audit = audit_folds(combined, folds)
    fold_audit_path = output_dir / "fold_audit.json"
    write_json(fold_audit_path, fold_audit)
    pd.DataFrame(fold_audit["distribution_records"]).to_csv(output_dir / "fold_distributions.csv", index=False, encoding="utf-8", lineterminator="\n")
    if fold_audit["status"] != "PASS":
        raise RuntimeError(f"Combined v2 fold audit failed; see {fold_audit_path}")

    versions = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "transformers", "torch", "stanza", "zeyrek"):
        try:
            from importlib.metadata import version
            versions[package] = version(package)
        except Exception:
            versions[package] = None
    provenance = {
        "schema_version": "combined-v2-provenance-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "Turkish AI-Generated Text Detection Study Combined v2",
        "historical_artifacts_modified": False,
        "random_seeds": {"global": int(config["protocol"]["seed"]), "model": 42},
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation(), "executable": sys.executable},
        "packages": versions,
        "models": _model_dependencies(),
        "model_identifiers": {"M5": "dbmdz/bert-base-turkish-cased"},
        "berturk_revision": config["models"]["M5"]["revision"],
        "source_hashes": observed_hashes,
        "dataset_sha256": sha256_file(dataset_path),
        "fold_sha256": sha256_file(folds_path),
        "config_sha256": sha256_file(config_path),
        "np_threshold_rule": {"alpha": 0.03, "delta": 0.05, "outer_training_humans": 1200, "k_exceed": np_umbrella_k(1200)},
    }
    provenance_path = provenance_dir / "build_provenance.json"
    write_json(provenance_path, provenance)
    manifest_files = [dataset_path, folds_path, audit_path, fold_audit_path, output_dir / "fold_distributions.csv", provenance_path, config_path]
    manifest = {"schema_version": "combined-v2-manifest-v1", "files": hash_tree(manifest_files)}
    manifest_path = provenance_dir / "manifest_sha256.json"
    write_json(manifest_path, manifest)
    return {"dataset": str(dataset_path), "folds": str(folds_path), "integrity": audit["status"], "fold_audit": fold_audit["status"], "dataset_sha256": provenance["dataset_sha256"], "fold_sha256": provenance["fold_sha256"], "manifest": str(manifest_path)}

