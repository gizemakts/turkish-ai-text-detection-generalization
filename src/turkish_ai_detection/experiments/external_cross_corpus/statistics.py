# -*- coding: utf-8 -*-
"""Post-seal external discrimination statistics.

This module requires both a code/config PREEXECUTION_FREEZE and complete
prediction sealing before loading external labels. It reports corpus-specific
AUROC, matched-cell method differences, descriptive same-realization external
minus internal AUROC, and prespecified exact-dedup/overlap sensitivities.

Calibration and frozen operating-point metrics remain conditional and are
not computed in this module before a separate compatibility decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


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
SEAL_REL = (
    "provenance/experiments/external_cross_corpus_v2/prediction_seal/"
    "PRESEAL_PREDICTION_MANIFEST.json"
)
OVERLAP_AUDIT_REL = (
    "provenance/experiments/external_cross_corpus_v2/overlap_audit/"
    "OVERLAP_AUDIT.json"
)
OVERLAP_MANIFEST_ROOT_REL = (
    "provenance/experiments/external_cross_corpus_v2/overlap_audit"
)
PRESEAL_ROOT_REL = "artifacts/predictions/external_cross_corpus/preseal"
RESULT_ROOT_REL = "results/external_cross_corpus/statistics"

BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 2607
CI_LOW = 2.5
CI_HIGH = 97.5


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


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
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


def verify_config(config_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    cfg = read_json(config_path)

    evaluation = cfg.get("evaluation")
    require(isinstance(evaluation, dict), "evaluation block missing")
    require(evaluation.get("primary_metric") == "auroc", "primary metric must be AUROC")
    require(
        evaluation.get("primary_external_pooling") is False,
        "pooled external AUROC must remain prohibited",
    )
    require(
        evaluation.get("labels_and_subgroups_join_only_after_complete_manifest_sealed")
        is True,
        "label join must remain post-seal only",
    )

    stats = cfg.get("statistics")
    require(isinstance(stats, dict), "statistics block missing")
    require(
        int(stats.get("bootstrap_replicates")) == BOOTSTRAP_REPLICATES,
        "bootstrap replicate count mismatch",
    )
    require(
        int(stats.get("bootstrap_seed")) == BOOTSTRAP_SEED,
        "bootstrap seed mismatch",
    )
    require(stats.get("ci") == "95% percentile", "CI contract mismatch")
    require(stats.get("same_resample_indices_across_methods_and_realizations") is True,
            "shared resample contract mismatch")
    require(stats.get("seeds_resampled") is False, "seeds must not be resampled")
    require(stats.get("folds_resampled") is False, "folds must not be resampled")
    require(
        stats.get("model_realizations_treated_as_independent_observations") is False,
        "realizations must not be treated as independent observations",
    )
    require(
        stats.get("cross_corpus_macro_summary") == "NOT_PART_OF_PROTOCOL",
        "cross-corpus macro summary must remain out of protocol",
    )

    expected_units = {
        "renklier2026": "source_group_id",
        "tum2024_tr": "text_sha256",
        "bostancioglu2026_tr": "text_sha256",
    }
    for dataset_id, unit in expected_units.items():
        require(
            stats.get(f"{dataset_id}_unit") == unit,
            f"bootstrap unit mismatch for {dataset_id}",
        )

    block = cfg.get("external_data_bindings")
    require(isinstance(block, dict), "external_data_bindings missing")
    primary = block.get("primary")
    require(isinstance(primary, list), "primary external bindings missing")

    by_id: dict[str, dict[str, Any]] = {}
    for item in primary:
        require(isinstance(item, dict), "invalid external binding")
        dataset_id = item.get("dataset_id")
        require(isinstance(dataset_id, str), "dataset_id missing")
        by_id[dataset_id] = item

    require(set(by_id) == set(DATASET_IDS), "unexpected external primary dataset set")
    return cfg, by_id


def verify_execution_guards(config_path: Path) -> dict[str, Any]:
    freeze_path = ROOT / FREEZE_REL
    require(freeze_path.is_file(), "statistics blocked before PREEXECUTION_FREEZE")
    freeze = read_json(freeze_path)
    from turkish_ai_detection.experiments.external_cross_corpus.freeze_v2 import verify_freeze
    verify_freeze(config_path, freeze_path, required_code_path=Path(__file__).resolve())

    own = Path(__file__).resolve()
    require(
        freeze_contains_binding(freeze, rel(own), sha256_file(own)),
        "PREEXECUTION_FREEZE does not bind this exact statistics.py",
    )
    seal_path = ROOT / SEAL_REL
    require(seal_path.is_file(), "statistics blocked before complete prediction seal")
    seal = read_json(seal_path)
    require(
        seal.get("status") == "SEALED_COMPLETE_LABEL_BLIND_PREDICTION_GRID",
        "prediction seal status is not complete",
    )
    grid = seal.get("grid", {})
    require(int(grid.get("expected_cells", -1)) == 135, "sealed expected grid != 135")
    require(int(grid.get("sealed_cells", -1)) == 135, "sealed prediction grid incomplete")
    sealed_config = seal.get("config", {})
    require(
        sealed_config.get("path") == rel(config_path)
        and sealed_config.get("sha256") == sha256_file(config_path),
        "prediction seal does not bind the active config",
    )
    sealed_freeze = seal.get("preexecution_freeze", {})
    require(
        sealed_freeze.get("path") == rel(freeze_path)
        and sealed_freeze.get("sha256") == sha256_file(freeze_path),
        "prediction seal does not bind the active PREEXECUTION_FREEZE",
    )

    # Critical: a status string / cell count alone is not a cryptographic
    # seal. Recheck EVERY sealed prediction and sidecar byte hash before the
    # first external label is loaded. This also rejects omitted/extra cells.
    expected_cells = {
        (dataset_id, method, seed, outer)
        for dataset_id in DATASET_IDS
        for method in METHODS
        for seed in SEEDS
        for outer in OUTER_FOLDS
    }
    cells = seal.get("cells")
    require(isinstance(cells, list) and len(cells) == 135,
            "prediction seal must contain exactly 135 cell records")
    observed_cells = set()
    for cell in cells:
        require(isinstance(cell, dict), "invalid sealed cell record")
        key = (
            cell.get("dataset_id"),
            cell.get("method"),
            cell.get("seed"),
            cell.get("outer_fold"),
        )
        require(key in expected_cells, f"unexpected sealed cell: {key}")
        require(key not in observed_cells, f"duplicate sealed cell: {key}")
        observed_cells.add(key)
        dataset_id, method, seed, outer = key
        pred_path = prediction_path(dataset_id, method, seed, outer)
        sidecar_path = pred_path.with_suffix(".json")

        for field, path in (("prediction", pred_path), ("sidecar", sidecar_path)):
            record = cell.get(field)
            require(isinstance(record, dict), f"missing sealed {field} record: {key}")
            require(path.is_file(), f"sealed {field} no longer exists: {path}")
            require(
                record.get("path") == rel(path)
                and record.get("sha256") == sha256_file(path),
                f"sealed {field} hash/path mismatch: {path}",
            )

        sidecar = read_json(sidecar_path)
        sidecar_prediction = sidecar.get("prediction", {})
        require(
            sidecar_prediction.get("path") == rel(pred_path)
            and sidecar_prediction.get("sha256") == sha256_file(pred_path),
            f"inference sidecar no longer binds prediction: {sidecar_path}",
        )
        for field in (
            "external_labels_read", "external_subgroups_read",
            "performance_metrics_computed", "external_adaptation",
            "calibration_applied",
        ):
            require(sidecar.get(field) is False,
                    f"label-blind inference guarantee failed ({field}): {sidecar_path}")
    require(observed_cells == expected_cells, "sealed prediction grid incomplete")

    overlap_path = ROOT / OVERLAP_AUDIT_REL
    require(overlap_path.is_file(), "pre-performance overlap audit missing")
    overlap = read_json(overlap_path)
    require(
        overlap.get("status") == "PASS_PREPERFORMANCE_OVERLAP_AUDIT",
        "overlap audit is not PASS",
    )
    require(overlap.get("performance_results_consulted") is False,
            "overlap audit consulted performance")
    require(overlap.get("primary_external_populations_modified") is False,
            "overlap audit modified primary populations")

    # The old overlap-audit report is immutable across active-code migration.
    # Its source config and exclusion-manifest hashes must still be correct
    # before any labels are joined with predictions.
    overlap_config = overlap.get("config", {})
    overlap_config_path = ROOT / overlap_config.get("path", "")
    require(overlap_config_path.is_file()
            and sha256_file(overlap_config_path) == overlap_config.get("sha256"),
            "historic overlap audit input contract was changed")
    records = overlap.get("corpora", [])
    require(isinstance(records, list) and len(records) == len(DATASET_IDS),
            "overlap audit does not bind all three corpora")
    require({item.get("dataset_id") for item in records} == set(DATASET_IDS),
            "overlap corpus binding mismatch")
    for item in records:
        require(item.get("primary_population_modified") is False,
                "overlap report indicates altered primary population")
        manifest = item.get("outputs", {}).get("exclusion_manifest", {})
        path = ROOT / manifest.get("path", "")
        require(path.is_file() and sha256_file(path) == manifest.get("sha256"),
                f"frozen overlap exclusion manifest changed for {item.get('dataset_id')}")

    return {
        "freeze": {"path": rel(freeze_path), "sha256": sha256_file(freeze_path)},
        "prediction_seal": {"path": rel(seal_path), "sha256": sha256_file(seal_path)},
        "overlap_audit": {"path": rel(overlap_path), "sha256": sha256_file(overlap_path)},
    }


def resolve_binary_label(frame: pd.DataFrame, label_contract: dict[str, Any]) -> pd.Series:
    human = int(label_contract["human"])
    ai = int(label_contract["ai"])
    require((human, ai) == (0, 1), "label contract must remain HUMAN=0, AI=1")

    if "binary_label" in frame.columns:
        y = pd.to_numeric(frame["binary_label"], errors="raise").astype(int)
        require(set(y.unique()).issubset({0, 1}), "binary_label outside {0,1}")
        return y

    require("label" in frame.columns, "external source has neither binary_label nor label")
    raw = frame["label"]

    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().all():
        y = numeric.astype(int)
        require(set(y.unique()).issubset({0, 1}), "numeric label outside {0,1}")
        return y

    mapped = raw.astype(str).str.strip().str.casefold().map(
        {
            "human": human,
            "ai": ai,
            "insan": human,
            "yz": ai,
        }
    )
    require(mapped.notna().all(), "unrecognized external label values")
    return mapped.astype(int)


def load_labeled_external(
    binding: dict[str, Any],
    label_contract: dict[str, Any],
    bootstrap_unit: str,
) -> pd.DataFrame:
    path = ROOT / binding["path"]
    require(path.is_file(), f"external source missing: {path}")
    require(sha256_file(path) == binding["sha256"], f"external source hash mismatch: {path}")

    frame = pd.read_csv(path)
    required = {"external_id", "text_sha256", bootstrap_unit}
    missing = sorted(required - set(frame.columns))
    require(not missing, f"external source missing columns {missing}: {path}")

    # For TUM/Bostancioglu the bootstrap unit IS text_sha256. Selecting
    # the column twice creates duplicate names and makes .str operations fail.
    projection = list(dict.fromkeys(["external_id", "text_sha256", bootstrap_unit]))
    out = frame.loc[:, projection].copy()
    out["binary_label"] = resolve_binary_label(frame, label_contract).to_numpy()

    out["external_id"] = out["external_id"].astype(str)
    out["text_sha256"] = out["text_sha256"].astype(str).str.lower()
    out[bootstrap_unit] = out[bootstrap_unit].astype(str)

    require(out["external_id"].is_unique, f"external_id not unique: {path}")
    require(set(out["binary_label"].unique()) == {0, 1}, f"both labels required: {path}")
    return out


def prediction_path(
    dataset_id: str,
    method: str,
    seed: int,
    outer_fold: int,
) -> Path:
    return (
        ROOT
        / PRESEAL_ROOT_REL
        / dataset_id
        / method
        / f"seed_{seed}"
        / f"outer_{outer_fold}.csv"
    )


def load_prediction(
    dataset_id: str,
    method: str,
    seed: int,
    outer_fold: int,
    labeled_external: pd.DataFrame,
) -> np.ndarray:
    path = prediction_path(dataset_id, method, seed, outer_fold)
    require(path.is_file(), f"sealed prediction missing: {path}")

    pred = pd.read_csv(
        path,
        usecols=[
            "external_id",
            "text_sha256",
            "method",
            "seed",
            "outer_fold",
            "raw_score_or_logit",
        ],
    )
    require(len(pred) == len(labeled_external), f"prediction row mismatch: {path}")
    require(pred["external_id"].astype(str).tolist() == labeled_external["external_id"].tolist(),
            f"prediction external_id identity/order mismatch: {path}")
    require(pred["text_sha256"].astype(str).str.lower().tolist()
            == labeled_external["text_sha256"].tolist(),
            f"prediction text_sha256 identity/order mismatch: {path}")
    require(pred["method"].astype(str).eq(method).all(), f"method mismatch: {path}")
    require(pd.to_numeric(pred["seed"], errors="raise").astype(int).eq(seed).all(),
            f"seed mismatch: {path}")
    require(pd.to_numeric(pred["outer_fold"], errors="raise").astype(int).eq(outer_fold).all(),
            f"outer_fold mismatch: {path}")

    score = pd.to_numeric(pred["raw_score_or_logit"], errors="raise").to_numpy(float)
    require(np.isfinite(score).all(), f"non-finite score: {path}")
    return score


def auc_or_nan(y: np.ndarray, score: np.ndarray) -> float:
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def seed_and_method_summary(
    cell_auc: dict[tuple[str, int, int], float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []

    for method in METHODS:
        seed_means: list[float] = []
        for seed in SEEDS:
            fold_values = np.asarray(
                [cell_auc[(method, seed, outer)] for outer in OUTER_FOLDS],
                dtype=float,
            )
            require(np.isfinite(fold_values).all(), f"non-finite fold AUROC for {method}/{seed}")
            seed_mean = float(fold_values.mean())
            seed_means.append(seed_mean)
            seed_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "mean_five_fold_auroc": seed_mean,
                    "min_fold_auroc": float(fold_values.min()),
                    "max_fold_auroc": float(fold_values.max()),
                }
            )

        arr = np.asarray(seed_means, dtype=float)
        method_rows.append(
            {
                "method": method,
                "mean_seed_mean_auroc": float(arr.mean()),
                "sample_sd_seed_means": float(arr.std(ddof=1)),
                "min_seed_mean_auroc": float(arr.min()),
                "max_seed_mean_auroc": float(arr.max()),
            }
        )

    return seed_rows, method_rows


COMPARISONS = (
    ("SUPCON", "CE_ONLY_MATCHED", "SUPCON_MINUS_CE"),
    ("LIT2026_REIMPL_V1_1", "CE_ONLY_MATCHED", "LIT_MINUS_CE"),
    ("LIT2026_REIMPL_V1_1", "SUPCON", "LIT_MINUS_SUPCON"),
)


def cluster_bootstrap_indices(
    units: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a cluster sample with replacement, keeping all rows per unit."""
    unique_units, inverse = np.unique(units, return_inverse=True)
    grouped_rows = [np.flatnonzero(inverse == i) for i in range(len(unique_units))]
    sampled = rng.integers(0, len(unique_units), size=len(unique_units))
    return np.concatenate([grouped_rows[i] for i in sampled])


def draw_shared_bootstrap_indices(
    y: np.ndarray,
    units: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[list[np.ndarray], int]:
    """Prespecified redraw of single-class draws, shared for all cells."""
    require(set(np.unique(y)) == {0, 1}, "bootstrap population lacks one class")
    require(len(y) == len(units), "bootstrap unit length mismatch")
    unique_units, inverse = np.unique(units, return_inverse=True)
    require(len(unique_units) > 1, "bootstrap requires multiple cluster units")
    grouped_rows = [np.flatnonzero(inverse == i) for i in range(len(unique_units))]
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    rejected = 0
    while len(draws) < replicates:
        sampled = rng.integers(0, len(unique_units), size=len(unique_units))
        idx = np.concatenate([grouped_rows[int(i)] for i in sampled])
        if np.unique(y[idx]).size < 2:
            rejected += 1
            continue
        draws.append(idx)
    return draws, rejected


def prepared_tied_rank_auc(
    y: np.ndarray,
    score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute exact score-tie groups for weighted Mann-Whitney AUROC."""
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    starts = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]
    return order, starts, y[order]


def weighted_tied_rank_auc(
    prepared: tuple[np.ndarray, np.ndarray, np.ndarray],
    row_weights: np.ndarray,
    total_positive: float,
    total_negative: float,
) -> float:
    """Exact ROC AUROC of a resample, without re-sorting duplicated rows.

    Weights are bootstrap cluster multiplicities. Equal decision scores are
    given half credit, matching sklearn.metrics.roc_auc_score tie behavior.
    """
    order, starts, sorted_y = prepared
    weights = row_weights[order]
    positive = weights * sorted_y
    negative = weights * (1 - sorted_y)
    pos_by_score = np.add.reduceat(positive, starts)
    neg_by_score = np.add.reduceat(negative, starts)
    negative_below = np.cumsum(neg_by_score) - neg_by_score
    return float(np.dot(pos_by_score, negative_below + 0.5 * neg_by_score)
                 / (total_positive * total_negative))


def bootstrap_cell_aucs(
    y: np.ndarray,
    units: np.ndarray,
    score_map: dict[tuple[str, int, int], np.ndarray],
) -> tuple[dict[tuple[str, int, int], np.ndarray], int]:
    """One shared external cluster resample across every matched model cell.

    The sorted ranks and tied-score groups are model-specific but fixed across
    replicates. Bootstrap cluster counts reproduce physical row duplication
    exactly. No seeds or folds are resampled.
    """
    require(set(np.unique(y)) == {0, 1}, "bootstrap population lacks one class")
    require(len(y) == len(units), "bootstrap unit length mismatch")
    unique_units, inverse = np.unique(units, return_inverse=True)
    require(len(unique_units) > 1, "bootstrap requires multiple units")

    prepared = {
        key: prepared_tied_rank_auc(y, score)
        for key, score in score_map.items()
    }
    out = {
        key: np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        for key in score_map
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    redraws = 0
    completed = 0
    while completed < BOOTSTRAP_REPLICATES:
        sampled_units = rng.integers(0, len(unique_units), size=len(unique_units))
        cluster_multiplicity = np.bincount(sampled_units, minlength=len(unique_units))
        row_weights = cluster_multiplicity[inverse].astype(float)
        total_positive = float(np.dot(row_weights, y))
        total_negative = float(np.dot(row_weights, 1 - y))
        if total_positive == 0 or total_negative == 0:
            redraws += 1
            continue

        for key, representation in prepared.items():
            auc = weighted_tied_rank_auc(
                representation, row_weights, total_positive, total_negative
            )
            require(np.isfinite(auc), f"non-finite bootstrap AUROC in cell {key}")
            out[key][completed] = auc
        completed += 1

    return out, redraws


def method_bootstrap_means(
    cell_boot: dict[tuple[str, int, int], np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        method: np.mean(
            [cell_boot[(method, seed, outer)]
             for seed in SEEDS for outer in OUTER_FOLDS],
            axis=0,
        )
        for method in METHODS
    }


def percentile_ci(values: np.ndarray) -> tuple[float, float, int]:
    require(len(values) == BOOTSTRAP_REPLICATES, "incomplete bootstrap replicates")
    require(np.isfinite(values).all(), "invalid bootstrap replicate in CI")
    low, high = np.percentile(values, [CI_LOW, CI_HIGH])
    return float(low), float(high), int(len(values))


def internal_prediction_path(method: str, seed: int, outer_fold: int) -> Path:
    if method == "LIT2026_REIMPL_V1_1":
        return (ROOT / "results/literature_comparison" / f"seed_{seed}"
                / f"outer_{outer_fold}_predictions.csv")
    objective = {"CE_ONLY_MATCHED": "ce", "SUPCON": "supcon"}[method]
    return (ROOT / "artifacts/checkpoints/external_cross_corpus"
            / "protocol_faithful_rerun"
            / f"{objective}_seed_{seed}_outer_{outer_fold}"
            / "internal_predictions.csv")


def load_internal_cell_aucs() -> tuple[dict[tuple[str, int, int], float], list[dict[str, Any]]]:
    """Never substitute historical CE/SupCon or development-pilot predictions."""
    aucs = {}
    bindings = []
    for method in METHODS:
        for seed in SEEDS:
            for outer in OUTER_FOLDS:
                prediction_path = internal_prediction_path(method, seed, outer)
                require(prediction_path.is_file(),
                        f"same-realization internal prediction missing: {prediction_path}")
                if method == "LIT2026_REIMPL_V1_1":
                    record_path = prediction_path.with_name(f"outer_{outer}_run.json")
                    record = read_json(record_path)
                    require(record.get("prediction_sha256") == sha256_file(prediction_path),
                            f"retained LIT2026 internal prediction hash mismatch: {prediction_path}")
                else:
                    record_path = prediction_path.with_name("run.json")
                    record = read_json(record_path)
                    require(record.get("status") == "SCIENTIFIC_PROTOCOL_FAITHFUL_RERUN",
                            f"development/non-scientific rerun rejected: {record_path}")
                    require(record.get("internal_prediction_sha256") == sha256_file(prediction_path),
                            f"rerun internal prediction hash mismatch: {prediction_path}")
                require(int(record.get("seed")) == seed and
                        int(record.get("outer_fold")) == outer,
                        f"internal cell identity mismatch: {record_path}")

                pred = pd.read_csv(prediction_path)
                require({"text_id", "pair_id", "label", "raw_decision_score"}
                        .issubset(pred.columns),
                        f"invalid internal prediction schema: {prediction_path}")
                require(len(pred) == 600 and pred["text_id"].astype(str).is_unique
                        and pred["pair_id"].astype(str).nunique() == 300,
                        f"internal test identity/count invalid: {prediction_path}")
                require(pd.to_numeric(pred["seed"], errors="raise").eq(seed).all(),
                        f"internal seed mismatch: {prediction_path}")
                require(pd.to_numeric(pred["outer_fold"], errors="raise").eq(outer).all(),
                        f"internal fold mismatch: {prediction_path}")
                y = resolve_binary_label(pred, {"human": 0, "ai": 1}).to_numpy(int)
                require(int(np.sum(y == 0)) == 300 and int(np.sum(y == 1)) == 300,
                        f"internal test class counts invalid: {prediction_path}")
                score = pd.to_numeric(pred["raw_decision_score"], errors="raise").to_numpy(float)
                require(np.isfinite(score).all(), f"internal scores non-finite: {prediction_path}")
                cell = (method, seed, outer)
                aucs[cell] = float(roc_auc_score(y, score))
                bindings.append({
                    "method": method, "seed": seed, "outer_fold": outer,
                    "prediction_path": rel(prediction_path),
                    "prediction_sha256": sha256_file(prediction_path),
                    "run_record_path": rel(record_path),
                    "run_record_sha256": sha256_file(record_path),
                    "internal_auroc": aucs[cell],
                })
    require(len(aucs) == 45, "expected 45 unique internal AUROC cells")
    return aucs, bindings


def exact_dedup_mask(
    cfg: dict[str, Any],
    dataset_id: str,
    primary: pd.DataFrame,
    label_contract: dict[str, Any],
) -> tuple[str, np.ndarray] | None:
    sens = [item for item in cfg["external_data_bindings"]["sensitivity"]
            if item["parent_primary"] == dataset_id
            and str(item["dataset_id"]).endswith("_exact_dedup")]
    if not sens:
        return None
    require(len(sens) == 1, f"ambiguous exact-dedup sensitivity for {dataset_id}")
    item = sens[0]
    path = ROOT / item["path"]
    require(path.is_file() and sha256_file(path) == item["sha256"],
            f"exact-dedup input hash mismatch: {path}")
    df = pd.read_csv(path)
    require(len(df) == int(item["rows"]), "exact-dedup input row count mismatch")
    require(df["external_id"].astype(str).is_unique, "dedup external ids not unique")
    require(df["external_id"].astype(str).isin(primary["external_id"]).all(),
            "dedup includes external ids outside primary")
    lookup = primary.set_index("external_id")
    ids = df["external_id"].astype(str)
    require((df["text_sha256"].astype(str).str.lower().to_numpy()
             == lookup.loc[ids, "text_sha256"].to_numpy()).all(),
            "dedup text hashes do not match the original population")
    require((resolve_binary_label(df, label_contract).to_numpy()
             == lookup.loc[ids, "binary_label"].to_numpy()).all(),
            "dedup labels do not match the original population")
    return str(item["dataset_id"]).upper(), primary["external_id"].isin(ids).to_numpy(bool)


def load_overlap_keep_ids(dataset_id: str, expected_ids: list[str]) -> set[str]:
    path = ROOT / OVERLAP_MANIFEST_ROOT_REL / f"{dataset_id}_exclusion_manifest.csv"
    require(path.is_file(), f"overlap exclusion manifest missing: {path}")
    manifest = pd.read_csv(
        path,
        usecols=["external_id", "exclude_from_overlap_excluded_sensitivity"],
    )
    manifest["external_id"] = manifest["external_id"].astype(str)
    require(manifest["external_id"].tolist() == expected_ids,
            f"overlap manifest identity/order mismatch: {path}")

    raw = manifest["exclude_from_overlap_excluded_sensitivity"]
    if raw.dtype == bool:
        excluded = raw
    else:
        excluded = raw.astype(str).str.strip().str.casefold().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        require(excluded.notna().all(), f"invalid exclusion boolean: {path}")

    return set(manifest.loc[~excluded.astype(bool), "external_id"].astype(str))


def analyze_population(
    dataset_id: str,
    population: str,
    external: pd.DataFrame,
    score_map: dict[tuple[str, int, int], np.ndarray],
    mask: np.ndarray,
    bootstrap_unit: str,
    internal_auc: dict[tuple[str, int, int], float],
) -> dict[str, pd.DataFrame]:
    y = external["binary_label"].to_numpy(int)[mask]
    units = external[bootstrap_unit].astype(str).to_numpy()[mask]
    if int(np.sum(y == 0)) < 2 or int(np.sum(y == 1)) < 2:
        # Predefined sensitivity NA rule: insufficient two-class retained sample.
        return {"status": pd.DataFrame([{
            "dataset_id": dataset_id, "population": population,
            "status": "NA_INSUFFICIENT_CLASSES",
            "rows": int(mask.sum()),
            "human_rows": int(np.sum(y == 0)),
            "ai_rows": int(np.sum(y == 1)),
        }])}

    masked_scores = {key: value[mask] for key, value in score_map.items()}
    cell_auc: dict[tuple[str, int, int], float] = {}
    cell_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in SEEDS:
            for outer in OUTER_FOLDS:
                key = (method, seed, outer)
                auc = auc_or_nan(y, masked_scores[key])
                require(np.isfinite(auc), f"AUROC undefined for {dataset_id}/{population}/{key}")
                cell_auc[key] = auc
                cell_rows.append({
                    "dataset_id": dataset_id, "population": population,
                    "method": method, "seed": seed, "outer_fold": outer,
                    "rows": int(mask.sum()),
                    "bootstrap_units": int(np.unique(units).size),
                    "auroc": auc,
                })
                transfer_rows.append({
                    "dataset_id": dataset_id, "population": population,
                    "method": method, "seed": seed, "outer_fold": outer,
                    "external_auroc": auc,
                    "same_realization_internal_auroc": internal_auc[key],
                    "external_minus_internal_auroc": auc - internal_auc[key],
                })

    seed_rows, method_rows = seed_and_method_summary(cell_auc)
    cell_boot, redraws = bootstrap_cell_aucs(y, units, masked_scores)
    boot_means = method_bootstrap_means(cell_boot)
    sample_info = {
        "dataset_id": dataset_id, "population": population,
        "rows": int(mask.sum()),
        "bootstrap_units": int(np.unique(units).size),
        "bootstrap_replicates_requested": BOOTSTRAP_REPLICATES,
        "bootstrap_replicates_valid": BOOTSTRAP_REPLICATES,
        "single_class_bootstrap_draws_redrawn": redraws,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }

    internal_method_mean = {
        method: float(np.mean([internal_auc[(method, seed, outer)]
                               for seed in SEEDS for outer in OUTER_FOLDS]))
        for method in METHODS
    }

    internal_shift_summaries: list[dict[str, Any]] = []
    for row in method_rows:
        method = row["method"]
        low, high, _valid = percentile_ci(boot_means[method])
        row.update({**sample_info,
                    "auroc_ci95_percentile_low": low,
                    "auroc_ci95_percentile_high": high})
        deltas = np.asarray([
            cell_auc[(method, seed, outer)] - internal_auc[(method, seed, outer)]
            for seed in SEEDS for outer in OUTER_FOLDS
        ])
        boot_deltas = boot_means[method] - internal_method_mean[method]
        d_low, d_high, _ = percentile_ci(boot_deltas)
        internal_shift_summaries.append({
            **sample_info, "method": method,
            "mean_external_minus_internal_auroc": float(deltas.mean()),
            "mean_same_realization_internal_auroc": internal_method_mean[method],
            "conditional_external_sample_ci95_low": d_low,
            "conditional_external_sample_ci95_high": d_high,
            "internal_auroc_held_fixed": True,
        })

    for row in seed_rows:
        row.update({"dataset_id": dataset_id, "population": population})

    comparison_cells: list[dict[str, Any]] = []
    comparison_summary: list[dict[str, Any]] = []
    for minuend, subtrahend, name in COMPARISONS:
        seed_means = []
        for seed in SEEDS:
            seed_differences = []
            for outer in OUTER_FOLDS:
                diff = cell_auc[(minuend, seed, outer)] - cell_auc[(subtrahend, seed, outer)]
                comparison_cells.append({
                    "dataset_id": dataset_id, "population": population,
                    "comparison": name, "minuend_method": minuend,
                    "subtrahend_method": subtrahend,
                    "seed": seed, "outer_fold": outer,
                    "matched_auroc_difference": diff,
                })
                seed_differences.append(diff)
            seed_means.append(float(np.mean(seed_differences)))
        pair_boot = boot_means[minuend] - boot_means[subtrahend]
        lo, hi, _ = percentile_ci(pair_boot)
        comparison_summary.append({
            **sample_info,
            "comparison": name,
            "minuend_method": minuend,
            "subtrahend_method": subtrahend,
            "mean_matched_cell_auc_difference": float(np.mean(seed_means)),
            "seed_42_mean_difference": seed_means[0],
            "seed_314_mean_difference": seed_means[1],
            "seed_2026_mean_difference": seed_means[2],
            "sample_sd_seed_mean_differences": float(np.std(seed_means, ddof=1)),
            "matched_difference_ci95_percentile_low": lo,
            "matched_difference_ci95_percentile_high": hi,
        })

    return {
        "cell": pd.DataFrame(cell_rows),
        "seed": pd.DataFrame(seed_rows),
        "method": pd.DataFrame(method_rows),
        "comparison_cell": pd.DataFrame(comparison_cells),
        "comparison_summary": pd.DataFrame(comparison_summary),
        "transfer_cell": pd.DataFrame(transfer_rows),
        "transfer_summary": pd.DataFrame(internal_shift_summaries),
        "status": pd.DataFrame([{
            **sample_info, "status": "COMPLETE",
            "human_rows": int(np.sum(y == 0)),
            "ai_rows": int(np.sum(y == 1)),
        }]),
    }


def run_statistics(config_path: Path) -> dict[str, Any]:
    cfg, bindings = verify_config(config_path)
    guards = verify_execution_guards(config_path)

    result_root = ROOT / RESULT_ROOT_REL
    output_names = {
        "cell": "cell_auroc.csv",
        "seed": "seed_mean_auroc.csv",
        "method": "method_auroc_summary.csv",
        "comparison_cell": "matched_cell_differences.csv",
        "comparison_summary": "matched_difference_summary.csv",
        "transfer_cell": "internal_transfer_shift_cells.csv",
        "transfer_summary": "internal_transfer_shift_summary.csv",
        "status": "population_analysis_status.csv",
    }
    paths = {name: result_root / filename for name, filename in output_names.items()}
    manifest_path = result_root / "STATISTICS_MANIFEST.json"
    for path in (manifest_path, *paths.values()):
        require(not path.exists(), f"refusing to overwrite existing statistics output: {path}")

    # These 45 internal values are the frozen reference observations for
    # descriptive transfer deltas and must NEVER be substituted with CE/SupCon
    # historical or development-pilot results.
    internal_auc, internal_bindings = load_internal_cell_aucs()
    label_contract = cfg["external_data_bindings"]["label_contract"]
    stats_cfg = cfg["statistics"]
    all_tables: dict[str, list[pd.DataFrame]] = {name: [] for name in output_names}

    for dataset_id in DATASET_IDS:
        bootstrap_unit = stats_cfg[f"{dataset_id}_unit"]
        external = load_labeled_external(bindings[dataset_id], label_contract, bootstrap_unit)
        score_map = {
            (method, seed, outer): load_prediction(
                dataset_id, method, seed, outer, external
            )
            for method in METHODS for seed in SEEDS for outer in OUTER_FOLDS
        }
        masks: list[tuple[str, np.ndarray]] = [
            ("PRIMARY_RELEASED", np.ones(len(external), dtype=bool))
        ]
        keep_ids = load_overlap_keep_ids(dataset_id, external["external_id"].tolist())
        masks.append(("OVERLAP_EXCLUDED_SENSITIVITY",
                      external["external_id"].isin(keep_ids).to_numpy(bool)))
        exact = exact_dedup_mask(cfg, dataset_id, external, label_contract)
        if exact is not None:
            masks.append(exact)

        for population, mask in masks:
            population_tables = analyze_population(
                dataset_id, population, external, score_map, mask,
                bootstrap_unit, internal_auc
            )
            for name, frame in population_tables.items():
                all_tables[name].append(frame)

    output_frames = {
        name: pd.concat(parts, ignore_index=True)
        for name, parts in all_tables.items()
    }
    for name in paths:
        require(name in output_frames, f"missing analysis table {name}")
        atomic_csv(output_frames[name], paths[name])

    manifest = {
        "schema_version": "external-cross-corpus-prespecified-statistics-v2",
        "status": "COMPLETE_PRESPECIFIED_EXTERNAL_AUROC_STATISTICS",
        "created_utc": utc_now(),
        "config": {"path": rel(config_path), "sha256": sha256_file(config_path)},
        "guards": guards,
        "implementation": {"path": rel(Path(__file__)), "sha256": sha256_file(Path(__file__))},
        "primary_metric": "auroc",
        "primary_external_pooling": False,
        "matched_differences": [item[2] for item in COMPARISONS],
        "exact_dedup_sensitivity_prespecified_only": True,
        "internal_transfer_difference": "external_auc - same_realization_internal_auc",
        "internal_transfer_bootstrap_internal_auc_held_fixed": True,
        "internal_bindings": internal_bindings,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "ci": "95% percentile",
            "single_class_draws_redrawn_and_counted": True,
            "shared_resamples_across_methods_realizations_and_matched_pairs": True,
            "seeds_resampled": False,
            "folds_resampled": False,
            "model_realizations_treated_as_independent_observations": False,
            "interpretation": "Estimation intervals only, not familywise confirmatory tests",
        },
        "conditional_calibration_and_threshold_metrics": (
            "NOT_COMPUTED_HERE; requires separate frozen mapping compatibility decision."
        ),
        "outputs": {
            name: {"path": rel(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    atomic_json(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "manifest_path": rel(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "outputs": {name: len(output_frames[name]) for name in output_names},
        "primary_external_pooling": False,
        "calibration_or_threshold_metrics_computed": False,
    }


def validate_only(config_path: Path) -> dict[str, Any]:
    cfg, bindings = verify_config(config_path)
    stats = cfg["statistics"]

    return {
        "status": "PASS_EXTERNAL_STATISTICS_STATIC_CONTRACT",
        "config_path": rel(config_path),
        "config_sha256": sha256_file(config_path),
        "implementation_path": rel(Path(__file__)),
        "implementation_sha256": sha256_file(Path(__file__)),
        "primary_metric": cfg["evaluation"]["primary_metric"],
        "primary_external_pooling": cfg["evaluation"]["primary_external_pooling"],
        "bootstrap_replicates": int(stats["bootstrap_replicates"]),
        "bootstrap_seed": int(stats["bootstrap_seed"]),
        "bootstrap_units": {
            dataset_id: stats[f"{dataset_id}_unit"]
            for dataset_id in DATASET_IDS
        },
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "outer_folds": list(OUTER_FOLDS),
        "primary_external_rows": {
            dataset_id: int(bindings[dataset_id]["rows"])
            for dataset_id in DATASET_IDS
        },
        "conditional_calibration_and_threshold_metrics": (
            "DEFERRED_TO_COMPATIBILITY_GATE"
        ),
        "matched_method_differences": [item[2] for item in COMPARISONS],
        "same_realization_internal_delta": True,
        "exact_dedup_sensitivities": True,
        "single_class_bootstrap_draws_redrawn": True,
        "note": (
            "Validation-only mode does not read predictions, labels, subgroup "
            "metadata, or performance results and does not require the final "
            "PREEXECUTION_FREEZE or prediction seal."
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

    result = validate_only(config_path) if args.validate_only else run_statistics(config_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
