# -*- coding: utf-8 -*-
"""Build the prespecified label-blind near-duplicate clusters and sensitivity split.

This script is intentionally result-blind: it never reads model predictions, metrics,
or labels for similarity/clustering. It requires METHOD_PREEXECUTION_FREEZE.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedGroupKFold

def _resolve_project_root() -> Path:
    for _candidate in Path(__file__).resolve().parents:
        if (_candidate / ".gitignore").is_file():
            return _candidate
    raise RuntimeError("Turkish AI-Generated Text Detection Study project root could not be resolved")

ROOT = _resolve_project_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
SCRIPTS = ROOT / "src"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from turkish_ai_detection.data.study import sha256_file, write_json  # noqa: E402
from turkish_ai_detection.reproducibility.reproduction_policy import validate_current_migrated_reproduction  # noqa: E402

CONFIG = ROOT / "configs" / "near_duplicate_sensitivity.json"
DATA = ROOT / "data" / "study" / "paired_study.csv"
METHOD_FREEZE = ROOT / "provenance" / "experiments" / "near_duplicate_sensitivity" / "METHOD_PREEXECUTION_FREEZE.json"
EDGE_OUT = ROOT / "data" / "study" / "near_duplicate_edges_v1.csv"
CLUSTER_OUT = ROOT / "data" / "study" / "near_duplicate_clusters_v1.csv"
SPLIT_OUT = ROOT / "data" / "study" / "cluster_aware_folds_v1.csv"
AUDIT_OUT = ROOT / "provenance" / "experiments" / "near_duplicate_sensitivity" / "CLUSTER_SPLIT_AUDIT.json"

TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_tokens(text: object) -> list[str]:
    value = unicodedata.normalize("NFC", str(text))
    value = value.translate(TR_LOWER).lower()
    return TOKEN_RE.findall(value)


def shingles(text: object, k: int = 5) -> list[str]:
    toks = normalize_tokens(text)
    if not toks:
        return []
    if len(toks) < k:
        return ["__SHORT__" + " ".join(toks)]
    return [" ".join(toks[i:i+k]) for i in range(len(toks) - k + 1)]


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def component_id(pair_ids: list[str]) -> str:
    payload = "\n".join(sorted(map(str, pair_ids))).encode("utf-8")
    return "ndc_" + hashlib.sha256(payload).hexdigest()[:16]


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def verify_method_freeze(cfg: dict) -> dict:
    if not METHOD_FREEZE.is_file():
        raise RuntimeError("near-duplicate method is locked until METHOD_PREEXECUTION_FREEZE.json exists")
    freeze = json.loads(METHOD_FREEZE.read_text(encoding="utf-8"))
    if freeze.get("status") != "METHOD_FROZEN_READY_TO_BUILD_CLUSTERS":
        raise RuntimeError("invalid near-duplicate method freeze status")
    freeze["_current_reproduction"] = validate_current_migrated_reproduction(
        ROOT, "near_duplicate_sensitivity", ROOT / "src/turkish_ai_detection/experiments/near_duplicate_sensitivity/run_fit.py"
    )
    if freeze["files"]["dataset"]["sha256"] != sha256_file(DATA):
        raise RuntimeError("dataset changed after method freeze")
    if sha256_file(DATA) != cfg["dataset_sha256"]:
        raise RuntimeError("dataset hash does not match prespecified config")
    return freeze


def build_similarity_edges(data: pd.DataFrame, threshold: float, k: int) -> tuple[pd.DataFrame, dict]:
    texts = data["text"].astype(str).tolist()
    analyzer = lambda doc: shingles(doc, k=k)
    vectorizer = CountVectorizer(analyzer=analyzer, binary=True, dtype=np.int32)
    x = vectorizer.fit_transform(texts)
    if not sparse.isspmatrix_csr(x):
        x = x.tocsr()
    sizes = np.asarray(x.sum(axis=1)).ravel().astype(np.int64)
    intersections = (x @ x.T).tocoo()

    pair_ids = data["pair_id"].astype(str).to_numpy()
    text_ids = data["text_id"].astype(str).to_numpy()
    rows = []
    same_pair_high = 0
    cross_pair_candidates = 0
    for i, j, inter in zip(intersections.row, intersections.col, intersections.data):
        if i >= j or inter <= 0:
            continue
        union = int(sizes[i] + sizes[j] - int(inter))
        if union <= 0:
            continue
        sim = float(inter) / float(union)
        if sim < threshold:
            continue
        if pair_ids[i] == pair_ids[j]:
            same_pair_high += 1
            continue
        cross_pair_candidates += 1
        rows.append({
            "text_id_a": text_ids[i],
            "text_id_b": text_ids[j],
            "pair_id_a": pair_ids[i],
            "pair_id_b": pair_ids[j],
            "intersection_shingles": int(inter),
            "union_shingles": union,
            "exact_jaccard": sim,
        })
    edge = pd.DataFrame(rows, columns=[
        "text_id_a", "text_id_b", "pair_id_a", "pair_id_b",
        "intersection_shingles", "union_shingles", "exact_jaccard",
    ])
    if not edge.empty:
        edge = edge.sort_values(
            ["exact_jaccard", "pair_id_a", "pair_id_b", "text_id_a", "text_id_b"],
            ascending=[False, True, True, True, True],
        ).reset_index(drop=True)
    stats = {
        "text_rows": int(len(data)),
        "vocabulary_size_5_shingles": int(len(vectorizer.vocabulary_)),
        "nonzero_document_shingle_sets": int(np.sum(sizes > 0)),
        "same_pair_high_similarity_row_pairs_ignored": int(same_pair_high),
        "cross_pair_high_similarity_row_edges": int(cross_pair_candidates),
    }
    return edge, stats


def build_components(data: pd.DataFrame, edge: pd.DataFrame) -> pd.DataFrame:
    pair_meta = data.drop_duplicates("pair_id")[["pair_id", "domain", "generator"]].copy()
    if len(pair_meta) != 1500 or not pair_meta["pair_id"].is_unique:
        raise RuntimeError("expected exactly 1500 unique pair_ids")
    pair_meta["pair_id"] = pair_meta["pair_id"].astype(str)
    uf = UnionFind(pair_meta["pair_id"].tolist())
    for row in edge.itertuples(index=False):
        uf.union(str(row.pair_id_a), str(row.pair_id_b))

    groups = defaultdict(list)
    for pid in pair_meta["pair_id"]:
        groups[uf.find(pid)].append(pid)
    mapping = {}
    size_map = {}
    for members in groups.values():
        cid = component_id(members)
        for pid in members:
            mapping[pid] = cid
            size_map[pid] = len(members)

    pair_meta["near_duplicate_cluster_id"] = pair_meta["pair_id"].map(mapping)
    pair_meta["cluster_pair_count"] = pair_meta["pair_id"].map(size_map).astype(int)
    pair_meta["is_multi_pair_cluster"] = pair_meta["cluster_pair_count"].gt(1)
    pair_meta["stratum"] = pair_meta["domain"].astype(str) + "×" + pair_meta["generator"].astype(str)
    return pair_meta.sort_values("pair_id").reset_index(drop=True)


def build_split(cluster_frame: pd.DataFrame, n_splits: int, seed: int) -> tuple[pd.DataFrame, dict]:
    frame = cluster_frame.copy().reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    frame["outer_fold"] = -1
    x_dummy = np.zeros((len(frame), 1), dtype=np.int8)
    y = frame["stratum"].astype(str).to_numpy()
    groups = frame["near_duplicate_cluster_id"].astype(str).to_numpy()
    try:
        for fold, (_train_idx, test_idx) in enumerate(splitter.split(x_dummy, y, groups)):
            frame.loc[test_idx, "outer_fold"] = int(fold)
    except Exception as exc:
        raise RuntimeError(f"NOT_FEASIBLE_AS_PRESPECIFIED: StratifiedGroupKFold failed: {exc}") from exc

    if set(frame["outer_fold"].astype(int)) != set(range(n_splits)):
        raise RuntimeError("NOT_FEASIBLE_AS_PRESPECIFIED: incomplete five-fold assignment")
    if frame.groupby("near_duplicate_cluster_id")["outer_fold"].nunique().max() != 1:
        raise RuntimeError("cluster integrity failure: component split across outer folds")

    all_strata = set(frame["stratum"].astype(str))
    fold_records = []
    all_strata_present = True
    for fold in range(n_splits):
        part = frame[frame.outer_fold.eq(fold)]
        seen = set(part["stratum"].astype(str))
        if seen != all_strata:
            all_strata_present = False
        counts = {
            str(k): int(v)
            for k, v in part["stratum"].value_counts().sort_index().items()
        }
        fold_records.append({
            "outer_fold": int(fold),
            "pair_count": int(len(part)),
            "row_count": int(2 * len(part)),
            "cluster_count": int(part["near_duplicate_cluster_id"].nunique()),
            "stratum_pair_counts": counts,
            "all_ten_strata_present": seen == all_strata,
        })
    if not all_strata_present:
        raise RuntimeError(
            "NOT_FEASIBLE_AS_PRESPECIFIED: at least one outer fold lacks one or more domain×generator strata; "
            "do not alter threshold/representation/random_state"
        )
    return frame, {"fold_records": fold_records, "all_ten_strata_present_in_every_fold": True}


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    freeze = verify_method_freeze(cfg)
    for path in (EDGE_OUT, CLUSTER_OUT, SPLIT_OUT, AUDIT_OUT):
        if path.exists():
            raise RuntimeError(f"refusing overwrite of prespecified clustering artifact: {path}")

    data = pd.read_csv(DATA).copy()
    needed = {"text_id", "pair_id", "domain", "generator", "text"}
    if not needed <= set(data.columns):
        raise RuntimeError(f"dataset schema missing: {sorted(needed - set(data.columns))}")
    if len(data) != 3000 or data["text_id"].nunique() != 3000 or data["pair_id"].nunique() != 1500:
        raise RuntimeError("Combined-v2 identity/count mismatch")
    if data["text"].isna().any() or data["text"].astype(str).str.strip().eq("").any():
        raise RuntimeError("empty text encountered")

    method = cfg["method"]
    threshold = float(method["similarity"]["threshold"])
    k = int(method["representation"]["shingle_size"])
    if threshold != 0.8 or k != 5:
        raise RuntimeError("prespecified near-duplicate method changed")

    edge, sim_stats = build_similarity_edges(data, threshold=threshold, k=k)
    clusters = build_components(data, edge)
    split, split_stats = build_split(
        clusters,
        n_splits=int(cfg["cluster_aware_split"]["n_splits"]),
        seed=int(cfg["cluster_aware_split"]["random_state"]),
    )

    atomic_csv(edge, EDGE_OUT)
    atomic_csv(clusters, CLUSTER_OUT)
    split_cols = [
        "pair_id", "domain", "generator", "stratum",
        "near_duplicate_cluster_id", "cluster_pair_count", "is_multi_pair_cluster", "outer_fold",
    ]
    atomic_csv(split[split_cols], SPLIT_OUT)

    cluster_sizes = clusters.drop_duplicates("near_duplicate_cluster_id")[
        ["near_duplicate_cluster_id", "cluster_pair_count"]
    ]["cluster_pair_count"].to_numpy(int)
    multi = clusters[clusters.is_multi_pair_cluster]
    audit = {
        "schema_version": "combined-v2-near-duplicate-cluster-split-audit-v1",
        "created_utc": utc_now(),
        "status": "PASS",
        "model_outcomes_consulted": False,
        "labels_consulted_for_clustering": False,
        "method_freeze_sha256": sha256_file(METHOD_FREEZE),
        "config_sha256": sha256_file(CONFIG),
        "builder_sha256": sha256_file(Path(__file__)),
        "reproduction_mode": freeze["_current_reproduction"]["reproduction_mode"],
        "historical_exact_reproduction": freeze["_current_reproduction"]["historical_exact_reproduction"],
        "dataset_sha256": sha256_file(DATA),
        "method": cfg["method"],
        "similarity_stats": sim_stats,
        "cluster_stats": {
            "total_pairs": int(len(clusters)),
            "total_components_including_singletons": int(clusters.near_duplicate_cluster_id.nunique()),
            "multi_pair_cluster_count": int(
                clusters.loc[clusters.cluster_pair_count.gt(1), "near_duplicate_cluster_id"].nunique()
            ),
            "pairs_in_multi_pair_clusters": int(len(multi)),
            "singleton_pair_count": int(np.sum(clusters.cluster_pair_count.eq(1))),
            "largest_cluster_pair_count": int(cluster_sizes.max()),
            "mean_cluster_pair_count": float(cluster_sizes.mean()),
        },
        "split_stats": split_stats,
        "files": {
            "edges": {"path": EDGE_OUT.relative_to(ROOT).as_posix(), "sha256": sha256_file(EDGE_OUT)},
            "clusters": {"path": CLUSTER_OUT.relative_to(ROOT).as_posix(), "sha256": sha256_file(CLUSTER_OUT)},
            "split": {"path": SPLIT_OUT.relative_to(ROOT).as_posix(), "sha256": sha256_file(SPLIT_OUT)},
        },
        "next_step": "Review cluster/split audit only. Do not modify method. If PASS, create the scientific split freeze before any RQ5 model fit.",
    }
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(AUDIT_OUT, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
