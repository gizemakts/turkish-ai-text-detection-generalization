from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

import pandas as pd


def project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".gitignore").is_file():
            return candidate
    raise RuntimeError("Turkish AI-Generated Text Detection Study project root could not be resolved")


ROOT = project_root()
DATA = ROOT / "data" / "study" / "paired_study.csv"
PROV = ROOT / "provenance" / "experiments" / "author_disjoint_newspaper_20261013"
FOLDS = PROV / "author_disjoint_folds.csv"
FREEZE = PROV / "AUTHOR_DISJOINT_FREEZE_MANIFEST.json"
CONFIG = ROOT / "results" / "author_disjoint_newspaper" / "FROZEN_EXPERIMENT_CONFIG.json"
OUT = ROOT / "results" / "author_disjoint_newspaper"
SEEDS = (42, 314, 2026)
OUTERS = (0, 1, 2, 3, 4)
OBJECTIVES = ("ce", "supcon")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_hash(values) -> str:
    payload = "\n".join(sorted(map(str, values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def load_config_and_freeze() -> tuple[dict, dict]:
    require(CONFIG.is_file(), f"missing frozen config: {CONFIG}")
    require(FREEZE.is_file(), f"missing freeze manifest: {FREEZE}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    require(freeze.get("status") == "FROZEN_NOT_EXECUTED", "freeze is not FROZEN_NOT_EXECUTED")
    require(config.get("status") == "FROZEN_NOT_EXECUTED", "config is not FROZEN_NOT_EXECUTED")
    require(tuple(config["seeds"]) == SEEDS, "seed plan changed")
    require(tuple(config["outer_folds"]) == OUTERS, "outer-fold plan changed")
    require(config["planned_training_runs"] == 30, "planned run count changed")
    for key, path in (("source_dataset", DATA), ("fold_assignment", FOLDS), ("frozen_config", CONFIG)):
        require(freeze["files"][key]["sha256"] == sha256_file(path), f"freeze hash mismatch: {key}")
    for key, record in freeze.get("files", {}).items():
        path = ROOT / record["path"]
        require(path.is_file(), f"missing frozen file: {key}")
        require(record["sha256"] == sha256_file(path), f"frozen file changed: {key}")
    for key, record in freeze.get("authoritative_reused_files", {}).items():
        path = ROOT / record["path"]
        require(path.is_file(), f"missing authoritative reused file: {key}")
        require(record["sha256"] == sha256_file(path), f"authoritative reused file changed: {key}")
    for key, record in freeze.get("implementation_files", {}).items():
        path = ROOT / record["path"]
        require(path.is_file(), f"missing frozen implementation: {key}")
        require(record["sha256"] == sha256_file(path), f"implementation changed after freeze: {key}")
    return config, freeze


def load_data_and_folds() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(DATA).copy()
    folds = pd.read_csv(FOLDS).copy()
    folds["author"] = folds["author"].astype(str).map(lambda x: unicodedata.normalize("NFC", x))
    return data, folds


def split_pair_ids(data: pd.DataFrame, folds: pd.DataFrame, outer: int) -> tuple[set[str], set[str]]:
    test_pairs = set(folds.loc[folds.author_fold.eq(outer), "pair_id"].astype(str))
    all_pairs = set(data.pair_id.astype(str))
    train_pairs = all_pairs - test_pairs
    require(not (train_pairs & test_pairs), "pair leakage")
    return train_pairs, test_pairs

