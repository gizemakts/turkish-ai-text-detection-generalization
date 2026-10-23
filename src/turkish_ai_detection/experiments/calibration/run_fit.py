# -*- coding: utf-8 -*-
"""Single inner-fit runner for COMBINED_V2_CALIBRATION_V1.

Trains exactly one objective/seed/outer/inner model on 900 outer-training pairs
and scores the held-out 300 outer-training pairs. It never reads outer-test
labels or outer-test predictions and never computes calibration/test metrics.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

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

from turkish_ai_detection.data.study import resolve_frozen_record_path, sha256_file, write_json  # noqa: E402
from turkish_ai_detection.reproducibility.reproduction_policy import validate_current_migrated_reproduction  # noqa: E402
from turkish_ai_detection.evaluation.metrics import condition_text  # noqa: E402
from turkish_ai_detection.experiments.baseline.berturk import MODEL_NAME, REVISION, encode_head_tail  # noqa: E402
from turkish_ai_detection.experiments.model_development.supcon.train import (  # noqa: E402
    BoundedCollator, DATASET_SHA256, DATA_PATH, FOLD_PATH, FOLD_SHA256,
    build_model, verify_inputs,
)
from turkish_ai_detection.experiments.primary.run_fit import (  # noqa: E402
    ALLOWED_OUTERS, ALLOWED_SEEDS, atomic_csv, make_dataset,
    make_seeded_trainer_class, pair_hash, score, training_arguments,
)

EXPERIMENT = "COMBINED_V2_CALIBRATION_V1"
CONDITION = "A"
CONFIG = ROOT / "configs" / "calibration.json"
FREEZE = ROOT / "provenance" / "experiments" / "calibration" / "PREEXECUTION_FREEZE.json"
OUT = ROOT / "results" / "calibration"
STATE = ROOT / "artifacts" / "checkpoints" / "calibration"
INNER_FOLDS = (0, 1, 2, 3)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_freeze() -> dict:
    if not FREEZE.is_file():
        raise RuntimeError("RQ4 execution locked until PREEXECUTION_FREEZE.json exists")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_READY_TO_RUN":
        raise RuntimeError("invalid RQ4 pre-execution freeze status")
    freeze["_current_reproduction"] = validate_current_migrated_reproduction(
        ROOT, "calibration", Path(__file__)
    )
    if freeze.get("scientific_rq4_outputs_present_before_freeze") is not False:
        raise RuntimeError("freeze does not certify clean RQ4 pre-execution state")
    if freeze.get("calibration_results_unblinded_before_freeze") is not False:
        raise RuntimeError("freeze does not certify calibration-result blinding")
    return freeze


def run_one(objective: str, seed: int, outer: int, inner: int) -> dict:
    import torch
    import transformers
    from transformers import AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("RQ4 inner fits require CUDA")
    if objective not in {"ce", "supcon"}:
        raise ValueError(objective)
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed not prespecified: {seed}")
    if outer not in ALLOWED_OUTERS:
        raise ValueError(f"outer fold not prespecified: {outer}")
    if inner not in INNER_FOLDS:
        raise ValueError(f"inner fold not prespecified: {inner}")

    freeze = verify_freeze()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    observed = verify_inputs()
    if observed["dataset"] != DATASET_SHA256 or observed["folds"] != FOLD_SHA256:
        raise RuntimeError("frozen dataset/fold identity mismatch")
    if cfg["dataset_sha256"] != DATASET_SHA256 or cfg["fold_sha256"] != FOLD_SHA256:
        raise RuntimeError("RQ4 config dataset/fold hash mismatch")

    data = pd.read_csv(DATA_PATH).copy()
    folds = pd.read_csv(FOLD_PATH)
    if len(data) != 3000 or data.text_id.nunique() != 3000 or folds.pair_id.nunique() != 1500:
        raise RuntimeError("Combined-v2 identity/count changed")
    data["model_text"] = condition_text(data, CONDITION)

    inner_col = f"inner_fold_outer{outer}"
    outer_test_pairs = set(folds.loc[folds.outer_fold.eq(outer), "pair_id"])
    outer_train_pairs = set(folds.loc[folds.outer_fold.ne(outer), "pair_id"])
    held_pairs = set(
        folds.loc[folds.outer_fold.ne(outer) & folds[inner_col].eq(inner), "pair_id"]
    )
    fit_pairs = outer_train_pairs - held_pairs

    if len(outer_train_pairs) != 1200 or len(outer_test_pairs) != 300:
        raise RuntimeError("unexpected outer pair counts")
    if len(fit_pairs) != 900 or len(held_pairs) != 300:
        raise RuntimeError("unexpected inner pair counts")
    if fit_pairs & held_pairs or (fit_pairs | held_pairs) != outer_train_pairs:
        raise RuntimeError("invalid inner split partition")
    if (fit_pairs | held_pairs) & outer_test_pairs:
        raise RuntimeError("outer-test leakage into RQ4 inner fit")

    fit = data[data.pair_id.isin(fit_pairs)].copy()
    held = data[data.pair_id.isin(held_pairs)].copy()
    if len(fit) != 1800 or len(held) != 600:
        raise RuntimeError("unexpected inner row counts")

    base = OUT / "inner_oof" / objective / f"seed_{seed}"
    pred_path = base / f"outer_{outer}_inner_{inner}_predictions.csv"
    sidecar_path = base / f"outer_{outer}_inner_{inner}_run.json"
    tag = f"{objective}_seed_{seed}_outer_{outer}_inner_{inner}"

    binding = {
        "experiment": EXPERIMENT,
        "objective": objective,
        "seed": int(seed),
        "outer_fold": int(outer),
        "inner_fold": int(inner),
        "condition": CONDITION,
        "dataset_sha256": DATASET_SHA256,
        "fold_sha256": FOLD_SHA256,
        "config_sha256": sha256_file(CONFIG),
        "runner_sha256": sha256_file(Path(__file__)),
        "preexecution_freeze_sha256": sha256_file(FREEZE),
        "reproduction_mode": freeze["_current_reproduction"]["reproduction_mode"],
        "historical_exact_reproduction": freeze["_current_reproduction"]["historical_exact_reproduction"],
        "outer_train_pair_count": 1200,
        "outer_test_pair_count": 300,
        "fit_pair_count": 900,
        "heldout_pair_count": 300,
        "fit_pair_sha256": pair_hash(fit_pairs),
        "heldout_pair_sha256": pair_hash(held_pairs),
        "outer_test_fit_overlap": 0,
        "outer_test_heldout_overlap": 0,
    }

    if pred_path.exists() or sidecar_path.exists():
        if not pred_path.exists() or not sidecar_path.exists():
            raise RuntimeError(f"incomplete resume state: {tag}")
        saved = json.loads(sidecar_path.read_text(encoding="utf-8"))
        expected = {**binding, "prediction_sha256": sha256_file(pred_path)}
        if {k: saved.get(k) for k in expected} != expected:
            raise RuntimeError(f"resume binding mismatch: {tag}")
        return saved

    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)
    encodings = [encode_head_tail(tokenizer, text) for text in data.model_text.astype(str)]
    collator = BoundedCollator(tokenizer)
    model = build_model()
    TrainerClass = make_seeded_trainer_class(objective, seed)
    trainer = TrainerClass(
        model=model,
        args=training_arguments(STATE / tag, seed),
        train_dataset=make_dataset(fit, encodings),
        data_collator=collator,
    )
    trainer.train()
    if trainer.last_group_audit is None:
        raise RuntimeError("no completed true-32 objective group observed")

    raw = score(model, held, encodings, collator)
    pred = held[["text_id", "pair_id", "label", "domain", "generator"]].copy()
    pred["outer_fold"] = int(outer)
    pred["inner_fold"] = int(inner)
    pred["seed"] = int(seed)
    pred["objective"] = objective
    pred["condition"] = CONDITION
    pred["raw_decision_score"] = raw

    if len(pred) != 600 or pred.text_id.nunique() != 600 or pred.pair_id.nunique() != 300:
        raise RuntimeError("invalid RQ4 inner prediction shape")
    if not np.isfinite(pred.raw_decision_score.to_numpy(float)).all():
        raise RuntimeError("non-finite RQ4 inner raw score")
    atomic_csv(pred, pred_path)

    record = {
        **binding,
        "schema_version": "combined-v2-calibration-inner-fit-v1",
        "created_utc": utc_now(),
        "status": "COMPLETE",
        "model": MODEL_NAME,
        "revision": REVISION,
        "calibration_fit": False,
        "threshold_fit": False,
        "scientific_outer_test_rows_read": 0,
        "last_group_audit": trainer.last_group_audit,
        "prediction_sha256": sha256_file(pred_path),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_model": torch.cuda.get_device_name(0),
    }
    write_json(sidecar_path, record)

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", required=True, choices=["ce", "supcon"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--outer-fold", required=True, type=int)
    parser.add_argument("--inner-fold", required=True, type=int)
    args = parser.parse_args()
    record = run_one(args.objective, args.seed, args.outer_fold, args.inner_fold)
    safe = {
        "status": record["status"],
        "experiment": record["experiment"],
        "objective": record["objective"],
        "seed": record["seed"],
        "outer_fold": record["outer_fold"],
        "inner_fold": record["inner_fold"],
        "prediction_sha256": record["prediction_sha256"],
        "scientific_metrics_computed": False,
    }
    print(json.dumps(safe, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
