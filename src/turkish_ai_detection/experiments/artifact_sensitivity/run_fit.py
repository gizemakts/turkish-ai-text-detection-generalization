# -*- coding: utf-8 -*-
"""Single-fit runner for the frozen Condition-B sensitivity experiment."""
from __future__ import annotations

import argparse
import gc
import json
import sys
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

from turkish_ai_detection.data.study import sha256_file, write_json  # noqa: E402
from turkish_ai_detection.reproducibility.reproduction_policy import validate_current_migrated_reproduction  # noqa: E402
from turkish_ai_detection.evaluation.metrics import condition_text  # noqa: E402
from turkish_ai_detection.experiments.baseline.berturk import MODEL_NAME, REVISION, encode_head_tail  # noqa: E402
from turkish_ai_detection.experiments.model_development.supcon.train import (  # noqa: E402
    BoundedCollator, DATASET_SHA256, DATA_PATH, FOLD_PATH, FOLD_SHA256, build_model,
)
from turkish_ai_detection.experiments.primary.run_fit import (  # noqa: E402
    atomic_csv, make_dataset, make_seeded_trainer_class, metric_record,
    pair_hash, score, training_arguments, utc_now,
)

EXPERIMENT = "CONDITION_B_SENSITIVITY_V1"
CONDITION = "B"
CONFIG_PATH = ROOT / "configs" / "artifact_sensitivity.json"
OUT = ROOT / "results" / "artifact_sensitivity"
PROV = ROOT / "provenance" / "experiments" / "artifact_sensitivity"
STATE = ROOT / "artifacts" / "checkpoints" / "artifact_sensitivity"
FREEZE_PATH = PROV / "PREEXECUTION_FREEZE.json"

PRIMARY_RUNNER = ROOT / "src" / "turkish_ai_detection/experiments/primary/run_fit.py"
PRIMARY_CONFIG = ROOT / "configs" / "primary_evaluation.json"
PRIMARY_FREEZE = ROOT / "provenance" / "experiments" / "primary" / "PREEXECUTION_FREEZE.json"
PRIMARY_MANIFEST = ROOT / "results" / "primary" / "final_statistics" / "final_statistics_manifest.json"

EVALUATE = ROOT / "src" / "turkish_ai_detection/evaluation/metrics.py"
CORE = ROOT / "src" / "turkish_ai_detection/features/core.py"
SUPCON_V2 = ROOT / "src" / "turkish_ai_detection/experiments/model_development/supcon/train.py"
M5 = ROOT / "src" / "turkish_ai_detection/experiments/baseline/berturk.py"

PRIMARY_RUNNER_SHA256 = "eafe1e66c5d8324cf547571144a1a2f70f9cf1b4406c22f9eabd4e0b8587b92e"
PRIMARY_CONFIG_SHA256 = "7e825c0f97f1763c6b4db69164feda04743f9fa30f27e254e1c52f953dd07f13"
PRIMARY_FREEZE_SHA256 = "162a120d624c830e3b0f93a07504a30be58ac602f4089aa2349527d52882548e"
PRIMARY_MANIFEST_SHA256 = "ed30975ecb16f49715352f6b11ef1c29aa6e4df2e0d2a0bf55f93a69fcc2187a"

ALLOWED_SEEDS = (42, 314, 2026)
ALLOWED_OUTERS = (0, 1, 2, 3, 4)


def verify_lineage_and_freeze():
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    deps = cfg["dependency_lineage"]

    expected = [
        (PRIMARY_RUNNER, PRIMARY_RUNNER_SHA256, "primary runner"),
        (PRIMARY_CONFIG, PRIMARY_CONFIG_SHA256, "primary config"),
        (PRIMARY_FREEZE, PRIMARY_FREEZE_SHA256, "primary freeze"),
        (PRIMARY_MANIFEST, PRIMARY_MANIFEST_SHA256, "primary final manifest"),
        (EVALUATE, deps["evaluation_metrics_sha256"], "evaluation metrics"),
        (CORE, deps["turkish_ai_detection_core_sha256"], "turkish_ai_detection_core"),
        (SUPCON_V2, deps["supcon_runner_sha256"], "SupCon V2 runner"),
        (M5, deps["berturk_helpers_sha256"], "BERTurk/tokenization dependency"),
    ]
    for path, sha, label in expected:
        if not path.is_file():
            raise RuntimeError(f"missing {label}: {path}")
        if sha256_file(path) != sha:
            raise RuntimeError(f"{label} hash mismatch")

    if not FREEZE_PATH.is_file():
        raise RuntimeError("Condition-B training locked until PREEXECUTION_FREEZE.json exists")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_READY_TO_RUN":
        raise RuntimeError("invalid Condition-B freeze")
    freeze["_current_reproduction"] = validate_current_migrated_reproduction(
        ROOT, "artifact_sensitivity", Path(__file__)
    )
    if freeze.get("condition_b_results_unblinded_before_freeze") is not False:
        raise RuntimeError("freeze does not certify blinded Condition-B state")
    return freeze, cfg


def run_one(objective: str, seed: int, outer: int) -> dict:
    import torch
    import transformers
    from transformers import AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Condition-B run requires CUDA")
    if objective not in {"ce", "supcon"}:
        raise ValueError(objective)
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed not prespecified: {seed}")
    if outer not in ALLOWED_OUTERS:
        raise ValueError(f"outer fold not prespecified: {outer}")

    freeze, cfg = verify_lineage_and_freeze()
    if sha256_file(DATA_PATH) != DATASET_SHA256 or sha256_file(FOLD_PATH) != FOLD_SHA256:
        raise RuntimeError("frozen dataset/fold identity changed")
    if cfg["dataset_sha256"] != DATASET_SHA256 or cfg["fold_sha256"] != FOLD_SHA256:
        raise RuntimeError("Condition-B config identity mismatch")

    data = pd.read_csv(DATA_PATH).copy()
    folds = pd.read_csv(FOLD_PATH)
    if len(data) != 3000 or data.text_id.nunique() != 3000 or data.pair_id.nunique() != 1500:
        raise RuntimeError("Combined-v2 identity/count changed")
    data["model_text"] = condition_text(data, CONDITION)
    if data["model_text"].isna().any() or (data["model_text"].astype(str).str.len() == 0).any():
        raise RuntimeError("Condition-B transform produced empty/NA scientific text")

    test_pairs = set(folds.loc[folds.outer_fold.eq(outer), "pair_id"])
    train_pairs = set(folds.loc[folds.outer_fold.ne(outer), "pair_id"])
    if train_pairs & test_pairs:
        raise RuntimeError("outer train/test pair leakage")

    train = data[data.pair_id.isin(train_pairs)].copy()
    test = data[data.pair_id.isin(test_pairs)].copy()
    if len(train) != 2400 or len(test) != 600:
        raise RuntimeError("unexpected Condition-B train/test row count")

    tag = f"{objective}_seed_{seed}_outer_{outer}"
    out_dir = OUT / objective / f"seed_{seed}"
    pred_path = out_dir / f"outer_{outer}_predictions.csv"
    sidecar_path = out_dir / f"outer_{outer}_run.json"

    binding = {
        "experiment": EXPERIMENT,
        "objective": objective,
        "seed": int(seed),
        "outer_fold": int(outer),
        "condition": CONDITION,
        "dataset_sha256": DATASET_SHA256,
        "fold_sha256": FOLD_SHA256,
        "config_sha256": sha256_file(CONFIG_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "preexecution_freeze_sha256": sha256_file(FREEZE_PATH),
        "reproduction_mode": freeze["_current_reproduction"]["reproduction_mode"],
        "historical_exact_reproduction": freeze["_current_reproduction"]["historical_exact_reproduction"],
        "primary_runner_sha256": sha256_file(PRIMARY_RUNNER),
        "primary_final_manifest_sha256": sha256_file(PRIMARY_MANIFEST),
        "condition_transform_dependency_sha256": sha256_file(CORE),
        "train_pair_count": int(len(train_pairs)),
        "test_pair_count": int(len(test_pairs)),
        "train_pair_sha256": pair_hash(train_pairs),
        "test_pair_sha256": pair_hash(test_pairs),
        "train_test_pair_overlap": 0,
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
        train_dataset=make_dataset(train, encodings),
        data_collator=collator,
    )
    trainer.train()
    if trainer.last_group_audit is None:
        raise RuntimeError("no completed true-32 objective group observed")

    raw = score(model, test, encodings, collator)
    prediction = test[["text_id", "pair_id", "label", "domain", "generator"]].copy()
    prediction["outer_fold"] = int(outer)
    prediction["seed"] = int(seed)
    prediction["objective"] = objective
    prediction["condition"] = CONDITION
    prediction["raw_decision_score"] = raw

    if len(prediction) != 600 or not prediction.text_id.is_unique:
        raise RuntimeError("invalid Condition-B prediction shape")
    if not np.isfinite(prediction.raw_decision_score.to_numpy(float)).all():
        raise RuntimeError("non-finite Condition-B score")

    atomic_csv(prediction, pred_path)
    metrics = metric_record(prediction)

    record = {
        **binding,
        "schema_version": "combined-v2-condition-b-single-fit-v1",
        "created_utc": utc_now(),
        "status": "COMPLETE",
        "model": MODEL_NAME,
        "revision": REVISION,
        "train_rows": len(train),
        "test_rows": len(test),
        "outer_test_rows_used_for_training": 0,
        "outer_test_used_for_model_selection": False,
        "calibration_used": False,
        "threshold_used": False,
        "lambda_supcon": 0.0 if objective == "ce" else 0.1,
        "temperature": None if objective == "ce" else 0.07,
        "last_group_audit": trainer.last_group_audit,
        "metrics": metrics,
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
    p = argparse.ArgumentParser()
    p.add_argument("--objective", required=True, choices=["ce", "supcon"])
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--outer-fold", required=True, type=int, choices=list(ALLOWED_OUTERS))
    a = p.parse_args()
    record = run_one(a.objective, a.seed, a.outer_fold)
    safe = {
        "status": record["status"],
        "experiment": record["experiment"],
        "objective": record["objective"],
        "seed": record["seed"],
        "outer_fold": record["outer_fold"],
        "condition": "B",
        "prediction_sha256": record["prediction_sha256"],
        "condition_b_metrics_blinded": True,
    }
    print("TURKISH_AI_DETECTION_SAFE_RESULT\t" + json.dumps(safe, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
