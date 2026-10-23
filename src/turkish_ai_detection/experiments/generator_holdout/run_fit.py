# -*- coding: utf-8 -*-
"""Single-fit runner for GENERATOR_HOLDOUT_V1.

Uses the already-validated shared CE/SupCon training implementation from
turkish_ai_detection/experiments/primary/run_fit.py. One invocation trains one objective/seed/held-out
generator combination. OOD metrics are written to a sealed sidecar and are not
printed during confirmatory execution.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
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
    BoundedCollator, DATASET_SHA256, DATA_PATH, build_model, verify_inputs,
)
from turkish_ai_detection.experiments.primary.run_fit import (  # noqa: E402
    atomic_csv, make_dataset, make_seeded_trainer_class, metric_record,
    pair_hash, score, training_arguments, utc_now,
)

EXPERIMENT = "GENERATOR_HOLDOUT_V1"
CONDITION = "A"
CONFIG_PATH = ROOT / "configs" / "generator_holdout.json"
OUT = ROOT / "results" / "generator_holdout"
PROV = ROOT / "provenance" / "experiments" / "generator_holdout"
STATE = ROOT / "artifacts" / "checkpoints" / "generator_holdout"
FREEZE_PATH = PROV / "PREEXECUTION_FREEZE.json"
PRIMARY_RUNNER = ROOT / "src" / "turkish_ai_detection/experiments/primary/run_fit.py"
PRIMARY_CONFIG = ROOT / "configs" / "primary_evaluation.json"
PRIMARY_FREEZE = ROOT / "provenance" / "experiments" / "primary" / "PREEXECUTION_FREEZE.json"
PRIMARY_FINAL_MANIFEST = (
    ROOT / "results" / "primary"
    / "final_statistics" / "final_statistics_manifest.json"
)

ALLOWED_GENERATORS = ("Claude", "GPT", "Gemini", "Kimi", "Qwen")
ALLOWED_SEEDS = (42, 314, 2026)
PRIMARY_RUNNER_SHA256 = "eafe1e66c5d8324cf547571144a1a2f70f9cf1b4406c22f9eabd4e0b8587b92e"
PRIMARY_CONFIG_SHA256 = "7e825c0f97f1763c6b4db69164feda04743f9fa30f27e254e1c52f953dd07f13"
PRIMARY_FREEZE_SHA256 = "162a120d624c830e3b0f93a07504a30be58ac602f4089aa2349527d52882548e"
PRIMARY_FINAL_MANIFEST_SHA256 = "ed30975ecb16f49715352f6b11ef1c29aa6e4df2e0d2a0bf55f93a69fcc2187a"


def slug(value: str) -> str:
    return value.lower().replace(" ", "_")


def verify_lineage() -> None:
    required = (
        (PRIMARY_RUNNER, PRIMARY_RUNNER_SHA256, "primary runner"),
        (PRIMARY_CONFIG, PRIMARY_CONFIG_SHA256, "primary config"),
        (PRIMARY_FREEZE, PRIMARY_FREEZE_SHA256, "primary freeze"),
        (PRIMARY_FINAL_MANIFEST, PRIMARY_FINAL_MANIFEST_SHA256, "primary final-statistics manifest"),
    )
    for path, expected, label in required:
        if not path.is_file():
            raise RuntimeError(f"missing {label}: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"{label} hash mismatch: {observed}")


def verify_preexecution_freeze() -> dict:
    if not FREEZE_PATH.is_file():
        raise RuntimeError("generator-holdout training is locked until PREEXECUTION_FREEZE.json exists")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_READY_TO_RUN":
        raise RuntimeError("invalid generator-holdout pre-execution freeze status")
    freeze["_current_reproduction"] = validate_current_migrated_reproduction(
        ROOT, "generator_holdout", Path(__file__)
    )
    if freeze.get("generator_ood_results_unblinded_before_freeze") is not False:
        raise RuntimeError("freeze does not certify blinded OOD state")
    return freeze


def split_for_generator(data: pd.DataFrame, heldout: str):
    pair_generator_counts = data.groupby("pair_id")["generator"].nunique()
    if not (pair_generator_counts == 1).all():
        raise RuntimeError("generator metadata is not pair-consistent")

    test_pairs = set(data.loc[data["generator"].eq(heldout), "pair_id"].astype(str))
    train_pairs = set(data.loc[~data["generator"].eq(heldout), "pair_id"].astype(str))
    if train_pairs & test_pairs:
        raise RuntimeError("generator-holdout train/test pair leakage")

    pair_as_str = data["pair_id"].astype(str)
    train = data[pair_as_str.isin(train_pairs)].copy()
    test = data[pair_as_str.isin(test_pairs)].copy()

    if len(train) != 2400 or len(test) != 600:
        raise RuntimeError(f"unexpected generator split row counts: train={len(train)}, test={len(test)}")
    if train["pair_id"].nunique() != 1200 or test["pair_id"].nunique() != 300:
        raise RuntimeError("unexpected generator split pair counts")
    if heldout in set(train["generator"].astype(str)):
        raise RuntimeError("held-out generator leaked into training")
    if set(test["generator"].astype(str)) != {heldout}:
        raise RuntimeError("test partition is not generator-pure")

    train_counts = train["label"].astype(str).value_counts().to_dict()
    test_counts = test["label"].astype(str).value_counts().to_dict()
    if train_counts.get("HUMAN", 0) != 1200 or train_counts.get("AI", 0) != 1200:
        raise RuntimeError("training class balance changed")
    if test_counts.get("HUMAN", 0) != 300 or test_counts.get("AI", 0) != 300:
        raise RuntimeError("test class balance changed")

    test_domain_pairs = test.groupby("domain")["pair_id"].nunique().to_dict()
    if test_domain_pairs.get("ABSTRACT", 0) != 200 or test_domain_pairs.get("NEWSPAPER", 0) != 100:
        raise RuntimeError(f"held-out generator domain composition changed: {test_domain_pairs}")

    return train, test, train_pairs, test_pairs


def run_one(objective: str, seed: int, heldout: str) -> dict:
    import torch
    import transformers
    from transformers import AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("generator holdout requires CUDA")
    if objective not in {"ce", "supcon"}:
        raise ValueError(objective)
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed not prespecified: {seed}")
    if heldout not in ALLOWED_GENERATORS:
        raise ValueError(f"generator not prespecified: {heldout}")

    verify_lineage()
    freeze = verify_preexecution_freeze()
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    observed = verify_inputs()
    if observed["dataset"] != DATASET_SHA256 or cfg["dataset_sha256"] != DATASET_SHA256:
        raise RuntimeError("frozen Combined-v2 dataset identity changed")

    data = pd.read_csv(DATA_PATH).copy()
    if len(data) != 3000 or data["text_id"].nunique() != 3000 or data["pair_id"].nunique() != 1500:
        raise RuntimeError("Combined-v2 count/identity changed")
    data["model_text"] = condition_text(data, CONDITION)

    train, test, train_pairs, test_pairs = split_for_generator(data, heldout)

    tag = f"{objective}_seed_{seed}_holdout_{slug(heldout)}"
    out_dir = OUT / objective / f"seed_{seed}"
    pred_path = out_dir / f"holdout_{slug(heldout)}_predictions.csv"
    sidecar_path = out_dir / f"holdout_{slug(heldout)}_run.json"

    binding = {
        "experiment": EXPERIMENT,
        "objective": objective,
        "seed": int(seed),
        "heldout_generator": heldout,
        "condition": CONDITION,
        "dataset_sha256": DATASET_SHA256,
        "config_sha256": sha256_file(CONFIG_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "preexecution_freeze_sha256": sha256_file(FREEZE_PATH),
        "reproduction_mode": freeze["_current_reproduction"]["reproduction_mode"],
        "historical_exact_reproduction": freeze["_current_reproduction"]["historical_exact_reproduction"],
        "primary_runner_sha256": sha256_file(PRIMARY_RUNNER),
        "primary_config_sha256": sha256_file(PRIMARY_CONFIG),
        "primary_freeze_sha256": sha256_file(PRIMARY_FREEZE),
        "primary_final_manifest_sha256": sha256_file(PRIMARY_FINAL_MANIFEST),
        "train_pair_count": len(train_pairs),
        "test_pair_count": len(test_pairs),
        "train_pair_sha256": pair_hash(train_pairs),
        "test_pair_sha256": pair_hash(test_pairs),
        "train_test_pair_overlap": 0,
        "train_generators": sorted(set(train["generator"].astype(str))),
        "test_generators": [heldout],
    }

    if pred_path.exists() or sidecar_path.exists():
        if not pred_path.exists() or not sidecar_path.exists():
            raise RuntimeError(f"incomplete resume state: {tag}")
        saved = json.loads(sidecar_path.read_text(encoding="utf-8"))
        expected = {**binding, "prediction_sha256": sha256_file(pred_path)}
        if {key: saved.get(key) for key in expected} != expected:
            raise RuntimeError(f"resume binding mismatch: {tag}")
        return saved

    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)
    encodings = [encode_head_tail(tokenizer, text) for text in data["model_text"].astype(str)]
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
    prediction["heldout_generator"] = heldout
    prediction["seed"] = int(seed)
    prediction["objective"] = objective
    prediction["condition"] = CONDITION
    prediction["raw_decision_score"] = raw

    if len(prediction) != 600 or prediction["text_id"].nunique() != 600:
        raise RuntimeError("invalid generator-holdout prediction shape")
    if not np.isfinite(prediction["raw_decision_score"]).all():
        raise RuntimeError("non-finite generator-holdout score")

    atomic_csv(prediction, pred_path)
    metrics = metric_record(prediction)

    record = {
        **binding,
        "schema_version": "combined-v2-generator-holdout-single-fit-v1",
        "created_utc": utc_now(),
        "status": "COMPLETE",
        "model": MODEL_NAME,
        "revision": REVISION,
        "train_rows": len(train),
        "test_rows": len(test),
        "heldout_generator_training_rows": 0,
        "heldout_generator_used_for_model_selection": False,
        "calibration_used": False,
        "threshold_used": False,
        "test_rows_used_for_training": 0,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", required=True, choices=["ce", "supcon"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--heldout-generator", required=True, choices=list(ALLOWED_GENERATORS))
    args = parser.parse_args()

    record = run_one(args.objective, args.seed, args.heldout_generator)
    safe = {
        "status": record["status"],
        "experiment": record["experiment"],
        "objective": record["objective"],
        "seed": record["seed"],
        "heldout_generator": record["heldout_generator"],
        "prediction_sha256": record["prediction_sha256"],
        "ood_metrics_blinded": True,
    }
    print("TURKISH_AI_DETECTION_SAFE_RESULT\t" + json.dumps(safe, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
