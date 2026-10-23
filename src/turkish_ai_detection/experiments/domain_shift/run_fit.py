# -*- coding: utf-8 -*-
"""Single-fit runner for DOMAIN_SHIFT_V1.

One invocation trains one objective/seed/direction using only the source domain
and evaluates only on the held-out target domain. Scientific target-domain
metrics are saved but never printed during the blinded run.
"""
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
from turkish_ai_detection.evaluation.metrics import condition_text, labels  # noqa: E402
from turkish_ai_detection.experiments.baseline.berturk import MODEL_NAME, REVISION, encode_head_tail  # noqa: E402
from turkish_ai_detection.experiments.model_development.supcon.train import (  # noqa: E402
    BalancedClassSampler, BoundedCollator, DATASET_SHA256, DATA_PATH, build_model,
)
from turkish_ai_detection.experiments.primary.run_fit import (  # noqa: E402
    atomic_csv, make_dataset, make_seeded_trainer_class, metric_record,
    pair_hash, score, training_arguments, utc_now,
)

EXPERIMENT = "DOMAIN_SHIFT_V1"
CONDITION = "A"
CONFIG_PATH = ROOT / "configs" / "domain_shift.json"
OUT = ROOT / "results" / "domain_shift"
PROV = ROOT / "provenance" / "experiments" / "domain_shift"
STATE = ROOT / "artifacts" / "checkpoints" / "domain_shift"
FREEZE_PATH = PROV / "PREEXECUTION_FREEZE.json"

PRIMARY_RUNNER = ROOT / "src" / "turkish_ai_detection/experiments/primary/run_fit.py"
PRIMARY_CONFIG = ROOT / "configs" / "primary_evaluation.json"
PRIMARY_FREEZE = ROOT / "provenance" / "experiments" / "primary" / "PREEXECUTION_FREEZE.json"
PRIMARY_MANIFEST = (
    ROOT / "results" / "primary"
    / "final_statistics" / "final_statistics_manifest.json"
)

ALLOWED_SEEDS = (42, 314, 2026)
DIRECTIONS = {
    "abstract_to_newspaper": ("ABSTRACT", "NEWSPAPER"),
    "newspaper_to_abstract": ("NEWSPAPER", "ABSTRACT"),
}
PRIMARY_RUNNER_SHA256 = "eafe1e66c5d8324cf547571144a1a2f70f9cf1b4406c22f9eabd4e0b8587b92e"
PRIMARY_CONFIG_SHA256 = "7e825c0f97f1763c6b4db69164feda04743f9fa30f27e254e1c52f953dd07f13"
PRIMARY_FREEZE_SHA256 = "162a120d624c830e3b0f93a07504a30be58ac602f4089aa2349527d52882548e"
PRIMARY_MANIFEST_SHA256 = "ed30975ecb16f49715352f6b11ef1c29aa6e4df2e0d2a0bf55f93a69fcc2187a"


def verify_primary_lineage() -> None:
    required = (
        (PRIMARY_RUNNER, PRIMARY_RUNNER_SHA256, "primary runner"),
        (PRIMARY_CONFIG, PRIMARY_CONFIG_SHA256, "primary config"),
        (PRIMARY_FREEZE, PRIMARY_FREEZE_SHA256, "primary freeze"),
        (PRIMARY_MANIFEST, PRIMARY_MANIFEST_SHA256, "primary final-statistics manifest"),
    )
    for path, expected, label in required:
        if not path.is_file():
            raise RuntimeError(f"missing {label}: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"{label} hash mismatch: {observed}")


def verify_preexecution_freeze() -> dict:
    if not FREEZE_PATH.is_file():
        raise RuntimeError("domain-shift training is locked until PREEXECUTION_FREEZE.json exists")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_READY_TO_RUN":
        raise RuntimeError("invalid domain-shift freeze status")
    freeze["_current_reproduction"] = validate_current_migrated_reproduction(
        ROOT, "domain_shift", Path(__file__)
    )
    if freeze.get("domain_shift_results_unblinded_before_freeze") is not False:
        raise RuntimeError("freeze does not certify blinded domain-shift state")
    return freeze


def split_for_direction(data: pd.DataFrame, direction: str):
    if direction not in DIRECTIONS:
        raise ValueError(direction)
    train_domain, test_domain = DIRECTIONS[direction]

    if not (data.groupby("pair_id")["domain"].nunique() == 1).all():
        raise RuntimeError("domain metadata is not pair-consistent")

    train = data[data["domain"].astype(str).eq(train_domain)].copy()
    test = data[data["domain"].astype(str).eq(test_domain)].copy()

    train_pairs = set(train["pair_id"].astype(str))
    test_pairs = set(test["pair_id"].astype(str))
    if train_pairs & test_pairs:
        raise RuntimeError("cross-domain train/test pair leakage")

    expected = {
        "abstract_to_newspaper": (2000, 1000, 1000, 500, 1984),
        "newspaper_to_abstract": (1000, 500, 2000, 1000, 992),
    }[direction]
    train_rows, train_pair_n, test_rows, test_pair_n, usable_rows = expected

    if len(train) != train_rows or train["pair_id"].nunique() != train_pair_n:
        raise RuntimeError(f"unexpected source-domain size for {direction}")
    if len(test) != test_rows or test["pair_id"].nunique() != test_pair_n:
        raise RuntimeError(f"unexpected target-domain size for {direction}")

    for frame, n_per_class, label_name in (
        (train, train_rows // 2, "train"),
        (test, test_rows // 2, "test"),
    ):
        counts = frame["label"].astype(str).value_counts().to_dict()
        if counts.get("HUMAN", 0) != n_per_class or counts.get("AI", 0) != n_per_class:
            raise RuntimeError(f"class balance changed in {label_name}: {direction}")

    sampler = BalancedClassSampler(labels(train), batch_size=16, seed=42)
    if len(sampler) != usable_rows:
        raise RuntimeError(f"unexpected true-32 usable rows for {direction}: {len(sampler)}")

    return train, test, train_pairs, test_pairs, usable_rows


def run_one(objective: str, seed: int, direction: str) -> dict:
    import torch
    import transformers
    from transformers import AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("domain-shift run requires CUDA")
    if objective not in {"ce", "supcon"}:
        raise ValueError(objective)
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed not prespecified: {seed}")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction not prespecified: {direction}")

    verify_primary_lineage()
    freeze = verify_preexecution_freeze()
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    if not DATA_PATH.is_file() or sha256_file(DATA_PATH) != DATASET_SHA256:
        raise RuntimeError("frozen Combined-v2 dataset identity changed")
    if cfg["dataset_sha256"] != DATASET_SHA256:
        raise RuntimeError("domain-shift config dataset hash mismatch")

    data = pd.read_csv(DATA_PATH).copy()
    if len(data) != 3000 or data["text_id"].nunique() != 3000 or data["pair_id"].nunique() != 1500:
        raise RuntimeError("Combined-v2 identity/count changed")
    data["model_text"] = condition_text(data, CONDITION)

    train, test, train_pairs, test_pairs, usable_rows = split_for_direction(data, direction)
    train_domain, test_domain = DIRECTIONS[direction]

    tag = f"{objective}_seed_{seed}_{direction}"
    out_dir = OUT / objective / f"seed_{seed}"
    pred_path = out_dir / f"{direction}_predictions.csv"
    sidecar_path = out_dir / f"{direction}_run.json"

    binding = {
        "experiment": EXPERIMENT,
        "objective": objective,
        "seed": int(seed),
        "direction": direction,
        "train_domain": train_domain,
        "test_domain": test_domain,
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
        "primary_final_manifest_sha256": sha256_file(PRIMARY_MANIFEST),
        "train_pair_count": len(train_pairs),
        "test_pair_count": len(test_pairs),
        "train_pair_sha256": pair_hash(train_pairs),
        "test_pair_sha256": pair_hash(test_pairs),
        "train_test_pair_overlap": 0,
        "sampler_usable_train_rows_per_epoch": int(usable_rows),
        "sampler_dropped_train_rows_per_epoch": int(len(train) - usable_rows),
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
    prediction["direction"] = direction
    prediction["train_domain"] = train_domain
    prediction["test_domain"] = test_domain
    prediction["seed"] = int(seed)
    prediction["objective"] = objective
    prediction["condition"] = CONDITION
    prediction["raw_decision_score"] = raw

    if len(prediction) != len(test) or prediction["text_id"].nunique() != len(test):
        raise RuntimeError("invalid domain-shift prediction shape")
    if not np.isfinite(prediction["raw_decision_score"]).all():
        raise RuntimeError("non-finite domain-shift score")

    atomic_csv(prediction, pred_path)
    metrics = metric_record(prediction)

    record = {
        **binding,
        "schema_version": "combined-v2-domain-shift-single-fit-v1",
        "created_utc": utc_now(),
        "status": "COMPLETE",
        "model": MODEL_NAME,
        "revision": REVISION,
        "train_rows": len(train),
        "test_rows": len(test),
        "heldout_domain_training_rows": 0,
        "heldout_domain_used_for_model_selection": False,
        "test_rows_used_for_training": 0,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", required=True, choices=["ce", "supcon"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--direction", required=True, choices=list(DIRECTIONS))
    args = parser.parse_args()

    record = run_one(args.objective, args.seed, args.direction)
    safe = {
        "status": record["status"],
        "experiment": record["experiment"],
        "objective": record["objective"],
        "seed": record["seed"],
        "direction": record["direction"],
        "test_domain": record["test_domain"],
        "prediction_sha256": record["prediction_sha256"],
        "domain_shift_metrics_blinded": True,
    }
    print("TURKISH_AI_DETECTION_SAFE_RESULT\t" + json.dumps(safe, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
