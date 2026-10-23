# -*- coding: utf-8 -*-
"""Protocol-faithful CE/SupCon rerun with terminal checkpoint persistence.

Scientific execution is prohibited while the V2 contract remains
DRAFT_FOR_REVIEW; two development-only pilot cells are explicitly allowed.
Scientific training math is delegated unchanged to the bound primary runner.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".gitignore").is_file():
            return candidate
    raise RuntimeError("project root not found")


ROOT = _root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from turkish_ai_detection.data.study import sha256_file, write_json  # noqa: E402
from turkish_ai_detection.evaluation.metrics import condition_text  # noqa: E402
from turkish_ai_detection.experiments.external_cross_corpus.audit_semantic_equivalence import (  # noqa: E402
    assert_semantic_equivalence,
)
from turkish_ai_detection.experiments.primary import run_fit as primary  # noqa: E402
from turkish_ai_detection.experiments.external_cross_corpus.freeze_v2 import verify_freeze  # noqa: E402

CONTRACT = ROOT / "configs" / "external_cross_corpus.json"
COMMON_SEEDS = (42, 314, 2026)
OUTER_FOLDS = (0, 1, 2, 3, 4)
DEVELOPMENT_CELLS = {("ce", 42, 0), ("supcon", 42, 0)}
FREEZE_PATH = ROOT / "provenance" / "experiments" / "external_cross_corpus_v2" / "PREEXECUTION_FREEZE.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_v2_contract(development_only: bool) -> tuple[dict, dict]:
    cfg = json.loads(CONTRACT.read_text(encoding="utf-8"))
    audit = assert_semantic_equivalence(CONTRACT)
    bindings = cfg["internal_training_bindings"]
    for name in ("dataset", "folds", "primary_historical_config", "primary_freeze"):
        record = bindings[name]
        if sha256_file(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"V2 identity mismatch: {name}")
    if development_only:
        if cfg.get("status") != "DRAFT_FOR_REVIEW" or cfg.get("execution_permitted") is not False:
            raise RuntimeError("development pilot requires the reviewed draft state")
    else:
        if cfg.get("status") != "FROZEN_READY_TO_RUN" or cfg.get("execution_permitted") is not True:
            raise RuntimeError("scientific rerun requires a separate frozen V2 contract")
        freeze_path = FREEZE_PATH
        if not freeze_path.is_file():
            raise RuntimeError("scientific rerun blocked: PREEXECUTION_FREEZE missing")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        verify_freeze(CONTRACT, freeze_path, required_code_path=Path(__file__).resolve())

        def bound(node: object, path: str, digest: str) -> bool:
            if isinstance(node, dict):
                if node.get("path") == path and node.get("sha256") == digest:
                    return True
                return any(bound(value, path, digest) for value in node.values())
            if isinstance(node, list):
                return any(bound(value, path, digest) for value in node)
            return False

        required_files = []
        for required_file in required_files:
            relative = required_file.resolve().relative_to(ROOT.resolve()).as_posix()
            if not bound(freeze, relative, sha256_file(required_file)):
                raise RuntimeError(f"PREEXECUTION_FREEZE missing binding: {relative}")
    return cfg, audit


def verify_structural_identity(objective: str, seed: int, outer: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    observed = primary.verify_inputs()
    if observed["dataset"] != primary.DATASET_SHA256 or observed["folds"] != primary.FOLD_SHA256:
        raise RuntimeError("primary input identity mismatch")
    data = pd.read_csv(primary.DATA_PATH).copy()
    folds = pd.read_csv(primary.FOLD_PATH)
    if len(data) != 3000 or data["text_id"].nunique() != 3000:
        raise RuntimeError("study row identity changed")
    data["model_text"] = condition_text(data, primary.CONDITION)
    test_pairs = set(folds.loc[folds.outer_fold.eq(outer), "pair_id"])
    train_pairs = set(folds.loc[folds.outer_fold.ne(outer), "pair_id"])
    if train_pairs & test_pairs:
        raise RuntimeError("outer train/test pair leakage")
    train = data[data.pair_id.isin(train_pairs)].copy()
    test = data[data.pair_id.isin(test_pairs)].copy()
    if len(train) != 2400 or len(test) != 600:
        raise RuntimeError("unexpected outer split row count")

    reference_path = ROOT / "results" / "primary" / objective / f"seed_{seed}" / f"outer_{outer}_run.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected = {
        "objective": objective,
        "seed": seed,
        "outer_fold": outer,
        "condition": primary.CONDITION,
        "dataset_sha256": primary.DATASET_SHA256,
        "fold_sha256": primary.FOLD_SHA256,
        "train_pair_count": 1200,
        "test_pair_count": 300,
        "train_pair_sha256": primary.pair_hash(train_pairs),
        "test_pair_sha256": primary.pair_hash(test_pairs),
        "train_test_pair_overlap": 0,
        "physical_batch_size": primary.PHYSICAL_BATCH_SIZE,
        "effective_batch_size": primary.EFFECTIVE_BATCH_SIZE,
        "gradient_accumulation_steps": primary.GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": 2e-5,
        "epochs": 3,
        "weight_decay": 0.01,
        "warmup_steps": 0.1,
        "lambda_supcon": 0.0 if objective == "ce" else primary.LAMBDA_SUPCON,
        "temperature": None if objective == "ce" else primary.TEMPERATURE,
    }
    mismatches = {key: (reference.get(key), value) for key, value in expected.items() if reference.get(key) != value}
    if mismatches:
        raise RuntimeError(f"structural identity mismatch: {mismatches}")
    reference_prediction = ROOT / "results" / "primary" / objective / f"seed_{seed}" / f"outer_{outer}_predictions.csv"
    if sha256_file(reference_prediction) != reference["prediction_sha256"]:
        raise RuntimeError("retained reference prediction hash mismatch")
    return train, test, reference


def save_terminal_payload(model, payload_dir: Path, metadata: dict) -> dict:
    from safetensors.torch import save_file

    payload_dir.mkdir(parents=True, exist_ok=False)
    model_path = payload_dir / "model.safetensors"
    temporary = payload_dir / "model.safetensors.tmp"
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(state, str(temporary))
    os.replace(temporary, model_path)
    config_path = payload_dir / "config.json"
    write_json(config_path, metadata)
    return {
        "model": model_path.relative_to(ROOT).as_posix(),
        "model_sha256": sha256_file(model_path),
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
    }


def reload_and_verify_payload(payload: dict, frame: pd.DataFrame, encodings: list[dict], collator, expected: np.ndarray) -> dict:
    import torch
    from safetensors.torch import load_file

    reloaded = primary.build_model()
    state = load_file(str(ROOT / payload["model"]), device="cpu")
    reloaded.load_state_dict(state, strict=True)
    reloaded.to("cuda")
    observed = primary.score(reloaded, frame, encodings, collator)
    np.testing.assert_allclose(observed, expected, rtol=1e-7, atol=1e-7)
    if not np.array_equal(observed > 0.0, expected > 0.0):
        raise RuntimeError("checkpoint roundtrip changed score-sign decisions")
    result = {
        "status": "PASS_CHECKPOINT_ROUNDTRIP",
        "rtol": 1e-7,
        "atol": 1e-7,
        "max_abs_difference": float(np.max(np.abs(observed - expected))),
        "argmax_sign_match": bool(np.array_equal(observed > 0.0, expected > 0.0)),
    }
    del reloaded
    torch.cuda.empty_cache()
    return result


def numerical_qa(prediction: pd.DataFrame, reference_path: Path) -> dict:
    reference = pd.read_csv(reference_path)
    identity = ["text_id", "pair_id", "label", "domain", "generator"]
    if not prediction[identity].reset_index(drop=True).equals(reference[identity].reset_index(drop=True)):
        raise RuntimeError("QA reference row identity mismatch")
    new = prediction["raw_decision_score"].to_numpy(float)
    old = reference["raw_decision_score"].to_numpy(float)
    old_metrics = primary.metric_record(reference)
    new_metrics = primary.metric_record(prediction)
    return {
        "status": "QA_DIAGNOSTIC_NOT_ACCEPTANCE_GATE",
        "prespecified_rtol": 1e-4,
        "prespecified_atol": 1e-5,
        "allclose_at_prespecified_values": bool(np.allclose(new, old, rtol=1e-4, atol=1e-5)),
        "max_abs_difference": float(np.max(np.abs(new - old))),
        "mean_abs_difference": float(np.mean(np.abs(new - old))),
        "rmse": float(np.sqrt(np.mean((new - old) ** 2))),
        "sign_agreement_fraction": float(np.mean((new > 0.0) == (old > 0.0))),
        "metric_differences": {key: float(new_metrics[key] - old_metrics[key]) for key in ("roc_auc", "average_precision", "std_pauc_003")},
        "interpretation": "Review diagnostic only; it cannot override structural identity or semantic-equivalence gates and is not an exact-reproduction acceptance rule.",
    }


def run_one(objective: str, seed: int, outer: int, development_only: bool) -> dict:
    import torch
    import transformers
    from transformers import AutoTokenizer, set_seed

    if objective not in {"ce", "supcon"} or seed not in COMMON_SEEDS or outer not in OUTER_FOLDS:
        raise ValueError("cell is outside the fixed V2 grid")
    if development_only and (objective, seed, outer) not in DEVELOPMENT_CELLS:
        raise RuntimeError("development pilots are limited to CE42/fold0 and SupCon42/fold0")
    if not torch.cuda.is_available():
        raise RuntimeError("protocol-faithful rerun requires CUDA")

    cfg, audit = verify_v2_contract(development_only)
    train, test, reference = verify_structural_identity(objective, seed, outer)
    mode = "DEVELOPMENT_ONLY_NOT_SCIENTIFIC" if development_only else "SCIENTIFIC_PROTOCOL_FAITHFUL_RERUN"
    root = (ROOT / "artifacts" / "development" if development_only else ROOT / "artifacts" / "checkpoints")
    cell_dir = root / "external_cross_corpus" / "protocol_faithful_rerun" / f"{objective}_seed_{seed}_outer_{outer}"
    if cell_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing payload: {cell_dir}")

    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(primary.MODEL_NAME, revision=primary.REVISION)
    data = pd.read_csv(primary.DATA_PATH).copy()
    data["model_text"] = condition_text(data, primary.CONDITION)
    encodings = [primary.encode_head_tail(tokenizer, text) for text in data.model_text.astype(str)]
    collator = primary.BoundedCollator(tokenizer)
    model = primary.build_model()
    TrainerClass = primary.make_seeded_trainer_class(objective, seed)
    trainer = TrainerClass(
        model=model,
        args=primary.training_arguments(cell_dir / "trainer_state", seed),
        train_dataset=primary.make_dataset(train, encodings),
        data_collator=collator,
    )
    trainer.train()
    if trainer.last_group_audit is None:
        raise RuntimeError("no completed true-32 objective group observed")

    raw = primary.score(model, test, encodings, collator)
    prediction = test[["text_id", "pair_id", "label", "domain", "generator"]].copy()
    prediction["outer_fold"] = outer
    prediction["seed"] = seed
    prediction["objective"] = objective
    prediction["condition"] = primary.CONDITION
    prediction["raw_decision_score"] = raw
    if len(prediction) != 600 or not prediction.text_id.is_unique or not np.isfinite(raw).all():
        raise RuntimeError("invalid rerun internal predictions")

    metadata = {
        "schema_version": "external-cross-corpus-v2-protocol-faithful-rerun-payload-v1",
        "status": mode,
        "objective": objective,
        "seed": seed,
        "outer_fold": outer,
        "model": primary.MODEL_NAME,
        "revision": primary.REVISION,
        "architecture": "Delegated to bound primary.build_model",
        "tokenization": "Delegated to bound primary.encode_head_tail",
    }
    payload = save_terminal_payload(model, cell_dir / "checkpoint", metadata)
    roundtrip = reload_and_verify_payload(payload, test, encodings, collator, raw)

    prediction_path = cell_dir / "internal_predictions.csv"
    prediction.to_csv(prediction_path, index=False)
    qa = numerical_qa(
        prediction,
        ROOT / "results" / "primary" / objective / f"seed_{seed}" / f"outer_{outer}_predictions.csv",
    )
    record = {
        **metadata,
        "created_utc": utc_now(),
        "contract_sha256": sha256_file(CONTRACT),
        "runner_sha256": sha256_file(Path(__file__)),
        "semantic_audit": audit,
        "structural_reference_sidecar_sha256": sha256_file(
            ROOT / "results" / "primary" / objective / f"seed_{seed}" / f"outer_{outer}_run.json"
        ),
        "structural_reference_prediction_sha256": reference["prediction_sha256"],
        "train_pair_sha256": reference["train_pair_sha256"],
        "test_pair_sha256": reference["test_pair_sha256"],
        "payload": payload,
        "checkpoint_roundtrip": roundtrip,
        "internal_prediction_path": prediction_path.relative_to(ROOT).as_posix(),
        "internal_prediction_sha256": sha256_file(prediction_path),
        "numerical_comparison": qa,
        "training_runtime_provenance": {
            "lineage": "RERUN_CE_SUPCON",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    write_json(cell_dir / "run.json", record)
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", required=True, choices=["ce", "supcon"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--outer-fold", required=True, type=int)
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    record = run_one(args.objective, args.seed, args.outer_fold, args.development_only)
    print(json.dumps({
        "status": record["status"],
        "objective": record["objective"],
        "seed": record["seed"],
        "outer_fold": record["outer_fold"],
        "metrics_not_printed": True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
