# -*- coding: utf-8 -*-
"""One frozen NEWSPAPER author-disjoint CE or SupCon fit.

This wrapper reuses the authoritative primary model, true-32 trainer,
tokenization, optimizer, and scoring implementation. It changes only the
train/test population. It does not calculate or print scientific metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import (
    CONFIG, DATA, FOLDS, FREEZE, OBJECTIVES, OUT, OUTERS, ROOT, SEEDS,
    load_config_and_freeze, load_data_and_folds, pair_hash, require,
    sha256_file, split_pair_ids,
)

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from turkish_ai_detection.data.study import write_json  # noqa: E402
from turkish_ai_detection.evaluation.metrics import condition_text  # noqa: E402
from turkish_ai_detection.experiments.primary import run_fit as primary  # noqa: E402


EXPERIMENT = "NEWSPAPER_AUTHOR_DISJOINT_SENSITIVITY_V1"
STATE = ROOT / "artifacts" / "checkpoints" / "author_disjoint_newspaper"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_csv(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def run_one(objective: str, seed: int, outer: int) -> dict:
    import torch
    import transformers
    from transformers import AutoTokenizer, set_seed

    require(torch.cuda.is_available(), "author-disjoint CE/SupCon execution requires CUDA")
    require(objective in OBJECTIVES, f"objective not frozen: {objective}")
    require(seed in SEEDS, f"seed not frozen: {seed}")
    require(outer in OUTERS, f"fold not frozen: {outer}")
    config, _freeze = load_config_and_freeze()
    data, folds = load_data_and_folds()
    require(len(data) == 3000 and data.text_id.nunique() == 3000, "dataset identity/count changed")

    train_pairs, test_pairs = split_pair_ids(data, folds, outer)
    data["model_text"] = condition_text(data, "A")
    train = data[data.pair_id.astype(str).isin(train_pairs)].copy()
    test = data[data.pair_id.astype(str).isin(test_pairs)].copy()
    require(len(train_pairs) == 1400 and len(train) == 2800, "unexpected training population")
    require(len(test_pairs) == 100 and len(test) == 200, "unexpected test population")
    require(set(test.domain.astype(str)) == {"NEWSPAPER"}, "test contains non-NEWSPAPER rows")
    require(set(train.pair_id.astype(str)).isdisjoint(set(test.pair_id.astype(str))), "pair leakage")
    test_authors = set(test.author.astype(str))
    train_news_authors = set(train.loc[train.domain.eq("NEWSPAPER"), "author"].astype(str))
    require(test_authors.isdisjoint(train_news_authors), "author leakage")

    tag = f"{objective}_seed_{seed}_fold_{outer}"
    out_dir = OUT / objective / f"seed_{seed}"
    pred_path = out_dir / f"fold_{outer}_predictions.csv"
    run_path = out_dir / f"fold_{outer}_run.json"
    state_dir = STATE / tag
    wrapper_sha = sha256_file(Path(__file__))
    binding = {
        "experiment": EXPERIMENT,
        "objective": objective,
        "seed": int(seed),
        "author_fold": int(outer),
        "condition": "A",
        "dataset_sha256": sha256_file(DATA),
        "fold_sha256": sha256_file(FOLDS),
        "config_sha256": sha256_file(CONFIG),
        "wrapper_sha256": wrapper_sha,
        "authoritative_primary_runner_sha256": sha256_file(Path(primary.__file__)),
        "freeze_sha256": sha256_file(FREEZE),
        "train_pair_count": 1400,
        "test_pair_count": 100,
        "train_pair_sha256": pair_hash(train_pairs),
        "test_pair_sha256": pair_hash(test_pairs),
        "train_test_pair_overlap": 0,
        "train_newspaper_author_count": 8,
        "test_newspaper_author_count": 2,
        "train_test_author_overlap": 0,
    }

    if pred_path.exists() or run_path.exists():
        require(pred_path.is_file() and run_path.is_file(), f"incomplete resume state: {tag}")
        saved = json.loads(run_path.read_text(encoding="utf-8"))
        expected = {**binding, "prediction_sha256": sha256_file(pred_path), "status": "COMPLETE"}
        require(all(saved.get(k) == v for k, v in expected.items()), f"resume binding mismatch: {tag}")
        return {"status": "SKIPPED_VERIFIED_COMPLETE", "tag": tag}
    if state_dir.exists() and any(state_dir.rglob("*")):
        raise RuntimeError(
            f"non-empty scratch state without complete prediction/sidecar: {tag}; "
            "manual inspection is required and silent overwrite is forbidden"
        )

    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(primary.MODEL_NAME, revision=primary.REVISION)
    encodings = [primary.encode_head_tail(tokenizer, text) for text in data.model_text.astype(str)]
    collator = primary.BoundedCollator(tokenizer)
    model = primary.build_model()
    trainer_class = primary.make_seeded_trainer_class(objective, seed)
    trainer = trainer_class(
        model=model,
        args=primary.training_arguments(state_dir, seed),
        train_dataset=primary.make_dataset(train, encodings),
        data_collator=collator,
    )
    trainer.train()
    require(trainer.last_group_audit is not None, "no true-32 objective group observed")
    raw_score = primary.score(model, test, encodings, collator)
    prediction = test[["text_id", "pair_id", "label", "domain", "generator", "author"]].copy()
    prediction["author_fold"] = int(outer)
    prediction["seed"] = int(seed)
    prediction["objective"] = objective
    prediction["condition"] = "A"
    prediction["raw_decision_score"] = raw_score
    require(len(prediction) == 200 and prediction.text_id.nunique() == 200, "invalid prediction shape")
    require(np.isfinite(prediction.raw_decision_score.to_numpy(float)).all(), "non-finite score")
    atomic_csv(prediction, pred_path)

    record = {
        **binding,
        "schema_version": "author-disjoint-newspaper-single-fit-v1",
        "created_utc": utc_now(),
        "status": "COMPLETE",
        "model": primary.MODEL_NAME,
        "revision": primary.REVISION,
        "physical_batch_size": primary.PHYSICAL_BATCH_SIZE,
        "effective_batch_size": primary.EFFECTIVE_BATCH_SIZE,
        "gradient_accumulation_steps": primary.GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": config["training"]["learning_rate"],
        "epochs": config["training"]["epochs"],
        "weight_decay": config["training"]["weight_decay"],
        "warmup_steps": config["training"]["warmup_steps"],
        "lambda_supcon": 0.0 if objective == "ce" else primary.LAMBDA_SUPCON,
        "temperature": None if objective == "ce" else primary.TEMPERATURE,
        "calibration_used": False,
        "threshold_used": False,
        "outer_test_rows_used_for_training": 0,
        "persistent_checkpoint_saved": False,
        "last_group_audit": trainer.last_group_audit,
        "prediction_sha256": sha256_file(pred_path),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_model": torch.cuda.get_device_name(0),
    }
    write_json(run_path, record)
    return {"status": "COMPLETE", "tag": tag, "prediction": str(pred_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=OBJECTIVES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--outer-fold", type=int, choices=OUTERS, required=True)
    args = parser.parse_args()
    print(json.dumps(run_one(args.objective, args.seed, args.outer_fold), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

