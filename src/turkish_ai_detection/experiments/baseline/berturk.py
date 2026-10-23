# -*- coding: utf-8 -*-
"""Compute-heavy, fixed-definition BERTurk (M5) runner for Combined v2.

The explicit budget flag prevents accidental execution. Each outer fold uses
four inner models for training-only OOF calibration/thresholding and one model
fit on the complete outer-training partition. No outer-evaluation observation
is passed to Trainer during fitting.
"""
from __future__ import annotations

import argparse
import gc
import json
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

from turkish_ai_detection.data.study import sha256_file, write_json
from turkish_ai_detection.evaluation.metrics import (
    _fold_metric_row, _prediction_rows, calibrated_score, condition_text,
    fit_platt, labels, summarize, threshold_from_training_oof,
    update_baseline_table,
)

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
REVISION = "b6e1de16c983e0f2c70664591ea3f22810072608"
SEED = 42
MAX_MODEL_TOKENS = 512
HEAD_WORDPIECES = 128
TAIL_WORDPIECES = 382


def encode_head_tail(tokenizer, text: str) -> dict[str, list[int]]:
    """Apply the frozen 128+382 WordPiece policy before model-safe encoding."""
    # This first call is only a length/content inspection. Its untruncated IDs
    # are never passed to BERT.
    inspected_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    )["input_ids"]
    selected_ids = (
        inspected_ids
        if len(inspected_ids) <= MAX_MODEL_TOKENS - 2
        else inspected_ids[:HEAD_WORDPIECES] + inspected_ids[-TAIL_WORDPIECES:]
    )
    # Every tokenizer call capable of producing a model input is explicitly
    # bounded for Transformers v5. For long text, this naive prefix result is
    # deliberately not used; the frozen head/tail IDs are wrapped below.
    bounded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_MODEL_TOKENS,
        return_attention_mask=True,
    )
    if len(inspected_ids) <= MAX_MODEL_TOKENS - 2:
        encoded = bounded
    else:
        input_ids = [tokenizer.cls_token_id] + selected_ids + [tokenizer.sep_token_id]
        encoded = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }
    assert len(encoded["input_ids"]) <= MAX_MODEL_TOKENS
    assert len(encoded["attention_mask"]) == len(encoded["input_ids"])
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def install_model_length_assertion(model):
    """Assert at the BERT call boundary that no input tensor exceeds 512."""
    def assert_length(_module, positional, keyword):
        input_ids = keyword.get("input_ids")
        if input_ids is None and positional:
            input_ids = positional[0]
        if input_ids is not None:
            assert input_ids.shape[-1] <= MAX_MODEL_TOKENS, (
                f"refusing BERT input of {input_ids.shape[-1]} tokens; "
                f"maximum is {MAX_MODEL_TOKENS}"
            )

    return model.register_forward_pre_hook(assert_length, with_kwargs=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["A", "B"], default="A")
    parser.add_argument("--approve-compute", action="store_true", help="acknowledge 25 fixed BERTurk fine-tunes")
    parser.add_argument("--smoke-test", action="store_true", help="validate v5 API, CUDA, and input bounds without training")
    parser.add_argument("--data", default="data/study/paired_study.csv")
    parser.add_argument("--folds", default="data/study/nested_folds.csv")
    args = parser.parse_args()
    if not args.approve_compute and not args.smoke_test:
        raise RuntimeError("M5 budget guard: pass --approve-compute to run 25 fixed BERTurk fine-tunes")

    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("M5 Combined v2 requires CUDA; refusing an unintended multi-day CPU run")
    set_seed(SEED)
    data_path = ROOT / args.data; folds_path = ROOT / args.folds
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)

    def training_arguments(output_dir: Path):
        return TrainingArguments(
            output_dir=str(output_dir), learning_rate=2e-5,
            per_device_train_batch_size=16, per_device_eval_batch_size=32,
            num_train_epochs=3, weight_decay=0.01, warmup_steps=0.1,
            save_strategy="no", logging_strategy="epoch", report_to=[],
            seed=SEED, data_seed=SEED,
        )

    base_collator = DataCollatorWithPadding(tokenizer)

    def guarded_collator(features):
        batch = base_collator(features)
        assert batch["input_ids"].shape[-1] <= MAX_MODEL_TOKENS, (
            f"collator produced {batch['input_ids'].shape[-1]} tokens"
        )
        return batch

    if args.smoke_test:
        smoke_dir = ROOT / "artifacts" / "checkpoints" / "berturk" / "smoke_test"
        smoke_args = training_arguments(smoke_dir)
        samples = [
            encode_head_tail(tokenizer, "Bu kısa bir Türkçe duman testidir."),
            encode_head_tail(tokenizer, "uzunörnek " * 700),
        ]
        raw_long_length = len(tokenizer(
            "uzunörnek " * 700,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"])
        batch = guarded_collator(samples)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, revision=REVISION, num_labels=2
        ).to("cuda")
        hook = install_model_length_assertion(model)
        try:
            cuda_batch = {key: value.to("cuda") for key, value in batch.items()}
            with torch.no_grad():
                model(**cuda_batch)
        finally:
            hook.remove()
            del model
            gc.collect(); torch.cuda.empty_cache()
        print(json.dumps({
            "status": "PASS",
            "training_started": False,
            "transformers_version": __import__("transformers").__version__,
            "cuda": True,
            "raw_long_wordpieces": raw_long_length,
            "short_model_tokens": len(samples[0]["input_ids"]),
            "long_model_tokens": len(samples[1]["input_ids"]),
            "batch_tensor_tokens": int(batch["input_ids"].shape[-1]),
            "warmup_steps": smoke_args.warmup_steps,
            "truncation": "head128+tail382",
        }, ensure_ascii=False, indent=2))
        return 0

    data = pd.read_csv(data_path); folds = pd.read_csv(folds_path)
    data["model_text"] = condition_text(data, args.condition)

    encoded = [encode_head_tail(tokenizer, text) for text in data["model_text"].astype(str)]
    encoded_data = Dataset.from_dict({"labels": labels(data).tolist(), "input_ids": [x["input_ids"] for x in encoded], "attention_mask": [x["attention_mask"] for x in encoded]})

    run_tmp = ROOT / "artifacts" / "checkpoints" / "berturk" / f"condition_{args.condition}"
    run_tmp.mkdir(parents=True, exist_ok=True)

    def subset(frame: pd.DataFrame):
        positions = data.index.get_indexer(frame.index)
        if (positions < 0).any():
            raise RuntimeError("M5 subset contains rows outside pretokenized data")
        return encoded_data.select(positions.tolist())

    def train_score(train: pd.DataFrame, evaluation: pd.DataFrame, tag: str) -> np.ndarray:
        model = trainer = None
        try:
            set_seed(SEED)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, revision=REVISION, num_labels=2)
            install_model_length_assertion(model)
            training_args = training_arguments(run_tmp / tag)
            trainer = Trainer(model=model, args=training_args, train_dataset=subset(train), data_collator=guarded_collator)
            trainer.train()
            logits = trainer.predict(subset(evaluation)).predictions
            return np.asarray(logits[:, 1] - logits[:, 0], dtype=float)
        finally:
            if trainer is not None:
                try:
                    trainer.accelerator.free_memory()
                except Exception:
                    pass
            del trainer, model
            gc.collect(); torch.cuda.empty_cache()

    all_predictions, all_fold_metrics, isolation = [], [], []
    for outer in range(5):
        test_pairs = set(folds.loc[folds.outer_fold.eq(outer), "pair_id"])
        train_pairs = set(folds.loc[folds.outer_fold.ne(outer), "pair_id"])
        train = data[data.pair_id.isin(train_pairs)].copy(); test = data[data.pair_id.isin(test_pairs)].copy()
        y_train, y_test = labels(train), labels(test)
        inner_column = f"inner_fold_outer{outer}"
        inner_oof = np.full(len(train), np.nan)
        for inner in range(4):
            validation_pairs = set(folds.loc[folds.outer_fold.ne(outer) & folds[inner_column].eq(inner), "pair_id"])
            validation = train.pair_id.isin(validation_pairs).to_numpy()
            fit_pairs = set(train.loc[~validation, "pair_id"])
            if fit_pairs & validation_pairs or (fit_pairs | validation_pairs) & test_pairs:
                raise RuntimeError("M5 partition leakage")
            inner_oof[validation] = train_score(train.loc[~validation], train.loc[validation], f"outer_{outer}_inner_{inner}")
            isolation.append({"outer_fold": outer, "inner_fold": inner, "fit_pairs": len(fit_pairs), "validation_pairs": len(validation_pairs), "outer_evaluation_overlap": 0})
        if not np.isfinite(inner_oof).all():
            raise RuntimeError("incomplete M5 inner-OOF scores")
        calibrator = fit_platt(inner_oof, y_train)
        inner_probability = calibrated_score(calibrator, inner_oof)
        threshold, k = threshold_from_training_oof(inner_probability, y_train)
        decision = train_score(train, test, f"outer_{outer}_full")
        probability = calibrated_score(calibrator, decision); prediction = (probability > threshold).astype(int)
        all_predictions.extend(_prediction_rows(test, outer, "M5", args.condition, decision, probability, prediction, threshold))
        all_fold_metrics.append(_fold_metric_row(y_test, probability, prediction, outer, "M5", args.condition, None, threshold, k))
        # Durable per-fold checkpoint contains predictions only, never a reusable
        # model selected with outer data.
        pd.DataFrame(all_predictions).to_csv(run_tmp / "partial_outer_oof.csv", index=False)
        pd.DataFrame(all_fold_metrics).to_csv(run_tmp / "partial_fold_metrics.csv", index=False)

    output_dir = ROOT / "results" / "model_comparison" / "baseline"; stem = f"M5_{args.condition}"
    prediction_frame = pd.DataFrame(all_predictions); fold_frame = pd.DataFrame(all_fold_metrics)
    prediction_frame.to_csv(output_dir / f"{stem}_outer_oof.csv", index=False)
    fold_frame.to_csv(output_dir / f"{stem}_fold_metrics.csv", index=False)
    pooled = summarize(prediction_frame); pooled.to_csv(output_dir / f"{stem}_pooled_metrics.csv", index=False)
    write_json(output_dir / f"{stem}_fit_isolation_audit.json", {"status": "PASS", "records": isolation, "outer_evaluation_data_used_for_fitting": False})
    provenance = {
        "schema_version": "combined-v2-m5-run-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME, "revision": REVISION, "condition": args.condition,
        "seed": SEED, "learning_rate": 2e-5, "epochs": 3, "batch_size": 16,
        "warmup_ratio": 0.1, "weight_decay": 0.01, "truncation": "head128+tail382",
        "dataset_sha256": sha256_file(data_path), "fold_sha256": sha256_file(folds_path),
        "calibration": "Platt on outer-training inner-OOF raw decisions",
        "threshold": "NP order statistic alpha=0.03 delta=0.05 on calibrated outer-training inner-OOF human scores",
        "score_warning": "calibrated_study_ai_score is not a true real-world posterior probability",
    }
    write_json(ROOT / "provenance" / "experiments" / "model_comparison" / f"{stem}_run.json", provenance)
    update_baseline_table(output_dir)
    print(pooled.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
