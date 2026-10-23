# -*- coding: utf-8 -*-
"""Single-fit runner for LIT2026_REIMPL_V1.

This is a same-data reimplementation of the 2026 BERTurk literature recipe.
It is not a reproduction of the paper's published score.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
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
from turkish_ai_detection.evaluation.metrics import condition_text, _pauc  # noqa: E402
from turkish_ai_detection.experiments.model_development.supcon.train import DATA_PATH, DATASET_SHA256, FOLD_PATH, FOLD_SHA256  # noqa: E402

EXPERIMENT = "LIT2026_REIMPL_V1_1"
CONDITION = "A"
CONFIG_PATH = ROOT / "configs" / "literature_comparison.json"
OUT = ROOT / "results" / "literature_comparison"
PROV = ROOT / "provenance" / "experiments" / "literature_comparison"
STATE = ROOT / "artifacts" / "checkpoints" / "literature_comparison"
FREEZE_PATH = PROV / "PREEXECUTION_FREEZE.json"

PRIMARY_RUNNER = ROOT / "src" / "turkish_ai_detection/experiments/primary/run_fit.py"
PRIMARY_CONFIG = ROOT / "configs" / "primary_evaluation.json"
PRIMARY_FREEZE = ROOT / "provenance" / "experiments" / "primary" / "PREEXECUTION_FREEZE.json"
PRIMARY_MANIFEST = ROOT / "results" / "primary" / "final_statistics" / "final_statistics_manifest.json"

SOURCE_TRANSCRIPTION = PROV / "LIT2026_SOURCE_TRANSCRIPTION.md"

ALLOWED_SEEDS = (42, 314, 2026)
ALLOWED_OUTERS = (0, 1, 2, 3, 4)
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
MODEL_REVISION = "b6e1de16c983e0f2c70664591ea3f22810072608"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def pair_hash(values) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode("utf-8")).hexdigest()


def stable_rank(pair_id, outer):
    key = f"LIT2026_VAL_V1|outer={outer}|pair={pair_id}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def make_train_val_split(data: pd.DataFrame, outer_train_pairs: set, outer: int):
    pairs = (
        data[data.pair_id.isin(outer_train_pairs)]
        [["pair_id", "generator", "domain"]]
        .drop_duplicates("pair_id")
        .copy()
    )
    if len(pairs) != 1200:
        raise RuntimeError(f"unexpected outer-training pair count: {len(pairs)}")
    if not (data.groupby("pair_id")["generator"].nunique() == 1).all():
        raise RuntimeError("generator metadata is not pair-consistent")
    if not (data.groupby("pair_id")["domain"].nunique() == 1).all():
        raise RuntimeError("domain metadata is not pair-consistent")

    selected = []
    group_records = []
    for (generator, domain), g in pairs.groupby(["generator", "domain"], sort=True):
        g = g.copy()
        target = int(round(len(g) * 0.125))
        # Expected Combined-v2 counts are exactly divisible for this 70/10/20 adaptation.
        if target <= 0:
            raise RuntimeError("validation stratum target is zero")
        g["_rank"] = g.pair_id.map(lambda p: stable_rank(p, outer))
        g = g.sort_values(["_rank", "pair_id"])
        chosen = g.head(target)
        selected.extend(chosen.pair_id.astype(str).tolist())
        group_records.append({
            "generator": str(generator),
            "domain": str(domain),
            "outer_train_pairs": int(len(g)),
            "validation_pairs": int(target),
        })

    val_pairs = set(selected)
    train_pairs = set(map(str, outer_train_pairs)) - val_pairs
    if len(val_pairs) != 150 or len(train_pairs) != 1050:
        raise RuntimeError(
            f"literature 70/10/20 adaptation failed: train={len(train_pairs)}, val={len(val_pairs)}"
        )
    if train_pairs & val_pairs:
        raise RuntimeError("train/validation pair overlap")
    return train_pairs, val_pairs, group_records


def metrics(frame: pd.DataFrame) -> dict:
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score,
        f1_score, precision_score, recall_score, roc_auc_score,
    )
    y = frame.label.astype(str).eq("AI").astype(int).to_numpy()
    s = frame.raw_decision_score.to_numpy(float)
    pred = frame.predicted_label.astype(int).to_numpy()
    return {
        "n": int(len(frame)),
        "n_human": int((y == 0).sum()),
        "n_ai": int((y == 1).sum()),
        "roc_auc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "standardized_pauc_003": float(_pauc(y, s)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_ai": float(precision_score(y, pred, zero_division=0)),
        "recall_ai": float(recall_score(y, pred, zero_division=0)),
        "f1_ai": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def verify_freeze():
    if not FREEZE_PATH.is_file():
        raise RuntimeError("LIT2026 scientific training is locked until PREEXECUTION_FREEZE.json exists")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_READY_TO_RUN":
        raise RuntimeError("invalid LIT2026 freeze")
    freeze["_current_reproduction"] = validate_current_migrated_reproduction(
        ROOT, "literature_comparison", Path(__file__)
    )
    if freeze.get("lit2026_results_unblinded_before_freeze") is not False:
        raise RuntimeError("freeze does not certify blinded LIT2026 state")
    return freeze


def run_one(seed: int, outer: int):
    import torch
    import transformers
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("LIT2026 reimplementation requires CUDA")
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed not prespecified: {seed}")
    if outer not in ALLOWED_OUTERS:
        raise ValueError(f"outer fold not prespecified: {outer}")

    freeze = verify_freeze()
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if sha256_file(DATA_PATH) != DATASET_SHA256 or sha256_file(FOLD_PATH) != FOLD_SHA256:
        raise RuntimeError("frozen dataset/fold identity changed")
    if cfg["dataset_sha256"] != DATASET_SHA256 or cfg["fold_sha256"] != FOLD_SHA256:
        raise RuntimeError("LIT2026 config identity mismatch")
    if sha256_file(SOURCE_TRANSCRIPTION) != cfg["authoritative_literature_source"]["source_transcription_sha256"]:
        raise RuntimeError("literature source transcription changed")

    data = pd.read_csv(DATA_PATH).copy()
    folds = pd.read_csv(FOLD_PATH)
    if len(data) != 3000 or data.text_id.nunique() != 3000 or data.pair_id.nunique() != 1500:
        raise RuntimeError("Combined-v2 identity/count changed")
    data["model_text"] = condition_text(data, CONDITION)

    outer_test_pairs = set(map(str, folds.loc[folds.outer_fold.eq(outer), "pair_id"]))
    outer_train_pairs = set(map(str, folds.loc[folds.outer_fold.ne(outer), "pair_id"]))
    if outer_test_pairs & outer_train_pairs:
        raise RuntimeError("outer train/test pair leakage")
    train_pairs, val_pairs, val_strata = make_train_val_split(data, outer_train_pairs, outer)
    if (train_pairs & outer_test_pairs) or (val_pairs & outer_test_pairs):
        raise RuntimeError("outer test leaked into train/validation")

    pair_text = data.pair_id.astype(str)
    train = data[pair_text.isin(train_pairs)].copy()
    val = data[pair_text.isin(val_pairs)].copy()
    test = data[pair_text.isin(outer_test_pairs)].copy()
    if (len(train), len(val), len(test)) != (2100, 300, 600):
        raise RuntimeError(f"unexpected 70/10/20 row counts: {len(train)}, {len(val)}, {len(test)}")
    for name, frame, expected in (("train", train, 1050), ("val", val, 150), ("test", test, 300)):
        if frame.pair_id.nunique() != expected:
            raise RuntimeError(f"unexpected {name} pair count")

    tag = f"seed_{seed}_outer_{outer}"
    out_dir = OUT / f"seed_{seed}"
    pred_path = out_dir / f"outer_{outer}_predictions.csv"
    sidecar_path = out_dir / f"outer_{outer}_run.json"

    binding = {
        "experiment": EXPERIMENT,
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
        "source_transcription_sha256": sha256_file(SOURCE_TRANSCRIPTION),
        "train_pair_count": 1050,
        "validation_pair_count": 150,
        "test_pair_count": 300,
        "train_pair_sha256": pair_hash(train_pairs),
        "validation_pair_sha256": pair_hash(val_pairs),
        "test_pair_sha256": pair_hash(outer_test_pairs),
        "train_validation_overlap": 0,
        "train_test_overlap": 0,
        "validation_test_overlap": 0,
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)

    def encode_frame(frame):
        enc = tokenizer(
            frame.model_text.astype(str).tolist(),
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        y = frame.label.astype(str).eq("AI").astype(int).tolist()
        return Dataset.from_dict({**enc, "labels": y})

    train_ds = encode_frame(train)
    val_ds = encode_frame(val)
    test_ds = encode_frame(test)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred = np.argmax(logits, axis=1)
        return {"f1": float(f1_score(labels, pred, zero_division=0))}

    args = TrainingArguments(
        output_dir=str(STATE / tag),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_steps=500,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        max_grad_norm=0.0,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="no",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to=[],
        seed=int(seed),
        data_seed=int(seed),
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    pred_out = trainer.predict(test_ds)
    logits = np.asarray(pred_out.predictions, dtype=float)
    if logits.shape != (600, 2):
        raise RuntimeError(f"unexpected literature logits shape: {logits.shape}")
    raw = logits[:, 1] - logits[:, 0]
    pred = np.argmax(logits, axis=1).astype(int)

    prediction = test[["text_id", "pair_id", "label", "domain", "generator"]].copy()
    prediction["outer_fold"] = int(outer)
    prediction["seed"] = int(seed)
    prediction["method"] = "LIT2026_REIMPL"
    prediction["condition"] = CONDITION
    prediction["raw_decision_score"] = raw
    prediction["predicted_label"] = pred
    if not np.isfinite(raw).all():
        raise RuntimeError("non-finite LIT2026 score")
    atomic_csv(prediction, pred_path)

    m = metrics(prediction)
    state = trainer.state
    record = {
        **binding,
        "schema_version": "combined-v2-lit2026-single-fit-v1",
        "status": "COMPLETE",
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "literature_doi": cfg["authoritative_literature_source"]["doi"],
        "train_rows": 2100,
        "validation_rows": 300,
        "test_rows": 600,
        "validation_strata": val_strata,
        "outer_test_rows_used_for_training": 0,
        "outer_test_used_for_model_selection": False,
        "best_model_checkpoint": state.best_model_checkpoint,
        "best_validation_f1": None if state.best_metric is None else float(state.best_metric),
        "trained_epochs": float(state.epoch or 0.0),
        "max_epochs": 5,
        "early_stopping_patience": 2,
        "warmup_steps": 500,
        "batch_size": 8,
        "max_sequence_length": 512,
        "truncation": "first_512_tokens",
        "weight_decay": 0.01,
        "learning_rate": 2e-5,
        "optimizer": "adamw_torch",
        "mixed_precision": "fp16",
        "gradient_clipping": "disabled",
        "metrics": m,
        "prediction_sha256": sha256_file(pred_path),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_model": torch.cuda.get_device_name(0),
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(sidecar_path, record)

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--outer-fold", required=True, type=int, choices=list(ALLOWED_OUTERS))
    a = p.parse_args()
    record = run_one(a.seed, a.outer_fold)
    safe = {
        "status": record["status"],
        "experiment": record["experiment"],
        "method": "LIT2026_REIMPL",
        "seed": record["seed"],
        "outer_fold": record["outer_fold"],
        "prediction_sha256": record["prediction_sha256"],
        "lit2026_metrics_blinded": True,
    }
    print("TURKISH_AI_DETECTION_SAFE_RESULT\t" + json.dumps(safe, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
