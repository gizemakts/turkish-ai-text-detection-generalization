# -*- coding: utf-8 -*-
"""Single-fit runner for the prespecified Turkish AI Detection primary CE-vs-SupCon ablation.

One invocation trains exactly one objective/seed/outer-fold combination.
No calibration or threshold fitting occurs in this primary discrimination stage.

Examples
--------
python src/turkish_ai_detection/experiments/primary/run_fit.py --objective ce --seed 42 --outer-fold 0
python src/turkish_ai_detection/experiments/primary/run_fit.py --objective supcon --seed 42 --outer-fold 0
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

from turkish_ai_detection.data.study import sha256_file, write_json  # noqa: E402
from turkish_ai_detection.evaluation.metrics import condition_text, labels, _pauc  # noqa: E402
from turkish_ai_detection.experiments.baseline.berturk import (  # noqa: E402
    HEAD_WORDPIECES, MAX_MODEL_TOKENS, MODEL_NAME, REVISION,
    TAIL_WORDPIECES, encode_head_tail,
)
from turkish_ai_detection.experiments.model_development.supcon.train import (  # noqa: E402
    BalancedClassSampler, BoundedCollator, CLS_DIM, DATASET_SHA256,
    DATA_PATH, EFFECTIVE_BATCH_SIZE, FOLD_PATH, FOLD_SHA256,
    GRADIENT_ACCUMULATION_STEPS, LAMBDA_SUPCON, PHYSICAL_BATCH_SIZE,
    PROJECTION_DIM, TEMPERATURE, build_model, supervised_contrastive_loss,
)

EXPERIMENT = "COMBINED_V2_PRIMARY_ABLATION_V1"
CONDITION = "A"
CONFIG_PATH = ROOT / "configs" / "primary_evaluation.json"
OUT = ROOT / "results" / "primary"
PROV = ROOT / "provenance" / "experiments" / "primary"
STATE = ROOT / "artifacts" / "checkpoints" / "primary"

ALLOWED_SEEDS = (42, 137, 314, 2026, 3407)
ALLOWED_OUTERS = (0, 1, 2, 3, 4)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pair_hash(values) -> str:
    """Stable fingerprint of the unordered training/test pair IDs."""
    return hashlib.sha256(
        "\n".join(sorted(map(str, values))).encode("utf-8")
    ).hexdigest()




def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def training_arguments(output_dir: Path, seed: int):
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=PHYSICAL_BATCH_SIZE,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_steps=0.1,
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
        seed=int(seed),
        data_seed=int(seed),
    )


def make_seeded_trainer_class(objective: str, seed: int):
    import torch
    import torch.nn.functional as F
    from transformers import Trainer

    if objective not in {"ce", "supcon"}:
        raise ValueError(objective)

    class PrimaryAblationTrainer(Trainer):
        """Same two-microbatch true-32 path for both objectives."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._pending_microbatch = None
            self.last_group_audit = None
            self.study_seed = int(seed)
            self.study_objective = objective

        def _get_train_sampler(self, train_dataset=None):
            dataset = train_dataset if train_dataset is not None else self.train_dataset
            return BalancedClassSampler(
                dataset["labels"],
                self._train_batch_size,
                self.study_seed,
            )

        def training_step(self, model, inputs, num_items_in_batch=None):
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()
            inputs = self._prepare_inputs(inputs)
            batch_labels = inputs.pop("labels").long()

            with self.compute_loss_context_manager():
                logits, _cls, projection = model(**inputs, return_components=True)

            current = {
                "logits": logits,
                "projection": projection,
                "labels": batch_labels,
            }

            if self._pending_microbatch is None:
                if len(batch_labels) != PHYSICAL_BATCH_SIZE:
                    raise RuntimeError("first microbatch must contain exactly 16 examples")
                self._pending_microbatch = current
                return logits.detach().sum() * 0.0

            first = self._pending_microbatch
            self._pending_microbatch = None

            all_logits = torch.cat((first["logits"], current["logits"]), dim=0)
            all_projection = torch.cat(
                (first["projection"], current["projection"]), dim=0
            )
            all_labels = torch.cat((first["labels"], current["labels"]), dim=0)

            if len(all_labels) != EFFECTIVE_BATCH_SIZE:
                raise RuntimeError("primary ablation requires exactly 32 candidates")
            counts = torch.bincount(all_labels, minlength=2)
            if counts.tolist() != [16, 16]:
                raise RuntimeError(
                    f"candidate group must be 16 HUMAN + 16 AI, observed {counts.tolist()}"
                )

            ce = F.cross_entropy(all_logits, all_labels, reduction="mean")
            if self.study_objective == "supcon":
                supcon = supervised_contrastive_loss(
                    all_projection, all_labels, TEMPERATURE
                )
                total = ce + LAMBDA_SUPCON * supcon
                supcon_value = float(supcon.detach().cpu())
            else:
                # The projection forward is intentionally retained because the same
                # model.forward path is used, but projection is excluded from the CE
                # objective graph. This is the only causal objective difference.
                total = ce
                supcon_value = None

            accelerator_gas = int(
                getattr(self.accelerator, "gradient_accumulation_steps", 1)
            )
            if accelerator_gas != 1:
                raise RuntimeError(
                    "Primary ablation requires Transformers-v5 Trainer-managed "
                    f"accumulation with Accelerator GAS=1; observed {accelerator_gas}"
                )

            self.last_group_audit = {
                "objective": self.study_objective,
                "candidate_size": int(len(all_labels)),
                "human": int(counts[0]),
                "ai": int(counts[1]),
                "ce": float(ce.detach().cpu()),
                "supcon": supcon_value,
                "total": float(total.detach().cpu()),
                "accelerator_gas": accelerator_gas,
            }

            self.accelerator.backward(total)
            return total.detach()

    return PrimaryAblationTrainer


def make_dataset(frame: pd.DataFrame, encodings: list[dict]):
    from datasets import Dataset

    positions = frame.index.to_numpy(int)
    return Dataset.from_dict({
        "labels": labels(frame).tolist(),
        "input_ids": [encodings[i]["input_ids"] for i in positions],
        "attention_mask": [encodings[i]["attention_mask"] for i in positions],
    })


def score(model, frame, encodings, collator):
    import torch
    from torch.utils.data import DataLoader

    dataset = make_dataset(frame, encodings)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collator,
    )
    values = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            inputs = {key: value.to("cuda") for key, value in batch.items()}
            logits, _cls, _projection = model.representations(**inputs)
            values.extend(
                (logits[:, 1] - logits[:, 0]).detach().cpu().numpy().tolist()
            )
    return np.asarray(values, dtype=float)


def metric_record(frame: pd.DataFrame) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = frame["label"].astype(str).eq("AI").astype(int).to_numpy()
    s = frame["raw_decision_score"].to_numpy(float)
    return {
        "n": int(len(frame)),
        "n_human": int(np.sum(y == 0)),
        "n_ai": int(np.sum(y == 1)),
        "roc_auc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "std_pauc_003": float(_pauc(y, s)),
    }


def run_one(objective: str, seed: int, outer: int) -> dict:
    import torch
    import transformers
    from transformers import AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("primary ablation requires CUDA")
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed not prespecified: {seed}")
    if outer not in ALLOWED_OUTERS:
        raise ValueError(f"outer fold not prespecified: {outer}")
    if objective not in {"ce", "supcon"}:
        raise ValueError(objective)

    # Validate the active publication configuration and input structure, not the
    # byte identity of a historical, differently named research runner.
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"missing primary config: {CONFIG_PATH}")
    if not DATA_PATH.is_file() or not FOLD_PATH.is_file():
        raise FileNotFoundError("authorized Combined-v2 data and fold inputs are required")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if seed not in config["seeds"] or outer not in config["outer_folds"]:
        raise ValueError("seed/fold is absent from the active primary config")

    # Validate only the active dataset and split identities. Historical code
    # source hashes are intentionally not part of the public runner contract.
    observed = {
        "dataset": sha256_file(DATA_PATH),
        "folds": sha256_file(FOLD_PATH),
    }
    expected = {
        "dataset": DATASET_SHA256,
        "folds": FOLD_SHA256,
    }
    if observed != expected:
        raise RuntimeError("active dataset/fold SHA-256 identity mismatch")
    if config.get("dataset_sha256") != DATASET_SHA256:
        raise RuntimeError("primary config dataset identity mismatch")
    if config.get("fold_sha256") != FOLD_SHA256:
        raise RuntimeError("primary config fold identity mismatch")

    data = pd.read_csv(DATA_PATH).copy()
    folds = pd.read_csv(FOLD_PATH)
    if len(data) != 3000 or data.text_id.nunique() != 3000:
        raise RuntimeError("unexpected Combined-v2 observation count or duplicate text IDs")
    required_data = {"pair_id", "text_id", "text", "label", "domain", "generator"}
    required_folds = {"pair_id", "outer_fold"}
    if not required_data.issubset(data.columns) or not required_folds.issubset(folds.columns):
        raise RuntimeError("required paired study / outer fold schema is missing")
    if data.pair_id.nunique() != 1500:
        raise RuntimeError("expected 1,500 distinct human–AI pairs")
    if set(data.label.astype(str).str.upper()) != {"HUMAN", "AI"}:
        raise RuntimeError("expected HUMAN/AI labels")
    if not data.groupby("pair_id").label.apply(lambda x: sorted(x.astype(str).str.upper().tolist()) == ["AI", "HUMAN"]).all():
        raise RuntimeError("each study pair must contain exactly one HUMAN and one AI text")
    if folds.pair_id.duplicated().any() or set(folds.pair_id) != set(data.pair_id):
        raise RuntimeError("outer fold assignments must cover each pair exactly once")
    if set(folds.outer_fold) != set(config["outer_folds"]):
        raise RuntimeError("outer fold identifiers do not match the active config")

    data["model_text"] = condition_text(data, CONDITION)
    test_pairs = set(folds.loc[folds.outer_fold.eq(outer), "pair_id"])
    train_pairs = set(folds.loc[folds.outer_fold.ne(outer), "pair_id"])

    if train_pairs & test_pairs:
        raise RuntimeError("outer train/test pair leakage")

    train = data[data.pair_id.isin(train_pairs)].copy()
    test = data[data.pair_id.isin(test_pairs)].copy()

    if len(train) != 2400 or len(test) != 600:
        raise RuntimeError("unexpected primary train/test row count")
    if set(train.pair_id) & set(test.pair_id):
        raise RuntimeError("pair overlap after row filtering")

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
        "train_pair_sha256": pair_hash(train_pairs),
        "test_pair_sha256": pair_hash(test_pairs),
        "train_pair_count": int(len(train_pairs)),
        "test_pair_count": int(len(test_pairs)),
        "train_test_pair_overlap": 0,
    }

    if pred_path.exists() or sidecar_path.exists():
        raise FileExistsError(
            f"output already exists for {tag}; use a clean validation copy so "
            "historical results are not overwritten or mistaken for a fresh reproduction"
        )

    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)
    encodings = [
        encode_head_tail(tokenizer, text)
        for text in data.model_text.astype(str)
    ]
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
        raise RuntimeError("no completed true-32 objective group was observed")

    raw = score(model, test, encodings, collator)
    prediction = test[
        ["text_id", "pair_id", "label", "domain", "generator"]
    ].copy()
    prediction["outer_fold"] = int(outer)
    prediction["seed"] = int(seed)
    prediction["objective"] = objective
    prediction["condition"] = CONDITION
    prediction["raw_decision_score"] = raw

    if len(prediction) != 600 or not prediction.text_id.is_unique:
        raise RuntimeError("invalid primary outer prediction shape")
    if not np.isfinite(prediction.raw_decision_score).all():
        raise RuntimeError("non-finite primary raw score")

    atomic_csv(prediction, pred_path)
    prediction_sha256 = sha256_file(pred_path)
    metrics = metric_record(prediction)

    record = {
        **binding,
        "schema_version": "combined-v2-primary-ablation-single-fit-v1",
        "created_utc": utc_now(),
        "status": "COMPLETE",
        "model": MODEL_NAME,
        "revision": REVISION,
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": 2e-5,
        "epochs": 3,
        "weight_decay": 0.01,
        "warmup_steps": 0.1,
        "lambda_supcon": 0.0 if objective == "ce" else LAMBDA_SUPCON,
        "temperature": None if objective == "ce" else TEMPERATURE,
        "calibration_used": False,
        "threshold_used": False,
        "outer_test_rows_used_for_training": 0,
        "last_group_audit": trainer.last_group_audit,
        "metrics": metrics,
        "prediction_sha256": prediction_sha256,
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
    args = parser.parse_args()
    record = run_one(args.objective, args.seed, args.outer_fold)
    # Do not reveal outer-test discrimination metrics during the confirmatory run.
    # Scientific results remain sealed in the per-fit sidecar until all 50 fits
    # complete and the prespecified statistics script performs the unblinding.
    safe = {
        "status": record["status"],
        "experiment": record["experiment"],
        "objective": record["objective"],
        "seed": record["seed"],
        "outer_fold": record["outer_fold"],
        "metrics_blinded": True,
    }
    print(json.dumps(safe, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
