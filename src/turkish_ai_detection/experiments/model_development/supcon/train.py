# -*- coding: utf-8 -*-
"""Supervised contrastive BERTurk training with 32-example contrastive groups.

The training objective combines cross-entropy and supervised contrastive loss
using a single backward pass per effective batch.
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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from turkish_ai_detection.data.study import sha256_file, write_json  # noqa: E402
from turkish_ai_detection.evaluation.metrics import (  # noqa: E402
    _fold_metric_row, _pauc, calibrated_score, condition_text, fit_platt,
    labels, summarize, threshold_from_training_oof,
)
from turkish_ai_detection.experiments.baseline.berturk import (  # noqa: E402
    HEAD_WORDPIECES, MAX_MODEL_TOKENS, MODEL_NAME, REVISION, SEED,
    TAIL_WORDPIECES, encode_head_tail,
)

EXPERIMENT = "COMBINED_V2_SUPCON_V2"
MODEL_ID = "C1R"
CONDITION = "A"
DATASET_SHA256 = "ea5797ac19348957472626d1a23b740de531cd19577f613e1a73894e34d66b44"
FOLD_SHA256 = "16ffae53a5d9027a95e12f3975fd062f17038ef5bf8970a74dd3ef7339d9a100"
FREEZE_SHA256 = "12fe84d69e96fd3d0ffba37e8572257ee1ab428fa481d6ab0ddc28db4a44e040"
LAMBDA_SUPCON = 0.1
TEMPERATURE = 0.07
CLS_DIM = 768
PROJECTION_DIM = 128
PHYSICAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
EFFECTIVE_BATCH_SIZE = 32
EXPECTED_FITS = 25
BACKWARD_SCALING_POLICY = "transformers_v5_trainer_managed_gas_accelerator_gas_must_be_1"

DATA_PATH = ROOT / "data" / "study" / "paired_study.csv"
FOLD_PATH = ROOT / "data" / "study" / "nested_folds.csv"
CONFIG_PATH = ROOT / "configs" / "models" / "supervised_contrastive.json"
FREEZE_PATH = ROOT / "provenance" / "history" / "study_development" / "legacy_workflow_records" / "freezes" / "COMBINED_V2_BASELINE_FROZEN_20260930T203203Z.json"
F1_MANIFEST = ROOT / "provenance" / "history" / "study_development" / "legacy_workflow_records" / "training" / "feature_fusion" / "run_manifest.json"
OUT = ROOT / "results" / "validation" / "supcon_training_validation"
PROV = ROOT / "provenance" / "experiments" / "supcon_training_validation"
PARTS = OUT / "parts"
STATE = ROOT / "artifacts" / "checkpoints" / "supervised_contrastive"
READINESS_PATH = OUT / "readiness_audit.json"
SMOKE_PATH = OUT / "smoke_test_report.json"


def utc_now() -> str:
    return __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()


def pair_hash(values) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode("utf-8")).hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def verify_inputs() -> dict:
    observed = {"freeze": sha256_file(FREEZE_PATH), "dataset": sha256_file(DATA_PATH), "folds": sha256_file(FOLD_PATH)}
    expected = {"freeze": FREEZE_SHA256, "dataset": DATASET_SHA256, "folds": FOLD_SHA256}
    if observed != expected:
        raise RuntimeError(f"frozen input hash mismatch: {observed}")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    for relative in ("outputs/combined_v2/M4_A_outer_oof.csv", "outputs/combined_v2/M5_A_outer_oof.csv"):
        physical = ROOT / relative.replace("outputs/combined_v2/", "results/historical/model_evaluation/baseline/")
        if sha256_file(physical) != freeze["result_artifact_hashes"][relative]:
            raise RuntimeError(f"frozen reference changed: {relative}")
    f1 = json.loads(F1_MANIFEST.read_text(encoding="utf-8"))
    relative = "outputs/combined_v2/feature_fusion_v1/F1_A_outer_oof.csv"
    if f1.get("status") != "COMPLETE" or sha256_file(ROOT / "results/historical/model_development/feature_fusion/F1_A_outer_oof.csv") != f1["outputs"][relative]:
        raise RuntimeError("completed F1 reference is missing or hash-mismatched")
    return observed


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    verify_inputs()
    data = pd.read_csv(DATA_PATH)
    if len(data) != 3000 or data.text_id.nunique() != 3000:
        raise RuntimeError("Combined-v2 dataset identity/count changed")
    data = data.copy(); data["model_text"] = condition_text(data, CONDITION)
    return data, pd.read_csv(FOLD_PATH)


def supervised_contrastive_masks(labels_tensor):
    import torch
    labels_tensor = labels_tensor.reshape(-1)
    same = labels_tensor[:, None].eq(labels_tensor[None, :])
    self_mask = torch.eye(len(labels_tensor), dtype=torch.bool, device=labels_tensor.device)
    positives = same & ~self_mask
    negatives = ~same
    return positives, negatives, self_mask


def supervised_contrastive_loss(embeddings, labels_tensor, temperature: float = TEMPERATURE):
    import torch
    import torch.nn.functional as F
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    normalized = F.normalize(embeddings, p=2, dim=1)
    positives, _negatives, self_mask = supervised_contrastive_masks(labels_tensor)
    positive_counts = positives.sum(dim=1)
    if torch.any(positive_counts == 0):
        raise ValueError("every SupCon anchor requires at least one same-label positive")
    logits = normalized @ normalized.T / float(temperature)
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    mean_positive_log_probability = (log_probability.masked_fill(~positives, 0.0).sum(dim=1) / positive_counts)
    loss = -mean_positive_log_probability.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite supervised contrastive loss")
    return loss


class BalancedClassSampler:
    """Balanced microbatches paired into complete 32-example SupCon groups.

    A balanced remainder smaller than 32 is dropped after the deterministic
    epoch shuffle. Thus an 1,800-row inner fit uses 1,792 rows (896/class),
    while a 2,400-row outer fit uses every row. No partial SupCon group exists.
    """
    def __init__(self, labels_values, batch_size: int = PHYSICAL_BATCH_SIZE, seed: int = SEED):
        values = np.asarray(labels_values, dtype=int)
        if batch_size < 4 or batch_size % 2:
            raise ValueError("balanced sampler requires an even batch size >= 4")
        if set(values.tolist()) != {0, 1}:
            raise ValueError("balanced sampler requires both binary labels")
        if np.sum(values == 0) != np.sum(values == 1):
            raise ValueError("prespecified Combined-v2 sampler requires equal class counts")
        self.labels = values; self.batch_size = int(batch_size); self.seed = int(seed); self.epoch = 0
        self.effective_group_size = EFFECTIVE_BATCH_SIZE
        self.usable_size = (len(values) // self.effective_group_size) * self.effective_group_size
        if self.usable_size < self.effective_group_size:
            raise ValueError("training partition cannot form one complete 32-example SupCon group")

    def __len__(self):
        return self.usable_size

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        class_zero = rng.permutation(np.flatnonzero(self.labels == 0))
        class_one = rng.permutation(np.flatnonzero(self.labels == 1))
        usable_per_class = self.usable_size // 2
        class_zero = class_zero[:usable_per_class]
        class_one = class_one[:usable_per_class]
        half = self.batch_size // 2; order = []
        for start in range(0, len(class_zero), half):
            zero = class_zero[start:start + half]
            one = class_one[start:start + half]
            batch = np.concatenate((zero, one))
            order.extend(rng.permutation(batch).tolist())
        return iter(order)


def build_model():
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel
    from transformers.modeling_outputs import SequenceClassifierOutput

    class SupConModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = AutoModel.from_pretrained(MODEL_NAME, revision=REVISION)
            self.config = self.bert.config
            if int(self.config.hidden_size) != CLS_DIM:
                raise RuntimeError("BERTurk CLS dimension changed")
            self.dropout = torch.nn.Dropout(0.1)
            self.classifier = torch.nn.Linear(CLS_DIM, 2)
            self.projection = torch.nn.Sequential(
                torch.nn.Linear(CLS_DIM, 256), torch.nn.GELU(), torch.nn.Linear(256, PROJECTION_DIM),
            )
            self.accepts_loss_kwargs = False

        def representations(self, input_ids, attention_mask=None, **kwargs):
            if input_ids.shape[-1] > MAX_MODEL_TOKENS:
                raise AssertionError("C1R refuses BERT inputs over 512 tokens")
            encoded = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            cls = encoded.last_hidden_state[:, 0, :]
            if cls.shape[-1] != CLS_DIM:
                raise AssertionError("C1R CLS representation is not 768-dimensional")
            projection = F.normalize(self.projection(cls), p=2, dim=1)
            if projection.shape[-1] != PROJECTION_DIM:
                raise AssertionError("C1R projection is not 128-dimensional")
            logits = self.classifier(self.dropout(cls))
            return logits, cls, projection

        def forward(self, input_ids=None, attention_mask=None, labels=None, return_components=False, **kwargs):
            logits, cls, projection = self.representations(input_ids, attention_mask, **kwargs)
            if return_components:
                return logits, cls, projection
            loss = None
            if labels is not None:
                ce = F.cross_entropy(logits, labels.long())
                supcon = supervised_contrastive_loss(projection, labels, TEMPERATURE)
                loss = ce + LAMBDA_SUPCON * supcon
            return SequenceClassifierOutput(loss=loss, logits=logits)

    return SupConModel()


def training_arguments(output_dir: Path):
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=str(output_dir), learning_rate=2e-5,
        per_device_train_batch_size=PHYSICAL_BATCH_SIZE,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=3, weight_decay=0.01, warmup_steps=0.1,
        save_strategy="no", logging_strategy="epoch", report_to=[],
        seed=SEED, data_seed=SEED,
    )


class BoundedCollator:
    def __init__(self, tokenizer):
        from transformers import DataCollatorWithPadding
        self.base = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        batch = self.base(features)
        if batch["input_ids"].shape[-1] > MAX_MODEL_TOKENS:
            raise AssertionError("C1 collator produced an overlength input")
        return batch


def make_trainer_class():
    import torch
    import torch.nn.functional as F
    from transformers import Trainer
    class BalancedTrainer(Trainer):
        """Retain two forward graphs and backpropagate one true 32-row loss."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._pending_supcon_microbatch = None
            self.last_group_audit = None

        def _get_train_sampler(self, train_dataset=None):
            dataset = train_dataset if train_dataset is not None else self.train_dataset
            return BalancedClassSampler(dataset["labels"], self._train_batch_size, SEED)

        def training_step(self, model, inputs, num_items_in_batch=None):
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()
            inputs = self._prepare_inputs(inputs)
            batch_labels = inputs.pop("labels").long()
            with self.compute_loss_context_manager():
                logits, _cls, projection = model(**inputs, return_components=True)
            current = {"logits": logits, "projection": projection, "labels": batch_labels}
            if self._pending_supcon_microbatch is None:
                if len(batch_labels) != PHYSICAL_BATCH_SIZE:
                    raise RuntimeError("first C1 microbatch is not a complete 16-example batch")
                self._pending_supcon_microbatch = current
                # The Trainer loop is inside an accumulation window here. No
                # backward call and therefore no optimizer step is possible.
                return logits.detach().sum() * 0.0

            first = self._pending_supcon_microbatch
            self._pending_supcon_microbatch = None
            all_logits = torch.cat((first["logits"], current["logits"]), dim=0)
            all_projection = torch.cat((first["projection"], current["projection"]), dim=0)
            all_labels = torch.cat((first["labels"], current["labels"]), dim=0)
            if len(all_labels) != EFFECTIVE_BATCH_SIZE:
                raise RuntimeError("C1R requires exactly 32 candidates per SupCon objective")
            counts = torch.bincount(all_labels, minlength=2)
            if counts.tolist() != [16, 16]:
                raise RuntimeError(f"C1R candidate group must be 16 HUMAN + 16 AI, observed {counts.tolist()}")
            ce = F.cross_entropy(all_logits, all_labels, reduction="mean")
            supcon = supervised_contrastive_loss(all_projection, all_labels, TEMPERATURE)
            total = ce + LAMBDA_SUPCON * supcon
            positives, negatives, self_mask = supervised_contrastive_masks(all_labels)
            projection_norms = torch.linalg.vector_norm(all_projection, dim=1)
            cross_microbatch = torch.zeros_like(positives)
            cross_microbatch[:PHYSICAL_BATCH_SIZE, PHYSICAL_BATCH_SIZE:] = True
            cross_microbatch[PHYSICAL_BATCH_SIZE:, :PHYSICAL_BATCH_SIZE] = True
            self.last_group_audit = {
                "candidate_size": len(all_labels), "similarity_shape": [len(all_labels), len(all_labels)],
                "human": int(counts[0]), "ai": int(counts[1]),
                "positive_pairs": int(positives.sum()), "negative_pairs": int(negatives.sum()),
                "self_pairs_positive": int((positives & self_mask).sum()),
                "cross_microbatch_positive_pairs": int((positives & cross_microbatch).sum()),
                "cross_microbatch_negative_pairs": int((negatives & cross_microbatch).sum()),
                "projection_norm_min": float(projection_norms.min().detach().cpu()),
                "projection_norm_max": float(projection_norms.max().detach().cpu()),
                "ce": float(ce.detach().cpu()), "supcon": float(supcon.detach().cpu()),
                "total": float(total.detach().cpu()),
            }
            # Transformers v5 Trainer manages gradient accumulation itself and
            # constructs Accelerator with GAS=1. Therefore the already-combined
            # 32-example mean objective must be backpropagated exactly once with
            # NO additional x2 scaling. A runtime guard prevents silent behavior
            # changes if a future Trainer/Accelerate integration differs.
            accelerator_gas = int(getattr(self.accelerator, "gradient_accumulation_steps", 1))
            if accelerator_gas != 1:
                raise RuntimeError(
                    "C1R requires Transformers Trainer-managed accumulation with "
                    f"Accelerator gradient_accumulation_steps=1; observed {accelerator_gas}"
                )
            self.accelerator.backward(total)
            return total.detach()
    return BalancedTrainer


def make_dataset(frame: pd.DataFrame, encodings: list[dict]):
    from datasets import Dataset
    positions = frame.index.to_numpy(int)
    return Dataset.from_dict({
        "labels": labels(frame).tolist(),
        "input_ids": [encodings[i]["input_ids"] for i in positions],
        "attention_mask": [encodings[i]["attention_mask"] for i in positions],
    })


def sampler_batch_audit(labels_values, batch_size=PHYSICAL_BATCH_SIZE):
    sampler = BalancedClassSampler(labels_values, batch_size, SEED)
    order = list(iter(sampler)); values = np.asarray(labels_values, int)
    rows = []
    for number, start in enumerate(range(0, len(order), batch_size)):
        batch_labels = values[np.asarray(order[start:start + batch_size])]
        counts = {str(label): int(np.sum(batch_labels == label)) for label in (0, 1)}
        rows.append({"batch": number, "size": len(batch_labels), "human": counts["0"], "ai": counts["1"], "both_labels": set(batch_labels.tolist()) == {0, 1}, "minimum_per_class": min(counts.values())})
    return rows


def smoke_test() -> dict:
    import torch
    import transformers
    from datasets import Dataset
    from transformers import AutoTokenizer, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("C1 smoke test requires CUDA")
    verify_inputs(); set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)
    short = encode_head_tail(tokenizer, "Bu kısa bir Türkçe karşıt öğrenme testidir.")
    long_text = "uzunörnek " * 700; long = encode_head_tail(tokenizer, long_text)
    raw_long = len(tokenizer(long_text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"])
    encoded = [short if i % 2 == 0 else long for i in range(EFFECTIVE_BATCH_SIZE)]
    smoke_labels = [i % 2 for i in range(EFFECTIVE_BATCH_SIZE)]
    dataset = Dataset.from_dict({
        "labels": smoke_labels, "input_ids": [row["input_ids"] for row in encoded],
        "attention_mask": [row["attention_mask"] for row in encoded],
    })
    args = training_arguments(STATE / "smoke_test")
    model = build_model().to("cuda"); collator = BoundedCollator(tokenizer)
    TrainerClass = make_trainer_class()
    trainer = TrainerClass(model=model, args=args, train_dataset=dataset, data_collator=collator)
    loader = trainer.get_train_dataloader()
    trainer.create_optimizer(); trainer.optimizer.zero_grad(set_to_none=True)
    model.train(); torch.cuda.reset_peak_memory_stats()
    iterator = iter(loader); first_batch = next(iterator); second_batch = next(iterator)
    observed_batches = []
    for batch in (first_batch, second_batch):
        counts = torch.bincount(batch["labels"].detach().cpu(), minlength=2).tolist()
        observed_batches.append({"size": len(batch["labels"]), "human": counts[0], "ai": counts[1]})
    classifier_before = model.classifier.weight.detach().clone()
    first_loss = trainer.training_step(model, first_batch)
    classifier_after_first = model.classifier.weight.detach().clone()
    optimizer_step_after_first = False
    if trainer.last_group_audit is not None or trainer._pending_supcon_microbatch is None:
        raise RuntimeError("C1R smoke: first microbatch incorrectly completed a contrastive group")
    second_loss = trainer.training_step(model, second_batch)
    if trainer.last_group_audit is None or trainer._pending_supcon_microbatch is not None:
        raise RuntimeError("C1R smoke: second microbatch did not complete the 32-example group")
    backbone_gradient = next((p.grad for p in model.bert.parameters() if p.grad is not None), None)
    projection_gradient = model.projection[-1].weight.grad
    classifier_gradient = model.classifier.weight.grad
    trainer.optimizer.step(); trainer.optimizer.zero_grad(set_to_none=True)
    classifier_after_step = model.classifier.weight.detach().clone()
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    group = trainer.last_group_audit
    record = {
        "schema_version": "combined-v2-supcon-v2-smoke-v1", "created_utc": utc_now(),
        "status": "PASS", "experiment": EXPERIMENT, "condition": CONDITION,
        "training_launched": False, "trainer_train_called": False,
        "cuda_available": True, "gpu_model": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__, "cuda_version": torch.version.cuda,
        "model": MODEL_NAME, "revision": REVISION,
        "raw_long_wordpieces": raw_long, "short_model_tokens": len(short["input_ids"]),
        "long_model_tokens": len(long["input_ids"]), "maximum_batch_tokens": MAX_MODEL_TOKENS,
        "truncation": f"head{HEAD_WORDPIECES}+tail{TAIL_WORDPIECES}",
        "cls_dimension": CLS_DIM, "projection_dimension": PROJECTION_DIM,
        "projection_norm_min": group["projection_norm_min"], "projection_norm_max": group["projection_norm_max"],
        "contrastive_candidate_size": group["candidate_size"], "similarity_matrix_shape": group["similarity_shape"],
        "candidate_human": group["human"], "candidate_ai": group["ai"],
        "self_pairs_positive": group["self_pairs_positive"],
        "cross_microbatch_positive_pairs": group["cross_microbatch_positive_pairs"],
        "cross_microbatch_negative_pairs": group["cross_microbatch_negative_pairs"],
        "ce_loss": group["ce"], "supcon_loss": group["supcon"], "combined_loss": group["total"],
        "ce_loss_finite": bool(np.isfinite(group["ce"])), "supcon_loss_finite": bool(np.isfinite(group["supcon"])), "combined_loss_finite": bool(np.isfinite(group["total"])),
        "lambda_supcon": LAMBDA_SUPCON, "temperature": TEMPERATURE,
        "backbone_gradient": backbone_gradient is not None and bool(torch.isfinite(backbone_gradient).all()),
        "classification_head_gradient": classifier_gradient is not None and bool(torch.isfinite(classifier_gradient).all()),
        "projection_head_gradient": projection_gradient is not None and bool(torch.isfinite(projection_gradient).all()),
        "physical_batch_size": PHYSICAL_BATCH_SIZE, "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE, "observed_microbatches": observed_batches,
        "both_labels_each_microbatch": all(row["human"] >= 2 and row["ai"] >= 2 for row in observed_batches),
        "first_microbatch_loss_return": float(first_loss.cpu()), "second_microbatch_combined_loss_return": float(second_loss.cpu()),
        "optimizer_step_after_first_microbatch": optimizer_step_after_first,
        "parameters_unchanged_after_first_microbatch": bool(torch.equal(classifier_before, classifier_after_first)),
        "optimizer_step_completed_after_32_examples": bool(not torch.equal(classifier_before, classifier_after_step)),
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "training_arguments_instantiated": True, "trainer_initialized": True,
        "runner_sha256": sha256_file(Path(__file__)), "config_sha256": sha256_file(CONFIG_PATH),
        "dataset_sha256": sha256_file(DATA_PATH), "fold_sha256": sha256_file(FOLD_PATH),
        "condition_B_executed": False,
    }
    required = [record[key] for key in ("ce_loss_finite", "supcon_loss_finite", "combined_loss_finite", "backbone_gradient", "classification_head_gradient", "projection_head_gradient", "both_labels_each_microbatch", "parameters_unchanged_after_first_microbatch", "optimizer_step_completed_after_32_examples")]
    structure_ok = record["contrastive_candidate_size"] == 32 and record["similarity_matrix_shape"] == [32, 32] and record["candidate_human"] == record["candidate_ai"] == 16 and record["self_pairs_positive"] == 0 and record["cross_microbatch_positive_pairs"] > 0 and record["cross_microbatch_negative_pairs"] > 0
    if not all(required) or not structure_ok or not np.allclose([record["projection_norm_min"], record["projection_norm_max"]], 1.0, atol=1e-5):
        record["status"] = "FAIL"; raise RuntimeError(f"C1 smoke failed: {record}")
    OUT.mkdir(parents=True, exist_ok=True); PROV.mkdir(parents=True, exist_ok=True)
    write_json(SMOKE_PATH, record); write_json(PROV / "smoke_test_report.json", record)
    del trainer, model, loader
    gc.collect(); torch.cuda.empty_cache()
    return record


def validate_prediction_part(part, expected, outer, role, inner):
    required = {"text_id", "pair_id", "label", "domain", "generator", "outer_fold", "role", "model", "condition", "raw_score"}
    if not required <= set(part.columns) or len(part) != 600 or part.text_id.nunique() != 600:
        raise RuntimeError("invalid C1 prediction part")
    if set(part.outer_fold) != {outer} or set(part.role) != {role} or set(part.model) != {MODEL_ID} or set(part.condition) != {CONDITION}:
        raise RuntimeError("C1 part identity mismatch")
    if role == "inner" and ("inner_fold" not in part or set(part.inner_fold.astype(int)) != {int(inner)}):
        raise RuntimeError("C1 inner fold mismatch")
    columns = ["text_id", "pair_id", "label", "domain", "generator"]
    if not part[columns].sort_values("text_id").reset_index(drop=True).equals(expected[columns].sort_values("text_id").reset_index(drop=True)):
        raise RuntimeError("C1 part metadata mismatch")
    if not np.isfinite(part.raw_score).all(): raise RuntimeError("C1 non-finite prediction")


def validate_final_outer_oof(frame):
    if len(frame) != 3000 or frame.text_id.nunique() != 3000 or not frame.text_id.is_unique:
        raise RuntimeError("C1R final outer OOF must contain exactly 3000 unique texts")
    if set(frame.model) != {MODEL_ID} or set(frame.condition) != {CONDITION}:
        raise RuntimeError("C1R final identity mismatch")


def cls_moments(cls_values: np.ndarray, labels_values: np.ndarray, outer: int) -> pd.DataFrame:
    rows = []
    for label_name, selector in (("ALL", np.ones(len(labels_values), bool)), ("HUMAN", labels_values == 0), ("AI", labels_values == 1)):
        values = cls_values[selector]
        for dimension in range(CLS_DIM):
            column = values[:, dimension]
            rows.append({"outer_fold": outer, "label": label_name, "dimension": dimension, "count": len(column), "sum": column.sum(), "sumsq": np.square(column).sum(), "min": column.min(), "max": column.max()})
    return pd.DataFrame(rows)


def projection_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    projection_columns = [f"projection_{index:03d}" for index in range(PROJECTION_DIM)]
    rows = []
    groups = [("ALL_DATA", "ALL", frame)]
    groups += [("DOMAIN", str(value), group) for value, group in frame.groupby("domain", sort=True)]
    groups += [("GENERATOR", str(value), group) for value, group in frame.groupby("generator", sort=True)]
    for scope, value, group in groups:
        z = group[projection_columns].to_numpy(float); y = group.label.eq("AI").astype(int).to_numpy()
        z0, z1 = z[y == 0], z[y == 1]
        def within(matrix):
            return float((np.square(matrix.sum(axis=0)).sum() - len(matrix)) / (len(matrix) * (len(matrix) - 1)))
        between = float(z0.mean(axis=0) @ z1.mean(axis=0))
        c0, c1 = z0.mean(axis=0), z1.mean(axis=0)
        cosine = float((c0 @ c1) / (np.linalg.norm(c0) * np.linalg.norm(c1)))
        rows.append({"scope": scope, "scope_value": value, "n": len(group), "human_within_cosine": within(z0), "ai_within_cosine": within(z1), "between_class_cosine": between, "centroid_cosine_distance": 1.0 - cosine, "centroid_euclidean_separation": float(np.linalg.norm(c0 - c1))})
    return pd.DataFrame(rows)


def run_full() -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, set_seed
    if not torch.cuda.is_available(): raise RuntimeError("C1 full run requires CUDA")
    if not READINESS_PATH.exists() or not SMOKE_PATH.exists(): raise RuntimeError("current readiness and smoke records required")
    runner_hash, config_hash = sha256_file(Path(__file__)), sha256_file(CONFIG_PATH)
    for path, allowed in ((READINESS_PATH, "READY_NOT_LAUNCHED"), (SMOKE_PATH, "PASS")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != allowed or record.get("runner_sha256") != runner_hash or record.get("config_sha256") != config_hash:
            raise RuntimeError(f"stale or failed gate: {path}")
    data, folds = load_inputs(); set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)
    encodings = [encode_head_tail(tokenizer, text) for text in data.model_text.astype(str)]
    collator = BoundedCollator(tokenizer); TrainerClass = make_trainer_class()
    OUT.mkdir(parents=True, exist_ok=True); PROV.mkdir(parents=True, exist_ok=True); PARTS.mkdir(parents=True, exist_ok=True); STATE.mkdir(parents=True, exist_ok=True)
    common = {"experiment": EXPERIMENT, "model_id": MODEL_ID, "condition": CONDITION, "condition_B_executed": False, "dataset_sha256": DATASET_SHA256, "fold_sha256": FOLD_SHA256, "runner_sha256": runner_hash, "config_sha256": config_hash, "model": MODEL_NAME, "revision": REVISION, "lambda_supcon": LAMBDA_SUPCON, "temperature": TEMPERATURE, "physical_batch_size": PHYSICAL_BATCH_SIZE, "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS, "effective_batch_size": EFFECTIVE_BATCH_SIZE, "sampler": "training-label-only-balanced", "backward_scaling_policy": BACKWARD_SCALING_POLICY}
    completed = []

    def score(model, frame, return_representations=False):
        dataset = make_dataset(frame, encodings); loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collator)
        scores, cls_rows, projection_rows = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch.pop("labels", None); inputs = {key: value.to("cuda") for key, value in batch.items()}
                logits, cls, projection = model.representations(**inputs)
                scores.extend((logits[:, 1] - logits[:, 0]).detach().cpu().numpy().tolist())
                if return_representations:
                    cls_rows.append(cls.detach().cpu().numpy()); projection_rows.append(projection.detach().cpu().numpy())
        if return_representations:
            return np.asarray(scores), np.vstack(cls_rows), np.vstack(projection_rows)
        return np.asarray(scores), None, None

    def train_or_reuse(fit, evaluation, outer, role, inner):
        tag = f"outer_{outer}_inner_{inner}" if role == "inner" else f"outer_{outer}_full"
        prediction_path = PARTS / f"{tag}.csv"; sidecar_path = PARTS / f"{tag}.json"
        projection_path = PARTS / f"{tag}_projection.csv"; cls_path = PARTS / f"{tag}_cls_moments.csv"
        fit_pairs, evaluation_pairs = set(fit.pair_id), set(evaluation.pair_id)
        outer_test = set(folds.loc[folds.outer_fold.eq(outer), "pair_id"])
        expected_fit = 900 if role == "inner" else 1200
        if fit_pairs & evaluation_pairs or len(fit_pairs) != expected_fit or len(evaluation_pairs) != 300: raise RuntimeError("C1R partition failure")
        if role == "inner" and (fit_pairs | evaluation_pairs) & outer_test: raise RuntimeError("C1R outer leakage")
        if role == "outer" and evaluation_pairs != outer_test: raise RuntimeError("C1R outer identity mismatch")
        sampler_audit = sampler_batch_audit(labels(fit), PHYSICAL_BATCH_SIZE)
        if not all(row["both_labels"] and row["minimum_per_class"] >= 2 for row in sampler_audit): raise RuntimeError("C1R sampler batch contract failed")
        binding = {**common, "outer_fold": outer, "role": role, "inner_fold": inner, "fit_pair_count": len(fit_pairs), "evaluation_pair_count": len(evaluation_pairs), "fit_pair_sha256": pair_hash(fit_pairs), "evaluation_pair_sha256": pair_hash(evaluation_pairs), "outer_test_fit_overlap": len(fit_pairs & outer_test), "sampler_batches": len(sampler_audit), "minimum_observed_per_class": min(row["minimum_per_class"] for row in sampler_audit)}
        expected_files = [prediction_path, sidecar_path] + ([projection_path, cls_path] if role == "outer" else [])
        if any(path.exists() for path in expected_files):
            if not all(path.exists() for path in expected_files): raise RuntimeError(f"incomplete C1R resume state {tag}")
            saved = json.loads(sidecar_path.read_text(encoding="utf-8"))
            expected_binding = {**binding, "prediction_sha256": sha256_file(prediction_path)}
            if role == "outer": expected_binding.update({"projection_sha256": sha256_file(projection_path), "cls_moments_sha256": sha256_file(cls_path)})
            if {key: saved.get(key) for key in expected_binding} != expected_binding: raise RuntimeError(f"C1R resume binding mismatch {tag}")
            prediction = pd.read_csv(prediction_path); validate_prediction_part(prediction, evaluation, outer, role, inner)
            completed.append({"tag": tag, "reused": True, "prediction_sha256": expected_binding["prediction_sha256"]})
            return prediction
        model = trainer = None
        try:
            set_seed(SEED); model = build_model()
            trainer = TrainerClass(model=model, args=training_arguments(STATE / tag), train_dataset=make_dataset(fit, encodings), data_collator=collator)
            trainer.train()
            values, cls_values, projection_values = score(model, evaluation, role == "outer")
        finally:
            if trainer is not None:
                try: trainer.accelerator.free_memory()
                except Exception: pass
        prediction = evaluation[["text_id", "pair_id", "label", "domain", "generator"]].copy()
        prediction["outer_fold"] = outer; prediction["role"] = role
        if role == "inner": prediction["inner_fold"] = int(inner)
        prediction["model"] = MODEL_ID; prediction["condition"] = CONDITION; prediction["raw_score"] = values
        validate_prediction_part(prediction, evaluation, outer, role, inner); atomic_csv(prediction, prediction_path)
        extra = {}
        if role == "outer":
            projection = prediction[["text_id", "pair_id", "label", "domain", "generator", "outer_fold"]].copy()
            for index in range(PROJECTION_DIM): projection[f"projection_{index:03d}"] = projection_values[:, index]
            atomic_csv(projection, projection_path)
            moments = cls_moments(cls_values, labels(evaluation), outer); atomic_csv(moments, cls_path)
            extra = {"projection_sha256": sha256_file(projection_path), "cls_moments_sha256": sha256_file(cls_path)}
        sidecar = {**binding, "created_utc": utc_now(), "prediction_sha256": sha256_file(prediction_path), **extra}
        write_json(sidecar_path, sidecar); completed.append({"tag": tag, "reused": False, "prediction_sha256": sidecar["prediction_sha256"]})
        write_json(OUT / "progress.json", {"status": "IN_PROGRESS", "completed_fits": len(completed), "expected_fits": EXPECTED_FITS, "parts": completed, "condition_B_executed": False})
        del trainer, model
        gc.collect(); torch.cuda.empty_cache()
        return prediction

    predictions, fold_metrics, leakage = [], [], []
    for outer in range(5):
        test_pairs = set(folds.loc[folds.outer_fold.eq(outer), "pair_id"]); train_pairs = set(folds.loc[folds.outer_fold.ne(outer), "pair_id"])
        train, test = data[data.pair_id.isin(train_pairs)].copy(), data[data.pair_id.isin(test_pairs)].copy(); inner_col = f"inner_fold_outer{outer}"
        inner_parts = []
        for inner in range(4):
            held_pairs = set(folds.loc[folds.outer_fold.ne(outer) & folds[inner_col].eq(inner), "pair_id"]); held_mask = train.pair_id.isin(held_pairs)
            fit, held = train.loc[~held_mask].copy(), train.loc[held_mask].copy(); inner_parts.append(train_or_reuse(fit, held, outer, "inner", inner))
            leakage.append({"outer_fold": outer, "inner_fold": inner, "fit_pairs": 900, "evaluation_pairs": 300, "outer_test_fit_overlap": 0, "batch_sampler_source": "fit labels only"})
        inner_oof = pd.concat(inner_parts, ignore_index=True)
        aligned = train[["text_id", "label"]].merge(inner_oof[["text_id", "raw_score"]], on="text_id", validate="one_to_one")
        y_train = labels(aligned); inner_scores = aligned.raw_score.to_numpy(float)
        calibrator = fit_platt(inner_scores, y_train); inner_probability = calibrated_score(calibrator, inner_scores)
        threshold, k = threshold_from_training_oof(inner_probability, y_train)
        outer_raw = train_or_reuse(train, test, outer, "outer", None)
        probability = calibrated_score(calibrator, outer_raw.raw_score.to_numpy(float)); prediction = (probability > threshold).astype(int)
        result = outer_raw[["text_id", "pair_id", "label", "domain", "generator", "outer_fold"]].copy(); result["model"] = MODEL_ID; result["condition"] = CONDITION; result["raw_decision_score"] = outer_raw.raw_score; result["calibrated_study_ai_score"] = probability; result["binary_prediction"] = prediction; result["binary_decision_threshold"] = threshold
        atomic_csv(result, OUT / f"outer_{outer}_predictions.csv")
        write_json(OUT / f"outer_{outer}_calibration_threshold.json", {"outer_fold": outer, "calibration": "Platt logistic", "calibration_fit_source": "outer-training inner-OOF only", "calibration_fit_rows": 2400, "outer_test_rows_used": 0, "coefficient": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0]), "threshold": threshold, "k_exceed": k, "threshold_fit_source": "calibrated outer-training inner-OOF human scores only", "outer_test_labels_used": False})
        y_test = labels(result); fold_metrics.append(_fold_metric_row(y_test, probability, prediction, outer, MODEL_ID, CONDITION, None, threshold, k)); predictions.append(result)
        leakage.append({"outer_fold": outer, "inner_fold": None, "fit_pairs": 1200, "evaluation_pairs": 300, "outer_test_fit_overlap": 0, "calibration_outer_rows": 0, "threshold_outer_rows": 0})
    prediction_frame = pd.concat(predictions, ignore_index=True).sort_values("text_id").reset_index(drop=True); validate_final_outer_oof(prediction_frame)
    atomic_csv(prediction_frame, OUT / "C1_A_outer_oof.csv"); atomic_csv(pd.DataFrame(fold_metrics), OUT / "C1_A_fold_metrics.csv"); atomic_csv(summarize(prediction_frame), OUT / "C1_A_pooled_metrics.csv")
    projection_frame = pd.concat([pd.read_csv(PARTS / f"outer_{outer}_full_projection.csv") for outer in range(5)], ignore_index=True).sort_values("text_id").reset_index(drop=True)
    atomic_csv(projection_frame, OUT / "C1_A_projection_representations.csv")
    atomic_csv(pd.concat([pd.read_csv(PARTS / f"outer_{outer}_full_cls_moments.csv") for outer in range(5)], ignore_index=True), OUT / "C1_A_cls_summary_statistics.csv")
    atomic_csv(projection_metrics(projection_frame), OUT / "C1_A_representation_metrics.csv")
    write_json(OUT / "leakage_audit.json", {"status": "PASS", "records": leakage, "outer_test_used_for_training_batching_calibration_or_threshold": False, "condition_B_executed": False})
    bootstrap_comparisons(prediction_frame)
    files = [path for path in OUT.rglob("*") if path.is_file() and "training_state" not in path.parts]
    manifest = {"schema_version": "combined-v2-supcon-v2-run-v1", "created_utc": utc_now(), "status": "COMPLETE", "experiment": EXPERIMENT, "condition": CONDITION, "condition_B_executed": False, "completed_fits": len(completed), "expected_fits": EXPECTED_FITS, "frozen_baseline_modified": False, "runner_sha256": runner_hash, "config_sha256": config_hash, "dataset_sha256": DATASET_SHA256, "fold_sha256": FOLD_SHA256, "freeze_sha256": FREEZE_SHA256, "outputs": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in files}}
    write_json(PROV / "run_manifest.json", manifest); write_json(OUT / "progress.json", {"status": "COMPLETE", "completed_fits": len(completed), "expected_fits": EXPECTED_FITS, "parts": completed, "condition_B_executed": False})
    return manifest


def bootstrap_comparisons(c1):
    from sklearn.metrics import confusion_matrix, roc_auc_score
    references = {"M5": ROOT / "results/historical/model_evaluation/baseline/M5_A_outer_oof.csv", "M4": ROOT / "results/historical/model_evaluation/baseline/M4_A_outer_oof.csv", "F1": ROOT / "results/historical/model_development/feature_fusion/F1_A_outer_oof.csv"}
    c1 = c1.sort_values("text_id").reset_index(drop=True); frames = {name: pd.read_csv(path).sort_values("text_id").reset_index(drop=True) for name, path in references.items()}
    for name, frame in frames.items():
        if not c1[["text_id", "pair_id", "label"]].equals(frame[["text_id", "pair_id", "label"]]): raise RuntimeError(f"C1R comparison alignment failed: {name}")
    pairs = np.asarray(sorted(c1.pair_id.unique())); indices = {pair: np.flatnonzero(c1.pair_id.to_numpy() == pair) for pair in pairs}; rng = np.random.default_rng(42); draws = []
    for bootstrap_id in range(1, 2001):
        idx = np.concatenate([indices[pair] for pair in rng.choice(pairs, len(pairs), replace=True)]); y = c1.label.iloc[idx].eq("AI").astype(int).to_numpy(); c_score = c1.calibrated_study_ai_score.to_numpy(float)[idx]; c_pred = c1.binary_prediction.to_numpy(int)[idx]; c_tn, c_fp, c_fn, c_tp = confusion_matrix(y, c_pred, labels=[0, 1]).ravel()
        for name, frame in frames.items():
            score = frame.calibrated_study_ai_score.to_numpy(float)[idx]; pred = frame.binary_prediction.to_numpy(int)[idx]; tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            draws.append({"bootstrap_id": bootstrap_id, "comparator": name, "delta_roc_auc": roc_auc_score(y, c_score) - roc_auc_score(y, score), "delta_std_pauc_003": _pauc(y, c_score) - _pauc(y, score), "delta_fpr": c_fp/(c_fp+c_tn)-fp/(fp+tn), "delta_tpr": c_tp/(c_tp+c_fn)-tp/(tp+fn)})
    draw_frame = pd.DataFrame(draws); summary = []
    for name in frames:
        subset = draw_frame[draw_frame.comparator.eq(name)]
        for metric in ("roc_auc", "std_pauc_003", "fpr", "tpr"):
            values = subset[f"delta_{metric}"].to_numpy(float); summary.append({"comparison": f"C1-{name}", "metric": metric, "mean_delta": values.mean(), "ci95_lower": np.quantile(values,.025), "ci95_upper": np.quantile(values,.975), "B":2000,"seed":42,"unit":"pair_id"})
    atomic_csv(draw_frame, OUT / "pair_bootstrap_draws.csv"); atomic_csv(pd.DataFrame(summary), OUT / "pair_bootstrap_summary.csv")


def main() -> int:
    parser = argparse.ArgumentParser(); modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke-test", action="store_true"); modes.add_argument("--authorize-25-condition-a-finetunes", action="store_true")
    args = parser.parse_args()
    if args.smoke_test: print(json.dumps(smoke_test(), ensure_ascii=False, indent=2)); return 0
    print(json.dumps(run_full(), ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
