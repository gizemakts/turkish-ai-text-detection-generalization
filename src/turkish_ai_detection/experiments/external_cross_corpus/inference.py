# -*- coding: utf-8 -*-
"""Label-blind external inference for the external cross-corpus study.

Scientific scope
----------------
This module ONLY produces pre-seal raw prediction artifacts. It never reads
external labels or subgroup metadata and never computes performance metrics.

Input projection (exact):
    external_id, text, text_sha256

Pre-seal CSV columns produced:
    external_id, text_sha256, method, seed, outer_fold, raw_score_or_logit

Model scoring:
- CE_ONLY_MATCHED / SUPCON:
  same pinned BERTurk, primary.encode_head_tail, primary.BoundedCollator,
  primary.build_model, and logit_1 - logit_0 decision score used by the
  protocol-faithful rerun.
- LIT2026_REIMPL_V1_1:
  pinned BERTurk tokenizer, first-512 truncation with max-length padding, and
  logit_1 - logit_0, matching the retained literature reimplementation.

Execution is intentionally blocked until a V2 PREEXECUTION_FREEZE exists and
binds this exact inference.py byte hash and the active config byte hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


METHODS = (
    "CE_ONLY_MATCHED",
    "SUPCON",
    "LIT2026_REIMPL_V1_1",
)
SEEDS = (42, 314, 2026)
OUTER_FOLDS = (0, 1, 2, 3, 4)

CONFIG_REL = "configs/external_cross_corpus.json"
FREEZE_REL = (
    "provenance/experiments/external_cross_corpus_v2/PREEXECUTION_FREEZE.json"
)
LIT_BINDING_REL = (
    "provenance/experiments/external_cross_corpus_v2/"
    "LIT2026_CHECKPOINT_BINDING.json"
)
LIT_IDENTITY_REPORT_REL = (
    "provenance/experiments/external_cross_corpus_v2/"
    "LIT2026_PRE_EXTERNAL_IDENTITY.json"
)

PRESEAL_ROOT_REL = (
    "artifacts/predictions/external_cross_corpus/preseal"
)

INPUT_COLUMNS = ["external_id", "text", "text_sha256"]
OUTPUT_COLUMNS = [
    "external_id",
    "text_sha256",
    "method",
    "seed",
    "outer_fold",
    "raw_score_or_logit",
]


def find_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / ".gitignore").is_file():
            return candidate
    raise RuntimeError("project root not found via .gitignore")


ROOT = find_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turkish_ai_detection.experiments.primary import run_fit as primary  # noqa: E402
from turkish_ai_detection.experiments.literature_comparison import run_fit as lit  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_utf8(text: object) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def iter_dicts(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from iter_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_dicts(value)


def freeze_contains_binding(
    freeze: dict[str, Any],
    path_rel: str,
    sha256: str,
) -> bool:
    for node in iter_dicts(freeze):
        if node.get("path") == path_rel and node.get("sha256") == sha256:
            return True
    return False


def verify_execution_freeze(config_path: Path) -> dict[str, Any]:
    freeze_path = ROOT / FREEZE_REL
    require(
        freeze_path.is_file(),
        "external inference is blocked before PREEXECUTION_FREEZE",
    )
    freeze = read_json(freeze_path)
    cfg = read_json(config_path)
    from turkish_ai_detection.experiments.external_cross_corpus.freeze_v2 import verify_freeze
    verify_freeze(config_path, freeze_path, required_code_path=Path(__file__).resolve())
    require(cfg.get("status") == "FROZEN_READY_TO_RUN" and
            cfg.get("execution_permitted") is True,
            "external execution config is not frozen/approved")

    own_path = Path(__file__).resolve()
    require(
        freeze_contains_binding(freeze, rel(own_path), sha256_file(own_path)),
        "PREEXECUTION_FREEZE does not bind this exact inference.py",
    )
    return {
        "path": rel(freeze_path),
        "sha256": sha256_file(freeze_path),
    }


def verify_config(config_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    cfg = read_json(config_path)

    evaluation = cfg.get("evaluation")
    require(isinstance(evaluation, dict), "evaluation config missing")
    require(
        evaluation.get("prediction_first_label_later") is True,
        "prediction_first_label_later must be true",
    )
    require(
        evaluation.get("prediction_input_columns_exact") == INPUT_COLUMNS,
        "prediction input contract changed",
    )

    allowed = evaluation.get("preseal_output_columns_allowed")
    require(isinstance(allowed, list), "preseal allowed-column contract missing")
    require(
        set(OUTPUT_COLUMNS).issubset(set(allowed)),
        "planned pre-seal output columns are outside the allowed contract",
    )

    forbidden = set(evaluation.get("preseal_columns_forbidden", []))
    require(
        not forbidden.intersection(OUTPUT_COLUMNS),
        "planned output intersects forbidden pre-seal columns",
    )
    require(
        evaluation.get("labels_and_subgroups_join_only_after_complete_manifest_sealed")
        is True,
        "label join must remain post-seal only",
    )
    require(
        evaluation.get("external_adaptation") is False,
        "external adaptation must remain prohibited",
    )

    bindings = cfg.get("external_data_bindings")
    require(isinstance(bindings, dict), "external_data_bindings missing")
    primary_bindings = bindings.get("primary")
    require(isinstance(primary_bindings, list), "primary external bindings missing")

    by_id: dict[str, dict[str, Any]] = {}
    for item in primary_bindings:
        require(isinstance(item, dict), "invalid primary binding")
        dataset_id = item.get("dataset_id")
        require(isinstance(dataset_id, str), "dataset_id missing")
        require(dataset_id not in by_id, f"duplicate dataset_id: {dataset_id}")
        by_id[dataset_id] = item

    require(
        set(by_id) == {"renklier2026", "tum2024_tr", "bostancioglu2026_tr"},
        f"unexpected primary external dataset set: {sorted(by_id)}",
    )
    return cfg, by_id


def load_label_blind_external(binding: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / binding["path"]
    require(path.is_file(), f"external dataset missing: {path}")
    require(
        sha256_file(path) == binding["sha256"],
        f"external dataset hash mismatch: {path}",
    )

    # Important: pandas is instructed to read ONLY the three permitted columns.
    frame = pd.read_csv(path, usecols=INPUT_COLUMNS)
    require(
        list(frame.columns) == INPUT_COLUMNS,
        f"unexpected projected column order: {list(frame.columns)}",
    )
    require(len(frame) == int(binding["rows"]), "external row count mismatch")
    require(frame["external_id"].notna().all(), "external_id contains null")
    require(frame["external_id"].astype(str).is_unique, "external_id is not unique")
    require(frame["text"].notna().all(), "text contains null")
    require(frame["text_sha256"].notna().all(), "text_sha256 contains null")

    frame["external_id"] = frame["external_id"].astype(str)
    frame["text"] = frame["text"].astype(str)
    frame["text_sha256"] = frame["text_sha256"].astype(str).str.lower()

    require((frame["text"].str.len() > 0).all(), "external text contains empty string")

    recomputed = [sha256_utf8(value) for value in frame["text"]]
    require(
        recomputed == frame["text_sha256"].tolist(),
        "text_sha256 does not match prepared UTF-8 text bytes",
    )
    return frame


def ce_supcon_payload_dir(method: str, seed: int, outer_fold: int) -> Path:
    objective = {
        "CE_ONLY_MATCHED": "ce",
        "SUPCON": "supcon",
    }[method]
    return (
        ROOT
        / "artifacts"
        / "checkpoints"
        / "external_cross_corpus"
        / "protocol_faithful_rerun"
        / f"{objective}_seed_{seed}_outer_{outer_fold}"
        / "checkpoint"
    )


def verify_frozen_training_runner(
    config_path: Path,
    freeze_path: Path,
) -> dict[str, str]:
    cfg = read_json(config_path)
    runner_binding = cfg["internal_training_bindings"][
        "dedicated_protocol_faithful_rerun_runner"
    ]
    require(
        runner_binding.get("path")
        == "src/turkish_ai_detection/experiments/external_cross_corpus/rerun_fit.py",
        "unexpected runner path in active contract",
    )

    from turkish_ai_detection.experiments.external_cross_corpus.freeze_v2 import verify_freeze

    freeze = verify_freeze(
        config_path,
        freeze_path,
        required_code_path=Path(__file__).resolve(),
    )
    for record in freeze["bindings"]:
        if record.get("path") == runner_binding["path"]:
            return {
                "path": record["path"],
                "sha256": record["sha256"],
            }
    raise RuntimeError("V2 freeze does not bind the dedicated rerun runner")


def verify_ce_supcon_payload(method: str, seed: int, outer_fold: int) -> dict[str, Any]:
    """Authenticate the finalized, protocol-faithful realization before model loading."""
    require(method in {"CE_ONLY_MATCHED", "SUPCON"}, "unsupported objective")
    require(seed in SEEDS and outer_fold in OUTER_FOLDS, "outside frozen grid")
    objective = {"CE_ONLY_MATCHED": "ce", "SUPCON": "supcon"}[method]
    payload_dir = ce_supcon_payload_dir(method, seed, outer_fold)
    cell_dir = payload_dir.parent
    model_path = payload_dir / "model.safetensors"
    meta_path = payload_dir / "config.json"
    run_path = cell_dir / "run.json"
    prediction_path = cell_dir / "internal_predictions.csv"

    meta = read_json(meta_path)
    run = read_json(run_path)
    require(meta.get("status") == "SCIENTIFIC_PROTOCOL_FAITHFUL_RERUN", "non-final checkpoint metadata")
    require(run.get("status") == "SCIENTIFIC_PROTOCOL_FAITHFUL_RERUN", "non-final training sidecar")
    for document in (meta, run):
        require(document.get("objective") == objective, "rerun objective mismatch")
        require(document.get("seed") == seed, "rerun seed mismatch")
        require(document.get("outer_fold") == outer_fold, "rerun fold mismatch")
        require(document.get("model") == primary.MODEL_NAME, "rerun model mismatch")
        require(document.get("revision") == primary.REVISION, "rerun revision mismatch")

    contract_path = ROOT / CONFIG_REL
    cfg = read_json(contract_path)
    freeze_path = ROOT / FREEZE_REL
    runner_identity = verify_frozen_training_runner(contract_path, freeze_path)
    runner_binding = cfg["internal_training_bindings"]["dedicated_protocol_faithful_rerun_runner"]
    runner_path = ROOT / runner_binding["path"]
    require(runner_binding["path"] ==
            "src/turkish_ai_detection/experiments/external_cross_corpus/rerun_fit.py",
            "unexpected runner path in active contract")
    require(
        runner_identity["path"] == runner_binding["path"],
        "V2 freeze rerun runner path mismatch",
    )
    require(sha256_file(runner_path) == runner_identity["sha256"],
            "V2 freeze rerun runner binding mismatch")
    require(run.get("contract_sha256") == sha256_file(contract_path), "training contract hash mismatch")
    require(run.get("runner_sha256") == sha256_file(runner_path), "training runner hash mismatch")
    require(run.get("checkpoint_roundtrip", {}).get("status") == "PASS_CHECKPOINT_ROUNDTRIP",
            "final checkpoint did not pass roundtrip")
    require(run.get("internal_prediction_path") == rel(prediction_path), "internal prediction path mismatch")
    require(run.get("internal_prediction_sha256") == sha256_file(prediction_path),
            "final internal prediction hash mismatch")

    reference_dir = ROOT / "results" / "primary" / objective / f"seed_{seed}"
    ref_run_path = reference_dir / f"outer_{outer_fold}_run.json"
    ref_pred_path = reference_dir / f"outer_{outer_fold}_predictions.csv"
    ref_run = read_json(ref_run_path)
    require(run.get("structural_reference_sidecar_sha256") == sha256_file(ref_run_path),
            "historical training sidecar binding mismatch")
    require(run.get("structural_reference_prediction_sha256") == sha256_file(ref_pred_path) ==
            ref_run.get("prediction_sha256"), "historical prediction binding mismatch")
    for field in ("train_pair_sha256", "test_pair_sha256"):
        require(run.get(field) == ref_run.get(field), f"training pair identity mismatch: {field}")
    audit = run.get("semantic_audit", {})
    require(audit.get("status") == "PASS_STATIC_SEMANTIC_EQUIVALENCE", "training semantic audit absent")
    require(audit.get("contract_sha256") == sha256_file(contract_path), "training semantic audit contract mismatch")
    require(audit.get("runner_sha256") == sha256_file(runner_path), "training semantic audit runner mismatch")

    payload = run.get("payload", {})
    expected = (("model", "model_sha256", model_path),
                ("config", "config_sha256", meta_path))
    for path_key, hash_key, path in expected:
        require(payload.get(path_key) == rel(path), f"final payload path mismatch: {path_key}")
        require(payload.get(hash_key) == sha256_file(path), f"final payload hash mismatch: {path_key}")

    return {
        "run_record_path": rel(run_path),
        "run_record_sha256": sha256_file(run_path),
        "internal_prediction_path": rel(prediction_path),
        "internal_prediction_sha256": run["internal_prediction_sha256"],
        "payload_dir": rel(payload_dir),
        "model_path": rel(model_path),
        "model_sha256": payload["model_sha256"],
        "metadata_path": rel(meta_path),
        "metadata_sha256": payload["config_sha256"],
        "training_runner_sha256": run["runner_sha256"],
        "training_contract_sha256": run["contract_sha256"],
    }


def load_ce_supcon(method: str, seed: int, outer_fold: int):
    # Authentication, hashes and training lineage are checked before loading weights.
    checkpoint = verify_ce_supcon_payload(method, seed, outer_fold)
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(primary.MODEL_NAME, revision=primary.REVISION)
    model = primary.build_model()
    state = load_file(str(ROOT / checkpoint["model_path"]), device="cpu")
    model.load_state_dict(state, strict=True)
    model.to("cuda")
    model.eval()
    return model, tokenizer, checkpoint


def score_ce_supcon(model, tokenizer, texts: list[str]) -> np.ndarray:
    import torch
    from datasets import Dataset
    from torch.utils.data import DataLoader

    encodings = [primary.encode_head_tail(tokenizer, text) for text in texts]
    dataset = Dataset.from_dict(
        {
            "input_ids": [item["input_ids"] for item in encodings],
            "attention_mask": [item["attention_mask"] for item in encodings],
        }
    )
    collator = primary.BoundedCollator(tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collator,
    )

    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            require("labels" not in batch, "label unexpectedly entered inference batch")
            inputs = {key: value.to("cuda") for key, value in batch.items()}
            logits, _cls, _projection = model.representations(**inputs)
            score = logits[:, 1] - logits[:, 0]
            values.extend(score.detach().cpu().numpy().tolist())

    return np.asarray(values, dtype=float)


def lit_identity_grid() -> tuple[dict[tuple[int, int], dict[str, Any]],
                                 dict[tuple[int, int], dict[str, Any]]]:
    """Validate the full, explicitly indexed 15-cell LIT binding against its identity report.

    This is label-blind and reads only JSON. Selected payload bytes are checked
    separately, immediately before the checkpoint is loaded.
    """
    binding_path = ROOT / LIT_BINDING_REL
    identity_path = ROOT / LIT_IDENTITY_REPORT_REL
    binding = read_json(binding_path)
    identity = read_json(identity_path)
    require(binding.get("protocol_id") == "EXTERNAL_CROSS_CORPUS_V2", "LIT protocol mismatch")
    require(binding.get("method") == identity.get("method") == "LIT2026_REIMPL_V1_1",
            "LIT method mismatch")
    require(identity.get("status") == "PASS_LIT2026_PRE_EXTERNAL_IDENTITY",
            "LIT2026 pre-external identity gate not PASS")
    require(identity.get("binding_path") == LIT_BINDING_REL, "LIT identity binding path mismatch")
    require(identity.get("binding_sha256") == sha256_file(binding_path),
            "LIT identity report does not match current binding bytes")
    require(identity.get("binding_status") == binding.get("status"), "LIT binding status mismatch")
    require(binding.get("checkpoint_count") == identity.get("checkpoint_count") == 15,
            "LIT checkpoint count mismatch")
    require(identity.get("grid", {}).get("seeds") == list(SEEDS) and
            identity.get("grid", {}).get("outer_folds") == list(OUTER_FOLDS) and
            identity.get("grid", {}).get("cells") == 15, "LIT identity grid mismatch")
    contract = binding.get("shared_inference_contract", {})
    require(contract.get("model") == lit.MODEL_NAME and
            contract.get("model_and_tokenizer_revision") == lit.MODEL_REVISION,
            "LIT tokenizer/model revision mismatch")

    def cells(items: Any, label: str) -> dict[tuple[int, int], dict[str, Any]]:
        require(isinstance(items, list) and len(items) == 15, f"LIT {label} cell list invalid")
        result: dict[tuple[int, int], dict[str, Any]] = {}
        for item in items:
            require(isinstance(item, dict), f"LIT {label} entry invalid")
            key = (item.get("seed"), item.get("outer_fold"))
            require(key not in result, f"duplicate LIT {label} realization: {key}")
            result[key] = item
        require(set(result) == {(s, f) for s in SEEDS for f in OUTER_FOLDS},
                f"incomplete or unexpected LIT {label} grid")
        return result

    bindings = cells(binding.get("bindings"), "binding")
    reports = cells(identity.get("cells"), "identity")
    for key in bindings:
        bound, verified = bindings[key], reports[key]
        require(verified.get("status") == "PASS", f"LIT identity cell not PASS: {key}")
        require(Path(bound["current_model"]).name == "model.safetensors",
                f"LIT unsupported model payload: {key}")
        require(Path(bound["current_model"]).parent == Path(bound["checkpoint_config"]).parent,
                f"LIT model/config directory mismatch: {key}")
        require(Path(bound["current_model"]).parent.name == verified.get("checkpoint_basename") ==
                Path(bound["historical_checkpoint"]).name,
                f"LIT selected checkpoint basename mismatch: {key}")
        for left, right in (("current_model_sha256", "model_sha256"),
                            ("checkpoint_config_sha256", "checkpoint_config_sha256"),
                            ("run_sidecar_sha256", "run_sidecar_sha256"),
                            ("retained_internal_prediction_sha256", "retained_internal_prediction_sha256")):
            require(bound.get(left) == verified.get(right),
                    f"LIT identity SHA-256 mismatch {key}: {left}")
    return bindings, reports


def strict_lit_path(path_value: str) -> Path:
    require(isinstance(path_value, str) and bool(path_value), "empty LIT binding path")
    p = Path(path_value)
    require(not p.is_absolute() and ".." not in p.parts, "unsafe LIT binding path")
    selected = (ROOT / p).resolve()
    require(selected.is_relative_to(ROOT.resolve()), "LIT binding path outside repository")
    return selected


def resolve_lit_checkpoint(seed: int, outer_fold: int) -> tuple[Path, dict[str, Any]]:
    require(seed in SEEDS and outer_fold in OUTER_FOLDS, "LIT realization outside grid")
    bindings, _reports = lit_identity_grid()
    cell = bindings[(seed, outer_fold)]
    model_path = strict_lit_path(cell["current_model"])
    config_path = strict_lit_path(cell["checkpoint_config"])
    sidecar_path = strict_lit_path(cell["run_sidecar"])
    prediction_path = strict_lit_path(cell["retained_internal_prediction"])
    require(model_path.parent == config_path.parent, "LIT checkpoint parent mismatch")
    for name, path, expected in (
        ("model", model_path, cell["current_model_sha256"]),
        ("checkpoint_config", config_path, cell["checkpoint_config_sha256"]),
        ("run_sidecar", sidecar_path, cell["run_sidecar_sha256"]),
        ("retained_internal_prediction", prediction_path, cell["retained_internal_prediction_sha256"]),
    ):
        require(sha256_file(path) == expected, f"LIT selected {name} SHA-256 mismatch")

    record = {
        "binding_path": LIT_BINDING_REL,
        "binding_sha256": sha256_file(ROOT / LIT_BINDING_REL),
        "identity_report_path": LIT_IDENTITY_REPORT_REL,
        "identity_report_sha256": sha256_file(ROOT / LIT_IDENTITY_REPORT_REL),
        "checkpoint_dir": rel(model_path.parent),
        "model_path": rel(model_path),
        "model_sha256": cell["current_model_sha256"],
        "checkpoint_config_path": rel(config_path),
        "checkpoint_config_sha256": cell["checkpoint_config_sha256"],
        "run_sidecar_path": rel(sidecar_path),
        "run_sidecar_sha256": cell["run_sidecar_sha256"],
        "retained_internal_prediction_path": rel(prediction_path),
        "retained_internal_prediction_sha256": cell["retained_internal_prediction_sha256"],
    }
    return model_path.parent, record


def load_lit(seed: int, outer_fold: int):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    checkpoint_dir, checkpoint = resolve_lit_checkpoint(seed, outer_fold)

    tokenizer = AutoTokenizer.from_pretrained(
        lit.MODEL_NAME,
        revision=lit.MODEL_REVISION,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_dir),
        local_files_only=True,
    )
    require(int(model.config.num_labels) == 2, "LIT2026 checkpoint num_labels != 2")
    model.to("cuda")
    model.eval()
    return model, tokenizer, checkpoint


def score_lit(model, tokenizer, texts: list[str]) -> np.ndarray:
    import torch

    values: list[float] = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(texts), 32):
            batch_text = texts[start : start + 32]
            encoded = tokenizer(
                batch_text,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt",
            )
            encoded = {key: value.to("cuda") for key, value in encoded.items()}

            # Historical Trainer evaluation used fp16=True. CUDA autocast mirrors
            # that inference precision while preserving the exact raw-score rule.
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):
                logits = model(**encoded).logits

            require(
                logits.ndim == 2 and logits.shape[1] == 2,
                f"unexpected LIT2026 logits shape: {tuple(logits.shape)}",
            )
            # Historical Trainer code: np.asarray(pred_out.predictions, dtype=float)
            # followed by raw = logits[:, 1] - logits[:, 0].  Subtract only
            # AFTER converting the two fp16 logits to NumPy float64.
            logits_float64 = np.asarray(logits.detach().cpu().numpy(), dtype=float)
            score = logits_float64[:, 1] - logits_float64[:, 0]
            values.extend(score.tolist())

    return np.asarray(values, dtype=float)


def output_paths(
    dataset_id: str,
    method: str,
    seed: int,
    outer_fold: int,
) -> tuple[Path, Path]:
    root = (
        ROOT
        / PRESEAL_ROOT_REL
        / dataset_id
        / method
        / f"seed_{seed}"
    )
    csv_path = root / f"outer_{outer_fold}.csv"
    json_path = root / f"outer_{outer_fold}.json"
    return csv_path, json_path


def run_one(
    config_path: Path,
    dataset_id: str,
    method: str,
    seed: int,
    outer_fold: int,
) -> dict[str, Any]:
    import torch

    require(torch.cuda.is_available(), "external inference requires CUDA")
    require(method in METHODS, f"unsupported method: {method}")
    require(seed in SEEDS, f"seed outside fixed grid: {seed}")
    require(outer_fold in OUTER_FOLDS, f"outer fold outside fixed grid: {outer_fold}")

    _cfg, bindings = verify_config(config_path)
    require(dataset_id in bindings, f"unknown primary dataset_id: {dataset_id}")

    freeze_record = verify_execution_freeze(config_path)

    csv_path, json_path = output_paths(dataset_id, method, seed, outer_fold)
    require(not csv_path.exists(), f"refusing to overwrite pre-seal prediction: {csv_path}")
    require(not json_path.exists(), f"refusing to overwrite inference sidecar: {json_path}")

    if method in {"CE_ONLY_MATCHED", "SUPCON"}:
        model, tokenizer, checkpoint = load_ce_supcon(method, seed, outer_fold)
        external = load_label_blind_external(bindings[dataset_id])
        texts = external["text"].tolist()
        raw = score_ce_supcon(model, tokenizer, texts)
        score_contract = (
            "primary.encode_head_tail + primary.BoundedCollator + "
            "primary.build_model representations; raw_score=logit_1-logit_0"
        )
    else:
        model, tokenizer, checkpoint = load_lit(seed, outer_fold)
        external = load_label_blind_external(bindings[dataset_id])
        texts = external["text"].tolist()
        raw = score_lit(model, tokenizer, texts)
        score_contract = (
            "pinned BERTurk tokenizer; truncation=True,max_length=512,"
            "padding=max_length; fp16 CUDA autocast; "
            "NumPy float64 logits then raw_score=logit_1-logit_0"
        )

    require(len(raw) == len(external), "prediction row count mismatch")
    require(np.isfinite(raw).all(), "non-finite raw external score")

    prediction = external.loc[:, ["external_id", "text_sha256"]].copy()
    prediction["method"] = method
    prediction["seed"] = int(seed)
    prediction["outer_fold"] = int(outer_fold)
    prediction["raw_score_or_logit"] = raw
    prediction = prediction.loc[:, OUTPUT_COLUMNS]

    # Belt-and-suspenders: no external text or forbidden metadata may be emitted.
    require(list(prediction.columns) == OUTPUT_COLUMNS, "output column contract mismatch")
    forbidden = {
        "text",
        "label",
        "binary_label",
        "generator",
        "domain",
        "scenario",
        "generation_type",
        "subgroup_metadata",
    }
    require(
        not forbidden.intersection(prediction.columns),
        "forbidden pre-seal output column detected",
    )

    atomic_csv(prediction, csv_path)

    sidecar = {
        "schema_version": "external-cross-corpus-preseal-inference-v1",
        "status": "COMPLETE_LABEL_BLIND_PRESEAL_INFERENCE",
        "created_utc": utc_now(),
        "dataset_id": dataset_id,
        "method": method,
        "seed": int(seed),
        "outer_fold": int(outer_fold),
        "rows": int(len(prediction)),
        "input_projection_exact": INPUT_COLUMNS,
        "output_columns_exact": OUTPUT_COLUMNS,
        "external_labels_read": False,
        "external_subgroups_read": False,
        "performance_metrics_computed": False,
        "external_adaptation": False,
        "calibration_applied": False,
        "score_contract": score_contract,
        "external_source": {
            "path": bindings[dataset_id]["path"],
            "sha256": bindings[dataset_id]["sha256"],
        },
        "checkpoint": checkpoint,
        "preexecution_freeze": freeze_record,
        "inference_implementation": {
            "path": rel(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "prediction": {
            "path": rel(csv_path),
            "sha256": sha256_file(csv_path),
        },
    }
    atomic_json(json_path, sidecar)

    # Sidecar is intentionally created only after the prediction hash is known.
    return {
        "status": sidecar["status"],
        "dataset_id": dataset_id,
        "method": method,
        "seed": seed,
        "outer_fold": outer_fold,
        "rows": len(prediction),
        "prediction_path": rel(csv_path),
        "prediction_sha256": sidecar["prediction"]["sha256"],
        "sidecar_path": rel(json_path),
        "sidecar_sha256": sha256_file(json_path),
        "external_labels_read": False,
        "performance_metrics_computed": False,
    }


def validate_only(config_path: Path) -> dict[str, Any]:
    _cfg, bindings = verify_config(config_path)
    lit_bindings, _lit_report = lit_identity_grid()
    return {
        "status": "PASS_LABEL_BLIND_INFERENCE_STATIC_CONTRACT",
        "lit_binding_verified_cells": len(lit_bindings),
        "lit_binding_sha256": sha256_file(ROOT / LIT_BINDING_REL),
        "config_path": rel(config_path),
        "config_sha256": sha256_file(config_path),
        "implementation_path": rel(Path(__file__)),
        "implementation_sha256": sha256_file(Path(__file__)),
        "primary_external_datasets": sorted(bindings),
        "input_columns_exact": INPUT_COLUMNS,
        "output_columns": OUTPUT_COLUMNS,
        "note": (
            "Validation-only mode does not read external CSV content, load "
            "checkpoints, perform inference, or require PREEXECUTION_FREEZE."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / CONFIG_REL,
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dataset-id", choices=(
        "renklier2026",
        "tum2024_tr",
        "bostancioglu2026_tr",
    ))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--outer-fold", type=int, choices=OUTER_FOLDS)
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_path = config_path.resolve()

    if args.validate_only:
        print(json.dumps(validate_only(config_path), ensure_ascii=False, indent=2))
        return 0

    missing = [
        name
        for name, value in (
            ("--dataset-id", args.dataset_id),
            ("--method", args.method),
            ("--seed", args.seed),
            ("--outer-fold", args.outer_fold),
        )
        if value is None
    ]
    require(not missing, f"missing required execution arguments: {', '.join(missing)}")

    result = run_one(
        config_path=config_path,
        dataset_id=args.dataset_id,
        method=args.method,
        seed=int(args.seed),
        outer_fold=int(args.outer_fold),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
