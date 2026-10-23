"""CPU-only authenticity/completeness verification for the 20 added terminal cells."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "robustness_analyses"
BASE = ROOT / "artifacts/checkpoints/external_cross_corpus/protocol_faithful_rerun"
EXTENSION = OUT / "external/EXTERNAL_5SEED_EXTENSION_CONFIG.json"
WRAPPER = OUT / "code/external_5seed_extension.py"
BASE_CONFIG = ROOT / "configs/external_cross_corpus.json"
BASE_RUNNER = ROOT / "src/turkish_ai_detection/experiments/external_cross_corpus/rerun_fit.py"
FOLDS = ROOT / "data/study/nested_folds.csv"

EXPECTED = {
    "base_config": "0a1314ee51fe79f820404ec7cd4e845709411a43ff0388376170f602f300d76c",
    "base_runner": "96f2af846f15cb3bb66bea7e8addba3f49da0c1b3457f96b8b1eb1d46f30e558",
    "folds": "16ffae53a5d9027a95e12f3975fd062f17038ef5bf8970a74dd3ef7339d9a100",
    "extension": "2f8654fd52ed1388714b212a0dfeb66c8c5a54a886811321e4bf1bad290f035a",
}
SEEDS = (137, 3407)
OBJECTIVES = ("ce", "supcon")
METHOD = {"ce": "CE_ONLY_MATCHED", "supcon": "SUPCON"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def main() -> int:
    require(sha(BASE_CONFIG) == EXPECTED["base_config"], "active terminal config hash changed")
    require(sha(BASE_RUNNER) == EXPECTED["base_runner"], "terminal training runner hash changed")
    require(sha(FOLDS) == EXPECTED["folds"], "outer-fold mapping hash changed")
    require(sha(EXTENSION) == EXPECTED["extension"], "five-seed extension config hash changed")

    fold_frame = pd.read_csv(FOLDS)
    require(len(fold_frame) == 1500 and fold_frame.pair_id.nunique() == 1500, "fold table identity/count failure")
    require(fold_frame.groupby("outer_fold").size().to_dict() == {0: 300, 1: 300, 2: 300, 3: 300, 4: 300}, "outer-fold counts changed")

    expected_dirs = {f"{obj}_seed_{seed}_outer_{fold}" for obj in OBJECTIVES for seed in SEEDS for fold in range(5)}
    observed_dirs = {p.name for p in BASE.iterdir() if p.is_dir() and any(f"_seed_{seed}_" in p.name for seed in SEEDS)}
    require(observed_dirs == expected_dirs, f"added-seed directory grid mismatch; missing={sorted(expected_dirs-observed_dirs)}, extra={sorted(observed_dirs-expected_dirs)}")

    from turkish_ai_detection.experiments.external_cross_corpus import inference
    inference.SEEDS = (42, 137, 314, 2026, 3407)

    rows = []
    for objective in OBJECTIVES:
        for seed in SEEDS:
            for fold in range(5):
                tag = f"{objective}_seed_{seed}_outer_{fold}"
                cell = BASE / tag
                run_path = cell / "run.json"
                model_path = cell / "checkpoint/model.safetensors"
                config_path = cell / "checkpoint/config.json"
                pred_path = cell / "internal_predictions.csv"
                for path in (run_path, model_path, config_path, pred_path):
                    require(path.is_file() and path.stat().st_size > 0, f"missing/empty required file: {path}")

                authenticated = inference.verify_ce_supcon_payload(METHOD[objective], seed, fold)
                run = json.loads(run_path.read_text(encoding="utf-8"))
                meta = json.loads(config_path.read_text(encoding="utf-8"))
                ref_path = ROOT / f"results/primary/{objective}/seed_{seed}/outer_{fold}_run.json"
                ref = json.loads(ref_path.read_text(encoding="utf-8"))
                extension = run.get("five_seed_extension", {})
                require(extension.get("config_sha256") == EXPECTED["extension"], f"extension provenance absent: {tag}")
                require(extension.get("base_recipe_unchanged") is True, f"extension recipe flag absent: {tag}")
                require(extension.get("wrapper_sha256") == sha(WRAPPER), f"extension wrapper binding mismatch: {tag}")
                require(run.get("status") == meta.get("status") == "SCIENTIFIC_PROTOCOL_FAITHFUL_RERUN", f"completion status failure: {tag}")
                require(run.get("objective") == meta.get("objective") == objective, f"objective mismatch: {tag}")
                require(run.get("seed") == meta.get("seed") == seed, f"seed mismatch: {tag}")
                require(run.get("outer_fold") == meta.get("outer_fold") == fold, f"fold mismatch: {tag}")
                require(run.get("contract_sha256") == EXPECTED["base_config"], f"contract mismatch: {tag}")
                require(run.get("runner_sha256") == EXPECTED["base_runner"], f"runner mismatch: {tag}")
                require(run.get("train_pair_sha256") == ref.get("train_pair_sha256"), f"train-fold identity mismatch: {tag}")
                require(run.get("test_pair_sha256") == ref.get("test_pair_sha256"), f"test-fold identity mismatch: {tag}")
                require(ref.get("fold_sha256") == EXPECTED["folds"], f"reference fold binding mismatch: {tag}")

                with safe_open(str(model_path), framework="pt", device="cpu") as handle:
                    keys = list(handle.keys())
                    require(len(keys) > 0, f"empty/unreadable safetensors payload: {tag}")
                    first_shape = list(handle.get_slice(keys[0]).get_shape())

                pred = pd.read_csv(pred_path)
                require(len(pred) == 600 and pred.text_id.nunique() == 600, f"internal prediction shape failure: {tag}")
                require(pred.seed.eq(seed).all() and pred.outer_fold.eq(fold).all() and pred.objective.eq(objective).all(), f"internal prediction identity failure: {tag}")
                require(np.isfinite(pred.raw_decision_score.to_numpy(float)).all(), f"non-finite internal predictions: {tag}")
                rows.append({
                    "tag": tag, "method": METHOD[objective], "objective": objective, "seed": seed,
                    "outer_fold": fold, "status": "PASS_AUTHENTIC_COMPLETE_READABLE",
                    "model_path": authenticated["model_path"], "model_sha256": authenticated["model_sha256"],
                    "config_path": authenticated["metadata_path"], "config_sha256": authenticated["metadata_sha256"],
                    "run_record_path": authenticated["run_record_path"], "internal_prediction_path": authenticated["internal_prediction_path"],
                    "safetensors_key_count": len(keys), "first_tensor_shape": json.dumps(first_shape),
                })

    report = pd.DataFrame(rows).sort_values(["objective", "seed", "outer_fold"])
    require(len(report) == 20 and report.tag.nunique() == 20, "verified cell count is not 20")
    report.to_csv(OUT / "EXTERNAL_5SEED_POSTTRAIN_VERIFICATION.csv", index=False)
    summary = {
        "status": "PASS_ALL_20_TERMINAL_CHECKPOINTS",
        "verified_cells": 20,
        "seeds": list(SEEDS),
        "outer_folds": list(range(5)),
        "objectives": list(OBJECTIVES),
        "base_config_sha256": EXPECTED["base_config"],
        "base_runner_sha256": EXPECTED["base_runner"],
        "fold_sha256": EXPECTED["folds"],
        "extension_config_sha256": EXPECTED["extension"],
        "gpu_inference_performed": False,
    }
    (OUT / "EXTERNAL_5SEED_POSTTRAIN_VERIFICATION.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
