from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from turkish_ai_detection.experiments.external_cross_corpus.freeze_v2 import (
    build_freeze,
    verify_freeze,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "configs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "src" / "runner.py").write_text("runner-v1", encoding="utf-8")
    (tmp_path / "data" / "folds.csv").write_text("fold\n0\n", encoding="utf-8")
    runner_hash = hashlib.sha256((tmp_path / "src" / "runner.py").read_bytes()).hexdigest()
    folds_hash = hashlib.sha256((tmp_path / "data" / "folds.csv").read_bytes()).hexdigest()
    contract = tmp_path / "configs" / "external.json"
    contract.write_text(
        json.dumps(
            {
                "status": "DRAFT_FOR_REVIEW",
                "execution_permitted": False,
                "internal_training_bindings": {
                    "folds": {"path": "data/folds.csv", "sha256": folds_hash},
                    "runner": {"path": "src/runner.py", "sha256": runner_hash},
                },
            }
        ),
        encoding="utf-8",
    )
    return contract, tmp_path / "freeze.json"


def test_candidate_freeze_is_deterministic_and_verifiable(tmp_path):
    contract, freeze_path = _fixture(tmp_path)
    first = build_freeze(contract, freeze_path, allow_draft=True)
    second_path = tmp_path / "freeze-second.json"
    second = build_freeze(contract, second_path, allow_draft=True)
    assert first["freeze_sha256"] == second["freeze_sha256"]
    assert verify_freeze(contract, freeze_path, require_execution=False)["freeze_sha256"] == first["freeze_sha256"]


def test_freeze_rejects_missing_and_tampered_bindings(tmp_path):
    contract, freeze_path = _fixture(tmp_path)
    build_freeze(contract, freeze_path, allow_draft=True)
    (tmp_path / "src" / "runner.py").write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="binding mismatch"):
        verify_freeze(contract, freeze_path, require_execution=False)


def test_freeze_rejects_missing_declared_file(tmp_path):
    contract, freeze_path = _fixture(tmp_path)
    config = json.loads(contract.read_text(encoding="utf-8"))
    config["internal_training_bindings"]["folds"]["path"] = "data/missing.txt"
    contract.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="file is missing"):
        build_freeze(contract, freeze_path, allow_draft=True)


def test_freeze_rejects_malformed_and_draft_execution(tmp_path):
    contract, freeze_path = _fixture(tmp_path)
    build_freeze(contract, freeze_path, allow_draft=True)
    with pytest.raises(RuntimeError, match="execution remains disabled"):
        verify_freeze(contract, freeze_path)
    freeze_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid V2 freeze schema"):
        verify_freeze(contract, freeze_path, require_execution=False)


def test_freeze_rejects_wrong_code_identity(tmp_path):
    contract, freeze_path = _fixture(tmp_path)
    build_freeze(contract, freeze_path, allow_draft=True)
    code = tmp_path / "src" / "runner.py"
    code.write_text("new", encoding="utf-8")
    with pytest.raises(RuntimeError, match="binding mismatch"):
        verify_freeze(contract, freeze_path, require_execution=False, required_code_path=code)
