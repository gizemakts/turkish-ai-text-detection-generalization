from pathlib import Path

import pytest

from turkish_ai_detection.reproducibility.reproduction_policy import (
    CURRENT_MIGRATED,
    EXACT_UNAVAILABLE,
    WORKFLOWS,
    historical_exact_reproduction_status,
    expected_run_hashes,
    validate_current_migrated_reproduction,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNERS = {
    "generator_holdout": ROOT / "src/turkish_ai_detection/experiments/generator_holdout/run_fit.py",
    "domain_shift": ROOT / "src/turkish_ai_detection/experiments/domain_shift/run_fit.py",
    "artifact_sensitivity": ROOT / "src/turkish_ai_detection/experiments/artifact_sensitivity/run_fit.py",
    "literature_comparison": ROOT / "src/turkish_ai_detection/experiments/literature_comparison/run_fit.py",
    "calibration": ROOT / "src/turkish_ai_detection/experiments/calibration/run_fit.py",
    "near_duplicate_sensitivity": ROOT / "src/turkish_ai_detection/experiments/near_duplicate_sensitivity/run_fit.py",
}


@pytest.mark.private_evidence
@pytest.mark.parametrize("workflow", sorted(WORKFLOWS))
def test_exact_historical_reproduction_is_explicitly_unavailable(workflow):
    status = historical_exact_reproduction_status(ROOT, workflow)
    assert status["status"] == EXACT_UNAVAILABLE
    assert len(status["historical_config_sha256"]) == 64
    assert len(status["historical_runner_sha256"]) == 64


@pytest.mark.private_evidence
@pytest.mark.parametrize("workflow", sorted(WORKFLOWS))
def test_current_migrated_reproduction_uses_active_bytes_and_semantic_audit(workflow):
    status = validate_current_migrated_reproduction(ROOT, workflow, RUNNERS[workflow])
    assert status["reproduction_mode"] == CURRENT_MIGRATED
    assert status["historical_exact_reproduction"] == EXACT_UNAVAILABLE
    assert status["current_config_path"] == WORKFLOWS[workflow].config
    assert status["semantic_equivalence_audit"].endswith(
        "active_config_semantic_equivalence_20261006.csv"
    )


def test_current_migrated_policy_fails_closed_without_semantic_audit(tmp_path):
    with pytest.raises(RuntimeError, match="missing active config"):
        validate_current_migrated_reproduction(
            tmp_path, "generator_holdout", tmp_path / "runner.py"
        )


@pytest.mark.private_evidence
def test_run_hash_binding_distinguishes_historical_and_migrated_records():
    workflow = "generator_holdout"
    binding = WORKFLOWS[workflow]
    import json
    freeze = json.loads((ROOT / binding.freeze).read_text(encoding="utf-8"))
    policy = validate_current_migrated_reproduction(ROOT, workflow, RUNNERS[workflow])

    historical = expected_run_hashes(
        {}, freeze, policy, config_key="config", runner_key="runner"
    )
    migrated = expected_run_hashes(
        {
            "reproduction_mode": CURRENT_MIGRATED,
            "historical_exact_reproduction": EXACT_UNAVAILABLE,
        },
        freeze,
        policy,
        config_key="config",
        runner_key="runner",
    )
    assert historical == (
        freeze["files"]["config"]["sha256"],
        freeze["files"]["runner"]["sha256"],
    )
    assert migrated == (policy["current_config_sha256"], policy["current_runner_sha256"])
    assert migrated != historical
