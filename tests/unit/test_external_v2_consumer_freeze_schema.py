from pathlib import Path
import json

import pytest

# Only the original private evidence satisfies these historical binding assertions.
pytestmark = pytest.mark.private_evidence

from turkish_ai_detection.experiments.external_cross_corpus.inference import (
    verify_frozen_training_runner,
)
from turkish_ai_detection.experiments.external_cross_corpus.freeze_v2 import verify_freeze


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "external_cross_corpus.json"
FREEZE = (
    ROOT
    / "provenance"
    / "experiments"
    / "external_cross_corpus_v2"
    / "PREEXECUTION_FREEZE.json"
)


def test_v2_freeze_uses_contract_identity_for_active_config():
    freeze = verify_freeze(CONFIG, FREEZE, require_execution=True)
    assert freeze["contract"]["path"] == "configs/external_cross_corpus.json"
    assert not any(
        item.get("path") == "configs/external_cross_corpus.json"
        for item in freeze["bindings"]
    )


def test_v2_freeze_exactly_binds_stage4_consumers():
    paths = {item["path"] for item in verify_freeze(CONFIG, FREEZE)["bindings"]}
    assert {
        "src/turkish_ai_detection/experiments/external_cross_corpus/inference.py",
        "src/turkish_ai_detection/experiments/external_cross_corpus/seal_predictions.py",
        "src/turkish_ai_detection/experiments/external_cross_corpus/statistics.py",
    } <= paths


def test_v2_freeze_runner_binding_is_the_training_identity_source():
    binding = verify_frozen_training_runner(CONFIG, FREEZE)
    assert binding["path"] == (
        "src/turkish_ai_detection/experiments/external_cross_corpus/rerun_fit.py"
    )
    assert len(binding["sha256"]) == 64


def test_v2_freeze_rejects_tampered_training_runner_binding(tmp_path):
    tampered = json.loads(FREEZE.read_text(encoding="utf-8"))
    for record in tampered["bindings"]:
        if record["path"].endswith("/rerun_fit.py"):
            record["sha256"] = "0" * 64
            break
    else:
        pytest.fail("training runner binding missing from production freeze")

    from turkish_ai_detection.experiments.external_cross_corpus.freeze_v2 import (
        canonical_json,
        sha256_bytes,
    )

    payload = dict(tampered)
    payload.pop("freeze_sha256")
    tampered["freeze_sha256"] = sha256_bytes(canonical_json(payload))
    candidate = tmp_path / "tampered-runner-freeze.json"
    candidate.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(RuntimeError, match="binding mismatch"):
        verify_frozen_training_runner(CONFIG, candidate)
