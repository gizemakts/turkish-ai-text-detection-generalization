from __future__ import annotations

import json
from pathlib import Path

import pytest

# Only the original private evidence satisfies these historical binding assertions.
pytestmark = pytest.mark.private_evidence

from turkish_ai_detection.experiments.external_cross_corpus.calibration_compatibility import (
    verify_v2_runtime_freeze,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "external_cross_corpus.json"
FREEZE = (
    ROOT
    / "provenance"
    / "experiments"
    / "external_cross_corpus_v2"
    / "PREEXECUTION_FREEZE.json"
)
RUNNER = (
    ROOT
    / "src"
    / "turkish_ai_detection"
    / "experiments"
    / "external_cross_corpus"
    / "rerun_fit.py"
)


def test_v2_freeze_accepts_scientific_config_identity_and_runner_binding():
    freeze, sources = verify_v2_runtime_freeze(CONFIG, FREEZE, RUNNER)
    assert freeze["contract"]["path"] == "configs/external_cross_corpus.json"
    assert sources["runner"]["path"].endswith(
        "src/turkish_ai_detection/experiments/external_cross_corpus/rerun_fit.py"
    )


def test_v2_freeze_rejects_tampered_bound_file_digest(tmp_path):
    tampered = json.loads(FREEZE.read_text(encoding="utf-8"))
    for record in tampered["bindings"]:
        if record["path"] == "src/turkish_ai_detection/experiments/external_cross_corpus/rerun_fit.py":
            record["sha256"] = "0" * 64
            break
    else:
        pytest.fail("runner binding missing from production freeze")
    tampered["freeze_sha256"] = None
    payload = dict(tampered)
    payload.pop("freeze_sha256")
    from turkish_ai_detection.experiments.external_cross_corpus.freeze_v2 import canonical_json, sha256_bytes
    tampered["freeze_sha256"] = sha256_bytes(canonical_json(payload))
    candidate = tmp_path / "tampered-freeze.json"
    candidate.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(RuntimeError, match="binding mismatch"):
        verify_v2_runtime_freeze(CONFIG, candidate, RUNNER)
