# -*- coding: utf-8 -*-
"""Review-only checks. Does not train, infer, seal, or compute metrics."""
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src/turkish_ai_detection/experiments/external_cross_corpus"
TASKS = [
    ("audit_semantic_equivalence.py", []),
    ("inference.py", ["--validate-only"]),
    ("seal_predictions.py", ["--validate-only"]),
    ("statistics.py", ["--validate-only"]),
    ("secondary_audits.py", ["--validate-only"]),
    ("calibration_compatibility.py", ["--validate-only"]),
    ("conditional_metrics.py", ["--validate-only"]),
]
for script, flags in TASKS:
    print(f"CHECK {script}", flush=True)
    cmd = [sys.executable, "-B", "-X", "utf8", str(PACKAGE / script), *flags]
    code = subprocess.call(cmd, cwd=str(ROOT))
    if code:
        raise SystemExit(code)
print("PASS_EXTERNAL_REVIEW_STATIC_CHECKS")
