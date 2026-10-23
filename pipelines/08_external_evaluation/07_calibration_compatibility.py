# -*- coding: utf-8 -*-
"""Thin CLI for internal-only calibration mapping compatibility QA."""
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src/turkish_ai_detection/experiments/external_cross_corpus/calibration_compatibility.py"
if len(sys.argv) != 2 or sys.argv[1] not in {"--validate-only", "--execute"}:
    raise SystemExit("Specify --validate-only now; --execute only after frozen final reruns.")
flags = ["--validate-only"] if sys.argv[1] == "--validate-only" else []
raise SystemExit(subprocess.call([sys.executable, "-B", "-X", "utf8", str(MODULE), *flags], cwd=str(ROOT)))
