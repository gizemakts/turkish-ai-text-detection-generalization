# -*- coding: utf-8 -*-
"""Thin runner for post-seal conditional metrics; no automatic execution."""
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src/turkish_ai_detection/experiments/external_cross_corpus/conditional_metrics.py"
if len(sys.argv) != 2 or sys.argv[1] not in {"--validate-only", "--execute"}:
    raise SystemExit("Specify --validate-only now; --execute only after prediction seal.")
flags = ["--validate-only"] if sys.argv[1] == "--validate-only" else []
raise SystemExit(subprocess.call([sys.executable, "-B", "-X", "utf8", str(MODULE), *flags], cwd=str(ROOT)))
