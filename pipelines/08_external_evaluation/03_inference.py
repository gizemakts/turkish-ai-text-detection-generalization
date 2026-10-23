# -*- coding: utf-8 -*-
"""Thin CLI entrypoint; scientific implementation lives in src/."""
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if len(sys.argv) == 1:
    raise SystemExit("Explicit CLI arguments required; no automatic execution.")
CODE = ROOT / "src/turkish_ai_detection/experiments/external_cross_corpus" / 'inference.py'
raise SystemExit(subprocess.call([sys.executable, "-B", "-X", "utf8", str(CODE), *sys.argv[1:]], cwd=str(ROOT)))
