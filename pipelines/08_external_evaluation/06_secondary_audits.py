# -*- coding: utf-8 -*-
"""Explicit CLI entrypoint for prespecified post-seal descriptive audits."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / 'src/turkish_ai_detection/experiments/external_cross_corpus/secondary_audits.py'
runpy.run_path(str(TARGET), run_name='__main__')
