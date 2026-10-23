# -*- coding: utf-8 -*-
"""Build the immutable-input, additive Turkish AI-Generated Text Detection Study Combined v2 baseline."""
from __future__ import annotations

import sys

import argparse
import json
from pathlib import Path

def _resolve_project_root() -> Path:
    for _candidate in Path(__file__).resolve().parents:
        if (_candidate / ".gitignore").is_file():
            return _candidate
    raise RuntimeError("Turkish AI-Generated Text Detection Study project root could not be resolved")

ROOT = _resolve_project_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from turkish_ai_detection.data.study import build_combined_study


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/study.json")
    args = parser.parse_args()
    result = build_combined_study(Path(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
