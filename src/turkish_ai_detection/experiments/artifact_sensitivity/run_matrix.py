# -*- coding: utf-8 -*-
"""Dry-run by default; executes the frozen 30-fit Condition-B matrix only with --execute."""
from __future__ import annotations

import sys
import argparse, json, subprocess, sys
from pathlib import Path

def _resolve_project_root() -> Path:
    for _candidate in Path(__file__).resolve().parents:
        if (_candidate / ".gitignore").is_file():
            return _candidate
    raise RuntimeError("Turkish AI-Generated Text Detection Study project root could not be resolved")

ROOT = _resolve_project_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
PYTHON = ROOT / ".venv-gpu" / "Scripts" / "python.exe"
RUNNER = ROOT / "src" / "turkish_ai_detection/experiments/artifact_sensitivity/run_fit.py"
FREEZE = ROOT / "provenance" / "experiments" / "artifact_sensitivity" / "PREEXECUTION_FREEZE.json"
SEEDS = [42, 314, 2026]
OUTERS = [0,1,2,3,4]
OBJECTIVES = ["ce", "supcon"]


def commands():
    for outer in OUTERS:
        for seed in SEEDS:
            for objective in OBJECTIVES:
                yield [
                    str(PYTHON), str(RUNNER),
                    "--objective", objective,
                    "--seed", str(seed),
                    "--outer-fold", str(outer),
                ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()

    if a.execute:
        if not FREEZE.is_file():
            raise RuntimeError("execution locked: required pre-execution freeze is missing; see docs/reproducibility.md")
        freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
        if freeze.get("status") != "FROZEN_READY_TO_RUN":
            raise RuntimeError("execution locked: invalid Condition-B freeze")

    for i, cmd in enumerate(commands(), 1):
        print(f"[{i:02d}/30] " + " ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
        if not a.execute:
            continue
        completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            raise SystemExit(completed.returncode)
        safe = None
        for line in completed.stdout.splitlines():
            if line.startswith("TURKISH_AI_DETECTION_SAFE_RESULT\t"):
                safe = line.split("\t", 1)[1]
        if safe is None:
            raise RuntimeError("child completed without safe sentinel")
        print(json.dumps(json.loads(safe), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
