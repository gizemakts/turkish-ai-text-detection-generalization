# -*- coding: utf-8 -*-
"""Dry-run or execute the 200 prespecified RQ4 inner fits."""
from __future__ import annotations
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
PYTHON = Path(sys.executable)
RUNNER = ROOT / "src" / "turkish_ai_detection" / "experiments" / "calibration" / "run_fit.py"
FREEZE = ROOT / "provenance" / "experiments" / "calibration" / "PREEXECUTION_FREEZE.json"
SEEDS = [42, 137, 314, 2026, 3407]
OUTERS = [0, 1, 2, 3, 4]
INNERS = [0, 1, 2, 3]
OBJECTIVES = ["ce", "supcon"]


def commands():
    for seed in SEEDS:
        for outer in OUTERS:
            for objective in OBJECTIVES:
                for inner in INNERS:
                    yield [str(PYTHON), str(RUNNER), "--objective", objective,
                           "--seed", str(seed), "--outer-fold", str(outer),
                           "--inner-fold", str(inner)]


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--execute", action="store_true"); args = p.parse_args()
    if args.execute:
        if not FREEZE.is_file():
            raise RuntimeError("execution locked: required pre-execution freeze is missing; see docs/reproducibility.md")
        freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
        if freeze.get("status") != "FROZEN_READY_TO_RUN":
            raise RuntimeError("execution locked: invalid RQ4 freeze")
    cmds = list(commands())
    if len(cmds) != 200:
        raise RuntimeError("RQ4 matrix cardinality is not 200")
    for i, cmd in enumerate(cmds, 1):
        print(f"[{i:03d}/200] " + " ".join(f'\"{x}\"' if " " in x else x for x in cmd))
        if args.execute:
            done = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            if done.returncode != 0:
                print(done.stdout); print(done.stderr, file=sys.stderr); return done.returncode
            if done.stdout.strip(): print(done.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
