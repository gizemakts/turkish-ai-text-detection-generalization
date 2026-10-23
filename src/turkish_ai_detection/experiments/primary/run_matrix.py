# -*- coding: utf-8 -*-
"""Generate or execute the 50 prespecified primary ablation commands.

Default behavior is dry-run: commands are printed only.
Use --execute only after the identity audit and one CE + one SupCon smoke/single-fit
preflight have been reviewed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
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
RUNNER = ROOT / "src" / "turkish_ai_detection/experiments/primary/run_fit.py"
SEEDS = [42, 137, 314, 2026, 3407]
OUTERS = [0, 1, 2, 3, 4]
OBJECTIVES = ["ce", "supcon"]


def commands():
    for seed in SEEDS:
        for outer in OUTERS:
            for objective in OBJECTIVES:
                yield [
                    str(PYTHON),
                    str(RUNNER),
                    "--objective", objective,
                    "--seed", str(seed),
                    "--outer-fold", str(outer),
                ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    for index, cmd in enumerate(commands(), 1):
        print(f"[{index:02d}/50] " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
        if args.execute:
            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                check=False,
                text=True,
                capture_output=True,
            )
            if completed.returncode != 0:
                print(completed.stdout)
                print(completed.stderr, file=sys.stderr)
                raise SystemExit(completed.returncode)
            # The child runner prints only safe completion metadata; training
            # progress/outer-test metrics stay blinded until finalization.
            safe_text = completed.stdout.strip()
            if safe_text:
                print(safe_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
