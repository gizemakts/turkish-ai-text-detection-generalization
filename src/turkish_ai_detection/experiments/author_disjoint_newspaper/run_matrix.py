"""Dry-run by default; execute the frozen 30-run matrix only with --execute."""
from __future__ import annotations

import argparse
import subprocess
import sys

from .common import OBJECTIVES, OUTERS, ROOT, SEEDS, load_config_and_freeze


def commands(objective_filter=None):
    runner = ROOT / "src" / "turkish_ai_detection" / "experiments" / "author_disjoint_newspaper" / "run_fit.py"
    objectives = (objective_filter,) if objective_filter else OBJECTIVES
    for objective in objectives:
        for seed in SEEDS:
            for outer in OUTERS:
                yield [sys.executable, "-m", "turkish_ai_detection.experiments.author_disjoint_newspaper.run_fit",
                       "--objective", objective, "--seed", str(seed), "--outer-fold", str(outer)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--objective", choices=OBJECTIVES)
    args = parser.parse_args()
    load_config_and_freeze()
    planned = list(commands(args.objective))
    for index, command in enumerate(planned, 1):
        print(f"[{index:02d}/{len(planned):02d}] " + " ".join(command))
        if args.execute:
            completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
            if completed.returncode:
                print(completed.stdout)
                print(completed.stderr, file=sys.stderr)
                return completed.returncode
            if completed.stdout.strip():
                print(completed.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

