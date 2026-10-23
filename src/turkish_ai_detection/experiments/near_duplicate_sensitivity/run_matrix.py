# -*- coding: utf-8 -*-
"""Dry-run or execute the 50 prespecified RQ5 cluster-aware fits."""
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
PYTHON=Path(sys.executable); RUNNER=ROOT/'src'/'turkish_ai_detection/experiments/near_duplicate_sensitivity/run_fit.py'
FREEZE=ROOT/'provenance'/'experiments'/'near_duplicate_sensitivity'/'SPLIT_PREEXECUTION_FREEZE.json'
SEEDS=[42,137,314,2026,3407]; OUTERS=[0,1,2,3,4]; OBJECTIVES=['ce','supcon']
def commands():
    for seed in SEEDS:
        for outer in OUTERS:
            for objective in OBJECTIVES:
                yield [str(PYTHON),str(RUNNER),'--objective',objective,'--seed',str(seed),'--outer-fold',str(outer)]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--execute',action='store_true'); a=p.parse_args(); cmds=list(commands())
    if len(cmds)!=50: raise RuntimeError('RQ5 matrix cardinality != 50')
    if a.execute:
        if not FREEZE.is_file(): raise RuntimeError('execution locked: required pre-execution freeze is missing; see docs/reproducibility.md')
        fr=json.loads(FREEZE.read_text(encoding='utf-8'))
        if fr.get('status')!='FROZEN_READY_TO_RUN': raise RuntimeError('invalid RQ5 scientific freeze')
    for i,cmd in enumerate(cmds,1):
        print(f'[{i:02d}/50] '+' '.join(f'"{x}"' if ' ' in x else x for x in cmd),flush=True)
        if a.execute:
            done=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,check=False)
            if done.returncode!=0:
                if done.stdout: print(done.stdout)
                if done.stderr: print(done.stderr,file=sys.stderr)
                return done.returncode
            if done.stdout.strip(): print(done.stdout.strip(),flush=True)
    return 0
if __name__=='__main__': raise SystemExit(main())
