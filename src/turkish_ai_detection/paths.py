"""Repository-relative paths for Turkish AI-Generated Text Detection Study.

Importing this module is read-only. Set ``TURKISH_AI_DETECTION_PROJECT_ROOT`` only when
running from outside the checkout.
"""
from __future__ import annotations

import os
from pathlib import Path


def _looks_like_root(path: Path) -> bool:
    """Recognize the checkout independently of optional/private datasets."""
    return (path / "README.md").is_file() and all(
        (path / name).is_dir() for name in ("src", "configs", "results")
    )


def _resolve_root() -> Path:
    configured = os.environ.get("TURKISH_AI_DETECTION_PROJECT_ROOT")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if not _looks_like_root(candidate):
            raise RuntimeError(f"TURKISH_AI_DETECTION_PROJECT_ROOT is not a Turkish AI-Generated Text Detection Study checkout: {candidate}")
        return candidate

    cwd = Path.cwd().resolve()
    for candidate in (Path(__file__).resolve().parents[2], cwd, *cwd.parents):
        if _looks_like_root(candidate):
            return candidate
    raise RuntimeError("Turkish AI-Generated Text Detection Study repository root not found. See REPRODUCIBILITY.md.")


PROJECT_ROOT = _resolve_root()
DATA_DIR = PROJECT_ROOT / "data"
BASE = DATA_DIR
ABSTRACT_DIR = DATA_DIR / "abstract"
RAW = ABSTRACT_DIR / "RAW"
P2 = ABSTRACT_DIR / "PROCESSED_V2"
R2 = ABSTRACT_DIR / "RESULTS_V2"
M2DIR = ABSTRACT_DIR / "MODELS_V2"
V2 = PROJECT_ROOT / "provenance" / "datasets" / "abstract"
PROTOCOL_DIR = PROJECT_ROOT / "provenance" / "protocols"
CODE_DIR = PROJECT_ROOT / "src" / "turkish_ai_detection"
RESOURCE_DIR = PROJECT_ROOT / "resources"
PIPELINE_SCRIPT_DIR = PROJECT_ROOT / "pipelines"
NOTEBOOK_DIR = PROJECT_ROOT / "provenance" / "history" / "notebooks"
OUTPUT_DIR = PROJECT_ROOT / "results"
CONFIG_DIR = PROJECT_ROOT / "configs"
TEST_DIR = PROJECT_ROOT / "tests"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
FINAL_MODEL_DIR = ARTIFACT_DIR / "models" / "berturk" / "final"

os.environ.setdefault("TURKISH_AI_DETECTION_PROJECT_ROOT", str(PROJECT_ROOT))
os.environ.setdefault("TURKISH_AI_DETECTION_BASE", str(DATA_DIR))
os.environ.setdefault("TURKISH_AI_DETECTION_CODE_DIR", str(CODE_DIR))
os.environ.setdefault("TURKISH_AI_DETECTION_RESOURCE_DIR", str(RESOURCE_DIR))
