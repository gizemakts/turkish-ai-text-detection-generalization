"""Read-only package and optional local-resource checks."""
from __future__ import annotations

import importlib
import importlib.metadata as metadata
from pathlib import Path

from turkish_ai_detection.paths import PROJECT_ROOT, RESOURCE_DIR

REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "transformers": "transformers",
    "torch": "torch",
}


def validate_package() -> list[str]:
    failures: list[str] = []
    for distribution, module in REQUIRED_IMPORTS.items():
        try:
            metadata.version(distribution)
            importlib.import_module(module)
        except Exception as exc:
            failures.append(f"dependency {distribution}: {type(exc).__name__}: {exc}")
    for relative in (
        Path("configs/study.json"),
        Path("src/turkish_ai_detection/paths.py"),
        Path("resources/turkish_stopwords_v1.txt"),
        Path("resources/conjunctions_primary_v1.txt"),
    ):
        if not (PROJECT_ROOT / relative).is_file():
            failures.append(f"missing package file: {relative.as_posix()}")
    if not RESOURCE_DIR.is_dir():
        failures.append("missing resources directory")
    return failures


def data_availability_messages() -> list[str]:
    messages: list[str] = []
    study = PROJECT_ROOT / "data" / "study" / "paired_study.csv"
    if not study.is_file():
        messages.append("The paired human-written inputs are required for training; see DATA_ACCESS.md.")
    weights = list((PROJECT_ROOT / "artifacts").rglob("model.safetensors"))
    if not weights:
        messages.append("Pretrained or trained weights are needed for model inference; see REPRODUCIBILITY.md.")
    return messages


def require_study_data() -> tuple[bool, str]:
    """Check data availability separately from repository identification."""
    study = PROJECT_ROOT / "data" / "study" / "paired_study.csv"
    if study.is_file():
        return True, ""
    return False, "The paired human-written inputs are required for training; see DATA_ACCESS.md."
