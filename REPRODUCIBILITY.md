# Reproducibility

The repository provides the study's code, experimental configurations, corpus partition metadata, protocol specifications, aggregate results and numerical tests. The AI-generated ABSTRACT and NEWSPAPER texts are provided, but their paired human-written sources, trained checkpoints and per-document model predictions are not distributed; consequently, the published training and external evaluation runs cannot be reproduced from this archive alone.

## Setup

Use Python 3.13 and the dependencies in `requirements.txt` and `constraints-cuda.txt`. Model training requires a compatible GPU environment. The BERTurk encoder is `dbmdz/bert-base-turkish-cased` at revision `b6e1de16c983e0f2c70664591ea3f22810072608`.

From the repository root, after installing the required dependencies:

```bash
python -m pip install -e . --no-deps
python -m turkish_ai_detection.cli status
python -m pytest -q tests/unit -m "not private_evidence"
```

The `private_evidence` marker identifies tests that require non-redistributed run records. Public tests exercise numerical behavior and protocol consistency but do not retrain the full study. Pipeline entry points are listed in `EXPERIMENTS.md`; data sources and access conditions are in `DATA_ACCESS.md`.

## Results

The reported measurements and their corresponding aggregate source files are indexed in `RESULTS_INDEX.csv`. The accompanying article and Online Resource 1 provide the full methodology, statistical interpretation and limitations.
