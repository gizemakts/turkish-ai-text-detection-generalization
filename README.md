# Turkish AI-Generated Text Detection: Training Objectives and Cross-Corpus Generalization

Research code and aggregate results accompanying **Beyond Near-Perfect Within-Corpus Discrimination: Training Objectives and Cross-Corpus Generalization in Turkish AI-Generated Text Detection**.

This study compares matched BERTurk cross-entropy (CE) and CE + 0.1 supervised contrastive learning (SupCon) training objectives. Experiments address pair-disjoint evaluation, generator holdout, domain transfer, author-disjoint and corpus-construction sensitivity, calibration, and transfer across three external Turkish corpora.

## Experimental design

| Component | Specification |
|---|---|
| Combined-v2 | 1,500 human-AI pairs (1,000 ABSTRACT; 500 NEWSPAPER) |
| Encoder | `dbmdz/bert-base-turkish-cased`, fixed revision in the study configuration |
| Training objectives | CE; CE + 0.1 SupCon (temperature 0.07) |
| Main experiment | Five seeds, five pair-disjoint outer folds, 25 fits per objective |
| Other experiments | Generator holdout, bidirectional domain transfer, artifact sensitivity, near-duplicate and author-disjoint evaluation, calibration and external transfer |
| External datasets | Renklier2026, TUM2024_TR, Bostancioglu2026_TR |
| Exploratory baselines | Adapted LIT2026, TF-IDF M4 and Meric2 |

Mean internal ROC-AUC was 0.999926 for CE and 0.999784 for SupCon. Performance decreased on some external corpora; transferred operating thresholds were not consistently reliable. The article and its supplementary material provide the statistical analysis.

## Repository layout

| Path | Contents |
|---|---|
| `src/turkish_ai_detection/` | Dataset construction, feature extraction, model training, evaluation and statistical analyses |
| `pipelines/` | Workflow entry points |
| `scripts/` | Additional experiment and analysis utilities |
| `configs/` | Model and experiment specifications |
| `data/` | AI-generated ABSTRACT/NEWSPAPER texts, text-free fold and cluster assignments, and source metadata |
| `results/` | Aggregate numerical results and supplementary analyses |
| `supplementary_analyses/` | Additional baseline and sensitivity analyses |
| `provenance/` | Recorded experimental protocols and dataset provenance |
| `tests/` | Unit and protocol tests |

`RESULTS.md`, `RESULTS_INDEX.csv` and `docs/PUBLICATION_CROSSWALK.md` map the included outputs to the article and Online Resource 1. Generation model labels are documented in `provenance/protocols/NEWSPAPER_GENERATION_MODELS.md` and are also present in the source dataset's `model` column.

## Installation and tests

Python 3.13 is recommended. Install dependencies described in `ENVIRONMENT.md` before running:

```bash
python -m pip install -e . --no-deps
python -m turkish_ai_detection.cli status
python -m pytest -q tests/unit -m "not private_evidence"
```

The repository uses a source-checkout layout. Model training requires the original datasets, pretrained model assets and compatible hardware; the archive does not contain the original human-written source texts, trained checkpoints or per-document predictions. See `DATA_ACCESS.md` and `REPRODUCIBILITY.md` for those requirements and the scope of the public tests.

## Citation and license

Citation information is provided in `CITATION.cff`. The MIT License applies to author-owned code only and does not extend to third-party datasets or model weights.
