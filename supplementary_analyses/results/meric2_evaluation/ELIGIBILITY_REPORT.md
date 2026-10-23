# Meric2 evaluation dataset and protocol

The Meric2 dataset contains 11,600 records. One AI text was blank and 24 further rows were within-class normalized duplicates. The analysis uses the remaining 11,575 records (5,776 human, 5,799 AI). No exact normalized-text overlap was observed with Combined-v2, Renklier2026, TUM2024_TR or Bostancioglu2026_TR.

The evaluation uses CE and SupCon checkpoints trained on Combined-v2 only, restricted to seeds 42, 314 and 2026 and five outer folds per seed. The TF-IDF M4 baseline uses five fold-specific estimators. A separate earlier Meric2-augmented CE pilot is excluded; Meric2 is not used to fit or select the reported models, calibrators or thresholds.

Integer word-count matching yields 3,969 observations per class. These matched observations are artificial length-control groups, not matched source texts. All reported Meric2 results are exploratory: the earlier pilot result was known before this stress test, and the corpus does not provide author-level or generation-prompt lineage for paired-source inference.
