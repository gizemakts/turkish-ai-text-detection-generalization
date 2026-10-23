# Experiment map

The article reports Tables 1-5 and Figures 1-2, with Tables S1-S27 and Figures S1-S13 in Online Resource 1. Source-to-publication mappings are in `RESULTS.md` and `RESULTS_INDEX.csv`.

| Experiment | Main implementation | Configuration | Included evidence | Status/qualification |
|---|---|---|---|---|
| Five-seed × five-fold CE/SupCon | `src/turkish_ai_detection/experiments/primary/` | `configs/primary_evaluation.json` | `results/primary/final_statistics/` | Primary matched comparison; CE vs CE + 0.1 SupCon. |
| Adapted LIT2026 | `src/turkish_ai_detection/experiments/literature_comparison/` | `configs/literature_comparison.json` | `results/literature_comparison/final_statistics/` | Adapted comparator, not an exact reproduction of the source paper. |
| Generator holdout | `src/turkish_ai_detection/experiments/generator_holdout/` | `configs/generator_holdout.json` | `results/generator_holdout/final_statistics/` | Three seeds, five generator families, CE/SupCon. |
| Bidirectional domain transfer | `src/turkish_ai_detection/experiments/domain_shift/` | `configs/domain_shift.json` | `results/domain_shift/final_statistics/` | Post-unsealing stress test. |
| Condition B sensitivity | `src/turkish_ai_detection/experiments/artifact_sensitivity/` | `configs/artifact_sensitivity.json` | `results/artifact_sensitivity/final_statistics/` | Artifact-reduced, not artifact-free. |
| Near-duplicate-aware evaluation | `src/turkish_ai_detection/experiments/near_duplicate_sensitivity/` | `configs/near_duplicate_sensitivity.json` | cluster/fold metadata and aggregate statistics | Five seeds/folds; cluster bootstrap. |
| Author-disjoint evaluation | `src/turkish_ai_detection/experiments/author_disjoint_newspaper/` | `results/author_disjoint_newspaper/FROZEN_EXPERIMENT_CONFIG.json` | aggregate, non-author-identifying statistics | Post-hoc sensitivity; per-author tables excluded. |
| Internal calibration | `src/turkish_ai_detection/experiments/calibration/` | `configs/calibration.json` | aggregate calibration/threshold results | Inner-OOF Platt and NP-inspired thresholds. |
| External cross-corpus | `src/turkish_ai_detection/experiments/external_cross_corpus/` | `configs/external_cross_corpus.json` | aggregate external metrics and five-seed extension | Provider datasets and checkpoints excluded. |
| Calibration/threshold transfer | external-cross-corpus modules | `configs/posthoc_terminal_calibration.json` | post-hoc calibration and compatibility summaries | Transfer reliability is limited; not a deployment calibration. |
| TF-IDF M4 external baseline | `supplementary_analyses/scripts/external_tfidf_baseline.py` | `supplementary_analyses/configs/tfidf_external.json` | tuning/fold/summary tables | Exploratory; no serialized estimator included. |
| Meric2 stress test | `supplementary_analyses/results/meric2_evaluation/code/` | `supplementary_analyses/results/meric2_evaluation/configs/analysis_config.json` | aggregate metrics and validation summaries | Post-hoc exploratory; dataset and per-text predictions excluded. |
| Surface/length/truncation/error probes | supplementary length-control analysis | embedded/frozen analysis settings | selected aggregate tables | Exploratory diagnostics. |
| Named-entity/temporal cues | `results/robustness_analyses/code/` | recorded NER model specification | aggregate cue/surface results | Per-text feature tables excluded. |
| External seed-level tests | `results/robustness_analyses/EXTERNAL_CE_SUPCON_SEED_LEVEL.csv` | five-seed primary external summaries | sign-flip, paired t-test and Holm analysis | Seed-level inference remains limited by five seeds. |



