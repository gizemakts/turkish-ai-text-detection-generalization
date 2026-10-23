# Included results

## Publication numbering

The final article contains Tables 1-5 and Figures 1-2. Online Resource 1 contains Tables S1-S27 and Figures S1-S13. The external dataset description is S25, adapted LIT2026 comparison is S26, and the calibration-transfer summary is S27. The study workflow is Figure S12; paired domain/external effects are Figure S13.

The five-seed external analysis underlying Table S20 is in `results/robustness_analyses/`. The article and Online Resource 1 give the reported tables and interpretation.


This repository includes aggregate and text-free results. Human-written source texts, per-text scores, derived features and external dataset payloads are not distributed; the AI-generated ABSTRACT and NEWSPAPER texts are provided under `data/`. See `RESULTS_INDEX.csv` for file-level mappings.

| Result package | Original source | Method / design | Seed and fold structure | Metrics / aggregation | Publication mapping |
|---|---|---|---|---|---|
| Primary | `results/primary/final_statistics/` | Matched CE and SupCon | 5 seeds × 5 outer folds; folds pooled within seed | ROC-AUC, AP, standardized pAUC; paired seed deltas; pair bootstrap | Table 2. |
| LIT2026 | `results/literature_comparison/final_statistics/` | Adapted comparator on the same evaluation design | 3 seeds × 5 folds | Same-data discrimination and paired bootstrap summaries | Supplementary Table S26. |
| Generator holdout | `results/generator_holdout/final_statistics/` | Leave-one-generator-family-out CE/SupCon | 3 seeds × 5 held-out families | Per-family and macro metrics; paired pair-bootstrap | Supplementary Table S13. |
| Domain transfer | `results/domain_shift/final_statistics/` | ABSTRACT→NEWSPAPER and reverse | 3 seeds per direction | Direction/seed ROC-AUC and paired bootstrap | Table 3. |
| Condition B | `results/artifact_sensitivity/final_statistics/` | Condition A vs artifact-reduced B | 3 seeds × 5 folds × 2 objectives | Per-seed metrics and paired bootstrap contrasts | Supplementary Table S14. |
| Near-duplicate | `results/near_duplicate_sensitivity/final_statistics/` | Cluster-aware folds | 5 seeds × 5 folds | Seed summaries; cluster bootstrap | Supplementary Table S15. |
| Internal calibration | `results/calibration/final_statistics/` | Inner-OOF Platt and frozen thresholds | 5 seeds × 5 outer folds with inner support fits | Brier, ECE, log loss, FPR/TPR and bootstrap summaries | Supplementary Table S18 (plus S1-S2). |
| External cross-corpus | `results/external_cross_corpus/statistics/` | CE, SupCon, adapted LIT2026 | CE/SupCon realization grid; common 3-seed comparison with LIT | ROC-AUC and matched differences; seed means | Table 4; Figure 1; Supplementary Table S20. |
| External AP/pAUC | `results/external_ap_pauc_s23_20261017/` | CE, SupCon and M4 in the article; additional exploratory LIT2026 evidence, where documented | Method-specific seed/fold grids | AP, standardized pAUC, ROC-AUC reconciliation | Supplementary Table S23 (Panels A-B). |
| External calibration transfer | `results/posthoc_terminal_calibration/` and transfer validation | Inner-OOF maps applied to terminal checkpoints | 3 seeds × 5 folds × CE/SupCon | Brier/ECE/log loss, reliability, threshold transfer, compatibility | Supplementary Tables S1-S7, S22, S27; Figure 2 and Figures S1-S7. |
| Author-disjoint | `results/author_disjoint_newspaper/final_statistics/` | Held-out NEWSPAPER authors | 3 seeds × 5 author folds × CE/SupCon | Fold/seed ROC-AUC; pair and author-cluster bootstrap | Supplementary; author-identifying tables excluded. |
| M4 external | `supplementary_analyses/results/tfidf_external/` | TF-IDF + logistic regression | 5 fold-specific fits with inner tuning | Fold and corpus summaries | Exploratory baseline. |
| Meric2 | `supplementary_analyses/results/meric2_evaluation/` | Fixed CE/SupCon grid plus M4 | 3 seeds × 5 folds for transformers | ROC-AUC, AP, standardized pAUC; length strata | Post-hoc exploratory stress test. |
| Length control | `supplementary_analyses/results/length_control/` | Word-count-only and length-matched analysis | 5 seeds × 5 folds for CE/SupCon | ROC-AUC and matched/unrestricted contrasts | Supplementary exploratory diagnostics. |
| Surface, truncation, error, cue diagnostics | selected supplementary/robustness tables | Saved-score diagnostic analyses | As recorded per file | Aggregate probe, subgroup, bootstrap and cue metrics | Supplementary Tables S8-S12 and S20; Figures S8-S11, according to the individual artifact. |

Bootstrap draw files are retained where they are already aggregate resampling outputs without text or row-level source content. Their intervals are generally conditional on fixed model realizations and do not represent full retraining or future-corpus uncertainty.

