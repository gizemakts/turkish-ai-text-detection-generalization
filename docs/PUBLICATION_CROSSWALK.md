# Publication results crosswalk

This index links the article and Online Resource 1 to their aggregate result files. Filenames are those in this archive. The published tables and figures remain the authoritative formatted presentation; internal intermediate records are not substituted for published results.

| Published item | Corresponding archived source | Checkable reference |
|---|---|---|
| Table 1 (dataset composition) | `configs/study.json`; text-free partitions under `data/study/` | 1,500 HUMAN-AI pairs; 1,000 ABSTRACT and 500 NEWSPAPER |
| Table 2 (matched objectives) | `results/primary/final_statistics/` | CE ROC-AUC 0.999926; SupCon ROC-AUC 0.999784 |
| Table 3 (bidirectional transfer) | `results/domain_shift/final_statistics/` | ABSTRACT to NEWSPAPER: 0.9604 / 0.9418; reverse: 0.9033 / 0.9170 (CE / SupCon) |
| Table 4, Panels A-B; Figure 1 | `results/robustness_analyses/EXTERNAL_CE_SUPCON_5SEED_SUMMARY.csv`; `EXTERNAL_CE_SUPCON_5SEED_BOOTSTRAP.csv`; `EXTERNAL_CE_SUPCON_SEED_LEVEL.csv` in the same folder | Five-seed CE/SupCon external ROC-AUC and conditional bootstrap contrasts |
| Table 4, Panel C | `results/external_cross_corpus/statistics/` | Common three-seed adapted LIT2026 comparison |
| Table 5; Table S23 | `supplementary_analyses/results/tfidf_external/tfidf_external_summary.csv`; `results/external_ap_pauc_s23_20261017/` | Exploratory lexical M4 and method-specific AP/pAUC summaries |
| Figure 2; Tables S3-S6 and S27 | `results/posthoc_terminal_calibration/` and associated external calibration transfer outputs | Frozen internal-threshold transfer; nominal FPR 0.03 |
| Tables S11-S12; Figure S11 | `results/robustness_analyses/NEWSPAPER_CUE_CLASS_COMPARISON.csv`; `NEWSPAPER_SURFACE_PROBE_ABLATION.csv`; `surface_probe/NEWSPAPER_SURFACE_PROBE_CONTRASTS.csv` | Newspaper cues and author-disjoint surface-probe analyses |
| Table S20 | `results/robustness_analyses/EXTERNAL_CE_SUPCON_SEED_LEVEL.csv` and the five-seed statistical outputs | Five-seed paired differences and seed-level inference, *not* the archived three-seed exploratory report |
| Table S24 | `supplementary_analyses/results/meric2_evaluation/` | Exploratory Meric2 analysis, not an additional confirmatory corpus |
| Table S26 | `results/literature_comparison/final_statistics/` | Adapted LIT2026 internal comparison |

The complete file-level index is `../RESULTS_INDEX.csv`. Its `mapping_status` is `PUBLICATION_LEVEL_MAPPED` for entries linked to a final table, figure or its methodological support, and `HISTORICAL_NOT_FINAL_FIVE_SEED` for the older three-seed exploratory records. Linking an artifact to a publication item does not imply that every field in an intermediate file is reproduced in that item.

Inference must preserve the different units of uncertainty: pair-based or cluster-based resampling assesses evaluation-set variation conditional on fixed models, whereas seed-level comparisons assess variation among the selected trained realizations. The code archive alone does not establish robustness to new corpora.
