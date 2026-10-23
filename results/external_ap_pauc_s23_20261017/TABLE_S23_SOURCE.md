# Supplementary Table S23 - source data

Panel A presents the TF-IDF M4 external fold measurements and inner cross-validation selections in `Table_S23_Panel_A_M4_fold_metrics_preserved.csv` and `Table_S23_Panel_A_M4_inner_selection_preserved.csv`. In each outer fold, M4 selected `C=10` using only outer-training observations and four-fold inner cross-validation for standardized pAUC at FPR <= 0.03.

## Panel B. Exploratory external AP and low-FPR discrimination

| Corpus | AI prevalence | CE AP | SupCon AP | M4 AP | CE standardized pAUC | SupCon standardized pAUC | M4 standardized pAUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Renklier2026 | 0.749791 | 0.902330 | 0.916499 | 0.927300 | 0.573212 | 0.570727 | 0.583027 |
| TUM2024_TR | 0.500000 | 0.709370 | 0.714198 | 0.711711 | 0.599147 | 0.599784 | 0.553681 |
| Bostancioglu2026_TR | 0.497423 | 0.756910 | 0.844769 | 0.916771 | 0.587476 | 0.612666 | 0.718833 |

**Caption.** Supplementary Table S23. Panel A reports fold-specific external results and inner model selection for the TF-IDF M4 baseline. Panel B compares post-hoc exploratory average precision and standardized partial ROC-AUC (FPR <= 0.03) for CE_ONLY_MATCHED, CE + 0.1 × SupCon, and TF-IDF M4 on the same three external evaluation populations.

**Note.** AI was the positive class. Metrics were calculated separately for each realization without pooling scores. CE and SupCon values summarize 25 realizations per corpus by first averaging five outer folds within each seed and then averaging the five seed means. M4 values average five dependent outer-fold estimators; those estimators must not be interpreted as five independent seeds. Standardized pAUC was computed with `roc_auc_score(y_true, y_score, max_fpr=0.03)`, for which random ranking equals 0.5. AP must be interpreted relative to corpus-specific AI prevalence. Evaluation on a common population supports descriptive comparison but does not make the different training-realization structures inferentially equivalent. No confidence intervals or significance tests were computed.
