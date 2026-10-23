# NEWSPAPER surface-probe ablation

Completed: `2026-10-14T15:07:49.520105+03:00`.

Primary estimates use five established author-disjoint folds. Each held-out fold contains two newspaper authors; no held-out newspaper author occurs in training. ABSTRACT texts and the other eight newspaper authors form each training set. Standardization and logistic-regression fitting occur inside each training fold.

The original full-surface pair-disjoint result was exactly reproduced at ROC-AUC `0.998772`. The author-disjoint baseline uses the same original surface features under the primary author-disjoint split.

| Probe | ROC-AUC | 95% ten-author cluster interval |
|---|---:|---:|
| BASELINE_EXISTING_FULL_SURFACE_AUTHOR_DISJOINT | 0.997424 | [0.994596, 1.000000] |
| P1_CUE_ONLY | 0.884568 | [0.790288, 0.956378] |
| P2_EXISTING_MINUS_CUE | 0.997004 | [0.993608, 1.000000] |
| P3_FULL | 0.998604 | [0.996416, 0.999944] |

P2 removes only `year_date_per_100_words`, the sole original feature directly duplicated by the frozen temporal/cue family. It retains `digit_token_fraction` because that broader feature is not a direct duplicate.

Interpretation: high P1 means the cue family itself carries label information; high P2 means other surface features retain strong separability; P3 > P2 indicates incremental information from the cue family. P3 approximately equal to P2 does not establish independence or non-use because correlated or redundant signals may remain.
