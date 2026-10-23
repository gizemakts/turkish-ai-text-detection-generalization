# NEWSPAPER Author-Disjoint Sensitivity — Outcome-Blind Protocol Freeze

**Freeze date:** 2026-10-13  
**Status:** `FROZEN_NOT_EXECUTED`  
**Label:** post-hoc, outcome-blind frozen sensitivity analysis  
**Not:** preregistered or confirmatory

## Authority and purpose

The sole data authority is `data/study/paired_study.csv`, SHA-256 `ea5797ac19348957472626d1a23b740de531cd19577f613e1a73894e34d66b44`. The analysis tests whether the high internal discrimination observed under pair-disjoint Combined-v2 folds remains when NEWSPAPER authors are also disjoint. No performance result was inspected or produced while defining this freeze.

## Deterministic assignment

For each NEWSPAPER pair, author identity is taken from its HUMAN row; the matched AI row inherits that author via `pair_id`. The original HUMAN author labels were NFC-normalized and sorted by UTF-8 bytes to establish the fixed fold assignment. The codes below replace the source-author names in the public documentation; sorting the public codes is not the original assignment procedure. No label, prediction, score, metric, or generator value participates in assignment.

| Fold | Held-out authors | Test pairs | Training pairs |
|---:|---|---:|---:|
| 0 | AUTHOR_03; AUTHOR_07 | 100 | 1,400 |
| 1 | AUTHOR_08; AUTHOR_01 | 100 | 1,400 |
| 2 | AUTHOR_04; AUTHOR_02 | 100 | 1,400 |
| 3 | AUTHOR_05; AUTHOR_09 | 100 | 1,400 |
| 4 | AUTHOR_06; AUTHOR_10 | 100 | 1,400 |

Each training population contains every ABSTRACT pair (1,000) and the 400 NEWSPAPER pairs belonging to the other eight authors. Each test population contains all 100 HUMAN–AI pairs belonging to the two held-out authors. Pair members may never be split. Author leakage and pair leakage must both be zero.

Every author has 50 pairs: ten AI counterparts from each of Claude, Gemini, GPT, Kimi, and Qwen. Consequently every held-out fold has 20 AI examples from each generator family. Additional generator stratification is neither needed nor permitted.

## Frozen methods and outcome

The main outcome is pooled NEWSPAPER author-disjoint ROC-AUC. Fold-level ROC-AUC is reported for all five held-out author blocks. Seed-level values pool the five held-out folds; method-level mean and sample SD summarize seeds 42, 314, and 2026.

The main methods are `CE_ONLY_MATCHED` and `SUPCON` (CE + 0.1 × SupCon, temperature 0.07). There are 5 folds × 3 seeds × 2 methods = 30 planned training runs. The backbone/revision, tokenizer, head-and-tail truncation, architecture, optimizer, learning rate, weight decay, epochs, batching, true-32 objective construction, and loss semantics are inherited without alteration from the corrected authoritative primary implementation.

The paired effect is `SUPCON − CE_ONLY_MATCHED`. The frozen percentile bootstrap uses 5,000 replicates, seed 2602, `pair_id` as the resampling unit, and the identical resample for both methods and all seeds.

## Interpretation frozen before outcomes

If discrimination remains high: the same author appearing in training and test does not appear necessary for high discrimination; temporal, prompt/generation-procedure, and other corpus-construction confounding remain.

If discrimination drops materially: source/author dependence will be reported as a contributing factor to internal discrimination, and the scope of the primary pair-disjoint interpretation will be narrowed accordingly.

The result is reported in either direction. Fold assignment, seeds, analysis, and interpretation rules may not be changed in response to outcomes.

## M4 diagnostic

M4 is separate from the main comparison and is labeled `diagnostic corpus-separability baseline`. It uses word (1,2) and `char_wb` (2,5) TF-IDF with `min_df=2`, sublinear TF, L2 logistic regression, and the existing C grid. TF-IDF is fitted only on each training population; held-out author text is transform-only. Inner C selection uses deterministic pair-grouped training-only splits. M4 is not evidence about the CE/SupCon objective contrast.

## Execution lock

The manual PowerShell launcher performs no action without `-Execute`. Before execution it checks CUDA and the frozen design. Completed runs are skipped only after their bindings and prediction hashes pass; incomplete or mismatched states fail closed. Silent overwrite is forbidden. Scientific statistics run only after all 30 prediction artifacts pass completeness and paired-alignment validation.

