# Scientific interpretation and limitations

This analysis examines word-count and truncation sensitivity using the specified evaluation folds.

## Findings

- Word count alone was weak for Combined-v2 (pooled OOF ROC-AUC 0.500) and differed by domain: ABSTRACT 0.502, NEWSPAPER 0.505.
- The ±10-word rule retained 1075/1500 pairs. Across five seeds, CE/SupCon mean pooled-OOF AUCs remained 0.999924/0.999725.
- Combined-v2 truncation affected 33.5% of HUMAN and 31.0% of AI texts. Domain × class results are reported separately; domain differences alone are not evidence of a class shortcut.
- Near-ceiling CE/SupCon discrimination persists under every prespecified pair-preserving length criterion. This reduces concern that gross pairwise word-count differences alone explain internal performance, but does not establish independence from length or correlated construction cues.

## Limitations

- Pair filtering changes the target evaluation population.
- Outer-fold estimates and repeated training seeds are dependent; fold rows are diagnostics, not independent replicates.
- Seed SD/ranges describe training-realization variability. The baseline pair bootstrap is conditional on the fixed folds and specification.
- High matched-subset performance cannot rule out subtler length effects or correlated corpus artifacts.
- This analysis does not resolve historical-period or generator-prompt confounding.
