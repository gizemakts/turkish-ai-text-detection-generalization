# Author-disjoint NEWSPAPER: statistical analysis

The author-disjoint experiment evaluates cross-author generalization. Ten authors contribute 50 human-AI pairs each. The design uses five author-held-out folds and three seeds per training objective. Author-level dependence is accounted for in the primary uncertainty analysis.

## Estimation

ROC-AUC is computed per held-out author and from the concatenated out-of-fold predictions for each seed. CE and SupCon use identical authors, folds, seeds, record identifiers and binary labels. The mean and sample standard deviation across three seeds summarize each method.

## Uncertainty

The primary bootstrap samples 10 authors with replacement, retaining all 50 human-AI pairs from each selected author. Each replicate uses the same sampled authors across methods and seeds. The analysis uses 5,000 bootstrap replicates and random seed 2602, with percentile 95% confidence intervals for CE ROC-AUC, SupCon ROC-AUC and their difference.

A pair-level bootstrap is included as a secondary sensitivity analysis using shared resampled pairs across methods and seeds. The primary intervals pertain to variation across the ten observed author clusters; they do not establish uncertainty for an unrestricted population of Turkish authors, publications, or time periods.

## Limitations

Only ten author clusters are observed. These analyses do not eliminate possible historical-period, source, generator-prompt or corpus-construction confounding.
