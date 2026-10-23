# Combined-v2 partition metadata

Combined-v2 consists of 1,500 HUMAN-AI pairs (3,000 observations), including 1,000 ABSTRACT pairs and 500 NEWSPAPER pairs. The complete text corpus is excluded from this public-code archive; only text-free partition metadata are included here.

Included files:

- `nested_folds.csv`: pair-disjoint outer/inner partition metadata.
- `near_duplicate_clusters_v1.csv`: lexical near-duplicate cluster assignments.
- `cluster_aware_folds_v1.csv`: grouping-aware sensitivity partition metadata.

The relevant experimental configurations are in `../../configs/`; data requirements and redistribution limits are in `../../DATA_ACCESS.md` and `../../REPRODUCIBILITY.md`.

`generator` identifies the AI generator stratum of a matched pair. Its presence on a HUMAN record denotes pair stratification, not attribution of human authorship to an AI model.
