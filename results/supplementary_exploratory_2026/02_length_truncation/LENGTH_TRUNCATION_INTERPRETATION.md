# Length and truncation sensitivity

Status: `COMPLETED_VALIDATED`; analysis class: `POST_HOC_EXPLORATORY`.

The table contains 725 subgroup-by-realization estimates with every seed/fold retained. Internal rows are pair-disjoint pooled OOF estimates per seed; external rows are individual fixed-model estimates. Dependency-aware 95% intervals use 500 cluster draws while seeds/folds remain fixed; the lower replicate count reflects the cost of recomputing every fixed-realization AUC/AP and is recorded explicitly. No raw scores were pooled across external models. Cells lacking both classes are recorded as not estimable (`NaN`).

WordPiece counts use the pinned BERTurk tokenizer. The truncation indicator is shared, while effective selection is explicitly different: CE/SupCon use the recorded head-and-tail rule and LIT2026 uses first-512. Subgroups are descriptive and do not establish deployment reliability.
