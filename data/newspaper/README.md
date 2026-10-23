# NEWSPAPER corpus

`ai_newspaper_columns_500.csv` contains the 500 AI-generated Turkish newspaper-style texts with `id`, `model`, and `generated_text` columns. The public CSV uses consistent `ai_0001` to `ai_0500` identifiers and replaces named newspaper columnists in generated texts with `[AUTHOR]`. The numerical suffixes retain the original pair alignment. These publication-only changes do not retroactively alter the reported model results. The paired human-written opinion columns are not distributed because their copyright and reuse conditions do not permit their publication as part of this repository.

The study uses 500 human-written columns from 2017 across ten authors and four newspapers. NEWSPAPER contributes 500 pairs to Combined-v2 and is also used in domain-transfer and author-disjoint evaluations. See `../../DATA_ACCESS.md`, `../../EXPERIMENTS.md` and `../../provenance/protocols/newspaper_provenance.json` for details.
