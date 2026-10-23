# Length-control analyses

The length-control analysis uses word counts, paired human-AI record identifiers and tokenizer-based truncation measurements from the existing study. It adds pairwise length summaries, a pair-disjoint word-count-only logistic baseline and a length-matched CE/SupCon sensitivity comparison. The latter restricts evaluation pairs by the specified length criterion without refitting the underlying models.

The encoder is `dbmdz/bert-base-turkish-cased` at revision `b6e1de16c983e0f2c70664591ea3f22810072608`. Primary CE/SupCon models use up to 512 tokens including special tokens, retaining 128 head and 382 tail WordPieces. The truncation diagnostics use token counts under the same tokenizer and input-length convention.

This is a post-hoc analysis of observed corpus properties. Matching on length alone does not control generation conditions, topics or other domain differences.
