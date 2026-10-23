# NEWSPAPER cue analysis

Completed: `2026-10-14T15:07:49.520105+03:00`.

Stanza `1.14.0` with Turkish `boun` tokenization and `starlang` NER was applied to all 1,000 NEWSPAPER texts (500 HUMAN–AI pairs; 10 newspaper authors). The frozen deterministic temporal definitions and 20-token entity–temporal proximity window were preserved.

## Selected density results

| Feature | HUMAN mean | AI mean | AI − HUMAN | 95% author-cluster interval |
|---|---:|---:|---:|---:|
| person_per_100_words | 1.662125 | 0.299549 | -1.362576 | [-1.723953, -0.974721] |
| organization_per_100_words | 1.322388 | 0.177014 | -1.145374 | [-1.550529, -0.800250] |
| location_gpe_per_100_words | 1.941767 | 0.536076 | -1.405692 | [-1.965231, -0.869046] |
| named_entity_total_per_100_words | 4.926281 | 1.012638 | -3.913643 | [-5.012499, -2.916173] |
| temporal_merged_per_100_words | 0.855934 | 0.205391 | -0.650543 | [-0.950723, -0.373345] |
| temporal_near_entity_per_100_words | 0.672717 | 0.044424 | -0.628293 | [-0.885818, -0.380622] |

Intervals resample the ten newspaper-author clusters. Pair-level intervals and descriptive length/author-adjusted coefficients are in the CSV. These are associational corpus-construction differences; proxy fields are not ground-truth factual-event labels, and class differences do not establish that BERTurk used the cues.
