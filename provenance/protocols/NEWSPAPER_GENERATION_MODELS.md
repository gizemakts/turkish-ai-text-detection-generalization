# Generation model labels

The AI components of ABSTRACT and NEWSPAPER were assembled from five generator groups. The model identifiers below are the values recorded in the `model` column of the corresponding generation datasets (`ai_abstracts_1000.csv` and `ai_newspaper_columns_500.csv`). Each ABSTRACT group contains 200 texts; each NEWSPAPER group contains 100 texts.

| Generator | Dataset model identifier | ABSTRACT texts | NEWSPAPER texts |
|---|---|---:|---:|
| GPT-5.5 (High) | `gpt_5-5-high` | 200 | 100 |
| Claude Opus 4.8 (High) | `claude-opus-4-8-high` | 200 | 100 |
| Qwen 3.7 (Max) | `qwen_3-7-max` | 200 | 100 |
| Gemini 3.1 Pro (extended thinking) | `gemini_pro-3-1-extended-thinking` | 200 | 100 |
| Kimi K2.6 (Fast/High) | `kimi_k-2-6_fast-high` | 200 | 100 |

The AI-generated texts and model identifiers are provided in `data/abstract/ai_abstracts_1000.csv` and `data/newspaper/ai_newspaper_columns_500.csv`. The original human-written texts are not redistributed. The model identifiers are part of the recorded dataset metadata, rather than inferred from individual output filenames. The construction and pairing procedures are described in `newspaper_provenance.json` and the accompanying article.
