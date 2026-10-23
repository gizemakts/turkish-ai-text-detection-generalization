# Data sources and access

This repository includes research code, experimental configurations, text-free evaluation partitions, aggregate results, and the generated-text components of ABSTRACT and NEWSPAPER. Human-written source texts, trained model checkpoints, and per-document model predictions are not distributed.

| Dataset | Description | Availability |
|---|---|---|
| Combined-v2 | 1,000 ABSTRACT and 500 NEWSPAPER human-AI pairs | The generated texts are provided in `data/abstract/` and `data/newspaper/`. Human-written source texts are not redistributed. |
| ABSTRACT | Turkish academic abstracts from 2005-2018 paired with generated texts | `data/abstract/ai_abstracts_1000.csv` contains generated texts and model identifiers. Original human abstracts are excluded because their access and reuse rights are not granted by this repository. |
| NEWSPAPER | Turkish opinion columns from 2017 paired with generated texts | `data/newspaper/ai_newspaper_columns_500.csv` contains generated texts and model identifiers. Original published columns are excluded because their access and reuse rights are not granted by this repository. The public NEWSPAPER AI file has normalized record identifiers and anonymized columnist-name mentions. |
| Renklier2026 | External Turkish evaluation corpus | Source described in `data/external_raw/EXTERNAL_DATASETS_SOURCE_LOCK.json`. |
| TUM2024_TR | External Turkish evaluation corpus | Hugging Face source recorded in the external-dataset source manifest. |
| Bostancioglu2026_TR | External Turkish evaluation corpus | Hugging Face source recorded in the external-dataset source manifest. |
| Meric2 | External exploratory corpus | [Turkish-AI-Generated-Content-Detection](https://github.com/meric2/Turkish-AI-Generated-Content-Detection). Its repository documents the `ai_data.csv` and `reviews.csv` files. |

The `data/study/` directory provides partitions and cluster metadata without the original texts. Reproducing published model fitting requires access to the excluded human-written data and necessary pretrained model assets. The provided generated-text files cannot by themselves reproduce the paired dataset.

## Licensing

The MIT License applies to the original code in this repository. It does not confer rights to redistribute third-party written works, external datasets, or model weights. The generated-text datasets are supplied as research data; usage remains subject to any applicable service and dataset terms.
