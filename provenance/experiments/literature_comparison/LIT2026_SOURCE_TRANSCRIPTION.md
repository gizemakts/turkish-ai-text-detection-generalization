# LIT2026 source transcription

Authoritative source used for the literature recipe:

- Ayhan Renklier & Ömer Faruk Sarıtaş (2026)
- "BERTurk tabanlı çok modelli yapay zekâ üretimli Türkçe metin tespiti"
- Niğde Ömer Halisdemir Üniversitesi Mühendislik Bilimleri Dergisi, volume 18
- DOI: 10.28948/ngumuh.1930411
- Published: 2026-09-09

The paper reports BERTurk (`dbmdz/bert-base-turkish-cased`) with a linear
768→2 classification layer and full fine-tuning. Section 3.3 / Table 5 reports:

- learning rate: 2e-5
- batch size: 8
- maximum epochs: 5
- early stopping: patience 2, based on validation F1
- maximum sequence length: 512 tokens
- weight decay: 0.01
- linear warmup: 500 steps
- mixed precision: FP16
- optimizer: AdamW

The paper uses a group-aware 70/10/20 train/validation/test split and states
that the first 512 tokens are used when text exceeds the model limit.

Important: LIT2026_REIMPL is a same-data recipe reimplementation on the
Turkish AI-Generated Text Detection Study corpus. It is not a reproduction of the paper's published score.

Adaptations frozen before LIT2026 results:
1. The frozen Turkish AI-Generated Text Detection Study 5 outer folds define the 20% test partition.
2. Within each outer-training pool, a deterministic pair-level validation
   partition of 150 pairs / 300 rows is selected, leaving 1050 pairs /
   2100 rows for training. This recreates 70/10/20 proportions over the
   3000-row Turkish AI-Generated Text Detection Study corpus.
3. The validation partition is stratified by generator × domain and is
   identical across the three training seeds; seeds therefore vary model
   stochasticity rather than the validation partition.
4. The frozen Turkish AI-Generated Text Detection Study population is not filtered with the literature
   paper's <30-word / duplicate removal rules, because doing so would change
   the common evaluation population. Condition A text is used unchanged.
5. The paper does not report gradient clipping, label smoothing, altered
   dropout, or class weighting. This reimplementation does not add those
   techniques.
6. Hugging Face `adamw_torch` is used as the concrete AdamW implementation.
7. Linear scheduling with 500 warmup steps is used; this operationalizes the
   paper's stated linear warmup while keeping the learning rate recipe fixed.

Published metrics from the paper belong to its own dataset and must not be
presented as a direct head-to-head score against Turkish AI-Generated Text Detection Study results.
