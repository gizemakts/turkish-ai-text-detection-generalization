# External cross-corpus evaluation

This pipeline implements the fixed-model external evaluation described in the article. Its source modules are in `src/turkish_ai_detection/experiments/external_cross_corpus/`. The active configuration is `configs/external_cross_corpus.json`; historical run identities and integrity evidence are retained under `provenance/experiments/external_cross_corpus_v2/`.

| Entry point | Purpose |
|---|---|
| `01_review.py` | Static configuration and protocol checks |
| `02_fit.py` | Train the recorded CE/SupCon seed-fold realizations |
| `07_calibration_compatibility.py` | Check internal compatibility of calibration mappings before external scoring |
| `03_inference.py` | Generate external scores without target-label fitting |
| `04_seal.py` | Verify and seal prediction outputs before label linkage |
| `05_statistics.py` | Calculate corpus-specific ROC-AUC and paired uncertainty |
| `06_secondary_audits.py` | Perform secondary external diagnostic analyses |
| `08_conditional_metrics.py` | Evaluate transferred calibration and thresholds after prediction sealing |

The numerical prefixes identify entry points, not an uninterrupted execution order. The calibration compatibility check precedes external inference; statistics and conditional metrics follow prediction sealing. Training and inference require the original approved data and checkpoints. Entry points that perform scientific computation require explicit arguments. The repository includes aggregate outputs but not the sealed per-text predictions needed to repeat the full workflow.

Internal Platt mappings and thresholds are not refitted on external labels. Estimated external false-positive rates are descriptive and do not constitute guaranteed operating characteristics.
