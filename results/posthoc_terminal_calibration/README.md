# Post-hoc calibration and threshold transfer

This supplementary analysis applies inner-out-of-fold Platt calibration and four internal nominal-FPR thresholds (0.01, 0.02, 0.03 and 0.05) to fixed external-model scores. It does not modify the primary raw-score ROC-AUC results.

With the required research data, predictions and dependencies available, the entry point is:

```bash
python src/turkish_ai_detection/experiments/external_cross_corpus/posthoc_terminal_calibration.py prepare
python src/turkish_ai_detection/experiments/external_cross_corpus/posthoc_terminal_calibration.py evaluate
```

`prepare` constructs the 30 internal mappings without external labels. `evaluate` uses sealed external scores. LIT2026 has no corresponding calibration mapping. The outputs are exploratory and do not establish external FPR control.
