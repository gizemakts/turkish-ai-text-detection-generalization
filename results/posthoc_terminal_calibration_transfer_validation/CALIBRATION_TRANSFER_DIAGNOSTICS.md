# Targeted internal calibration-transfer validation

## Outcome

**Decision: `CALIBRATION_TRANSFER_LIMITED`.** Authentic held-out internal predictions were identified for all 30/30 terminal checkpoints. Each prediction file is hash-bound to a protocol-faithful terminal run, its model payload, and the three sealed external inference sidecars for the same method/seed/fold. All outer-test memberships, labels, objectives, seeds, folds, mapping identities, source hashes, and zero-overlap checks passed.

## Fixed-mapping calibration

No mapping was fitted or selected in this validation. Existing Platt coefficients were applied unchanged to each terminal checkpoint's own 600 held-out outer-test rows (300 HUMAN, 300 AI). Inner-OOF and terminal samples contain different observations.

- CE terminal medians were Brier 0.0020, log loss 0.0104, and ECE 0.0026; mean observed FPRs were 1%: 0.0058, 2%: 0.0113, 3%: 0.0204, 5%: 0.0307.
- SupCon terminal medians were Brier 0.0021, log loss 0.0142, and ECE 0.0028; mean observed FPRs were 1%: 0.0051, 2%: 0.0111, 3%: 0.0187, 5%: 0.0293.

Calibration intercept/slope MLEs were estimable in 19/30 realizations. Non-estimable cells are explicitly marked; complete or quasi-complete class separation makes an unpenalized calibration slope non-finite rather than evidencing a slope of exactly one.

## Score-distribution diagnostics

Class-conditional means, SDs, quantiles, Wasserstein distances, KS statistics, and standardized mean differences are reported for every method/seed/fold. These diagnostics assess compatibility but cannot prove that inner-model mappings and terminal-model scores are exchangeable.

Although held-out calibration metrics were strong, the 90th percentile absolute class-conditional raw-score SMD was 1.046 for CE and 1.065 for SupCon, exceeding the prespecified descriptive cutoff of 1.0. The largest shifts were concentrated in the AI class. This score-scale evidence is why the final status is limited rather than fully supported.

## Threshold transfer

The unchanged 1%, 2%, 3%, and 5% thresholds used the strict rule `calibrated_probability > threshold`. Each realization has 300 held-out HUMAN observations, so FPR resolution is 1/300 = 0.00333. Per-realization Clopper–Pearson intervals describe binomial uncertainty conditional on the evaluated terminal model and held-out fold. They do not transfer the original Neyman–Pearson guarantee to terminal scores.

## Decision criteria

The status uses the descriptive criteria recorded in `VALIDATION_STATUS.json`: authenticated provenance; method-level median terminal Brier/log loss/ECE; median degradation from inner OOF; class-conditional raw-score SMD; and threshold deviations across realizations. These are transparent diagnostic cutoffs, not confirmatory tests or mathematical guarantees.

## Dependence and scope

The 15 realizations per method are dependent. Outer folds within a seed partition one dataset, and texts recur across seeds. Results are reported by realization and with descriptive method summaries only. No external inference, calibration refit, threshold optimization, bootstrap rerun, or primary-result modification occurred.
