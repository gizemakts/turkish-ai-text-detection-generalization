# Post-hoc calibration analysis

This analysis is a post-hoc exploratory secondary analysis, not a
prospectively prespecified component of the frozen V2 external evaluation.
The original checkpoints, sealed raw predictions, primary AUROC statistics,
historical calibration artifacts, and the 27 incompatible historical mapping
decisions were left unchanged.

For each of the 15 `CE_ONLY_MATCHED` and 15 `SUPCON` terminal
realizations (three seeds and five outer folds), Platt logistic calibration
was fitted only to the existing 2,400-row outer-training inner-OOF prediction
set (1,200 human and 1,200 AI observations). The four operating thresholds
(nominal FPR 1%, 2%, 3%, and 5%; delta 0.05) were obtained only from the
internal OOF human probabilities using the established `np_umbrella_k`
order-statistic helper. Every calibration source hash and zero-overlap
leakage assertion was recorded in `PARAMETER_MANIFEST.json`, which was frozen
before any external label was read.

The frozen mappings were then applied to the already sealed 135-cell raw
prediction grid; no external inference was rerun. External results include
per-realization Brier score, log loss, 10-bin equal-width ECE, confusion
counts, realized FPR/TPR/FNR, and the difference between realized and nominal
FPR. Matched CE--SupCon differences are provided by dataset, seed, and outer
fold. The threshold results are transfer observations under dataset shift:
an internal nominal FPR is not a guarantee of the same external FPR.

The three datasets show substantial variation in realized operating points,
particularly for the low-FPR thresholds. Therefore these results should be
reported descriptively, without confirmatory p-values or claims that one
calibrator generalizes better. Five outer folds per seed are dependent model
realizations, and repeated prediction rows are not independent texts. The
package consequently reports realization means and sample SDs and documents
the limitation that no additional bootstrap confidence interval was claimed
where the frozen statistical protocol did not define a valid paired
resampling estimand. LIT2026 remains unavailable by design because no
corresponding frozen model-specific calibration and low-FPR mapping exists.
