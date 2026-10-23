# Corpus-construction shortcut analysis

Status: `COMPLETED_VALIDATED`; analysis class: `POST_HOC_EXPLORATORY`.

The evaluation population was all 1,500 Combined-v2 HUMAN–AI pairs. Feature definitions are in the frozen protocol and script. The pair-disjoint, train-only-scaled surface probe achieved ROC-AUC 0.965 (pair-bootstrap 95% CI 0.959–0.970) and AP 0.963. The largest raw-scale mean paired difference was for `wordpiece_count` (-58.4, AI minus HUMAN); raw scales are not comparable across features.

These results demonstrate observable corpus separability using predefined surface measurements. They do **not** demonstrate that BERTurk causally relies on any feature. Score associations are exploratory correlations, and the existing Condition-B results remain unchanged.
