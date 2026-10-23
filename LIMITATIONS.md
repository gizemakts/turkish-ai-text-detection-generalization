# Study limitations

The main limitations relevant to interpreting the results are described in the accompanying article and Online Resource 1. In particular:

- Pair-disjoint evaluation on the constructed corpus does not measure generalization to unrelated sources. Near-ceiling internal discrimination contrasts with lower and variable performance across external corpora.
- Calibration and false-positive-rate thresholds estimated on internal folds do not carry a guaranteed operating point to other datasets.
- NEWSPAPER transfer, author-disjoint and surface-feature evaluations are sensitivity analyses on the study's constructed corpus. Exploratory external analyses should not be read as confirmatory evaluations.
- The three-seed CE/SupCon/LIT external comparison and the five-seed CE/SupCon extension have different comparison sets and must be interpreted separately.
- Resampling-based intervals conditional on trained realizations do not quantify all training, source-population or deployment uncertainty.

Reproduction from this code archive requires the original datasets and model assets described in `DATA_ACCESS.md` and `REPRODUCIBILITY.md`. These materials are excluded from the public download for data-access and redistribution reasons.
