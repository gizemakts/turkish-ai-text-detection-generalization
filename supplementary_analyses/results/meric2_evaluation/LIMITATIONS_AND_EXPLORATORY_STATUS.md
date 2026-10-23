# Limitations and exploratory status

- This is a post-hoc exploratory analysis. Meric2 and the earlier Meric2-augmented CE pilot had already been examined; that pilot reported no measurable improvement.
- Only uncontaminated, frozen Combined-v2 terminal checkpoints and Combined-v2-only M4 fits are eligible. The Meric2-augmented pilot checkpoint is prohibited.
- Transformer results are restricted to verifier-authorized seeds 42, 314, and 2026. Seeds 137 and 3407 are excluded; therefore this is a three-seed analysis with five dependent folds per seed. The five TF-IDF outer models are unchanged.
- Meric2 has no authoritative source-lineage, author, generator-version, prompt-condition, publication-date, or generation-date identifiers. HUMAN–AI source pairs cannot be recovered or invented.
- Exact-word-count matching controls only observed integer length. Its rank-aligned records are artificial bootstrap/balance blocks, not source pairs.
- After deduplication, observation-level independence remains unverified. Seed-level variability and dependent-fold summaries must be kept separate from evaluation-sample uncertainty.
- Lexical/surface differences and score correlations are associational. They cannot establish causal reliance, historical effects, or generation-instruction effects.
- Meric2 cannot reproduce the original restrictive-versus-permissive generation experiment.
- No Meric2 result may be used for model, checkpoint, threshold, feature, or hyperparameter selection.
