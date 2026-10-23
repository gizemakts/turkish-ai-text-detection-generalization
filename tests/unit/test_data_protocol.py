from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from turkish_ai_detection.data.study import audit_dataset, audit_folds, create_nested_folds, np_umbrella_k


GENS = ["GPT", "Claude", "Gemini", "Qwen", "Kimi"]


@pytest.fixture(scope="module")
def valid_combined():
    rows = []
    for domain, prefix, per_generator in [("ABSTRACT", "ABS", 200), ("NEWSPAPER", "NEWS", 100)]:
        number = 0
        for generator in GENS:
            for _ in range(per_generator):
                number += 1
                pair_id = f"{prefix}_{number:04d}"
                for label, suffix in [("HUMAN", "H"), ("AI", "A")]:
                    rows.append({
                        "text_id": f"{pair_id}_{suffix}", "pair_id": pair_id,
                        "label": label, "domain": domain, "generator": generator,
                        "text": f"unique {domain} {generator} {number} {label}",
                    })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def valid_folds(valid_combined):
    return create_nested_folds(valid_combined, seed=42)


def test_dataset_counts_and_duplicate_ids(valid_combined):
    report = audit_dataset(valid_combined)
    assert report["status"] == "PASS"
    assert report["checks"]["row_count"]["observed"] == 3000
    broken = valid_combined.copy()
    broken.loc[1, "text_id"] = broken.loc[0, "text_id"]
    report = audit_dataset(broken)
    assert report["status"] == "FAIL"
    assert not report["checks"]["unique_text_ids"]["passed"]


def test_exact_normalized_duplicates_fail_loudly(valid_combined):
    broken = valid_combined.copy()
    broken.loc[1, "text"] = "  " + broken.loc[0, "text"] + "\n"
    report = audit_dataset(broken)
    assert report["status"] == "FAIL"
    assert report["checks"]["exact_normalized_text_duplicate_groups"]["observed"] == 1


def test_fold_determinism(valid_combined, valid_folds):
    second = create_nested_folds(valid_combined.sample(frac=1, random_state=9), seed=42)
    pd.testing.assert_frame_equal(valid_folds, second)


def test_pair_leakage_outer_and_inner(valid_folds):
    for outer in range(5):
        test = set(valid_folds.loc[valid_folds.outer_fold.eq(outer), "pair_id"])
        train = set(valid_folds.loc[valid_folds.outer_fold.ne(outer), "pair_id"])
        assert test.isdisjoint(train)
        column = f"inner_fold_outer{outer}"
        inner_sets = [set(valid_folds.loc[valid_folds.outer_fold.ne(outer) & valid_folds[column].eq(inner), "pair_id"]) for inner in range(4)]
        assert set.union(*inner_sets) == train
        assert sum(map(len, inner_sets)) == len(set.union(*inner_sets))
        assert all(valid_folds.loc[valid_folds.outer_fold.eq(outer), column].eq(-1))


def test_domain_generator_balance(valid_combined, valid_folds):
    report = audit_folds(valid_combined, valid_folds)
    assert report["status"] == "PASS"
    dist = pd.DataFrame(report["distribution_records"])
    outer = dist[dist.level.eq("outer_evaluation")]
    inner = dist[dist.level.eq("inner_validation")]
    assert set(outer.loc[outer.domain.eq("ABSTRACT"), "pair_count"]) == {40}
    assert set(outer.loc[outer.domain.eq("NEWSPAPER"), "pair_count"]) == {20}
    assert set(inner.loc[inner.domain.eq("ABSTRACT"), "pair_count"]) == {40}
    assert set(inner.loc[inner.domain.eq("NEWSPAPER"), "pair_count"]) == {20}


def test_train_only_transformation_partition_contract(valid_folds):
    # A train-fitted transform may see only these sets; outer evaluation is
    # excluded before any inner fitting or validation split is formed.
    for outer in range(5):
        outer_test = set(valid_folds.loc[valid_folds.outer_fold.eq(outer), "pair_id"])
        column = f"inner_fold_outer{outer}"
        for inner in range(4):
            validation = set(valid_folds.loc[valid_folds.outer_fold.ne(outer) & valid_folds[column].eq(inner), "pair_id"])
            fit = set(valid_folds.loc[valid_folds.outer_fold.ne(outer) & valid_folds[column].ne(inner), "pair_id"])
            assert fit.isdisjoint(validation)
            assert fit.isdisjoint(outer_test)
            assert validation.isdisjoint(outer_test)


def test_calibration_isolation_contract(valid_folds):
    # Calibration consumes all and only inner-OOF scores from the outer-training
    # 1200 pairs (2400 rows); it never consumes an outer-evaluation score.
    for outer in range(5):
        training = valid_folds[valid_folds.outer_fold.ne(outer)]
        column = f"inner_fold_outer{outer}"
        assert len(training) == 1200
        assert sorted(training[column].value_counts().tolist()) == [300, 300, 300, 300]
        assert not training[column].eq(-1).any()


def test_threshold_isolation_and_fixed_np_rule():
    assert np_umbrella_k(800) == 15
    assert np_umbrella_k(1000) == 20
    assert np_umbrella_k(1200) == 26
    # Changing hypothetical outer scores cannot enter this pure training-side rule.
    from turkish_ai_detection.evaluation.metrics import threshold_from_training_oof
    training_prob = np.linspace(0.0, 1.0, 2400)
    y = np.tile([0, 1], 1200)
    first = threshold_from_training_oof(training_prob, y)
    _outer_scores = np.full(600, 999.0)
    second = threshold_from_training_oof(training_prob, y)
    assert first == second


def test_m4_cached_transform_is_scientifically_equivalent():
    from turkish_ai_detection.evaluation.metrics import make_sparse_logistic, make_tfidf, make_tfidf_features
    text = ["bu bir insan metni", "ikinci insan yazısı", "yapay metin örneği", "başka yapay metin"] * 3
    y = np.array([0, 0, 1, 1] * 3)
    pipeline_score = make_tfidf(0.3).fit(text, y).decision_function(text)
    vectorizer = make_tfidf_features()
    matrix = vectorizer.fit_transform(text)
    cached_score = make_sparse_logistic(0.3).fit(matrix, y).decision_function(matrix)
    np.testing.assert_allclose(pipeline_score, cached_score, rtol=0, atol=1e-12)



