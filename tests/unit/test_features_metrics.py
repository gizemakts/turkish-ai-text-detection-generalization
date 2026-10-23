"""Regression tests for the 2026-09-23 08A implementation corrections."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "src"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from turkish_ai_detection.features.core import (
    SURFACE_FEATURES,
    artifact_neutral_context_text,
    artifact_neutral_text,
    clopper_pearson_interval,
    ece_equal_width,
    letter_word_spans,
    paired_pair_bootstrap_auc_delta,
    p2_mes_decision,
    sentence_features_v6,
    surface_model_matrix,
    wordpiece_positions_overlapping_spans,
)


class _Tok:
    def __init__(self, start_char):
        self.start_char = start_char


class _Sent:
    def __init__(self, start_char):
        self.tokens = [_Tok(start_char)]


class _Doc:
    def __init__(self, starts):
        self.sentences = [_Sent(x) for x in starts]


class _OneSentenceNLP:
    def __call__(self, text):
        return _Doc([0])


def test_condition_b_context_preserves_digits_but_surface_text_removes_them():
    text = "Ama\u00e7: COVID-19 \u00f6rnekleri 653\u2019\u00fc i\u00e7erir \u2014 sonu\u00e7 42."
    context = artifact_neutral_context_text(text)
    surface = artifact_neutral_text(text)

    assert not context.lower().startswith("amaç:")
    assert "COVID-19" in context
    assert "653'ü" in context
    assert "42." in context
    assert "—" not in context and "-" in context

    assert "COVID-19" not in surface
    assert "653'ü" not in surface
    assert "42." not in surface
    assert "örnekleri" in surface and "içerir" in surface


def test_condition_b_sentence_lengths_count_only_letter_containing_whitespace_words():
    text = "ABC 123 p53 (%) son"
    out_all = sentence_features_v6(text, _OneSentenceNLP(), letter_words_only=False)
    out_letters = sentence_features_v6(text, _OneSentenceNLP(), letter_words_only=True)
    assert out_all["sentence_word_count"] == 5
    # ABC, p53 and son contain letters; 123 and (%) do not.
    assert out_letters["sentence_word_count"] == 3
    assert json.loads(out_letters["sentence_lengths"]) == [3]


def test_letter_word_spans_and_wordpiece_selection_keep_mixed_alphanumeric_words():
    text = "ABC 123 p53 %"
    spans = letter_word_spans(text)
    # Synthetic tokenizer offsets: ABC | 123 | p | 53 | %
    offsets = [(0, 3), (4, 7), (8, 9), (9, 11), (12, 13)]
    assert spans == [(0, 3), (8, 11)]
    assert wordpiece_positions_overlapping_spans(offsets, spans) == [0, 2, 3]


def test_m3_surface_matrix_is_log1p_of_protocol_features():
    row = {name: float(i) for i, name in enumerate(SURFACE_FEATURES)}
    df = pd.DataFrame([row])
    got = surface_model_matrix(df)
    expected = np.log1p(df[SURFACE_FEATURES].astype(float))
    assert list(got.columns) == SURFACE_FEATURES
    assert np.allclose(got.to_numpy(), expected.to_numpy())


def test_m3_surface_matrix_rejects_negative_values():
    row = {name: 0.0 for name in SURFACE_FEATURES}
    row[SURFACE_FEATURES[0]] = -1.0
    try:
        surface_model_matrix(pd.DataFrame([row]))
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative M3 input should fail")


def test_clopper_pearson_exact_boundaries_and_known_zero_event_upper_bound():
    lo, hi = clopper_pearson_interval(0, 100, alpha=.05)
    assert lo == 0.0
    assert np.isclose(hi, 1.0 - 0.025 ** (1.0 / 100.0), rtol=0, atol=1e-12)
    lo2, hi2 = clopper_pearson_interval(100, 100, alpha=.05)
    assert hi2 == 1.0
    assert np.isclose(lo2, 0.025 ** (1.0 / 100.0), rtol=0, atol=1e-12)


def test_ece_10_equal_width_includes_probability_one_in_last_bin():
    y = np.array([0, 1, 1])
    prob = np.array([0.1, 0.9, 1.0])
    # |0-.1|/3 + |1-.9|/3 + |1-1|/3 = 0.066666...
    assert np.isclose(ece_equal_width(y, prob, n_bins=10), 2.0 / 30.0)


def _synthetic_pair_oof():
    rows = []
    # Six complete human/AI pairs; probabilities are deliberately different
    # between M1 and M2 so the paired contrast is non-trivial.
    m1_h = [0.05, 0.10, 0.20, 0.30, 0.15, 0.25]
    m1_a = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65]
    m2_h = [0.10, 0.20, 0.35, 0.40, 0.25, 0.30]
    m2_a = [0.80, 0.70, 0.60, 0.55, 0.50, 0.45]
    a, b = [], []
    for i in range(6):
        pid = f'p{i}'
        for label, p1, p2 in [('human', m1_h[i], m2_h[i]), ('ai', m1_a[i], m2_a[i])]:
            sid = f'{pid}_{label}'
            a.append({'sample_id': sid, 'pair_id': pid, 'label': label, 'prob': p1})
            b.append({'sample_id': sid, 'pair_id': pid, 'label': label, 'prob': p2})
    return pd.DataFrame(a), pd.DataFrame(b)


def test_p2_pair_bootstrap_is_deterministic_and_pair_aligned():
    m1, m2 = _synthetic_pair_oof()
    one = paired_pair_bootstrap_auc_delta(m1, m2, B=50, seed=42)
    two = paired_pair_bootstrap_auc_delta(m1, m2, B=50, seed=42)
    assert one['B'] == 50 and one['seed'] == 42
    assert len(one['bootstrap_deltas']) == 50
    assert np.array_equal(one['bootstrap_deltas'], two['bootstrap_deltas'])
    assert np.isfinite(one['bootstrap_deltas']).all()


def test_p2_mes_three_way_decision_rule():
    assert p2_mes_decision(.02, .005, .03, sesoi=.01) == 'meaningful_incremental_contribution'
    assert p2_mes_decision(.005, -.01, .009, sesoi=.01) == 'no_meaningful_incremental_contribution'
    assert p2_mes_decision(.009, .001, .02, sesoi=.01) == 'uncertain'
