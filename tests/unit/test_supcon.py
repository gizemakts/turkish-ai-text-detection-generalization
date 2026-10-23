import json

import numpy as np
import pytest
import torch

from turkish_ai_detection.experiments.model_development.supcon import train as supcon

CONFIG = json.loads(supcon.CONFIG_PATH.read_text(encoding="utf-8"))


def test_positive_mask_uses_same_label_and_excludes_self_pairs():
    labels = torch.tensor([0, 0, 1, 1])
    positives, negatives, self_mask = supcon.supervised_contrastive_masks(labels)
    expected = torch.tensor([
        [False, True, False, False], [True, False, False, False],
        [False, False, False, True], [False, False, True, False],
    ])
    assert torch.equal(positives.cpu(), expected)
    assert not positives.diagonal().any()
    assert torch.equal(negatives, labels[:, None].ne(labels[None, :]))
    assert self_mask.diagonal().all()


def test_cross_microbatch_labels_define_positives_and_negatives():
    labels = torch.tensor([0, 1] * 16)
    positives, negatives, _ = supcon.supervised_contrastive_masks(labels)
    assert int(positives[:16, 16:].sum()) == 128
    assert int(negatives[:16, 16:].sum()) == 128


def test_supcon_loss_is_finite_stable_and_has_gradients():
    generator = torch.Generator().manual_seed(42)
    embedding = (torch.randn(16, 128, generator=generator) * 1000).requires_grad_()
    loss = supcon.supervised_contrastive_loss(embedding, torch.tensor([0, 1] * 8))
    assert torch.isfinite(loss)
    loss.backward()
    assert embedding.grad is not None and torch.isfinite(embedding.grad).all()


def test_temperature_is_exact_and_changes_the_loss():
    embedding = torch.tensor([[1., 0.], [.9, .1], [0., 1.], [.1, .9]])
    labels = torch.tensor([0, 0, 1, 1])
    frozen = supcon.supervised_contrastive_loss(embedding, labels, temperature=.07)
    changed = supcon.supervised_contrastive_loss(embedding, labels, temperature=.14)
    assert supcon.TEMPERATURE == CONFIG["objective"]["temperature"] == .07
    assert not torch.isclose(frozen, changed)
    with pytest.raises(ValueError, match="positive"):
        supcon.supervised_contrastive_loss(torch.eye(2), torch.tensor([0, 1]), .07)


def test_loss_normalizes_embeddings_and_is_scale_invariant():
    embedding = torch.tensor([[2., 0.], [1.8, .2], [0., 3.], [.3, 2.7]])
    labels = torch.tensor([0, 0, 1, 1])
    first = supcon.supervised_contrastive_loss(embedding, labels)
    second = supcon.supervised_contrastive_loss(embedding * 17.0, labels)
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)


def test_balanced_sampler_is_deterministic_and_training_label_only():
    labels = np.array([0, 1] * 900)
    first = list(supcon.BalancedClassSampler(labels, 16, 42))
    second = list(supcon.BalancedClassSampler(labels, 16, 42))
    assert first == second and len(first) == 1792
    assert np.bincount(labels[first], minlength=2).tolist() == [896, 896]
    sampler = supcon.BalancedClassSampler(labels, 16, 42)
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_architecture_objective_and_effective_batch_are_prespecified():
    assert CONFIG["architecture"]["classification_head"] == "Dropout(0.1) -> Linear(768, 2)"
    assert CONFIG["architecture"]["projection_head"] == "Linear(768, 256) -> GELU -> Linear(256, 128) -> L2 normalize"
    assert supcon.CLS_DIM == 768 and supcon.PROJECTION_DIM == 128
    assert supcon.LAMBDA_SUPCON == .1
    assert (supcon.PHYSICAL_BATCH_SIZE, supcon.GRADIENT_ACCUMULATION_STEPS, supcon.EFFECTIVE_BATCH_SIZE) == (16, 2, 32)
    assert CONFIG["batching"]["contrastive_candidate_size"] == 32
    assert CONFIG["batching"]["per_microbatch_supcon_forbidden"] is True
