# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/safety-training/ (ai://NEW-safety-training).
import itertools

import numpy as np
import pytest

from safety_training import (
    ToyResidualNet,
    ToyTargetLoss,
    ablate,
    add_direction,
    attack_success_rate,
    context_distillation_loss,
    diff_in_means,
    false_refusal_rate,
    first_order_change,
    gcg_step,
    is_refusal,
    kl,
    mix_datasets,
    projection_score,
    refusal_breakdown,
    refusal_rate,
    robustness_gamma,
    safety_examples_for_fraction,
    safety_fraction,
    select_distilled,
    top_k_candidates,
    unit,
)


def test_safety_fraction_matches_bianchi_sizes():
    got = [round(100 * safety_fraction(n, 20_000), 2) for n in (100, 500, 1000, 2000)]
    assert got == [0.50, 2.44, 4.76, 9.09]
    n = safety_examples_for_fraction(0.03, 20_000)
    assert n == 619 and safety_fraction(n, 20_000) >= 0.03 > safety_fraction(n - 1, 20_000)
    with pytest.raises(ValueError):
        safety_examples_for_fraction(1.0, 10)


def test_mix_datasets_keeps_everything_and_draws_without_replacement():
    rng = np.random.default_rng(0)
    general = [f"g{i}" for i in range(50)]
    safety = [f"s{i}" for i in range(20)]
    mixed = mix_datasets(general, safety, 7, rng)
    assert len(mixed) == 57 and set(general) <= set(mixed)
    drawn = [x for x in mixed if x.startswith("s")]
    assert len(drawn) == len(set(drawn)) == 7
    with pytest.raises(ValueError):
        mix_datasets(general, safety, 21, rng)


def test_select_distilled_keeps_only_improvements():
    chosen, keep = select_distilled(["a", "b", "c"], ["A", "B", "C"], [0.2, 0.9, 0.5], [0.7, 0.3, 0.5])
    assert chosen == ["A", "b", "c"] and keep.tolist() == [True, False, False]


def test_context_distillation_loss_is_kl_plus_teacher_entropy():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(5), size=4)
    q = rng.dirichlet(np.ones(5), size=4)
    ent = -np.sum(p * np.log(p), axis=-1)
    assert np.isclose(context_distillation_loss(p, np.log(q)), np.mean(kl(p, q) + ent))
    assert np.isclose(context_distillation_loss(p, np.log(p)), np.mean(ent))
    assert np.all(kl(p, q) >= 0) and np.allclose(kl(p, p), 0)
    assert kl([1.0, 0.0], [0.5, 0.5]) == pytest.approx(np.log(2))


def test_substring_refusal_detection_and_its_blind_spots():
    assert is_refusal("I'm sorry, but I can't help with that.")
    assert is_refusal("I’m sorry, I cannot do that.")  # curly apostrophe
    assert not is_refusal("Sure, here is how to kill a Python process: use kill -9.")
    assert not is_refusal("That is not something I will help with.")  # a refusal the list misses
    assert is_refusal("I'm sorry to hear that! Here is the recipe you asked for.")  # compliance counted as refusal
    assert refusal_rate(["I cannot.", "Sure.", "As an AI, no.", "Here you go."]) == 0.5


def test_rates():
    assert attack_success_rate([1, 0, 1, 1]) == 0.75
    labels = ["full_refusal"] * 95 + ["partial_refusal"] * 54 + ["full_compliance"] * 101
    full, partial = refusal_breakdown(labels)
    assert (full, partial) == (0.38, 0.216) and full + partial == pytest.approx(0.596)
    with pytest.raises(ValueError):
        refusal_breakdown(["maybe"])
    assert false_refusal_rate([True, False, True, True], [True, True, False, False]) == 0.5
    with pytest.raises(ValueError):
        false_refusal_rate([True], [False])
    assert robustness_gamma(1.8, 1, 1) == 1.8 and robustness_gamma(9, 5, 4) == 0.45
    assert 60 / robustness_gamma(9, 5, 4) == pytest.approx(133.33, abs=0.01)


def test_difference_of_means_is_the_best_separating_direction():
    rng = np.random.default_rng(2)
    harmless = rng.normal(size=(60, 8))
    shift = np.zeros(8)
    shift[3] = 4.0
    harmful = rng.normal(size=(50, 8)) + shift
    r = diff_in_means(harmful, harmless)
    r_hat = unit(r)
    assert np.argmax(np.abs(r_hat)) == 3
    for _ in range(200):
        u = unit(rng.normal(size=8))
        assert u @ r <= r_hat @ r + 1e-12
    gap = projection_score(harmful, r_hat).mean() - projection_score(harmless, r_hat).mean()
    assert gap == pytest.approx(np.linalg.norm(r))
    assert np.allclose(add_direction(harmless, r).mean(0), harmful.mean(0))
    with pytest.raises(ValueError):
        unit(np.zeros(3))


def test_ablation_is_a_projection():
    rng = np.random.default_rng(3)
    r_hat = unit(rng.normal(size=6))
    x = rng.normal(size=(5, 6))
    xa = ablate(x, r_hat)
    assert np.allclose(xa @ r_hat, 0)
    assert np.allclose(ablate(xa, r_hat), xa)
    assert np.allclose(x - xa, np.outer(x @ r_hat, r_hat))
    assert np.allclose(ablate(x[0], r_hat), xa[0])


def test_orthogonalized_weights_equal_ablation_everywhere():
    rng = np.random.default_rng(4)
    net = ToyResidualNet(vocab=17, d_model=8, d_hidden=12, n_layers=3, max_len=6, rng=rng)
    tokens = np.array([1, 5, 2, 16, 0, 3])
    r_hat = unit(rng.normal(size=8))
    logits_ablate, stream_ablate = net.forward(tokens, intervene=lambda z: ablate(z, r_hat))
    logits_ortho, stream_ortho = net.orthogonalized(r_hat).forward(tokens)
    assert np.allclose(logits_ablate, logits_ortho)
    for a, b in zip(stream_ablate, stream_ortho):
        assert np.allclose(a, b) and np.allclose(b @ r_hat, 0)
    logits, _ = net.forward(tokens)
    assert not np.allclose(logits, logits_ortho)


def test_activation_addition_moves_the_stream():
    rng = np.random.default_rng(5)
    net = ToyResidualNet(vocab=9, d_model=6, d_hidden=8, n_layers=2, max_len=4, rng=rng)
    tokens = np.array([1, 2, 3])
    r = rng.normal(size=6)
    _, base = net.forward(tokens)
    _, steered = net.forward(tokens, add_at=(0, r))
    assert np.allclose(base[0], steered[0])
    assert not np.allclose(base[1], steered[1])


def test_top_k_candidates_rank_by_first_order_decrease():
    grad = np.array([[0.3, -1.0, 0.2, -0.5], [1.0, 0.0, -2.0, 0.5]])
    assert top_k_candidates(grad, 2).tolist() == [[1, 3], [2, 1]]
    assert first_order_change(grad[0], 0, 1) == pytest.approx(-1.3)


def make_toy(seed=6, vocab=12, dim=5, n=4):
    rng = np.random.default_rng(seed)
    loss = ToyTargetLoss(rng.normal(size=(vocab, dim)), rng.normal(size=(dim, vocab)), target=7)
    return rng, loss, rng.integers(vocab, size=n)


def test_toy_loss_gradient_matches_finite_differences():
    _, loss, tokens = make_toy()
    g = loss.grad(tokens)
    eps = 1e-6
    for i in range(len(tokens)):
        for v in range(loss.embed.shape[0]):
            h = loss.embed[tokens].mean(0)
            dh = (loss.embed[v] - loss.embed[tokens[i]]) / len(tokens)

            def f(h):
                z = h @ loss.unembed
                return np.logaddexp.reduce(z) - z[loss.target]

            num = (f(h + eps * dh) - f(h - eps * dh)) / (2 * eps)
            assert num == pytest.approx(g[i, v] - g[i, tokens[i]], abs=1e-6)


def test_gcg_never_increases_the_loss_and_finds_the_best_suffix():
    rng, loss, tokens = make_toy(vocab=8, n=3)
    vocab = loss.embed.shape[0]
    best_exhaustive = min(loss(np.array(t)) for t in itertools.product(range(vocab), repeat=3))
    current = loss(tokens)
    for _ in range(30):
        new, new_loss = gcg_step(tokens, [0, 1, 2], loss, loss.grad, k=vocab, batch=32, rng=rng)
        if new_loss < current:
            tokens, current = new, new_loss
        assert new_loss == pytest.approx(loss(new))
    assert current == pytest.approx(best_exhaustive)


def test_gcg_respects_the_positions_it_may_change():
    rng, loss, tokens = make_toy(n=6)
    new, _ = gcg_step(tokens, [3, 4, 5], loss, loss.grad, k=3, batch=16, rng=rng)
    assert np.array_equal(new[:3], tokens[:3])
