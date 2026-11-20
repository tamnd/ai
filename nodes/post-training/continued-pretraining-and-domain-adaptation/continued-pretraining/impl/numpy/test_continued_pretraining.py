# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/continued-pretraining/ (ai://NEW-continued-pretraining).
import copy
import math

import numpy as np
import pytest
from continued_pretraining import (
    BigramLM,
    adaptation,
    compute_equivalent_replay,
    cosine_crossing,
    cosine_decay,
    ewc_penalty,
    extend_vocab,
    fisher_diagonal,
    forgetting,
    infinite_schedule,
    markov_pairs,
    mean_init_kl_bound,
    new_parameters,
    parmar_schedule,
    passes_over,
    replay_batches,
    rewarm_cosine,
    rope_rotate,
    rope_wavelengths,
    softmax,
    tokens_for_new,
    train,
)


def test_cosine_and_rewarm():
    assert math.isclose(cosine_decay(0, 3e-4, 3e-5), 3e-4) and math.isclose(cosine_decay(1, 3e-4, 3e-5), 3e-5)
    assert math.isclose(cosine_decay(0.5, 3e-4, 3e-5), 1.65e-4)
    lr = rewarm_cosine(np.arange(1001), 100, 1000, 3e-4, 3e-5)
    assert lr[0] == 0 and math.isclose(lr[50], 1.5e-4) and math.isclose(lr[100], 3e-4) and math.isclose(lr[1000], 3e-5)
    assert np.all(np.diff(lr[:101]) > 0) and np.all(np.diff(lr[100:]) <= 0)


def test_parmar_switch_point():
    start, end, s = parmar_schedule(4.5e-5)
    assert math.isclose(start, 4.5e-5) and math.isclose(end, 4.5e-7)
    assert abs(s - 0.711) < 1e-3 and math.isclose(cosine_decay(s, start, end), 9e-6)
    with pytest.raises(ValueError):
        cosine_crossing(1.0, 4.5e-5, 4.5e-7)


def test_infinite_schedule_phases_join():
    kw = {
        "warmup": 10,
        "cooldown": 40,
        "t_ann": 200,
        "t_end": 300,
        "eta_max": 3e-4,
        "eta_const": 1.5e-4,
        "eta_min": 3e-5,
    }
    lr = np.array([infinite_schedule(t, **kw) for t in range(301)])
    assert math.isclose(lr[10], 3e-4) and math.isclose(lr[50], 1.5e-4) and math.isclose(lr[199], 1.5e-4)
    assert math.isclose(lr[300], 3e-5) and math.isclose(lr[250], 1.5e-4 * (0.2**0.5))
    assert np.max(np.abs(np.diff(lr[10:]))) < 1e-5  # no jumps after warmup


def test_replay_accounting():
    new, old = compute_equivalent_replay(100e9, 0.05)
    assert math.isclose(new, 95e9) and math.isclose(old, 5e9)
    assert math.isclose(tokens_for_new(100, 0.5), 200) and math.isclose(tokens_for_new(75, 0.25), 100)
    with pytest.raises(ValueError):
        tokens_for_new(1, 1.0)
    assert math.isclose(passes_over(200e9, 0.95, 55e9), 190 / 55)


def test_replay_batches_fraction_and_order():
    rng = np.random.default_rng(0)
    got = list(replay_batches(1000, 30, 8, 0.25, 10, rng))
    assert all(len(o) == 2 and len(n) == 6 for o, n in got)
    news = np.concatenate([n for _, n in got])
    assert np.array_equal(news[:30], np.arange(30)) and np.array_equal(news[30:60], np.arange(30))
    assert all(len(o) == 0 for o, _ in replay_batches(10, 30, 8, 0.0, 3, rng))


def test_ewc_penalty_and_fisher():
    rng = np.random.default_rng(1)
    theta_old, fisher = rng.normal(size=5), rng.uniform(0, 2, size=5)
    theta = rng.normal(size=5)
    _val, g = ewc_penalty(theta, theta_old, fisher, lam=3.0)
    eps = 1e-6
    for i in range(5):
        e = np.zeros(5)
        e[i] = eps
        num = (
            ewc_penalty(theta + e, theta_old, fisher, 3.0)[0] - ewc_penalty(theta - e, theta_old, fisher, 3.0)[0]
        ) / (2 * eps)
        assert math.isclose(num, g[i], rel_tol=1e-6)
    assert ewc_penalty(theta_old, theta_old, fisher, 3.0)[0] == 0
    grads = rng.normal(size=(50, 5))
    assert np.allclose(fisher_diagonal(grads), (grads**2).mean(0))


def test_mean_init_bounds_the_change():
    rng = np.random.default_rng(2)
    V, H, k = 40, 6, 25
    embed, head = rng.normal(size=(V, H)), rng.normal(size=(V, H))
    e2, h2 = extend_vocab(embed, head, k)
    assert e2.shape == (V + k, H) and np.array_equal(e2[:V], embed) and np.array_equal(h2[:V], head)
    for _ in range(50):
        h = rng.normal(size=H) * 3
        p, q = softmax(head @ h), softmax(h2 @ h)
        assert q[V:].sum() <= k / (V + k) + 1e-12
        assert np.sum(p * np.log(p / q[:V])) <= mean_init_kl_bound(V, k) + 1e-12
    e3, h3 = extend_vocab(embed, head, k, rng=rng, init="normal")
    assert e3.shape == (V + k, H) and not np.allclose(h3[V:], h3[V])
    assert new_parameters(32000, 49953, 4096) == 147070976 and new_parameters(32000, 49953, 4096, tied=True) == 73535488
    assert abs(mean_init_kl_bound(32000, 17953) - 0.4453) < 1e-4


def test_rope_base():
    w4, w6 = rope_wavelengths(128, 1e4), rope_wavelengths(128, 1e6)
    assert math.isclose(w4[0], 2 * math.pi) and math.isclose(w6[0], 2 * math.pi)
    assert abs(w6[-1] / w4[-1] - 93.06) < 0.01
    rng = np.random.default_rng(3)
    q, k = rng.normal(size=(1, 16)), rng.normal(size=(1, 16))
    # Scores depend only on the offset, for any base.
    for base in (1e4, 5e5):
        s1 = rope_rotate(q, [7], base) @ rope_rotate(k, [3], base).T
        s2 = rope_rotate(q, [104], base) @ rope_rotate(k, [100], base).T
        assert np.allclose(s1, s2)
    assert np.allclose(rope_rotate(q, [0], 1e4), q)


def test_forgetting_replay_and_ewc_on_a_bigram_model():
    rng = np.random.default_rng(4)
    V = 6
    A = np.full((V, V), 0.02)
    A[np.arange(V), (np.arange(V) + 1) % V] = 0.9
    A /= A.sum(1, keepdims=True)
    B = np.full((V, V), 0.02)
    B[np.arange(V), (np.arange(V) + 2) % V] = 0.9
    B /= B.sum(1, keepdims=True)
    old, new = markov_pairs(A, 3000, rng), markov_pairs(B, 3000, rng)
    old_test, new_test = markov_pairs(A, 2000, rng), markov_pairs(B, 2000, rng)
    base = train(BigramLM(V, rng), old, 1500, 32, lambda t: 1.0, rng)
    before_old, before_new = base.loss(old_test), base.loss(new_test)

    def cont(**kw):
        return train(copy.deepcopy(base), new, 600, 32, lambda t: 1.0, np.random.default_rng(5), **kw)

    plain = cont()
    replay = cont(old=old, replay_fraction=0.25)
    fisher = fisher_diagonal(base.per_example_grads(old[:500]))
    ewc = cont(ewc=(base.W.ravel().copy(), fisher, 50.0))
    f_plain = forgetting(before_old, plain.loss(old_test))
    assert f_plain > 1.0 and adaptation(before_new, plain.loss(new_test)) > 1.0
    assert forgetting(before_old, replay.loss(old_test)) < 0.5 * f_plain
    assert forgetting(before_old, ewc.loss(old_test)) < f_plain
    # Replay pays on the new data: it fits B less well than continuing on B alone.
    assert replay.loss(new_test) > plain.loss(new_test)
