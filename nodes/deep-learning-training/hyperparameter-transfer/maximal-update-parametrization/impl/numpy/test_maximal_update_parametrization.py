# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/maximal-update-parametrization/ (ai://4VJECR).
"""Tests of the NumPy reference: the per-layer initialization std and learning rates of muP for SGD and Adam against
Table 3 of Yang et al. (2022) with the base-width constants of their Eq. 4, SP for contrast, the agreement of the two
at the base width, the 1/d attention scale, the gradients and the SGD and Adam steps of the MLP, the feature-update
size that muP keeps constant across widths and SP does not, and the worked example of the page."""

import numpy as np
import pytest

from maximal_update_parametrization import (
    MLP,
    attention_logits,
    feature_change,
    layer_hparams,
)

rng = np.random.default_rng(11)


def test_worked_example_adam():
    hp = layer_hparams(4096, 256, 0.01, 32, optimizer="adam", param="mup")
    assert hp["input"]["std"] == pytest.approx(1 / np.sqrt(32))
    assert hp["input"]["lr"] == pytest.approx(0.01)
    assert hp["hidden"]["std"] == pytest.approx(0.015625)
    assert hp["hidden"]["lr"] == pytest.approx(0.000625)
    assert hp["output"]["std"] == pytest.approx(0.00390625)
    assert hp["output"]["lr"] == pytest.approx(0.000625)
    assert hp["bias"] == {"std": 0.0, "lr": pytest.approx(0.01)}


def test_worked_example_sgd():
    hp = layer_hparams(4096, 256, 0.01, 32, optimizer="sgd", param="mup")
    assert hp["input"]["lr"] == pytest.approx(0.16)
    assert hp["hidden"]["lr"] == pytest.approx(0.01)
    assert hp["output"]["lr"] == pytest.approx(0.000625)
    assert hp["bias"]["lr"] == pytest.approx(0.16)
    assert hp["output"]["std"] == pytest.approx(0.00390625)
    assert hp["hidden"]["std"] == pytest.approx(0.015625)


def test_worked_example_sp():
    for opt in ("sgd", "adam"):
        hp = layer_hparams(4096, 256, 0.01, 32, optimizer=opt, param="sp")
        assert [hp[k]["lr"] for k in ("input", "hidden", "output", "bias")] == [0.01] * 4
        assert hp["output"]["std"] == pytest.approx(0.015625)
        assert hp["hidden"]["std"] == pytest.approx(0.015625)
        assert hp["input"]["std"] == pytest.approx(1 / np.sqrt(32))
        assert hp["bias"]["std"] == 0.0


def test_mup_equals_sp_at_the_base_width():
    for opt in ("sgd", "adam"):
        for n in (8, 100, 512):
            assert layer_hparams(n, n, 0.02, 7, 1.3, opt, "mup") == layer_hparams(n, n, 0.02, 7, 1.3, opt, "sp")


def test_width_exponents():
    """Doubling the width over the base: Table 3 with fan_in = width and fan_out = width."""
    for opt, exps in (
        ("sgd", {"input": 1, "hidden": 0, "output": -1, "bias": 1}),
        ("adam", {"input": 0, "hidden": -1, "output": -1, "bias": 0}),
    ):
        a = layer_hparams(1000, 10, 0.1, 3, optimizer=opt)
        b = layer_hparams(2000, 10, 0.1, 3, optimizer=opt)
        for k, e in exps.items():
            assert b[k]["lr"] / a[k]["lr"] == pytest.approx(2.0**e)
        assert (b["output"]["std"] / a["output"]["std"]) ** 2 == pytest.approx(0.25)
        assert (b["hidden"]["std"] / a["hidden"]["std"]) ** 2 == pytest.approx(0.5)
        assert b["input"]["std"] == a["input"]["std"]


def test_sigma_scales_every_std():
    a = layer_hparams(300, 30, 0.1, 4, sigma=1.0)
    b = layer_hparams(300, 30, 0.1, 4, sigma=2.5)
    for k in ("input", "hidden", "output"):
        assert b[k]["std"] == pytest.approx(2.5 * a[k]["std"])
        assert b[k]["lr"] == a[k]["lr"]


def test_invalid_arguments():
    for args in ((0, 8, 0.1, 3), (8, 0, 0.1, 3), (8, 8, 0.1, 0), (-4, 8, 0.1, 3)):
        with pytest.raises(ValueError):
            layer_hparams(*args)
    with pytest.raises(ValueError):
        layer_hparams(8, 8, 0.1, 3, optimizer="rmsprop")
    with pytest.raises(ValueError):
        layer_hparams(8, 8, 0.1, 3, param="ntk")
    with pytest.raises(ValueError):
        attention_logits(np.ones((2, 4)), np.ones((2, 4)), param="ntk")
    with pytest.raises(ValueError):
        attention_logits(np.ones((2, 4)), np.ones((2, 3)))


def test_attention_logits():
    q = rng.standard_normal((3, 256))
    k = rng.standard_normal((5, 256))
    raw = q @ k.T
    np.testing.assert_allclose(attention_logits(q, k, base_d_head=64), raw / 32)
    np.testing.assert_allclose(attention_logits(q, k, param="sp"), raw / 16)
    np.testing.assert_allclose(attention_logits(q, k, base_d_head=64, alpha=3.0), 3 * raw / 32)
    np.testing.assert_allclose(attention_logits(q, k, alpha=2.0, param="sp"), raw / 8)
    np.testing.assert_allclose(attention_logits(q, k), attention_logits(q, k, param="sp"))
    np.testing.assert_allclose(attention_logits(q, k, base_d_head=256), raw / 16)
    qb = rng.standard_normal((2, 3, 3, 8))
    kb = rng.standard_normal((2, 3, 4, 8))
    out = attention_logits(qb, kb, base_d_head=2)
    assert out.shape == (2, 3, 3, 4)
    np.testing.assert_allclose(out[1, 2], np.sqrt(2) / 8 * qb[1, 2] @ kb[1, 2].T)


def test_attention_of_aligned_vectors_stays_bounded_under_mup():
    vals_mup, vals_sp = [], []
    for d in (64, 1024, 16384):
        q = rng.standard_normal((1, d))
        vals_mup.append(float(attention_logits(q, q, base_d_head=64)[0, 0]))
        vals_sp.append(float(attention_logits(q, q, param="sp")[0, 0]))
    assert vals_mup[2] / vals_mup[0] < 1.2
    assert vals_sp[2] / vals_sp[0] > 12


def test_initialization_matches_the_hparams():
    net = MLP(20, 512, 3, 64, 0.01, sigma=1.5, optimizer="adam", seed=4)
    hp = net.hp
    assert np.std(net.params["U"]) == pytest.approx(hp["input"]["std"], rel=0.05)
    assert np.std(net.params["W"]) == pytest.approx(hp["hidden"]["std"], rel=0.01)
    assert np.std(net.params["V"]) == pytest.approx(hp["output"]["std"], rel=0.1)
    assert np.all(net.params["b1"] == 0) and np.all(net.params["b2"] == 0)
    assert net.params["U"].shape == (512, 20) and net.params["W"].shape == (512, 512)
    assert net.params["V"].shape == (3, 512)
    assert net.lr == {
        "U": hp["input"]["lr"],
        "b1": hp["bias"]["lr"],
        "W": hp["hidden"]["lr"],
        "b2": hp["bias"]["lr"],
        "V": hp["output"]["lr"],
    }


def test_forward_and_loss():
    net = MLP(4, 16, 2, 8, 0.1, seed=1)
    X = rng.standard_normal((5, 4))
    p = net.params
    p["b1"] += 0.1
    p["b2"] -= 0.2
    f, (h1, x1, h2, x2) = net.forward(X)
    np.testing.assert_allclose(h1, X @ p["U"].T + p["b1"])
    np.testing.assert_allclose(x2, np.tanh(np.tanh(h1) @ p["W"].T + p["b2"]))
    np.testing.assert_allclose(f, x2 @ p["V"].T)
    y = rng.standard_normal((5, 2))
    assert net.loss(X, y) == pytest.approx(0.5 * np.sum((f - y) ** 2) / 5)


def test_gradients_match_finite_differences():
    net = MLP(3, 6, 2, 3, 0.1, seed=2)
    for name in ("b1", "b2"):
        net.params[name] += rng.standard_normal(6) * 0.3
    X = rng.standard_normal((4, 3))
    y = rng.standard_normal((4, 2))
    g = net.grads(X, y)
    h = 1e-6
    for name, p in net.params.items():
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[i]
            p[i] = old + h
            up = net.loss(X, y)
            p[i] = old - h
            down = net.loss(X, y)
            p[i] = old
            assert (up - down) / (2 * h) == pytest.approx(g[name][i], abs=1e-7)


def test_sgd_step_uses_the_per_layer_rates():
    net = MLP(3, 32, 1, 8, 0.05, optimizer="sgd", seed=3)
    X = rng.standard_normal((4, 3))
    y = np.ones((4, 1))
    before = {k: v.copy() for k, v in net.params.items()}
    g = net.grads(X, y)
    loss = net.loss(X, y)
    assert net.step(X, y) == pytest.approx(loss)
    for k in before:
        np.testing.assert_allclose(net.params[k], before[k] - net.lr[k] * g[k])
    assert net.lr["U"] == pytest.approx(0.2) and net.lr["V"] == pytest.approx(0.0125)


def test_adam_steps_match_the_update_rule():
    net = MLP(3, 16, 1, 8, 0.01, optimizer="adam", seed=5, betas=(0.8, 0.95), eps=1e-6)
    X = rng.standard_normal((4, 3))
    y = np.ones((4, 1))
    p0 = {k: v.copy() for k, v in net.params.items()}
    g1 = net.grads(X, y)
    net.step(X, y)
    for k in p0:
        np.testing.assert_allclose(net.params[k], p0[k] - net.lr[k] * g1[k] / (np.abs(g1[k]) + 1e-6))
    p1 = {k: v.copy() for k, v in net.params.items()}
    g2 = net.grads(X, y)
    net.step(X, y)
    for k in p0:
        m = (0.8 * 0.2 * g1[k] + 0.2 * g2[k]) / (1 - 0.8**2)
        v = (0.95 * 0.05 * g1[k] ** 2 + 0.05 * g2[k] ** 2) / (1 - 0.95**2)
        np.testing.assert_allclose(
            net.params[k],
            p1[k] - net.lr[k] * m / (np.sqrt(v) + 1e-6),
            rtol=1e-10,
            atol=1e-14,
        )
    assert net.t == 2


def test_training_reduces_the_loss():
    X = rng.standard_normal((16, 5))
    y = np.sin(X[:, :1])
    for opt, lr in (("sgd", 0.05), ("adam", 0.01)):
        net = MLP(5, 128, 1, 32, lr, optimizer=opt, seed=6)
        first = net.step(X, y)
        for _ in range(60):
            last = net.step(X, y)
        assert last < 0.5 * first


def _sizes(param, opt, lr, widths=(64, 256, 1024)):
    X = np.random.default_rng(0).standard_normal((8, 16))
    y = np.ones((8, 1))
    out = []
    for n in widths:
        out.append(
            np.mean(
                [feature_change(MLP(16, n, 1, 64, lr, optimizer=opt, param=param, seed=s), X, y) for s in range(3)],
                axis=0,
            )
        )
    return np.array(out)


def test_mup_keeps_the_feature_update_constant_and_sp_does_not():
    for opt, lr in (("sgd", 0.05), ("adam", 0.003)):
        mup = _sizes("mup", opt, lr)
        sp = _sizes("sp", opt, lr)
        np.testing.assert_allclose(mup[0], sp[0])  # identical at the base width
        assert 0.7 < mup[-1, 0] / mup[0, 0] < 1.4
        assert 0.7 < mup[-1, 1] / mup[0, 1] < 1.4
        assert sp[-1, 1] / sp[0, 1] > 8  # the output grows like the width
        assert sp[-1, 0] / sp[0, 0] > (2 if opt == "sgd" else 10)


def test_feature_change_is_the_rms_change():
    net = MLP(4, 8, 1, 8, 0.1, optimizer="sgd", seed=7)
    X = rng.standard_normal((3, 4))
    y = np.ones((3, 1))
    twin = MLP(4, 8, 1, 8, 0.1, optimizer="sgd", seed=7)
    f0, (_, _, h0, _) = twin.forward(X)
    twin.step(X, y)
    f1, (_, _, h1, _) = twin.forward(X)
    dh, df = feature_change(net, X, y)
    assert dh == pytest.approx(np.sqrt(np.mean((h1 - h0) ** 2)))
    assert df == pytest.approx(np.sqrt(np.mean((f1 - f0) ** 2)))
