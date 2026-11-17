# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/constitutional-ai/ (ai://20KVDN).
# Check of derivations/ai-preference-label.md: the AI preference label as a logistic function of the log-odds, the
# soft-label loss and its minimizer, hard labels, clamping, the average over principles, the average over the two
# orders of the options, and the numbers of the worked example.
import numpy as np


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def soft_loss(d, p):
    return -p * log_sigmoid(d) - (1.0 - p) * log_sigmoid(-d)


def entropy(p):
    return -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)


def swap_average(d, b):
    return 0.5 * (sigmoid(d + b) + sigmoid(d - b))


def main():
    rng = np.random.default_rng(20261116)
    h = 1e-5
    # 1. The two-way normalized probability is sigma(l_A - l_B), and a common shift changes nothing.
    la, lb = rng.normal(-3, 2, 1000), rng.normal(-3, 2, 1000)
    p = np.exp(la) / (np.exp(la) + np.exp(lb))
    assert np.allclose(p, sigmoid(la - lb), rtol=1e-12)
    c = rng.normal(0, 5, 1000)
    assert np.allclose(np.exp(la + c) / (np.exp(la + c) + np.exp(lb + c)), p, rtol=1e-10)
    # 2. L_p'(D) = sigma(D) - p and L_p''(D) = sigma(D) sigma(-D) > 0, by central differences.
    d = rng.normal(0, 3, 1000)
    q = rng.uniform(0.01, 0.99, 1000)
    g = (soft_loss(d + h, q) - soft_loss(d - h, q)) / (2 * h)
    assert np.allclose(g, sigmoid(d) - q, atol=1e-8)
    h2 = 1e-4
    curv = (soft_loss(d + h2, q) - 2 * soft_loss(d, q) + soft_loss(d - h2, q)) / h2**2
    assert np.allclose(curv, sigmoid(d) * sigmoid(-d), atol=1e-6)
    assert np.all(sigmoid(d) * sigmoid(-d) > 0)
    # 3. The minimizer is log(p / (1 - p)) = l_A - l_B, and the minimum is the binary entropy H(p).
    grid = np.linspace(-12, 12, 240001)
    for pk in [0.03, 0.25, 0.5, 0.75, 0.9, 0.99]:
        star = np.log(pk / (1 - pk))
        assert abs(grid[np.argmin(soft_loss(grid, pk))] - star) < 2e-4
        assert np.isclose(soft_loss(star, pk), entropy(pk), rtol=1e-12)
    assert np.allclose(np.log(p / (1 - p)), la - lb, atol=1e-9)
    # 4. A hard label p = 1: the loss falls strictly everywhere and has no minimizer.
    ls = soft_loss(grid, 1.0)
    assert np.all(np.diff(ls) < 0) and ls[-1] > 0
    # 5. Clamping p to [a, 1 - a] bounds |D*| by log((1 - a) / a): log 1.5 for 40-60%, log 4 for 20-80%.
    for a, bound in [(0.4, np.log(1.5)), (0.2, np.log(4.0))]:
        assert np.isclose(np.log((1 - a) / a), bound, rtol=1e-15)
        pc = np.clip(rng.uniform(0, 1, 1000), a, 1 - a)
        assert np.all(np.abs(np.log(pc / (1 - pc))) <= bound + 1e-12)
    # 6. The loss is affine in p, so the expected loss over principles is the loss of the averaged label.
    pk = rng.uniform(0.05, 0.95, 16)
    wk = rng.dirichlet(np.ones(16))
    pbar = wk @ pk
    exp_loss = sum(w * soft_loss(d, x) for w, x in zip(wk, pk))
    assert np.allclose(exp_loss, soft_loss(d, pbar), rtol=1e-12)
    assert abs(grid[np.argmin(sum(w * soft_loss(grid, x) for w, x in zip(wk, pk)))] - np.log(pbar / (1 - pbar))) < 2e-4
    # 7. The average over the two orders is even in b and equals 1/2 at d = 0.
    bs = rng.normal(0, 4, 1000)
    ds = rng.normal(0, 3, 1000)
    assert np.allclose(swap_average(ds, bs), swap_average(ds, -bs), rtol=1e-14)
    assert np.allclose(swap_average(0.0, bs), 0.5, atol=1e-15)
    # 8. For d > 0 it falls strictly in |b| from sigma(d), stays above 1/2 and tends to 1/2.
    for dk in [0.05, 0.5, 1.0, 3.0]:
        b = np.linspace(0, 15, 1501)
        f = swap_average(dk, b)
        assert np.isclose(f[0], sigmoid(dk), rtol=1e-15)
        assert np.all(np.diff(f) < 0) and np.all(f > 0.5)
        assert abs(swap_average(dk, 40.0) - 0.5) < 1e-6
    # The worked example.
    pa = 0.45 / (0.45 + 0.15)
    assert np.isclose(pa, 0.75)
    assert np.isclose(np.log(pa / (1 - pa)), np.log(3.0))
    assert round(float(np.log(3.0)), 4) == 1.0986
    assert round(float(soft_loss(0.0, pa)), 4) == 0.6931
    assert np.isclose(sigmoid(0.0) - pa, -0.25)
    assert round(float(soft_loss(np.log(3.0), pa)), 4) == 0.5623 and np.isclose(soft_loss(np.log(3.0), pa), entropy(0.75))
    assert np.isclose(np.clip(pa, 0.4, 0.6), 0.6) and round(float(np.log(1.5)), 4) == 0.4055
    assert np.isclose((0.9 + 0.6) / 2, 0.75)
    s, t = np.log(3.0), np.log(0.55 / 0.45)
    dd, bb = (s + t) / 2, (s - t) / 2
    assert round(float(t), 4) == 0.2007 and round(float(dd), 4) == 0.6496 and round(float(bb), 4) == 0.4490
    assert np.isclose(swap_average(dd, bb), 0.65) and round(float(sigmoid(dd)), 4) == 0.6569
    assert 0.5 < swap_average(dd, bb) < sigmoid(dd)
    print("ok: the label as a logistic of the log-odds, the soft-label loss, its minimizer and entropy, hard labels, "
          "clamping, the average over principles, the average over orders, and the worked example")


if __name__ == "__main__":
    main()
