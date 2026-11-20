# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/consistency-model/ (ai://NEW-consistency-model).
"""Tests of the NumPy reference: the shared fixture, the counts, the hand-written gradients against finite
differences, the boundary condition, the closed-form consistency function of a Gaussian against the ODE, the exact
Euler step of consistency training, the toy of Proposition 1, the schedules, the progressive-distillation target, and
a model trained by distillation and one trained from scratch, sampled in one and two steps."""

import json
import math
import pathlib

import numpy as np
import pytest

import consistency_model as cm

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def close(a, b, rtol=1e-12, atol=1e-13, msg=""):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)


def three_gaussians(rng, n=6000):
    ang = np.array([0.0, 2 * math.pi / 3, 4 * math.pi / 3])
    means = 0.8 * np.stack([np.cos(ang), np.sin(ang)], 1)
    return means, 0.01, np.full(3, 1 / 3), means[rng.integers(0, 3, n)] + 0.1 * rng.standard_normal((n, 2))


def test_shared_fixture():
    P = dict(np.load(FIXTURES / "params-small.npz"))
    inp = dict(np.load(FIXTURES / "input-small.npz"))
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    p = {k: v for k, v in P.items() if not k.startswith("minus_")}
    pm = {k[6:]: v for k, v in P.items() if k.startswith("minus_")}
    cfg = cm.config(*(int(v) for v in inp["cfg"]))
    ts, n = cm.karras_steps(18), inp["n"]
    t_lo, t_hi = ts[n], ts[n + 1]
    x_hi, x_lo = cm.cd_pair(
        inp["x0"], inp["z"], t_lo, t_hi, lambda x, t: cm.score(x, t, inp["means"], 0.1, inp["weights"])
    )
    close(x_lo, want["x_lo"])
    close(cm.forward(p, cfg, x_hi, t_hi), want["f"])
    loss, grad = cm.consistency_loss(p, pm, cfg, x_hi, t_hi, x_lo, t_lo)
    assert loss == pytest.approx(float(want["cd_loss"]), rel=1e-12)
    for k, g in grad.items():
        close(g, want[f"cd_d{k}"], rtol=1e-10, msg=k)
    xc_hi, xc_lo = cm.ct_pair(inp["x0"], inp["z"], t_lo, t_hi)
    c = 0.00054 * math.sqrt(3.0)
    loss, grad = cm.consistency_loss(p, p, cfg, xc_hi, t_hi, xc_lo, t_lo, cm.ict_weight(t_lo, t_hi), c)
    assert loss == pytest.approx(float(want["ct_loss"]), rel=1e-12)
    for k, g in grad.items():
        close(g, want[f"ct_d{k}"], rtol=1e-10, msg=k)
    close(cm.ema(pm, p, 0.95)["W1"], want["ema"])
    close(cm.sample(lambda x, t: cm.forward(p, cfg, x, t), inp["xT"], [0.8], inp["noise"]), want["sample"])
    (a_t, s_t), (a_s, s_s), (a_s2, s_s2) = (cm.vp(v) for v in inp["lam"])
    z = inp["z_pd"]
    z_s2 = cm.ddim_step(cm.ddim_step(z, np.tanh(z), a_t, s_t, a_s, s_s), 0.5 * np.tanh(z), a_s, s_s, a_s2, s_s2)
    close(cm.pd_target(z, z_s2, a_t, s_t, a_s2, s_s2), want["pd_target"])


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = cm.config(k["dim"], k["hidden"], k["temb"])
    assert cm.param_count(cfg) == mod["params"]
    assert cm.flops_forward(cfg, k["B"]) == mod["flops_forward"]
    p = cm.init(np.random.default_rng(0), cfg)
    assert sum(v.size for v in p.values()) == mod["params"]
    assert cm.forward(p, cfg, np.zeros((k["B"], k["dim"])), 3.0).shape == (k["B"], k["dim"])


@pytest.mark.parametrize("c", [None, 0.3])
def test_gradient_by_finite_differences(c):
    rng = np.random.default_rng(1)
    cfg = cm.config(3, 4, 5)
    p = {k: v + 0.2 * rng.standard_normal(v.shape) for k, v in cm.init(rng, cfg).items()}
    pm = {k: v + 0.1 * rng.standard_normal(v.shape) for k, v in p.items()}
    t_hi, t_lo = np.array([0.01, 0.5, 3.0, 40.0]), np.array([0.005, 0.3, 2.0, 30.0])
    x_hi, x_lo = rng.standard_normal((4, 3)), rng.standard_normal((4, 3))
    w = np.array([1.0, 2.0, 0.5, 0.1])
    _, grad = cm.consistency_loss(p, pm, cfg, x_hi, t_hi, x_lo, t_lo, w, c)
    h = 1e-6
    for k in p:
        num = np.zeros_like(p[k])
        for i in np.ndindex(p[k].shape):
            q = {kk: vv.copy() for kk, vv in p.items()}
            q[k][i] += h
            up = cm.consistency_loss(q, pm, cfg, x_hi, t_hi, x_lo, t_lo, w, c)[0]
            q[k][i] -= 2 * h
            num[i] = (up - cm.consistency_loss(q, pm, cfg, x_hi, t_hi, x_lo, t_lo, w, c)[0]) / (2 * h)
        close(grad[k], num, rtol=1e-6, atol=1e-9, msg=k)


def test_boundary_condition():
    """f(x, eps) = x for every parameter value, since c_skip(eps) = 1 and c_out(eps) = 0."""
    rng = np.random.default_rng(2)
    assert cm.c_skip(cm.EPS) == 1.0 and cm.c_out(cm.EPS) == 0.0
    cfg = cm.config(4, 7, 6)
    for _ in range(3):
        p = {k: 5 * rng.standard_normal(v.shape) for k, v in cm.init(rng, cfg).items()}
        x = rng.standard_normal((5, 4))
        close(cm.forward(p, cfg, x, cm.EPS), x, rtol=0, atol=0)
    t = np.geomspace(cm.EPS, cm.T_MAX, 50)
    assert np.all(np.diff(cm.c_skip(t)) < 0) and np.all(np.diff(cm.c_out(t)) > 0)
    close(cm.c_skip(cm.T_MAX), 0.25 / (79.998**2 + 0.25))


def test_gaussian_consistency_function():
    """eq-gauss-cf against 400 Heun steps, to 1e-4, of the ODE, and self-consistency: every point of one path maps to its end."""
    rng = np.random.default_rng(3)
    mu, s2 = np.array([[0.5, -1.0]]), 0.2

    def sf(x, t):
        return cm.score(x, t, mu, s2, np.ones(1))

    x = rng.standard_normal((6, 2)) * 5
    for t in (0.1, 1.0, 10.0, 80.0):
        xt = x * t / 5
        close(cm.solve_ode(xt, t, cm.EPS, sf, 400), cm.gaussian_consistency(xt, t, mu, s2), rtol=1e-4, atol=1e-4)
    xT = 80.0 * rng.standard_normal((6, 2))
    path, ts = [xT], cm.karras_steps(400)[::-1]
    for a, b in zip(ts[:-1], ts[1:], strict=True):
        path.append(cm.teacher_step(path[-1], a, b, sf))
    ends = [cm.gaussian_consistency(xi, ti, mu, s2) for xi, ti in zip(path, ts, strict=True)]
    for e in ends:
        close(e, path[-1], rtol=1e-4, atol=1e-4)


def test_ct_euler_step_is_exact():
    """One Euler step of the ODE with the one-sample score estimate -(x_t - x) / t^2 from x + t_hi z lands on x + t_lo z,
    so the consistency-training pair is the distillation pair of that estimate."""
    rng = np.random.default_rng(4)
    x, z = rng.standard_normal((5, 3)), rng.standard_normal((5, 3))
    t_lo, t_hi = np.array([0.002, 0.1, 1.0, 5.0, 60.0]), np.array([0.01, 0.4, 1.7, 9.0, 80.0])

    def sf(xt, t):
        return -(xt - x) / cm._col(t, len(xt)) ** 2

    x_hi, x_lo = cm.cd_pair(x, z, t_lo, t_hi, sf, "euler")
    c_hi, c_lo = cm.ct_pair(x, z, t_lo, t_hi)
    close(x_hi, c_hi)
    close(x_lo, c_lo, rtol=1e-12, atol=1e-12)


def test_proposition_1_toy():
    """For data at xi and a model that is exact at theta = xi: with theta- = theta the gradient of the CT loss vanishes
    only at xi; with theta- frozen at another value the loss tends to E[(1 - smin / t)^2] (theta - theta-)^2 and so
    pulls theta to theta-, whatever the data."""
    rng = np.random.default_rng(5)
    ts, z = np.linspace(0.1, 1.0, 4001), rng.standard_normal((1, 2000))
    xi, smin = 0.7, ts[0]
    loss, grad = cm.toy_ct_loss(xi, xi, xi, ts, z)
    assert loss < 1e-30 and abs(grad) < 1e-15
    for theta in (-0.4, 1.3):
        _, g = cm.toy_ct_loss(theta, theta, xi, ts, z)
        lo, hi = ts[:-1], ts[1:]
        assert g == pytest.approx((2 * smin * (1 / lo - 1 / hi) * (1 - smin / hi) * (theta - xi)).mean(), rel=1e-9)
        limit = (2 * smin / hi**2 * (1 - smin / hi) * (theta - xi)).mean() * (ts[1] - ts[0])
        assert g == pytest.approx(limit, rel=1e-3)
        assert np.sign(g) == np.sign(theta - xi)
        loss, g = cm.toy_ct_loss(theta, 0.0, xi, ts, z)
        k = ((1 - smin / ts[1:]) ** 2).mean()
        assert loss == pytest.approx(k * theta**2, rel=1e-2)
        assert np.sign(g) == np.sign(theta)


def test_schedules():
    assert cm.cm_schedule(0, 800_000) == (2, pytest.approx(0.9))
    N, mu = cm.cm_schedule(800_000, 800_000)
    assert N == 151 and mu == pytest.approx(math.exp(2 * math.log(0.9) / 151))
    Ns = [cm.cm_schedule(k, 1000)[0] for k in range(1001)]
    assert all(b >= a for a, b in zip(Ns, Ns[1:], strict=False))
    steps = [cm.ict_steps(k, 400_000) for k in range(0, 400_000, 1000)]
    assert steps[0] == 11 and steps[-1] == 1281 and sorted(set(steps)) == [11, 21, 41, 81, 161, 321, 641, 1281]
    ts = cm.karras_steps(18)
    assert ts[0] == pytest.approx(cm.EPS) and ts[-1] == pytest.approx(cm.T_MAX) and np.all(np.diff(ts) > 0)
    q = cm.lognormal_probs(ts)
    assert q.sum() == pytest.approx(1.0) and np.all(q > 0)
    mid = np.sqrt(ts[:-1] * ts[1:])
    assert abs(np.log(mid[q.argmax()]) + 1.1) < 1.0
    close(cm.ict_weight(ts[:-1], ts[1:]) * np.diff(ts), np.ones(17))


def test_progressive_distillation_target():
    """One student DDIM step from z_t with the target x-prediction lands where two teacher steps land."""
    rng = np.random.default_rng(6)
    z = rng.standard_normal((5, 4))
    for lam in ([2.0, 1.0, 0.0], [-1.0, -2.0, -3.0], [6.0, 5.5, 4.0]):
        (a_t, s_t), (a_s, s_s), (a_s2, s_s2) = (cm.vp(v) for v in lam)
        z_s = cm.ddim_step(z, np.sin(z), a_t, s_t, a_s, s_s)
        z_s2 = cm.ddim_step(z_s, np.cos(z_s), a_s, s_s, a_s2, s_s2)
        x_tilde = cm.pd_target(z, z_s2, a_t, s_t, a_s2, s_s2)
        close(cm.ddim_step(z, x_tilde, a_t, s_t, a_s2, s_s2), z_s2, rtol=1e-10, atol=1e-12)
    a, s = cm.vp(np.array([10.0, 0.0, -10.0]))
    close(a**2 + s**2, np.ones(3))


def stats(x, means):
    d = np.sqrt(((x[:, None] - means[None]) ** 2).sum(-1))
    return float((d.min(1) < 0.3).mean()), np.bincount(d.argmin(1), minlength=3) / len(x)


def test_distillation_and_training():
    rng = np.random.default_rng(7)
    means, s2, w, x0 = three_gaussians(rng)

    def sf(x, t):
        return cm.score(x, t, means, s2, w)

    cfg = cm.config(2, 64, 8)
    xT = 80.0 * np.random.default_rng(5).standard_normal((2000, 2))
    ref = cm.solve_ode(xT, cm.T_MAX, cm.EPS, sf, 200)
    near_ref, _ = stats(ref, means)
    assert near_ref > 0.98
    p0 = cm.init(np.random.default_rng(1), cfg)
    err0 = np.sqrt(((cm.sample(lambda x, t: cm.forward(p0, cfg, x, t), xT) - ref) ** 2).sum(1)).mean()

    p, pm, trace = cm.train_cd(cm.init(rng, cfg), cfg, x0, sf, 3000, rng, lr=3e-3)
    one = cm.sample(lambda x, t: cm.forward(pm, cfg, x, t), xT)
    err = np.sqrt(((one - ref) ** 2).sum(1)).mean()
    near, props = stats(one, means)
    assert err < 0.2 < err0 and near > 0.65 and np.abs(props - 1 / 3).max() < 0.06

    p, trace = cm.train_ct(cm.init(rng, cfg), cfg, x0, 3000, rng, lr=3e-3)
    noise = np.random.default_rng(6).standard_normal((1, 2000, 2))
    one = cm.sample(lambda x, t: cm.forward(p, cfg, x, t), xT)
    two = cm.sample(lambda x, t: cm.forward(p, cfg, x, t), xT, [0.8], noise)
    (n1, props1), (n2, _) = stats(one, means), stats(two, means)
    assert n1 > 0.25 and n2 > max(n1, 0.6) and np.abs(props1 - 1 / 3).max() < 0.1


def test_worked_example():
    """The numbers of the worked example on the page: one Gaussian N(0, 0.5^2), a sample at T = 80 and at t = 1, the
    output the network must produce, a consistency training pair, and the Pseudo-Huber metric at c = 0.03."""
    mu, s2 = np.array([0.0]), 0.25
    assert cm.gaussian_consistency(np.array([[40.0]]), 80.0, mu, s2)[0, 0] == pytest.approx(0.25, abs=3e-6)
    assert cm.c_skip(80.0) == pytest.approx(3.906e-5, rel=1e-3) and cm.c_out(80.0) == pytest.approx(0.49998, rel=1e-5)
    assert (0.25 - cm.c_skip(80.0) * 40.0) / cm.c_out(80.0) == pytest.approx(0.4969, abs=1e-4)
    f1 = cm.gaussian_consistency(np.array([[1.0]]), 1.0, mu, s2)[0, 0]
    assert f1 == pytest.approx(0.4472, abs=1e-4)
    assert cm.c_skip(1.0) == pytest.approx(0.2006, abs=1e-4) and cm.c_out(1.0) == pytest.approx(0.4463, abs=1e-4)
    assert (f1 - cm.c_skip(1.0)) / cm.c_out(1.0) == pytest.approx(0.5525, abs=1e-4)
    x_hi, x_lo = cm.ct_pair(np.array([[0.3]]), np.array([[1.0]]), 1.0, 2.0)
    assert x_hi[0, 0] == pytest.approx(2.3) and x_lo[0, 0] == pytest.approx(1.3)
    euler = cm.teacher_step(x_hi, 2.0, 1.0, lambda x, t: -(x - 0.3) / np.asarray(t) ** 2, "euler")
    assert euler[0, 0] == pytest.approx(1.3)
    d = [cm.metric(np.array([[r]]), np.zeros((1, 1)), 0.03)[0] for r in (0.01, 0.1, 1.0)]
    assert d == pytest.approx([0.00162, 0.0744, 0.9704], abs=1e-4)
