# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weight-normalization/ (ai://5PC0VR).
# Check of derivations/norm-growth.md: a plain gradient step v' = v - eta grad_v gives ||v'||^2 = ||v||^2 +
# eta^2 ||grad_v||^2, so ||v|| never decreases; the effective step on the direction shrinks as ||v|| grows; and a
# heavy-ball momentum step is not orthogonal to v in general.
import numpy as np


def grad_v(v, g, grad_w):
    n = np.linalg.norm(v)
    return g / n * (grad_w - (grad_w @ v) / n**2 * v)


def main():
    rng = np.random.default_rng(7)
    k, steps = 6, 400
    a = rng.normal(size=(50, k))
    target = a @ rng.normal(size=k)
    for eta in (0.01, 0.1, 1.0):
        v, g, norms = rng.normal(size=k), 1.5, []
        for _ in range(steps):
            w = g * v / np.linalg.norm(v)
            gw = a.T @ (a @ w - target) / len(a)
            gv = grad_v(v, g, gw)
            new = v - eta * gv
            assert np.isclose(new @ new, v @ v + eta**2 * (gv @ gv), rtol=1e-12)
            norms.append(np.linalg.norm(new))
            v = new
        assert all(b >= a_ - 1e-12 for a_, b in zip(norms, norms[1:]))
    # the ratio form: with c = ||delta v|| / ||v||, ||v'|| = sqrt(1 + c^2) ||v||
    v, gv = np.array([3.0, 4.0]), grad_v(np.array([3.0, 4.0]), 2.0, np.array([1.0, 0.0]))
    c = np.linalg.norm(gv) / 5.0
    assert np.isclose(np.linalg.norm(v - gv), np.sqrt(1 + c**2) * 5.0) and np.isclose(np.linalg.norm(gv), 0.32)
    # scaling v by s at fixed w scales grad_v by 1 / s, so the turn of the direction per step falls like 1 / s^2
    for s_ in (0.5, 2.0, 10.0):
        gw = rng.normal(size=k)
        v = rng.normal(size=k)
        assert np.allclose(grad_v(s_ * v, 1.5, gw), grad_v(v, 1.5, gw) / s_)
    # momentum: the velocity mixes old gradients, which are orthogonal to old v, not to the current v
    v, vel, g, worst = rng.normal(size=k), np.zeros(k), 1.5, 0.0
    for _ in range(50):
        w = g * v / np.linalg.norm(v)
        gv = grad_v(v, g, a.T @ (a @ w - target) / len(a))
        vel = 0.9 * vel - 0.5 * gv
        worst = max(worst, abs(vel @ v) / (np.linalg.norm(vel) * np.linalg.norm(v) + 1e-300))
        v = v + vel
    assert worst > 1e-3
    print("ok: each plain step adds eta^2 ||grad_v||^2 to ||v||^2; with momentum the step leaves the tangent plane")


if __name__ == "__main__":
    main()
