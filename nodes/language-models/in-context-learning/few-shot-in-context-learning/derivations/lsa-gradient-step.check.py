# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/few-shot-in-context-learning/ (ai://NEW-few-shot-in-context-learning).
# Check of derivations/lsa-gradient-step.md: one linear self-attention layer with the weights of von Oswald et al.
# is one gradient step on the in-context least-squares loss, K layers are K steps, the query read-out, the
# rescaling of the weights, and the averaging estimate at W_0 = 0, eta = 1.
import numpy as np


def weights(dx, dy, eta, n, w0):
    d = dx + dy
    wk = np.zeros((d, d))
    wk[:dx, :dx] = np.eye(dx)
    wv = np.zeros((d, d))
    wv[dx:, :dx] = w0
    wv[dx:, dx:] = -np.eye(dy)
    return wk, wk.copy(), wv, (eta / n) * np.eye(d)


def layer(ctx, tokens, wk, wq, wv, p):
    # e <- e + P W_V sum_i e_i e_i^T W_K^T W_Q e, keys and values from the context tokens only.
    return tokens + (p @ wv @ ctx.T @ ctx @ wk.T @ wq @ tokens.T).T


def gd(x, y, w, eta):
    n = len(x)
    return w - (eta / n) * (w @ x.T - y.T) @ x


def main():
    rng = np.random.default_rng(20261116)
    for dx, dy, n, eta in [(1, 1, 3, 0.1), (5, 1, 10, 0.7), (4, 3, 7, 1.3), (10, 2, 40, 0.05)]:
        x = rng.normal(size=(n, dx))
        y = rng.normal(size=(n, dy))
        w0 = rng.normal(size=(dy, dx))
        xq = rng.normal(size=(1, dx))
        wk, wq, wv, p = weights(dx, dy, eta, n, w0)
        ctx = np.hstack([x, y])
        q = np.hstack([xq, -xq @ w0.T])
        # 1. One layer: context tokens become (x_j, y_j - Delta W x_j).
        w1 = gd(x, y, w0, eta)
        out = layer(ctx, ctx, wk, wq, wv, p)
        assert np.allclose(out[:, :dx], x, atol=1e-12)
        assert np.allclose(out[:, dx:], y - x @ (w1 - w0).T, atol=1e-10)
        # 2. Read-out: the query becomes (x_q, -W_1 x_q).
        oq = layer(ctx, q, wk, wq, wv, p)
        assert np.allclose(oq, np.hstack([xq, -xq @ w1.T]), atol=1e-10)
        # 3. K layers are K steps.
        c, t, w = ctx.copy(), q.copy(), w0.copy()
        for _ in range(5):
            c, t = layer(c, c, wk, wq, wv, p), layer(c, t, wk, wq, wv, p)
            w = gd(x, y, w, eta)
            assert np.allclose(c[:, dx:], y - x @ (w - w0).T, atol=1e-9)
            assert np.allclose(-t[:, dx:], xq @ w.T, atol=1e-9)
        # 4. Rescaling: (s P W_V, W_K^T W_Q / s) gives the same layer.
        for s in [-3.0, 0.5, 7.0]:
            o2 = layer(ctx, q, np.eye(dx + dy), wk.T @ wq / s, s * p @ wv, np.eye(dx + dy))
            assert np.allclose(o2, oq, atol=1e-10)
        # Letting the query act as a key and value changes the output unless W_0 = 0,
        # because then the query's value (0, W_0 x_q + W_0 x_q) is non-zero.
        both = np.vstack([ctx, q])
        o3 = layer(both, q, wk, wq, wv, p)
        assert not np.allclose(o3, oq)
        wk0, wq0, wv0, p0 = weights(dx, dy, eta, n, np.zeros((dy, dx)))
        q0 = np.hstack([xq, np.zeros((1, dy))])
        assert np.allclose(layer(np.vstack([ctx, q0]), q0, wk0, wq0, wv0, p0), layer(ctx, q0, wk0, wq0, wv0, p0), atol=1e-12)
        # 5. Averaging: W_0 = 0, eta = 1 predicts (1/N) sum_i y_i x_i^T x_q.
        wk, wq, wv, p = weights(dx, dy, 1.0, n, np.zeros((dy, dx)))
        pred = -layer(ctx, q0, wk, wq, wv, p)[:, dx:]
        assert np.allclose(pred, (xq @ x.T @ y) / n, atol=1e-12)
    print("lsa-gradient-step: ok")


if __name__ == "__main__":
    main()
