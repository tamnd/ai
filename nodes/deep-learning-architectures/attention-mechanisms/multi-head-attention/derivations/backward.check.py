# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multi-head-attention/ (ai://NEW-multi-head-attention).
# Check of derivations/backward.md: the gradients of multi-head attention against finite differences, the zero
# row sums of the score gradient, and the zero gradient of the masked scores.
import numpy as np


def forward(P, X, M, mask, H, dk, dv):
    T, S = X.shape[0], M.shape[0]
    heads, keep = [], []
    for r in range(H):
        Q = X @ P["Wq"][:, r * dk : (r + 1) * dk]
        K = M @ P["Wk"][:, r * dk : (r + 1) * dk]
        V = M @ P["Wv"][:, r * dv : (r + 1) * dv]
        Z = np.where(mask, Q @ K.T / np.sqrt(dk), -np.inf)
        E = np.exp(Z - Z.max(axis=1, keepdims=True))
        A = E / E.sum(axis=1, keepdims=True)
        heads.append(A @ V)
        keep.append((Q, K, V, A))
    C = np.concatenate(heads, axis=1)
    assert C.shape == (T, H * dv) and S == M.shape[0]
    return C @ P["Wo"], C, keep


def num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + h
        up = f()
        x[i] = old - h
        down = f()
        x[i] = old
        g[i] = (up - down) / (2 * h)
    return g


def main():
    rng = np.random.default_rng(20170612)
    for H, dk, dv, cross in [(1, 3, 3, False), (2, 2, 3, False), (3, 2, 1, True), (2, 4, 2, True)]:
        Md, T = 5, 4
        S = 3 if cross else T
        P = {"Wq": rng.normal(size=(Md, H * dk)), "Wk": rng.normal(size=(Md, H * dk)),
             "Wv": rng.normal(size=(Md, H * dv)), "Wo": rng.normal(size=(H * dv, Md))}
        X = rng.normal(size=(T, Md))
        Mem = rng.normal(size=(S, Md)) if cross else X
        mask = np.tile(np.arange(S) < 2, (T, 1)) if cross else np.tril(np.ones((T, T), bool))
        Ybar = rng.normal(size=(T, Md))
        Y, C, keep = forward(P, X, Mem, mask, H, dk, dv)
        g = {"Wo": C.T @ Ybar, "Wq": np.zeros_like(P["Wq"]), "Wk": np.zeros_like(P["Wk"]), "Wv": np.zeros_like(P["Wv"])}
        Cbar = Ybar @ P["Wo"].T
        Xbar, Mbar = np.zeros_like(X), np.zeros_like(Mem)
        for r, (Q, K, V, A) in enumerate(keep):
            Obar = Cbar[:, r * dv : (r + 1) * dv]
            Vbar, Abar = A.T @ Obar, Obar @ V.T
            Zbar = A * (Abar - np.sum(A * Abar, axis=1, keepdims=True))
            assert np.allclose(Zbar.sum(axis=1), 0, atol=1e-12)
            assert np.all(Zbar[~mask] == 0)
            Qbar, Kbar = Zbar @ K / np.sqrt(dk), Zbar.T @ Q / np.sqrt(dk)
            cq, cv = slice(r * dk, (r + 1) * dk), slice(r * dv, (r + 1) * dv)
            g["Wq"][:, cq], g["Wk"][:, cq], g["Wv"][:, cv] = X.T @ Qbar, Mem.T @ Kbar, Mem.T @ Vbar
            Xbar += Qbar @ P["Wq"][:, cq].T
            Mbar += Kbar @ P["Wk"][:, cq].T + Vbar @ P["Wv"][:, cv].T
        if not cross:
            Xbar, Mbar = Xbar + Mbar, None

        def loss():
            return float(np.sum(forward(P, X, X if not cross else Mem, mask, H, dk, dv)[0] * Ybar))

        for name in P:
            n = num_grad(loss, P[name])
            assert np.linalg.norm(g[name] - n) <= 1e-6 * (1 + np.linalg.norm(n)), (name, H, cross)
        n = num_grad(loss, X)
        assert np.linalg.norm(Xbar - n) <= 1e-6 * (1 + np.linalg.norm(n)), ("X", H, cross)
        if cross:
            n = num_grad(loss, Mem)
            assert np.linalg.norm(Mbar - n) <= 1e-6 * (1 + np.linalg.norm(n))
            assert np.all(Mbar[2:] == 0)
    print("ok: the gradients match finite differences, the score gradients sum to 0 by rows and vanish where masked")


if __name__ == "__main__":
    main()
