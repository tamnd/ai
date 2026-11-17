# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/attention-mechanism/ (ai://NEW-attention-mechanism).
"""The attention mechanism of Bahdanau, Cho and Bengio (2015) and the forms of Luong, Pham and Manning (2015), the
NumPy reference.

RNNsearch is the model of Bahdanau et al.'s Appendix A.2: a bidirectional GRU encoder whose two states at each
position, concatenated, are the annotations h_j (eq-annot); a GRU decoder whose state s_i is updated with a context
c_i that the alignment model recomputes at every step from s_{i-1} (eq-score, eq-weights, eq-context); and a deep
output with one maxout layer (eq-output). forward() runs it with teacher forcing and backward() is its gradient,
written out by hand. Rows are vectors, so a linear map is x @ W with W of shape (in, out), and the three gate
matrices of a GRU sit side by side in the column order (update z, reset r, candidate). As in the paper's equations
there are no biases. attention=False is the paper's RNNencdec: c_i is fixed to the two final encoder states.

The second half is the attention layer of Luong et al. on its own: the scores dot, general, concat and location
(eq-luong-scores), global attention, local-p attention (eq-local-p) and the attentional state (eq-luong-out).
"""

import numpy as np

GATES = 3  # z, r and the candidate state


def config(src_vocab=7, tgt_vocab=6, embed=3, hidden=4, align=5, maxout=3, attention=True):
    """m = embed, n = hidden, n' = align and l = maxout in the notation of Bahdanau et al.'s Appendix A.2.3."""
    return {"src_vocab": src_vocab, "tgt_vocab": tgt_vocab, "embed": embed, "hidden": hidden, "align": align,
            "maxout": maxout, "attention": attention}  # fmt: skip


def shapes(cfg):
    Kx, Ky, m, n, a, l = (cfg[k] for k in ("src_vocab", "tgt_vocab", "embed", "hidden", "align", "maxout"))
    g = GATES * n
    return {
        "E_x": (Kx, m), "fwd_W": (m, g), "fwd_U": (n, g), "bwd_W": (m, g), "bwd_U": (n, g),
        "E_y": (Ky, m), "dec_W": (m, g), "dec_U": (n, g), "dec_C": (2 * n, g), "W_s": (n, n),
        "W_a": (n, a), "U_a": (2 * n, a), "v_a": (a,),
        "U_o": (n, 2 * l), "V_o": (m, 2 * l), "C_o": (2 * n, 2 * l), "W_o": (l, Ky),
    }  # fmt: skip


def param_count(cfg):
    return sum(int(np.prod(s)) for s in shapes(cfg).values())


def init(rng, cfg, scale=0.01):
    """Bahdanau et al.'s B.1 draws W_a and U_a with standard deviation 0.001, v_a at 0 and the other matrices with
    standard deviation 0.01 (the recurrent ones orthogonal); scale stands in for 0.01 and v_a is drawn as well, so
    that small examples are not degenerate."""
    return {k: rng.normal(0.0, scale, s) for k, s in shapes(cfg).items()}


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def gru_step(x, h, W, U, c=None, C=None):
    """Cho et al.'s gated unit, Bahdanau et al.'s A.1.1: z and r from x, h (and c); the candidate from x, r * h (and
    c); the new state (1 - z) h + z candidate."""
    n = h.shape[-1]
    a = x @ W + (0.0 if c is None else c @ C)
    zr = sigmoid(a[:, : 2 * n] + h @ U[:, : 2 * n])
    z, r = zr[:, :n], zr[:, n:]
    cand = np.tanh(a[:, 2 * n :] + (r * h) @ U[:, 2 * n :])
    return (1.0 - z) * h + z * cand, (x, h, c, z, r, cand)


def gru_step_backward(dh_new, W, U, C, cache, grads, kW, kU, kC=None):
    """Add the gradients of W, U (and C) to grads and return those of x, h and c."""
    x, h, c, z, r, cand = cache
    n = h.shape[-1]
    dz = dh_new * (cand - h) * z * (1.0 - z)
    da_h = dh_new * z * (1.0 - cand**2)
    d_rh = da_h @ U[:, 2 * n :].T
    dr = d_rh * h * r * (1.0 - r)
    da = np.concatenate([dz, dr, da_h], -1)  # (B, 3n)
    grads[kW] += x.T @ da
    grads[kU][:, : 2 * n] += h.T @ da[:, : 2 * n]
    grads[kU][:, 2 * n :] += (r * h).T @ da_h
    dh = dh_new * (1.0 - z) + da[:, : 2 * n] @ U[:, : 2 * n].T + d_rh * r
    dc = None
    if kC is not None:
        grads[kC] += c.T @ da
        dc = da @ C.T
    return da @ W.T, dh, dc


def encode(params, src, src_mask):
    """eq-annot: the forward GRU reads x_1..x_S, the backward GRU x_S..x_1, and h_j = [fwd h_j; bwd h_j]. A padded
    position leaves the state as it was, so the backward GRU of a short sentence starts from 0 at its last word."""
    B, S = src.shape
    n = params["fwd_U"].shape[0]
    ex = params["E_x"][src]  # (B, S, m)
    m = src_mask[..., None].astype(float)  # (B, S, 1)
    Hf, Hb = np.zeros((B, S, n)), np.zeros((B, S, n))
    cf, cb = [None] * S, [None] * S
    h = np.zeros((B, n))
    for t in range(S):
        new, cf[t] = gru_step(ex[:, t], h, params["fwd_W"], params["fwd_U"])
        h = Hf[:, t] = m[:, t] * new + (1.0 - m[:, t]) * h
    h = np.zeros((B, n))
    for t in reversed(range(S)):
        new, cb[t] = gru_step(ex[:, t], h, params["bwd_W"], params["bwd_U"])
        h = Hb[:, t] = m[:, t] * new + (1.0 - m[:, t]) * h
    return np.concatenate([Hf, Hb], -1), (ex, m, cf, cb)


def align(params, s_prev, H, HU, src_mask):
    """eq-score and eq-weights: e_ij = v_a . tanh(s_{i-1} W_a + h_j U_a), alpha_i = softmax_j(e_i) over the real
    positions; eq-context: c_i = sum_j alpha_ij h_j. HU = H U_a does not depend on i and is computed once."""
    u = np.tanh((s_prev @ params["W_a"])[:, None] + HU)  # (B, S, n')
    e = np.where(src_mask, u @ params["v_a"], -np.inf)  # (B, S)
    w = np.exp(e - e.max(-1, keepdims=True))
    alpha = w / w.sum(-1, keepdims=True)
    return np.einsum("bs,bsk->bk", alpha, H), alpha, u


def forward(params, cfg, src, tgt, src_mask=None):
    """Teacher-forced logits (B, T, K_y) of p(y_i | y_<i, x) for i = 1..T, each from s_{i-1}, y_{i-1} and c_i, with
    y_0 an all-zero embedding. Also returns the cache for backward and the alignments alpha (B, T, S)."""
    src_mask = np.ones(src.shape, bool) if src_mask is None else src_mask
    B, T = tgt.shape
    H, enc = encode(params, src, src_mask)
    n = params["W_s"].shape[0]
    HU = H @ params["U_a"]
    s = np.tanh(H[:, 0, n:] @ params["W_s"])  # s_0 = tanh(W_s bwd h_1)
    ey = np.concatenate([np.zeros((B, 1, params["E_y"].shape[1])), params["E_y"][tgt[:, :-1]]], 1)  # (B, T, m)
    fixed = np.concatenate([H[:, -1, :n], H[:, 0, n:]], -1)  # RNNencdec: the last forward and first backward states
    logits, alphas, steps = [], [], []
    for i in range(T):
        if cfg["attention"]:
            c, alpha, u = align(params, s, H, HU, src_mask)
        else:
            c, alpha, u = fixed, np.zeros(src.shape), None
        tt = s @ params["U_o"] + ey[:, i] @ params["V_o"] + c @ params["C_o"]  # (B, 2l)
        pick = tt[:, 0::2] >= tt[:, 1::2]
        t_max = np.where(pick, tt[:, 0::2], tt[:, 1::2])  # maxout over pairs
        logits.append(t_max @ params["W_o"])
        s_new, gc = gru_step(ey[:, i], s, params["dec_W"], params["dec_U"], c, params["dec_C"])
        steps.append((s, c, alpha, u, pick, t_max, gc))
        alphas.append(alpha)
        s = s_new
    logits = np.stack(logits, 1)
    cache = {"src": src, "tgt": tgt, "src_mask": src_mask, "H": H, "enc": enc, "ey": ey, "steps": steps,
             "s0": steps[0][0], "logits": logits}  # fmt: skip
    return logits, np.stack(alphas, 1), cache


def nll(logits, tgt, tgt_mask=None):
    """eq-loss: the mean of -log p(y_i | y_<i, x) over the real target tokens."""
    tgt_mask = np.ones(tgt.shape, bool) if tgt_mask is None else tgt_mask
    z = logits - logits.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    return -np.take_along_axis(logp, tgt[..., None], -1)[..., 0][tgt_mask].mean()


def backward(params, cfg, cache, tgt_mask=None):
    """The gradient of nll with respect to every parameter, by the chain rule through eq-output, the decoder GRU,
    eq-context, eq-weights and eq-score, the initial state and both encoder GRUs."""
    p, src, tgt, H, ey = params, cache["src"], cache["tgt"], cache["H"], cache["ey"]
    tgt_mask = np.ones(tgt.shape, bool) if tgt_mask is None else tgt_mask
    B, T = tgt.shape
    n = p["W_s"].shape[0]
    g = {k: np.zeros_like(v) for k, v in p.items()}
    z = cache["logits"] - cache["logits"].max(-1, keepdims=True)
    dlogits = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
    dlogits[np.arange(B)[:, None], np.arange(T), tgt] -= 1.0
    dlogits *= tgt_mask[..., None] / tgt_mask.sum()
    dH, dHU, dey = np.zeros_like(H), np.zeros(H.shape[:2] + p["v_a"].shape), np.zeros_like(ey)
    dfixed = np.zeros((B, 2 * n))
    ds = np.zeros((B, n))  # the gradient reaching s_i from the steps after i
    for i in reversed(range(T)):
        s, c, alpha, u, pick, t_max, gc = cache["steps"][i]
        dx, ds_prev, dc = gru_step_backward(ds, p["dec_W"], p["dec_U"], p["dec_C"], gc, g, "dec_W", "dec_U", "dec_C")
        dey[:, i] += dx
        g["W_o"] += t_max.T @ dlogits[:, i]
        dt = dlogits[:, i] @ p["W_o"].T
        dtt = np.zeros((B, 2 * dt.shape[1]))
        dtt[:, 0::2], dtt[:, 1::2] = dt * pick, dt * ~pick
        g["U_o"] += s.T @ dtt
        g["V_o"] += ey[:, i].T @ dtt
        g["C_o"] += c.T @ dtt
        ds_prev += dtt @ p["U_o"].T
        dey[:, i] += dtt @ p["V_o"].T
        dc += dtt @ p["C_o"].T
        if cfg["attention"]:
            dH += alpha[..., None] * dc[:, None]
            dalpha = np.einsum("bk,bsk->bs", dc, H)
            de = alpha * (dalpha - (alpha * dalpha).sum(-1, keepdims=True))
            g["v_a"] += np.einsum("bs,bsk->k", de, u)
            dpre = de[..., None] * p["v_a"] * (1.0 - u**2)  # (B, S, n')
            dHU += dpre
            dq = dpre.sum(1)
            g["W_a"] += s.T @ dq
            ds_prev += dq @ p["W_a"].T
        else:
            dfixed += dc
        ds = ds_prev
    np.add.at(g["E_y"], tgt[:, :-1], dey[:, 1:])
    da = ds * (1.0 - cache["s0"] ** 2)
    g["W_s"] += H[:, 0, n:].T @ da
    dH[:, 0, n:] += da @ p["W_s"].T
    g["U_a"] += np.einsum("bsk,bsa->ka", H, dHU)
    dH += dHU @ p["U_a"].T
    dH[:, -1, :n] += dfixed[:, :n]
    dH[:, 0, n:] += dfixed[:, n:]
    ex, m, cf, cb = cache["enc"]
    dex = np.zeros_like(ex)
    S = src.shape[1]
    for direction, order, cs, dHd in (("fwd", reversed(range(S)), cf, dH[..., :n]), ("bwd", range(S), cb, dH[..., n:])):
        dh = np.zeros((B, n))
        for t in order:
            dh = dh + dHd[:, t]
            dx, dh_prev, _ = gru_step_backward(m[:, t] * dh, p[direction + "_W"], p[direction + "_U"], None, cs[t], g,
                                               direction + "_W", direction + "_U")  # fmt: skip
            dex[:, t] += dx
            dh = dh_prev + (1.0 - m[:, t]) * dh
    np.add.at(g["E_x"], src, dex)
    return g


def greedy_decode(params, cfg, src, max_len, eos=0):
    """Pick the most probable word at each step until <EOS> (token 0); return the tokens and the alignments, one row
    of alpha per emitted word, the matrices of Bahdanau et al.'s Figure 3."""
    out = []
    for x in src:
        x = x[None]
        H, _ = encode(params, x, np.ones(x.shape, bool))
        n = params["W_s"].shape[0]
        HU, s = H @ params["U_a"], np.tanh(H[:, 0, n:] @ params["W_s"])
        e, ys, rows = np.zeros((1, params["E_y"].shape[1])), [], []
        for _ in range(max_len):
            if cfg["attention"]:
                c, alpha, _ = align(params, s, H, HU, np.ones(x.shape, bool))
            else:
                c, alpha = np.concatenate([H[:, -1, :n], H[:, 0, n:]], -1), np.zeros(x.shape)
            tt = s @ params["U_o"] + e @ params["V_o"] + c @ params["C_o"]
            y = int(np.argmax(np.maximum(tt[:, 0::2], tt[:, 1::2]) @ params["W_o"]))
            s, _ = gru_step(e, s, params["dec_W"], params["dec_U"], c, params["dec_C"])
            e = params["E_y"][[y]]
            ys.append(y)
            rows.append(alpha[0])
            if y == eos:
                break
        out.append((np.array(ys), np.array(rows)))
    return out


# The attention layer of Luong, Pham and Manning (2015): a target state h (B, n) attends to source states Hs (B, S, n).


def luong_score(kind, h, Hs, Wa=None, va=None):
    """eq-luong-scores. dot: h . hs; general: h Wa . hs; concat: va . tanh([h; hs] Wa); location: h Wa, a score per
    source position that does not look at the source states (Wa is then (n, S))."""
    if kind == "dot":
        return np.einsum("bk,bsk->bs", h, Hs)
    if kind == "general":
        return np.einsum("bk,bsk->bs", h @ Wa, Hs)
    if kind == "concat":
        n = h.shape[-1]
        return np.tanh((h @ Wa[:n])[:, None] + Hs @ Wa[n:]) @ va
    if kind == "location":
        return h @ Wa
    raise ValueError(kind)


def softmax(e, mask):
    e = np.where(mask, e, -np.inf)
    w = np.exp(e - e.max(-1, keepdims=True))
    return w / w.sum(-1, keepdims=True)


def global_attention(kind, h, Hs, mask, Wa=None, va=None):
    """Luong et al.'s Eq. 7: a_t = softmax over the source of score(h_t, hs), and c_t = sum_s a_t(s) hs."""
    a = softmax(luong_score(kind, h, Hs, Wa, va), mask)
    return a, np.einsum("bs,bsk->bk", a, Hs)


def local_p_attention(kind, h, Hs, lengths, D, Wp, vp, Wa=None, va=None):
    """eq-local-p: p_t = S sigmoid(vp . tanh(h Wp)) with S the source length, the weights of global attention
    restricted to the window [p_t - D, p_t + D] (cut at the ends of the sentence) and multiplied by
    exp(-(s - p_t)^2 / (2 sigma^2)) with sigma = D / 2. The window takes the integer positions s with
    |s - p_t| <= D, so p_t, a real number, moves the window and the Gaussian and gets a gradient through the latter.
    The Gaussian is applied after the softmax and the weights are not renormalized, as Eq. 11 is written."""
    S = Hs.shape[1]
    pos = np.arange(S)
    p_t = lengths * sigmoid(np.tanh(h @ Wp) @ vp)  # (B,)
    window = (np.abs(pos - p_t[:, None]) <= D) & (pos < lengths[:, None])
    a = softmax(luong_score(kind, h, Hs, Wa, va), window)
    a = a * np.exp(-((pos - p_t[:, None]) ** 2) / (2.0 * (D / 2.0) ** 2))
    return p_t, a, np.einsum("bs,bsk->bk", a, Hs)


def attentional_state(c, h, Wc):
    """eq-luong-out: h~_t = tanh([c_t; h_t] Wc), the vector fed to the softmax layer and, with input feeding, to the
    next step's input."""
    return np.tanh(np.concatenate([c, h], -1) @ Wc)
