# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/jamba/ (ai://2V1JK1).
"""Jamba of Lieber et al. (2024), the NumPy reference: a hybrid language model of attention and Mamba layers with
mixture-of-experts MLPs, its forward and backward passes, generation with a cache, and its counts.

Tokens are embedded to D numbers and pass through n_blocks Jamba blocks of l layers each. Every layer is
r <- r + mixer(rmsnorm(r)) then r <- r + ffn(rmsnorm(r)) (eq-layer). The mixer of layer i of a block is causal
self-attention with grouped-query heads and no positional encoding when i = l // 2, and a Mamba block otherwise, so a
block has a : m = 1 : (l - 1) (eq-layout). The ffn is a mixture of n SwiGLU experts of which the router picks the top
K for each token when i % e = e - 1, and a single SwiGLU MLP otherwise (eq-moe). The Mamba block is that of Gu and
Dao with the discretization of their reference code and an RMSNorm on Delta's low-rank input, on B and on C, the
internal activations that the paper normalizes to stop the loss spikes. A final RMSNorm and an untied output layer
give the next-token scores. The loss is the cross-entropy plus the load-balancing term of every MoE layer (eq-aux).
Everything is float64.
"""

import math

import numpy as np


def config(vocab=13, d_model=8, n_blocks=1, block_len=4, moe_every=2, n_experts=4, top_k=2, d_ff=12, n_heads=2,
           n_kv=1, d_state=4, d_conv=3, expand=2, dt_rank=2, aux=0.01, eps=1e-6):  # fmt: skip
    """Return the config; the defaults are the example config of shapes.yaml."""
    sizes = (vocab, d_model, n_blocks, block_len, moe_every, n_experts, top_k, d_ff, n_heads, n_kv, d_state, d_conv,
             expand, dt_rank)  # fmt: skip
    if min(sizes) < 1:
        raise ValueError("the sizes must be at least 1")
    if d_model % n_heads or n_heads % n_kv:
        raise ValueError("n_heads must divide d_model and n_kv must divide n_heads")
    if block_len % moe_every or (block_len // 2) % moe_every == moe_every - 1:
        raise ValueError("moe_every must divide block_len, and the attention layer must keep a plain MLP")
    if top_k > n_experts:
        raise ValueError("top_k must be at most n_experts")
    return {"vocab": vocab, "d_model": d_model, "n_blocks": n_blocks, "block_len": block_len, "moe_every": moe_every,
            "n_experts": n_experts, "top_k": top_k, "d_ff": d_ff, "n_heads": n_heads, "n_kv": n_kv,
            "head_dim": d_model // n_heads, "d_state": d_state, "d_conv": d_conv, "expand": expand,
            "d_inner": expand * d_model, "dt_rank": dt_rank, "n_layers": n_blocks * block_len, "aux": aux,
            "eps": eps}  # fmt: skip


def layers(cfg):
    """eq-layout: the (mixer, ffn) kind of every layer, mixer "attn" or "mamba", ffn "mlp" or "moe"."""
    l, e = cfg["block_len"], cfg["moe_every"]
    return [
        ("attn" if i % l == l // 2 else "mamba", "moe" if i % e == e - 1 else "mlp") for i in range(cfg["n_layers"])
    ]


def mixer_shapes(cfg, kind):
    D, A, G = cfg["d_model"], cfg["n_heads"] * cfg["head_dim"], cfg["n_kv"] * cfg["head_dim"]
    if kind == "attn":
        return {"wq": (D, A), "wk": (D, G), "wv": (D, G), "wo": (A, D)}
    I, N, K, R = cfg["d_inner"], cfg["d_state"], cfg["d_conv"], cfg["dt_rank"]
    return {"in_proj": (D, 2 * I), "conv_w": (I, K), "conv_b": (I,), "x_proj": (I, R + 2 * N), "dt_norm": (R,),
            "B_norm": (N,), "C_norm": (N,), "dt_w": (R, I), "dt_b": (I,), "A_log": (I, N), "D": (I,),
            "out_proj": (I, D)}  # fmt: skip


def ffn_shapes(cfg, kind):
    D, F, n = cfg["d_model"], cfg["d_ff"], cfg["n_experts"]
    if kind == "mlp":
        return {"w_gate": (D, F), "w_up": (D, F), "w_down": (F, D)}
    return {"router": (D, n), "e_gate": (n, D, F), "e_up": (n, D, F), "e_down": (n, F, D)}


def param_shapes(cfg):
    D, V = cfg["d_model"], cfg["vocab"]
    shapes = {"embed": (V, D)}
    for i, (m, f) in enumerate(layers(cfg)):
        shapes |= {f"l{i}_norm1": (D,), f"l{i}_norm2": (D,)}
        shapes |= {f"l{i}_{k}": s for k, s in (mixer_shapes(cfg, m) | ffn_shapes(cfg, f)).items()}
    return shapes | {"norm_f": (D,), "head": (D, V)}


def param_count(cfg, active=False):
    """eq-params from the formula, not the shapes. active=True counts the experts that one token uses, K of n."""
    D, V, F, n, K = (cfg[k] for k in ("d_model", "vocab", "d_ff", "n_experts", "top_k"))
    G, I, N, Kc, R = cfg["n_kv"] * cfg["head_dim"], cfg["d_inner"], cfg["d_state"], cfg["d_conv"], cfg["dt_rank"]
    attn = 2 * D * D + 2 * D * G + 2 * D
    mamba = 3 * I * D + I * (Kc + 1) + I * (R + 2 * N) + R + 2 * N + (R + 1) * I + I * N + I + 2 * D
    mlp = 3 * D * F
    moe = D * n + (K if active else n) * 3 * D * F
    l, e, nb = cfg["block_len"], cfg["moe_every"], cfg["n_blocks"]
    return 2 * V * D + D + nb * (attn + (l - 1) * mamba + (l - l // e) * mlp + (l // e) * moe)


def flops(cfg, B, T):
    """eq-flops: the forward FLOPs of the products on B sequences of T tokens, a multiply-add as 2. The attention
    scores and their weighted sum are over all T x T pairs, as computed; the elementwise work (the norms, the
    activations, the softmax, the convolution and the state update of the scan) is left out."""
    D, V, F, n, K, H, hd = (cfg[k] for k in ("d_model", "vocab", "d_ff", "n_experts", "top_k", "n_heads", "head_dim"))
    G, I, N, R = cfg["n_kv"] * hd, cfg["d_inner"], cfg["d_state"], cfg["dt_rank"]
    attn = 2 * B * T * (2 * D * D + 2 * D * G) + 4 * B * H * T * T * hd
    mamba = 2 * B * T * (2 * D * I + I * (R + 2 * N) + R * I + I * N + I * D)
    mlp = 2 * B * T * 3 * D * F
    moe = 2 * B * T * (D * n + K * 3 * D * F)
    l, e, nb = cfg["block_len"], cfg["moe_every"], cfg["n_blocks"]
    return nb * (attn + (l - 1) * mamba + (l - l // e) * mlp + (l // e) * moe) + 2 * B * T * D * V


def cache_numbers(cfg, B, T):
    """eq-cache: the numbers that generation keeps after T tokens: keys and values of every attention layer, which
    grow with T, and the convolution window and SSM state of every Mamba layer, which do not."""
    G, I = cfg["n_kv"] * cfg["head_dim"], cfg["d_inner"]
    return sum(2 * B * T * G if m == "attn" else B * I * (cfg["d_conv"] - 1 + cfg["d_state"]) for m, _ in layers(cfg))


def init(rng, cfg, scale=0.1, dt_min=1e-3, dt_max=1e-1):
    """Random weights N(0, scale^2); RMSNorm weights 1; and, in the Mamba layers, A_log = log(1..N), D = 1 and the
    bias of Delta so that softplus(dt_b) is log-uniform in [dt_min, dt_max], as in the reference code of Mamba."""
    p = {k: rng.normal(0.0, scale, s) for k, s in param_shapes(cfg).items()}
    I, N = cfg["d_inner"], cfg["d_state"]
    for k in p:
        if k.endswith("norm") or k.endswith(("norm1", "norm2", "norm_f")):
            p[k] = np.ones_like(p[k])
    for i, (m, _) in enumerate(layers(cfg)):
        if m == "mamba":
            dt = np.exp(rng.uniform(np.log(dt_min), np.log(dt_max), I))
            p[f"l{i}_dt_b"] = dt + np.log(-np.expm1(-dt))  # the inverse of softplus
            p[f"l{i}_A_log"] = np.log(np.tile(np.arange(1.0, N + 1), (I, 1)))
            p[f"l{i}_D"] = np.ones(I)
    return p


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def softplus(x):
    return np.logaddexp(0.0, x)


def silu(x):
    return x * sigmoid(x)


def silu_grad(x):
    s = sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def rmsnorm(x, w, eps):
    r = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return x * r * w, r


def rmsnorm_backward(dy, x, w, r):
    xh = x * r
    g = dy * w
    return r * (g - xh * (g * xh).mean(-1, keepdims=True)), (dy * xh).reshape(-1, x.shape[-1]).sum(0)


def attention(p, cfg, u, q_):
    """eq-attn: causal self-attention of (B, T, D) inputs; head h reads the keys and values of group h // (H / G).
    There is no positional encoding: the order reaches the attention only through the Mamba layers before it."""
    B, T, _ = u.shape
    G, hd = cfg["n_kv"], cfg["head_dim"]
    Q = cfg["n_heads"] // G
    q = (u @ p[q_ + "wq"]).reshape(B, T, G, Q, hd)
    k = (u @ p[q_ + "wk"]).reshape(B, T, G, hd)
    v = (u @ p[q_ + "wv"]).reshape(B, T, G, hd)
    s = np.einsum("btgqd,bsgd->bgqts", q, k) / math.sqrt(hd)
    a = softmax(np.where(np.tril(np.ones((T, T), bool)), s, -np.inf))
    o = np.einsum("bgqts,bsgd->btgqd", a, v).reshape(B, T, -1)
    return o @ p[q_ + "wo"], {"u": u, "q": q, "k": k, "v": v, "a": a, "o": o}


def attention_backward(dout, p, cfg, c, q_):
    u, q, k, v, a, o = (c[n] for n in ("u", "q", "k", "v", "a", "o"))
    D, hd = u.shape[-1], cfg["head_dim"]
    g = {"wo": o.reshape(-1, o.shape[-1]).T @ dout.reshape(-1, D)}
    do = (dout @ p[q_ + "wo"].T).reshape(q.shape)
    da = np.einsum("btgqd,bsgd->bgqts", do, v)
    dv = np.einsum("bgqts,btgqd->bsgd", a, do)
    ds = a * (da - (da * a).sum(-1, keepdims=True)) / math.sqrt(hd)
    dq = np.einsum("bgqts,bsgd->btgqd", ds, k).reshape(*u.shape[:2], -1)
    dk = np.einsum("bgqts,btgqd->bsgd", ds, q).reshape(*u.shape[:2], -1)
    dv = dv.reshape(*u.shape[:2], -1)
    uf = u.reshape(-1, D)
    g |= {n: uf.T @ d.reshape(-1, d.shape[-1]) for n, d in (("wq", dq), ("wk", dk), ("wv", dv))}
    return g, dq @ p[q_ + "wq"].T + dk @ p[q_ + "wk"].T + dv @ p[q_ + "wv"].T


def causal_conv(x, w, b):
    """Depthwise causal convolution over the positions of (B, T, I): out[t] = b + sum_k w[:, k] x[t - K + 1 + k]."""
    K, T = w.shape[1], x.shape[1]
    xp = np.concatenate([np.zeros((x.shape[0], K - 1, x.shape[2])), x], 1)
    return b + sum(w[:, k] * xp[:, k : k + T] for k in range(K))


def mamba(p, cfg, u, q_):
    """eq-mamba: the Mamba block on (B, T, D) inputs, with RMSNorm on Delta's low-rank input, B and C."""
    I, N, R, eps = cfg["d_inner"], cfg["d_state"], cfg["dt_rank"], cfg["eps"]
    xz = u @ p[q_ + "in_proj"]
    x, z = xz[..., :I], xz[..., I:]
    xc = causal_conv(x, p[q_ + "conv_w"], p[q_ + "conv_b"])
    xa = silu(xc)
    dbl = xa @ p[q_ + "x_proj"]
    dtl, Bm, C = dbl[..., :R], dbl[..., R : R + N], dbl[..., R + N :]
    dtn, s_dt = rmsnorm(dtl, p[q_ + "dt_norm"], eps)
    Bn, s_B = rmsnorm(Bm, p[q_ + "B_norm"], eps)
    Cn, s_C = rmsnorm(C, p[q_ + "C_norm"], eps)
    dtp = dtn @ p[q_ + "dt_w"] + p[q_ + "dt_b"]
    dt = softplus(dtp)
    A = -np.exp(p[q_ + "A_log"])
    dA, dB = np.exp(dt[..., None] * A), dt[..., None] * Bn[:, :, None, :]
    hs = np.empty_like(dA)
    h = np.zeros(dA[:, 0].shape)
    for t in range(dA.shape[1]):  # the selective scan, h_t = A_bar_t h_{t-1} + B_bar_t x_t
        h = dA[:, t] * h + dB[:, t] * xa[:, t, :, None]
        hs[:, t] = h
    y = np.einsum("btin,btn->bti", hs, Cn) + p[q_ + "D"] * xa
    g = silu(z)
    c = dict(u=u, x=x, z=z, xc=xc, xa=xa, dtl=dtl, Bm=Bm, C=C, dtn=dtn, Bn=Bn, Cn=Cn, s_dt=s_dt, s_B=s_B, s_C=s_C,
             dtp=dtp, dt=dt, A=A, dA=dA, dB=dB, hs=hs, y=y, g=g)  # fmt: skip
    return (y * g) @ p[q_ + "out_proj"], c


def mamba_backward(dout, p, cfg, c, q_):
    I, R, K = cfg["d_inner"], cfg["dt_rank"], cfg["d_conv"]
    g = {"out_proj": np.einsum("bti,btd->id", c["y"] * c["g"], dout)}
    dyg = dout @ p[q_ + "out_proj"].T
    dy, dz = dyg * c["g"], dyg * c["y"] * silu_grad(c["z"])
    g["D"] = (dy * c["xa"]).sum((0, 1))
    dxa = dy * p[q_ + "D"]
    hs, dA, dB, xa, Cn = c["hs"], c["dA"], c["dB"], c["xa"], c["Cn"]
    T = xa.shape[1]
    ddA, ddB, dCn = np.zeros_like(dA), np.zeros_like(dB), np.zeros_like(Cn)
    gh = np.zeros_like(hs[:, 0])
    for t in reversed(range(T)):  # the scan run backwards: gh is dLoss/dh_t
        gh = gh + dy[:, t, :, None] * Cn[:, t, None, :]
        dCn[:, t] = np.einsum("bi,bin->bn", dy[:, t], hs[:, t])
        ddA[:, t] = gh * (hs[:, t - 1] if t > 0 else 0.0)
        ddB[:, t] = gh * xa[:, t, :, None]
        dxa[:, t] += (gh * dB[:, t]).sum(-1)
        gh = gh * dA[:, t]
    dt, A = c["dt"], c["A"]
    ddt = (ddA * dA * A).sum(-1) + (ddB * c["Bn"][:, :, None, :]).sum(-1)
    g["A_log"] = (ddA * dA * dt[..., None]).sum((0, 1)) * A
    dBn = (ddB * dt[..., None]).sum(2)
    ddtp = ddt * sigmoid(c["dtp"])
    g["dt_w"] = np.einsum("btr,bti->ri", c["dtn"], ddtp)
    g["dt_b"] = ddtp.sum((0, 1))
    ddtl, g["dt_norm"] = rmsnorm_backward(ddtp @ p[q_ + "dt_w"].T, c["dtl"], p[q_ + "dt_norm"], c["s_dt"])
    dBm, g["B_norm"] = rmsnorm_backward(dBn, c["Bm"], p[q_ + "B_norm"], c["s_B"])
    dC, g["C_norm"] = rmsnorm_backward(dCn, c["C"], p[q_ + "C_norm"], c["s_C"])
    ddbl = np.concatenate([ddtl, dBm, dC], -1)
    g["x_proj"] = np.einsum("bti,btk->ik", xa, ddbl)
    dxc = (dxa + ddbl @ p[q_ + "x_proj"].T) * silu_grad(c["xc"])
    g["conv_b"] = dxc.sum((0, 1))
    xp = np.concatenate([np.zeros((xa.shape[0], K - 1, I)), c["x"]], 1)
    g["conv_w"] = np.stack([(dxc * xp[:, k : k + T]).sum((0, 1)) for k in range(K)], 1)
    dxp = np.zeros_like(xp)
    for k in range(K):
        dxp[:, k : k + T] += dxc * p[q_ + "conv_w"][:, k]
    dxz = np.concatenate([dxp[:, K - 1 :], dz], -1)
    g["in_proj"] = np.einsum("btd,btj->dj", c["u"], dxz)
    return g, dxz @ p[q_ + "in_proj"].T


def swiglu(x, wg, wu, wd):
    hg, hu = x @ wg, x @ wu
    m = silu(hg) * hu
    return m @ wd, (x, hg, hu, m)


def swiglu_backward(dy, wg, wu, wd, c):
    x, hg, hu, m = c
    dm = dy @ wd.T
    dhg, dhu = dm * hu * silu_grad(hg), dm * silu(hg)
    flat = lambda a: a.reshape(-1, a.shape[-1])  # noqa: E731
    grads = (flat(x).T @ flat(dhg), flat(x).T @ flat(dhu), flat(m).T @ flat(dy))
    return grads, dhg @ wg.T + dhu @ wu.T


def moe(p, cfg, u, q_):
    """eq-moe and eq-aux: each token goes to the K experts with the largest router probabilities, and its output is
    the sum of their outputs weighted by those probabilities; only the K chosen experts run on it. aux is the
    load-balancing term alpha n sum_j f_j P_j, f_j the share of the K M assignments that go to expert j and P_j the
    mean probability of j over the M tokens."""
    B, T, D = u.shape
    n, K = cfg["n_experts"], cfg["top_k"]
    x = u.reshape(-1, D)
    probs = softmax(x @ p[q_ + "router"])
    top = np.argsort(-probs, axis=-1, kind="stable")[:, :K]
    w = np.take_along_axis(probs, top, -1)
    out, parts = np.zeros_like(x), []
    for j in range(n):
        rows, slot = np.nonzero(top == j)
        y, c = swiglu(x[rows], p[q_ + "e_gate"][j], p[q_ + "e_up"][j], p[q_ + "e_down"][j])
        out[rows] += w[rows, slot, None] * y
        parts.append((rows, slot, y, c))
    f = np.bincount(top.ravel(), minlength=n) / top.size
    aux = cfg["aux"] * n * float((f * probs.mean(0)).sum())
    return out.reshape(B, T, D), aux, {"x": x, "probs": probs, "w": w, "f": f, "parts": parts}


def moe_backward(dout, p, cfg, c, q_):
    x, probs, w, n = c["x"], c["probs"], c["w"], cfg["n_experts"]
    shape, dout = dout.shape, dout.reshape(x.shape)
    dprobs, dx = np.zeros_like(probs), np.zeros_like(x)
    g = {k: np.zeros_like(p[q_ + k]) for k in ("e_gate", "e_up", "e_down")}
    for j, (rows, slot, y, cj) in enumerate(c["parts"]):
        go = dout[rows]
        dprobs[rows, j] += (go * y).sum(-1)
        (g["e_gate"][j], g["e_up"][j], g["e_down"][j]), dxj = swiglu_backward(
            w[rows, slot, None] * go, p[q_ + "e_gate"][j], p[q_ + "e_up"][j], p[q_ + "e_down"][j], cj
        )
        dx[rows] += dxj
    dprobs += cfg["aux"] * n * c["f"] / len(x)  # the gradient of aux; f is piecewise constant
    dlogits = probs * (dprobs - (dprobs * probs).sum(-1, keepdims=True))
    g["router"] = x.T @ dlogits
    return g, (dx + dlogits @ p[q_ + "router"].T).reshape(shape)


def forward(p, cfg, tokens):
    """The (B, T, V) next-token scores of a (B, T) array of token ids, the sum of the aux terms, and a cache."""
    r, eps, cache = p["embed"][tokens], cfg["eps"], []
    B, T = tokens.shape
    aux = 0.0
    for i, (m, f) in enumerate(layers(cfg)):
        q_ = f"l{i}_"
        u, s1 = rmsnorm(r, p[q_ + "norm1"], eps)
        out, cm = (attention if m == "attn" else mamba)(p, cfg, u, q_)
        r_mid = r + out
        v, s2 = rmsnorm(r_mid, p[q_ + "norm2"], eps)
        if f == "moe":
            out, a, cf = moe(p, cfg, v, q_)
            aux += a
        else:
            out, cf = swiglu(v, p[q_ + "w_gate"], p[q_ + "w_up"], p[q_ + "w_down"])
        cache.append({"r": r, "s1": s1, "r_mid": r_mid, "s2": s2, "mixer": cm, "ffn": cf})
        r = r_mid + out
    hf, sf = rmsnorm(r, p["norm_f"], eps)
    return hf @ p["head"], aux, {"tokens": tokens, "layers": cache, "r": r, "sf": sf, "hf": hf}


def nll(logits, targets):
    """The mean cross-entropy of the next tokens."""
    m = logits.max(-1, keepdims=True)
    logp = logits - m - np.log(np.exp(logits - m).sum(-1, keepdims=True))
    return float(-np.take_along_axis(logp, targets[..., None], -1).mean())


def lm_loss(p, cfg, tokens):
    """eq-loss: the cross-entropy of predicting tokens[:, 1:] from tokens[:, :-1], plus the aux terms."""
    logits, aux, _ = forward(p, cfg, tokens[:, :-1])
    return nll(logits, tokens[:, 1:]) + aux


def backward(p, cfg, cache, logits, targets):
    """The gradients of eq-loss with respect to every parameter."""
    dl = softmax(logits)
    np.put_along_axis(dl, targets[..., None], np.take_along_axis(dl, targets[..., None], -1) - 1.0, -1)
    dl /= targets.size
    V, D = p["embed"].shape
    grads = {"head": cache["hf"].reshape(-1, D).T @ dl.reshape(-1, V)}
    dr, grads["norm_f"] = rmsnorm_backward(dl @ p["head"].T, cache["r"], p["norm_f"], cache["sf"])
    for i in reversed(range(cfg["n_layers"])):
        m, f = layers(cfg)[i]
        q_, c = f"l{i}_", cache["layers"][i]
        if f == "moe":
            g, dv = moe_backward(dr, p, cfg, c["ffn"], q_)
        else:
            (gg, gu, gd), dv = swiglu_backward(dr, p[q_ + "w_gate"], p[q_ + "w_up"], p[q_ + "w_down"], c["ffn"])
            g = {"w_gate": gg, "w_up": gu, "w_down": gd}
        dn, g["norm2"] = rmsnorm_backward(dv, c["r_mid"], p[q_ + "norm2"], c["s2"])
        dr = dr + dn
        gm, du = (attention_backward if m == "attn" else mamba_backward)(dr, p, cfg, c["mixer"], q_)
        g |= gm
        dn, g["norm1"] = rmsnorm_backward(du, c["r"], p[q_ + "norm1"], c["s1"])
        dr = dr + dn
        grads |= {q_ + k: v for k, v in g.items()}
    grads["embed"] = np.zeros_like(p["embed"])
    np.add.at(grads["embed"], cache["tokens"], dr)
    return grads


def init_state(cfg, B):
    """The state of generation before the first token: empty keys and values for each attention layer, and a zero
    convolution window and SSM state for each Mamba layer."""
    G, hd, I = cfg["n_kv"], cfg["head_dim"], cfg["d_inner"]
    return [(np.zeros((B, 0, G, hd)), np.zeros((B, 0, G, hd))) if m == "attn" else
            (np.zeros((B, cfg["d_conv"] - 1, I)), np.zeros((B, I, cfg["d_state"]))) for m, _ in layers(cfg)]  # fmt: skip


def step(p, cfg, token, state):
    """One token of generation: (B,) ids and the state give the (B, V) scores and the next state. The keys and
    values of the attention layers grow by one position; the state of the Mamba layers keeps its size."""
    I, N, R, K, eps = cfg["d_inner"], cfg["d_state"], cfg["dt_rank"], cfg["d_conv"], cfg["eps"]
    G, hd = cfg["n_kv"], cfg["head_dim"]
    Q = cfg["n_heads"] // G
    r, new = p["embed"][token], []
    for i, ((m, f), (s0, s1)) in enumerate(zip(layers(cfg), state)):
        q_ = f"l{i}_"
        u, _ = rmsnorm(r, p[q_ + "norm1"], eps)
        Bsz = u.shape[0]
        if m == "attn":
            q = (u @ p[q_ + "wq"]).reshape(Bsz, G, Q, hd)
            ks = np.concatenate([s0, (u @ p[q_ + "wk"]).reshape(Bsz, 1, G, hd)], 1)
            vs = np.concatenate([s1, (u @ p[q_ + "wv"]).reshape(Bsz, 1, G, hd)], 1)
            a = softmax(np.einsum("bgqd,bsgd->bgqs", q, ks) / math.sqrt(hd))
            r = r + np.einsum("bgqs,bsgd->bgqd", a, vs).reshape(Bsz, -1) @ p[q_ + "wo"]
            new.append((ks, vs))
        else:
            xz = u @ p[q_ + "in_proj"]
            x, z = xz[:, :I], xz[:, I:]
            win = np.concatenate([s0, x[:, None]], 1)
            xa = silu(p[q_ + "conv_b"] + (win * p[q_ + "conv_w"].T).sum(1))
            dbl = xa @ p[q_ + "x_proj"]
            dtn = rmsnorm(dbl[:, :R], p[q_ + "dt_norm"], eps)[0]
            Bn = rmsnorm(dbl[:, R : R + N], p[q_ + "B_norm"], eps)[0]
            Cn = rmsnorm(dbl[:, R + N :], p[q_ + "C_norm"], eps)[0]
            dt = softplus(dtn @ p[q_ + "dt_w"] + p[q_ + "dt_b"])
            h = np.exp(dt[..., None] * -np.exp(p[q_ + "A_log"])) * s1 + dt[..., None] * Bn[:, None] * xa[..., None]
            y = np.einsum("bin,bn->bi", h, Cn) + p[q_ + "D"] * xa
            r = r + (y * silu(z)) @ p[q_ + "out_proj"]
            new.append((win[:, 1:] if K > 1 else s0, h))
        v, _ = rmsnorm(r, p[q_ + "norm2"], eps)
        if f == "moe":
            r = r + moe(p, cfg, v[:, None], q_)[0][:, 0]
        else:
            r = r + swiglu(v, p[q_ + "w_gate"], p[q_ + "w_up"], p[q_ + "w_down"])[0]
    hf, _ = rmsnorm(r, p["norm_f"], eps)
    return hf @ p["head"], new


def kv_cache_bytes(n_attn, n_kv, head_dim, T, bytes_per=2):
    """The bytes of the keys and values of n_attn attention layers with n_kv heads of head_dim after T tokens."""
    return 2 * n_attn * n_kv * head_dim * T * bytes_per


def worked_example():
    """The example config, a block of 4 layers: what generation keeps after 1000 tokens, against the same block
    with attention in all four layers."""
    cfg = config()
    full = 4 * 2 * 1000 * cfg["n_kv"] * cfg["head_dim"]
    return {"params": param_count(cfg), "active": param_count(cfg, active=True), "cache": cache_numbers(cfg, 1, 1000),
            "cache_all_attention": full}  # fmt: skip
