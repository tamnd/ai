# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sequence-to-sequence-model/ (ai://N149H2).
"""A sequence-to-sequence model of two deep LSTMs in JAX, as pure functions of the parameters.

The parameters have the names and the layout of the NumPy reference, so the shared fixture loads as it is. The loops
over the source and the target are jax.lax.scan, and the gradient of eq-loss is left to jax.grad. Token 0 is <EOS>.
"""

import jax
import jax.numpy as jnp

EOS = 0


def config(src_vocab=7, tgt_vocab=6, embed_size=3, hidden_size=4, n_layers=2, reverse=True):
    """Return the config of the model."""
    if min(embed_size, hidden_size, n_layers) < 1:
        raise ValueError("the sizes and the number of layers must be at least 1")
    if min(src_vocab, tgt_vocab) < 2:
        raise ValueError("a vocabulary needs <EOS> and at least one other token")
    return {
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "embed_size": embed_size,
        "hidden_size": hidden_size,
        "n_layers": n_layers,
        "reverse": bool(reverse),
    }


def layer_names(prefix, n_layers):
    return [(f"{prefix}{l}_W", f"{prefix}{l}_U", f"{prefix}{l}_b") for l in range(n_layers)]  # noqa: E741


def init(key, cfg, scale=0.08):
    """All parameters uniform in [-scale, scale], as in Section 3.4 of Sutskever et al."""
    E, H, L = cfg["embed_size"], cfg["hidden_size"], cfg["n_layers"]
    shapes = {"src_embed": (cfg["src_vocab"], E), "tgt_embed": (cfg["tgt_vocab"], E)}
    for prefix in ("enc", "dec"):
        for l, (w, u, b) in enumerate(layer_names(prefix, L)):  # noqa: E741
            shapes |= {w: (E if l == 0 else H, 4 * H), u: (H, 4 * H), b: (4 * H,)}
    shapes |= {"out_W": (H, cfg["tgt_vocab"]), "out_b": (cfg["tgt_vocab"],)}
    keys = jax.random.split(key, len(shapes))
    return {
        n: jax.random.uniform(k, s, minval=-scale, maxval=scale)
        for k, (n, s) in zip(keys, shapes.items(), strict=True)
    }


def step(params, prefix, n_layers, x, state):
    """One step of the L-layer stack `prefix` on x (B, E) from state (h, c), each (L, B, H)."""
    hs, cs = [], []
    for l, (w, u, b) in enumerate(layer_names(prefix, n_layers)):  # noqa: E741
        z = x @ params[w] + state[0][l] @ params[u] + params[b]  # (B, 4 * H)
        i, f, g, o = jnp.split(z, 4, axis=-1)
        c = jax.nn.sigmoid(f) * state[1][l] + jax.nn.sigmoid(i) * jnp.tanh(g)
        x = jax.nn.sigmoid(o) * jnp.tanh(c)
        hs.append(x)
        cs.append(c)
    return x, (jnp.stack(hs), jnp.stack(cs))


def encode(params, cfg, src):
    """The vector v = (h, c), each (L, B, H), for the source tokens (B, S)."""
    L, H = cfg["n_layers"], cfg["hidden_size"]
    order = src[:, ::-1] if cfg["reverse"] else src
    x = params["src_embed"][order]  # (B, S, E)
    zeros = jnp.zeros((L, src.shape[0], H), x.dtype)

    def body(state, x_t):
        return step(params, "enc", L, x_t, state)[1], None

    v, _ = jax.lax.scan(body, (zeros, zeros), jnp.swapaxes(x, 0, 1))
    return v


def decoder_inputs(tgt):
    return jnp.concatenate([jnp.full_like(tgt[:, :1], EOS), tgt[:, :-1]], axis=1)


def forward(params, cfg, src, tgt):
    """Teacher-forced logits (B, T, V) of p(y_t | v, y_<t)."""
    y = params["tgt_embed"][decoder_inputs(tgt)]  # (B, T, E)

    def body(state, y_t):
        top, state = step(params, "dec", cfg["n_layers"], y_t, state)
        return state, top

    _, tops = jax.lax.scan(body, encode(params, cfg, src), jnp.swapaxes(y, 0, 1))
    return jnp.swapaxes(tops, 0, 1) @ params["out_W"] + params["out_b"]


def loss(params, cfg, src, tgt):
    """eq-loss: -(1/B) sum_b sum_t log p(y_t | v, y_<t)."""
    logp = jax.nn.log_softmax(forward(params, cfg, src, tgt), axis=-1)
    return -jnp.take_along_axis(logp, tgt[..., None], axis=-1).sum() / tgt.shape[0]


def greedy(params, cfg, src, max_len):
    """The most likely token at every step, (B, max_len), with <EOS> repeated after a sequence has ended."""
    B = src.shape[0]

    def body(carry, _):
        state, prev, done = carry
        top, state = step(params, "dec", cfg["n_layers"], params["tgt_embed"][prev], state)
        tok = jnp.where(done, EOS, jnp.argmax(top @ params["out_W"] + params["out_b"], axis=-1))
        return (state, tok, done | (tok == EOS)), tok

    init_carry = (encode(params, cfg, src), jnp.full((B,), EOS), jnp.zeros((B,), bool))
    _, toks = jax.lax.scan(body, init_carry, None, length=max_len)
    return toks.T


def param_count(cfg):
    E, H, L, V = cfg["embed_size"], cfg["hidden_size"], cfg["n_layers"], cfg["tgt_vocab"]
    stack = 4 * H * (E + H) + 4 * H + (L - 1) * (8 * H * H + 4 * H)
    return (cfg["src_vocab"] + V) * E + 2 * stack + H * V + V
