# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sequence-to-sequence-model/ (ai://N149H2).
"""A sequence-to-sequence model of two deep LSTMs, the NumPy reference, in the form of Sutskever et al. (2014).

The source tokens x_1..x_S, reversed when cfg["reverse"] is set, are embedded and read by an L-layer LSTM encoder.
Its final hidden and cell states of every layer are the fixed-size vector v of eq-factor, and they are the initial
state of an L-layer LSTM decoder with its own weights. The decoder reads <EOS>, y_1, ..., y_{T-1}, the target shifted
by one (teacher forcing), and a softmax layer on its top state gives p(y_t | v, y_1..y_{t-1}) at every step (eq-step).
Token 0 is <EOS> in both vocabularies: it ends every source and every target, and it starts the decoder.

Each LSTM layer is the one of the LSTM page: one bias, the four gates stacked as input, forget, cell candidate and
output, the order of torch.nn.LSTM. The training loss is eq-loss, the negative log-likelihood of each target
sequence, summed over its tokens and averaged over the batch; backward() differentiates it by hand.
"""

import numpy as np

EOS = 0
GATES = ("i", "f", "g", "o")


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
    """The parameter names of the LSTM layers of the encoder ("enc") or the decoder ("dec")."""
    return [(f"{prefix}{l}_W", f"{prefix}{l}_U", f"{prefix}{l}_b") for l in range(n_layers)]


def init(rng, cfg, scale=0.08):
    """Return the parameters, all uniform in [-scale, scale], the initialization of Sutskever et al. (Section 3.4)."""
    E, H, L = cfg["embed_size"], cfg["hidden_size"], cfg["n_layers"]
    shapes = {"src_embed": (cfg["src_vocab"], E), "tgt_embed": (cfg["tgt_vocab"], E)}
    for prefix in ("enc", "dec"):
        for l, (w, u, b) in enumerate(layer_names(prefix, L)):
            shapes |= {w: (E if l == 0 else H, 4 * H), u: (H, 4 * H), b: (4 * H,)}
    shapes |= {"out_W": (H, cfg["tgt_vocab"]), "out_b": (cfg["tgt_vocab"],)}
    return {name: rng.uniform(-scale, scale, shape) for name, shape in shapes.items()}


def sigmoid(z):
    """The logistic function, without overflow for large negative z."""
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def log_softmax(z):
    """log softmax over the last axis."""
    m = z.max(axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))


def lstm_cell(W, U, b, x, h, c):
    """One LSTM step: return (h_new, c_new, gates) for x (B, I), h (B, H) and c (B, H)."""
    H = h.shape[-1]
    z = x @ W + h @ U + b  # (B, 4 * H)
    i, f = sigmoid(z[:, :H]), sigmoid(z[:, H : 2 * H])  # (B, H) each
    g, o = np.tanh(z[:, 2 * H : 3 * H]), sigmoid(z[:, 3 * H :])  # (B, H) each
    c_new = f * c + i * g  # (B, H)
    return o * np.tanh(c_new), c_new, (i, f, g, o)


def run_stack(params, prefix, x, h0, c0):
    """Run the L-layer LSTM `prefix` over x (B, T, I) from the state h0, c0 (L, B, H).

    Return (top, h_T, c_T, caches): the top layer's state at every step (B, T, H), the final states (L, B, H) and,
    for backward, the input, the states and the gates of every layer.
    """
    B, T, _ = x.shape
    L, H = h0.shape[0], h0.shape[-1]
    inp, h_T, c_T, caches = x, np.empty((L, B, H)), np.empty((L, B, H)), []
    for l, (w, u, b) in enumerate(layer_names(prefix, L)):
        h, c = h0[l], c0[l]
        hs, cs = np.empty((B, T, H)), np.empty((B, T, H))
        gates = {name: np.empty((B, T, H)) for name in GATES}
        for t in range(T):
            h, c, gs = lstm_cell(params[w], params[u], params[b], inp[:, t], h, c)
            hs[:, t], cs[:, t] = h, c
            for name, v in zip(GATES, gs):
                gates[name][:, t] = v
        caches.append({"x": inp, "h0": h0[l], "c0": c0[l], "h": hs, "c": cs} | gates)
        h_T[l], c_T[l] = h, c
        inp = hs
    return inp, h_T, c_T, caches


def lstm_backward(W, U, cache, dh, dh_T, dc_T):
    """Backpropagation through time for one layer: return (dx, dW, dU, db, dh0, dc0)."""
    x, hs, cs = cache["x"], cache["h"], cache["c"]
    T = x.shape[1]
    dW, dU, db = np.zeros_like(W), np.zeros_like(U), np.zeros(W.shape[1])
    dx = np.zeros_like(x)
    dh_next, dc_next = dh_T.copy(), dc_T.copy()  # (B, H) each
    for t in reversed(range(T)):
        i, f, g, o = (cache[name][:, t] for name in GATES)
        c_prev = cs[:, t - 1] if t > 0 else cache["c0"]
        h_prev = hs[:, t - 1] if t > 0 else cache["h0"]
        tc = np.tanh(cs[:, t])
        dh_t = dh[:, t] + dh_next  # (B, H)
        dc = dc_next + dh_t * o * (1.0 - tc**2)  # (B, H)
        dz = np.concatenate(
            [dc * g * i * (1.0 - i), dc * c_prev * f * (1.0 - f), dc * i * (1.0 - g**2), dh_t * tc * o * (1.0 - o)],
            axis=-1,
        )  # (B, 4 * H)
        dW += x[:, t].T @ dz
        dU += h_prev.T @ dz
        db += dz.sum(axis=0)
        dx[:, t] = dz @ W.T
        dh_next, dc_next = dz @ U.T, dc * f
    return dx, dW, dU, db, dh_next, dc_next


def stack_backward(params, prefix, caches, dtop, dh_T, dc_T, grads):
    """Backpropagate through the stack `prefix`; add the weight gradients to grads and return (dx, dh0, dc0)."""
    L = len(caches)
    dh0, dc0 = np.empty_like(dh_T), np.empty_like(dc_T)
    d = dtop
    for l in reversed(range(L)):
        w, u, b = layer_names(prefix, L)[l]
        d, dW, dU, db, dh0[l], dc0[l] = lstm_backward(params[w], params[u], caches[l], d, dh_T[l], dc_T[l])
        grads[w] += dW
        grads[u] += dU
        grads[b] += db
    return d, dh0, dc0


def source_order(cfg, src):
    """The order in which the encoder reads the source (B, S): reversed when cfg["reverse"] is set."""
    return src[:, ::-1] if cfg["reverse"] else src


def encode(params, cfg, src):
    """Read the source tokens (B, S); return the vector v = (h, c), each (L, B, H), and the cache of the encoder."""
    B = src.shape[0]
    L, H = cfg["n_layers"], cfg["hidden_size"]
    order = source_order(cfg, src)
    zeros = np.zeros((L, B, H))
    _, h, c, caches = run_stack(params, "enc", params["src_embed"][order], zeros, zeros)
    return (h, c), {"order": order, "caches": caches}


def decoder_inputs(tgt):
    """The decoder reads <EOS>, y_1, ..., y_{T-1}: the target (B, T) shifted right by one, with <EOS> in front."""
    return np.concatenate([np.full((tgt.shape[0], 1), EOS, dtype=tgt.dtype), tgt[:, :-1]], axis=1)


def forward(params, cfg, src, tgt, dec_in=None):
    """Return (logits, cache): logits (B, T, V) of p(y_t | v, y_<t) for every step of the target tgt (B, T).

    dec_in (B, T) are the tokens that the decoder reads; by default decoder_inputs(tgt), teacher forcing.
    """
    v, enc = encode(params, cfg, src)
    dec_in = decoder_inputs(tgt) if dec_in is None else dec_in
    top, _, _, caches = run_stack(params, "dec", params["tgt_embed"][dec_in], *v)
    logits = top @ params["out_W"] + params["out_b"]  # (B, T, V)
    return logits, {"enc": enc, "dec_in": dec_in, "dec": caches, "top": top, "logits": logits, "tgt": tgt, "v": v}


def nll(logits, tgt):
    """eq-loss: -(1/B) sum_b sum_t log p(y_t), for logits (B, T, V) and targets (B, T)."""
    logp = log_softmax(logits)
    return -np.take_along_axis(logp, tgt[..., None], axis=-1).sum() / tgt.shape[0]


def loss(params, cfg, src, tgt):
    """The training loss of eq-loss on one batch."""
    return nll(forward(params, cfg, src, tgt)[0], tgt)


def backward(params, cfg, cache):
    """The gradient of eq-loss with respect to every parameter, by hand."""
    logits, tgt, top = cache["logits"], cache["tgt"], cache["top"]
    B = tgt.shape[0]
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    dlogits = np.exp(log_softmax(logits))  # (B, T, V): softmax minus one-hot, over B
    np.put_along_axis(dlogits, tgt[..., None], np.take_along_axis(dlogits, tgt[..., None], axis=-1) - 1.0, axis=-1)
    dlogits /= B
    grads["out_W"] = np.einsum("bth,btv->hv", top, dlogits)
    grads["out_b"] = dlogits.sum(axis=(0, 1))
    dtop = dlogits @ params["out_W"].T  # (B, T, H)
    zeros = np.zeros_like(cache["v"][0])
    dy, dh_v, dc_v = stack_backward(params, "dec", cache["dec"], dtop, zeros, zeros, grads)
    np.add.at(grads["tgt_embed"], cache["dec_in"], dy)
    # The vector v is the only path from the source to the loss: its gradient is the encoder's final-state gradient.
    enc = cache["enc"]
    B, S = enc["order"].shape
    dx, _, _ = stack_backward(params, "enc", enc["caches"], np.zeros((B, S, cfg["hidden_size"])), dh_v, dc_v, grads)
    np.add.at(grads["src_embed"], enc["order"], dx)
    return grads


def loss_and_grads(params, cfg, src, tgt):
    """Return (loss, grads) of eq-loss on one batch."""
    logits, cache = forward(params, cfg, src, tgt)
    return nll(logits, tgt), backward(params, cfg, cache)


def decoder_step(params, tokens, state):
    """One step of the decoder for a batch of previous tokens (B,): return (log p (B, V), new state)."""
    h0, c0 = state
    L = h0.shape[0]
    x = params["tgt_embed"][tokens]  # (B, E)
    h, c = np.empty_like(h0), np.empty_like(c0)
    for l, (w, u, b) in enumerate(layer_names("dec", L)):
        h[l], c[l], _ = lstm_cell(params[w], params[u], params[b], x, h0[l], c0[l])
        x = h[l]
    return log_softmax(x @ params["out_W"] + params["out_b"]), (h, c)


def sequence_log_prob(params, cfg, src, tgt):
    """log p(y | x) (B,) for whole target sequences, by eq-factor: the sum of the log-probabilities of the steps."""
    logp = log_softmax(forward(params, cfg, src, tgt)[0])
    return np.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0].sum(axis=1)


def greedy_decode(params, cfg, src, max_len):
    """Pick the most likely token at every step; return one list per source sentence, ending in <EOS> if it ended."""
    state, _ = encode(params, cfg, src)
    B = src.shape[0]
    tokens, out, done = np.full(B, EOS), [[] for _ in range(B)], np.zeros(B, bool)
    for _ in range(max_len):
        logp, state = decoder_step(params, tokens, state)
        tokens = logp.argmax(axis=-1)
        for i in np.flatnonzero(~done):
            out[i].append(int(tokens[i]))
        done |= tokens == EOS
        if done.all():
            break
    return out


def beam_search(params, cfg, src, beam, max_len):
    """The left-to-right beam search of Sutskever et al. (Section 3.2) for one source sentence src (S,).

    At each step every partial hypothesis is extended by every token, and only the `beam` most likely extensions
    are kept; an extension that ends in <EOS> leaves the beam for the set of complete hypotheses. Return (tokens,
    log p) of the most likely complete hypothesis, or of the most likely partial one if none ended in max_len steps.
    The search stops early once no partial hypothesis can beat the best complete one, since log p only falls.
    """
    if beam < 1:
        raise ValueError("the beam must hold at least one hypothesis")
    (h, c), _ = encode(params, cfg, src[None])
    hyps, scores = [[]], np.zeros(1)  # the partial hypotheses and their log-probabilities
    complete = []
    for _ in range(max_len):
        last = np.array([y[-1] if y else EOS for y in hyps])
        logp, (h, c) = decoder_step(params, last, (h, c))  # one row of the state per partial hypothesis
        total = scores[:, None] + logp  # (n, V)
        flat = np.argsort(-total, axis=None, kind="stable")[:beam]
        keep, new_scores = [], []
        for k in flat:
            j, tok = divmod(int(k), total.shape[1])
            if tok == EOS:
                complete.append((hyps[j] + [tok], float(total[j, tok])))
            else:
                keep.append((j, tok))
                new_scores.append(total[j, tok])
        if not keep:
            break
        rows = np.array([j for j, _ in keep])
        hyps = [hyps[j] + [tok] for j, tok in keep]
        scores, h, c = np.array(new_scores), h[:, rows], c[:, rows]
        if complete and max(s for _, s in complete) >= scores.max():
            break
    if complete:
        return max(complete, key=lambda p: p[1])
    best = int(np.argmax(scores))
    return hyps[best], float(scores[best])


def schedule(kind, i, k, c=0.0, eps=0.0):
    """The probability eps_i of feeding the true previous token at training step i, in scheduled sampling.

    Bengio et al. (2015, Section 2.4): linear max(eps, k - c i), exponential k^i with k < 1, inverse sigmoid
    k / (k + exp(i / k)) with k >= 1.
    """
    if kind == "linear":
        return max(eps, k - c * i)
    if kind == "exponential":
        if not 0 < k < 1:
            raise ValueError("the exponential schedule needs 0 < k < 1")
        return k**i
    if kind == "inverse_sigmoid":
        if k < 1:
            raise ValueError("the inverse sigmoid schedule needs k >= 1")
        return k / (k + np.exp(i / k))
    raise ValueError(f"unknown schedule {kind!r}")


def time_lags(n, reverse):
    """The distance, in steps of the combined sequence, from source word i to the target word i that it aligns to.

    The encoder reads n words at steps 1..n and the decoder predicts target word i at step n + i; reversing the
    source puts word i at step n + 1 - i. Return the n lags, for i = 1..n.
    """
    return [n + i - ((n + 1 - i) if reverse else i) for i in range(1, n + 1)]


def lstm_params(n_in, H):
    """The parameters of one LSTM layer with one bias: 4 H (n_in + H) + 4 H."""
    return 4 * H * (n_in + H) + 4 * H


def stack_params(cfg):
    """The parameters of the encoder, or of the decoder: L LSTM layers, the first reading embeddings."""
    E, H, L = cfg["embed_size"], cfg["hidden_size"], cfg["n_layers"]
    return lstm_params(E, H) + (L - 1) * lstm_params(H, H)


def param_count(cfg):
    """All the parameters: two embeddings, the encoder, the decoder and the softmax layer."""
    E, H, V = cfg["embed_size"], cfg["hidden_size"], cfg["tgt_vocab"]
    return (cfg["src_vocab"] + V) * E + 2 * stack_params(cfg) + H * V + V


def weight_count(cfg):
    """The same without the biases, the count of Sutskever et al. (Section 3.4)."""
    E, H, L, V = cfg["embed_size"], cfg["hidden_size"], cfg["n_layers"], cfg["tgt_vocab"]
    return (cfg["src_vocab"] + V) * E + 2 * (4 * H * (E + H) + (L - 1) * 8 * H * H) + H * V


def recurrent_weight_count(cfg):
    """The weights of the LSTM layers of the encoder and the decoder, without biases."""
    E, H, L = cfg["embed_size"], cfg["hidden_size"], cfg["n_layers"]
    return 2 * (4 * H * (E + H) + (L - 1) * 8 * H * H)


def state_size(cfg):
    """The numbers of the vector v: the hidden and the cell state of every layer, 2 L H."""
    return 2 * cfg["n_layers"] * cfg["hidden_size"]


def flops(cfg, B, S, T):
    """FLOPs of forward(), a multiply-add as 2, the products only: both stacks and the softmax layer."""
    E, H, L, V = cfg["embed_size"], cfg["hidden_size"], cfg["n_layers"], cfg["tgt_vocab"]
    per_step = 8 * H * (E + H) + (L - 1) * 16 * H * H
    return B * (S + T) * per_step + 2 * B * T * H * V
