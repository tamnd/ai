# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/bidirectional-recurrent-neural-network/ (ai://Z7Z0J3).
"""A deep bidirectional recurrent network for sequence labelling, the NumPy reference.

Every layer has a forward direction, which reads its input from t = 1 to T (eq-forward), and a backward direction
with its own weights, which reads the same input from t = T to 1 (eq-backward). The output of the layer at step t
is the concatenation [h_fwd_t, h_bwd_t] (eq-concat), and it is the input of the next layer at step t, as in the deep
bidirectional LSTM of Graves, Mohamed and Hinton (2013). An affine layer on the top layer gives a score for each of
K classes at every step (eq-output), and the loss is the cross-entropy of the labels, summed over the steps of each
sequence and averaged over the batch (eq-loss).

The cell of both directions is an LSTM (cfg["cell"] = "lstm", one bias, the gates stacked as input, forget, cell
candidate and output, the order of torch.nn.LSTM) or a tanh unit ("tanh"). With cfg["bidirectional"] False only
the forward directions exist, and the network is an ordinary deep unidirectional one.

A batch may hold sequences of different lengths: lengths[b] is the number of real steps of sequence b, the steps
after it are padding. The backward direction of sequence b starts at its own last real step, not at T, and the
outputs at padded steps are zero, which is what torch.nn.utils.rnn.pack_padded_sequence does. Both directions
start from a zero state. backward() differentiates eq-loss by hand.
"""

import numpy as np

CELLS = {"lstm": 4, "tanh": 1}  # the number of blocks of weights of each cell


def config(input_size=3, hidden_size=4, n_layers=2, n_classes=5, cell="lstm", bidirectional=True):
    """Return the config of the network."""
    if min(input_size, hidden_size, n_layers) < 1:
        raise ValueError("the sizes and the number of layers must be at least 1")
    if n_classes < 2:
        raise ValueError("a classifier needs at least two classes")
    if cell not in CELLS:
        raise ValueError(f"cell must be one of {sorted(CELLS)}")
    return {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "n_layers": n_layers,
        "n_classes": n_classes,
        "cell": cell,
        "bidirectional": bool(bidirectional),
    }


def directions(cfg):
    return ("fwd", "bwd") if cfg["bidirectional"] else ("fwd",)


def param_shapes(cfg):
    """The name and shape of every parameter, in a fixed order."""
    I, H, K = cfg["input_size"], cfg["hidden_size"], cfg["n_classes"]  # noqa: E741
    G, D = CELLS[cfg["cell"]], len(directions(cfg))
    shapes = {}
    for l in range(cfg["n_layers"]):  # noqa: E741
        for d in directions(cfg):
            n_in = I if l == 0 else D * H  # an upper layer reads both directions of the layer below
            shapes |= {f"l{l}_{d}_W": (n_in, G * H), f"l{l}_{d}_U": (H, G * H), f"l{l}_{d}_b": (G * H,)}
    return shapes | {"out_W": (D * H, K), "out_b": (K,)}


def init(rng, cfg, scale=0.1):
    """Return the parameters, all uniform in [-scale, scale]."""
    return {name: rng.uniform(-scale, scale, shape) for name, shape in param_shapes(cfg).items()}


def param_count(cfg):
    """eq-params: G H (I_l + H) + G H per direction and layer, I_1 = I and I_l = D H above, plus D H K + K."""
    I, H, L, K = cfg["input_size"], cfg["hidden_size"], cfg["n_layers"], cfg["n_classes"]  # noqa: E741
    G, D = CELLS[cfg["cell"]], len(directions(cfg))
    first = D * (G * H * (I + H) + G * H)
    upper = (L - 1) * D * (G * H * (D * H + H) + G * H)
    return first + upper + D * H * K + K


def flops(cfg, B, T):
    """FLOPs of one forward pass over B sequences of T steps: the matrix products only, a multiply-add as 2."""
    I, H, L, K = cfg["input_size"], cfg["hidden_size"], cfg["n_layers"], cfg["n_classes"]  # noqa: E741
    G, D = CELLS[cfg["cell"]], len(directions(cfg))
    layers = D * 2 * B * T * G * H * (I + H) + (L - 1) * D * 2 * B * T * G * H * (D * H + H)
    return layers + 2 * B * T * D * H * K


def sigmoid(z):
    """The logistic function, without overflow for large negative z."""
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def log_softmax(z):
    """log softmax over the last axis."""
    m = z.max(axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))


def check_lengths(lengths, B, T):
    lengths = np.full(B, T) if lengths is None else np.asarray(lengths)
    if lengths.shape != (B,) or lengths.min() < 1 or lengths.max() > T:
        raise ValueError("lengths must hold one length in 1..T per sequence")
    return lengths


def reverse_padded(x, lengths):
    """Reverse the first lengths[b] steps of every sequence b of x (B, T, ...); the padding stays where it is.

    This is the order in which the backward direction reads a padded batch. It is its own inverse.
    """
    B, T = x.shape[:2]
    t = np.arange(T)[None, :]
    n = np.asarray(lengths)[:, None]
    return x[np.arange(B)[:, None], np.where(t < n, n - 1 - t, t)]


def run_direction(cell, W, U, b, x):
    """Run one direction over x (B, T, n_in), already in its reading order, from a zero state.

    Return the state of every step (B, T, H) and the cache of backward: the input, the states, the cell states and
    the activated gates (or, for a tanh cell, nothing more).
    """
    B, T, _ = x.shape
    H = U.shape[0]
    h, c = np.zeros((B, H)), np.zeros((B, H))
    hs, cs, acts = np.empty((B, T, H)), np.empty((B, T, H)), np.empty((B, T, W.shape[1]))
    for t in range(T):
        z = x[:, t] @ W + h @ U + b  # (B, G * H)
        if cell == "tanh":
            h = np.tanh(z)
            acts[:, t] = h
        else:
            i, f = sigmoid(z[:, :H]), sigmoid(z[:, H : 2 * H])
            g, o = np.tanh(z[:, 2 * H : 3 * H]), sigmoid(z[:, 3 * H :])
            c = f * c + i * g
            h = o * np.tanh(c)
            acts[:, t] = np.concatenate([i, f, g, o], axis=-1)
        hs[:, t], cs[:, t] = h, c
    return hs, {"x": x, "h": hs, "c": cs, "acts": acts}


def direction_backward(cell, W, U, cache, dh):
    """Backpropagation through time for one direction, in its reading order: return (dx, dW, dU, db)."""
    x, hs, cs, acts = cache["x"], cache["h"], cache["c"], cache["acts"]
    B, T, _ = x.shape
    H = U.shape[0]
    dW, dU, db, dx = np.zeros_like(W), np.zeros_like(U), np.zeros(W.shape[1]), np.zeros_like(x)
    dh_next, dc_next = np.zeros((B, H)), np.zeros((B, H))
    for t in reversed(range(T)):
        h_prev = hs[:, t - 1] if t > 0 else np.zeros((B, H))
        dh_t = dh[:, t] + dh_next
        if cell == "tanh":
            dz = dh_t * (1.0 - hs[:, t] ** 2)
        else:
            i, f, g, o = (acts[:, t, k * H : (k + 1) * H] for k in range(4))
            c_prev = cs[:, t - 1] if t > 0 else np.zeros((B, H))
            tc = np.tanh(cs[:, t])
            dc = dc_next + dh_t * o * (1.0 - tc**2)
            dz = np.concatenate(
                [dc * g * i * (1.0 - i), dc * c_prev * f * (1.0 - f), dc * i * (1.0 - g**2), dh_t * tc * o * (1.0 - o)],
                axis=-1,
            )
            dc_next = dc * f
        dW += x[:, t].T @ dz
        dU += h_prev.T @ dz
        db += dz.sum(axis=0)
        dx[:, t] = dz @ W.T
        dh_next = dz @ U.T
    return dx, dW, dU, db


def mask_of(lengths, T):
    """1.0 at the real steps and 0.0 at the padding, (B, T, 1)."""
    return (np.arange(T)[None, :] < np.asarray(lengths)[:, None])[..., None].astype(float)


def forward(params, cfg, x, lengths=None):
    """Return (logits, cache): the class scores (B, T, K) at every step of the inputs x (B, T, I).

    The scores at padded steps are out_b alone, since the top layer is zero there; the loss ignores them.
    """
    B, T, _ = x.shape
    lengths = check_lengths(lengths, B, T)
    mask = mask_of(lengths, T)
    inp, layers = x, []
    for l in range(cfg["n_layers"]):  # noqa: E741
        outs, caches = [], {}
        for d in directions(cfg):
            W, U, b = (params[f"l{l}_{d}_{n}"] for n in "WUb")
            seq = inp if d == "fwd" else reverse_padded(inp, lengths)  # eq-forward reads 1..T, eq-backward T..1
            hs, caches[d] = run_direction(cfg["cell"], W, U, b, seq)
            outs.append(hs if d == "fwd" else reverse_padded(hs, lengths))
        inp = np.concatenate(outs, axis=-1) * mask  # eq-concat: (B, T, D * H)
        layers.append(caches)
    logits = inp @ params["out_W"] + params["out_b"]  # eq-output
    return logits, {"lengths": lengths, "mask": mask, "layers": layers, "top": inp, "logits": logits}


def nll(logits, labels, lengths=None):
    """eq-loss: -(1/B) sum_b sum_{t <= lengths[b]} log p(label_bt | x_b), for logits (B, T, K) and labels (B, T)."""
    B, T = labels.shape
    mask = mask_of(check_lengths(lengths, B, T), T)[..., 0]
    logp = np.take_along_axis(log_softmax(logits), labels[..., None], axis=-1)[..., 0]
    return -(logp * mask).sum() / B


def loss(params, cfg, x, labels, lengths=None):
    """The training loss eq-loss on one batch."""
    return nll(forward(params, cfg, x, lengths)[0], labels, lengths)


def backward(params, cfg, cache, labels):
    """The gradient of eq-loss with respect to every parameter, by hand. Return (grads, dx)."""
    logits, mask, lengths = cache["logits"], cache["mask"], cache["lengths"]
    B = labels.shape[0]
    H = cfg["hidden_size"]
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    dlogits = np.exp(log_softmax(logits))  # softmax minus one-hot, at the real steps, over B
    np.put_along_axis(
        dlogits, labels[..., None], np.take_along_axis(dlogits, labels[..., None], axis=-1) - 1.0, axis=-1
    )
    dlogits *= mask / B
    grads["out_W"] = np.einsum("bti,btk->ik", cache["top"], dlogits)
    grads["out_b"] = dlogits.sum(axis=(0, 1))
    dout = dlogits @ params["out_W"].T  # (B, T, D * H)
    for l in reversed(range(cfg["n_layers"])):  # noqa: E741
        dout = dout * mask
        dinp = 0.0
        for k, d in enumerate(directions(cfg)):
            W, U = params[f"l{l}_{d}_W"], params[f"l{l}_{d}_U"]
            dh = dout[..., k * H : (k + 1) * H]
            if d == "bwd":
                dh = reverse_padded(dh, lengths)
            dx, dW, dU, db = direction_backward(cfg["cell"], W, U, cache["layers"][l][d], dh)
            grads[f"l{l}_{d}_W"], grads[f"l{l}_{d}_U"], grads[f"l{l}_{d}_b"] = dW, dU, db
            dinp = dinp + (dx if d == "fwd" else reverse_padded(dx, lengths))
        dout = dinp
    return grads, dout
