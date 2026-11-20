# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/bidirectional-recurrent-neural-network/ (ai://Z7Z0J3).
"""A deep bidirectional recurrent network for sequence labelling, the PyTorch version, with the parameters of the
NumPy reference.

Every layer has a forward and a backward direction with their own weights; the output of a layer at step t is
[h_fwd_t, h_bwd_t], the input of the layer above at step t; an affine layer on the top layer gives K class scores
at every step. The cell is an LSTM (one bias, gates in the order of torch.nn.LSTM) or a tanh unit. A padded batch
carries the length of each sequence: the backward direction of a sequence starts at its own last step, and the
outputs at padded steps are zero. to_torch() returns the torch.nn.LSTM or torch.nn.RNN with the same weights.
"""

import torch

CELLS = {"lstm": 4, "tanh": 1}


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


def param_count(cfg):
    I, H, L, K = cfg["input_size"], cfg["hidden_size"], cfg["n_layers"], cfg["n_classes"]  # noqa: E741
    G, D = CELLS[cfg["cell"]], len(directions(cfg))
    return D * (G * H * (I + H) + G * H) + (L - 1) * D * (G * H * (D * H + H) + G * H) + D * H * K + K


def reverse_padded(x, lengths):
    """Reverse the first lengths[b] steps of every sequence b of x (B, T, ...); its own inverse."""
    B, T = x.shape[:2]
    t = torch.arange(T)[None, :]
    n = lengths[:, None]
    idx = torch.where(t < n, n - 1 - t, t)  # (B, T)
    return x[torch.arange(B)[:, None], idx]


def mask_of(lengths, T, dtype):
    return (torch.arange(T)[None, :] < lengths[:, None])[..., None].to(dtype)


class BiRNN(torch.nn.Module):
    def __init__(self, cfg, scale=0.1):
        super().__init__()
        self.cfg = cfg
        I, H, K = cfg["input_size"], cfg["hidden_size"], cfg["n_classes"]  # noqa: E741
        G, D = CELLS[cfg["cell"]], len(directions(cfg))
        shapes = {}
        for l in range(cfg["n_layers"]):  # noqa: E741
            for d in directions(cfg):
                n_in = I if l == 0 else D * H
                shapes |= {f"l{l}_{d}_W": (n_in, G * H), f"l{l}_{d}_U": (H, G * H), f"l{l}_{d}_b": (G * H,)}
        shapes |= {"out_W": (D * H, K), "out_b": (K,)}
        for name, shape in shapes.items():
            self.register_parameter(name, torch.nn.Parameter(torch.empty(shape).uniform_(-scale, scale)))

    @classmethod
    def from_params(cls, params, cfg):
        """A network with the given arrays, by name, as written by the NumPy reference."""
        model = cls(cfg)
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(torch.as_tensor(params[name]))
        return model

    def run_direction(self, l, d, x):  # noqa: E741
        """One direction of layer l over x (B, T, n_in), in its reading order, from a zero state: (B, T, H)."""
        W, U, b = (getattr(self, f"l{l}_{d}_{n}") for n in "WUb")
        B, T, _ = x.shape
        h = c = x.new_zeros(B, self.cfg["hidden_size"])
        hs = []
        for t in range(T):
            z = x[:, t] @ W + h @ U + b
            if self.cfg["cell"] == "tanh":
                h = torch.tanh(z)
            else:
                i, f, g, o = z.chunk(4, dim=-1)
                c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
                h = torch.sigmoid(o) * torch.tanh(c)
            hs.append(h)
        return torch.stack(hs, dim=1)

    def features(self, x, lengths=None):
        """The top layer (B, T, D * H) for the inputs x (B, T, I); zero at the padded steps."""
        B, T, _ = x.shape
        lengths = torch.full((B,), T) if lengths is None else torch.as_tensor(lengths)
        mask = mask_of(lengths, T, x.dtype)
        inp = x
        for l in range(self.cfg["n_layers"]):  # noqa: E741
            outs = []
            for d in directions(self.cfg):
                if d == "fwd":
                    outs.append(self.run_direction(l, d, inp))
                else:
                    outs.append(reverse_padded(self.run_direction(l, d, reverse_padded(inp, lengths)), lengths))
            inp = torch.cat(outs, dim=-1) * mask
        return inp

    def forward(self, x, lengths=None):
        """The class scores (B, T, K) at every step."""
        return self.features(x, lengths) @ self.out_W + self.out_b

    def to_torch(self):
        """The torch.nn.LSTM (or torch.nn.RNN with tanh) with these weights: weight_ih is W.T, weight_hh is U.T,
        bias_ih is b and bias_hh is zero; the backward direction is the parameter set with the suffix _reverse."""
        cfg = self.cfg
        kind = torch.nn.LSTM if cfg["cell"] == "lstm" else torch.nn.RNN
        net = kind(
            cfg["input_size"],
            cfg["hidden_size"],
            num_layers=cfg["n_layers"],
            batch_first=True,
            bidirectional=cfg["bidirectional"],
            dtype=self.out_W.dtype,
        )
        with torch.no_grad():
            for l in range(cfg["n_layers"]):  # noqa: E741
                for d in directions(cfg):
                    suffix = f"l{l}" + ("_reverse" if d == "bwd" else "")
                    getattr(net, f"weight_ih_{suffix}").copy_(getattr(self, f"l{l}_{d}_W").T)
                    getattr(net, f"weight_hh_{suffix}").copy_(getattr(self, f"l{l}_{d}_U").T)
                    getattr(net, f"bias_ih_{suffix}").copy_(getattr(self, f"l{l}_{d}_b"))
                    getattr(net, f"bias_hh_{suffix}").zero_()
        return net


def nll(logits, labels, lengths=None):
    """eq-loss: the cross-entropy of the labels, summed over the real steps of each sequence, averaged over the batch."""
    B, T = labels.shape
    lengths = torch.full((B,), T) if lengths is None else torch.as_tensor(lengths)
    mask = mask_of(lengths, T, logits.dtype)[..., 0]
    logp = torch.log_softmax(logits, dim=-1).gather(-1, labels[..., None])[..., 0]
    return -(logp * mask).sum() / B
