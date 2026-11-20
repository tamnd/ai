# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoregressive-model/ (ai://NEW-autoregressive-model).
"""Autoregressive models in PyTorch, with autograd for the backward pass: a MADE module built from masked linear
layers (eq-made, eq-made-masks) with the negative log-likelihood of eq-nll, ancestral sampling and the mask ensemble of
eq-ensemble; NADE's shared hidden layer (eq-nade) as one cumulative sum; and a WaveNet-style stack of dilated causal
Conv1d layers with gated units (eq-causal-conv, eq-gated) over mu-law codes (eq-mu-law).

The weights have the shape (n_in, n_out) and a layer computes h @ W + b, as in the NumPy reference, so from_params
loads the same arrays. The convolution weights of the reference have the shape (k, C_in, C_out); Conv1d wants
(C_out, C_in, k), and the sequence is padded on the left only.
"""

import itertools
import math

import torch
import torch.nn.functional as F
from torch import nn


def config(d_data=4, hidden=(6,), direct=True):
    hidden = tuple(int(h) for h in hidden)
    if d_data < 2 or min(hidden, default=1) < 1:
        raise ValueError("MADE needs d_data >= 2 and positive hidden widths")
    if not hidden and not direct:
        raise ValueError("with no hidden layer the direct connection is the only path")
    return {"d_data": int(d_data), "hidden": hidden, "direct": bool(direct)}


def param_shapes(cfg):
    D, widths = cfg["d_data"], (cfg["d_data"],) + cfg["hidden"]
    s = {}
    for i in range(len(cfg["hidden"])):
        s[f"W{i + 1}"], s[f"b{i + 1}"] = (widths[i], widths[i + 1]), (widths[i + 1],)
    if cfg["hidden"]:
        s["V"] = (widths[-1], D)
    s["c"] = (D,)
    if cfg["direct"]:
        s["A"] = (D, D)
    return s


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg, batch):
    """eq-flops: the matrix products of one pass, a multiply-add as 2."""
    w = (cfg["d_data"],) + cfg["hidden"] + (cfg["d_data"],)
    n = sum(a * b for a, b in itertools.pairwise(w)) + (cfg["d_data"] ** 2 if cfg["direct"] else 0)
    return 2 * batch * n


def degrees(cfg, order=None, generator=None):
    """m0 = order (a permutation of 1..D), then each hidden layer's degrees in [min m_{l-1}, D - 1], drawn uniformly
    if a generator is given, else cycled through that range."""
    D = cfg["d_data"]
    m = [torch.arange(1, D + 1) if order is None else torch.as_tensor(order, dtype=torch.long)]
    for n in cfg["hidden"]:
        lo = int(m[-1].min())
        if generator is None:
            m.append(lo + torch.arange(n) % (D - lo))
        else:
            m.append(torch.randint(lo, D, (n,), generator=generator))
    return m


def masks(cfg, m):
    """eq-made-masks, each mask of the shape of its weight."""
    dt = torch.get_default_dtype()
    out = {f"W{i}": (m[i][None, :] >= m[i - 1][:, None]).to(dt) for i in range(1, len(m))}
    if cfg["hidden"]:
        out["V"] = (m[0][None, :] > m[-1][:, None]).to(dt)
    if cfg["direct"]:
        out["A"] = (m[0][None, :] > m[0][:, None]).to(dt)
    return out


class MaskedLinear(nn.Module):
    """h @ (W * M) + b with a fixed 0/1 mask M; the masked entries get zero gradient."""

    mask: torch.Tensor

    def __init__(self, n_in, n_out, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_in, n_out))
        self.bias = nn.Parameter(torch.zeros(n_out)) if bias else None
        self.register_buffer("mask", torch.ones(n_in, n_out))

    def forward(self, h):
        out = h @ (self.weight * self.mask)
        return out if self.bias is None else out + self.bias


class MADE(nn.Module):
    """eq-made: masked ReLU hidden layers, a masked output layer V and a masked direct connection A; forward returns
    the logits of p(x_d = 1 | inputs before d) for every d."""

    def __init__(self, cfg, m=None):
        super().__init__()
        self.cfg = cfg
        widths = (cfg["d_data"],) + cfg["hidden"]
        self.layers = [MaskedLinear(widths[i], widths[i + 1]) for i in range(len(cfg["hidden"]))]
        self.hidden = nn.ModuleList(self.layers)
        self.out = MaskedLinear(widths[-1], cfg["d_data"], bias=False) if cfg["hidden"] else None
        self.c = nn.Parameter(torch.zeros(cfg["d_data"]))
        self.direct = MaskedLinear(cfg["d_data"], cfg["d_data"], bias=False) if cfg["direct"] else None
        for layer in self.layers:
            nn.init.normal_(layer.weight, 0.0, math.sqrt(2.0 / layer.weight.shape[0]))
        self.set_masks(degrees(cfg) if m is None else m)

    def set_masks(self, m):
        """Install the masks of the degrees m; m[0] is the ordering."""
        self.order = m[0]
        mk = masks(self.cfg, m)
        for i, layer in enumerate(self.layers):
            layer.mask.copy_(mk[f"W{i + 1}"])
        if self.out is not None:
            self.out.mask.copy_(mk["V"])
        if self.direct is not None:
            self.direct.mask.copy_(mk["A"])

    def named_params(self):
        """The parameters under the names of the NumPy reference: W1, b1, ..., V, c, A."""
        p = {}
        for i, layer in enumerate(self.layers):
            p[f"W{i + 1}"], p[f"b{i + 1}"] = layer.weight, layer.bias
        if self.out is not None:
            p["V"] = self.out.weight
        p["c"] = self.c
        if self.direct is not None:
            p["A"] = self.direct.weight
        return p

    @classmethod
    def from_params(cls, params, cfg, m):
        model = cls(cfg, m)
        with torch.no_grad():
            for k, v in model.named_params().items():
                v.copy_(torch.as_tensor(params[k]))
        return model

    def forward(self, x):
        h, a = x, self.c
        for layer in self.layers:
            h = torch.relu(layer(h))
        if self.out is not None:
            a = a + self.out(h)
        if self.direct is not None:
            a = a + self.direct(x)
        return a.expand(x.shape[0], -1)

    def log_prob(self, x):
        """log p(x) = sum_d x_d a_d - softplus(a_d), per row."""
        a = self(x)
        return (x * a - F.softplus(a)).sum(1)

    @torch.no_grad()
    def sample(self, n, generator=None):
        """Ancestral sampling: D passes, each drawing the next input in the ordering from its conditional."""
        x = torch.zeros(n, self.cfg["d_data"])
        for d in torch.argsort(self.order):
            p = torch.sigmoid(self(x)[:, d])
            x[:, d] = (torch.rand(n, generator=generator) < p).to(x.dtype)
        return x


def nll(model, x):
    """eq-nll: the mean negative log-likelihood over the batch."""
    return -model.log_prob(x).mean()


def all_binary(D):
    return torch.tensor(list(itertools.product([0.0, 1.0], repeat=D)))


def ensemble_log_prob(model, mask_list, x):
    """eq-ensemble: the log of the mean probability over a list of degree sets sharing the weights."""
    lp = []
    for m in mask_list:
        model.set_masks(m)
        lp.append(model.log_prob(x))
    return torch.logsumexp(torch.stack(lp), 0) - math.log(len(mask_list))


def train(model, data, steps, generator, mask_list=None, batch=100, lr=1e-2):
    """Maximum likelihood with Adam, cycling through the degree sets of mask_list if given; returns the losses."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    trace = []
    for s in range(steps):
        if mask_list:
            model.set_masks(mask_list[s % len(mask_list)])
        idx = torch.randint(0, len(data), (batch,), generator=generator)
        loss = nll(model, data[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        trace.append(loss.item())
    return trace


# NADE: h_i = sigm(c + sum_{j<i} x_j W[j]), p(x_i = 1 | x_<i) = sigm(b_i + h_i . V[:, i]).


def nade_logits(q, x):
    """eq-nade in O(BDH): an exclusive cumulative sum over the rows of W weighted by x."""
    terms = x[:, :, None] * q["W"][None]
    pre = q["c"] + torch.cumsum(terms, 1) - terms
    return q["b"] + torch.einsum("bdh,hd->bd", torch.sigmoid(pre), q["V"])


# WaveNet-style stack over K-valued sequences: logits[t] predict x_t from x_<t.


def wavenet_shapes(K, C, kernel, dilations):
    s: dict[str, tuple[int, ...]] = {"E": (K, C)}
    for i in range(len(dilations)):
        s.update({f"F{i}": (kernel, C, C), f"G{i}": (kernel, C, C), f"R{i}": (C, C), f"S{i}": (C, C)})
    s.update({"O": (C, K), "o": (K,)})
    return s


def receptive_field(kernel, dilations):
    """eq-receptive-field."""
    return 1 + (kernel - 1) * sum(dilations)


def causal_conv(h, W, dilation):
    """eq-causal-conv with Conv1d: h (B, T, C_in), W (k, C_in, C_out) as in the reference; (k - 1) dilation zeros on the
    left and none on the right, so out[t] sees h[t] and earlier."""
    k = W.shape[0]
    hp = F.pad(h.transpose(1, 2), ((k - 1) * dilation, 0))
    return F.conv1d(hp, W.permute(2, 1, 0), dilation=dilation).transpose(1, 2)


def wavenet_logits(w, x, dilations):
    """Logits (B, T, K) for integer sequences x (B, T): embed x shifted by one step, gated residual layers (eq-gated)
    with skip connections, a ReLU and an output layer."""
    h = F.pad(w["E"][x], (0, 0, 1, 0))[:, :-1]
    skip = torch.zeros_like(h)
    for i, d in enumerate(dilations):
        z = torch.tanh(causal_conv(h, w[f"F{i}"], d)) * torch.sigmoid(causal_conv(h, w[f"G{i}"], d))
        skip = skip + z @ w[f"S{i}"]
        h = h + z @ w[f"R{i}"]
    return torch.relu(skip) @ w["O"] + w["o"]


def wavenet_log_prob(w, x, dilations):
    lp = torch.log_softmax(wavenet_logits(w, x, dilations), -1)
    return lp.gather(-1, x[..., None])[..., 0].sum(1)


@torch.no_grad()
def wavenet_sample(w, dilations, n, T, generator=None):
    """T passes, each drawing x_t from the softmax at t given the values drawn so far."""
    x = torch.zeros(n, T, dtype=torch.long)
    for t in range(T):
        pr = torch.softmax(wavenet_logits(w, x, dilations)[:, t], -1)
        x[:, t] = torch.multinomial(pr, 1, generator=generator)[:, 0]
    return x


def mu_law_encode(x, K=256, mu=255.0):
    """eq-mu-law, then K uniform bins of [-1, 1]."""
    y = torch.sign(x) * torch.log1p(mu * x.abs()) / math.log1p(mu)
    return torch.clamp(torch.floor((y + 1.0) / 2.0 * K), 0, K - 1).long()
