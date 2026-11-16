# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/long-short-term-memory/ (ai://NEW-long-short-term-memory).
"""One LSTM layer with a forget gate and no peepholes as a torch.nn.Module, the PyTorch reference.

It computes eq-gates, eq-cell and eq-hidden of the page with torch operations, as the NumPy reference does, and
leaves the gradient to autograd. The parameters W, U and b have the names and the layout of the NumPy reference,
with the gates stacked in the order input, forget, cell candidate, output, so from_params loads the shared fixture.
torch.nn.LSTM stores the transposes, W.T as weight_ih_l0 and U.T as weight_hh_l0, and two biases that add up to b.
"""

import torch


def config(input_size=3, hidden_size=4):
    """Return the config of the layer."""
    if min(input_size, hidden_size) < 1:
        raise ValueError("the sizes must be at least 1")
    return {"input_size": input_size, "hidden_size": hidden_size}


class LSTM(torch.nn.Module):
    """One LSTM layer over a batch of sequences (B, T, I)."""

    def __init__(self, cfg, forget_bias=0.0):
        super().__init__()
        n_in, H = cfg["input_size"], cfg["hidden_size"]
        k = H**-0.5
        self.W = torch.nn.Parameter(torch.empty(n_in, 4 * H).uniform_(-k, k))  # (I, 4 * H)
        self.U = torch.nn.Parameter(torch.empty(H, 4 * H).uniform_(-k, k))  # (H, 4 * H)
        self.b = torch.nn.Parameter(torch.empty(4 * H).uniform_(-k, k))  # (4 * H,)
        with torch.no_grad():
            self.b[H : 2 * H] += forget_bias

    @classmethod
    def from_params(cls, params, cfg):
        """Build the layer from the arrays W, U and b, as in fixtures/params-small.npz."""
        model = cls(cfg)
        model.load_state_dict({k: torch.as_tensor(v) for k, v in params.items()}, strict=True)
        return model

    def cell(self, x, h, c):
        """One step: return (h_new, c_new, gates) for x (B, I), h (B, H) and c (B, H)."""
        z = x @ self.W + h @ self.U + self.b  # (B, 4 * H)
        zi, zf, zg, zo = z.chunk(4, dim=-1)  # (B, H) each
        i, f, g, o = (
            torch.sigmoid(zi),
            torch.sigmoid(zf),
            torch.tanh(zg),
            torch.sigmoid(zo),
        )
        c_new = f * c + i * g  # (B, H): the additive update of eq-cell
        h_new = o * torch.tanh(c_new)  # (B, H)
        return h_new, c_new, {"i": i, "f": f, "g": g, "o": o}

    def forward(self, x, h0=None, c0=None):
        """Return (h, cache): h (B, T, H) is the hidden state of every step, as in the NumPy reference."""
        B, T, _ = x.shape
        H = self.U.shape[0]
        h = x.new_zeros(B, H) if h0 is None else h0  # (B, H)
        c = x.new_zeros(B, H) if c0 is None else c0  # (B, H)
        hs, cs, gates = [], [], {"i": [], "f": [], "g": [], "o": []}
        for t in range(T):
            h, c, gs = self.cell(x[:, t], h, c)
            hs.append(h)
            cs.append(c)
            for name, v in gs.items():
                gates[name].append(v)
        cache = {
            "c": torch.stack(cs, 1),
            "h_T": h,
            "c_T": c,
        }  # (B, T, H), (B, H), (B, H)
        cache |= {name: torch.stack(v, 1) for name, v in gates.items()}  # (B, T, H) each
        return torch.stack(hs, 1), cache  # (B, T, H)

    def to_torch_lstm(self):
        """Return a torch.nn.LSTM with batch_first=True and the same weights: bias_ih = b and bias_hh = 0."""
        n_in, H = self.W.shape[0], self.U.shape[0]
        lib = torch.nn.LSTM(n_in, H, batch_first=True).to(self.W.dtype)
        p = dict(lib.named_parameters())
        with torch.no_grad():
            p["weight_ih_l0"].copy_(self.W.T)  # (4 * H, I)
            p["weight_hh_l0"].copy_(self.U.T)  # (4 * H, H)
            p["bias_ih_l0"].copy_(self.b)  # (4 * H,)
            p["bias_hh_l0"].zero_()  # (4 * H,)
        return lib


def param_count(cfg):
    """The number of parameters, from the sizes alone."""
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    return 4 * H * n_in + 4 * H * H + 4 * H
