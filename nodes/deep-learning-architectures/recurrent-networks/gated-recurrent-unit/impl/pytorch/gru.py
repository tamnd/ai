# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/gated-recurrent-unit/ (ai://NEW-gated-recurrent-unit).
"""One GRU layer as a torch.nn.Module, the PyTorch reference.

It computes eq-gates, eq-candidate and eq-update of the page with torch operations, as the NumPy reference does, and
leaves the gradient to autograd. The parameters W, U, b_i and b_h have the names and the layout of the NumPy
reference, with the blocks in the order reset, update, candidate, so from_params loads the shared fixture.
torch.nn.GRU stores the transposes, W.T as weight_ih_l0 and U.T as weight_hh_l0, and b_i and b_h as bias_ih_l0 and
bias_hh_l0; with reset="after" the two compute the same function.
"""

import torch

RESETS = ("after", "before")


def config(input_size=3, hidden_size=4):
    """Return the config of the layer."""
    if min(input_size, hidden_size) < 1:
        raise ValueError("the sizes must be at least 1")
    return {"input_size": input_size, "hidden_size": hidden_size}


class GRU(torch.nn.Module):
    """One GRU layer over a batch of sequences (B, T, I)."""

    def __init__(self, cfg, reset="after"):
        super().__init__()
        if reset not in RESETS:
            raise ValueError(f"reset must be one of {RESETS}")
        self.reset = reset
        n_in, H = cfg["input_size"], cfg["hidden_size"]
        k = H**-0.5
        self.W = torch.nn.Parameter(torch.empty(n_in, 3 * H).uniform_(-k, k))  # (I, 3 * H)
        self.U = torch.nn.Parameter(torch.empty(H, 3 * H).uniform_(-k, k))  # (H, 3 * H)
        self.b_i = torch.nn.Parameter(torch.empty(3 * H).uniform_(-k, k))  # (3 * H,)
        self.b_h = torch.nn.Parameter(torch.empty(3 * H).uniform_(-k, k))  # (3 * H,)

    @classmethod
    def from_params(cls, params, cfg, reset="after"):
        """Build the layer from the arrays W, U, b_i and b_h, as in fixtures/params-small.npz."""
        model = cls(cfg, reset)
        model.load_state_dict({k: torch.as_tensor(v) for k, v in params.items()}, strict=True)
        return model

    def cell(self, x, h):
        """One step: return (h_new, gates) for x (B, I) and h (B, H)."""
        H = h.shape[-1]
        a = x @ self.W + self.b_i  # (B, 3 * H)
        ar, az, an = a.split(H, dim=-1)  # (B, H) each
        if self.reset == "after":
            m = h @ self.U + self.b_h  # (B, 3 * H): one product for the three blocks
            mr, mz, mn = m.split(H, dim=-1)
            r, z = torch.sigmoid(ar + mr), torch.sigmoid(az + mz)
            n = torch.tanh(an + r * mn)  # (B, H): eq-candidate
        else:
            m = h @ self.U[:, : 2 * H] + self.b_h[: 2 * H]  # (B, 2 * H)
            mr, mz = m.split(H, dim=-1)
            r, z = torch.sigmoid(ar + mr), torch.sigmoid(az + mz)
            n = torch.tanh(an + (r * h) @ self.U[:, 2 * H :] + self.b_h[2 * H :])  # (B, H): eq-candidate-cho
        h_new = (1.0 - z) * n + z * h  # (B, H): eq-update
        return h_new, {"r": r, "z": z, "n": n}

    def forward(self, x, h0=None):
        """Return (h, cache): h (B, T, H) is the state of every step, as in the NumPy reference."""
        B, T, _ = x.shape
        H = self.U.shape[0]
        h = x.new_zeros(B, H) if h0 is None else h0  # (B, H)
        hs, gates = [], {"r": [], "z": [], "n": []}
        for t in range(T):
            h, gs = self.cell(x[:, t], h)
            hs.append(h)
            for name, v in gs.items():
                gates[name].append(v)
        cache = {"h_T": h} | {name: torch.stack(v, 1) for name, v in gates.items()}  # (B, H), (B, T, H) each
        return torch.stack(hs, 1), cache  # (B, T, H)

    def to_torch_gru(self):
        """Return a torch.nn.GRU with batch_first=True and the same weights; it matches reset="after" only."""
        n_in, H = self.W.shape[0], self.U.shape[0]
        lib = torch.nn.GRU(n_in, H, batch_first=True).to(self.W.dtype)
        p = dict(lib.named_parameters())
        with torch.no_grad():
            p["weight_ih_l0"].copy_(self.W.T)  # (3 * H, I)
            p["weight_hh_l0"].copy_(self.U.T)  # (3 * H, H)
            p["bias_ih_l0"].copy_(self.b_i)  # (3 * H,)
            p["bias_hh_l0"].copy_(self.b_h)  # (3 * H,)
        return lib


def param_count(cfg):
    """The number of parameters, from the sizes alone."""
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    return 3 * H * n_in + 3 * H * H + 6 * H
