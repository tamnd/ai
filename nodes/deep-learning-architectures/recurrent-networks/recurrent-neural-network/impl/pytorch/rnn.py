# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/recurrent-neural-network/ (ai://NEW-recurrent-neural-network).
"""One simple (Elman) recurrent layer as a torch.nn.Module, the PyTorch reference.

It computes eq-step of the page with torch operations, as the NumPy reference does, and leaves backpropagation
through time to autograd. The parameters W, U, b_i and b_h have the names and the layout of the NumPy reference, so
from_params loads the shared fixture. torch.nn.RNN stores the transposes, W.T as weight_ih_l0 and U.T as
weight_hh_l0, and b_i and b_h as bias_ih_l0 and bias_hh_l0; the two compute the same function.
"""

import torch

NONLINEARITIES = ("tanh", "relu")


def config(input_size=3, hidden_size=4):
    """Return the config of the layer."""
    if min(input_size, hidden_size) < 1:
        raise ValueError("the sizes must be at least 1")
    return {"input_size": input_size, "hidden_size": hidden_size}


class RNN(torch.nn.Module):
    """One simple recurrent layer over a batch of sequences (B, T, I)."""

    def __init__(self, cfg, nonlinearity="tanh"):
        super().__init__()
        if nonlinearity not in NONLINEARITIES:
            raise ValueError(f"nonlinearity must be one of {NONLINEARITIES}")
        self.nonlinearity = nonlinearity
        n_in, H = cfg["input_size"], cfg["hidden_size"]
        k = H**-0.5
        self.W = torch.nn.Parameter(torch.empty(n_in, H).uniform_(-k, k))  # (I, H)
        self.U = torch.nn.Parameter(torch.empty(H, H).uniform_(-k, k))  # (H, H)
        self.b_i = torch.nn.Parameter(torch.empty(H).uniform_(-k, k))  # (H,)
        self.b_h = torch.nn.Parameter(torch.empty(H).uniform_(-k, k))  # (H,)

    @classmethod
    def from_params(cls, params, cfg, nonlinearity="tanh"):
        """Build the layer from the arrays W, U, b_i and b_h, as in fixtures/params-small.npz."""
        model = cls(cfg, nonlinearity)
        model.load_state_dict({k: torch.as_tensor(v) for k, v in params.items()}, strict=True)
        return model

    def cell(self, x, h):
        """One step of eq-step: return (h_new, a) for x (B, I) and h (B, H)."""
        a = x @ self.W + self.b_i + h @ self.U + self.b_h  # (B, H)
        return (torch.tanh(a) if self.nonlinearity == "tanh" else torch.relu(a)), a

    def forward(self, x, h0=None):
        """Return (h, cache): h (B, T, H) is the state of every step, as in the NumPy reference."""
        B, T, _ = x.shape
        H = self.U.shape[0]
        h = x.new_zeros(B, H) if h0 is None else h0  # (B, H)
        hs, As = [], []
        for t in range(T):
            h, a = self.cell(x[:, t], h)
            hs.append(h)
            As.append(a)
        return torch.stack(hs, 1), {"h_T": h, "a": torch.stack(As, 1)}  # (B, T, H), (B, H), (B, T, H)

    def to_torch_rnn(self):
        """Return a torch.nn.RNN with batch_first=True, the same nonlinearity and the same weights."""
        n_in, H = self.W.shape[0], self.U.shape[0]
        lib = torch.nn.RNN(n_in, H, nonlinearity=self.nonlinearity, batch_first=True).to(self.W.dtype)
        p = dict(lib.named_parameters())
        with torch.no_grad():
            p["weight_ih_l0"].copy_(self.W.T)  # (H, I)
            p["weight_hh_l0"].copy_(self.U.T)  # (H, H)
            p["bias_ih_l0"].copy_(self.b_i)  # (H,)
            p["bias_hh_l0"].copy_(self.b_h)  # (H,)
        return lib


def clip_by_norm(grads, threshold):
    """Algorithm 1 of Pascanu et al.: rescale the whole gradient to norm threshold when its norm is above it."""
    norm = torch.sqrt(sum((g * g).sum() for g in grads))
    if norm >= threshold:
        return [g * (threshold / norm) for g in grads], norm
    return list(grads), norm


def param_count(cfg):
    """The number of parameters, from the sizes alone."""
    n_in, H = cfg["input_size"], cfg["hidden_size"]
    return H * n_in + H * H + 2 * H
