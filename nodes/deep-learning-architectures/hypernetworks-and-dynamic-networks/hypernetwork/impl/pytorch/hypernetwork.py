# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/hypernetwork/ (ai://NEW-hypernetwork).
"""Hypernetworks in PyTorch: the static hypernetwork with the convolutional network of the shape contract, the
HyperRNN of eq-scaled, and the HyperLSTM cell of the paper's Appendix A.2.2 with the initialization of A.2.3.

The parameters have the names and the layout of the NumPy reference, so the shared fixture loads as it is
(from_params). The gradients are left to autograd. ConvNet is the same network with ordinary kernels, for the
comparison of the experiment test. The embeddings of the HyperRNN and the HyperLSTM come from the hyper cell's new
state, as in the NumPy reference.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _param(shape):
    return nn.Parameter(torch.zeros(shape, dtype=torch.double))


class StaticHyperNetwork(nn.Module):
    """eq-static: E embeddings of size N_z to E basic kernels (E, N, N, f, f), in the layout of a convolution."""

    def __init__(self, n_emb, n_basic, fsize, n_z):
        super().__init__()
        N, f, d = n_basic, fsize, n_z
        self.f = f
        self.z = nn.Parameter(torch.randn(n_emb, n_z, dtype=torch.double))
        self.W_in = nn.Parameter(torch.randn(N, d, n_z, dtype=torch.double) / math.sqrt(n_z))
        self.B_in = _param((N, d))
        self.W_out = nn.Parameter(torch.randn(f, N * f, d, dtype=torch.double) / math.sqrt(d))
        self.B_out = _param((f, N * f))

    def forward(self):
        a = torch.einsum("idz,ez->eid", self.W_in, self.z) + self.B_in
        K = torch.einsum("uwd,eid->eiuw", self.W_out, a) + self.B_out  # (E, N_in, f, N_out f)
        E, Nin, f, w = K.shape
        return K.reshape(E, Nin, f, w // f, f).permute(0, 3, 1, 2, 4)  # (E, N_out, N_in, f, f)


def tile(basic, m):
    """eq-tile for an m x m grid: (m^2, N, N, f, f) to (m N, m N, f, f)."""
    E, N, _, f, _ = basic.shape
    return basic.reshape(m, m, N, N, f, f).permute(1, 2, 0, 3, 4, 5).reshape(m * N, m * N, f, f)


class HyperConvNet(nn.Module):
    """The network of shapes.yaml: a stem convolution, depth convolutions with generated kernels, ReLUs, a global
    average pool and a linear head."""

    def __init__(self, c_in=1, n_basic=2, mult=2, fsize=3, depth=2, n_z=2, classes=3):
        super().__init__()
        M = mult * n_basic
        self.mult, self.depth, self.pad = mult, depth, fsize // 2
        self.stem_w = nn.Parameter(torch.randn(M, c_in, fsize, fsize, dtype=torch.double) * math.sqrt(2 / (fsize**2 * c_in)))
        self.stem_b = _param((M,))
        self.hyper = StaticHyperNetwork(depth * mult**2, n_basic, fsize, n_z)
        with torch.no_grad():
            self.hyper.W_out.mul_(math.sqrt(2 / (fsize**2 * M)))
        self.conv_b = nn.ParameterList([_param((M,)) for _ in range(depth)])
        self.head_w = nn.Parameter(torch.randn(M, classes, dtype=torch.double) / math.sqrt(M))
        self.head_b = _param((classes,))

    def kernels(self):
        basic, m = self.hyper(), self.mult
        return [tile(basic[j * m * m : (j + 1) * m * m], m) for j in range(self.depth)]

    def forward(self, x):
        h = F.relu(F.conv2d(x, self.stem_w, self.stem_b, padding=self.pad))
        for w, b in zip(self.kernels(), self.conv_b):
            h = F.relu(F.conv2d(h, w, b, padding=self.pad))
        return h.mean((2, 3)) @ self.head_w + self.head_b

    def named_like_numpy(self):
        """The parameters under the names of the NumPy reference."""
        out = {"stem_w": self.stem_w, "stem_b": self.stem_b, "head_w": self.head_w, "head_b": self.head_b}
        out |= {k: getattr(self.hyper, k) for k in ("z", "W_in", "B_in", "W_out", "B_out")}
        return out | {f"conv{j}_b": b for j, b in enumerate(self.conv_b)}

    @classmethod
    def from_params(cls, params, **cfg):
        n_z = params["z"].shape[1]
        model = cls(params["stem_w"].shape[1], params["W_in"].shape[0], cfg.get("mult", 2),
                    params["stem_w"].shape[2], cfg.get("depth", 2), n_z, params["head_w"].shape[1])  # fmt: skip
        with torch.no_grad():
            for k, t in model.named_like_numpy().items():
                t.copy_(torch.as_tensor(params[k]))
        return model


class ConvNet(nn.Module):
    """The same network with ordinary kernels, one weight for every entry."""

    def __init__(self, c_in, width, fsize, depth, classes):
        super().__init__()
        self.pad = fsize // 2
        self.convs = nn.ModuleList([nn.Conv2d(c_in if j == 0 else width, width, fsize, padding=self.pad).double()
                                    for j in range(depth + 1)])  # fmt: skip
        self.head = nn.Linear(width, classes).double()

    def forward(self, x):
        for c in self.convs:
            x = F.relu(c(x))
        return self.head(x.mean((2, 3)))


class HyperRNN(nn.Module):
    """eq-scaled with the hyper cell of eq-hyper-cell, tanh in both."""

    def __init__(self, n_x, n_h, n_hat, n_z):
        super().__init__()
        shapes = {"W_h": (n_h, n_h), "W_x": (n_h, n_x), "W_hz": (n_h, n_z), "W_xz": (n_h, n_z), "W_bz": (n_h, n_z),
                  "b_0": (n_h,), "W_hhat": (n_hat, n_hat), "W_xhat": (n_hat, n_h + n_x), "b_hat": (n_hat,),
                  "W_hh": (n_z, n_hat), "b_hh": (n_z,), "W_hx": (n_z, n_hat), "b_hx": (n_z,), "W_hb": (n_z, n_hat)}  # fmt: skip
        for k, s in shapes.items():
            setattr(self, k, nn.Parameter(torch.randn(s, dtype=torch.double) * 0.5 / math.sqrt(s[-1] if len(s) > 1 else 4)))
        self.n_h, self.n_hat = n_h, n_hat

    def forward(self, x, return_scales=False):
        B, T, _ = x.shape
        h, hhat = x.new_zeros(B, self.n_h), x.new_zeros(B, self.n_hat)
        hs, ds = [], []
        for t in range(T):
            xhat = torch.cat([h, x[:, t]], -1)
            hhat = torch.tanh(hhat @ self.W_hhat.T + xhat @ self.W_xhat.T + self.b_hat)
            z_h, z_x, z_b = hhat @ self.W_hh.T + self.b_hh, hhat @ self.W_hx.T + self.b_hx, hhat @ self.W_hb.T
            d_h, d_x = z_h @ self.W_hz.T, z_x @ self.W_xz.T
            h = torch.tanh(d_h * (h @ self.W_h.T) + d_x * (x[:, t] @ self.W_x.T) + z_b @ self.W_bz.T + self.b_0)
            hs.append(h)
            ds.append(d_h)
        out = torch.stack(hs, 1)
        return (out, torch.stack(ds, 1)) if return_scales else out

    @classmethod
    def from_params(cls, params):
        n_h, n_x = params["W_x"].shape
        model = cls(n_x, n_h, params["W_hhat"].shape[0], params["W_hz"].shape[1])
        with torch.no_grad():
            for k, t in model.named_parameters():
                t.copy_(torch.as_tensor(params[k]))
        return model


def layer_norm(a, eps=1e-5):
    return F.layer_norm(a, a.shape[-1:], eps=eps)


class HyperLSTM(nn.Module):
    """The HyperLSTM, eqs. 10 to 13 of the paper with the dropout of eq. 13 on phi(g_t) in training; every array has
    a leading axis of 4, the gates i, g, f, o. reset_parameters is the initialization of Appendix A.2.3."""

    def __init__(self, n_x, n_h, n_hat, n_z, ln=True, dropout=0.0):
        super().__init__()
        self.shapes = {"W_h": (4, n_h, n_h), "W_x": (4, n_h, n_x), "W_hz": (4, n_h, n_z), "W_xz": (4, n_h, n_z),
                       "W_bz": (4, n_h, n_z), "b_0": (4, n_h), "W_hhat": (4, n_hat, n_hat),
                       "W_xhat": (4, n_hat, n_h + n_x), "b_hat": (4, n_hat), "W_hh": (4, n_z, n_hat), "b_hh": (4, n_z),
                       "W_hx": (4, n_z, n_hat), "b_hx": (4, n_z), "W_hb": (4, n_z, n_hat)}  # fmt: skip
        for k, s in self.shapes.items():
            setattr(self, k, _param(s))
        self.n_h, self.n_hat, self.n_z, self.ln, self.dropout = n_h, n_hat, n_z, ln, dropout
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for k in ("W_h", "W_x", "W_hhat", "W_xhat"):
                for y in range(4):
                    nn.init.orthogonal_(getattr(self, k)[y])
            for k in ("b_hat", "W_hh", "W_hx", "b_0", "W_bz"):
                getattr(self, k).zero_()
            self.b_hh.fill_(1.0)
            self.b_hx.fill_(1.0)
            self.W_hb.normal_(0.0, 0.01)
            self.W_hz.fill_(0.1 / self.n_z)
            self.W_xz.fill_(0.1 / self.n_z)

    def forward(self, x, return_scales=False):
        norm = layer_norm if self.ln else (lambda a: a)
        B, T, _ = x.shape
        h, c = x.new_zeros(B, self.n_h), x.new_zeros(B, self.n_h)
        hhat, chat = x.new_zeros(B, self.n_hat), x.new_zeros(B, self.n_hat)
        hs, ds = [], []
        for t in range(T):
            xhat = torch.cat([h, x[:, t]], -1)
            gh = norm(torch.einsum("ykj,bj->ybk", self.W_hhat, hhat) + torch.einsum("ykj,bj->ybk", self.W_xhat, xhat)
                      + self.b_hat[:, None])  # fmt: skip
            chat = torch.sigmoid(gh[2]) * chat + torch.sigmoid(gh[0]) * torch.tanh(gh[1])
            hhat = torch.sigmoid(gh[3]) * torch.tanh(norm(chat))
            z_h = torch.einsum("yzk,bk->ybz", self.W_hh, hhat) + self.b_hh[:, None]
            z_x = torch.einsum("yzk,bk->ybz", self.W_hx, hhat) + self.b_hx[:, None]
            z_b = torch.einsum("yzk,bk->ybz", self.W_hb, hhat)
            d_h = torch.einsum("ynz,ybz->ybn", self.W_hz, z_h)
            d_x = torch.einsum("ynz,ybz->ybn", self.W_xz, z_x)
            bias = torch.einsum("ynz,ybz->ybn", self.W_bz, z_b) + self.b_0[:, None]
            g = norm(d_h * torch.einsum("ynj,bj->ybn", self.W_h, h) + d_x * torch.einsum("ynj,bj->ybn", self.W_x, x[:, t])
                     + bias)  # fmt: skip
            cand = F.dropout(torch.tanh(g[1]), self.dropout, self.training)
            c = torch.sigmoid(g[2]) * c + torch.sigmoid(g[0]) * cand
            h = torch.sigmoid(g[3]) * torch.tanh(norm(c))
            hs.append(h)
            ds.append(d_h.transpose(0, 1))
        out = torch.stack(hs, 1)
        return (out, torch.stack(ds, 1)) if return_scales else out

    @classmethod
    def from_params(cls, params, ln=True):
        _, n_h, n_x = params["W_x"].shape
        model = cls(n_x, n_h, params["W_hhat"].shape[1], params["W_hz"].shape[2], ln=ln)
        with torch.no_grad():
            for k, t in model.named_parameters():
                t.copy_(torch.as_tensor(params[k]))
        return model
