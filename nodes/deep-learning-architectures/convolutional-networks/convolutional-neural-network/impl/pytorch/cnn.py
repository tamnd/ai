# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/convolutional-neural-network/ (ai://GNXZCP).
"""The convolutional layer and the small network of the page in PyTorch; the NumPy reference is the oracle.

conv2d forms the windows with torch.nn.functional.unfold (im2col) and contracts them with the kernel, so that the
tests can compare it with torch.nn.functional.conv2d, the library's own layer. The network CNN uses the library
layers, torch.nn.Conv2d, torch.nn.MaxPool2d and torch.nn.Linear, with the sizes of shapes.yaml.
"""

import torch
import torch.nn.functional as F

KERNEL = 5
POOL = 2


def config(channels=1, c1=6, c2=16, f1=120, f2=84, classes=10, size=32):
    if min(channels, c1, c2, f1, f2, classes) < 1 or size <= 12 or (size - 12) % 4:
        raise ValueError("all widths must be at least 1, and size - 12 a positive multiple of 4")
    return {"channels": channels, "c1": c1, "c2": c2, "f1": f1, "f2": f2, "classes": classes, "size": size}


def param_count(cfg):
    flat = cfg["c2"] * ((cfg["size"] - 12) // 4) ** 2
    convs = KERNEL**2 * (cfg["channels"] * cfg["c1"] + cfg["c1"] * cfg["c2"]) + cfg["c1"] + cfg["c2"]
    return convs + flat * cfg["f1"] + cfg["f1"] + cfg["f1"] * cfg["f2"] + cfg["f2"] + cfg["f2"] * cfg["classes"] + cfg["classes"]


def conv2d(x, w, b=None, stride=1, pad=0, mode="zeros"):
    """x (B, C, H, W) and w (D, C, k, k) give (B, D, H', W'), by im2col and one matrix product."""
    if mode not in ("zeros", "circular"):
        raise ValueError("mode is zeros or circular")
    if mode == "circular" and pad:
        x = F.pad(x, (pad, pad, pad, pad), mode="circular")  # (B, C, H + 2p, W + 2p)
        pad = 0
    B, _, H, W = x.shape
    D, C, k, _ = w.shape
    Ho, Wo = (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1
    cols = F.unfold(x, k, padding=pad, stride=stride)  # (B, C * k * k, H' * W')
    y = (w.reshape(D, C * k * k) @ cols).reshape(B, D, Ho, Wo)  # eq: eq-conv  (B, D, H', W')
    return y if b is None else y + b[None, :, None, None]


class CNN(torch.nn.Module):
    """Two 5x5 convolutions with ReLU and 2x2 max pooling, then three fully connected layers."""

    def __init__(self, cfg):
        super().__init__()
        flat = cfg["c2"] * ((cfg["size"] - 12) // 4) ** 2
        self.conv1 = torch.nn.Conv2d(cfg["channels"], cfg["c1"], KERNEL)
        self.conv2 = torch.nn.Conv2d(cfg["c1"], cfg["c2"], KERNEL)
        self.pool = torch.nn.MaxPool2d(POOL, POOL)  # eq: eq-maxpool
        self.fc1 = torch.nn.Linear(flat, cfg["f1"])
        self.fc2 = torch.nn.Linear(cfg["f1"], cfg["f2"])
        self.fc3 = torch.nn.Linear(cfg["f2"], cfg["classes"])

    @classmethod
    def from_params(cls, params, cfg):
        """Load the parameter tree of the NumPy reference; its dense weights are (n_in, n_out)."""
        net = cls(cfg).double()
        with torch.no_grad():
            for name in ("conv1", "conv2", "fc1", "fc2", "fc3"):
                w = torch.as_tensor(params[name]["w"])
                getattr(net, name).weight.copy_(w if name.startswith("conv") else w.T)
                getattr(net, name).bias.copy_(torch.as_tensor(params[name]["b"]))
        return net

    def forward(self, x):
        """Return (logits (B, K), cache) for images x (B, C, R, R)."""
        h1 = F.relu(self.conv1(x))  # (B, C1, R - 4, R - 4)
        q1 = self.pool(h1)  # (B, C1, (R - 4) / 2, (R - 4) / 2)
        h2 = F.relu(self.conv2(q1))  # (B, C2, (R - 12) / 2, (R - 12) / 2)
        q2 = self.pool(h2)  # (B, C2, (R - 12) / 4, (R - 12) / 4)
        z1 = F.relu(self.fc1(q2.flatten(1)))  # (B, F1)
        z2 = F.relu(self.fc2(z1))  # (B, F2)
        return self.fc3(z2), {"conv1": h1, "pool1": q1, "conv2": h2, "pool2": q2, "fc1": z1, "fc2": z2}  # (B, K)
