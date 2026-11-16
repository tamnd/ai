# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/resnet/ (ai://NEW-resnet).
"""The residual network of He et al. (2016), the PyTorch reference.

It computes eq-residual, eq-projection and eq-bottleneck of the page as the NumPy reference does, and leaves the
gradient to autograd. The modules hold their weights as bare parameters named as in the NumPy parameter tree, and
the running statistics of each norm as the buffers mean and var, so from_params loads the shared fixture as it is.
In train mode the norms use the statistics of the batch (and leave the buffers alone); in eval mode, the buffers.
"""

import torch
import torch.nn.functional as F

EPS = 1e-5
STAGES = 4
EXPANSION = 4


def config(width=64, blocks=(3, 4, 6, 3), classes=1000, stride_on="1x1"):
    """Return the config. stride_on is "1x1", as in the paper, or "3x3", as in the ResNet-50 v1.5 of later code."""
    if len(blocks) != STAGES or min(width, classes, *blocks) < 1 or stride_on not in ("1x1", "3x3"):
        raise ValueError("width, classes and the 4 block counts must be at least 1; stride_on is 1x1 or 3x3")
    cfg = {"width": width, "classes": classes, "stride_on": stride_on, "blocks": tuple(blocks)}
    cfg.update({f"n{s + 1}": n for s, n in enumerate(blocks)})
    return cfg


def param_count(cfg):
    """The closed form of shapes.yaml: per block 17 m^2 + 12 m, plus 5 m c_in + 8 m for a projection, and so on."""
    w, K = cfg["width"], cfg["classes"]
    n, c_in = 149 * w + 32 * w * K + K, w
    for s, blocks in enumerate(cfg["blocks"]):
        m = w * 2**s
        n += 5 * m * c_in + 13 * m * m + 20 * m + (blocks - 1) * (17 * m * m + 12 * m)
        c_in = EXPANSION * m
    return n


def _conv_w(c_out, c_in, k):
    """He et al. (2015) initialization: a Gaussian with variance 2 / (k^2 c_in)."""
    return torch.nn.Parameter(torch.randn(c_out, c_in, k, k) * (2.0 / (k * k * c_in)) ** 0.5)


class Norm(torch.nn.Module):
    """Batch normalization of each channel, with a scale gamma, a shift beta and running mean and var."""

    def __init__(self, c):
        super().__init__()
        self.gamma, self.beta = torch.nn.Parameter(torch.ones(c)), torch.nn.Parameter(torch.zeros(c))
        self.register_buffer("mean", torch.zeros(c))
        self.register_buffer("var", torch.ones(c))

    def forward(self, x):
        if self.training:
            return F.batch_norm(x, None, None, self.gamma, self.beta, training=True, eps=EPS)
        return F.batch_norm(x, self.mean, self.var, self.gamma, self.beta, training=False, eps=EPS)


class Bottleneck(torch.nn.Module):
    """1x1 from c_in to m, 3x3 at m, 1x1 from m to 4m; a 1x1 projection shortcut when projection is true."""

    def __init__(self, c_in, m, stride, projection, stride_on="1x1"):
        super().__init__()
        self.s1, self.s2 = (stride, 1) if stride_on == "1x1" else (1, stride)
        self.stride = stride
        self.conv1, self.bn1 = _conv_w(m, c_in, 1), Norm(m)
        self.conv2, self.bn2 = _conv_w(m, m, 3), Norm(m)
        self.conv3, self.bn3 = _conv_w(EXPANSION * m, m, 1), Norm(EXPANSION * m)
        if projection:
            self.proj, self.bn_proj = _conv_w(EXPANSION * m, c_in, 1), Norm(EXPANSION * m)

    def forward(self, x):
        a = F.relu(self.bn1(F.conv2d(x, self.conv1, stride=self.s1)))  # (B, m, H', H')
        b = F.relu(self.bn2(F.conv2d(a, self.conv2, stride=self.s2, padding=1)))  # (B, m, H', H')
        c = self.bn3(F.conv2d(b, self.conv3))  # eq: eq-bottleneck  (B, 4m, H', H')
        if hasattr(self, "proj"):
            shortcut = self.bn_proj(F.conv2d(x, self.proj, stride=self.stride))  # eq: eq-projection
        else:
            shortcut = x
        y = F.relu(c + shortcut)  # eq: eq-residual
        return y, {"a": a, "b": b, "c": c, "shortcut": shortcut, "y": y}


class Stem(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.conv, self.bn = _conv_w(w, 3, 7), Norm(w)

    def forward(self, x):
        conv = F.conv2d(x, self.conv, stride=2, padding=3)  # (B, C, R/2, R/2)
        return F.max_pool2d(F.relu(self.bn(conv)), 3, stride=2, padding=1), conv  # (B, C, R/4, R/4)


class Head(torch.nn.Module):
    def __init__(self, c, K):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(K, c) * c**-0.5)
        self.b = torch.nn.Parameter(torch.zeros(K))

    def forward(self, h):
        pooled = h.mean(dim=(2, 3))  # (B, 32C)
        return pooled @ self.W.T + self.b, pooled  # (B, K)


class ResNet(torch.nn.Module):
    def __init__(self, cfg, zero_gamma=False):
        super().__init__()
        self.cfg = cfg
        w = cfg["width"]
        self.stem = Stem(w)
        stages, c_in = [], w
        for s, n in enumerate(cfg["blocks"]):
            m = w * 2**s
            blocks = []
            for i in range(n):
                blocks.append(Bottleneck(c_in, m, 2 if (s > 0 and i == 0) else 1, i == 0, cfg["stride_on"]))
                c_in = EXPANSION * m
                if zero_gamma:
                    with torch.no_grad():
                        blocks[-1].bn3.gamma.zero_()
            stages.append(torch.nn.ModuleList(blocks))
        self.stages = torch.nn.ModuleList(stages)
        self.fc = Head(c_in, cfg["classes"])

    @classmethod
    def from_params(cls, params, cfg, state=None):
        """Build the model from flat trees, their paths joined by "/", as in fixtures/params-small.npz."""
        model = cls(cfg)
        flat = dict(params) | dict(state or {})
        tensors = {k.replace("/", "."): torch.as_tensor(v) for k, v in flat.items()}
        model.load_state_dict(tensors, strict=state is not None)
        return model

    def forward(self, x):
        """Return (logits (B, K), cache) for images x (B, 3, R, R)."""
        h, conv = self.stem(x)
        cache = {"stem": {"conv": conv, "y": h}, "blocks": {}}
        for s, stage in enumerate(self.stages):
            for i, block in enumerate(stage):
                h, cache["blocks"][(s, i)] = block(h)
        logits, cache["pooled"] = self.fc(h)
        return logits, cache
