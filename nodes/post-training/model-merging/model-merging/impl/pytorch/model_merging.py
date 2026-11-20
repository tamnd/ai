# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/model-merging/ (ai://NEW-model-merging).
"""Model merging on PyTorch state dicts.

A model is a state dict (name -> tensor) of one shared architecture; the methods based on task vectors assume every
fine-tuned model started from the same pre-trained state dict. This file has:

* `uniform_soup` and `interpolate` (WiSE-FT), averages of floating-point tensors;
* `task_vector` and `add_task_vectors`, task arithmetic;
* `ties_merge`, which trims each task vector to its top-k fraction by magnitude over all parameters at once, elects a
  sign per parameter and takes the disjoint mean;
* `dare`, random drop and rescale of a fine-tuned model's delta parameters, with an explicit generator;
* `diagonal_fisher`, the exact diagonal Fisher of a classifier, with the expectation over labels summed over the
  classes, and `fisher_merge`.

Non-floating tensors (counters such as BatchNorm's num_batches_tracked) are copied from the first state dict.
"""

import torch
import torch.nn.functional as F


def _floating(sd):
    return [k for k, v in sd.items() if torch.is_floating_point(v)]


def uniform_soup(states):
    out = dict(states[0])
    for k in _floating(states[0]):
        out[k] = torch.stack([s[k] for s in states]).mean(dim=0)
    return out


def interpolate(state_0, state_1, alpha):
    """(1 - alpha) * state_0 + alpha * state_1, WiSE-FT's weight-space ensemble."""
    out = dict(state_0)
    for k in _floating(state_0):
        out[k] = torch.lerp(state_0[k], state_1[k], alpha)
    return out


def task_vector(state_ft, state_pre):
    return {k: state_ft[k] - state_pre[k] for k in _floating(state_pre)}


def add_task_vectors(state_pre, taus, lam):
    """state_pre + lam * sum of the task vectors."""
    out = dict(state_pre)
    for k in _floating(state_pre):
        out[k] = state_pre[k] + lam * sum(t[k] for t in taus)
    return out


def _flatten(tau, keys):
    return torch.cat([tau[k].reshape(-1) for k in keys])


def _unflatten(vec, like, keys):
    out, i = {}, 0
    for k in keys:
        n = like[k].numel()
        out[k] = vec[i : i + n].view_as(like[k])
        i += n
    return out


def ties_merge(state_pre, states_ft, k=0.2, lam=1.0):
    """TIES-merging: trim (keep the top-k fraction of each task vector by magnitude, over all its parameters),
    elect (the sign of the sum of the trimmed vectors) and disjoint merge (the mean of the entries that agree with
    the elected sign). Returns the merged state dict."""
    keys = _floating(state_pre)
    taus = torch.stack([_flatten(task_vector(s, state_pre), keys) for s in states_ft])  # (n, d)
    keep = round(k * taus.shape[1])
    trimmed = torch.zeros_like(taus)
    if keep > 0:
        idx = torch.topk(taus.abs(), keep, dim=1, sorted=False).indices
        trimmed.scatter_(1, idx, taus.gather(1, idx))
    sign = torch.sign(trimmed.sum(dim=0))
    agree = (torch.sign(trimmed) == sign) & (trimmed != 0)
    count = agree.sum(dim=0)
    tau_m = torch.where(agree, trimmed, torch.zeros_like(trimmed)).sum(dim=0) / count.clamp(min=1)
    return add_task_vectors(state_pre, [_unflatten(tau_m, state_pre, keys)], lam)


def dare(state_ft, state_pre, p, generator=None):
    """Drop each delta parameter with probability p and rescale the survivors by 1 / (1 - p)."""
    if not 0.0 <= p < 1.0:
        raise ValueError("p must be in [0, 1)")
    out = dict(state_ft)
    for k in _floating(state_pre):
        delta = state_ft[k] - state_pre[k]
        keep = torch.rand(delta.shape, generator=generator, dtype=delta.dtype) >= p
        out[k] = state_pre[k] + torch.where(keep, delta / (1 - p), torch.zeros_like(delta))
    return out


def diagonal_fisher(model, inputs):
    """E_x sum_y p(y|x) (d log p(y|x) / d theta)^2 for a classifier returning logits, averaged over the inputs and
    summed exactly over the classes; returns name -> tensor shaped like each parameter."""
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    fisher = {n: torch.zeros_like(p) for n, p in params.items()}
    for x in inputs:
        logp = F.log_softmax(model(x[None]), dim=-1)[0]
        probs = logp.exp().detach()
        for y in range(logp.shape[0]):
            grads = torch.autograd.grad(logp[y], list(params.values()), retain_graph=True)
            for (n, _), g in zip(params.items(), grads, strict=True):
                fisher[n] += probs[y] * g**2
    return {n: f / len(inputs) for n, f in fisher.items()}


def fisher_merge(states, fishers, lambdas=None, target=0, tol=0.0):
    """Per parameter, sum_i lambda_i F_i theta_i / sum_i lambda_i F_i; where the denominator is at most tol the
    parameter keeps its value in states[target]. fishers[i] maps names to diagonal Fishers; names without a Fisher
    (buffers, heads that are not merged) are copied from states[target]."""
    n = len(states)
    lam = [1.0 / n] * n if lambdas is None else list(lambdas)
    out = dict(states[target])
    for k in fishers[0]:
        num = sum(lam[i] * fishers[i][k] * states[i][k] for i in range(n))
        den = sum(lam[i] * fishers[i][k] for i in range(n))
        out[k] = torch.where(den > tol, num / den.clamp(min=torch.finfo(den.dtype).tiny), states[target][k])
    return out
