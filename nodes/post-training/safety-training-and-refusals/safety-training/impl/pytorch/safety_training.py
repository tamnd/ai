# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/safety-training/ (ai://NEW-safety-training).
"""Safety training's model-level pieces in PyTorch, on a tiny decoder.

* context_distillation_loss: KL from the teacher (the same model with a safety preprompt, under no_grad) to the
  student (no preprompt), over the response tokens only.
* refusal_direction, ablation hooks, activation addition and orthogonalize_, which folds directional ablation into
  the weights: every matrix and bias that writes to the residual stream loses its component along the direction.
* gcg_step: one step of greedy coordinate gradient on an adversarial suffix, with the gradient taken through a
  one-hot matrix times the embedding table.
"""

import torch
import torch.nn.functional as F
from torch import nn


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.mlp_in = nn.Linear(d_model, d_ff)
        self.mlp_out = nn.Linear(d_ff, d_model)

    def attention(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.attn_out(y.transpose(1, 2).reshape(b, t, d))

    def forward(self, x, hook):
        x = hook(x + self.attention(self.ln1(x)))
        return hook(x + self.mlp_out(F.gelu(self.mlp_in(self.ln2(x)))))


class TinyDecoder(nn.Module):
    """A pre-norm decoder. The residual stream is written by tok, pos, and each block's attn_out and mlp_out."""

    def __init__(self, vocab, d_model=32, n_heads=4, d_ff=64, n_layers=2, max_len=32):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_ff) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, ids=None, embeds=None, hook=None, add_at=None, return_stream=False):
        """Logits for token ids, or for input embeddings (used by GCG).

        hook(x) is applied to every residual activation: after the embeddings and after each attention and MLP
        write. add_at = (layer, vector) adds the vector to the stream entering that block."""
        hook = hook or (lambda x: x)
        x = self.tok(ids) if embeds is None else embeds
        x = hook(x + self.pos(torch.arange(x.shape[1], device=x.device)))
        stream = [x]
        for i, block in enumerate(self.blocks):
            if add_at is not None and add_at[0] == i:
                x = x + add_at[1]
            x = block(x, hook)
            stream.append(x)
        logits = self.head(self.ln_f(x))
        return (logits, stream) if return_stream else logits


# Context distillation.


def context_distillation_loss(model, prompt_ids, preprompt_ids, response_ids):
    """Mean over response tokens of KL(p(. | preprompt, prompt, y<t) || q(. | prompt, y<t)).

    The teacher is the model itself with the safety preprompt in front, evaluated without gradients; the student is
    the model without it. Llama 2 instead samples a response from the teacher and fine-tunes on it, which estimates
    the same objective up to the teacher's entropy."""
    student_in = torch.cat([prompt_ids, response_ids], dim=1)
    teacher_in = torch.cat([preprompt_ids, prompt_ids, response_ids], dim=1)
    n = response_ids.shape[1]
    with torch.no_grad():
        teacher = F.log_softmax(model(teacher_in)[:, -n - 1 : -1], dim=-1)
    student = F.log_softmax(model(student_in)[:, -n - 1 : -1], dim=-1)
    return F.kl_div(student, teacher, log_target=True, reduction="none").sum(-1).mean()


# The refusal direction.


@torch.no_grad()
def mean_activation(model, prompts, layer, position=-1):
    """Mean residual activation entering block `layer` (stream index) at a token position, over a list of prompts."""
    acts = [model(ids=p, return_stream=True)[1][layer][:, position] for p in prompts]
    return torch.cat(acts).mean(0)


def refusal_direction(model, harmful, harmless, layer, position=-1):
    """Difference of mean activations between harmful and harmless prompts, and its unit vector."""
    r = mean_activation(model, harmful, layer, position) - mean_activation(model, harmless, layer, position)
    return r, r / r.norm()


def ablation_hook(r_hat):
    """x -> x - (x . r_hat) r_hat on the last axis."""
    return lambda x: x - (x @ r_hat).unsqueeze(-1) * r_hat


@torch.no_grad()
def orthogonalize_(model, r_hat):
    """Edit the weights in place so that nothing writes along r_hat.

    nn.Linear stores weight as (out, in) and computes x W^T + b, so for a write to the residual stream the edit is
    W <- (I - r r^T) W and b <- (I - r r^T) b; embedding tables have the residual dimension last and become E (I -
    r r^T)."""
    proj = torch.eye(r_hat.numel(), dtype=r_hat.dtype) - torch.outer(r_hat, r_hat)
    model.tok.weight.copy_(model.tok.weight @ proj)
    model.pos.weight.copy_(model.pos.weight @ proj)
    for block in model.blocks:
        for lin in (block.attn_out, block.mlp_out):
            lin.weight.copy_(proj @ lin.weight)
            lin.bias.copy_(proj @ lin.bias)
    return model


# Greedy coordinate gradient.


def target_loss(model, embeds, target_ids):
    """Cross-entropy of the target tokens given everything before them; embeds already contains the target."""
    n = target_ids.shape[1]
    logits = model(embeds=embeds)[:, -n - 1 : -1]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))


def token_gradients(model, ids, suffix_slice, target_ids):
    """Gradient of the target loss with respect to the one-hot vectors of the suffix tokens, shape (len, vocab)."""
    w = model.tok.weight
    one_hot = F.one_hot(ids[0, suffix_slice], w.shape[0]).to(w.dtype).requires_grad_(True)
    embeds = model.tok(ids).detach()
    embeds = torch.cat([embeds[:, : suffix_slice.start], (one_hot @ w)[None], embeds[:, suffix_slice.stop :]], dim=1)
    loss = target_loss(model, embeds, target_ids)
    (grad,) = torch.autograd.grad(loss, one_hot)
    return grad


def gcg_step(model, ids, suffix_slice, target_ids, k, batch, generator):
    """One GCG step: top-k tokens per suffix position by -gradient, batch random single-token swaps, exact losses, and
    the best of them. ids must end with target_ids. Returns the new ids and their loss."""
    grad = token_gradients(model, ids, suffix_slice, target_ids)
    top = (-grad).topk(k, dim=-1).indices
    n_pos = top.shape[0]
    cands = ids.repeat(batch, 1)
    pos = torch.randint(n_pos, (batch,), generator=generator)
    choice = torch.randint(k, (batch,), generator=generator)
    cands[torch.arange(batch), suffix_slice.start + pos] = top[pos, choice]
    with torch.no_grad():
        losses = torch.stack([target_loss(model, model.tok(c[None]), target_ids) for c in cands])
    best = losses.argmin()
    return cands[best : best + 1], losses[best]
