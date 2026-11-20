# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/system-prompt/ (ai://NEW-system-prompt).
import torch
import torch.nn.functional as F
from system_prompt import TinyDecoder, cfg_logprobs, masked_lm_loss, persona_vector, split_softmax, trait_projection

torch.set_default_dtype(torch.float64)


def model(seed=0, vocab=29):
    torch.manual_seed(seed)
    return TinyDecoder(vocab).eval()


def test_split_softmax_raises_the_prompt_share_at_every_layer():
    m = model()
    ids = torch.randint(29, (2, 24))
    plain, steered = [], []
    m(ids, record=plain)
    m(ids, n_prompt=6, k=0.5, record=steered)
    with torch.no_grad():
        same = m(ids, n_prompt=6, k=1.0)
        assert torch.allclose(same, m(ids))
    for a in steered:
        assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)))
    # In the first layer the inputs are equal, so the shares are related exactly by pi -> pi^0.5 for queries after
    # the prompt; deeper layers see different inputs and are only checked to sum to 1.
    pi0, pi1 = plain[0][..., 6:, :6].sum(-1), steered[0][..., 6:, :6].sum(-1)
    assert torch.allclose(pi1, pi0**0.5)
    assert torch.allclose(split_softmax(plain[0], 6, 0.5)[..., :6, :], plain[0][..., :6, :])


def test_cfg_reduces_to_prompting_at_alpha_one():
    m = model(1)
    sys_ids, ctx = torch.randint(29, (1, 5)), torch.randint(29, (1, 7))
    with torch.no_grad():
        lc = F.log_softmax(m(torch.cat([sys_ids, ctx], 1))[:, -1], -1)
        lu = F.log_softmax(m(ctx)[:, -1], -1)
        assert torch.allclose(cfg_logprobs(m, sys_ids, ctx, 1.0), lc)
        assert torch.allclose(cfg_logprobs(m, sys_ids, ctx, 0.0), lu)
        g = cfg_logprobs(m, sys_ids, ctx, 3.0)
        assert torch.isclose(g.exp().sum(), torch.tensor(1.0))
        i = (lc - lu).argmax()
        assert g[0, i] > lc[0, i]


def test_masked_loss_only_sees_selected_targets():
    m = model(2)
    ids = torch.randint(29, (1, 20))
    mask = torch.zeros(1, 20)
    mask[:, 15:] = 1  # the last assistant message
    loss = masked_lm_loss(m, ids, mask)
    # Changing an earlier token changes the loss only through the context, never as a target: with a mask of ones on
    # the last 5 positions, the loss equals the mean of those 5 token losses computed directly.
    logits = m(ids)[:, 14:19]
    direct = F.cross_entropy(logits[0], ids[0, 15:20])
    assert torch.isclose(loss, direct)
    # Training on the mask lowers the masked loss.
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    m.train()
    for _ in range(30):
        opt.zero_grad()
        masked_lm_loss(m, ids, mask).backward()
        opt.step()
    assert masked_lm_loss(m, ids, mask).item() < loss.item()


def test_persona_vector_steering_moves_the_projection():
    m = model(3)
    trait = [(torch.tensor([[1, 2, 3, 4, 5, 6]]), 3), (torch.tensor([[1, 2, 3, 4, 7, 6]]), 3)]
    plain = [(torch.tensor([[8, 9, 10, 11, 12, 13]]), 3), (torch.tensor([[8, 9, 10, 11, 14, 13]]), 3)]
    v = persona_vector(m, trait, plain, layer=1)
    assert v.shape == (32,) and v.norm() > 0
    ids = torch.tensor([[8, 9, 10, 11]])
    with torch.no_grad():
        _, s0 = m(ids, return_stream=True)
        _, s1 = m(ids, add_at=(1, 0.5 * v), return_stream=True)
    v_hat = v / v.norm()
    # add_at acts on the stream entering block 1, after stream[1] is recorded; the block output moves by the residual.
    assert torch.allclose(s0[1], s1[1])
    assert not torch.allclose(s0[2], s1[2])
    before = trait_projection(m, ids, v, 1)
    shifted = (s0[1][:, -1] + 0.5 * v) @ v_hat
    assert torch.isclose(shifted - before, 0.5 * v.norm()).all()
