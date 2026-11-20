# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sequence-to-sequence-model/ (ai://N149H2).
"""A sequence-to-sequence model of two deep LSTMs, the PyTorch version, with the parameters of the NumPy reference.

The source, reversed when cfg["reverse"] is set, is read by an L-layer LSTM encoder whose final (h, c) of every layer
is the vector v; an L-layer LSTM decoder with its own weights starts from v, reads <EOS>, y_1, ..., y_{T-1} and a
softmax layer gives p(y_t | v, y_<t). Token 0 is <EOS>. forward_scheduled() trains with scheduled sampling (Bengio
et al., 2015): at each step it feeds the true previous token with probability eps and the model's own otherwise.
"""

import torch

EOS = 0


def config(src_vocab=7, tgt_vocab=6, embed_size=3, hidden_size=4, n_layers=2, reverse=True):
    """Return the config of the model."""
    if min(embed_size, hidden_size, n_layers) < 1:
        raise ValueError("the sizes and the number of layers must be at least 1")
    if min(src_vocab, tgt_vocab) < 2:
        raise ValueError("a vocabulary needs <EOS> and at least one other token")
    return {
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "embed_size": embed_size,
        "hidden_size": hidden_size,
        "n_layers": n_layers,
        "reverse": bool(reverse),
    }


def layer_names(prefix, n_layers):
    return [(f"{prefix}{l}_W", f"{prefix}{l}_U", f"{prefix}{l}_b") for l in range(n_layers)]


def decoder_inputs(tgt):
    """<EOS>, y_1, ..., y_{T-1}: the target (B, T) shifted right by one."""
    return torch.cat([torch.full_like(tgt[:, :1], EOS), tgt[:, :-1]], dim=1)


class Seq2Seq(torch.nn.Module):
    def __init__(self, cfg, scale=0.08):
        super().__init__()
        self.cfg = cfg
        E, H, L = cfg["embed_size"], cfg["hidden_size"], cfg["n_layers"]
        shapes = {"src_embed": (cfg["src_vocab"], E), "tgt_embed": (cfg["tgt_vocab"], E)}
        for prefix in ("enc", "dec"):
            for l, (w, u, b) in enumerate(layer_names(prefix, L)):
                shapes |= {w: (E if l == 0 else H, 4 * H), u: (H, 4 * H), b: (4 * H,)}
        shapes |= {"out_W": (H, cfg["tgt_vocab"]), "out_b": (cfg["tgt_vocab"],)}
        for name, shape in shapes.items():  # U(-0.08, 0.08), as in Section 3.4 of Sutskever et al.
            self.register_parameter(name, torch.nn.Parameter(torch.empty(shape).uniform_(-scale, scale)))

    @classmethod
    def from_params(cls, params, cfg):
        """A model with the given arrays, by name, as written by the NumPy reference."""
        model = cls(cfg)
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(torch.as_tensor(params[name]))
        return model

    def cell(self, prefix, l, x, h, c):
        w, u, b = (getattr(self, n) for n in layer_names(prefix, self.cfg["n_layers"])[l])
        z = x @ w + h @ u + b  # (B, 4 * H)
        i, f, g, o = z.chunk(4, dim=-1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c

    def step(self, prefix, x, state):
        """One step of the stack `prefix` on x (B, E) from state (h, c), each (L, B, H); return (top, state)."""
        hs, cs = [], []
        for l in range(self.cfg["n_layers"]):
            h, c = self.cell(prefix, l, x, state[0][l], state[1][l])
            hs.append(h)
            cs.append(c)
            x = h
        return x, (torch.stack(hs), torch.stack(cs))

    def encode(self, src):
        """The vector v = (h, c), each (L, B, H), for the source tokens (B, S)."""
        order = src.flip(1) if self.cfg["reverse"] else src
        x = self.src_embed[order]  # (B, S, E)
        zeros = x.new_zeros(self.cfg["n_layers"], src.shape[0], self.cfg["hidden_size"])
        state = (zeros, zeros)
        for t in range(src.shape[1]):
            _, state = self.step("enc", x[:, t], state)
        return state

    def logits(self, top):
        return top @ self.out_W + self.out_b

    def forward(self, src, tgt, dec_in=None):
        """Teacher-forced logits (B, T, V); dec_in replaces the inputs <EOS>, y_1, ..., y_{T-1} if given."""
        state = self.encode(src)
        dec_in = decoder_inputs(tgt) if dec_in is None else dec_in
        y = self.tgt_embed[dec_in]  # (B, T, E)
        tops = []
        for t in range(dec_in.shape[1]):
            top, state = self.step("dec", y[:, t], state)
            tops.append(top)
        return self.logits(torch.stack(tops, dim=1))

    def forward_scheduled(self, src, tgt, eps, generator=None, sample=True):
        """Scheduled sampling: at every step after the first, a coin with P(heads) = eps chooses, for each sequence,
        the true previous token or the model's own: a token sampled from its last distribution, or its most likely
        token if sample is False. The model's tokens are not differentiated through. Return the logits (B, T, V).
        """
        state = self.encode(src)
        prev = torch.full_like(tgt[:, 0], EOS)
        out = []
        for t in range(tgt.shape[1]):
            top, state = self.step("dec", self.tgt_embed[prev], state)
            logits = self.logits(top)
            out.append(logits)
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                own = torch.multinomial(probs, 1, generator=generator)[:, 0] if sample else probs.argmax(-1)
                coin = torch.rand(tgt.shape[0], generator=generator, dtype=probs.dtype) < eps
            prev = torch.where(coin, tgt[:, t], own)
        return torch.stack(out, dim=1)

    def greedy(self, src, max_len):
        """The most likely token at every step, (B, max_len), with <EOS> repeated after a sequence has ended."""
        state = self.encode(src)
        prev = torch.full((src.shape[0],), EOS, dtype=torch.long)
        out, done = [], torch.zeros(src.shape[0], dtype=torch.bool)
        for _ in range(max_len):
            top, state = self.step("dec", self.tgt_embed[prev], state)
            prev = torch.where(done, EOS, self.logits(top).argmax(-1))
            out.append(prev)
            done |= prev == EOS
        return torch.stack(out, dim=1)

    def to_torch_lstm(self, prefix):
        """A torch.nn.LSTM with the weights of the stack `prefix`: weight_ih is W.T, weight_hh U.T, bias_hh zero."""
        E, H, L = self.cfg["embed_size"], self.cfg["hidden_size"], self.cfg["n_layers"]
        lstm = torch.nn.LSTM(E, H, num_layers=L, batch_first=True, dtype=self.out_W.dtype)
        with torch.no_grad():
            for l, (w, u, b) in enumerate(layer_names(prefix, L)):
                getattr(lstm, f"weight_ih_l{l}").copy_(getattr(self, w).T)
                getattr(lstm, f"weight_hh_l{l}").copy_(getattr(self, u).T)
                getattr(lstm, f"bias_ih_l{l}").copy_(getattr(self, b))
                getattr(lstm, f"bias_hh_l{l}").zero_()
        return lstm


def nll(logits, tgt):
    """eq-loss: the negative log-likelihood of each target sequence, summed over its tokens, averaged over the batch."""
    logp = torch.log_softmax(logits, dim=-1)
    return -logp.gather(-1, tgt[..., None]).sum() / tgt.shape[0]


def param_count(cfg):
    E, H, L, V = cfg["embed_size"], cfg["hidden_size"], cfg["n_layers"], cfg["tgt_vocab"]
    stack = 4 * H * (E + H) + 4 * H + (L - 1) * (8 * H * H + 4 * H)
    return (cfg["src_vocab"] + V) * E + 2 * stack + H * V + V
