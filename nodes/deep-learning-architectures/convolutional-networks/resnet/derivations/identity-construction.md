---
id: identity-construction
derives: [c-identity-construction]
kind: construction
assumes: ["the block computes relu(BN_3(F(x)) + x) with an identity shortcut", "the input x of the block is the output of a ReLU, so every entry of x is at least 0", "the added blocks keep the width of the stage"]
uses: [relu@al]
status: full
check: identity-construction.check.py
lean: null
sources:
  - {work: he-2016-deep-residual-learning, locator: "Section 1"}
  - {work: goyal-2017-accurate-large-minibatch-sgd, locator: "Section 5.1"}
---

### Statement

1. A bottleneck block with an identity shortcut whose last batch normalization has scale $\gamma = 0$ and shift $\beta = 0$ returns its input unchanged on every input with non-negative entries, in training and in inference mode.
2. So inserting such a block after any block of a residual network leaves the function of the network unchanged, and a residual network with more blocks of the same widths can compute every function that the shallower one computes.

### Assumptions

1. The block computes $y = \operatorname{relu}(\mathrm{BN}_3(F'(x)) + x)$, where $F'$ is the first two convolutions and normalizations and the third convolution, and $\mathrm{BN}_3(z) = \gamma \hat z + \beta$ with $\hat z$ the normalized $z$. The shortcut is the identity (Eq. 1 of He et al. 2016). Used in step 1.
2. The input $x$ is the output of the stem or of an earlier block, each of which ends with a ReLU, so $x \ge 0$ entry by entry. Used in step 2.
3. The inserted block takes and returns $4 m_s$ channels in stage $s$, the shape of the blocks around it. Used in step 3.

### Steps

1. With $\gamma = 0$ and $\beta = 0$, $\mathrm{BN}_3(z) = 0 \cdot \hat z + 0 = 0$ for any $z$, whether $\hat z$ uses the batch statistics or the running ones, so $y = \operatorname{relu}(0 + x) = \operatorname{relu}(x)$.
2. By assumption 2, $x \ge 0$, and $\operatorname{relu}(x) = \max(x, 0) = x$. So $y = x$.
3. By assumption 3, the inserted block fits between two blocks of the stage, and by step 2 it passes its input through unchanged; the blocks after it see the same input as before. Inserting any number of such blocks keeps the function, which is the construction of He et al. (2016, Section 1): the added layers are identities and the others are copied from the shallower model.
4. Setting $\gamma = 0$ at initialization, as Goyal et al. (2017, Section 5.1) do, starts every block of a network at this identity, so that the signal initially propagates through the shortcuts.

### Result and check

$$ \gamma_3 = \beta_3 = 0,\; x \ge 0 \;\Longrightarrow\; \operatorname{relu}(\mathrm{BN}_3(F'(x)) + x) = x . $$

The check `identity-construction.check.py` builds the blocks of the reference implementation with $\gamma = 0$ in the last normalization, runs them in both modes on random non-negative inputs, and inserts them into a small network to verify that the logits do not change.
