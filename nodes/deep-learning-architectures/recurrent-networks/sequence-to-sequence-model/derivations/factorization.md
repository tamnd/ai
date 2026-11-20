---
id: factorization
derives: [c-factor-sum, c-loss-steps, c-normalized, c-prefix-bound]
kind: identity
assumes: ["a decoder that gives, for every prefix y_1..y_{t-1} and the vector v, a distribution p(y_t | v, y_1..y_{t-1}) over a vocabulary that contains <EOS>", "a target sequence ends at its first <EOS>"]
uses: [cross-entropy@m2, maximum-likelihood-estimation@m2]
status: full
check: factorization.check.py
lean: null
sources:
  - {work: sutskever-2014-sequence-to-sequence-learning, locator: "Section 2"}
---

### Statement

1. With $p(y_1, \dots, y_T \mid x) = \prod_{t=1}^{T} p(y_t \mid v, y_1, \dots, y_{t-1})$, the log-likelihood of a target is the sum of the log-probabilities of its tokens, and eq-loss is the sum over the steps of the cross-entropy between the one-hot target and the decoder's distribution, averaged over the batch.
2. Its gradient with respect to the logits $z_t$ of step $t$ is $(\operatorname{softmax}(z_t) - e_{y_t}) / B$.
3. For every $n$, the probabilities of the sequences that end with <EOS> within $n$ steps, plus those of the $n$-token prefixes that contain no <EOS>, add up to 1. So the probabilities of all finite sequences ending in <EOS> add up to at most 1, and to exactly 1 when the probability of a prefix with no <EOS> goes to 0 as its length grows.
4. The probability that a target starts with $y_1, \dots, y_k$ never increases with $k$. In particular, a sequence is never more likely than any of its prefixes followed by <EOS> is, unless the conditionals of the extra tokens are 1.

### Assumptions

1. Each conditional is a softmax, so it is positive and sums to 1 over the vocabulary. Used in steps 2, 3 and 4.
2. A target ends at its first <EOS>, so the sequences that end at different steps are different outcomes. Used in step 3.

### Steps

1. The log of a product is the sum of the logs, so $\log p(y \mid x) = \sum_t \log p(y_t \mid v, y_{<t})$. For a softmax, $-\log p(y_t \mid \cdot) = -\log \operatorname{softmax}(z_t)_{y_t}$ is the cross-entropy between $e_{y_t}$ and $\operatorname{softmax}(z_t)$. Summing over $t$ and averaging over the batch is eq-loss. This is statement 1.
2. The derivative of $-\log \operatorname{softmax}(z)_y$ with respect to $z$ is $\operatorname{softmax}(z) - e_y$. Each step's logits enter only their own term, and the batch average divides by $B$. This is statement 2.
3. Let $P_n$ be the sum of the statement over $n$. $P_0 = 1$, the empty prefix. Each open prefix of $n$ tokens splits by its next token: the <EOS> extension ends a sequence, and the others are open prefixes of $n + 1$ tokens. The conditionals of that split sum to 1, so $P_{n+1} = P_n$. The completed part is non-decreasing in $n$ and bounded by 1, and it equals 1 minus the mass of the open prefixes. This is statement 3.
4. Going from $k$ to $k + 1$ tokens multiplies the probability by one conditional, which is at most 1. This is statement 4.

### Result and check

$$ -\log p(y \mid x) = \sum_{t=1}^{T} -\log p(y_t \mid v, y_{<t}), \qquad \frac{\partial \mathcal L}{\partial z_t} = \frac{\operatorname{softmax}(z_t) - e_{y_t}}{B} . $$

The check `factorization.check.py` adds up the probabilities of every output of up to $n$ tokens on random models, compares the loss with the sum of the per-step cross-entropies, compares the gradient with respect to the logits with $\operatorname{softmax} - e_y$ by finite differences, and checks that the prefix probabilities never increase.
