---
id: units-and-context
derives: [c-unit-conversion, c-bpb-formula, c-canonical-bound, c-sliding-window, c-reduction-scale]
kind: proof
assumes: ["the token, word and byte counts of a text are positive", "the model assigns positive probability to the scored tokens", "1 <= stride <= window"]
uses: [cross-entropy@m2]
status: full
check: units-and-context.check.py
lean: null
sources:
  - {work: gao-2020-pile-800gb-dataset, locator: "p. 6"}
  - {work: mielke-2019-what-kind-of-language-hard-to-language-model, locator: "p. 3"}
---

### Statement

A text has total negative log-likelihood $S$ nats under a model, and $N_T$ tokens, $N_W$ words and $N_B$ UTF-8 bytes.
Write $\ell = S/N_T$ for the per-token loss and $\mathrm{PPL}_U = e^{S/N_U}$ for the perplexity per unit $U$.

1. $\mathrm{PPL}_W = \mathrm{PPL}_T^{\,N_T/N_W}$, and in general $\ln \mathrm{PPL}_U = (N_T/N_U)\,\ell$.
2. The bits per byte are $\mathrm{BPB} = S/(N_B \ln 2) = (N_T/N_B)\,\ell/\ln 2$, and the per-byte perplexity is $2^{\mathrm{BPB}}$.
3. If a model over subword tokens defines the probability of a string $s$ as the sum over its segmentations, $p(s) = \sum_{\sigma} p(\sigma)$, then scoring only the canonical segmentation $\sigma^*$ gives $p(\sigma^*) \le p(s)$, so the string-level perplexity computed from $\sigma^*$ is at least the one computed from $p(s)$.
4. Score a sequence of $n$ tokens with a model of context $W$ by windows that start at $0, s, 2s, \dots$ with $1 \le s \le W$, end at $\min(\text{start} + W, n)$, and score only the tokens that no earlier window scored. Then every token is scored exactly once, there are $1 + \lceil (n - W)/s \rceil$ windows when $n > W$ and one window otherwise, and in every window after the first, each scored token has at least $W - s$ tokens of the window before it.
5. A relative cut $r \in [0, 1)$ of perplexity, from $\mathrm{PPL}$ to $(1 - r)\,\mathrm{PPL}$, removes $-\ln(1 - r)$ nats per unit whatever the starting value, which is a fraction $-\ln(1 - r)/\ln \mathrm{PPL}$ of the cross-entropy.

### Assumptions

- The counts $N_T$, $N_W$, $N_B$ are positive and the scored tokens have positive probability, so $S$ is finite.
- In item 3, every segmentation has nonnegative probability.
- In item 4, $n \ge 1$; the context before a window's start is dropped.
- In item 5, $\mathrm{PPL} > 1$.

### Steps

1. $\ln \mathrm{PPL}_U = S/N_U = (N_T/N_U)(S/N_T)$; for $U = W$, $\mathrm{PPL}_W = e^{(N_T/N_W)\ell} = (e^{\ell})^{N_T/N_W}$.
2. A loss of $S$ nats is $S/\ln 2$ bits; dividing by $N_B$ gives $S/(N_B \ln 2) = (N_T/N_B)\,\ell/\ln 2$, and $2^{\mathrm{BPB}} = e^{S/N_B} = \mathrm{PPL}_B$.
3. The sum includes the term $p(\sigma^*)$ and every other term is nonnegative, so $p(\sigma^*) \le p(s)$; then $-\ln p(\sigma^*)/N_U \ge -\ln p(s)/N_U$ and $\exp$ is increasing.
4. Window $k$ starts at $b_k = ks$. The scored range of window $k$ is $[e_{k-1}, e_k)$ with $e_{-1} = 0$ and $e_k = \min(b_k + W, n)$, and since $e_k$ increases strictly until it reaches $n$, these ranges tile $[0, n)$. The loop stops at the first $k$ with $b_k + W \ge n$, that is $k = \lceil (n - W)/s \rceil$ when $n > W$ and $k = 0$ otherwise. For $k \ge 1$, the previous window was full, $e_{k-1} = b_{k-1} + W = b_k + W - s$, so the first scored token of window $k$ has $e_{k-1} - b_k = W - s$ tokens of the window before it, and later tokens have more.
5. $\ln \mathrm{PPL} - \ln((1 - r)\mathrm{PPL}) = -\ln(1 - r)$; divide by $\ln \mathrm{PPL}$ for the fraction.

### Result and check

Perplexities over different units are powers of each other, with the ratio of the counts as the exponent; bits per byte is the Pile formula; scoring one segmentation bounds the string perplexity from above; strided windows score each token once with at least $W - s$ tokens of in-window context after the first window; and a fixed relative cut of perplexity is a fixed number of nats.
`units-and-context.check.py` checks items 1, 2 and 5 on random counts and losses, item 3 by enumerating every segmentation of short strings under a random unigram model over subwords, and item 4 for all $n \le 200$, $W \le 40$ and $1 \le s \le W$; it ends with `units-and-context: ok`.
