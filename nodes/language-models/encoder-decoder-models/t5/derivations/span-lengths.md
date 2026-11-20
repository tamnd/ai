---
id: span-lengths
derives: [c-span-lengths, c-span-iid, c-span-rate]
kind: count
assumes: ["a sequence of n tokens, of which N = round(r n) are corrupted in m = round(N / mu) spans of mean length mu", "each span replaced by one sentinel in the input; the target is each sentinel followed by its span, and a final sentinel", "for i.i.d. corruption, each token corrupted independently with probability r and each run of consecutive corrupted tokens replaced by one sentinel"]
uses: [linearity-of-expectation@m2]
status: full
check: span-lengths.check.py
lean: null
sources:
  - {work: raffel-2020-t5, locator: "pp. 12 and 13", note: "Section 3.1.4, the sentinels and the targets of the baseline objective"}
  - {work: raffel-2020-t5, locator: "pp. 23 and 24", note: "Section 3.3.4, the span-corruption objective and its example of 500 tokens"}
---

### Statement

1. Span corruption turns $n$ tokens into an input of $n - N + m$ tokens and a target of $N + m + 1$.
2. In the paper's example, 500 tokens at 15% in 25 spans, $N = 75$ and the mean span length is 3; the input has 450 tokens and the target 101.
3. When each token is corrupted independently with probability $r$ and runs are merged, the expected number of spans is $r + (n - 1)\,r(1 - r)$. For long sequences the mean span length tends to $1/(1 - r)$, 1.18 at 15%. At $n = 512$ the expected target has 143.1 tokens and the expected input 500.5, against 104 and 461 for spans of mean length 3.
4. At 50% corruption with a mean span length of 3, the target of 512 tokens grows to 342 tokens, against 104 at 15%; an objective that predicts the whole sequence has a target of 512.

### Assumptions

1. Spans separated by at least one kept token, so that the $m$ sentinels of the input stand between kept tokens. Used in step 1.
2. Independent corruption in statement 3, the i.i.d. objective of Section 3.3.4. Used in step 3.

### Steps

1. The input keeps the $n - N$ uncorrupted tokens and gains one sentinel per span, $n - N + m$. The target holds the $N$ corrupted tokens, a sentinel before each of the $m$ spans and the final sentinel, $N + m + 1$. This is statement 1.
2. With $n = 500$ and $r = 0.15$, $N = 75$; with $m = 25$, $\mu = 75/25 = 3$. Statement 1 gives $500 - 75 + 25 = 450$ and $75 + 25 + 1 = 101$. This is statement 2.
3. A span starts at token $i$ when token $i$ is corrupted and token $i - 1$ is not, or $i$ is the first token. By linearity of expectation the number of spans has mean $r + (n - 1)\,r(1 - r)$. The expected number of corrupted tokens is $nr$, so the mean span length tends to $nr / (n r (1 - r)) = 1/(1 - r)$, $1/0.85 = 1.176$. At $n = 512$ the expected spans are $0.15 + 511 \cdot 0.1275 = 65.30$; statement 1 in expectation gives a target of $76.8 + 65.30 + 1 = 143.1$ and an input of $512 - 76.8 + 65.30 = 500.5$. With $\mu = 3$, $N = \operatorname{round}(76.8) = 77$ and $m = \operatorname{round}(77/3) = 26$, so the target has $77 + 26 + 1 = 104$ tokens and the input $512 - 77 + 26 = 461$. This is statement 3.
4. At $r = 0.5$, $N = 256$ and $m = \operatorname{round}(256/3) = 85$, so the target has $256 + 85 + 1 = 342$ tokens. This is statement 4.

### Result and check

$$ |\text{input}| = n - N + m , \qquad |\text{target}| = N + m + 1 , \qquad \mathbb{E}[m_{\text{i.i.d.}}] = r + (n - 1)\,r(1 - r) . $$

The check `span-lengths.check.py` runs random_spans_mask and span_corrupt of the NumPy reference for many lengths, rates and span lengths, compares the lengths with statement 1, and estimates the i.i.d. counts of statement 3 by simulation.
