---
id: beam-search
derives: [c-beam-greedy, c-beam-exhaustive, c-beam-stop, c-greedy-miss, c-beam-cost]
kind: algorithm
assumes: ["the beam search of eq-beam: at each step every partial hypothesis is extended by every token, the `beam` most likely extensions are kept, and an extension ending in <EOS> moves to the set of complete hypotheses", "the search returns the most likely complete hypothesis", "log-probabilities are compared exactly, with ties broken in the same order by both searches"]
uses: [beam-search@al]
status: full
check: beam-search.check.py
lean: null
sources:
  - {work: sutskever-2014-sequence-to-sequence-learning, locator: "Section 3.2"}
---

### Statement

1. A beam of 1 is the greedy decoder: it returns the sequence built by taking the most likely token at every step.
2. Over at most $n$ steps and a vocabulary of $V$ tokens, <EOS> included, a beam of at least $V (V - 1)^{n-1}$ never discards an extension, so it returns the most likely of all outputs of at most $n$ tokens that end in <EOS>.
3. Stopping as soon as the best complete hypothesis is at least as likely as every partial one changes nothing in the result.
4. Neither greedy decoding nor a small beam is exact. In the worked example, a model with $V = 3$ whose first-step probabilities are 0.2207 for <EOS>, 0.4918 and 0.2875: greedy returns an output of probability 0.0867, a beam of 2 one of 0.1793, and a beam of 3 the most likely output, <EOS> alone, of 0.2207.
5. Over $n$ steps the search runs the decoder on at most $n \cdot \text{beam}$ hypotheses and scores at most $n \cdot \text{beam} \cdot V$ extensions; an exhaustive search over the same outputs scores $\sum_{t=1}^{n} V (V-1)^{t-1}$.

### Assumptions

1. The probability of a prefix never increases when it is extended (statement 4 of the derivation `factorization`). Used in step 3.
2. Ties are broken the same way by both searches; with a stable sort, by the order of the hypotheses and of the tokens. Used in step 1.

### Steps

1. With a beam of 1 there is one partial hypothesis. Its $V$ extensions are ranked, and only the best is kept: the most likely next token. If that token is <EOS>, the hypothesis completes and the beam is empty, so the search ends with it. This is the greedy rule, step for step. This is statement 1.
2. After $t - 1$ steps without pruning there are at most $(V - 1)^{t-1}$ partial hypotheses, those with no <EOS>. Their extensions at step $t$ number at most $V (V - 1)^{t-1} \le V (V - 1)^{n-1}$. A beam at least that large keeps them all. Every output of at most $n$ tokens ending in <EOS> then enters the complete set, and the search returns the most likely of them. This is statement 2.
3. Every later complete hypothesis extends a current partial one, so by assumption 1 it is at most as likely as that partial one. Once the best complete hypothesis is at least as likely as every partial one, no later hypothesis can beat it. This is statement 3.
4. At the first step greedy takes the token of 0.4918. At the second, <EOS> has 0.3646 and the same token 0.3707, so greedy continues and ends with probability $0.4918 \times 0.3707 \times 0.4755 = 0.0867$. A beam of 2 keeps the two tokens of 0.4918 and 0.2875, never <EOS>, and its best output has 0.1793. A beam of 3 also keeps <EOS> at the first step, which no later output beats. This is statement 4.
5. Each step scores the $V$ extensions of each of at most beam partial hypotheses. Exhaustive scoring extends every open prefix, $(V-1)^{t-1}$ of them at step $t$. This is statement 5.

### Result and check

$$ \text{beam} = 1 \iff \text{greedy}, \qquad \text{beam} \ge V (V-1)^{n-1} \implies \text{exact over } |y| \le n . $$

The check `beam-search.check.py` compares a beam of 1 with greedy decoding and a beam of $V(V-1)^{n-1}$ with brute force on random models, runs the search with and without early stopping, and recomputes the numbers of the worked example.
