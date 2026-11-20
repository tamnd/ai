---
id: instruction-tuning-arithmetic
derives: [c-it-rates, c-it-epochs, c-it-example, c-it-t0rule, c-it-chain, c-it-grad, c-it-weights, c-it-flops, c-it-we-loss]
kind: proof
assumes: ["datasets are sampled independently with fixed rates and uniformly within a dataset", "the model factorizes the sequence probability by the chain rule", "the loss is the masked mean of per-token cross-entropies over the logits", "training costs about 6 FLOPs per parameter per token"]
uses: []
status: full
check: instruction-tuning-arithmetic.check.py
lean: null
sources:
  - {work: wei-2022-finetuned-language-models-zero-shot, locator: "p. 4"}
  - {work: sanh-2022-multitask-prompted-training, locator: "p. 6"}
  - {work: chung-2022-scaling-instruction-finetuned, locator: "pp. 4 to 6"}
---

### Statement

Let $M$ datasets have $e_1, \dots, e_M$ examples, and let $K$ be the cap.

1. (Capped mixing.) Under examples-proportional mixing with cap $K$, dataset $m$ is sampled with rate $r_m = \min(e_m, K) / Z$, where $Z = \sum_n \min(e_n, K)$. Every dataset with $e_m \ge K$ gets the same rate $K/Z$.
2. (Passes over the data.) After $S$ samples, a dataset with $e_m \le K$ is expected to be passed over $S/Z$ times, the same for all such datasets, and a dataset with $e_m > K$ is passed over $S K / (e_m Z)$ times, fewer.
3. (Example.) For $e = (1000, 3000, 100000)$ and $K = 3000$ the rates are $1/7, 3/7, 3/7$, against $0.0096, 0.0288, 0.9615$ with no cap. After $S = 70{,}000$ samples the datasets are passed over 10, 10 and 0.3 times.
4. (T0's rule.) T0 replaces $e$ by $\tilde e = e$ for $e \le 500{,}000$ and by $500{,}000/T$ above, with $T$ the number of templates. For $T > 1$ this is not monotone: with $T = 10$, $\tilde e(500{,}000) = 500{,}000$ and $\tilde e(500{,}001) = 50{,}000$.
5. (Response-only loss.) For an autoregressive model and a sequence made of a prompt $x_{1:P}$ followed by a response $y_{1:R}$, the sum of the per-token losses at the response positions is $-\log p(y \mid x)$, and the sum over all positions is $-\log p(x) - \log p(y \mid x)$.
6. (Its gradient.) Let $\ell_t = -\log \mathrm{softmax}(z_t)_{y_t}$ and $L = \sum_t m_t \ell_t / \sum_t m_t$, with mask $m_t \in \{0, 1\}$. Then $\partial L / \partial z_t = m_t\,(\mathrm{softmax}(z_t) - \mathbf{e}_{y_t}) / \sum_s m_s$, zero at every masked position.
7. (Token or sequence averaging.) For a batch of sequences with $R_1, \dots, R_B$ response tokens, the token-level mean $\sum_b \sum_t \ell_{b,t} / \sum_b R_b$ gives sequence $b$ the weight $R_b / \sum_c R_c$ on its own mean loss, while the per-sequence mean gives each sequence the weight $1/B$. For $R = (10, 90)$ the weights are $(0.1, 0.9)$ and $(0.5, 0.5)$.
8. (Compute.) With $C \approx 6ND$, PaLM 540B pre-trained on 780B tokens costs $2.53 \times 10^{24}$ FLOPs and finetuning on 1.4B tokens costs $4.54 \times 10^{21}$, about 0.81 of the $5.6 \times 10^{21}$ that Chung et al. list. The token ratio is $1.4/780 \approx 0.18\%$ and the listed FLOP ratio $5.6 \times 10^{21} / 2.5 \times 10^{24} \approx 0.22\%$.
9. (Worked loss.) For the prompt $(5, 6, 7)$, the response $(8, 9)$ and end token 1, the targets are $(6, 7, 8, 9, 1)$ with mask $(0, 0, 1, 1, 1)$. If the model gives the three masked targets probabilities $0.5$, $0.25$ and $0.8$, the summed response loss is $-\log 0.1 = 2.302585$ and its mean is $0.767528$. If it also gives the two prompt targets probabilities $0.01$ and $0.02$, the full-sequence loss sums to $-\log(2 \times 10^{-5}) = 10.819778$, mean $2.163956$.

### Assumptions

1. Items 1 to 3 treat sampling as independent draws of a dataset with probability $r_m$ and then of an example uniformly within it. Shuffling a fixed epoch gives the same expected counts.
2. Item 5 needs a model that factorizes $p(x, y) = \prod_t p(\text{token}_t \mid \text{tokens}_{<t})$, as decoder-only models do. An encoder-decoder model such as T0 conditions on the prompt in the encoder and only ever computes $-\log p(y \mid x)$.
3. Item 8 uses the approximation of about 6 FLOPs per parameter per token for a forward and backward pass. Chung et al. do not say how they counted, and padding or packing can change the count.

### Steps

1. The rates are nonnegative and sum to 1 by construction. If $e_m \ge K$ then $\min(e_m, K) = K$, so $r_m = K/Z$ for every such $m$.
2. In $S$ samples dataset $m$ is drawn $S r_m$ times in expectation, so each of its $e_m$ examples is drawn $S r_m / e_m$ times. For $e_m \le K$ this is $S e_m / (Z e_m) = S/Z$. For $e_m > K$ it is $S K / (e_m Z) < S/Z$.
3. $Z = 1000 + 3000 + 3000 = 7000$, so the rates are $1000/7000 = 1/7$ and $3000/7000 = 3/7$ twice. With no cap $Z = 104{,}000$ and the rates are $1000/104000$, $3000/104000$ and $100000/104000$. With $S = 70{,}000$, $S/Z = 10$ for the two small datasets and $70000 \cdot 3000 / (100000 \cdot 7000) = 0.3$ for the large one.
4. The two values follow from the definition; the jump at $e = 500{,}000$ is from $500{,}000$ down to $500{,}000/T$.
5. By the chain rule, $\log p(x, y) = \sum_{t \le P} \log p(x_t \mid x_{<t}) + \sum_{r \le R} \log p(y_r \mid x, y_{<r})$. The second sum is $\log p(y \mid x)$ and the first is $\log p(x)$.
6. $\partial \ell_t / \partial z_t = \mathrm{softmax}(z_t) - \mathbf{e}_{y_t}$, the standard gradient of softmax cross-entropy, and $\ell_t$ does not depend on $z_s$ for $s \ne t$. The factor $m_t / \sum_s m_s$ carries through.
7. The token-level mean is $\sum_b (R_b / \sum_c R_c) \bar\ell_b$ with $\bar\ell_b = \sum_t \ell_{b,t}/R_b$. For $R = (10, 90)$, $\sum R = 100$ and the weights are $0.1$ and $0.9$.
8. $6 \times 540 \times 10^9 \times 780 \times 10^9 = 2.527 \times 10^{24}$ and $6 \times 540 \times 10^9 \times 1.4 \times 10^9 = 4.536 \times 10^{21}$, and $4.536/5.6 = 0.81$.
9. $0.5 \times 0.25 \times 0.8 = 0.1$, so the sum of the three losses is $\log 10 = 2.302585$, and $2.302585/3 = 0.767528$. Adding the prompt positions multiplies the probability by $0.01 \times 0.02$, giving $-\log(2 \times 10^{-5}) = 10.819778$ over five positions.

### Result and check

$$ r_m = \frac{\min(e_m, K)}{\sum_n \min(e_n, K)}, \qquad \sum_{r} -\log p(y_r \mid x, y_{<r}) = -\log p(y \mid x), \qquad \frac{\partial L}{\partial z_t} = \frac{m_t\,(\mathrm{softmax}(z_t) - \mathbf{e}_{y_t})}{\sum_s m_s} . $$

The check `instruction-tuning-arithmetic.check.py` verifies the rates and pass counts on random dataset sizes and against a simulation of 200,000 draws. It recomputes the worked example and the jump in T0's rule. It verifies the chain-rule identity on a random bigram model by enumerating every sequence of a short length, and the gradient of the masked loss against central finite differences. It recomputes the batch weights, the compute figures and the worked loss.
