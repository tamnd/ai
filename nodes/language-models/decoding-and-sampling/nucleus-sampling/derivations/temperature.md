---
id: temperature
derives: [c-temp-order, c-temp-limits, c-temp-entropy, c-temp-nucleus, eq-temperature]
kind: proof
assumes: ["u_1, ..., u_V are finite logits", "the tempered distribution is softmax(u / t) with t > 0", "ties among equal logits are broken by a fixed order of the tokens"]
uses: [shannon-entropy@m2]
status: full
check: temperature.check.py
lean: null
sources:
  - {work: holtzman-2020-curious-case-neural-text-degeneration, locator: "p. 6"}
---

### Statement

Let $u_1, \dots, u_{|V|}$ be finite logits and $\pi_t = \operatorname{softmax}(u / t)$ for $t > 0$, as in eq-temperature. Write $\beta = 1/t$. Then

1. $\pi_t(i) > \pi_t(j)$ exactly when $u_i > u_j$, so the order of the tokens, and with it the top-$k$ set, does not depend on $t$;
2. as $t \to 0^+$, $\pi_t$ tends to the uniform distribution on $\arg\max_i u_i$, and as $t \to \infty$ to the uniform distribution on $V$;
3. the entropy $H(\pi_t)$ has derivative $\operatorname{Var}_{\pi_t}(u)/t^3 \ge 0$, so it is nondecreasing in $t$, and strictly increasing unless all logits are equal;
4. for every $m$, the mass $\pi_t$ gives to the $m$ tokens with the largest logits is nonincreasing in $t$, so for a fixed $p$ the nucleus of $\pi_t$ has a size $m(t)$ that is nondecreasing in $t$.

### Assumptions

1. The logits are finite, so every token has positive probability for every $t > 0$.
2. $t > 0$. The limit $t \to 0^+$ is greedy decoding, which eq-temperature does not define at $t = 0$.
3. Equal logits are ordered by a fixed rule, so "the $m$ largest" names one set for every $t$.

### Steps

1. $\pi_t(i)/\pi_t(j) = \exp((u_i - u_j)/t)$, which is above $1$ exactly when $u_i > u_j$, for every $t > 0$.
2. Let $u^\ast = \max_i u_i$ and $A = \arg\max_i u_i$. Dividing numerator and denominator by $e^{u^\ast/t}$ gives $\pi_t(i) = e^{(u_i - u^\ast)/t} / \big(|A| + \sum_{j \notin A} e^{(u_j - u^\ast)/t}\big)$. As $t \to 0^+$ each term with $u_j < u^\ast$ tends to $0$, so $\pi_t(i) \to 1/|A|$ on $A$ and $0$ elsewhere. As $t \to \infty$ every exponent tends to $0$ and $\pi_t(i) \to 1/|V|$.
3. With $Z(\beta) = \sum_i e^{\beta u_i}$, $\log \pi(i) = \beta u_i - \log Z$, so $H = \log Z - \beta\, \mathbb{E}_\pi[u]$. Differentiating, $\frac{d}{d\beta} \log Z = \mathbb{E}_\pi[u]$ and $\frac{d}{d\beta} \mathbb{E}_\pi[u] = \mathbb{E}_\pi[u^2] - \mathbb{E}_\pi[u]^2 = \operatorname{Var}_\pi(u)$. Hence $\frac{dH}{d\beta} = \mathbb{E}_\pi[u] - \mathbb{E}_\pi[u] - \beta \operatorname{Var}_\pi(u) = -\beta \operatorname{Var}_\pi(u)$, and with $\frac{d\beta}{dt} = -1/t^2$, $\frac{dH}{dt} = \operatorname{Var}_\pi(u)/t^3$. The variance is $0$ only if all logits are equal, since every token has positive probability.
4. Let $S$ be the $m$ tokens with the largest logits and $T$ the rest, and let $R(\beta) = \sum_{j \in T} e^{\beta u_j} / \sum_{i \in S} e^{\beta u_i}$, so the mass of $S$ is $1/(1 + R)$. Then $\frac{d}{d\beta} \log R = \mathbb{E}[u \mid T] - \mathbb{E}[u \mid S]$, the two means taken under $\pi$ restricted to each part. Every logit in $T$ is at most every logit in $S$, so this is $\le 0$, and $R$ is nonincreasing in $\beta$. The mass of $S$ is therefore nondecreasing in $\beta$, that is nonincreasing in $t$. By item 1 the sorted order is the same for every $t$, so the nucleus size is the least $m$ whose top-$m$ mass reaches $p$. If $t_1 < t_2$ and the top-$m$ mass reaches $p$ at $t_2$, it reaches $p$ at $t_1$ as well, so $m(t_1) \le m(t_2)$.

### Result and check

$$ \frac{d}{dt} H(\pi_t) = \frac{\operatorname{Var}_{\pi_t}(u)}{t^3} \ge 0, \qquad t_1 < t_2 \implies |V^{(p)}_{t_1}| \le |V^{(p)}_{t_2}| . $$

The check `temperature.check.py` draws random logits, including ties, and checks item 1 on a grid of $t$. It checks the two limits of item 2 at $t$ equal to one fiftieth of the gap between the largest logit and the next distinct one, and at $t = 10^{4}$. It compares the derivative of item 3 with a central difference, and checks over a grid of $t$ that every top-$m$ mass falls and every nucleus size grows. It also evaluates the worked example of the node: the logits $\log(0.5, 0.2, 0.15, 0.1, 0.05)$ at $t = 0.5$ give about $(0.769, 0.123, 0.069, 0.031, 0.008)$, and the nucleus at $p = 0.8$ shrinks from 3 tokens to 2.
