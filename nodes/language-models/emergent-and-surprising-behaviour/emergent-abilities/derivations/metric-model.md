---
id: metric-model
derives: [c-mm-smooth, c-mm-midpoint, c-mm-width, c-mm-linear, c-mm-resolution]
kind: proof
assumes: ["the per-token loss is l(N) = (c/N)^alpha with c > 0 and alpha > 0", "the L tokens of an answer are right independently, each with probability exp(-l(N))", "the n test questions are scored independently"]
uses: []
status: full
check: metric-model.check.py
lean: null
sources:
  - {work: schaeffer-2023-emergent-abilities-mirage, locator: "p. 4"}
  - {work: hu-2024-predicting-emergent-abilities-passuntil, locator: "p. 5"}
---

### Statement

Let $\ell(N) = (c/N)^{\alpha}$ with $c, \alpha > 0$, let $u = \ln N$, and for an answer of $L \ge 1$ tokens let

$$ A_L(N) = e^{-L\ell(N)}, \qquad E_L(N) = L\,(1 - e^{-\ell(N)}) . $$

Then

1. $A_L$ is infinitely differentiable and strictly increasing in $u$;
2. $A_L(N^\ast) = 1/e$ exactly at $N^\ast = c\,L^{1/\alpha}$;
3. $\max_u dA_L/du = \alpha/e$, reached at $N^\ast$, and the interval of $u$ on which $0.1 \le A_L \le 0.9$ has length $\ln(\ln 10 / \ln(10/9))/\alpha$, neither depending on $L$;
4. $\ln(-\ln A_L) = \ln L + \alpha \ln c - \alpha u$, and $L\ell e^{-\ell} \le E_L \le L\ell$, so $E_L/(L\ell) \to 1$ as $N \to \infty$;
5. if each of $n$ questions is answered right independently with probability $A_L$, the number right is $0$ with probability $(1 - A_L)^n \ge 1 - nA_L$.

### Assumptions

1. The loss is a pure power law in $N$ with no irreducible term, as in Schaeffer et al.; they write it as $(N/c)^{\alpha}$ with a negative exponent, which is the same law with $\alpha$ replaced by $-\alpha$.
2. Tokens are right independently, so the probability of a fully right answer is the product of the per-token probabilities.
3. Edit distance is approximated by the expected number of wrong tokens, as in Schaeffer et al.

### Steps

1. $L\ell(N) = L c^{\alpha} e^{-\alpha u}$ is smooth and strictly decreasing in $u$, and $x \mapsto e^{-x}$ is smooth and strictly decreasing, so $A_L = e^{-L\ell}$ is smooth and strictly increasing in $u$.
2. $A_L = 1/e$ exactly when $L\ell = 1$, that is $(c/N)^{\alpha} = 1/L$, that is $N = c\,L^{1/\alpha}$.
3. Put $y = L\ell$. Then $dy/du = -\alpha y$ and $dA_L/du = -e^{-y}\,dy/du = \alpha\,y e^{-y}$. The function $y e^{-y}$ has derivative $(1 - y)e^{-y}$, so its maximum over $y > 0$ is $1/e$ at $y = 1$, the point $N^\ast$ of step 2. The level $A_L = a$ is reached where $y = -\ln a$, that is $u = \ln c + (\ln L - \ln(-\ln a))/\alpha$. The difference between the levels $0.9$ and $0.1$ is $(\ln(-\ln 0.1) - \ln(-\ln 0.9))/\alpha = \ln(\ln 10/\ln(10/9))/\alpha$, in which $L$ cancels.
4. $-\ln A_L = L\ell = L c^{\alpha} N^{-\alpha}$; taking logarithms gives the line. For $x \ge 0$, $1 - e^{-x} \le x$, and $1 - e^{-x} \ge x e^{-x}$ because $e^{x} - 1 \ge x$. With $x = \ell$ and a factor $L$, $L\ell e^{-\ell} \le E_L \le L\ell$, so the ratio $E_L/(L\ell)$ lies in $[e^{-\ell}, 1]$, which tends to $1$ as $\ell \to 0$, that is as $N \to \infty$.
5. The number right is binomial with parameters $n$ and $A_L$, so it is $0$ with probability $(1 - A_L)^n$. Bernoulli's inequality $(1 - a)^n \ge 1 - na$ for $0 \le a \le 1$ and integer $n \ge 1$ gives the bound.

### Result and check

$$ N^\ast = c\,L^{1/\alpha}, \qquad \max_u \frac{dA_L}{du} = \frac{\alpha}{e}, \qquad u_{0.9} - u_{0.1} = \frac{1}{\alpha}\ln\frac{\ln 10}{\ln(10/9)} \approx \frac{3.084}{\alpha} . $$

The check `metric-model.check.py` draws random $c$, $\alpha$ and $L$ and checks items 1 to 4 on a fine grid of $\ln N$, with the slope by a central difference and the levels $0.1$ and $0.9$ by bisection. It checks item 5 against the exact binomial probability and against a simulation with a fixed seed, and it prints, for $\alpha = 0.5$ and $L = 1, 4, 16$, how far right the rise moves while its width stays at about $6.17$ in $\ln N$.
