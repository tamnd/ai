---
id: fourier-identity
derives: [c-fourier-peak, c-fourier-prime, c-fourier-all-frequencies, c-fourier-113]
kind: closed-form
assumes: ["p is a prime", "every weight alpha_k is positive", "every frequency k lies in 1, ..., p - 1"]
uses: []
status: full
check: fourier-identity.check.py
lean: null
sources:
  - {work: nanda-2023-progress-measures-grokking, locator: "p. 3"}
  - {work: nanda-2023-progress-measures-grokking, locator: "p. 4"}
  - {work: nanda-2023-progress-measures-grokking, locator: "p. 6"}
---

### Statement

Let $p$ be a prime, $K \subseteq \{1, \dots, p-1\}$ a nonempty set of frequencies, $w_k = 2\pi k/p$, and $\alpha_k > 0$ for $k \in K$.
For inputs $a, b \in \{0, \dots, p-1\}$ define the logit of each answer $c \in \{0, \dots, p-1\}$ as

$$ \ell(c) = \sum_{k \in K} \alpha_k \cos\big(w_k (a + b - c)\big). $$

Then (i) $\ell(c) = \sum_k \alpha_k$ at $c^* = (a + b) \bmod p$ and $\ell(c) < \sum_k \alpha_k$ at every other $c$, so the largest logit is at $c^*$ and only there;
(ii) $\ell(c)$ is computed from $\cos(w_k a)$, $\sin(w_k a)$, $\cos(w_k b)$, $\sin(w_k b)$, $\cos(w_k c)$ and $\sin(w_k c)$ alone, by
$$ \cos(w_k(a+b)) = \cos(w_k a)\cos(w_k b) - \sin(w_k a)\sin(w_k b), \quad \sin(w_k(a+b)) = \sin(w_k a)\cos(w_k b) + \cos(w_k a)\sin(w_k b) $$
and
$$ \cos(w_k(a+b-c)) = \cos(w_k(a+b))\cos(w_k c) + \sin(w_k(a+b))\sin(w_k c); $$
(iii) with $K = \{1, \dots, (p-1)/2\}$ and every $\alpha_k = 1$, $\ell(c^*) = (p-1)/2$ and $\ell(c) = -1/2$ at every other $c$;
(iv) for $p = 113$, $K = \{14, 35, 41, 42, 52\}$ and every $\alpha_k = 1$, the right answer has logit 5 and the largest wrong logit is $2.5898$, at $c \equiv a + b + 16$, so the rule is right on all $113^2 = 12{,}769$ pairs.
If $p$ is not prime, (i) can fail: a frequency $k$ that shares a factor with $p$ gives a wrong answer the same logit as the right one.

### Assumptions

1. $p$ is prime and $1 \le k \le p - 1$ for every $k \in K$.
2. $\alpha_k > 0$ for every $k \in K$.
3. $a$, $b$ and $c$ are integers; only their residues mod $p$ matter.

### Steps

1. Write $d = (a + b - c) \bmod p$. Since $w_k p = 2\pi k$, each term depends on $a + b - c$ only through $d$: $\cos(w_k(a+b-c)) = \cos(2\pi k d / p)$.
2. At $c = c^*$, $d = 0$ and every cosine is 1, so $\ell(c^*) = \sum_k \alpha_k$.
3. At $c \ne c^*$, $d \in \{1, \dots, p-1\}$. As $p$ is prime and $1 \le k, d \le p-1$, $p$ does not divide $kd$, so $kd/p$ is not an integer and $\cos(2\pi kd/p) < 1$. With $\alpha_k > 0$, every term is strictly below $\alpha_k$ and $\ell(c) < \sum_k \alpha_k$. This is (i).
4. The angle-sum formulas for cosine and sine give the first two identities of (ii); the cosine of a difference, $\cos(x - y) = \cos x \cos y + \sin x \sin y$ with $x = w_k(a+b)$ and $y = w_k c$, gives the third.
5. For (iii), $\sum_{k=0}^{p-1} e^{2\pi i k d/p} = 0$ for $d \not\equiv 0$, since it is a geometric sum of a $p$-th root of unity other than 1. Its real part is $1 + 2\sum_{k=1}^{(p-1)/2} \cos(2\pi kd/p) = 0$, because the terms $k$ and $p - k$ have the same cosine. So the sum over $k = 1, \dots, (p-1)/2$ is $-1/2$; at $d = 0$ it is $(p-1)/2$.
6. For (iv), the wrong logits depend only on $d$, so the 112 values $\sum_{k \in K} \cos(2\pi kd/113)$ for $d = 1, \dots, 112$ cover every wrong answer of every pair; their maximum is $2.5898$ at $d = 97$, that is $c \equiv a + b - 97 \equiv a + b + 16$.
7. For a composite modulus, take $p = 12$ and $k = 3$: at $d = 4$, $kd/p = 1$ and the cosine is 1, so the answer $c^* - 4$ ties with $c^*$.

### Result and check

The rule "answer the $c$ with the largest $\sum_k \alpha_k \cos(w_k(a+b-c))$" computes $a + b \bmod p$ exactly for any prime $p$, any nonempty set of frequencies and any positive weights; a single frequency suffices.
This is the algorithm that Nanda et al. (2023) read from the weights of a one-layer transformer trained on addition mod 113, with the five key frequencies of (iv) (p. 4).
The trig identity on p. 4 of the paper has a misprint, $\cos(w_k a)\cos(w_k a)$ for $\cos(w_k a)\cos(w_k b)$; the form on p. 6 is the one above.
`fourier-identity.check.py` evaluates the logits through the identities of (ii), checks (i) on all pairs for every prime below 60 with random positive weights and random frequency sets, checks (iii) for the same primes, recomputes the numbers of (iv) at $p = 113$, and shows the tie of step 7.
