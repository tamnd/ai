---
id: memorization-arithmetic
derives: [c-ma-slope, c-ma-per-copy, c-ma-loglinear, c-ma-zipf, c-ma-uniform]
kind: proof
assumes: ["the expected number of generations of a sequence is a power law in its number of copies", "the extractable fraction is linear in the logarithm of the parameter count", "the frequencies of the subpopulations are fixed numbers, and the examples of a subpopulation share one label drawn from ten balanced classes"]
uses: []
status: full
check: memorization-arithmetic.check.py
lean: null
sources:
  - {work: kandpal-2022-deduplicating-mitigates-privacy, locator: "p. 1"}
  - {work: carlini-2023-quantifying-memorization, locator: "p. 4"}
  - {work: feldman-2019-does-learning-require-memorization, locator: "p. 5"}
---

### Statement

1. (Duplication.) Let a sequence with $d \ge 1$ copies in the training data be generated $E(d) = c\,d^{s}$ times in expectation, with $c > 0$. Then $E(10)/E(1) = 10^{s}$, and the generations per copy are $E(d)/d = c\,d^{s-1}$. Sampling from the training data itself gives $E(d) \propto d$, that is $s = 1$. A ratio $E(10)/E(1) = 1000$ gives $s = 3$, and then $E(10)/10 = 100\,E(1)$.
2. (Model size.) Let the extractable fraction of a model with $N$ parameters be $F(N) = a + b \log_{10} N$ with $b > 0$. Doubling $N$ adds $b \log_{10} 2$, which is $0.0572$ for $b = 0.19$. Since $F \le 1$, the relation cannot hold for $N > 10^{(1 - a)/b}$.
3. (Long tail.) Let $N$ subpopulations have frequencies $\alpha_1, \dots, \alpha_N$ summing to 1, draw $n$ examples, and define
$$ \tau_1 = \frac{\sum_i \alpha_i^2 (1 - \alpha_i)^{n-1}}{\sum_i \alpha_i (1 - \alpha_i)^{n-1}}, \qquad \sigma = \sum_i \alpha_i (1 - \alpha_i)^{n-1} . $$
Then $\sigma$ is the expected fraction of the $n$ examples whose subpopulation appears exactly once. A learner that labels each such example right only with probability $1/10$ leaves $0.9\,n\sigma$ of them wrong in expectation, so the bound of Feldman adds at least $0.9\,\tau_1 n \sigma$ to the optimal error.
   1. For $\alpha_i \propto 1/i$ with $N = n = 50{,}000$: $\tau_1 n \approx 0.472$, $\min_i \alpha_i \cdot n = 1/H_N \approx 0.088$, $\sigma \approx 0.170$ and $0.9\,\tau_1 n \sigma \approx 0.072$.
   2. For $\alpha_i = 1/K$: $\tau_1 = 1/K$ and $\sigma = (1 - 1/K)^{n-1}$. With $n = 50{,}000$ and $K = 5{,}000$, $\sigma \approx 4.5 \times 10^{-5}$ and the added error is about $4.1 \times 10^{-4}$; with $K = 1{,}000$, $\sigma \approx 1.9 \times 10^{-22}$.

### Assumptions

1. Item 1 takes the power law as exact; Kandpal et al. fit it to measured counts, which scatter around the line.
2. Item 2 takes the log-linear fit of Carlini et al. (2023) as exact over the range where it is applied.
3. Item 3 uses the special case of Feldman's bound for examples seen once, with a fixed list of frequencies in place of his random prior, so the expectations over the prior become sums over the subpopulations. The frequencies $\alpha_i = 1/(i H_N)$, with $H_N = \sum_{i \le N} 1/i$, are the plain Zipf law; Feldman's own example smooths the prior, so his figures differ slightly.

### Steps

1. $E(10)/E(1) = c\,10^{s}/c = 10^{s}$, which equals 1000 exactly when $s = 3$. A sampler that draws whole sequences from the training data produces a sequence with probability proportional to its number of copies, so $E(d) \propto d$ and $s = 1$. $E(d)/d = c\,d^{s-1}$ has derivative $c(s-1)d^{s-2} > 0$ when $s > 1$, and for $s = 3$, $E(10)/10 = 100c = 100\,E(1)$.
2. $F(2N) - F(N) = b(\log_{10} 2N - \log_{10} N) = b \log_{10} 2$, and $0.19 \times 0.30103 = 0.0572$. $F(N) > 1$ exactly when $\log_{10} N > (1 - a)/b$.
3. Subpopulation $i$ appears exactly once with probability $n\alpha_i(1 - \alpha_i)^{n-1}$, and then contributes one example, so the expected number of such examples is $n\sigma$. Each is wrong with probability $0.9$, independently of which it is, so the expected count of wrong ones is $0.9\,n\sigma$, and $\tau_1$ times this count is the added error. For equal frequencies both sums of $\tau_1$ have $N = K$ equal terms, and their ratio is $\alpha = 1/K$; $\sigma = K \cdot (1/K)(1 - 1/K)^{n-1}$. The numbers in items 3.1 and 3.2 are evaluated directly: $H_{50000} \approx 11.397$, $(1 - 1/5000)^{49999} = e^{49999 \ln 0.9998} \approx e^{-10.0} \approx 4.54 \times 10^{-5}$, and $0.9 \times 10 \times 4.54 \times 10^{-5} \approx 4.1 \times 10^{-4}$.

### Result and check

$$ \frac{E(10)}{E(1)} = 10^{s}, \qquad F(2N) - F(N) = b \log_{10} 2, \qquad \text{added error} \ge 0.9\,\tau_1 n \sigma \approx \begin{cases} 0.072 & \text{Zipf} \\ 4.1 \times 10^{-4} & \text{uniform}, K = 5000 . \end{cases} $$

The check `memorization-arithmetic.check.py` evaluates the power law at random exponents and confirms items 1 and 2, then computes $\tau_1$, $\sigma$ and the added error for the Zipf and uniform cases in full double precision. It simulates 200 draws of 50,000 examples from the Zipf law and confirms that the measured fraction of examples whose subpopulation appears once is within 0.005 of $\sigma$.
