---
id: hallucination-arithmetic
derives: [c-ha-gap, c-ha-toy, c-ha-uniform, c-ha-score, c-ha-penalty, c-ha-ab]
kind: proof
assumes: ["the training facts are n independent draws from a fixed distribution over facts", "the generator cannot tell unseen true facts from false factoids", "the grader's scores are fixed in advance and the model's probability of being right on a question is a number q"]
uses: []
status: full
check: hallucination-arithmetic.check.py
lean: null
sources:
  - {work: kalai-2024-calibrated-must-hallucinate, locator: "pp. 1, 4 and 5"}
  - {work: kalai-2025-why-language-models-hallucinate, locator: "pp. 3, 4 and 13"}
---

### Statement

Let the training data be $n$ independent draws from a distribution $p = (p_1, p_2, \dots)$ over facts. Write $\mathrm{MF}$ for the fraction of draws whose fact appears exactly once, and $p(U)$ for the missing mass, the total probability of the facts that do not appear.

1. (Good-Turing.) $\mathbb{E}[\mathrm{MF}] - \mathbb{E}[p(U)] = \sum_i p_i^2 (1 - p_i)^{n-1}$, and $0 \le \sum_i p_i^2 (1 - p_i)^{n-1} \le 1/n$.
2. (A generator that must guess.) Suppose a generator reproduces the facts it has seen at their observed frequencies, with total mass $1 - \mathrm{MF}$, and spreads the mass $\mathrm{MF}$ evenly over $U$ unseen true facts and $M$ false factoids, which it cannot tell apart. Its hallucination rate is $\mathrm{MF} \cdot M / (M + U)$, which tends to $\mathrm{MF}$ when $M \gg U$.
3. (The uniform example.) With $N$ equally likely facts and $n = N$ draws, $\mathbb{E}[p(U)] = (1 - 1/N)^N \to 1/e \approx 0.368$. For $N = 100{,}000$ and $M = 100N$, the generator of item 2 hallucinates on about 36.7% of its generations.
4. (Confidence targets.) Under a target $t \in [0, 1)$ a correct answer earns 1, an error costs $t/(1 - t)$ and an abstention earns 0. Answering with probability $q$ of being right has expected score $(q - t)/(1 - t)$, positive exactly when $q > t$. With $t = 0$ answering is never worse than abstaining. The penalty is 1, 3 and 9 for $t = 0.5$, $0.75$ and $0.9$, and a penalty of 2 corresponds to $t = 2/3$.
5. (Guessing under binary grading.) If a model is right with probability $q_j$ on question $j$ of a set where it is unsure, guessing on all of them instead of abstaining raises its expected binary score by $\sum_j q_j$ and its expected number of errors by $\sum_j (1 - q_j)$.

### Assumptions

1. Items 1 to 3 treat facts as independent draws, as Kalai and Vempala do for arbitrary facts such as their 5W example. Facts that follow a rule, such as sums, are outside the model.
2. Item 2 is a toy. It is calibrated in the sense that the mass it puts on unseen factoids equals the Good-Turing estimate of how often a new draw is unseen; the theorem of Kalai and Vempala covers every calibrated model, with error terms this toy does not need.
3. Items 4 and 5 assume the model knows its own probability $q$ of being right. A miscalibrated model applies the rule to a wrong $q$.

### Steps

1. Fact $i$ appears exactly once with probability $n p_i (1 - p_i)^{n-1}$, so $\mathbb{E}[\mathrm{MF}] = \sum_i p_i (1 - p_i)^{n-1}$. It is absent with probability $(1 - p_i)^n$, so $\mathbb{E}[p(U)] = \sum_i p_i (1 - p_i)^n$. The difference is $\sum_i p_i (1 - p_i)^{n-1} (1 - (1 - p_i)) = \sum_i p_i^2 (1 - p_i)^{n-1} \ge 0$. The function $x (1 - x)^{n-1}$ on $[0, 1]$ peaks at $x = 1/n$ with value $(1/n)(1 - 1/n)^{n-1} \le 1/n$, so the sum is at most $\sum_i p_i \cdot (1/n) = 1/n$.
2. A generation is false exactly when the generator draws from the spread mass, probability $\mathrm{MF}$, and then lands on one of the $M$ false factoids, probability $M / (M + U)$. The product is $\mathrm{MF} \cdot M / (M + U)$, and $M / (M + U) \to 1$ as $M / U \to \infty$.
3. Each fact has $p_i = 1/N$, so $\mathbb{E}[p(U)] = (1 - 1/N)^N$ and $\mathbb{E}[\mathrm{MF}] = (1 - 1/N)^{N-1}$, both about $e^{-1} \approx 0.3679$ for $N = 100{,}000$. The expected number of unseen true facts is $U = N (1 - 1/N)^N \approx 36{,}788$. With $M = 10^7$, $M / (M + U) \approx 0.9963$ and the rate is $0.3679 \times 0.9963 \approx 0.3665$. The gap of item 1 is $N \cdot N^{-2} (1 - 1/N)^{N-1} = \mathbb{E}[\mathrm{MF}]/N \approx 3.7 \times 10^{-6}$, below $1/n = 10^{-5}$.
4. The expected score is $q - (1 - q) t/(1 - t) = (q(1 - t) - t + qt)/(1 - t) = (q - t)/(1 - t)$. The denominator is positive, so the sign is that of $q - t$, and abstaining scores 0. With $t = 0$ the score is $q \ge 0$. The penalties are $0.5/0.5 = 1$, $0.75/0.25 = 3$ and $0.9/0.1 = 9$, and $t/(1 - t) = 2$ gives $t = 2/3$.
5. Under binary grading a guess on question $j$ scores 1 with probability $q_j$ and 0 otherwise, and an abstention scores 0. Expectations add over questions, which gives both sums.

### Result and check

$$ 0 \le \mathbb{E}[\mathrm{MF}] - \mathbb{E}[p(U)] \le \frac{1}{n}, \qquad \text{hallucination rate} = \mathrm{MF} \cdot \frac{M}{M + U}, \qquad \mathbb{E}[\text{score}] = \frac{q - t}{1 - t} . $$

Kalai et al. (2025) list the penalties for $t = 0.5$, $0.75$ and $0.9$ as 1, 2 and 9. The formula they state gives 3 for $t = 0.75$; 2 would be the penalty for $t = 2/3$.

The check `hallucination-arithmetic.check.py` verifies the identity and the bound of item 1 exactly on 400 random distributions, and compares the expected monofact fraction and missing mass with 300 simulated draws from a Zipf law. It recomputes the uniform example and simulates five training sets of 100,000 draws and the generator's output. It checks the expected score and its sign for six targets on a grid of $q$, the penalties, and the gain from guessing against simulated binary grading on 500 random sets of questions.
