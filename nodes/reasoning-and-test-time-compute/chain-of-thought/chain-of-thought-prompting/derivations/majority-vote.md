---
id: majority-vote
derives: [c-binary-vote, c-binary-step, c-binary-limit, c-even-ties, c-first-seen-ties, c-plurality-limit, c-dataset-mixture, c-worked-numbers, eq-binary-vote, eq-binary-step]
kind: proof
assumes: ["the m sampled answers to one question are independent and identically distributed given the question", "the answers lie in a finite answer set", "the probabilities of the answers are fixed numbers of the model, the prompt and the decoding settings, not quantities the paper reports"]
uses: [binomial-distribution@m2, hoeffdings-inequality@m2, independent-and-identically-distributed-random-variables@m2, weak-law-of-large-numbers@m2]
status: full
check: majority-vote.check.py
lean: null
sources:
  - {work: wang-2022-self-consistency-chain-of-thought, locator: "p. 3"}
  - {work: wang-2022-self-consistency-chain-of-thought, locator: "p. 5"}
  - {work: wang-2022-self-consistency-chain-of-thought, locator: "p. 2"}
---

### Statement

Fix one question. Sample $m$ chains of thought and read off their final answers $a_1, \dots, a_m$, which lie in a finite answer set $A$. Let $q(a)$ be the probability that one sample gives the answer $a$, let $c$ be the correct answer, and write $p = q(c)$. The majority vote of Wang et al. returns an answer with the most votes, $\arg\max_a \sum_i \mathbb 1(a_i = a)$.

1. **Two answers, odd $m$.** If only $c$ and one wrong answer occur, the vote is correct with probability
   $$ P_m(p) = \sum_{j = (m+1)/2}^{m} \binom{m}{j} p^j (1 - p)^{m - j} . $$
2. **The step.** For odd $m$ and $h = (m + 1)/2$,
   $$ P_{m+2}(p) - P_m(p) = \binom{m}{h} \, p^{h} (1 - p)^{h} \, (2p - 1) . $$
   So with $p > 1/2$ the accuracy rises strictly with every two more samples from $P_1 = p$, and with $p < 1/2$ it falls strictly.
3. **The limit.** With $p > 1/2$, $1 - P_m(p) \le \exp(-2m(p - 1/2)^2)$, so $P_m \to 1$; with $p < 1/2$, $P_m(p) \le \exp(-2m(1/2 - p)^2)$, so $P_m \to 0$.
4. **Even $m$.** If a tie is broken by a fair coin, $P_{2k}(p) = P_{2k-1}(p)$.
5. **First-seen ties.** Breaking a tie in favour of the tied answer that was sampled first has the same distribution as breaking it uniformly at random among the tied answers.
6. **Many answers.** Let $q^\ast = \max_{a \ne c} q(a)$. If $p > q^\ast$, the vote is wrong with probability at most $(|A| - 1)\exp(-m(p - q^\ast)^2/2)$, whatever the tie rule, so it tends to $c$; if $q^\ast > p$, it returns $c$ with probability at most $\exp(-m(q^\ast - p)^2/2)$. The correct answer only has to be the most likely answer, not to have probability above $1/2$.
7. **A benchmark.** The vote accuracy on a benchmark is the mean of the per-question vote accuracies. With one question where $p = 0.9$ against one wrong answer, and one where $q = (0.3, 0.5, 0.2)$ with the correct answer first, a single sample is right with probability $0.6$ on average, and the vote accuracy tends to $(1 + 0)/2 = 0.5$.
8. **Numbers.** $P_3(0.6) = 0.648$, $P_5(0.6) = 0.68256$, $P_{21}(0.6) \approx 0.8256$, and $P_3(0.4) = 0.352$.

### Assumptions

1. **Independent samples.** Given the question and the prompt, the $m$ answers are independent and identically distributed. This part is from the paper: Wang et al. sample their 40 outputs per question independently from the decoder (p. 5), with one prompt and one temperature, so each output has the same distribution. Used in every item.
2. **A fixed answer set.** The answers are parsed into a finite set $A$, as Wang et al. require for self-consistency (p. 3). Used in every item.
3. **This page's model, not the paper's.** The per-question probabilities $q(a)$, the reduction to two answers in items 1 to 4, and the condition that the correct answer is the most likely one are the model of this page. Wang et al. state the informal version as a hypothesis, that correct reasoning processes tend to agree more in their final answer than incorrect ones (p. 2), and do not give these formulas.
4. **Independence is per question.** Nothing is assumed about how questions relate to each other: item 7 only uses linearity of the mean.
5. Items 1 to 3 take $m$ odd so there are no ties; item 4 takes a fair coin for ties.

### Steps

1. Let $X_m$ be the number of correct answers among $m$. By assumption 1 it is binomial with parameters $m$ and $p$. With two answers and odd $m$, the vote is correct exactly when $X_m \ge (m + 1)/2$, which gives item 1.
2. Write $X_{m+2} = X_m + Y$, where $Y \in \{0, 1, 2\}$ counts the correct answers among two more samples and is independent of $X_m$. The vote changes only when $X_m = h - 1$ and $Y = 2$, which turns a wrong vote right, or when $X_m = h$ and $Y = 0$, which turns a right vote wrong. Hence $P_{m+2} - P_m = \Pr(X_m = h - 1)\, p^2 - \Pr(X_m = h)\,(1 - p)^2$. Since $(h - 1) + h = m$, $\binom{m}{h-1} = \binom{m}{h}$, and $m - (h - 1) = h$, so $\Pr(X_m = h - 1) = \binom{m}{h} p^{h-1}(1-p)^{h}$ and $\Pr(X_m = h) = \binom{m}{h} p^{h}(1-p)^{h-1}$. Substituting, $P_{m+2} - P_m = \binom{m}{h} p^h (1-p)^h \,[p - (1 - p)]$, which is item 2. For $0 < p < 1$ every factor except $2p - 1$ is positive.
3. The vote is wrong when $X_m/m \le 1/2$, that is when the mean of $m$ independent indicators in $[0, 1]$ falls $p - 1/2$ below its expectation $p$. Hoeffding's inequality bounds this by $\exp(-2m(p - 1/2)^2)$. The case $p < 1/2$ is the same bound for the event $X_m/m \ge 1/2$.
4. Let $Z = X_{2k-1}$. With $2k$ samples and a fair coin, the vote is correct if $Z \ge k + 1$; if $Z = k$, it is correct when the last sample is correct and wins the coin otherwise, $p + (1 - p)/2$; if $Z = k - 1$, it wins the coin when the last sample is correct, $p/2$. With $2k - 1$ samples the vote is correct iff $Z \ge k$. The difference is $\tfrac12[\,p \Pr(Z = k - 1) - (1 - p)\Pr(Z = k)\,]$, and as in step 2 the two products are both $\binom{2k-1}{k} p^k (1-p)^k$, so it is 0.
5. Given the counts of every answer, every order of the $m$ samples with those counts has the same probability $\prod_a q(a)^{n_a}$, because the samples are independent and identically distributed. So the order is a uniformly random arrangement of the multiset, and swapping the labels of two tied answers is a bijection between the orders where one is first and those where the other is first. Each tied answer is therefore first with the same probability.
6. For a wrong answer $a$, let $D_i = \mathbb 1(a_i = a) - \mathbb 1(a_i = c)$, which lies in $[-1, 1]$ with mean $q(a) - p$. The vote can return $a$ only if $\sum_i D_i \ge 0$, and Hoeffding's inequality for variables of range 2 bounds this by $\exp(-2m(p - q(a))^2/4)$ when $p > q(a)$. A union bound over the $|A| - 1$ wrong answers gives the first half of item 6. If $q(a^\ast) = q^\ast > p$, the vote can return $c$ only if $\sum_i [\mathbb 1(a_i = c) - \mathbb 1(a_i = a^\ast)] \ge 0$, bounded in the same way. No tie rule enters, because the events bound every outcome where $a$ ties or beats $c$.
7. The benchmark accuracy is $\frac1n\sum_j \Pr(\text{vote}_j = c_j)$, and the single-sample accuracy is $\frac1n \sum_j q_j(c_j)$. In the example, $\frac12(0.9 + 0.3) = 0.6$. By item 3 the first question's vote accuracy tends to 1, and by item 6 the second's tends to 0, since the wrong answer of probability $0.5$ beats the correct one of $0.3$. The mean tends to $0.5 < 0.6$, so voting lowers the benchmark accuracy here.
8. Evaluate item 1: $P_3(0.6) = 0.6^3 + 3 \cdot 0.6^2 \cdot 0.4 = 0.216 + 0.432 = 0.648$; $P_5(0.6) = 0.07776 + 0.2592 + 0.3456 = 0.68256$; $P_3(0.4) = 0.064 + 0.288 = 0.352$; $P_{21}(0.6)$ is the exact fraction $393687311953653/476837158203125 \approx 0.825622$.

### Result and check

$$ P_m(p) = \sum_{j > m/2} \binom{m}{j} p^j (1 - p)^{m-j}, \qquad P_{m+2} - P_m = \binom{m}{\tfrac{m+1}{2}} \big(p(1 - p)\big)^{\frac{m+1}{2}} (2p - 1) . $$

Majority voting over independent samples helps a question exactly when the correct answer is the single most likely answer of one sample, and the help grows with $m$; on a question where a wrong answer is more likely, it hurts by the same mechanism.

The check `majority-vote.check.py` computes item 1 in exact fractions and compares it with a sum over all $2^m$ outcomes for $m \le 11$ and eight values of $p$. It checks the step identity of item 2 exactly for odd $m$ up to 41 and twenty rational $p$, and the sign of the step. It checks the Hoeffding bounds of item 3 for $m$ up to 201, and item 4 by enumerating the outcomes with a fair coin. It checks item 5 by enumerating all orders of samples over three answers for $m \le 7$, and item 6 by exact enumeration of the counts over three answers for $m$ up to 61, against the bound and against 20000 seeded simulated votes. It evaluates the example of item 7 with 201 samples for the first question and 151 for the second, where the vote accuracies are within $10^{-40}$ of 1 and about $0.0027$, and the numbers of item 8.
