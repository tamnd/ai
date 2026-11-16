---
id: stochastic-approximation
derives: [c-contraction, c-target-mean, eq-bellman-optimality, eq-sa-form]
kind: bound
assumes: ["a finite Markov decision process", "0 <= gamma < 1", "bounded rewards", "a table with one entry per state-action pair"]
uses: [markov-decision-process@al, bellman-optimality-equation@al, bellman-contraction@al, fixed-point@al, conditional-expectation@m2, robbins-monro-algorithm@m2]
status: full
lean: null
sources:
  - {work: sutton-barto-2018-reinforcement-learning, locator: "p. 131"}
  - {work: sutton-barto-2018-reinforcement-learning, locator: "p. 33"}
  - {work: watkins-dayan-1992-q-learning, locator: "Section 2"}
---

### Statement

Let $\mathcal{S}$ and $\mathcal{A}$ be finite, let $0 \le \gamma < 1$, and let $p(s', r \mid s, a)$ be the dynamics of the process.
For a table $Q$, the Bellman optimality operator is

$$ (\mathcal{T}_* Q)(s, a) = \mathbb{E}\big[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') \;\big|\; S_t = s, A_t = a \big], $$

with $\max_{a'} Q(s', a') = 0$ when $s'$ is terminal.

1. $\lVert \mathcal{T}_* Q - \mathcal{T}_* Q' \rVert_\infty \le \gamma \lVert Q - Q' \rVert_\infty$ for any two tables $Q$ and $Q'$. So $\mathcal{T}_*$ has exactly one fixed point, $q_*$, and the iterates $\mathcal{T}_*^k Q$ converge to $q_*$ from any table.
2. Let $Y_t = R_{t+1} + \gamma \max_{a'} Q_t(S_{t+1}, a')$ be the target of the Q-learning update. Given the history up to the choice of $A_t$, its mean is $(\mathcal{T}_* Q_t)(S_t, A_t)$.
3. So the update of eq-q-update is a stochastic approximation step toward the fixed point of $\mathcal{T}_*$:
$Q_{t+1}(S_t, A_t) = Q_t(S_t, A_t) + \alpha_t \big[ (\mathcal{T}_* Q_t)(S_t, A_t) - Q_t(S_t, A_t) + w_t \big]$, where the noise $w_t$ has mean zero given the history.

### Assumptions

1. $\mathcal{S}$ and $\mathcal{A}$ are finite, so a table is a vector of finite length and the max norm is defined. Used in every step.
2. $0 \le \gamma < 1$. Used in step 4 for the contraction and in step 5 for the fixed point.
3. The rewards are bounded, so every expectation below is finite. Used in steps 1 and 6.
4. The Markov property: given $S_t = s$ and $A_t = a$, the pair $(S_{t+1}, R_{t+1})$ has the law $p(\cdot, \cdot \mid s, a)$ whatever the earlier history and the current table. Used in step 6.
5. $Q_t$ and $A_t$ are fixed by the history up to the choice of $A_t$. Used in step 6.

### Steps

1. Write $P(s' \mid s, a) = \sum_r p(s', r \mid s, a)$ and $\bar r(s, a) = \sum_{s', r} r \, p(s', r \mid s, a)$. By the definition of the expectation, $(\mathcal{T}_* Q)(s, a) = \bar r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q(s', a')$, where the sum runs over the nonterminal states.
2. For two tables and one state $s'$, $\lvert \max_{a'} Q(s', a') - \max_{a'} Q'(s', a') \rvert \le \max_{a'} \lvert Q(s', a') - Q'(s', a') \rvert \le \lVert Q - Q' \rVert_\infty$. The first inequality holds because if $Q(s', b)$ is the larger maximum, then $\max Q - \max Q' \le Q(s', b) - Q'(s', b)$; the other order is the same with $Q$ and $Q'$ swapped.
3. Subtract the two forms of step 1 for the same $(s, a)$: the reward terms cancel, and $\lvert (\mathcal{T}_* Q)(s, a) - (\mathcal{T}_* Q')(s, a) \rvert \le \gamma \sum_{s'} P(s' \mid s, a) \, \lVert Q - Q' \rVert_\infty$ by the triangle inequality and step 2.
4. The probabilities $P(s' \mid s, a)$ over the nonterminal states sum to at most 1. So the right side of step 3 is at most $\gamma \lVert Q - Q' \rVert_\infty$. Taking the max over $(s, a)$ gives part 1 of the statement: $\mathcal{T}_*$ is a contraction of modulus $\gamma < 1$ (assumption 2). {#c-contraction}
5. The tables with the max norm form a complete metric space (assumption 1: a finite-dimensional real vector space). By the Banach fixed-point theorem, a contraction of such a space has exactly one fixed point, and its iterates converge to it from any start. The fixed point is the solution of the Bellman optimality equation, eq-bellman-optimality, which is $q_*$.
6. Condition on the history $\mathcal{F}_t$ up to the choice of $A_t$. By assumption 5, $Q_t$ is fixed, so $\max_{a'} Q_t(s', a')$ is a fixed function of $s'$. By assumption 4, $\mathbb{E}[Y_t \mid \mathcal{F}_t] = \sum_{s', r} p(s', r \mid S_t, A_t) \big[ r + \gamma \max_{a'} Q_t(s', a') \big]$, which is $(\mathcal{T}_* Q_t)(S_t, A_t)$ by step 1. This is part 2. {#c-target-mean}
7. Define $w_t = Y_t - (\mathcal{T}_* Q_t)(S_t, A_t)$. By step 6, $\mathbb{E}[w_t \mid \mathcal{F}_t] = 0$. Substituting $Y_t = (\mathcal{T}_* Q_t)(S_t, A_t) + w_t$ into eq-q-update gives part 3.

### Result and check

The Q-learning update moves one entry a fraction $\alpha_t$ toward a noisy sample of $\mathcal{T}_* Q_t$, and $\mathcal{T}_*$ is a $\gamma$-contraction whose only fixed point is $q_*$.
This is the shape of the Robbins and Monro scheme: a step toward a fixed point, with noise of mean zero.
This derivation does not prove convergence.
The proof of Watkins and Dayan (1992, Section 2 and Section 3) needs, in addition, step sizes for each pair whose sum is infinite and whose sum of squares is finite, and every pair tried infinitely often.
Sutton and Barto (2018, p. 33) note that the step sizes $1/n$ meet both conditions, and a constant step size does not meet the second.

The check is numeric, in the tests of `impl/numpy`.
`test_bellman_contraction` draws 100 random models and pairs of tables, and asserts step 4.
`test_fixed_point_matches_oracle` asserts that the $q_*$ of an independent policy iteration oracle is a fixed point of $\mathcal{T}_*$, and that 400 iterates of $\mathcal{T}_*$ from zero reach it.
`test_target_mean_is_operator` compares the mean of 20,000 sampled targets with $(\mathcal{T}_* Q)(s, a)$, within four standard errors.
`test_converges_to_oracle` runs Q-learning with the step sizes $1/n(s, a)$ on a random model and reaches the $q_*$ of the oracle within 0.05.
