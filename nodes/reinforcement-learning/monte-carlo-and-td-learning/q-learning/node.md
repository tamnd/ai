---
tag: KDH97E
kind: method
slug: q-learning
name: Q-learning
aliases: [Watkins Q-learning, tabular Q-learning, one-step Q-learning]
area: reinforcement-learning/monte-carlo-and-td-learning
level: core
summary: >-
  An off-policy temporal-difference control method that moves the value of each visited state-action pair toward the reward plus the discounted maximum value of the next state.
purpose: control
uses: [markov-decision-process@al, bellman-optimality-equation@al, temporal-difference-learning@al, epsilon-greedy@al, stochastic-approximation@m2]
related_to: [sarsa@al, expected-sarsa@al, double-q-learning@al, dqn@al, value-iteration@al, watkins-q-lambda@al, deadly-triad@al]
hyperparameters:
  - {name: alpha, meaning: "the step size: the fraction of the TD error added to the entry", typical: "in (0, 1] in the algorithm of Sutton and Barto; per-pair step sizes such as 1/n for the convergence theorem"}
  - {name: gamma, meaning: "the discount rate of future rewards", typical: "0 <= gamma < 1 for the convergence theorem; 1 for episodic tasks such as cliff walking"}
  - {name: epsilon, meaning: "the probability of a random action in the epsilon-greedy behaviour policy", typical: "a small epsilon > 0, for example 0.1"}
cost:
  train: "one max over the actions of the next state and one table entry written per transition; a table of |S| x |A| values"
  inference: "one argmax over the actions of the current state"
history:
  year: 1989
  people: [Chris Watkins, Peter Dayan]
  note: "Watkins (1989), PhD thesis, University of Cambridge; the convergence proof is Watkins and Dayan (1992), Machine Learning 8."
prereqs: [markov-decision-process@al, bellman-optimality-equation@al, temporal-difference-learning@al, epsilon-greedy@al, robbins-monro-algorithm@m2]
same_as:
  al: q-learning
sources:
  - {book: "Sutton and Barto (2018), Reinforcement Learning: An Introduction, second edition", locator: "pp. 131 to 136, p. 140", note: "Sections 6.5 to 6.7: Q-learning, cliff walking, Expected Sarsa and Double Q-learning; the bibliographical remarks"}
  - {paper: "Watkins and Dayan (1992)", title: "Q-learning", locator: "Machine Learning 8, 279 to 292; Sections 2 and 4", note: "the convergence theorem, its conditions and its extensions"}
  - {paper: "Watkins (1989)", title: "Learning from Delayed Rewards", locator: "PhD thesis, University of Cambridge", note: "the thesis that introduced Q-learning"}
  - {paper: "Mnih et al. (2015)", title: "Human-level control through deep reinforcement learning", locator: "Nature 518, 529 to 533", note: "the deep Q-network"}
  - {paper: "van Hasselt, Guez and Silver (2016)", title: "Deep Reinforcement Learning with Double Q-learning", locator: "AAAI 2016; arXiv 1509.06461v3, the abstract", note: "overestimation in DQN and Double DQN"}
---

Q-learning is a reinforcement learning method that learns the value of each action in each state from experience, with no model of the environment.
After each step it moves one table entry toward the reward plus the discounted maximum value of the next state.
The table then approaches the optimal action values whatever exploratory policy collected the data.
It is the base of many deep reinforcement learning agents, such as the deep Q-network.

## In plain words

Keep a table with a score for every action in every state.
Take an action, see the reward and where you land.
Your new guess for that action is the reward plus the maximum score you now see in the next state.
Move the old score part of the way toward that guess.
Because the guess uses the maximum over the next actions, not the one you will try, you can explore freely and still learn the scores of optimal behaviour.

## Setting

Q-learning solves the control problem of a Markov decision process: find a policy that maximizes the expected discounted return.
The agent sees the state $S_t$, picks an action $A_t$, and gets a reward $R_{t+1}$ and a next state $S_{t+1}$.
It does not know the dynamics $p(s', r \mid s, a)$; it only samples them.
The tabular form needs finite sets of states $\mathcal{S}$ and actions $\mathcal{A}$ and one table entry per pair.
The target is the optimal action value function $q_*$, the solution of the Bellman optimality equation

$$ q_*(s, a) = \mathbb{E}\big[ R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') \;\big|\; S_t = s, A_t = a \big]. $$ {#eq-bellman-optimality}

A greedy policy with respect to $q_*$ is optimal.

## The method

The update, as given by Sutton and Barto {eq:sutton-barto-2018-reinforcement-learning#6.8}, is

$$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \big[ R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \big]. $$ {#eq-q-update}

Q-learning (Watkins, 1989) is an off-policy temporal-difference control algorithm defined by this update. {#c-update}
The bracket is the TD error $\delta_t$.
When $S_{t+1}$ is terminal, the max term is zero.
Only the entry of the pair $(S_t, A_t)$ changes.

The procedure, for each episode:

1. Start in a state $S$.
2. Choose $A$ from $S$ with a policy derived from $Q$, for example $\varepsilon$-greedy: a uniformly random action with probability $\varepsilon$, a greedy action otherwise.
3. Take $A$, and observe $R$ and $S'$.
4. Apply eq-q-update to $Q(S, A)$.
5. Set $S \leftarrow S'$, and go back to step 2 until $S$ is terminal.

In the algorithm of Sutton and Barto, the parameters are a step size $\alpha \in (0, 1]$ and a small $\varepsilon > 0$, and $Q$ starts arbitrary except that $Q(\text{terminal}, \cdot) = 0$. {#c-algorithm-box}

Sutton and Barto write that $Q$ directly approximates $q_*$, independent of the policy being followed. {#c-approximates-qstar}
That is what makes the method off-policy: the behaviour policy explores, and the target uses the greedy action.
The policy still decides which pairs are visited and updated, and correct convergence needs all pairs to keep being updated. {#c-policy-role}

Sarsa, the on-policy method of the same family, uses the action that the behaviour policy actually takes next:

$$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \big[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \big]. $$ {#eq-sarsa}

The two differ only in the target, and the cliff walking experiment below shows what that changes.

## Derivation

The Bellman optimality operator $\mathcal{T}_*$ maps a table $Q$ to the right side of eq-bellman-optimality with $Q$ in place of $q_*$.
For a finite process with $0 \le \gamma < 1$, $\mathcal{T}_*$ is a contraction of modulus $\gamma$ in the max norm, so its only fixed point is $q_*$. {#c-contraction}
Given the history, the mean of the target $R_{t+1} + \gamma \max_a Q_t(S_{t+1}, a)$ is $(\mathcal{T}_* Q_t)(S_t, A_t)$. {#c-target-mean}
So each update is a noisy step toward the fixed point of a contraction, a stochastic approximation scheme of the Robbins and Monro kind:

$$ Q_{t+1}(S_t, A_t) = Q_t(S_t, A_t) + \alpha_t \big[ (\mathcal{T}_* Q_t)(S_t, A_t) - Q_t(S_t, A_t) + w_t \big], \qquad \mathbb{E}[w_t \mid \text{history}] = 0. $$ {#eq-sa-form}

The full derivation, with each step and its reason, is in [derivations/stochastic-approximation.md](derivations/stochastic-approximation.md).

The derivation shows the shape, not the convergence.
Watkins and Dayan (1992) prove convergence.
In their notation, $x$ is the state and $Q_n$ is the table after the $n$-th episode.
The index $n^i(x, a)$ marks the $i$-th time that action $a$ is tried in state $x$.
Their theorem: given bounded rewards $\lvert r_n \rvert \le R$, learning rates $0 \le \alpha_n < 1$, and, for all $x$ and $a$, $\sum_i \alpha_{n^i(x,a)} = \infty$ and $\sum_i \alpha_{n^i(x,a)}^2 < \infty$, $Q_n(x, a) \to q_*(x, a)$ as $n \to \infty$ for all $x$ and $a$, with probability 1. {#c-wd-theorem}
The setting is a discounted process with $\gamma < 1$.
Their description assumes a look-up table, and they cite Watkins (1989) for the fact that Q-learning may not converge correctly with other representations.
The most important implicit condition is an infinite number of episodes for each starting state and action. {#c-wd-conditions}
The step sizes $\alpha_n = 1/n$ meet both sums, and a constant step size does not meet the second one: the estimates then keep varying in response to the latest rewards. {#c-step-sizes}
Watkins and Dayan sketch two extensions where the result holds: the undiscounted case with absorbing goal states, and updates of many entries in each iteration. {#c-wd-extensions}

## Worked example

Take two states, $s_0$ and $s_1$, two actions, $a_0$ and $a_1$, $\alpha = 0.5$ and $\gamma = 0.9$.
The table starts as below, one row per state.

| | $a_0$ | $a_1$ |
|---|---|---|
| $s_0$ | 1 | 2 |
| $s_1$ | 4 | 3 |

Three transitions follow, each from the table that the previous one left.

| Step | Transition $(s, a, r, s')$ | Target | Old value | New value |
|---|---|---|---|---|
| 1 | $(s_0, a_0, 1, s_1)$ | $1 + 0.9 \cdot \max(4, 3) = 4.6$ | 1 | $1 + 0.5 \cdot (4.6 - 1) = 2.8$ |
| 2 | $(s_1, a_1, 2, \text{terminal})$ | $2$ | 3 | $3 + 0.5 \cdot (2 - 3) = 2.5$ |
| 3 | $(s_0, a_1, 0, s_1)$ | $0 + 0.9 \cdot \max(4, 2.5) = 3.6$ | 2 | $2 + 0.5 \cdot (3.6 - 2) = 2.8$ |

The table ends as $s_0 = (2.8, 2.8)$ and $s_1 = (4, 2.5)$.
In step 2 the next state is terminal, so the target is the reward alone.

Sarsa on the transition of step 1, with the next action $a_1$ chosen by exploration, uses $Q(s_1, a_1) = 3$ instead of the max.
Its target is $1 + 0.9 \cdot 3 = 3.7$, and the new value is $1 + 0.5 \cdot 2.7 = 2.35$.
The test `test_worked_example` of the NumPy reference checks these numbers.

## Hyperparameters

The **step size** $\alpha$ sets how far one transition moves an entry.
The algorithm of Sutton and Barto takes $\alpha \in (0, 1]$, while the convergence theorem needs a decreasing step size for each pair, such as $1/n$.
A constant $\alpha$ tracks a changing problem, but the estimates do not settle.

The **discount rate** $\gamma$ weighs future rewards.
The contraction argument needs $\gamma < 1$; episodic tasks with sure termination may use $\gamma = 1$, as cliff walking does.

The **exploration rate** $\varepsilon$ of an $\varepsilon$-greedy behaviour policy sets how often a random action is tried.
Q-learning learns the greedy values whatever $\varepsilon$ is, but the return earned while learning drops as $\varepsilon$ grows.
The cliff walking example of Sutton and Barto uses $\varepsilon = 0.1$.

The **initial values** also matter.
Sutton and Barto describe optimistic initial values as a simple trick to encourage exploration, quite effective on stationary problems but far from a generally useful approach. {#c-optimistic}

No framework ships tabular Q-learning with defaults; libraries of deep Q-learning set their own.

## Cost

Let $\lvert \mathcal{S} \rvert$ be the number of states and $\lvert \mathcal{A} \rvert$ the number of actions.
The table holds $\lvert \mathcal{S} \rvert \cdot \lvert \mathcal{A} \rvert$ values.
Each transition costs one max over the $\lvert \mathcal{A} \rvert$ values of the next state and one write.
An $\varepsilon$-greedy choice costs one argmax over $\lvert \mathcal{A} \rvert$ values.
The method keeps no model and no replay of old transitions.
The number of transitions needed to converge is the real cost: every pair must be tried many times, and the table grows with the product of the state and action sets.

## Evidence

Watkins and Dayan (1992) report that Q-learning converges to the optimum action-values with probability 1 so long as all actions are repeatedly sampled in all states and the action-values are represented discretely. {#c-wd-abstract}
On cliff walking with $\varepsilon = 0.1$, Q-learning learns the optimal path along the edge and occasionally falls off, while Sarsa learns a safer path and does better online (Sutton and Barto, 2018). {#c-cliff}
Mnih et al. (2015) report that a deep Q-network, seeing only pixels and the score, reached a level comparable to a professional human games tester across 49 Atari games. {#c-dqn-atari}

## Limitations and failure modes

**Overestimation.**
Q-learning uses a maximum over estimated values as an estimate of the maximum value, which can lead to a significant positive bias (Sutton and Barto, 2018). {#c-max-bias}
With many actions whose true values are equal, noise alone pushes the maximum of the estimates too high.
van Hasselt, Guez and Silver (2016) report substantial overestimations by DQN in some Atari games.
They report that their Double Q-learning form of DQN reduces them and does much better on several games. {#c-double-dqn}

**Online return.**
Q-learning learns the values of the greedy policy, not of the exploring policy that it follows.
On cliff walking, this makes its return while learning worse than that of Sarsa, as the Evidence section shows.
If $\varepsilon$ were gradually reduced, both methods would converge to the optimal policy. {#c-cliff-decay}

**Function approximation.**
The convergence theorem is for a table.
Sutton and Barto name the combination of function approximation, bootstrapping and off-policy training the deadly triad, where the danger of instability and divergence arises. {#c-deadly-triad}
Q-learning with a function approximator has all three.
Mnih et al. (2015) write that reinforcement learning is known to be unstable or even to diverge when a nonlinear function approximator such as a neural network represents the action-value function. {#c-dqn-unstable}

**Exploration.**
The guarantee needs every pair to be tried infinitely often; an $\varepsilon$-greedy policy in a large state space may take a very long time to reach rare pairs.

## Variants

- **Sarsa** uses the next action taken, eq-sarsa, and learns the values of the exploring policy.
- **Expected Sarsa** uses the expected value of the next state under a target policy. When that policy is greedy and the behaviour explores more, it is exactly Q-learning (Sutton and Barto, 2018). {#c-expected-sarsa}
- **Double Q-learning** keeps two tables, and uses one to pick the maximizing action and the other to value it. Each step updates one table, so the memory doubles but the computation per step does not (Sutton and Barto, 2018). {#c-double-learning}
- **Watkins's Q($\lambda$)** adds eligibility traces and cuts them to zero after the first non-greedy action (Sutton and Barto, 2018). {#c-q-lambda} The theorem of Watkins and Dayan does not extend trivially to updates that use rewards from more than one step. {#c-wd-no-lambda}
- **Deep Q-network (DQN)** represents $Q$ with a convolutional network. Mnih et al. (2015) add experience replay, which randomizes over the data, and targets that are only periodically updated. {#c-dqn-ideas}
- **Double DQN** brings the idea of Double Q-learning to DQN (van Hasselt, Guez and Silver, 2016).

## When to use it

Use tabular Q-learning when the states and actions are few enough for a table and there is no model of the environment.
It learns the optimal values from any data that keeps visiting every pair, including data from an exploring or older policy.
Use Sarsa or Expected Sarsa when the return earned while learning matters, as on cliff walking, where Sarsa learns a safer path.
Use dynamic programming, such as value iteration, when the model $p(s', r \mid s, a)$ is known.
Use Double Q-learning when the maximization bias matters, for example with many actions and noisy rewards.
For large state spaces, Q-learning needs function approximation, and then the convergence theorem no longer applies.

## Pitfalls

> [!pitfall]
> **Bootstrapping from a terminal state.** The value of a terminal state is zero, so the target is the reward alone. Code that reads a row of the table for the terminal state adds whatever that row holds. The test `test_terminal_has_no_bootstrap` puts large values in that row and checks that they are ignored.

> [!pitfall]
> **A time limit treated as termination.** When an episode is cut by a step limit, the next state is not terminal, and the target must still bootstrap from it. The test `test_truncation_bootstraps` checks that the two cases give different targets.

> [!pitfall]
> **The next action instead of the max.** A target that uses the value of the action actually taken next is Sarsa, not Q-learning, and it learns the values of the exploring policy. The test `test_max_not_next_action` checks both targets on a table where they differ.

> [!pitfall]
> **Ties always broken toward the first action.** With a table that starts at zero, `argmax` returns the first action, and the greedy part of the policy never tries the others. The test `test_ties_broken_at_random` checks that tied actions are picked about equally often.

> [!pitfall]
> **A constant step size in a stochastic task.** With a constant $\alpha$, the entries keep moving with the latest rewards and do not settle on $q_*$. The test `test_converges_to_oracle` uses the step sizes $1/n(s, a)$ and checks the result against an independent policy iteration oracle.

## History

Sutton and Barto write that Q-learning was introduced by Watkins (1989), and that Watkins and Dayan (1992) made his outline of a convergence proof rigorous. {#c-history-watkins}
They add that Jaakkola, Jordan and Singh (1994) and Tsitsiklis (1994) proved more general convergence results.
Watkins's thesis, *Learning from Delayed Rewards*, was written at the University of Cambridge.
They also credit the term action value to Watkins (1989), who may have been the first to use $\varepsilon$-greedy methods. {#c-history-action-value}
Mnih et al. (2015) combined Q-learning with a deep convolutional network, experience replay and a periodically updated target, and trained it on Atari 2600 games from pixels.
van Hasselt, Guez and Silver (2016) adapted Double Q-learning to that agent.
