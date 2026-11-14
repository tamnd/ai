---
tag: KDH97E
kind: method
slug: q-learning
name: Q-learning
aliases: [Watkins Q-learning, tabular Q-learning]
area: reinforcement-learning/monte-carlo-and-td-learning
level: core
summary: >-
  An off-policy temporal-difference method that learns optimal action values from sampled transitions.
purpose: control
uses: []
related_to: []
hyperparameters:
  - {name: alpha, meaning: "step size for each value update", typical: "0 < alpha <= 1 in the implementation; the convergence theorem has separate per-pair sum conditions"}
  - {name: gamma, meaning: "discount on future rewards", typical: "0 <= gamma < 1 for the discounted contraction argument"}
  - {name: epsilon, meaning: "probability of exploratory action under epsilon-greedy behavior", typical: "0 <= epsilon <= 1"}
cost:
  train: "one scan of the next state's actions and one table update per transition; one value per state-action pair"
  inference: "one scan of the current state's actions; one value per action"
history:
  note: "Watkins introduced Q-learning in his thesis; Watkins and Dayan published a convergence analysis."
prereqs: []
sources:
  - {paper: "Watkins (1989), Learning from Delayed Rewards", title: "Learning from Delayed Rewards", locator: "Title page"}
  - {paper: "Watkins and Dayan (1992), Q-learning", title: "Q-learning", locator: "p. 282"}
  - {book: "Sutton and Barto (2018), Reinforcement Learning: An Introduction", locator: "Sections 2.2, 2.5, 2.6, 3.1, 3.3, 3.5, 3.6, 6.4 to 6.7, 7.6, 11.3"}
  - {paper: "Mnih et al. (2015), Human-level control through deep reinforcement learning", title: "Human-level control through deep reinforcement learning", locator: "Nature 518, Abstract and Methods"}
---

Q-learning is an [[al:importance-sampling-off-policy|off-policy]] [[al:temporal-difference-learning|temporal-difference]] method that estimates the action values of an optimal policy from sampled transitions, even when another policy selects the actions. {#c-definition}
It can learn a greedy policy that is optimal in the limit while the behavior policy continues to explore. {#c-purpose}

## In plain words

After an action, compare its estimated value with the reward plus the best estimated value available next.
Move the old estimate partway toward that target.
The action that generated the data can differ from the action rule whose values are learned. {#c-plain}

## Setting

Q-learning is a control method for a finite [[al:markov-decision-process|Markov decision process]] (MDP) with sampled states, actions, rewards, and next states. {#c-setting}
The Markov property means that the distribution of the next state and reward depends only on the current state and action, not on earlier history. {#c-markov}
The goal is to estimate $q_*(s,a)$, the expected discounted return after taking action $a$ in state $s$ and then acting optimally. {#c-optimal-value}
The method uses a [[al:bellman-optimality-equation|Bellman optimality equation]] to define those values. {#c-bellman-link}
The [[al:epsilon-greedy|behavior policy]] may explore, while the target in the update evaluates a greedy policy.
The tabular update can use samples from deterministic or stochastic tasks. {#c-stochastic-setting} {#c-off-policy}

### Intuition

The target uses the largest estimated next value, even when an exploratory action produced the transition. {#c-intuition}
Repeated visits let later observations revise each table entry. {#c-intuition-revision}

## The method

Store an estimate $Q(s,a)$ for each state-action pair.
After observing $(s_t,a_t,r_{t+1},s_{t+1})$, compute

$$Y_t = r_{t+1} + \gamma \max_{a'} Q(s_{t+1},a')$$ {#eq-target}

For a terminal next state, set $Y_t=r_{t+1}$ and omit the future-value term. {#c-terminal}
Then update only the visited entry:

$$Q(s_t,a_t) \leftarrow Q(s_t,a_t)+\alpha_t\left[Y_t-Q(s_t,a_t)\right].$$ {#eq-update}

An [[al:epsilon-greedy|epsilon-greedy]] policy can select behavior actions by choosing a random action with probability $\epsilon$ and a greedy action otherwise. {#c-epsilon-greedy}

For each episode, apply the following tabular loop:

```text
Initialize the action-value table Q.
For each episode:
    Observe the initial state s.
    Until the task terminates:
        Choose action a from the behavior policy based on Q(s, ·).
        Execute a and observe reward r, next state s', and termination.
        Set target to r if terminal; otherwise set it to r + gamma max_b Q(s', b).
        Set Q(s, a) to Q(s, a) + alpha(s, a) [target - Q(s, a)].
        Set s to s' if the task continues.
Return Q and a greedy policy derived from Q.
```

Only the visited state-action entry changes at each transition. {#c-one-entry-alg}
This target takes a maximum over next actions, rather than using the sampled next action as [[al:sarsa|state, action, reward, state, action (Sarsa)]] does. {#c-target-distinction}

## Derivation

The [derivation](derivations/convergence.md) gives the Bellman equation, the conditional mean of the sample target, and the update's stochastic-approximation form. {#c-derivation-link}
It derives that the Bellman optimality operator is a $\gamma$-contraction in the sup norm when $\gamma<1$. {#c-contraction}
The [derivation](derivations/convergence.md) cites, but does not prove, the convergence theorem.
For a finite discounted MDP with bounded rewards, infinitely many visits to every state-action pair, and per-pair step sizes with divergent sums and convergent sums of squares, Q-learning converges with probability one to $q_*$. {#c-convergence}

## Worked example

Let $Q(s,a)=2$, let the observed reward be $1$, let $\gamma=0.5$, and let the next state's action values be $4$ and $3$.
With $\alpha=0.25$, the target is $1+0.5\max(4,3)=3$, and the updated entry is $2+0.25(3-2)=2.25$. {#c-example}
All other table entries retain their prior values in this update. {#c-one-entry}

## Hyperparameters

The step size $\alpha_t$ controls how far an update moves toward its target. {#c-alpha}
The NumPy update accepts $0<\alpha\leq1$, including a full replacement of the visited estimate. {#c-alpha-code}
The cited theorem separately requires $0<\alpha<1$ at each update, per-pair sums that diverge, and squared sums that converge. {#c-alpha-conditions}
The discounted contraction argument on this page requires $0\leq\gamma<1$. {#c-gamma}
The implementation also accepts $\gamma=1$ for episodic tasks with a terminal state; the contraction proof does not cover that case.
The exploration rate $\epsilon$ controls random behavior actions under epsilon-greedy selection. {#c-epsilon}
The starting $Q$ values affect early greedy choices before all actions have been sampled. {#c-initial-setting}
A visit-count step size such as $\alpha_t(s,a)=1/N_t(s,a)$ decreases on each visit and satisfies the theorem's per-pair sum conditions when visits continue. {#c-alpha-schedule}
A fixed positive step size gives more weight to later samples but does not satisfy the theorem's squared-sum condition. {#c-fixed-step}
A decaying $\epsilon$ can reduce exploratory actions as learning proceeds, but a schedule that removes visits to some pairs loses the theorem's coverage condition. {#c-epsilon-schedule}
Optimistic initial values can encourage early visits to actions whose values have not yet been revised. {#c-optimistic}
This page experiment uses its own step size, exploration rate, and discount settings, and does not reproduce Sutton and Barto’s cliff-walking figure. {#c-local-settings}
Its [configuration](exp/cliff-walking/experiment.yaml) uses a discount below one and records the episode window, step size, and exploration rate. {#c-cliff-settings}
This is one discounted configuration, while the book example uses an undiscounted task. {#c-cliff-discount}

## Cost

For $|S|$ states and $|A|$ actions, a full table stores $|S||A|$ values. {#c-storage}
If the next state's action values are scanned directly, each transition costs $O(|A|)$ time for the maximum and constant additional work for the update. {#c-cost}
Greedy action selection scans the current state's $|A|$ values. {#c-inference-cost}

## Evidence

Watkins and Dayan prove convergence for finite tabular problems under repeated sampling and step-size conditions. {#c-convergence-evidence}
Sutton and Barto’s Example 6.6 reports that Q-learning follows the optimal path while Sarsa follows a safer route during continued exploration. {#c-cliff-result}
Across {fact:q-learning/cliff.seeds} seeds, the online return is averaged over the last {fact:q-learning/cliff.window} of {fact:q-learning/cliff.episodes} episodes in each seed. {#c-cliff-window}
The [run record](exp/cliff-walking/results.json) reports Q-learning mean return {fact:q-learning/cliff.q_mean} with across-seed standard deviation {fact:q-learning/cliff.q_sd}. {#c-cliff-q-return}
Sarsa's mean is {fact:q-learning/cliff.sarsa_mean} with across-seed standard deviation {fact:q-learning/cliff.sarsa_sd}. {#c-cliff-sarsa-return}
The estimated Sarsa-minus-Q-learning return difference has a bootstrap 95 percent confidence interval from {fact:q-learning/cliff.ci_low} to {fact:q-learning/cliff.ci_high}. {#c-cliff-ci}
The run reports edge-path fractions of {fact:q-learning/cliff.q_edge} for Q-learning and {fact:q-learning/cliff.sarsa_edge} for Sarsa. {#c-cliff-path-check}
Greedy policies reached the goal in {fact:q-learning/cliff.q_goal} Q-learning seeds and {fact:q-learning/cliff.sarsa_goal} Sarsa seeds. {#c-cliff-goal}
The Sarsa edge-path fraction includes {fact:q-learning/cliff.sarsa_no_goal} greedy policies that did not reach the goal. {#c-cliff-path-caveat}

## Limitations and failure modes

A table needs a separate value for every state-action pair, so it is impractical when either set is large or continuous. {#c-table-limit}
With function approximation, [[al:temporal-difference-learning|bootstrapping]] and off-policy learning together can cause divergence; this combination is called the [[al:deadly-triad|deadly triad]]. {#c-deadly-triad}
Sutton and Barto describe [[al:double-q-learning|maximization bias]] in Q-learning targets in Section 6.7. {#c-max-bias-source}
For this page’s constructed example, two independent estimates of zero-valued actions are each $+1$ or $-1$ with equal probability, so their expected maximum is $1/2$. {#c-max-bias}
The convergence theorem requires every state-action pair to be visited infinitely often; without coverage, the theorem does not apply. {#c-coverage}
Sutton and Barto describe how a constant step size can track changing rewards. {#c-constant-alpha-track}
A positive constant step size used indefinitely fails the theorem’s squared-sum condition. {#c-constant-alpha}
An exploration schedule that stops visiting an action can leave its value uncorrected. {#c-exploration-failure}

## Variants

[[al:watkins-q-lambda|Watkins Q(lambda)]] extends Q-learning with eligibility traces and clears the trace after a non-greedy behavior action. {#c-q-lambda}
[[al:double-q-learning|Double Q-learning]] uses two estimates to reduce the maximization bias described above. {#c-double}
[[al:expected-sarsa|Expected Sarsa]] replaces the sampled next action with an expectation under a specified target policy. {#c-expected-sarsa}
[[al:n-step-td|Multi-step temporal-difference methods]] use rewards from several transitions before bootstrapping. {#c-n-step}
[[al:dqn|Deep Q-networks]] (DQN) use neural function approximation with Q-learning targets, experience replay, and a target network. {#c-dqn}
Their max target can inherit overestimation from noisy action estimates; the two-estimate idea in Double Q-learning addresses that mechanism in the tabular setting. {#c-dqn-bias}

### Related methods

[[al:sarsa|Sarsa]] uses the sampled next action in its target. {#c-related-sarsa}
[[al:expected-sarsa|Expected Sarsa]] averages over the next action under a chosen policy. {#c-related-expected}

## When to use it

Tabular Q-learning fits small finite tasks where the agent can sample transitions and ensure adequate coverage. {#c-when}
For on-policy control, compare it with [[al:sarsa|Sarsa]], which evaluates actions selected by the behavior policy. {#c-alternatives}

## Pitfalls

> [!pitfall]
> **Do not bootstrap through a terminal state.**
> Set the terminal target to the reward alone.
> `test_terminal_ignores_nonzero_next_row` checks this case.

> [!pitfall]
> **Keep time-limit truncation distinct from task termination.**
> Pass `terminal=False` for a time-limit truncation so the update keeps its bootstrap term.
> `test_time_limit_truncation_bootstraps` checks this behavior.

> [!pitfall]
> **Take the maximum, not the sampled next action.**
> Using the behavior action in the target changes the update to an on-policy Sarsa target.
> `test_max_not_sampled_action` checks this case.

> [!pitfall]
> **Update only the visited entry.**
> `test_worked_example_and_one_entry` checks the updated value and every untouched entry.

> [!pitfall]
> **Provide exploration coverage.**
> The convergence theorem needs repeated coverage, but this implementation does not test a general exploration policy.

> [!pitfall]
> **Choose initial values deliberately.**
> If unsampled actions keep initial values above sampled actions, greedy selection favors the unsampled actions. {#c-initial-values}
> `test_optimistic_initial_value_guides_greedy_choice` checks this case.

> [!pitfall]
> **Separate online return from the learned target.**
> On the cliff-walking task, Sarsa's sampled next action can account for exploratory risk in its updates, while Q-learning's greedy target does not.
> Sutton and Barto report that this distinction affects online behavior in their cliff-walking example. {#c-cliff-difference}

## Implementations

The [NumPy implementation](impl/numpy/q_learning.py) provides the tabular update and epsilon-greedy action selection.
Its tests cover the worked update, terminal handling, greedy target, policy frequencies, parameter errors, and deterministic and stochastic value-iteration comparisons.

## Experiments

The [cliff-walking experiment](exp/cliff-walking/experiment.yaml) compares tabular Q-learning and Sarsa.
The [local experiment](exp/cliff-walking/results.json) compares online returns and greedy paths on a tabular cliff grid across its recorded seeds. {#c-cliff-scope}
The [epsilon sweep](exp/epsilon-sweep/sweep.json) checks the same cliff task at several exploration rates. {#c-sweep-scope}
Mean online return decreases as $\epsilon$ rises across its recorded arms. {#c-sweep-outcome}
Its [results](exp/cliff-walking/results.json) record online return, its spread, and greedy routes in a discounted setting.

## History

Watkins introduced Q-learning in his University of Cambridge doctoral thesis in 1989. {#c-thesis}
Watkins and Dayan published their convergence analysis in *Machine Learning*, volume 8, pages 279 to 292, in 1992. {#c-watkins-dayan}
Mnih et al. reported a deep Q-network system for Atari game control in *Nature* in 2015. {#c-dqn-history}

## References
