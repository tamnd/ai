---
id: convergence
derives: [c-derivation-link, c-contraction, c-example, c-one-entry, c-storage, c-cost, c-inference-cost, c-max-bias, c-exploration-failure, c-constant-alpha, c-gamma, c-one-entry-alg, c-alpha-schedule, c-fixed-step, c-epsilon-schedule, c-dqn-bias]
kind: proof
assumes: ["finite discounted Markov decision process", "bounded rewards", "bounded conditional second moment of the sample noise"]
uses: []
status: partial
---

# Q-learning target and stochastic approximation

Assume a finite discounted Markov decision process with $0\leq\gamma<1$ and bounded rewards.
For any action-value function $Q$, define the Bellman optimality operator by

$$ (T_*Q)(s,a)=\mathbb{E}[R+\gamma\max_{a'}Q(S',a')\mid s,a]. $$ {#eq-bellman-operator}

At the optimal action-value function, the Bellman optimality equation is

$$ q_*(s,a)=\mathbb{E}[R+\gamma\max_{a'}q_*(S',a')\mid s,a]=(T_*q_*)(s,a). $$ {#eq-bellman-optimality}

This is the fixed-point definition of $q_*$ for a discounted finite task.

## Conditional mean of the sample target

Let $\mathcal{F}_t$ contain the history before observing transition $t$, including $(s_t,a_t)$ and the current table $Q_t$.
The Markov property gives the conditional distribution of $(R_{t+1},S_{t+1})$ from $(s_t,a_t)$.
The sampled target is $Y_t=R_{t+1}+\gamma\max_{a'}Q_t(S_{t+1},a')$ for nonterminal next states.
Therefore

$$\mathbb{E}[Y_t\mid\mathcal{F}_t]=(T_*Q_t)(s_t,a_t).$$

When $Q_t=q_*$, the fixed-point equation gives $\mathbb{E}[Y_t\mid\mathcal{F}_t]=q_*(s_t,a_t)$.
Thus $Y_t$ is a conditionally unbiased sample of the Bellman optimality backup at $q_*$.
For a terminal transition, the continuation value is zero, so the target is the observed reward.

## Stochastic-approximation form

Write the update for the visited component as

$$Q_{t+1}(s_t,a_t)=Q_t(s_t,a_t)+\alpha_t[Y_t-Q_t(s_t,a_t)].$$

Define the Bellman residual $h_t=(T_*Q_t)(s_t,a_t)-Q_t(s_t,a_t)$ and the noise $M_{t+1}=Y_t-(T_*Q_t)(s_t,a_t)$.
The conditional-mean result gives $\mathbb{E}[M_{t+1}\mid\mathcal{F}_t]=0$.
For the stochastic-approximation convergence argument, assume the conditional second moment of $M_{t+1}$ is bounded uniformly over time.
Substitution yields

$$Q_{t+1}(s_t,a_t)=Q_t(s_t,a_t)+\alpha_t[h_t+M_{t+1}].$$

This is a Robbins-Monro stochastic-approximation step: the mean update follows the Bellman residual, and the sampled transition supplies zero-mean noise.
The step size scales both the residual correction and the noise.

## Contraction

For two bounded action-value functions $Q_1,Q_2$, each action maximum changes by at most their largest component difference:

$$|\max_{a'}Q_1(s',a')-\max_{a'}Q_2(s',a')|\leq\lVert Q_1-Q_2\rVert_\infty.$$

To see this, put $d=\lVert Q_1-Q_2\rVert_\infty$.
For every action, $Q_1(s',a')\leq Q_2(s',a')+d$.
Taking maxima gives $\max_{a'}Q_1(s',a')\leq\max_{a'}Q_2(s',a')+d$.
Swapping $Q_1$ and $Q_2$ gives the reverse bound, so their maxima differ by at most $d$.
For any integrable random variable $X$, $|\mathbb{E}[X]|\leq\mathbb{E}[|X|]$.
The reward terms cancel because both Bellman backups use the same conditional reward distribution.
For each $(s,a)$, subtract the backups, apply the expectation inequality, and then use the bound on the maxima:

$$|(T_*Q_1-T_*Q_2)(s,a)|=\gamma|\mathbb{E}[\max_{a'}Q_1(S',a')-\max_{a'}Q_2(S',a')\mid s,a]|\leq\gamma\mathbb{E}[|\max_{a'}Q_1(S',a')-\max_{a'}Q_2(S',a')|\mid s,a]\leq\gamma d.$$

Taking the supremum over $(s,a)$ gives:

$$\lVert T_*Q_1-T_*Q_2\rVert_\infty\leq\gamma\lVert Q_1-Q_2\rVert_\infty.$$

Since $\gamma<1$, $T_*$ is a contraction in the sup norm.
The bounded action-value functions on a finite state-action set form a complete space under the sup norm.
Banach's fixed-point theorem therefore gives $T_*$ a unique fixed point.
The Bellman optimality equation identifies that fixed point as $q_*$.
The contraction explains why the expected backup has a unique target, while stochastic approximation addresses noisy sampled backups.

## Convergence conditions

Watkins and Dayan's tabular convergence theorem assumes a finite discrete task, bounded rewards, repeated visits to every state-action pair, and per-pair learning rates below one whose sum diverges while their squared sum converges.
That theorem supplies the convergence result with probability one; its proof is in the paper's appendix.
Convergence with probability one is cited from Watkins and Dayan, not derived here.
The conditional-mean and contraction results above do not alone prove convergence of noisy asynchronous updates.
The result concerns tabular estimates and does not establish convergence for arbitrary function approximation.

## Worked update and locality

For the page's example, $\max(4,3)=4$.
The target is $Y=1+0.5(4)=3$.
The temporal-difference error is $Y-Q(s,a)=3-2=1$.
The increment is $0.25(1)=0.25$, so the updated value is $2+0.25=2.25$.
The update assignment has $(s,a)$ as its only left-hand-side table entry.
Thus every other $(u,b)$ keeps its old value.
At a terminal next state the continuation return is zero by definition, so the target reduces to $Y=r$.

## Storage and scan cost

There is one table entry for each member of $S\times A$.
The product set has $|S||A|$ members, so the table stores $|S||A|$ values.
A direct maximum reads each of the $|A|$ next-state entries once, giving $O(|A|)$ time; arithmetic and the single write add $O(1)$.
A greedy inference action uses the same scan of the current-state row, giving $O(|A|)$ time and $O(1)$ extra space.

## Maximization bias example

Let the true values of two actions both be zero, and let each estimate independently equal $+1$ or $-1$ with probability $1/2$.
Each estimate has mean zero.
The four equally likely estimate pairs are $(+1,+1)$, $(+1,-1)$, $(-1,+1)$, and $(-1,-1)$.
Their maxima are $+1,+1,+1,-1$, respectively.
The expected maximum is $(1+1+1-1)/4=1/2$, above the true maximum of zero.

## Coverage and step sizes

Only a visited entry changes in each update, so an entry never visited again keeps its last estimate.
For a fixed positive step size $\alpha$ used infinitely often, $\sum_t\alpha^2$ diverges because it adds the same positive term on every visit.
Such a schedule fails the squared-sum condition of the cited theorem.

## Visit-count schedule

On the $n$th visit to a pair, set $\alpha_n=1/n$.
The harmonic sum $\sum_n 1/n$ diverges, while $\sum_n 1/n^2$ converges.
Thus this schedule meets the cited per-pair sum conditions if every pair continues to be visited.
A constant positive step size instead has a divergent squared sum, as shown above.
For epsilon-greedy behavior, lowering epsilon lowers the probability of the explicit random-action branch.
If a pair stops being visited, its entry stops changing, and the theorem's coverage premise fails.

## Function-approximation target

A DQN target also takes a maximum over estimated next-action values.
The constructed noisy-estimate example above shows how a maximum can exceed the maximum of the underlying values in expectation.
Double Q-learning separates action selection from evaluation in the tabular setting, so the same estimate is not used for both parts of the target.
This derivation does not prove a corresponding bound for DQN.
