---
tag: JKWQF4
kind: method
slug: speculative-decoding
name: Speculative decoding
aliases: [speculative sampling, SpS, draft-and-verify decoding, modified rejection sampling]
area: efficient-ai/speculative-and-parallel-decoding
level: core
summary: >-
  Sampling from a large language model several tokens per call: a small model drafts, the large model checks the drafts in parallel, and a rejection rule keeps the output distribution exact.
purpose: cutting the latency of sampling from a large language model when decoding is limited by memory bandwidth rather than arithmetic, while keeping the samples exactly distributed as the large model's
uses: [ancestral-sampling@al, total-variation-distance-between-probability-measures@m2, geometric-distribution@m2, linearity-of-expectation@m2]
related_to: [knowledge-distillation@al, beam-search@al, autoregressive-model@m2]
hyperparameters:
  - {name: gamma (K), meaning: "the number of tokens the draft model proposes per run of the target model", typical: "2 to 10 in Leviathan et al.'s analysis and 3 to 7 in their T5 runs; K = 4 in Chen et al.'s main table, with XSum fastest at K = 3"}
  - {name: draft model, meaning: "the small model that proposes tokens; it sets the acceptance rate alpha and the cost coefficient c", typical: "about two orders of magnitude smaller than the target in Leviathan et al.; a 4B wide, shallow model for the 70B Chinchilla in Chen et al."}
  - {name: sampling adjustment, meaning: "temperature, top-k or nucleus truncation applied to both distributions before the rule", typical: "the same adjustment for draft and target, as in both papers"}
cost:
  train: "none if an existing smaller model can serve as the draft; otherwise the training of a draft model"
  inference: "per run, gamma serial draft steps and one target call on gamma + 1 prefixes in parallel; on average (gamma c_hat + gamma + 1) / E[N] times the arithmetic of standard decoding, for E[N] tokens per run"
history:
  year: 2023
  note: "Stern et al. (2018) decoded several tokens in parallel for greedy decoding with a trained model; Leviathan, Kalman and Matias (arXiv 2022, ICML 2023) and Chen et al. (2023), independently, gave the rejection rule that keeps sampling exact; Cai et al. (2024) replaced the draft model with extra heads on the target."
prereqs: [autoregressive-model@m2, ancestral-sampling@al, total-variation-distance-between-probability-measures@m2]
same_as:
  arxiv: ["2211.17192", "2302.01318", "2401.10774"]
sources:
  - {paper: "Leviathan, Kalman and Matias (2023)", title: "Fast Inference from Transformers via Speculative Decoding", locator: "arXiv 2211.17192v2, Sections 1 to 4 and Appendix A", note: "speculative sampling, Algorithm 1, the acceptance rate, the expected tokens, walltime and arithmetic analysis, and the T5, GPT-like and LaMDA results"}
  - {paper: "Chen, Borgeaud, Irving, Lespiau, Sifre and Jumper (2023)", title: "Accelerating Large Language Model Decoding with Speculative Sampling", locator: "arXiv 2302.01318v1, Sections 1 to 5 and the proof of Theorem 1", note: "the same rule found independently, the draft-model design for Chinchilla 70B, and the speedups and the effect of K"}
  - {paper: "Cai, Li, Geng, Peng, Lee, Chen and Dao (2024)", title: "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", locator: "arXiv 2401.10774v3, the abstract and Sections 1 and 2", note: "extra decoding heads and tree attention instead of a draft model"}
  - {paper: "Shazeer (2019)", title: "Fast Transformer Decoding: One Write-Head is All You Need", locator: "arXiv 1911.02150v1, p. 1", note: "the memory-bandwidth bottleneck of incremental decoding and multi-query attention"}
  - {paper: "Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón and Sanghai (2023)", title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", locator: "arXiv 2305.13245v3, p. 1", note: "decoder inference bound by loading weights and keys and values, and grouped-query attention"}
---

Speculative decoding samples from a large autoregressive model, the target, several tokens per call instead of one.
A cheap draft model guesses the next $\gamma$ tokens, the target scores all the guesses in one parallel call, and a modified rejection rule keeps a prefix of the guesses and replaces the first rejected one, so that every token is still an exact sample from the target.
Leviathan, Kalman and Matias, who named it, and Chen et al., independently, introduced it; it is now a standard part of serving large language models.

## In plain words

A language model writes one token per forward pass, and each pass of a large model mostly waits for its weights to stream out of memory.
Checking five proposed tokens costs about as much as writing one, because the same weights serve all five positions.
So let a small model write a short draft and let the large model check it.
Where the draft agrees with what the large model would have done, keep it; at the first disagreement, take a corrected token and start again.
The trick is the rule for "agrees": it accepts a drafted token at random, with a probability chosen so that the tokens that come out are distributed exactly as if the large model had written them alone.

## Setting

Decoding $K$ tokens from an autoregressive model such as a Transformer takes $K$ serial runs of the model. {#c-le-serial}
Leviathan et al. build on two observations: hard language-modelling tasks often contain easier subtasks that more efficient models approximate well, and speculative execution with a new sampling method can run the large model in parallel on the outputs of the approximation model without changing the distribution. {#c-le-observations}
Speculative execution, a common optimization in processors such as branch prediction, does work in parallel before knowing whether it is needed; their method generalizes it to the stochastic setting. {#c-le-spec-exec}
The resources are there because inference from large models is often bottlenecked not on arithmetic but on memory bandwidth and communication, so additional computation may be available. {#c-le-memory}
Chen et al. make the same point: transformer sampling is typically memory-bandwidth bound, and scoring a short continuation in parallel with the target model has a latency comparable to sampling a single token from it. {#c-ch-bandwidth}
Chen et al. developed their method concurrently and independently of Leviathan et al. and state that the core idea is the same. {#c-ch-concurrent}
The bottleneck was named earlier by Shazeer (2019): incremental decoding cannot be parallelized over the sequence and is slow because it reloads the large key and value tensors at every step. {#c-sh-bandwidth}
Ainslie et al. (2023) add the decoder weights, which are also loaded at every decoding step. {#c-ai-bottleneck}
Multi-query attention attacks the bottleneck by sharing keys and values across heads, at a minor cost in quality, {#c-sh-mqa}
and grouped-query attention keeps a few key-value heads to get close to multi-head quality at close to multi-query speed. {#c-ai-gqa}
Those methods shrink the memory traffic per token by changing the model; speculative decoding leaves the model alone and spends the idle arithmetic on scoring several tokens per load.
Other techniques, such as distillation, sparsification, quantization and architecture modification, make inference more efficient for all tokens but usually also change the outputs of the model. {#c-le-other-methods}
Knowledge distillation can still help, by giving a better draft.
Leviathan et al. suggest that further improvements might be obtained from approximation models with custom training procedures, such as standard distillation with soft targets from $M_p$, or optimizing $M_q$ for $\alpha$ directly. {#c-le-distill-draft}

## The method

### Notation

$M_p$ is the target model and $M_q$ the more efficient approximation (draft) model for the same task; $p(x)$ and $q(x)$ are their distributions over the next token given a prefix. {#c-le-notation}
Greedy decoding, top-k, nucleus sampling and temperature can all be cast as standard sampling from an adjusted distribution, so $p$ and $q$ are taken after that adjustment and the method covers these decoding schemes. {#c-le-adjusted}
Chen et al. likewise state that their scheme works with nucleus, top-k and temperature sampling. {#c-ch-decoding-schemes}
Chen et al. write the letters the other way round: in their paper $q$ is the target and $p$ the draft.

### Speculative sampling

To sample $x \sim p$, sample $x \sim q$ instead, keep it if $q(x) \le p(x)$, and if $q(x) > p(x)$ reject it with probability $1 - p(x)/q(x)$ and sample again from $p'(x) = \mathrm{norm}(\max(0, p(x) - q(x)))$. {#c-le-rule}
In one formula, the draft token is accepted with probability

$$ \min\Big(1, \frac{p(x)}{q(x)}\Big) , $$ {#eq-accept}

and a rejected token is replaced by a draw from the residual

$$ p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{z} \max(0, p(z) - q(z))} . $$ {#eq-residual}

For any distributions $p(x)$ and $q(x)$, Leviathan et al. find that $x$ sampled in this way is indeed distributed as $p(x)$ (their Appendix A.1). {#c-le-exact}
Chen et al. call the same rule modified rejection sampling and prove that it recovers the target distribution. {#c-ch-theorem}

### Speculative decoding

Algorithm 1 of Leviathan et al. samples $\gamma$ guesses $x_1, \dots, x_\gamma$ from $M_q$ autoregressively, runs $M_p$ in parallel on the prefix and on the prefix extended by each guess to get $p_1, \dots, p_{\gamma+1}$, and accepts the guesses from left to right by speculative sampling. {#c-le-algorithm}
If a guess is rejected, an additional token is sampled from an adjusted distribution to fix the first one that was rejected; if all $\gamma$ are accepted, an additional token is sampled from $p_{\gamma+1}$, the bonus token. {#c-le-bonus}
Each run of the target therefore produces at least one new token and up to $\gamma + 1$, and the number of serial runs of the target can never exceed that of standard decoding. {#c-le-at-least-one}
Chen et al. describe the same loop: generate a draft of length $K$, score it with the target, and accept a subset of the draft from left to right with modified rejection sampling. {#c-ch-loop}

### Acceptance rate and speed

The acceptance rate $\beta$ for a prefix is the probability of accepting $x \sim q$ by speculative sampling, and $\alpha = E(\beta)$ measures how well $M_q$ approximates $M_p$. {#c-le-beta}
By their Theorem 3.5, $\beta = 1 - D_{LK}(p, q) = \sum_x \min(p(x), q(x))$. {#c-le-dlk}
Averaging over prefixes,

$$ \alpha = E\Big(\sum_x \min(p(x), q(x))\Big) . $$ {#eq-alpha}

Under the simplifying assumption that the $\beta$s are i.i.d., the number of tokens from one run is a capped geometric variable with success probability $1 - \alpha$ and cap $\gamma + 1$, with mean {#c-le-capped}

$$ E(\#\text{tokens}) = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha} . $$ {#eq-expected-tokens}

With $c$ the ratio between the time of one run of $M_q$ and one run of $M_p$, and assuming the $\gamma + 1$ target runs fit in the walltime of one, the expected walltime improvement is {#c-le-walltime}

$$ \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)} . $$ {#eq-walltime}

If $\alpha > c$ some $\gamma$ gives an improvement, by a factor of at least $(1 + \alpha)/(1 + c)$. {#c-le-corollary}
With $\hat c$ the ratio of arithmetic operations per token of $M_q$ to $M_p$, the expected total arithmetic grows by the factor {#c-le-ops}

$$ \frac{(1 - \alpha)(\gamma \hat c + \gamma + 1)}{1 - \alpha^{\gamma+1}} . $$ {#eq-ops}

## Derivation

Two derivations support the formulas; each lists its assumptions and has a numeric check.

**Speculative sampling is exact** (`speculative-sampling`).
Take distributions $p$ and $q$ on a finite vocabulary, $x \sim q$ and an independent uniform $u$.
The probability of accepting and outputting $x$ is $q(x)\min(1, p(x)/q(x)) = \min(p(x), q(x))$, rejection has probability $1 - \sum \min(p, q) = \sum \max(0, p - q)$, which is the normalizer of $p'$, so the output has probability $\min(p(y), q(y)) + \max(0, p(y) - q(y)) = p(y)$. {#c-ss-exact}
The acceptance probability is $\beta = \sum_x \min(p(x), q(x))$. {#c-ss-accept-mass}
Since $|a - b| = a + b - 2\min(a, b)$, $\beta = 1 - \mathrm{TV}(p, q)$, one minus the total variation distance. {#c-ss-tv}
So $\beta = 1$ exactly when $p = q$ and $\beta = 0$ exactly when the supports are disjoint. {#c-ss-extremes}
For greedy decoding $p$ is one-hot on a token $t$: a draft is accepted exactly when it is $t$, and the output is always $t$. {#c-ss-greedy}
A token with $p(y) = 0$ is never output, whatever $q$ proposes. {#c-ss-zero}

**Expected tokens per run** (`expected-tokens`).
Assume each draft position is accepted with probability $\alpha$ independently, given that the earlier ones were.
Then $N$, the tokens per run, has $P(N = n) = \alpha^{n-1}(1 - \alpha)$ for $n \le \gamma$ and $P(N = \gamma + 1) = \alpha^\gamma$. {#c-et-pmf}
Writing $N = 1 + \sum_{k=1}^{\gamma} \mathbb 1[\text{the first } k \text{ accepted}]$ gives $E[N] = \sum_{k=0}^{\gamma} \alpha^k$, which is eq-expected-tokens. {#c-et-mean}
It lies between 1 and $\min(\gamma + 1, 1/(1 - \alpha))$ and grows with both $\alpha$ and $\gamma$. {#c-et-bounds}
A run costs $\gamma c + 1$ target times and yields $E[N]$ tokens, which gives eq-walltime. {#c-et-walltime}
At $\gamma = 1$ the improvement is $(1 + \alpha)/(1 + c)$, above 1 exactly when $\alpha > c$. {#c-et-gamma-one}
A run spends $\gamma \hat c + \gamma + 1$ target units of arithmetic for $E[N]$ tokens, which gives eq-ops. {#c-et-ops}
At $c = \hat c = 0$ these formulas give, for example, $3.69\times$ speed and $1.63\times$ arithmetic at $\alpha = 0.8$, $\gamma = 5$. {#c-et-tables}

## Worked example

Take three tokens with target $p = (1/2, 1/3, 1/6)$ and draft $q = (1/4, 1/2, 1/4)$.
The acceptance probabilities of eq-accept are $1$, $2/3$ and $2/3$, and the residual of eq-residual is $\max(0, p - q) = (1/4, 0, 0)$, normalized to $(1, 0, 0)$.
Acceptance gives token 0 with probability $1/4$, token 1 with $1/2 \cdot 2/3 = 1/3$, token 2 with $1/4 \cdot 2/3 = 1/6$; the rejection mass is $1 - 3/4 = 1/4$, all of it going to token 0.
The output law is $(1/4 + 1/4, 1/3, 1/6) = p$.
Here $\alpha = 3/4$ and $\mathrm{TV}(p, q) = 1/4$.
If every position looked like this, $\gamma = 3$ would give $E[N] = (1 - (3/4)^4)/(1/4) = 175/64 \approx 2.73$ tokens per target call, and with $c = 0.05$ a walltime improvement of $2.73/1.15 \approx 2.38$, at $4.15/2.73 \approx 1.52$ times the arithmetic for $\hat c = 0.05$.
The tests `test_acceptance_probability_worked_example`, `test_residual` and `test_expected_tokens` check these numbers.

## Hyperparameters

Leviathan et al. give the optimal $\gamma$ as a function of $\alpha$ and $c$ in their Figure 3, and find that a draft model about two orders of magnitude smaller than the target usually performed best, balancing $\alpha$ and $c$. {#c-le-size}
Chen et al. find that as $K$ grows the speedup plateaus or regresses, with XSum fastest at $K = 3$, and that the variance of the time to generate a sequence grows with $K$. {#c-ch-k}
Larger $K$ may give marginally greater mean speedups in some cases, but the larger variance matters where P90 or P99 latency is of concern. {#c-ch-tail}

## Cost

In Leviathan et al.'s experiments, with $M_q$ typically a couple of orders of magnitude smaller than $M_p$, $c$ was always less than 0.05 and often negligibly close to 0. {#c-le-c}
For a draft of negligible cost the improvement of eq-walltime is bounded above by $1/(1 - \alpha)$. {#c-le-negligible}
Chen et al. note that the time per loop grows roughly linearly with $K$, because of the extra model calls. {#c-ch-linear}
Their draft model has 4B parameters and 8 layers, and samples at 1.8 ms per token against 14.1 ms per token for Chinchilla 70B on 16 TPU v4. {#c-ch-draft}
They trained a wide, shallow draft because a 7B model on 16 TPUs would be slower than on fewer, through communication overhead. {#c-ch-wide}

## Evidence

On T5-XXL, Leviathan et al. report a 2X-3X acceleration compared to the T5X implementation, with identical outputs. {#c-le-headline}
With T5-small as the draft, at batch size 1 on one TPU-v4, English-German translation runs 3.4X faster at temperature 0 and 2.6X at temperature 1, and CNN/DM summarization 3.1X and 2.3X. {#c-le-t5}
Drafts some two orders of magnitude below the target in size tend to give $\alpha$ between 0.5 and 0.9, and the sharper the adjusted distribution, the higher $\alpha$. {#c-le-alpha-range}
Even a bigram model has $\alpha$ of 0.2 on translation, which gives 1.25X at negligible cost. {#c-le-bigram}
Theoretical predictions mostly match measured runtimes, and Leviathan et al. trace the larger differences to optimization differences between their implementation and the baseline, and to the i.i.d. assumption being only approximate. {#c-le-predicted}
On Chinchilla 70B, Chen et al. report a 2 to 2.5 times decoding speedup, distributed over many devices, with no loss of sample quality. {#c-ch-headline}
On XSum with nucleus sampling, latency falls from 14.1 to 7.52 ms per token, 1.92 times faster, and on HumanEval to 5.73 ms per token, 2.46 times, with the benchmark scores within noise. {#c-ch-table}
On HumanEval and greedy XSum the speedup went past the theoretical memory bandwidth limit that the hardware sets on autoregressive sampling. {#c-ch-exceed}
The acceptance rate depends on the domain and the decoding method; Chen et al. attribute the larger gain on code to its many common subsequences. {#c-ch-domain}

## Limitations and failure modes

**It spends arithmetic to save time.**
Latency improves through more concurrency at the cost of more arithmetic operations, so the method does not help where additional computation is not available. {#c-le-limit}
At large batch sizes, where decoding is no longer bandwidth-bound, the spare compute that pays for verification shrinks; both papers measure at batch size 1.

**Exact only up to numerics.**
Chen et al. note that their samples are not identical to those of standard sampling, because of numerics and random seeds, but come from the same distribution. {#c-ch-numerics}
In code, a bug in the residual or the bonus token changes the distribution without any visible error, so tests must compare output frequencies with the target.

**A bad draft wastes the parallel call.**
By eq-walltime a draft with $\alpha$ near $c$ gains little.
Chen et al. note that a 7B draft would provide only a modest speedup with its optimal topology, without full utilisation of the hardware during drafting. {#c-ch-7b}
Cai et al. note that a separate draft model is challenging to obtain and maintain, and harder still to integrate into a distributed system. {#c-me-draft}

## Variants

**Blockwise parallel decoding.** Stern et al. (2018) also decode several tokens in parallel, but only greedily, with a custom-trained model, aiming at task quality rather than identical outputs. {#c-le-blockwise}

**Beam search.** Speculative decoding can be applied, with some performance penalty, to beam search: run beam search with $M_q$ for $\gamma$ steps and check all of the candidates with $M_p$ in parallel. {#c-le-beam}

**Lenience.** Leviathan et al. note that if some changes to the output distribution are allowed, with nice guarantees, further inference speed improvements are possible. {#c-le-lenience}

**Medusa.** Cai et al. add extra decoding heads to the target that predict several next tokens, build multiple candidate continuations and verify them at once with tree attention. {#c-me-heads}
Across different prompt types, Cai et al. report a speedup of 2.3 to 2.8 times without compromising the quality of generation. {#c-me-results}
The rejection sampling scheme of speculative decoding can be reused with the heads to generate responses with the same distribution as the original model, but it cannot further enhance the acceleration rate; a typical acceptance scheme instead selects reasonable candidates while maintaining a similar generation quality. {#c-me-typical}

## When to use it

Use it to cut the latency of sampling from a large model at small batch sizes, when a much smaller model trained on similar data is available and the accelerator has spare compute.
Pick $\gamma$ from a measured $\alpha$ and $c$ with eq-walltime, and keep the same sampling adjustment for draft and target.
Skip it when throughput, not latency, is the goal and the hardware is already saturated.

## Pitfalls

> [!pitfall]
> **Resampling from p instead of the residual.** Drawing the replacement from $p$ over-weights the tokens where $q > p$. The test `test_output_distribution_is_exactly_p` computes the exact law, and `test_verify_token_distribution_matches_p` samples it.

> [!pitfall]
> **Dropping the bonus token.** When all $\gamma$ drafts are accepted, a token from $p_{\gamma+1}$ is free; leaving it out loses one token per run. The test `test_verify_lengths_and_bonus_token` checks the length $\gamma + 1$.

> [!pitfall]
> **Swapped letters.** In Chen et al. $q$ is the target. Accepting with $q(x)/p(x)$ in Leviathan's notation samples the draft. The tests `test_acceptance_probability_worked_example` and `test_speculative_step_follows_the_target` check the direction.

> [!pitfall]
> **Different sampling adjustments.** Temperature or top-k must be applied to both distributions before the rule, or the output follows the wrong target. The test `test_greedy_one_hot_target` checks the greedy case.

## History

Stern et al. (2018) had decoded blocks of tokens in parallel for greedy decoding with a specially trained model.
Leviathan, Kalman and Matias posted speculative decoding in November 2022, and Chen et al. posted speculative sampling, found independently, in February 2023.
Medusa (Cai et al., 2024) and other variants later replaced the separate draft model with extra heads or trees of candidates.
