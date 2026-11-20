---
tag: NEW-nucleus-sampling
kind: method
slug: nucleus-sampling
name: Nucleus sampling
aliases: [top-p sampling, nucleus decoding, truncation sampling]
area: language-models/decoding-and-sampling
level: core
summary: >-
  Sampling each next token from the smallest set of most probable tokens whose total probability reaches p, renormalized, so that the pool of candidates grows and shrinks with the model's confidence.
purpose: generating open-ended text from a language model that is neither bland and repetitive, as maximization gives, nor incoherent, as sampling the full distribution gives
uses: [conditional-probability@m2, shannon-entropy@m2, ancestral-sampling@al]
related_to: [beam-search@al, autoregressive-model, decoder-only-transformer, speculative-decoding, n-gram-language-model]
hyperparameters:
  - {name: p, meaning: "the probability mass the nucleus must reach", typical: "0.9 or 0.95; Holtzman et al. report their main results at p = 0.95"}
  - {name: temperature t, meaning: "the divisor of the logits before the softmax; below 1 it sharpens the distribution, above 1 it flattens it", typical: "1 with nucleus sampling; 0.7 with top-k, as in Radford et al. (2019) and Fan et al. (2018)"}
  - {name: k (top-k), meaning: "the fixed number of candidates kept by top-k sampling", typical: "10 in Fan et al.; 30 or 40 after tuning in Meister et al."}
  - {name: p_base (min-p), meaning: "the fraction of the top token's probability below which tokens are dropped", typical: "0.1 in the figures of Nguyen et al."}
cost:
  train: "none; it changes only how tokens are drawn from a trained model"
  inference: "per step, a sort of the V next-token probabilities and a cumulative sum, O(V log V), small beside the forward pass of the model that produced them"
history:
  year: 2020
  people: [Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi]
  note: "Fan, Lewis and Dauphin (ACL 2018) sampled stories from the 10 most likely words; Holtzman et al. (arXiv April 2019, ICLR 2020) named neural text degeneration and proposed nucleus sampling; Meister et al. (TACL 2023), Hewitt, Manning and Liang (Findings of EMNLP 2022) and Nguyen et al. (ICLR 2025) changed the truncation rule."
prereqs: [autoregressive-model, conditional-probability@m2, ancestral-sampling@al]
same_as:
  arxiv: ["1904.09751", "1805.04833", "2210.15191", "2202.00666", "2407.01082"]
sources:
  - {paper: "Holtzman, Buys, Du, Forbes and Choi (2020)", title: "The Curious Case of Neural Text Degeneration", locator: "ICLR 2020, arXiv 1904.09751v2, pp. 1 to 9", note: "degeneration under maximization and pure sampling, the nucleus, top-k and temperature, and the GPT-2 Large comparison"}
  - {paper: "Fan, Lewis and Dauphin (2018)", title: "Hierarchical Neural Story Generation", locator: "ACL 2018, pp. 889 to 898, PDF pp. 5 and 6", note: "top-k random sampling with k = 10 instead of beam search"}
  - {paper: "Hewitt, Manning and Liang (2022)", title: "Truncation Sampling as Language Model Desmoothing", locator: "Findings of EMNLP 2022, pp. 3414 to 3427, PDF pp. 1, 2 and 5", note: "truncation as desmoothing, the over-truncation of top-p, and eta-sampling"}
  - {paper: "Meister, Pimentel, Wiher and Cotterell (2023)", title: "Locally Typical Sampling", locator: "TACL 11, pp. 102 to 121, PDF pp. 1, 2, 7 to 10", note: "top-k and nucleus as subset problems, and locally typical sampling"}
  - {paper: "Nguyen, Baker, Neo, Roush, Kirsch and Shwartz-Ziv (2025)", title: "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs", locator: "ICLR 2025, arXiv 2407.01082v3, pp. 1 to 7", note: "min-p sampling and its behaviour at high temperatures"}
---

Nucleus sampling, or top-p sampling, picks each next token of a generated text at random from a truncated version of the model's distribution: only the smallest set of most likely tokens whose probabilities add up to at least $p$ is kept, and the kept probabilities are rescaled to sum to one.
Holtzman et al. (2020) proposed it to avoid text degeneration by truncating the unreliable tail of the distribution and sampling from the dynamic nucleus of tokens that holds the vast majority of the probability mass. {#c-ho-proposal}
It is the most common member of a family of truncation rules, with top-k, min-p, typical and $\eta$-sampling, that differ only in where they cut.

## In plain words

A language model hands you, at every step, a probability for every word in its vocabulary.
Always taking the top word gives dull text that loops; drawing from all of them lets in thousands of unlikely words, and every so often one of them derails the text.
Nucleus sampling keeps the head of the list and throws away the tail, but it decides how long the head is from the numbers themselves.
After "The capital of France is" one word has nearly all the probability, so the head is a single word.
After "I bought a" hundreds of nouns share the probability, so the head is long.
You then draw from the head, in proportion to the probabilities the model gave.

## Setting

Holtzman et al. study open-ended generation: given a sequence of $m$ tokens $x_1 \dots x_m$ as context, the task is to generate the next $n$ continuation tokens so that the completed text forms a coherent continuation. {#c-ho-task}
They set it against directed generation, such as translation, data-to-text and summarization, where the output is a constrained transformation of the input and beam search is the usual decoder, since repetition and genericness are less of a problem there. {#c-ho-directed}
The model gives a next-token distribution $P(x \mid x_{1:i-1})$ at every step, and the decoding strategy turns those distributions into text.

Two obvious strategies fail.
Maximization-based decoding methods such as beam search lead to degeneration: output text that is bland, incoherent, or stuck in repetitive loops. {#c-ho-degeneration}
Pure sampling, drawing straight from the predicted probabilities, gives text that is incoherent and almost unrelated to the context. {#c-ho-pure-incoherent}
Holtzman et al. blame an unreliable tail of tens of thousands of candidate tokens with relatively low probability that are over-represented in the aggregate. {#c-ho-tail}
At each step most of the mass sits in the nucleus, a small subset of the vocabulary that tends to range between one and a thousand candidates. {#c-ho-nucleus-size}

## The method

Given the distribution $P(x \mid x_{1:i-1})$, Holtzman et al. define the top-$p$ vocabulary $V^{(p)} \subset V$ as the smallest set of tokens whose probabilities sum to at least $p$. {#c-ho-top-p-set}

$$ \sum_{x \in V^{(p)}} P(x \mid x_{1:i-1}) \ge p $$ {#eq-top-p-set}

With $p' = \sum_{x \in V^{(p)}} P(x \mid x_{1:i-1})$, the distribution is rescaled to

$$ P'(x \mid x_{1:i-1}) = \begin{cases} P(x \mid x_{1:i-1}) / p' & \text{if } x \in V^{(p)}, \\ 0 & \text{otherwise}, \end{cases} $$ {#eq-renormalize}

and the next token is sampled from $P'$. {#c-ho-renormalize}
In practice this means selecting the highest-probability tokens whose cumulative probability mass exceeds $p$, so the size of the sampling set adjusts to the shape of the distribution at each step. {#c-ho-practice}

One step of the algorithm:

1. Compute the next-token probabilities, after any temperature, from the logits.
2. Sort the tokens by probability, from the largest.
3. Take the shortest prefix of the sorted list whose probabilities sum to at least $p$; this is $V^{(p)}$.
4. Set the other probabilities to zero, divide the kept ones by their sum $p'$, and draw a token.
5. Append it to the context and repeat until an end-of-text token or a length limit.

Step 3 always keeps at least the top token, so the method never has an empty pool, and when the top token alone has probability at least $p$ the step is greedy.

**Top-k.** Top-$k$ sampling keeps instead the set $V^{(k)}$ of size $k$ that maximizes $\sum_{x \in V^{(k)}} P(x \mid x_{1:i-1})$, and rescales as in eq-renormalize; Holtzman et al. note that the kept mass can then vary wildly from step to step, in contrast to nucleus sampling. {#c-ho-top-k}

$$ V^{(k)} = \operatorname*{arg\,max}_{S \subseteq V,\ |S| = k} \sum_{x \in S} P(x \mid x_{1:i-1}) $$ {#eq-top-k-set}

**Temperature.** Given the logits $u_{1:|V|}$ and a temperature $t$, the softmax is re-estimated as

$$ p(x = V_l \mid x_{1:i-1}) = \frac{\exp(u_l / t)}{\sum_{l'} \exp(u_{l'} / t)}, $$ {#eq-temperature}

and setting $t \in [0, 1)$ skews the distribution toward high-probability events, which implicitly lowers the mass in the tail. {#c-ho-temperature}
Temperature changes every probability but never their order, so it composes with any truncation rule; the derivation `temperature` below shows that it changes the nucleus but not the top-$k$ set.

## Derivation

Two derivations support the statements; each lists its assumptions and has a numeric check.

**The nucleus is a sorted prefix** (`top-p-set`).
Sort the probabilities as $q_1 \ge q_2 \ge \dots \ge q_V$ and let $m$ be the least index with $q_1 + \dots + q_m \ge p$; the first $m$ tokens form a smallest set with mass at least $p$, and every smallest set has exactly $m$ tokens. {#c-tp-prefix}
The kept mass satisfies $p \le p' < p + q_m$. {#c-tp-mass}
The total variation distance between $P$ and the renormalized $P'$ of eq-renormalize is $1 - p'$, the mass removed, so it is at most $1 - p$; for top-$k$ it is $1 - p'_k$, which depends on the context. {#c-tp-tv}
If $q_1 \ge p$ the nucleus is the single top token and the step is greedy; if $p > 1 - q_s$, where $q_s$ is the smallest positive probability, the nucleus is every token with positive probability and the step is pure sampling. {#c-tp-limits}
The nucleus is monotone in $p$: raising $p$ never removes a token. {#c-tp-monotone}

**Temperature keeps the order and moves the mass** (`temperature`).
For $t > 0$ the map from logits to eq-temperature preserves the order of the tokens, so the top-$k$ set does not depend on $t$. {#c-temp-order}
As $t \to 0^+$ the distribution tends to the uniform distribution on the tokens with the largest logit, and as $t \to \infty$ to the uniform distribution on $V$. {#c-temp-limits}
The entropy of eq-temperature has derivative $\operatorname{Var}(u)/t^3 \ge 0$ in $t$, where the variance of the logits is taken under the tempered distribution, so the entropy never decreases as $t$ grows. {#c-temp-entropy}
Lowering $t$ raises the share of the top tokens, so for a fixed $p$ the nucleus at a lower temperature has at most as many tokens as at a higher one. {#c-temp-nucleus}

## Worked example

Take five tokens with probabilities $P = (0.5, 0.2, 0.15, 0.1, 0.05)$ and $p = 0.8$.
The cumulative sums are $0.5, 0.7, 0.85$, so the nucleus is the first three tokens, $p' = 0.85$, and $P' = (10/17, 4/17, 3/17, 0, 0) \approx (0.588, 0.235, 0.176, 0, 0)$.
The total variation from $P$ is $0.15 = 1 - p'$.
Top-$k$ with $k = 2$ keeps $0.7$ of the mass and gives $(5/7, 2/7, 0, 0, 0)$.
At temperature $t = 0.5$ the probabilities are squared and renormalized, which gives $(0.25, 0.04, 0.0225, 0.01, 0.0025)/0.325 \approx (0.769, 0.123, 0.069, 0.031, 0.008)$; the cumulative sums are $0.769$ and $0.892$, so the nucleus at $p = 0.8$ shrinks to two tokens, while the top-2 set is unchanged.
Min-p with $p_{\text{base}} = 0.25$ keeps the tokens with probability at least $0.25 \times 0.5 = 0.125$, the same three as the nucleus.

The rule adapts to the context.
On a flat distribution over 50 equally likely tokens the nucleus at $p = 0.95$ has 48 tokens; on the distribution $(0.96, 0.02, 0.01, 0.01)$ it has one, the top token, and the step is greedy.
The tests `test_worked_example`, `test_temperature_shrinks_nucleus`, `test_flat_and_peaked` and `test_min_p_worked_example` check these numbers.

## Hyperparameters

The main hyperparameter is $p$, the `top_p` of most generation libraries.
Holtzman et al. report their main comparison at $p = 0.95$, and Meister et al. found 0.9 or 0.95 best for nucleus sampling after tuning with MAUVE, as the Evidence section reports.
Values near 1 approach pure sampling and values near 0 approach greedy decoding, by the limits under Derivation.
The temperature $t$ is usually left at 1 with nucleus sampling, since lowering it also shrinks the nucleus; when both are set, say in which order they are applied, as under Pitfalls.
Top-$k$ is often combined with top-$p$ as a cap on the candidate count; with $k$ large it rarely binds.

## Cost

Training is unchanged.
At each step the sampler sorts the $|V|$ next-token probabilities and takes a cumulative sum, $O(|V| \log |V|)$ work, which is small beside the forward pass of the model that produced the logits.
A partial selection of the top tokens avoids the full sort when the nucleus is known to be small; min-p needs only the maximum, $O(|V|)$.
The method adds no memory beyond one vector of size $|V|$ per sequence.

## Evidence

Holtzman et al. generate from GPT-2 Large, 762M parameters, trained on WebText; they produce 5,000 passages that end at an end-of-document token or at 200 tokens, conditioned on the first paragraph, 1 to 40 tokens, of held-out WebText documents. {#c-ho-setup}
On these passages the perplexity of human text is 12.38; greedy decoding gives 1.50, beam search with $b = 16$ gives 1.48, pure sampling 22.73, top-$k$ with $k = 40$ gives 6.88, top-$k$ with $k = 40$ and $t = 0.7$ gives 3.48, and nucleus sampling with $p = 0.95$ gives 13.13. {#c-ho-perplexity}
In the repetition column of the same table, human text scores 0.28%, greedy decoding 73.66%, beam search with $b = 16$ 28.94% and nucleus sampling with $p = 0.95$ 0.36%. {#c-ho-repetition}
Across parameter settings, nucleus and top-$k$ sampling have the least repetition for reasonable parameter ranges, and temperature sampling repeats more unless the temperature is very high. {#c-ho-repetition-range}
In the vocabulary distributions pure sampling is the closest to human text, followed by nucleus sampling. {#c-ho-zipf}
On Self-BLEU, common values of $t \in [0.5, 1]$ and $k \in [1, 100]$ give high self-similarity, whereas the usual values $p \in [0.9, 1)$ closely match the human text. {#c-ho-self-bleu}
On HUSE, a combined human and statistical evaluation, nucleus sampling is the best overall decoding strategy. {#c-ho-huse}

The finding is not that maximization fails to find the best text; the per-token probability of natural text is on average much lower than that of beam search output, and natural language rarely stays in a high-probability zone for several steps in a row. {#c-ho-natural-not-max}
Holtzman et al. conjecture that this is an intrinsic property of human language, and conclude that training larger models with standard per-word objectives is unlikely to solve the problem on its own. {#c-ho-conjecture}

Other studies agree that truncated sampling beats maximization for open-ended text.
Fan et al. found top-$k$ sampling from the 10 most likely candidates substantially more effective than beam search, which tends to produce common phrases and repetitive text from the training set. {#c-fan-top-k}
In comparison with nucleus and top-$k$ sampling, locally typical sampling gives competitive quality on summarization and story generation while consistently reducing degenerate repetitions. {#c-me-results}

## Limitations and failure modes

**It cuts too much when the model is sure.**
Top-$p$ can truncate words with up to $1 - p$ probability simply because other high-probability words already cover $p$; after "My name", GPT-2 gives "is" probability 0.96, and less likely candidates such as "'s", "was" and "isn" are cut. {#c-he-overtruncate}
Hewitt et al. report the same for a document that starts with "Donald": GPT-2 Large gives "Trump" probability 0.96, and top-$p$ with $p = 0.95$ allows no other continuation. {#c-he-donald}

**It does not tame high temperatures.**
Nguyen et al. report that top-$p$ often struggles to balance quality and diversity, especially at higher temperatures, which leads to incoherent or repetitive outputs. {#c-mp-problem}
The reason is that a high temperature flattens the head and the tail together, so the nucleus at a fixed $p$ grows into the tail.

**It cannot fix the model.**
The rule removes low-probability tokens, but it keeps the model's preferences inside the nucleus, including its pull toward repetition: Holtzman et al. find that the probability of a repeated phrase increases with each repetition, a positive feedback loop. {#c-ho-repeat-loop}

**It is a rule for open-ended text.**
For directed tasks, where the output is tightly scoped by the input, beam search remains the usual decoder (see Setting).

## Variants

The variants all keep a subset $C$ of the vocabulary and renormalize as in eq-renormalize; they differ in how they choose $C$.

**Top-k.** Fan et al. generate stories with a top-$k$ random sampling scheme, sampling at each step from the $k = 10$ most likely candidates. {#c-fan-scheme}
They note that completely random sampling can introduce very unlikely words that damage generation, since the model has not seen such mistakes during training, and that the restriction to the 10 most likely candidates reduces that risk. {#c-fan-why}
Holtzman et al. argue that a constant $k$ is sub-optimal across contexts: if $k$ is small the text can be bland or generic where the head is flat, and if $k$ is large the top-$k$ set includes inappropriate candidates whose probability renormalization increases. {#c-ho-k-problem}

**Locally typical sampling.** Meister et al. hypothesize that for text to be perceived as natural, each word should have an information content close to its expected information content given the prior context. {#c-me-hypothesis}
Their rule sorts the words by the distance between their information content $-\log p(y \mid y_{<t})$ and the conditional entropy, and greedily takes words from that list until their cumulative probability exceeds a threshold $\tau$, at the same $O(|V| \log |V|)$ cost as nucleus sampling. {#c-me-algorithm}

$$ C(y_{<t}) = \operatorname*{arg\,min}_{C \subseteq V} \sum_{y \in C} \big| H(Y_t \mid Y_{<t} = y_{<t}) + \log p(y \mid y_{<t}) \big| \quad \text{s.t.} \quad \sum_{y \in C} p(y \mid y_{<t}) \ge \tau $$ {#eq-typical}

They found $\tau = 0.2$ best for story generation and $\tau = 0.95$ for summarization, and $k$ of 30 or 40 and nucleus $p$ of 0.9 or 0.95 best for the baselines. {#c-me-hparams}

**$\eta$-sampling.** Hewitt et al. view a language model as a mixture of a true distribution and a smoothing distribution, so that truncation is desmoothing: estimating a subset of the support of the true distribution. {#c-he-desmoothing}
The word comes from the n-gram language model, whose smoothing deliberately moves probability to continuations never seen in training; truncation takes such mass away again at sampling time.
$\eta$-sampling keeps the words with probability above

$$ \eta = \min\big(\epsilon,\ \alpha \exp(-h_{\theta, x_{<i}})\big), $$ {#eq-eta}

where $h_{\theta, x_{<i}}$ is the entropy of the model's next-token distribution and $\exp(-h)$ is the probability of a word in the uniform distribution with that entropy. {#c-he-eta}
They set $\alpha = \sqrt{\epsilon}$, which leaves one hyperparameter, and suggest thinking of $\epsilon \approx 0.0009$. {#c-he-alpha}
They report that $\eta$-sampling generates long English documents that humans find more plausible and is better at breaking out of repetition. {#c-he-results}

**Min-p.** Nguyen et al. set the threshold relative to the top token: with $p_{\max} = \max_v P(v \mid x_{1:t-1})$ and a base $p_{\text{base}} \in (0, 1]$, they keep the tokens whose probability is at least $p_{\text{base}} \times p_{\max}$ and sample from them in proportion to their probabilities. {#c-mp-rule}

$$ V_{\min} = \{ v \in V : P(v \mid x_{1:t-1}) \ge p_{\text{base}} \times p_{\max} \} $$ {#eq-min-p}

The threshold is applied after temperature scaling. {#c-mp-after-temperature}
On GPQA Main with Mistral 7B, min-p achieves higher accuracy than the other methods at every temperature they test. {#c-mp-gpqa}

## When to use it

Use nucleus sampling, or one of its variants, for open-ended generation from a language model: stories, dialogue, continuations, and sampling several candidates to choose from.
Start from $p$ between 0.9 and 0.95 at temperature 1, and lower $p$ or the temperature when the text wanders, raise them when it repeats.
For high temperatures, prefer a threshold that scales with the top probability, such as min-p, to a fixed mass.
For short answers that have one right output, and for directed tasks such as translation, greedy decoding or beam search is usually the better start.
Nucleus sampling decides which distribution each token is drawn from, while speculative decoding only changes how fast tokens are drawn: its rejection step reproduces whatever target distribution it is given, so the truncated $P'$ of eq-renormalize can be that target.

## Pitfalls

> [!pitfall]
> **Stopping one token early.** The nucleus is the smallest prefix whose sum reaches $p$, so it includes the token that crosses $p$. Keeping only the tokens whose cumulative sum stays below $p$ can return an empty set when the top token alone exceeds $p$. The test `test_includes_crossing_token` checks both cases.

> [!pitfall]
> **Renormalizing by the wrong mass.** Dividing by $p$ instead of the kept mass $p'$ gives probabilities that do not sum to one. The test `test_worked_example` checks $P'$ against exact fractions.

> [!pitfall]
> **Order of temperature and truncation.** Tempering after the nucleus is chosen leaves the set unchanged and only reweights it; tempering first, as min-p prescribes, changes the set. The test `test_temperature_shrinks_nucleus` checks that the two orders differ on the worked example.

> [!pitfall]
> **Ties at the boundary.** Equal probabilities around the cut can make the nucleus depend on the sort. The reference sorts stably by index, so the result is deterministic. The test `test_ties_are_deterministic` checks it.

## History

Fan, Lewis and Dauphin (2018) sampled stories from the 10 most likely words, and the high-quality GPT-2 samples of 2019 relied on top-$k$ sampling as well. {#c-ho-gpt2-top-k}
Holtzman et al. posted "The Curious Case of Neural Text Degeneration" in April 2019 and published it at ICLR 2020; it gave nucleus sampling its name and made the case against maximization for open-ended text.
Later work kept the idea of truncating the tail but changed the cut: locally typical sampling (Meister et al., posted in 2022, TACL 2023), $\eta$-sampling (Hewitt, Manning and Liang, 2022) and min-p (Nguyen et al., posted in 2024, ICLR 2025).
