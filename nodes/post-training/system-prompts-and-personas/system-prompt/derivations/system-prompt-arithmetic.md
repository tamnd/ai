# The arithmetic of system prompts

This derivation backs the mathematical claims of the system-prompt page. The check file
`system-prompt-arithmetic.check.py` tests every statement numerically.

## 1. What Ghost Attention trains on

A dialogue is [u_1, a_1, ..., u_n, a_n], user and assistant messages in turn. Ghost Attention keeps the instruction
only in u_1 and sets the loss to 0 on every token before the last assistant message a_n. With a per-token
cross-entropy the loss is then

L = −Σ_{j ∈ a_n} log p_θ(token_j | everything before it),

a sum over the tokens of a_n alone. The earlier assistant messages a_1, ..., a_{n−1} still sit in the context that
a_n is conditioned on, but they are never targets, so changing what they say changes no term's target and the
gradient only sees them as input. If every message has L tokens, the dialogue has 2nL message tokens and the loss
covers L of them, a fraction 1/(2n): one half for a single turn, 1/16 for eight turns.

## 2. Split-softmax keeps a distribution

Let α_1, ..., α_t be the attention weights of the last token, summing to 1, and let the first |s| tokens be the
system prompt, with share π = Σ_{i ≤ |s|} α_i, 0 < π < 1. Split-softmax replaces them with

α′_i = α_i π^k/π for i ≤ |s|,   α′_i = α_i (1 − π^k)/(1 − π) for i > |s|.

The new prompt share is Σ_{i ≤ |s|} α_i π^k/π = π^k, the rest gets (1 − π)(1 − π^k)/(1 − π) = 1 − π^k, and the
total is 1. Inside each group all weights are multiplied by the same factor, so their ratios are unchanged: the
prompt tokens are reweighted as a block, not reranked. For 0 < π < 1 the function k ↦ π^k is decreasing, so
π^k ≥ π^1 = π for k ≤ 1, with equality at k = 1; as k → 0 the share tends to 1. For π = 0.04 and k = 0.5 the share
becomes √0.04 = 0.2.

## 3. Why the share falls

If the last token gave every one of the t tokens in context the same weight 1/t, the system prompt's share would be
π(t) = |s|/t. A 100-token prompt then gets 100/1,000 = 0.1 at t = 1,000 and 100/4,000 = 0.025 at t = 4,000: every
token of conversation added after the prompt dilutes it. Real heads are far from uniform, which is why Li et al.
measure π(t) instead of assuming it, but the uniform case shows the direction of the effect.

## 4. Classifier-free guidance is a geometric mixture

Write p_c(w) for the next-token distribution with the system prompt and p_u(w) without. The guided score is

log p̂(w) = log p_u(w) + α (log p_c(w) − log p_u(w)) = (1 − α) log p_u(w) + α log p_c(w),

and normalizing over the vocabulary gives p̂(w) = p_c(w)^α p_u(w)^(1−α) / Z with Z = Σ_w p_c(w)^α p_u(w)^(1−α). At
α = 0 this is p_u, at α = 1 it is p_c (then Z = 1), and for α > 1 the tokens that the prompt makes more likely
gain further: the log-odds between two tokens under p̂ are α times their log-odds under p_c plus (1 − α) times
those under p_u.

## 5. The cost of repeating the prompt

System prompt repetition inserts the |s|-token prompt before each of n user turns, each time with probability p.
The number of insertions is binomial with mean pn, so the expected number of added tokens is p·n·|s|. With
|s| = 200, n = 8 and p = 1 that is 1,600 tokens, eight copies of the prompt.

## 6. The two extraction metrics

Zhang et al. call a guess g an exact match for a prompt p when every sentence of p is a substring of g, and an
approximate match when |LCS(p, g)|/|p| ≥ 0.9 over tokens. Neither implies the other.

* Take p = A B, two sentences of k words each with no word in common, and g = B A. Both sentences are substrings of
  g, so g is an exact match. A common subsequence of p and g that uses words from both A and B would need an A word
  before a B word in g, which never happens, so the longest one is A or B alone, of length k, and the recall is
  k/2k = 1/2.
* Take a 20-token prompt and a guess that leaves out one token. The LCS has 19 tokens, recall 0.95 ≥ 0.9, but the
  sentence missing the token is not a substring of g.

## 7. Interleaving defeats an n-gram filter

Suppose the model is told to write a symbol σ between every two words of its output, and σ is not a word of the
prompt. Every run of n ≥ 2 consecutive words of the output contains σ, so none of them occurs in the prompt; a
filter that blocks outputs sharing a 5-gram with the prompt lets the output through. Removing every σ gives back
the original word sequence, so the attacker recovers the prompt exactly. The defense works only as long as the
model cannot follow such formatting instructions, which matches what Zhang et al. find across model sizes.

## 8. Datamarking and encoding

Datamarking with a marker μ replaces every single space of a text with μ. If μ does not occur in the text, replacing
μ with a space gives back the text. Base64 writes each group of 3 bytes as 4 characters and pads the last group,
so n bytes become 4⌈n/3⌉ characters: 300 bytes become 400, a third more input for the model to read and decode.

## 9. Steering moves one coordinate

Let v ≠ 0 and v̂ = v/‖v‖. For h′ = h + αv, the projection onto v̂ is h′·v̂ = h·v̂ + α v·v/‖v‖ = h·v̂ + α‖v‖, and for any
u orthogonal to v, h′·u = h·u + α v·u = h·u. Adding a persona vector therefore moves the activation along that
direction by α‖v‖ and nowhere else; subtracting it, as in inference-time mitigation, moves it back by the same
amount.
