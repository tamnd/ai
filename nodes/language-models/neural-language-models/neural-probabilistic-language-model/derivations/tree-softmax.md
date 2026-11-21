---
id: tree-softmax
derives: [c-tree-normalised, c-huffman-depth]
kind: proof
assumes:
  - The vocabulary is the set of leaves of a full binary tree, every internal node having two children.
  - At internal node j the model goes to branch 0 with probability sigmoid(v_j · a + c_j) and to branch 1 with the rest.
uses: []
status: full
check: tree-softmax.check.py
lean: null
sources:
  - paper: morin-2005-hierarchical-neural-network-language-model
    locator: p. 3
  - paper: mikolov-2013-efficient-estimation-word-representations
    locator: p. 3
---

## Statement

The probability of a word is the product of the branch probabilities on the path from the root to its leaf. For
every hidden vector a these probabilities sum to one over the vocabulary, with no sum over the vocabulary. Computing
the probability of one word costs one sigmoid per node on its path. For a tree with |V| leaves and the smallest
possible maximum depth, the depth of each leaf is ⌊log2 |V|⌋ or ⌈log2 |V|⌉. For a Huffman tree built from word
frequencies q, the mean depth Σ_w q_w depth(w) lies between the unigram entropy H(q) in bits and H(q) + 1.
The unigram perplexity is 2^H(q).

## Assumptions

The tree and the counts are fixed before training; only v_j and c_j are learned.

## Steps

1. sigmoid(z) + sigmoid(−z) = 1, so the two branch probabilities at each node sum to one.
2. By induction from the leaves: the probabilities of the leaves below a node, divided by the probability of reaching
   the node, sum to sigmoid(z) · 1 + sigmoid(−z) · 1 = 1. At the root this is the whole vocabulary.
3. A path is a sequence of binary decisions, so a tree whose leaves all have depth at most L has at most 2^L leaves.
   The smallest L is ⌈log2 |V|⌉. Huffman's construction on equal counts merges the lightest pairs first, which gives
   depths ⌊log2 |V|⌋ and ⌈log2 |V|⌉ only.
4. Huffman codes are optimal prefix codes, and the expected length of an optimal binary prefix code lies in
   [H(q), H(q) + 1). The leaf depths are the code lengths.
5. For |V| = 10,000, log2 |V| = 13.29, so one word costs 13 or 14 sigmoids against 10,000 terms for the softmax, a
   ratio of about 750. Morin and Bengio's unigram entropy of about 9.16 bits would make a Huffman tree cheaper still;
   they used a WordNet tree instead.

## Result and check

The check builds Huffman trees for equal and Zipf counts, verifies that the leaves form a complete prefix code
(Σ 2^−depth = 1), that random branch probabilities give leaf probabilities summing to one, the depth bounds, and
that the mean depth lies in [H, H + 1).
