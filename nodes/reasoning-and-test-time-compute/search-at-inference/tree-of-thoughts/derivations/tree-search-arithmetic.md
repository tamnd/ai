# The arithmetic of searching over thoughts

This derivation backs the mathematical claims of the tree-of-thoughts page. The check file
`tree-search-arithmetic.check.py` tests every statement numerically, including the counts by brute-force enumeration.

A thought is one coherent intermediate step, such as one equation in the Game of 24. A state is the input together
with the thoughts so far. A search of depth T builds T thoughts before it reaches an answer.

## 1. The size of the Game of 24 tree

A state of the Game of 24 holds n numbers. One thought picks an unordered pair of them, C(n, 2) choices, and one of
six operations: a + b, a × b, a − b, b − a, a / b and b / a. The result replaces the pair, so the next state holds
n − 1 numbers. Counting every move,

moves(4) = 6 · 6 = 36,  moves(3) = 3 · 6 = 18,  moves(2) = 1 · 6 = 6,

and a full game tree from four numbers has 36 · 18 · 6 = 3888 leaves. Division by zero removes a few moves for some
inputs: the paper's example 4 9 10 13 has 3880 legal leaves, because 13 − 9 = 4 and 4 − 4 = 0 make some later
divisions illegal. A tree of 3888 leaves is small for a computer and large for a language model, whose every visited
node costs at least one model call. The point of the evaluator is to visit a few dozen of these nodes and not all of
them.

## 2. Chains against a tree under a step-wise model

Take a simple model of generation. At each of the T steps, a sampled thought keeps the problem solvable with
probability q, independently of the other samples, and a wrong step cannot be undone.

A single chain of thought succeeds only if all T of its steps are good: P_chain = q^T.

n independent chains, with an oracle that recognises a correct final answer, succeed if at least one chain does:
P_n = 1 − (1 − q^T)^n.

A breadth-first tree with k proposals per step, breadth b = 1 and a perfect evaluator keeps a good thought whenever at
least one of the k proposals is good, so P_tree = (1 − (1 − q)^k)^T.

With q = 0.4 and T = 3: P_chain = 0.064. Five chains generate 15 thoughts and give P_5 = 1 − 0.936^5 = 0.2816. The
tree with k = 5 also generates 15 thoughts and gives P_tree = 0.92224^3 = 0.7844. The tree wins because it spends its
samples where they are needed: a chain that goes wrong at step one wastes its two remaining steps, while the tree
redraws at the step that failed. The comparison holds only when the evaluator is good, which section 3 relaxes, and it
ignores the evaluator's own calls, which section 4 counts.

## 3. Evaluator errors compound with depth

Let the evaluator discard the only good state at a step with probability δ. With breadth one and no backtracking, the
search keeps the good path through T steps with probability (1 − δ)^T. At δ = 0.1 this is 0.729 for T = 3, the depth of
the Game of 24, and 0.3487 for T = 10, the number of clues in a 5 × 5 crossword. The same evaluator that is
nearly harmless on a short problem loses most good paths on a long one. Two remedies follow from the formula: a wider
beam (a good state survives if it ranks in the top b and not only first) and backtracking, which revisits a parent
when the subtree under the chosen child is judged dead.

## 4. The model calls of breadth-first search

Breadth-first search with depth T, breadth b and k proposals per state expands one state at the first step and b
states at each later step, so it expands E = 1 + b(T − 1) states. It produces k · E candidate thoughts and, with v
evaluation samples per candidate, makes v · k · E evaluation calls. For the Game of 24 settings (T = 3, b = 5, k = 5,
v = 3) these are 11 expansions, 55 candidates and 165 evaluation calls. The evaluation calls outnumber the generation
calls fifteen to one, which is why the paper's ablation that swaps the evaluator model changes the cost as much as the
accuracy.

## 5. Selection by upper confidence bound

Monte Carlo tree search variants such as LATS pick the child s of a node p that maximises

UCT(s) = V(s) + w · sqrt(ln N(p) / N(s)),

where V(s) is the mean value of s, N counts visits and w weighs exploration. With w = 1 and N(p) = 10, a child with
V = 0.6 visited 5 times scores 0.6 + sqrt(ln 10 / 5) = 1.2786, and a child with V = 0.2 visited once scores
0.2 + sqrt(ln 10) = 1.7174. The rarely visited child is chosen even though its mean value is lower. An unvisited child
has N(s) = 0 and an infinite score, so every child is tried once before any is tried twice.

## 6. The self-evaluation score

Self-evaluation guided beam search scores a step by E = C^λ · P^(1 − λ), where P is the probability the model gave the
step when generating it and C is the model's own judged probability that the step is correct. With λ = 0.5 the score is
the geometric mean. A fluent step with P = 0.9 that the model judges wrong, C = 0.2, scores sqrt(0.18) = 0.4243. A less
fluent step with P = 0.5 judged right, C = 0.9, scores sqrt(0.45) = 0.6708 and ranks higher. At λ = 0 the score is the
generation probability alone, and the search reduces to ordinary beam search over the model's likelihood.

## 7. Token accounting

The paper's cost table for the Game of 24 lists 6.7k completion tokens for 100 chain-of-thought samples, about 67
tokens per sample, and 5.5k completion tokens for the tree search. At equal completion tokens the tree search buys
5500 / 67 ≈ 82 chain samples. Per problem, the tree costs $0.74 with 74% success and the best of 100 chains costs $0.47
with 49% success, which is $1.00 and $0.96 per solved problem. The best of 100 chains is an oracle figure, because it
counts a problem solved if any sample is right, so the tree's advantage at equal money comes from not needing that
oracle.
