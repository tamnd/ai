---
tag: NEW-tree-of-thoughts
kind: method
slug: tree-of-thoughts
name: Tree of Thoughts
aliases: [ToT, tree-of-thought prompting, tree of thoughts prompting, deliberate problem solving, thought tree search, self-evaluation guided beam search, Reasoning via Planning, RAP, Language Agent Tree Search, LATS, Graph of Thoughts, GoT, Stream of Search]
area: reasoning-and-test-time-compute/search-at-inference
level: core
summary: >-
  Running a search over intermediate reasoning steps, where a language model proposes candidate next steps and judges partial solutions, so that weak branches are dropped and dead ends can be left.
purpose: reasoning
uses: [chain-of-thought-prompting, self-consistency, beam-search@al, monte-carlo-tree-search@al, heuristic-function@al, state-space-search@al]
related_to: [process-reward-model, test-time-scaling, a-star-search@al, ucb1@al]
hyperparameters:
  - {name: thought size, meaning: "how much text one step of the search holds", typical: "one equation in the Game of 24; one word in crosswords; a whole writing plan in creative writing"}
  - {name: proposals, meaning: "the number k of candidate thoughts generated from each state", typical: "5 in Yao et al.; a single propose prompt that lists several steps at once when the steps are short"}
  - {name: breadth, meaning: "the number b of states kept per depth in breadth-first or beam search", typical: "1 to 5 in Yao et al.; 5 for the Game of 24"}
  - {name: evaluation samples, meaning: "how many times each state is valued or how many votes are cast", typical: "3 value samples per thought in the Game of 24; 5 votes per step in creative writing"}
  - {name: value threshold and step limit, meaning: "the value below which depth-first search prunes a subtree, and the maximum number of steps it may take", typical: "pruning on an impossible verdict; 100 steps for crosswords"}
  - {name: exploration weight, meaning: "the weight w of the exploration term in UCT for Monte Carlo tree search variants", typical: "about 1; tuned per task"}
cost:
  train: "none; the method prompts a pretrained model, except for variants such as Stream of Search that train the model on search traces"
  inference: "tens to hundreds of model calls per problem, for proposals and for evaluations; 5 to 100 times the generated tokens of one chain of thought in Yao et al."
history:
  year: 2023
  people: [Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan]
  note: "Yao et al. (arXiv 2305.10601, May 2023, NeurIPS 2023) named Tree of Thoughts; in the same month Xie et al. posted self-evaluation guided beam search and Hao et al. RAP with Monte Carlo tree search; Graph of Thoughts (August 2023), LATS (October 2023, ICML 2024) and Stream of Search (April 2024) followed."
prereqs: [chain-of-thought-prompting, self-consistency, beam-search@al, state-space-search@al]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["2305.10601", "2305.00633", "2305.14992", "2310.04406", "2308.09687", "2404.03683"]
sources:
  - {paper: "Yao, Yu, Zhao, Shafran, Griffiths, Cao and Narasimhan (2023)", title: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", locator: "NeurIPS 2023, arXiv 2305.10601v2, pp. 1 to 9 and 14", note: "the framework, the three tasks, the ablations, the generation and evaluation swap and the cost tables"}
  - {paper: "Xie, Kawaguchi, Zhao, Zhao, Kan, He and Xie (2023)", title: "Self-Evaluation Guided Beam Search for Reasoning", locator: "NeurIPS 2023, arXiv 2305.00633v3, pp. 1, 4, 6, 8 and 18", note: "the combined score, stochastic beam search and the comparison at equal tokens"}
  - {paper: "Hao, Gu, Ma, Hong, Wang, Wang and Hu (2023)", title: "Reasoning with Language Model is Planning with World Model", locator: "arXiv 2305.14992v2, pp. 1, 2 and 8", note: "the model as world model and agent, with Monte Carlo tree search"}
  - {paper: "Zhou, Yan, Shlapentokh-Rothman, Wang and Wang (2024)", title: "Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models", locator: "ICML 2024, arXiv 2310.04406v3, pp. 1, 4 and 9", note: "Monte Carlo tree search with environment feedback and self-reflection, and its limits"}
  - {paper: "Besta, Blach, Kubicek, Gerstenberger and others (2024)", title: "Graph of Thoughts: Solving Elaborate Problems with Large Language Models", locator: "arXiv 2308.09687v4, pp. 1 and 2", note: "thoughts as vertices of a graph and aggregation"}
  - {paper: "Gandhi, Lee, Grand, Liu, Cheng, Sharma and Goodman (2024)", title: "Stream of Search (SoS): Learning to Search in Language", locator: "arXiv 2404.03683v1, p. 1", note: "training a model on serialized search traces"}
---

Tree of Thoughts turns a language model into the two halves of a classical search program: a generator that proposes candidate next steps and an evaluator that judges how promising a partial solution is.
A search algorithm, breadth-first, depth-first or Monte Carlo tree search, then decides which partial solutions to extend, which to drop and when to go back.
It applies to problems where one wrong early step ruins a chain of thought and where the model can judge a partial solution better than it can produce a whole one.

## In plain words

Chain-of-thought prompting writes one line of reasoning from left to right and never goes back.
If the first step is a poor choice, the rest of the chain is spent on a problem that can no longer be solved.
Tree of Thoughts asks the model for several possible first steps, asks it again which of them still look workable, keeps the best few and repeats from there.
When a line of attack turns out to be hopeless, the search can return to an earlier step and try another branch, which a single chain cannot do.

The price is many more calls to the model, for the proposals and especially for the judgements.
The method pays off when judging is easier than solving, as in puzzles where a partial state can be checked with a little arithmetic or common sense.

## Setting

A problem has an input x and a final output y, and the reasoning between them is cut into steps called thoughts.
ToT frames a problem as a search over a tree, where each node is a state s = [x, z1...i] holding the input and the sequence of thoughts so far. {#c-tot-state}
An instantiation of ToT answers four questions: how to decompose the process into thought steps, how to generate potential thoughts from each state, how to heuristically evaluate states, and what search algorithm to use. {#c-tot-four}
How large a thought should be depends on the task: small enough for the model to produce several good alternatives, and large enough for the model to judge whether the partial solution is on track.

Tree of Thoughts generalizes chain-of-thought prompting and enables exploration over coherent units of text, thoughts, that serve as intermediate steps toward problem solving, letting the model consider multiple reasoning paths, self-evaluate choices, and look ahead or backtrack when necessary. {#c-tot-frame}
The search does not change the model's weights; it changes how the model's outputs are combined into one answer.

## The method

### Generating thoughts

Sampling i.i.d. thoughts from a CoT prompt works better when the thought space is rich, such as a paragraph per thought, while proposing thoughts sequentially in one propose prompt works better when the thought space is more constrained, such as a word or a line, because proposing in the same context avoids duplication. {#c-tot-gen}
In the Game of 24 a propose prompt lists possible next equations together with the numbers left after each; in creative writing the model samples whole writing plans independently.

### Evaluating states

The evaluator plays the part of the heuristic function of classical search, but it is written in natural language and computed by the model itself.
The state evaluator either values each state independently with a value prompt that outputs a scalar or a classification such as sure/likely/impossible, or votes across states by comparing them in a vote prompt; the valuations do not need to be perfect, only approximately helpful for decision making. {#c-tot-eval}
A value prompt can do a short lookahead, such as checking that 5, 5 and 14 reach 24 by addition, or use common sense, such as noting that 1, 2 and 3 are too small to reach 24.
Values are noisy, so they are sampled several times and averaged; votes suit tasks such as writing, where comparing two passages is easier than scoring one alone.

### Searching

Breadth-first search keeps the b most promising states per step and was used where the tree depth is limited (T ≤ 3) with b ≤ 5; depth-first search explores the most promising state first, prunes a subtree when the evaluator deems it impossible below a value threshold, and backtracks to the parent state. {#c-tot-search}
Breadth-first search with breadth b is beam search over thoughts, with the model's judgement in place of the sequence probability.
Depth-first search with a pruning threshold is a backtracking search that stops a branch as soon as the evaluator calls it impossible.
Yao et al. left A* search and Monte Carlo tree search for future work, and later variants added them.

### Choosing the answer

Breadth-first search returns the best state at the final depth and asks the model to write the output from it.
Depth-first search stops at a complete solution or after a step limit, and returns the deepest state it explored.
IO, CoT, CoT-SC and self-refinement can be seen as special cases of ToT, trees of limited depth and breadth, and no extra training is needed: a pre-trained LM is sufficient. {#c-tot-special}
A chain of thought is a tree of breadth one with no evaluator, and self-consistency is a set of independent chains with a vote at the leaves.

## Derivation

The full derivation is in `derivations/tree-search-arithmetic.md`, and `tree-search-arithmetic.check.py` checks every step, including a brute-force count of the Game of 24 tree.

**The size of the tree.**
A Game of 24 state with n numbers has C(n, 2) · 6 moves, so a full tree from four numbers has 36 · 18 · 6 = 3888 leaves, slightly fewer for inputs where a move produces zero. {#c-d-size}

**Chains against a tree.**
Under a step-wise model where each sampled thought keeps the problem solvable with probability q, n chains succeed with probability 1 − (1 − q^T)^n and a breadth-one tree with k proposals and a perfect evaluator succeeds with (1 − (1 − q)^k)^T; at q = 0.4, T = 3 and fifteen thoughts each, five chains give 0.2816 and the tree 0.7844. {#c-d-chains}
Both spend fifteen generated thoughts; the tree wins by resampling at the step that failed instead of restarting the whole chain.

**Evaluator errors.**
An evaluator that discards the only good state with probability δ per step keeps it through T steps with probability (1 − δ)^T, which at δ = 0.1 is 0.729 for T = 3 and 0.3487 for T = 10. {#c-d-prune}
This is why depth-first search with backtracking, or a wider beam, matters for deeper problems such as crosswords.

**Model calls.**
Breadth-first search with depth T, breadth b, k proposals and v evaluation samples expands 1 + b(T − 1) states, produces k times as many candidates and makes v · k · (1 + b(T − 1)) evaluation calls: 11, 55 and 165 for T = 3, b = 5, k = 5, v = 3. {#c-d-calls}

**Upper confidence bounds.**
Monte Carlo tree search variants pick which branch to try next with UCT, the tree version of the UCB1 rule for multi-armed bandits.
With w = 1 and 10 parent visits, UCT scores a child with mean value 0.6 and 5 visits at 1.2786 and a child with mean value 0.2 and 1 visit at 1.7174, so the less visited child is selected; an unvisited child scores infinity. {#c-d-uct}

**The self-evaluation score.**
With the score C^λ · P^(1 − λ) at λ = 0.5, a step with generation probability 0.9 and judged correctness 0.2 scores 0.4243, below a step with 0.5 and 0.9 that scores 0.6708; at λ = 0 the score is the generation probability alone. {#c-d-xie}

**Tokens.**
At 67 completion tokens per chain-of-thought sample, the 5.5k completion tokens of a tree search buy about 82 samples, and the cost per solved problem is $1.00 for the tree and $0.96 for the best of 100 chains. {#c-d-tokens}
The best of 100 chains needs an oracle that knows which chain is right, while the tree picks its answer with the model's own evaluator.

## Worked example

Take the input 4 9 10 13.
From four numbers there are 36 possible first equations, and exact enumeration finds that only 7 of them leave numbers from which 24 can still be made, one of them 13 − 9 = 4, which leaves 4 4 10.
A model that picks a first step at random therefore keeps the problem solvable less than one time in five.
The step 4 + 9 = 13 leaves 10 13 13, from which no sequence of operations reaches 24, and a chain that starts with it fails however well it continues.

Tree of Thoughts asks the model for five candidate first steps, say 4 + 9 = 13, 13 − 9 = 4, 10 − 4 = 6, 4 × 9 = 36 and 9 + 10 = 19.
The evaluator looks at the numbers left: 4 4 10 is sure, since (10 − 4) × 4 = 24; 6 9 13 is sure, since (13 − 9) × 6 = 24; 4 13 19 is sure, since (19 − 13) × 4 = 24; 10 13 13 and 10 13 36 cannot reach 24 and should be judged impossible.
The search keeps the best states and expands them the same way, and two more steps reach (10 − 4) × (13 − 9) = 24.

The NumPy implementation simulates this with a stand-in model that proposes random legal steps and values states correctly except with a set error rate.
On 60 random problems, a single random chain almost never succeeds, while breadth-first search with ten proposals per step solves most of them, and adding evaluator noise or removing evaluation samples lowers the success rate, as the derivation predicts.
The PyTorch implementation trains a small network to predict whether a state can still reach 24 and uses it as the evaluator in beam search; with breadth 5 the learned evaluator solves far more problems than a random ranking of the same candidates.

## Hyperparameters

The thought size is the most important choice: thoughts that are too small give the evaluator nothing to judge, and thoughts that are too large leave the generator with too few alternatives.
Breadth and proposals trade calls for success; in the Game of 24, raising the breadth from 1 to 5 raised the success rate from 45% to 74%.
More evaluation samples reduce the noise of the evaluator, which section 3 of the derivation shows compounds with depth.
Depth-first search needs a step limit and a pruning rule; a rule that prunes too eagerly removes correct branches, and one that never prunes wastes the step budget.
For Monte Carlo tree search the exploration weight and the number of iterations play the role of the breadth.

## Cost

There is no training cost in the original method.
At inference, every expanded state costs a generation call and every candidate costs one or more evaluation calls, so the evaluation calls usually dominate.
In the Game of 24, ToT requires 5.5k completion tokens per problem, close to 100 CoT trials (6.7k tokens), at $0.74 per case against $0.47 for the best of 100 CoT samples, and ToT could require 5-100 times more generated tokens than CoT in general. {#c-tot-cost}
Calls within one depth are independent and can run in parallel, but the depths are sequential, so latency grows with the depth of the tree.

## Evidence

Yao et al. tested three tasks built to need search: the Game of 24, creative writing and mini crosswords, all with GPT-4.
For the Game of 24, Yao et al. scraped 1,362 games from 4nums.com, sorted by human solving time, and tested on the relatively hard games indexed 901 to 1,000; success requires a valid equation that equals 24 and uses each input number exactly once. {#c-tot-24-setup}
In the Game of 24, ToT runs breadth-first search keeping the best b = 5 candidates per step, prompting the model to evaluate each thought candidate as sure/maybe/impossible with regard to reaching 24, and samples values 3 times for each thought. {#c-tot-24-bfs}
In the Game of 24, IO, CoT and CoT-SC prompting achieve only 7.3%, 4.0% and 9.0% success rates, while ToT with a breadth of b = 1 achieves 45% and b = 5 achieves 74%. {#c-tot-24}
Best of 100 CoT samples achieve a success rate of 49%, still much worse than exploring more nodes in ToT, and around 60% of CoT samples already failed the task after generating the first step. {#c-tot-best100}
In creative writing, average GPT-4 coherency scores across 100 tasks are 7.56 for ToT against 6.19 for IO and 6.93 for CoT, and humans prefer ToT over CoT in 41 out of 100 passage pairs and CoT over ToT in 21. {#c-tot-writing}
The mini crosswords task uses 20 test games of 5 × 5 mini crosswords from GooBix, where the input describes the 5 horizontal clues and 5 vertical clues and the output is a board of 25 letters. {#c-tot-cross-setup}
In mini crosswords, IO and CoT prompting have a word-level success rate less than 16%, while ToT with depth-first search limited to 100 steps achieves a word-level success rate of 60% and solves 4 out of 20 games, and 7/20 when outputting from the oracle best DFS state. {#c-tot-cross}
Ablating the pruning makes performance generally worse, and an ablation without backtracking, similar to a greedy breadth-one search, performs poorly with a word-level success of only 20%. {#c-tot-ablate}
With GPT-3.5, ToT reaches 19% on the Game of 24 against 74% with GPT-4; GPT-4 generation with GPT-3.5 evaluation reaches 64% and GPT-3.5 generation with GPT-4 evaluation reaches 31%, suggesting the bottleneck is thought generation. {#c-tot-gen-bottleneck}

Related search methods reported gains on standard reasoning benchmarks.
Self-evaluation guided stochastic beam search surpasses the corresponding Codex-backboned baselines in few-shot accuracy by 6.34%, 9.56% and 5.46% on GSM8K, AQuA and StrategyQA. {#c-xie-gain}
With Llama-2, self-evaluation guided search achieves 46.1% on GSM8K with a cost of 12.6k tokens, compared with 41.8% for self-consistency at 13.9k tokens. {#c-xie-cost}
RAP achieves an average success rate of 64% on Blocksworld plan generation while CoT fails almost completely, and LLaMA-33B with RAP surpasses GPT-4 with CoT by 33% relative improvement. {#c-rap-result}
On GSM8K, RAP answers 48.8% of the problems correctly with 20 iterations of MCTS, outperforming chain-of-thought and least-to-most prompting with self-consistency. {#c-rap-gsm}
LATS reaches 92.7% accuracy for programming on HumanEval with GPT-4 and an average score of 75.9 on WebShop with GPT-3.5, comparable to gradient-based fine-tuning. {#c-lats-result}

## Limitations and failure modes

Deliberate search such as ToT might not be necessary for many tasks that GPT-4 already excels at, and search methods like ToT require more resources than sampling methods in order to improve task performances. {#c-tot-limits}
The three tasks of the original paper were designed to need search; on benchmarks where a chain of thought already succeeds, the extra calls buy little.

The evaluator is the weak point.
When a crossword is actually solved, the state evaluator might still deem some words impossible and prune, because 5 × 5 crosswords have some rare or obsolete words that GPT-4 cannot recognize; without pruning the search found the correct solution for 4/20 games, 3 of which ToT with pruning cannot solve within 100 steps. {#c-tot-prune-wrong}
A self-evaluator shares the blind spots of the generator, so an error the model cannot see when writing it may also pass when judged, and the derivation shows how such errors compound over depth.
A trained process reward model can replace the self-evaluation, at the cost of training data and of a new way to fail, since search can exploit a learned scorer's mistakes.

LATS has a higher computational cost than simpler prompting methods like ReAct or Reflexion, and it assumes the ability to revert to earlier states in decision-making environments, which may not be universally applicable. {#c-lats-limits}
The original method also needs a task-specific decomposition and prompts for proposing and judging, written by hand for each task.

## Variants

**Self-evaluation guided beam search.**
Xie et al. combine the language model probability with a correctness confidence C in [0, 1] that the model assigns to each reasoning step, and run stochastic beam search that samples k beams without replacement from a distribution proportional to exp(E/τ), which becomes vanilla beam search as τ → 0. {#c-xie-method}
Xie et al. set λ = 0.5 for all datasets; setting λ to 1 (E = C) or 0 (E = P) decreases GSM8K performance from 80.2% to 74.5% and 77.1%, respectively. {#c-xie-lambda}
Equation 3 of the paper writes the exponents the other way round from its text and appendix, and at λ = 0.5 the two forms agree.

**Reasoning via Planning (RAP).**
Reasoning via Planning (RAP) repurposes the LLM as both a world model and a reasoning agent and incorporates a planning algorithm based on Monte Carlo Tree Search for strategic exploration in the reasoning space. {#c-rap-method}
The model predicts the next state after an action, so the search can roll out plans without an external simulator.

**Language Agent Tree Search (LATS).**
LATS integrates Monte Carlo Tree Search to enable LMs as agents, along with LM-powered value functions and self-reflections, and incorporates an environment for external feedback. {#c-lats-method}
LATS selects nodes with the upper confidence bound UCT(s) = V(s) + w·sqrt(ln N(p)/N(s)), where N(s) is the number of visits to node s and p its parent. {#c-lats-uct}
UCT applies the UCB1 bandit rule at every node of the tree.

**Graph of Thoughts.**
Graph of Thoughts models an LLM thought as a vertex and a dependency between thoughts as an edge, so arbitrary thoughts can be aggregated by constructing vertices that have more than one incoming edge. {#c-got-method}
On sorting, Graph of Thoughts improves upon CoT and ToT by about 70% and 62% in the quality of sorting while reducing costs by more than 31% over ToT. {#c-got-result}

**Stream of Search.**
Stream of Search pretrains a language model on streams of search generated by heuristic solvers for the game of Countdown, which increases search accuracy by 25% over models trained to predict only the optimal search trajectory; finetuned with APA and STaR, the models solve 36% of previously unsolved problems. {#c-sos-result}
Instead of an outer search program, the model itself writes the search, with its dead ends and backtracking, as one long sequence.

**Informed search.**
A* search orders states by the cost so far plus a heuristic estimate of the cost to go, and a model-written evaluator can supply the estimate; Yao et al. list A* among the algorithms that could be plugged in.

## When to use it

Use it when a problem has natural steps, when early mistakes are fatal and cannot be repaired later in a chain, and when the model can judge partial solutions, by short lookahead or by an external check.
Puzzles, planning, code with tests and constrained writing fit these conditions.
For questions that one chain usually answers, self-consistency or a single chain is cheaper.
Test-time scaling studies find that the benefit of search over plain sampling depends on difficulty and budget, so the choice should be measured against sampling at the same number of tokens.

## Pitfalls

Comparing a search with hundreds of calls against a single chain overstates its value; compare at matched tokens or cost, and against the best of many samples where an oracle check is available.
An evaluator that scores whole answers rather than partial states cannot prune early, which removes most of the benefit.
Reporting the oracle best state instead of the state the search would return measures what the tree contains, not what the method outputs.
Prompts and decompositions tuned on the test tasks make results hard to transfer.

## History

Search with a heuristic is among the oldest ideas of artificial intelligence, from the state-space search of Newell and Simon's problem solvers to A* search and, for games, Monte Carlo tree search guided by learned values.
In 2022, chain-of-thought prompting and self-consistency made multi-step reasoning in language models a sampling problem.
In May 2023 three groups independently put search over reasoning steps on top of language models: Yao et al. named Tree of Thoughts, Xie et al. posted self-evaluation guided beam search, and Hao et al. posted RAP with Monte Carlo tree search.
Graph of Thoughts generalised the tree to a graph in August 2023, LATS added environment feedback for agents in October 2023, and Stream of Search trained models on serialized search in April 2024.
Later work replaced self-evaluation with trained process reward models and folded search into training, through reasoning models that learn to backtrack inside a single long chain of thought.

## References
