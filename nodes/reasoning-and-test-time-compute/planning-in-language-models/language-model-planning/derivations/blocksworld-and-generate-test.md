# The arithmetic of Blocksworld and generate-test planning

This derivation backs the mathematical claims of the language-model planning page. The check file
`blocksworld-and-generate-test.check.py` tests every statement, the state counts and plan lengths by enumerating
the state space and running breadth-first search.

Blocksworld has n labelled blocks, a table and one hand. The four actions are pick up a clear block from the table,
unstack a clear block from another block, put the held block down on the table, and stack the held block on a clear
block. A state is a set of towers plus what the hand holds.

## 1. How many states there are

With the hand empty, a state splits the n blocks into towers and orders each tower from bottom to top. The towers
form an unordered set, and the number of ways to split n labelled items into k non-empty ordered lists is the Lah
number L(n, k) = C(n − 1, k − 1) · n! / k!. Summing over k gives the number a(n) of hand-empty states, which
satisfies

a(n) = (2n − 1) · a(n − 1) − (n − 1)(n − 2) · a(n − 2), with a(0) = a(1) = 1.

So a(3) = 13, a(4) = 73 and a(5) = 501. A state with a block in the hand is a held block together with a hand-empty
arrangement of the other n − 1, so there are n · a(n − 1) such states. The totals are 13 + 3 · 3 = 22,
73 + 4 · 13 = 125 and 501 + 5 · 73 = 866 for n = 3, 4 and 5. Breadth-first search reaches every one of them from any
state, so the space is connected.

These numbers are small. A classical planner can search all 866 states of a five-block problem at once, which is why
Fast Downward solves the PlanBench test set in a fraction of a second per instance. The count grows faster than
exponentially: a(20) ≈ 3.277 · 10^20, so larger problems need a heuristic rather than blind search, but the
heuristics used in classical planning handle them.

## 2. How long a plan has to be

Let u be the number of blocks that start on another block and g the number that end on another block. One plan
always works: unstack every tower onto the table, which takes an unstack and a put-down for each of the u blocks
(2u actions), then build every goal tower from the bottom, which takes a pick-up and a stack for each of the g blocks
(2g actions). So every problem has a plan of at most 2u + 2g actions. Since at most n − 1 blocks can sit on other blocks,

optimal plan length ≤ 2u + 2g ≤ 4(n − 1).

For n = 3, 4 and 5 breadth-first search over all pairs of hand-empty states finds longest optimal plans of 8, 12 and
16 actions, so the bound is met exactly. The PlanBench test set of three to five blocks has plans of 2 to 16 steps,
which fits: 16 = 4 · (5 − 1).

A table-to-stack problem starts with every block on the table and asks for one tower of all n blocks. Here u = 0 and
g = n − 1, and no shorter plan exists, because each of the n − 1 blocks above the bottom one has to be picked up and
stacked at least once. The optimal plan has exactly 2(n − 1) actions. Four blocks need 6 actions and twenty blocks need 38. The procedure is the same at every size, which
is what makes these problems a clean test of whether a model has learned the procedure or the examples.

## 3. Generate and test with a sound verifier

Suppose each attempt of a plan generator is correct with probability p, attempts are independent, and a sound
verifier accepts exactly the correct plans. The loop stops at the first accepted plan, so with K attempts

P(success) = 1 − (1 − p)^K,

and every plan it returns is correct. With p = 0.4 and K = 15 the formula gives 0.99953. Valmeekam et al. measured
p = 0.40 for GPT-4 on 100 Blocksworld problems and 0.88 for the loop with a sound verifier and up to 15 rounds. The
gap means the attempts are far from independent: a problem the model gets wrong once it tends to get wrong again,
even with the verifier's feedback. The formula is an upper bound for a fixed p, and the measured rate says how
correlated the retries are.

## 4. Generate and test with an unsound verifier

Now let the verifier accept a correct plan with probability t, its true positive rate, and an incorrect plan with
probability f, its false positive rate. One round then accepts a correct plan with probability p t, accepts an
incorrect one with probability (1 − p) f, and continues with probability r = 1 − p t − (1 − p) f. If rounds are
independent, the probability that the loop returns a correct plan within K rounds is

P(correct) = p t / (p t + (1 − p) f) · (1 − r^K).

The first factor is the precision of the verifier on this generator, the share of accepted plans that are correct.
With p = 0.4 and the rates Valmeekam et al. measured for GPT-4 checking GPT-4 plans, t = 0.982 and f = 0.8445, the
precision is 0.3928 / (0.3928 + 0.5067) = 0.4367 and r = 0.1005, so the loop nearly always stops in the first round
and P(correct) is 0.4367 for any K of two or more rounds. An unsound verifier that accepts most wrong plans cannot
lift the loop much above the generator, however many rounds are allowed. A sound verifier has f = 0 and precision 1.

## 5. Time against a classical planner

On the 600 Blocksworld problems o1-preview took 40.43 seconds per problem on average and Fast Downward 0.265 seconds,
a ratio of 40.43 / 0.265 = 152.6. On Mystery Blocksworld the o1-preview time of 83.37 seconds is 314.6 times the
planner's.
