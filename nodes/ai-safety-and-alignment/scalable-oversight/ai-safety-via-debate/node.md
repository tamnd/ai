---
tag: N45PC9
kind: method
slug: ai-safety-via-debate
name: AI safety via debate
aliases: [debate, AI debate, the debate game, debate for scalable oversight]
area: ai-safety-and-alignment/scalable-oversight
level: core
summary: >-
  Two agents argue for competing answers in a zero-sum game and a judge picks the winner, so that a judge who cannot check an answer directly can still supervise agents stronger than itself.
purpose: supervising AI systems on questions too hard for a person to judge directly, by training two agents to argue and a person to judge the argument
uses: [zero-sum-game@al, nash-equilibrium@al, minimax@al, monte-carlo-tree-search@al]
related_to: [adversarial-training@al, perfect-information-game@al, game-tree@al]
hyperparameters:
  - {name: rounds and statement length, meaning: "how many statements each debater makes and how long each one may be", typical: "short statements; 750 characters per turn, of which up to 250 are quotes, in Michael et al. (2023)"}
  - {name: answer assignment, meaning: "whether the debaters choose the answers they defend or are assigned one correct and one incorrect answer", typical: "chosen in Irving et al. (2018); assigned in the experiments of Michael et al. and Khan et al."}
  - {name: precommit, meaning: "whether each debater states its answer before the argument and cannot change it", typical: "on; it was crucial in the MNIST experiment"}
  - {name: judge, meaning: "who decides the winner, and by what rule", typical: "a person, or an LLM judge that sees both orders of the debaters"}
  - {name: turn penalty, meaning: "the cost to the judge of asking for one more round", typical: "0.05 bits per turn in Michael et al."}
  - {name: optimisation for persuasiveness, meaning: "how hard the debaters are optimised to win the judge's approval", typical: "best-of-16 sampling with a preference model in Khan et al. (2024)"}
cost:
  train: "self-play of two debaters, often the same weights, through whole debates, with one judgement per debate; 10k Monte Carlo tree search rollouts per move in the MNIST experiment; the time of human judges when they judge"
  inference: "at test time the debate can stop after the two answers, so no more than the answer itself; a full debate per question when it is run at test time"
history:
  year: 2018
  people: [Geoffrey Irving, Paul Christiano, Dario Amodei]
  note: "Irving, Christiano and Amodei of OpenAI posted AI safety via debate as arXiv 1805.00899 in 2018. Human and LLM experiments on reading comprehension by Michael et al. (2023) and Khan et al. (ICML 2024) and the doubly-efficient protocols of Brown-Cohen, Irving and Piliouras (2023) followed."
prereqs: [zero-sum-game@al, nash-equilibrium@al, minimax@al]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1805.00899", "2311.08702", "2402.06782", "2311.14125"]
  al: null
sources:
  - {paper: "Irving et al. (2018)", title: "AI safety via debate", locator: "pp. 1, 3, 5, 6, 9 to 11, 14, 15 and 17 to 20", note: "the debate game, the complexity argument, the MNIST experiment and the open problems"}
  - {paper: "Michael et al. (2023)", title: "Debate Helps Supervise Unreliable Experts", locator: "pp. 1, 4 to 11, 13 and 14", note: "human debate against consultancy on QuALITY, error analysis, obfuscated arguments and limitations"}
  - {paper: "Khan et al. (2024)", title: "Debating with More Persuasive LLMs Leads to More Truthful Answers", locator: "pp. 1, 2 and 4 to 8, the title page", note: "LLM debaters and judges, persuasiveness, judge biases and limitations"}
  - {paper: "Brown-Cohen et al. (2023)", title: "Scalable AI Safety via Doubly-Efficient Debate", locator: "pp. 1, 2, 9 and 10", note: "debate protocols with efficient honest strategies, and the obfuscated argument problem"}
---

AI safety via debate trains two agents to argue for competing answers to a question while a judge, usually a person, decides who argued better.
It targets tasks where a person cannot check an answer directly but can follow an argument about it.
Irving, Christiano and Amodei (2018) proposed it on the bet that a lie is harder to defend than to expose, so that winning debates would reward honesty.

## In plain words

Suppose an AI gives an answer you cannot check yourself.
Debate adds a second AI whose only goal is to beat the first in front of you.
Each states an answer, then they take turns making short points, and you pick the winner.
A false claim gives the opponent something to attack, so if exposing a lie is easier than defending one, the side telling the truth should win.
You never have to solve the whole problem: the debaters narrow the disagreement until it rests on one small step you can check.
In theory this lets a weak judge supervise far stronger agents.
In practice it depends on people judging well, on the honest side always being able to find the flaw, and on training reaching good play.
Experiments with people and language models on reading comprehension found that debate helped judges more than hearing from a single expert.

## Setting

Training a system from human judgements of its behavior breaks down once the task is too hard for a person to judge. {#c-motivation}
Debate keeps a person in the loop but changes what they judge: an argument between two agents, not the answer alone.
The agents are trained by self-play on a zero-sum game: given a question or a proposed action, they take turns making short statements up to a limit, and a person then decides which agent gave the most true and useful information. {#c-def}
The agents themselves are unrestricted; only the training signal changes.

The game in Irving et al. has five steps: both agents see the question, each states an answer (the two may agree), they alternate statements, the judge sees the transcript and picks a winner, and each agent tries to maximize its chance of winning. {#c-game-steps}
At test time the game can stop once the two answers are stated, because the debate is a training device, though agents could run debates internally to sharpen their answers. {#c-test-time}

The method rests on one claim, that in the debate game lying is harder than refuting a lie; Irving et al. treat it as an empirical question in each setting. {#c-central-claim}
If it holds, they hope for three more things: that both agents try to tell the truth convincingly in every Nash equilibrium of the game, that self-play or another form of gradient descent can find approximate equilibria, and that agents at equilibrium are about as capable as agents trained without any safety measure. {#c-hoped-claims}
For questions too large to show a judge, such as a gigapixel image or the whole web, the agents reveal small pieces of it in their statements. {#c-large-queries}

## The method

**Short debates and a single path.**
A debate does not need to cover a whole argument.
Long arguments are long mostly because they branch, and an argument laid out as a sequence can be folded into a shallow tree: state the result of the first half, then let the opponent choose which half to examine. {#c-unbranched}
Each round the challenger picks the branch it believes is wrong, so the transcript follows one path down the tree, and the judge checks only the last step.

**The complexity-theory analogy.**
Irving et al. replace the human judge with a polynomial-time algorithm $H$ and give the agents unlimited computation.
Asking $H$ for answers directly gives the class P, which they compare to supervised learning; using $H$ only to check an answer supplied by the model gives NP, which they compare to reinforcement learning with one agent. {#c-p-np}
With $n$ alternating rounds, Alice and Bob produce a formula $\exists x_0 \forall x_1 \cdots H(q, x_0, x_1, \ldots)$; a fixed $n$ reaches the level $\Sigma_n \mathrm{P}$ of the polynomial hierarchy, and a number of rounds polynomial in the question reaches PSPACE. {#c-hierarchy}
Their Theorem 1 states that for every problem in PSPACE there is a polynomial-time judge for which optimal play of debate with polynomially many steps solves the problem. {#c-theorem-pspace}
So, by the analogy, debate with optimal play answers PSPACE questions where direct judging answers only NP. {#c-direct-np}
They call these arguments analogies only: no tractable learning algorithm is expected to reach all of PSPACE, and the point is that the supervision signal need not be the bottleneck, the capacity and training of the models can be. {#c-analogies-only}

**Bisection.**
Two of their examples make the analogy concrete.
To count the primes below $x$, Alice states the count for the whole range and for its lower half; a false total forces a false half, Bob names it, and the halving repeats until the judge tests one number for primality, having done only additions and subtractions on the way, so Alice's best opening is the truth. {#c-prime-example}
The same scheme checks a long deterministic simulation: Alice states the state after $n$ and after $n/2$ steps, Bob names the half he disputes, and they bisect until they disagree about a single step, which the judge can compute. {#c-simulation}

**The human-judge protocol of Michael et al.**
In their experiments with people, debaters are randomly assigned to the two answers, open with simultaneous statements, and then alternate with the judge until the judge stops the debate; part of each turn can be certified quotes from the hidden text, which the judge knows are verbatim. {#c-michael-protocol}
Each turn holds at most 750 characters, of which at most 250 are quotes. {#c-michael-limits-chars}
The judge's score is

$$ S_j = \log_2 p^*_{j,t} - 0.05\, t , $$ {#eq-judge-score}

as given by {eq:michael-2023-debate-helps-supervise#1}, where $p^*_{j,t}$ is the probability the judge puts on the correct answer at the end and $t$ the number of times the judge asked to continue.
A debater scores the base-2 logarithm of the probability the judge gives to its answer, with no length penalty; the judge's rule is strictly proper, so reporting true beliefs maximizes the expected score. {#c-michael-score}
Unlike the original proposal, Michael et al. assign the answers, let the judge question the debaters between turns, and ask the judge for probabilities instead of an automatic loss for a debater caught lying. {#c-michael-differences}

**The baseline: consultancy.**
To isolate what the opponent adds, Michael et al. compare debate with consultancy: one expert argues for one of the two answers, the wrong one half of the time, which they frame as a worst case of RLHF. {#c-michael-consultancy}
Khan et al. use the same contrast with language models: experts that can read the text argue for different answers with a quote tool that verifies their quotes, and a judge that cannot read the text chooses. {#c-khan-setup}

## Derivation

**Honesty wins the bisection debate.**
Write $C(a, b)$ for the number of primes in $[a, b)$, and let the judge insist that the two half counts add up to the count being defended.
On a range of $2^n$ numbers, Alice can win whatever Bob does exactly when her claimed count is $C(a, b)$; any other claim loses to the Bob who always challenges a half with a wrong count. {#c-bisect-count}
The step is one line: $C(a, b) = C(a, m) + C(m, b)$, so if the total is wrong, at least one half is wrong, and by induction the wrong range shrinks to one number that the judge tests.
Every play takes exactly $n = \log_2 (b - a)$ rounds, and the judge does $n$ subtractions and one primality test. {#c-bisect-rounds}
For a deterministic step function, the bisection over $2^k$ simulation steps behaves the same way: Alice wins against every Bob exactly when her final state is the true one, after $k$ rounds and one checked step. {#c-bisect-sim}

**The judge's score rewards honest probabilities.**
For a judge whose belief in an answer is $q$ with $0 < q < 1$, the expected score $q \log_2 p + (1 - q) \log_2 (1 - p)$ is strictly concave in the reported $p$ and largest at $p = q$, and the turn penalty of eq-judge-score does not depend on $p$, so it does not move the maximum. {#c-log-score-proper}
The derivation `bisection-debate` gives the steps, and its check solves both debates exactly by minimax on small instances.

## Worked example

Count the primes in $[0, 16)$: they are 2, 3, 5, 7, 11 and 13, so $C(0, 16) = 6$.
Alice lies and claims 7.
In each round she splits her current claim into a left and a right half, and Bob challenges a half whose count is wrong.

| round | range | Alice's claim | her split, left + right | true counts | Bob challenges |
|---|---|---|---|---|---|
| 1 | $[0, 16)$ | 7 | 4 + 3 | 4 + 2 | $[8, 16)$, claimed 3 |
| 2 | $[8, 16)$ | 3 | 1 + 2 | 1 + 1 | $[12, 16)$, claimed 2 |
| 3 | $[12, 16)$ | 2 | 1 + 1 | 1 + 0 | $[14, 16)$, claimed 1 |
| 4 | $[14, 16)$ | 1 | 0 + 1 | 0 + 0 | $[15, 16)$, claimed 1 |

The judge tests 15, which is not prime, so the claim of 1 is false and Alice loses after $4 = \log_2 16$ rounds, four subtractions and one primality test.
Had she claimed 6 and split truthfully, every half would be true and Bob would have nothing to win with.

For the judge's score of eq-judge-score: a judge who ends at once with 50% on each answer scores $\log_2 0.5 = -1$; a judge who continues twice and ends with 90% on the correct answer scores $\log_2 0.9 - 0.10 \approx -0.152 - 0.100 = -0.252$; a judge who continues once and ends with 60% scores $\log_2 0.6 - 0.05 \approx -0.787$.
The check `bisection-debate.check.py` verifies these numbers.

## Hyperparameters

**The game.**
The number of rounds and the length of a statement bound how much a judge must read.
Precommitting to an answer limits the liar: in the MNIST experiment the liar did much better when its lie could change between branches of the game tree. {#c-mnist-precommit}
Who moves first matters; there the second player to move had an advantage, since it can steal any winning strategy of the first player by ignoring the previous move. {#c-mnist-second}

**The judge.**
Michael et al. cap each turn at 750 characters with 250 of quotes and charge the judge 0.05 bits per extra turn, as in eq-judge-score.
Khan et al. run each LLM judgement twice with the debaters swapped and impose strict word limits, against position and verbosity biases. {#c-khan-judge-bias}

**The debaters.**
Khan et al. make debaters more persuasive by best-of-$N$ sampling and critique-and-refinement, scored by a preference model that, like the judge, cannot see the text.

## Cost

Training plays whole debates between two debaters, which are often the same weights, and needs one judgement per debate; with human judges the cost is mostly their time.
The MNIST experiment needed no training of the debaters: it played the game with plain Monte Carlo tree search, 10k rollouts per move, the PUCT rule with $c_{\text{puct}} = 1$, and nine games per image to model each possible precommitted lie. {#c-mnist-mcts}
Debates can be cheaper for the judge than a single expert: human debates ended after 2.7 rounds on average against 4 for consultancy, and showed the judge only 3.5% of the story on average, against 5.8% in consultancy. {#c-michael-efficiency}
At test time a trained debater can simply give its answer.

## Evidence

**MNIST.**
In the MNIST game, a fixed classifier trained to label digits from 6 or 4 revealed pixels is the judge; two players precommit to labels and reveal one pixel per turn, one of them told to be honest and one to lie, and the label with the higher logit wins. {#c-mnist-setup}
With precommit, debate lifted a judge that is 59.4% accurate on random 6-pixel masks to 88.9%, and a 48.2% judge on 4 pixels to 85.2%. {#c-mnist-result}
Lies that worked clustered on the digits 8 and 9, likely because only nonzero pixels can be shown, while the judge's own errors on random masks were spread out. {#c-mnist-lies}
Because the search ran on the test set with full access to the judge, the experiment models debaters with no generalization error. {#c-mnist-no-generalization}
A cat-or-dog version for human players suggested informally that lying was very hard, and formal human experiments were left open. {#c-cat-dog}

**Human debaters on reading comprehension.**
In Michael et al., judges who had not read a story saw arguments and short quotes from debaters who had, one arguing for the right answer and one for a wrong one. {#c-michael-setup}
Human debate reached 84% judge accuracy against 74% for consultancy, a significant gap, and debates were 68% as long. {#c-michael-result}
With GPT-4 as debater, AI debate and AI consultancy reached 78% and 80%, between the two human settings. {#c-michael-ai}
The AI debaters gave less than half the quote density of human ones and often repeated quotes. {#c-michael-ai-evidence}
Dishonest human consultants misled judges 40% of the time, while judges did better against dishonest AI consultants (87%) than honest ones (74%). {#c-michael-deceive}
More skilled debaters raised debate's accuracy and lowered consultancy's. {#c-michael-skill-trend}
Judges were best calibrated in human debate but overconfident above 90% in every setting. {#c-michael-calibration}

**LLM debaters.**
On QuALITY, Khan et al. found that debate raised accuracy to 76% for non-expert LLM judges and 88% for human judges, against naive baselines of 48% and 60%. {#c-khan-result}
Consultancy gave 78% for human judges and 54% for LLM judges. {#c-khan-consultancy}
Debaters were more persuasive when assigned the correct answer, and more so for stronger debaters, which the authors read as truth being easier to argue than a lie on this task. {#c-khan-truth-easier}
Optimising debaters for the judge's approval raised judge accuracy in self-play, while optimising consultants the same way lowered it. {#c-khan-persuasive}
The authors call this weak evidence that debate under optimal play yields truthful information. {#c-khan-weak-evidence}
Letting human judges question the debaters did not change accuracy. {#c-khan-interaction}
Dropping static-debate judgements below 75% confidence kept 65% of questions at 94% accuracy, and debate beat consultancy at every rejection threshold. {#c-khan-calibration}

**Efficient honest strategies.**
Brown-Cohen et al. point out that the original protocol needs the honest debater to simulate a deterministic system for exponentially many steps, and give protocols in which the honest side succeeds with polynomially many steps and can verify stochastic systems even against a dishonest side with exponentially many. {#c-bc-protocols}
In their model two polynomial-time provers face a much cheaper verifier, and under their assumptions any polynomial-time computation can be checked with a constant number of queries to the human-judgement oracle. {#c-bc-constant}

## Limitations and failure modes

**Obfuscated arguments.**
Brown-Cohen et al. name the obstacle to letting both debaters be unbounded the obfuscated argument problem, after Barnes and Christiano (2020): a dishonest debater can build in polynomial time an argument that the honest one would need exponential time to refute. {#c-obfuscated-bc}
Michael et al. describe it as a long argument of many steps that both debaters know is wrong but in which neither can say which step fails; they did not see it matter in their reading-comprehension setting. {#c-obfuscated-michael}
Their remedy is to collect probabilities so that the judge can stay uncertain, and in the worst case to award a debate only to positive proof, at the cost of some of the power of the original scheme. {#c-fail-gracefully}
A related limit is the stalemate, where both sides' best arguments are too complex for the judge to follow. {#c-stalemate}
The doubly-efficient protocols avoid the problem only for computations that a careful enough human could have checked step by step. {#c-bc-requirement}

**Honesty need not be the best strategy.**
Irving et al. ask whether people can check logical statements that use words they do not understand, and whether a debater can slip in a flawed step that looks valid without the opponent catching it. {#c-honesty-best}
Optimal play is out of reach in practice, and they call the hope that near-optimal agents stay honest far from a solid argument. {#c-approx-optimal}
They know of no theory that self-play converges stably, and sketch a bad cycle: both agents become honest, forget how to expose lies, and one returns to lying and wins for a while. {#c-no-convergence}
Michael et al. note that dishonest strategies may exist that break debate at equilibrium and that their human debaters never found, so part of their accuracy may come from the liars' mistakes. {#c-michael-nash}
Brown-Cohen et al. leave open what debate can verify when the human-judgement oracle is wrong on some queries, as reward models trained on human labels are. {#c-bc-fallible}
The bisection protocol fails for stochastic simulation: a dishonest debater can pick the noise so that each step looks plausible while the end result is very unlikely, and a natural protocol for this case is open. {#c-stochastic}
Debate also assumes evenly matched debaters, since a much stronger liar may lie in ways the other cannot refute; self-play with shared weights is the proposed fix. {#c-matched}

**Judge bias.**
Irving et al. worry about belief bias and expect that, above some level of judge skill, debate is limited only by model capacity; they suggest majority votes, weighting better judges, instructions and judge training. {#c-belief-bias}
The human experiments do not test this: Michael et al.'s judges started without strong opinions on the questions, so robustness to prior views and cognitive biases was not measured. {#c-michael-no-bias-test}
LLM judges showed strong position and verbosity biases in Khan et al., as above.
Human judges in Michael et al. often stopped too early, deciding after one round in a quarter of the debates they got wrong. {#c-michael-early}
That leaves no time to refute a lie, and the authors suggest requiring one more round after the first and after any change of mind. {#c-michael-swing}

**Narrow evidence.**
The experiments create the expertise gap by hiding a text from the judge: Michael et al. say it is unclear how their result carries over to larger knowledge gaps or to differences in reasoning and persuasion. {#c-michael-limits}
Khan et al. add that their debaters were RLHF models that lean toward honesty, and that their results depend on verified quotes, without which a liar could invent a story. {#c-khan-limits}

**Training to persuade.**
Training for maximal persuasion resembles training for deception, but rewarding both agents for honesty would let them collude on consistent lies, because debate relies on each side attacking the other. {#c-convincing-dangerous}
A strong misaligned debater might also talk a person into a harmful act; short statements make that harder and let the opponent expose the attempt. {#c-mind-hack}

## Variants

- **Symmetric games.** Against first-mover advantage: simultaneous moves, two debates with each agent moving first, or a komi as in Go if judges can give graded scores. {#c-variants-symmetric}
- **Shared activations.** Each agent sees the other's earlier activations, so a deliberate lie is easy to detect even if it is hard to refute. {#c-shared-activations}
- **Assigned answers, interactive judges and probabilities**, as in the experiments of Michael et al. and Khan et al. above.
- **Consultancy**, the one-expert baseline above.
- **Doubly-efficient debate**, with polynomial-time debaters and a constant number of human judgements, as in the evidence above.

**Related methods.**
Debate grew out of an attempt to understand the amplification approach to alignment of Christiano et al. (2018), to which it is closely related. {#c-amplification}
Like [[al:adversarial-training|adversarial training]], it sets two learners against each other, but here the opponent's role is to expose flaws to a judge, not to fool a classifier.
The analysis uses the game theory of [[al:zero-sum-game|zero-sum games]] and [[al:nash-equilibrium|Nash equilibria]], and the MNIST experiment plays the game with [[al:monte-carlo-tree-search|Monte Carlo tree search]].
Each debater sees the whole transcript, as in a [[al:perfect-information-game|perfect-information game]], but the judge who decides the game sees only what the debaters reveal of the hidden image or text.

## When to use it

Use debate when the judge can check small steps but not the final answer, when two capable debaters can be trained or prompted, and when the debaters can back their points with evidence the judge can verify, as the quote tool did in the experiments.
Prefer it to a single expert whose answer the judge must trust: in both human and LLM experiments, adding an opponent helped judges more than a lone consultant did, and the gap grew with debater skill.
Ask the judge for probabilities so that an unclear debate ends in low confidence, not in a confident mistake.
Do not rely on it for stochastic questions, for arguments too long for anyone to check step by step, or without measuring how good the judges are.

## Pitfalls

> [!pitfall]
> **No precommit.** A liar who can change its claim between branches wins far more often; fix the answers before the argument, as in the MNIST experiment.

> [!pitfall]
> **No additivity check in a bisection.** If the judge does not require the two halves to add up to the claim being defended, a liar can state true halves under a false total. The derivation uses this check in every step, and its check finds every false claim losing only with it in place.

> [!pitfall]
> **A first-mover edge.** One seat can win more often regardless of the truth; measure win rates by seat, or use a symmetric variant.

> [!pitfall]
> **One order for an LLM judge.** An LLM judge may favor one position; judge each debate in both orders, as Khan et al. do.

> [!pitfall]
> **Unverified evidence.** Without checked quotes or other trusted evidence, a dishonest debater can tell a consistent false story that the judge has no means to refute.

> [!pitfall]
> **Ending after one round.** A judge who stops at once gives the honest debater no chance to answer; require at least one round after the first and after any swing.

## History

AI safety via debate, by Geoffrey Irving, Paul Christiano and Dario Amodei of OpenAI, is arXiv 1805.00899; its v2 is dated 22 Oct 2018. {#c-hist-irving}
Debate Helps Supervise Unreliable Experts, by Michael et al. of New York University, Cohere and Anthropic, was posted as arXiv 2311.08702v1 on 15 Nov 2023. {#c-hist-michael}
Scalable AI Safety via Doubly-Efficient Debate, by Jonah Brown-Cohen and Geoffrey Irving of Google DeepMind with Georgios Piliouras, was posted as arXiv 2311.14125v1 on 23 Nov 2023. {#c-hist-bc}
Debating with More Persuasive LLMs Leads to More Truthful Answers, by Khan et al., appeared in the Proceedings of the 41st International Conference on Machine Learning in Vienna, PMLR 235, 2024. {#c-hist-khan}
