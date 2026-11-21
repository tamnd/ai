---
tag: NEW-language-model-planning
kind: finding
slug: language-model-planning
name: Planning ability of language models
aliases: [LLM planning, can LLMs plan, planning with large language models, PlanBench results, Mystery Blocksworld, LLM-Modulo, LLMs as planners]
area: reasoning-and-test-time-compute/planning-in-language-models
level: core
summary: >-
  Language models asked to write plans for classical planning problems fail often, fail badly when names are disguised, and do far better as plan proposers checked by a sound verifier.
related_to: [chain-of-thought-prompting, tree-of-thoughts, test-time-scaling, chain-of-thought-faithfulness, few-shot-in-context-learning]
history:
  year: 2022
  note: "Huang et al. (January 2022) prompted language models for household action plans; Valmeekam et al. released PlanBench in June 2022 (NeurIPS 2023 Datasets and Benchmarks) and the critical investigation in May 2023 (NeurIPS 2023); Stechly et al. (May 2024, NeurIPS 2024) tested chain of thought on Blocksworld; Valmeekam et al. (September 2024) evaluated OpenAI's o1; Kambhampati et al. (ICML 2024) proposed LLM-Modulo frameworks."
prereqs: [chain-of-thought-prompting, few-shot-in-context-learning]
same_as:
  arxiv: ["2206.10498", "2305.15771", "2402.01817", "2409.13373", "2405.04776", "2310.08118", "2201.07207"]
sources:
  - {paper: "Valmeekam, Marquez, Olmo, Sreedharan and Kambhampati (2023)", title: "PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change", locator: "NeurIPS 2023 Datasets and Benchmarks, arXiv 2206.10498v4, pp. 1 to 3, 7, 8 and 10", note: "the benchmark, its domains and tasks, and the GPT-4 and Instruct-GPT3 results"}
  - {paper: "Valmeekam, Marquez, Sreedharan and Kambhampati (2023)", title: "On the Planning Abilities of Large Language Models: A Critical Investigation", locator: "NeurIPS 2023, arXiv 2305.15771v2, pp. 1, 3, 6, 7 and 9", note: "autonomous and LLM-Modulo modes, obfuscation, fine-tuning, the human baseline, LPG seeding and VAL back-prompting"}
  - {paper: "Valmeekam, Stechly and Kambhampati (2024)", title: "LLMs Still Can't Plan; Can LRMs? A Preliminary Evaluation of OpenAI's o1 on PlanBench", locator: "arXiv 2409.13373v1, pp. 2 to 6", note: "recent LLMs and o1 on Blocksworld, Mystery Blocksworld, larger and unsolvable problems, time and cost"}
  - {paper: "Stechly, Valmeekam and Kambhampati (2024)", title: "Chain of Thoughtlessness? An Analysis of CoT in Planning", locator: "NeurIPS 2024, arXiv 2405.04776v3, pp. 1, 3 and 6", note: "chain-of-thought prompts of varying specificity and their failure to generalize with problem size"}
  - {paper: "Valmeekam, Marquez and Kambhampati (2023)", title: "Can Large Language Models Really Improve by Self-critiquing Their Own Plans?", locator: "arXiv 2310.08118v1, pp. 4 and 5", note: "GPT-4 as its own plan verifier against VAL, and levels of feedback"}
  - {paper: "Kambhampati, Valmeekam, Guan, Verma, Stechly, Bhambri, Saldyt and Murthy (2024)", title: "LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks", locator: "ICML 2024, arXiv 2402.01817v3, pp. 1, 2, 4 and 6", note: "the position and the LLM-Modulo framework"}
  - {paper: "Huang, Abbeel, Pathak and Mordatch (2022)", title: "Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents", locator: "arXiv 2201.07207v2, pp. 1 and 3", note: "household plans in VirtualHome and the trade-off between executability and correctness"}
  - {paper: "Hao et al. (2023)", title: "Reasoning with Language Model is Planning with World Model", locator: "arXiv 2305.14992, p. 2", note: "Monte Carlo tree search with the model as world model on Blocksworld"}
claim: >-
  Language models prompted to produce plans for classical planning problems such as Blocksworld solve a minority of instances, lose almost all of that success when the action and object names are replaced by unrelated words, and are not rescued by chain-of-thought prompting or by checking their own plans; reasoning models trained with reinforcement learning do much better on small problems but degrade on larger and unsolvable ones, while plans proposed by a language model and checked by a sound verifier are correct whenever they are returned.
first_reported: {work: valmeekam-2023-planbench, note: "PlanBench, first posted in June 2022, found GPT-3-family models and later GPT-4 solving a minority of Blocksworld plan-generation instances, with obfuscated domains and automatic validation of every plan"}
replications:
  - {work: valmeekam-2023-planning-abilities-critical-investigation, note: "GPT-4, GPT-3.5 and older GPT models on Blocksworld, Logistics and two obfuscations, with fine-tuning, a human baseline and LLM-Modulo modes"}
  - {work: valmeekam-2024-llms-still-cant-plan-o1, note: "ten LLMs from four families and the o1-preview and o1-mini reasoning models on the same 600 Blocksworld instances and their obfuscations"}
  - {work: stechly-2024-chain-of-thoughtlessness, note: "GPT-4, GPT-4-Turbo and Claude-3-Opus with chain-of-thought prompts of decreasing generality on Blocksworld problems of growing size"}
  - {work: valmeekam-2023-self-critiquing-plans, note: "GPT-4 as generator and as verifier of its own Blocksworld plans, compared with the sound verifier VAL"}
scale_range:
  - {range: "Instruct-GPT3 to GPT-4 on 600 Blocksworld problems of three to five blocks with plans of 2 to 16 steps", work: valmeekam-2023-planbench}
  - {range: "ten LLMs from Claude, GPT-4, LLaMA and Gemini families and two o1 models; 110 larger problems of 6 to 20 blocks with 20 to 40 step plans", work: valmeekam-2024-llms-still-cant-plan-o1}
  - {range: "table-to-stack problems with growing numbers of blocks, and scalable coin flip, last letter and arithmetic tasks", work: stechly-2024-chain-of-thoughtlessness}
failures:
  - {work: valmeekam-2024-llms-still-cant-plan-o1, note: "o1-preview solves 97.8% of the 600 Blocksworld instances and 52.8% of Mystery Blocksworld, far above any LLM, so the claim that models cannot plan at all does not hold for reasoning models on small problems"}
  - {work: hao-2023-reasoning-planning-world-model, note: "Monte Carlo tree search with the model as world model and reward reaches 64% on Blocksworld plan generation with LLaMA-33B"}
  - {work: huang-2022-language-models-zero-shot-planners, note: "in open-ended household tasks large models write plans that people judge as correct, though often not executable"}
explanations:
  - {name: "approximate retrieval", summary: "A language model reproduces plans that resemble plans it has seen, which works on common-sense versions of a domain and fails when disguised names remove the resemblance.", sources: [valmeekam-2023-planning-abilities-critical-investigation, kambhampati-2024-llm-modulo-frameworks]}
  - {name: "no internal check", summary: "Generating a token takes the same computation whatever the problem, and the model cannot reliably verify a plan, so errors in preconditions go uncaught unless something outside checks them.", sources: [kambhampati-2024-llm-modulo-frameworks, valmeekam-2023-self-critiquing-plans]}
  - {name: "no learned procedure", summary: "Demonstrations in a prompt teach the surface of the examples rather than the algorithm, so performance falls once problems grow past the size of the examples.", sources: [stechly-2024-chain-of-thoughtlessness]}
---

The planning ability of language models is the question of whether a model asked to produce a sequence of actions that reaches a goal can do so reliably.
The finding, from a series of studies on problems taken from the automated planning community, is that it cannot on its own.
Language models solve a minority of simple Blocksworld problems, almost none once the names of actions and objects are disguised, and they do not improve much with chain-of-thought prompts or by checking their own work.
Reasoning models trained with reinforcement learning, such as OpenAI's o1, solve most small problems, but still fail on larger ones and on problems that have no solution, at a cost far above a classical planner.
What does work is to treat the model as a source of candidate plans and have a sound verifier accept or reject them.

## In plain words

Blocksworld is a toy problem: some blocks sit on a table or on each other, a hand can move one clear block at a time, and the goal is a given arrangement.
A classical planner solves these problems in a fraction of a second and is guaranteed to be right.
The problems are easy for people too.

Language models often get them wrong.
They write plans that pick up a block that has another block on top of it, or that stop before the goal is reached.
Researchers then rename everything: the blocks become meaningless names and "pick up" becomes "attack".
The problem is the same, and a planner does not notice the change, but the models' success falls to almost nothing.
That suggests the models were drawing on what plans about blocks usually look like rather than working out what the rules allow.

The fix most of these studies support is to pair the model with a checker.
The model proposes a plan, a program that knows the rules checks it, and if the plan is wrong the checker says why and the model tries again.
The checker guarantees that any plan that comes out is correct; the model only has to propose good guesses often enough.

## The finding

The question needs problems where planning cannot be replaced by memory.
Most claims about LLM planning capabilities are based on common sense tasks, where it becomes hard to tell whether LLMs are planning or merely retrieving from their vast world knowledge. {#c-plb-motive}
Planning benchmarks avoid this by giving the full rules of a domain in the prompt and checking every plan mechanically.

On such problems the success rates are low.
LLMs' ability to generate executable plans autonomously is rather limited, with GPT-4 having an average success rate of about 12% across the domains, the highest rate among the models tested. {#c-cr-12}
These results strongly suggest that whatever accidental planning performance LLMs display is likely connected to pattern matching rather than reasoning, which should be robust to name change. {#c-cr-pattern}
Later models improved on the plain domain but not on the disguised one: Across vanilla LLMs on 600 Blocksworld problems the top result was LLaMA 3.1 405B with 62.6% accuracy, while on Mystery Blocksworld no LLM achieves even 5%. {#c-o1-llm}

Reasoning models changed the picture on small problems.
o1-preview correctly answers 97.8% of the 600 Blocksworld instances; on Mystery Blocksworld it answers 52.8% correctly and on Randomized Mystery Blocksworld 37.3%, sharply contrasting the flat zeroes of previous models. {#c-o1-main}
The same study found that this success fades on larger and unsolvable problems, which the Evidence section describes.

The authors' position, set out in an ICML 2024 paper, is that Auto-regressive LLMs cannot, by themselves, do planning or self-verification, and should be viewed as universal approximate knowledge sources. {#c-mo-pos}
The rest of this page reads that claim against the evidence, which supports the first half for language models used directly and complicates it for reasoning models.

## Evidence

**The benchmarks.**
PlanBench is an extensible benchmark suite based on the kinds of domains used in the automated planning community, especially in the International Planning Competition, to test the capabilities of LLMs in planning or reasoning about actions and change. {#c-plb-design}
PlanBench is initialized with two IPC domains, Blocksworld and Logistics, with obfuscated versions that use misleading words or random alphanumeric strings, about 26250 prompts in all. {#c-plb-domains}
A classical planning problem is the tuple P = ⟨D, I, G⟩ of a domain, an initial state and a goal specification; an action is executable only if its preconditions hold, and its effects add and delete predicates. {#c-plb-classical}
Blocksworld, the domain most results use, has four actions, pick up, put down, stack and unstack, each with preconditions such as "the block is clear" and "the hand is empty".

PlanBench tests more than plan generation.
In the plan verification task the LLM is asked whether a candidate plan is valid; if it answers no, it must identify the first inexecutable action and at least one missing precondition, or at least one missing goal condition. {#c-plb-verify}
Other tasks ask the model to predict the state after a plan, to replan after an unexpected change, to reuse part of a plan and to recognize a goal stated in a different order.
The Blocksworld domain has 600 instances which vary in the number of objects and the optimal plan length, and the Logistics domain 285 instances. {#c-plb-600}
The 600 Blocksworld instances range from three to five blocks and require plans of between 2 to 16 steps to solve. {#c-o1-range}

The obfuscated domains are the key control.
Mystery Blocksworld obfuscates action names, so that pickup becomes attack and unstack becomes feast, and predicate names, so that ontable becomes planet and handempty becomes harmony; from the perspective of standard planners the domains are essentially identical. {#c-cr-obf-names}
Obfuscated domains were originally introduced into the International Planning Competition by Drew McDermott in 1998, to check whether the competition planners were truly domain independent. {#c-cr-mcdermott}
A model that plans from the rules given in the prompt should solve the renamed domain as well as the original; a model that recalls typical block-moving plans should not.

The derivation blocksworld-and-generate-test works out the size of these problems.
Blocksworld with n blocks has a(n) hand-empty states, the sum of the Lah numbers, with a(n) = (2n − 1) a(n − 1) − (n − 1)(n − 2) a(n − 2): 13, 73 and 501 for three, four and five blocks, and 22, 125 and 866 states in all once a held block is allowed; a(20) is about 3.277 × 10^20. {#c-d-states}
Unstacking every tower and rebuilding the goal towers gives a plan of 2u + 2g ≤ 4(n − 1) actions, where u and g count the blocks on other blocks at the start and at the goal, and breadth-first search finds longest optimal plans of 8, 12 and 16 actions for three, four and five blocks. {#c-d-length}
The longest optimal plan for five blocks is 16 actions, which matches the 2 to 16 steps of the PlanBench test set.
The state spaces are small enough that a classical planner searches them almost instantly.

**Plan generation with language models.**
On Blocksworld plan generation GPT-4 solved 206 of 600 instances (34.3%) and Instruct-GPT3 41 of 600 (6.8%), and in plan verification GPT-4 answered 352 of 600 (58.6%) correctly. {#c-plb-table}
With zero-shot prompting and natural language input, GPT-4 solved 210 of 600 instances in Blocksworld but only 1 instance in the deceptive Mystery Blocksworld domain and 0 in the randomized mystery domain. {#c-cr-obf}
Renaming took GPT-4 from about a third of the instances to almost none.
Chain of thought prompting does not significantly improve performance over one-shot natural language prompts. {#c-cr-cot}
GPT-3 fine-tuned on 1,000 Blocksworld instances separate from the test set solved only 122 of the 600 test instances, approximately 20%. {#c-cr-ft}
In an IRB-approved user study, 39 of 50 participants (78%) came up with a valid plan for a Blocksworld instance, and 35 of the 39 (89.7%) came up with an optimal plan. {#c-cr-human}
People were given the same domain description and one example, so the gap is not explained by the wording of the prompt.

**Chain-of-thought prompting.**
Stechly, Valmeekam and Kambhampati tested whether a chain-of-thought prompt can teach a model a planning procedure.
They used table-to-stack problems, where every block starts on the table and the goal is one tower; the derivation gives their plan length.
A table-to-stack problem of n blocks needs exactly 2(n − 1) actions, 6 for four blocks and 38 for twenty, by the same procedure at every size. {#c-d-stack}
On table-to-stack problems GPT-4's zero-shot performance over 261 instances is 3.8%, and with the stacking chain-of-thought prompt it improves to 59.3%. {#c-co-261}
Of prompts of varying granularity on table-to-stack problems, only the most specific and least applicable prompt retains anywhere near this performance improvement. {#c-co-only}
Chain of thought prompts give meaningful performance improvements only when they are exceedingly specific to their problem class, and those improvements quickly deteriorate as the size n of the query stack grows past the size of stacks shown in the examples. {#c-co-claim}
Scalable versions of three synthetic problems previously used to study chain of thought, Coin Flip, Last Letter Concatenation and multi-step arithmetic, were built with replicated chain of thought prompts. {#c-co-other}
The pattern was the same in those tasks: chain of thought helped on problems the size of the examples and lost its effect on larger ones.

**Reasoning models.**
On 110 larger Blocksworld problems of 6 to 20 blocks requiring 20 to 40 step optimal plans, o1-preview only manages 23.63%. {#c-o1-large}
On 100 unsolvable Blocksworld instances o1-preview correctly identified only 27% as unsolvable, returned an empty plan in 19% and a full plan in the remaining 54%; on Randomized Mystery Blocksworld 11.5% of solvable instances are incorrectly claimed to be impossible. {#c-o1-unsolv}
A plan for an unsolvable problem is wrong by definition, and earlier models almost never said a problem was unsolvable, so o1's 27% is progress, but most of its answers to these problems were plans that could not work.
o1-preview took 40.43 seconds per Blocksworld instance and 83.37 seconds per Mystery Blocksworld instance on average, against 0.265 seconds for Fast Downward, which solved 600 of 600 instances. {#c-o1-time}
At 40.43 seconds per Blocksworld problem for o1-preview against 0.265 seconds for Fast Downward, the reasoning model is 152.6 times slower, and 314.6 times slower on Mystery Blocksworld at 83.37 seconds. {#c-d-time}
Per 100 instances o1-preview cost $42.12 and o1-mini $3.69, and LRMs are significantly more expensive than LLMs. {#c-o1-cost}
Classical planners like Fast Downward achieve 100% on the dataset in a fraction of the time, compute and cost, while providing guarantees that their answers are correct. {#c-o1-fd}

**Search with the model as a world model.**
RAP achieves an average success rate of 64% on Blocksworld plan generation while CoT fails almost completely, and LLaMA-33B with RAP surpasses GPT-4 with CoT by 33% relative improvement. {#c-rap-result}
Here the language model predicts the effects of actions and scores states, and Monte Carlo tree search chooses the actions, so search outside the model does part of the planning.
The tree-of-thoughts page covers this family of methods.

**LLM-Modulo: the model proposes, a verifier decides.**
An LLM-Modulo Framework combines LLMs with external critics, verifiers and humans, a name loosely inspired by SAT Modulo Theories. {#c-mo-frame}
In such a framework the guarantees, if any, come in terms of the soundness of the external verifier. {#c-mo-sound}
The idea is a generate-test loop: the model proposes a plan, a sound verifier checks it against the domain model, and the verifier's report is sent back to the model as the next prompt.
The evidence for the loop comes in two forms.
Seeding the local-search planner LPG with LLM plans made the average number of search steps much lower than with an empty or a random seed plan, so the LLM plans help the search of the underlying planner. {#c-cr-lpg}
In the Mystery Blocksworld domain the LLM seed plan gives only a meager reduction in search steps over the random plan and uses more steps than the empty plan. {#c-cr-mys-lpg}
The plans help most where common sense applies, which is consistent with the approximate-retrieval explanation.
With the sound verifier VAL back-prompting GPT-4 for up to 15 rounds on failed instances, GPT-4 solved 41 of 50 (82%) in Blocksworld, 35 of 50 (70%) in Logistics and 5 of 50 (10%) in deceptive Mystery Blocksworld. {#c-cr-back}

**The verifier has to be sound.**
The model checking itself does not work.
On 100 instances of the Blocksworld domain, with GPT-4 as generator, the generator LLM only without backprompting reached 40/100 (40%), the LLM+LLM system with backprompting 55/100 (55%) and the LLM+VAL system with backprompting 88/100 (88%). {#c-cq-table}
The verifier LLM judged 61 of 100 final plans correctly but produced 38 false positives, deeming invalid plans valid; its true positive rate was 98.2% and its false positive rate 84.45%. {#c-cq-fp}
In the LLM+VAL system, only binary feedback gave 74% accuracy and binary with first error or all errors feedback 86%; the amount of feedback provided to the LLM seems to have minimal influence on its performance improvement. {#c-cq-feedback}
The derivation makes the reason exact.
With a sound verifier and K independent attempts that each succeed with probability p, generate-test succeeds with probability 1 − (1 − p)^K and returns only correct plans; at p = 0.4 and K = 15 this is 0.99953. {#c-d-sound}
The measured 88% is well below that bound, so failed attempts on a problem are strongly correlated: the model tends to repeat its mistakes even when told what they are.
A verifier with true positive rate t and false positive rate f returns a correct plan with probability p t / (p t + (1 − p) f) · (1 − r^K), r = 1 − p t − (1 − p) f; at p = 0.4, t = 0.982 and f = 0.8445 this is 0.4367 for any K ≥ 2. {#c-d-unsound}
An unsound verifier that accepts most wrong plans stops the loop in the first round almost every time, so the loop returns little more than the generator's own success rate.

## Scope and conditions

The results come mostly from Blocksworld and Logistics, deterministic domains with full information and exact rules.
Problems were given with the full domain description, in natural language or in PDDL, and plans were checked by a validator rather than judged by people.
Most prompts were one-shot, with one solved example; this is in-context learning, and the results say that a solved example teaches the format of a plan more than the rules that make it valid.

In open-ended household tasks the picture is different.
Large models can produce action plans indistinguishable from those by humans, but frequently are not executable in the environment. {#c-hu-indist}
Biasing model generation and translating the steps to admissible actions improved the executability of action plans from 18% to 79%, without modifying model parameters. {#c-hu-exec}
Human evaluation reveals a trade-off between executability and correctness. {#c-hu-trade}
Such tasks have no exact rules to check, so correctness was judged by people; they test commonsense knowledge of how tasks go more than planning under given constraints.

Most results are from GPT-family models from 2022 to 2024, and the o1 results are a preliminary evaluation of o1-preview and o1-mini in their first week of release.
Later reasoning models may do better, and the obfuscated, larger and unsolvable problems are the tests most likely to separate planning from retrieval.

## Explanations

**Approximate retrieval.**
The collapse under renaming is the main evidence.
If the model used the rules in the prompt, the names would not matter; that they matter so much means the model relies on its knowledge of what such plans usually look like.
The LPG results point the same way: the model's plans are useful seeds in the plain domain and useless in the disguised one.

**No internal check.**
A model generating a plan writes one action after another, and nothing in the process checks that each action's preconditions hold.
The self-critique study found that the model is also a poor checker, accepting most invalid plans, so it cannot catch its own errors by being asked to.
Kambhampati and coauthors argue that a system that spends the same computation on each token whatever the problem cannot be doing principled reasoning on its own.
Reasoning models weaken this argument, since they spend more tokens on harder problems, and their results are much better.

**No learned procedure.**
Chain of thought could in principle teach a procedure, and table-to-stack needs only one: find the bottom block, then stack the rest in order.
The chain-of-thought study found that the gain stayed tied to problems like the examples, which is what one expects if the prompt is matched as a pattern rather than followed as an algorithm.
This links the finding to chain-of-thought faithfulness, which asks whether the written reasoning is what produced the answer.

## Implications

For building agents, the finding argues against trusting a language model's plan without a check.
Where an exact domain model exists, a classical planner is faster, cheaper and correct; the model is useful for translating a problem into the planner's language or for proposing candidate plans.
Where no exact model exists, the LLM-Modulo design asks for whatever critics can be built, such as checks of hard constraints, with the model's proposals filtered through them.

For evaluation, the finding argues for tests that a model cannot pass from memory: disguised names, larger problems than the examples, and problems with no solution.
A plan should be judged by executing it, not by how it reads.

For reasoning models, the results show what test-time computation buys on planning: much higher accuracy on small problems, at a cost per problem far above a planner, with no guarantee of correctness.
Test-time scaling, the broader finding that accuracy rises with inference computation, covers the general trade-off; planning is a case where the extra computation raises accuracy without bringing a guarantee.

## Open questions

Whether reinforcement-learned reasoning models will close the gap on larger and disguised problems is open; the o1 results are from early versions with limited reasoning budgets.
Whether a learned verifier can be good enough is also open.
If soundness guarantees are not insisted on, it is in principle possible to train LLMs discriminatively to learn to verify plans. {#c-mo-train}
While o1-preview may provide higher accuracy than LLMs, it still fails to provide any correctness guarantees, and it is unclear that it is at all cost-effective. {#c-o1-guar}
How far the Blocksworld results carry to planning problems with uncertainty, partial information or no exact model, which are the problems agents meet, has been tested much less.

## History

Huang, Abbeel, Pathak and Mordatch prompted GPT-3 and Codex for household action plans in January 2022.
Valmeekam and coauthors at Arizona State University posted PlanBench in June 2022, building on the planning-competition tradition of disguised domains, and reported GPT-4 results with the critical investigation in May 2023; both appeared at NeurIPS 2023.
The self-critique study followed in October 2023, the LLM-Modulo position paper in February 2024 and the chain-of-thought study in May 2024.
After OpenAI released o1-preview in September 2024, Valmeekam, Stechly and Kambhampati evaluated it on the same test set within a week.

## References
