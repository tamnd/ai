---
tag: NEW-does-scaling-lead-to-general-intelligence
kind: open-problem
slug: does-scaling-lead-to-general-intelligence
name: Does scaling lead to general intelligence?
aliases: [scaling hypothesis, is scale all you need, scaling and AGI]
area: ai-foundations/general-intelligence-concepts
level: research
summary: >-
  The open question whether training deep learning models with more parameters, more data and more compute, with no other change of method, leads to artificial general intelligence.
sources:
  - {paper: "Kaplan et al. (2020)", title: "Scaling Laws for Neural Language Models", locator: "the title page, the abstract, pp. 7 and 17", note: "the power laws of loss and where they are expected to break down"}
  - {paper: "Legg and Hutter (2007)", title: "A Collection of Definitions of Intelligence", locator: "the abstract, pp. 1, 2 and 9"}
  - {paper: "Chollet (2019)", title: "On the Measure of Intelligence", locator: "the abstract, pp. 19, 20 and 27", note: "the definition of intelligence and ARC"}
  - {paper: "Morris et al. (2024)", title: "Levels of AGI for Operationalizing Progress on the Path to AGI", locator: "ICML 2024, the abstract, p. 1, Sections 2 and 3", note: "case studies of definitions of AGI and the levels"}
  - {paper: "Villalobos et al. (2024)", title: "Will we run out of data? Limits of LLM scaling based on human-generated data", locator: "ICML 2024, the abstract, Fig. 1"}
  - {paper: "LeCun (2022)", title: "A Path Towards Autonomous Machine Intelligence", locator: "version 0.9.2, the title page, the abstract, Section 8.3.1"}
  - {paper: "Marcus (2018)", title: "Deep Learning: A Critical Appraisal", locator: "the abstract, pp. 6 to 7"}
  - {paper: "Marcus (2020)", title: "The Next Decade in AI: Four Steps Towards Robust Artificial Intelligence", locator: "the title page, the abstract"}
  - {paper: "Lake, Ullman, Tenenbaum and Gershman (2017)", title: "Building Machines That Learn and Think Like People", locator: "Behavioral and Brain Sciences 40, the abstract"}
  - {paper: "Chollet, Knoop, Kamradt and Landers (2024)", title: "ARC Prize 2024: Technical Report", locator: "the abstract, Sections 1, 1.2, 3 and 5"}
  - {paper: "Lake and Baroni (2018)", title: "Generalization without systematicity", locator: "ICML 2018, the abstract"}
  - {paper: "Dziri et al. (2023)", title: "Faith and Fate: Limits of Transformers on Compositionality", locator: "NeurIPS 2023, the abstract, Section 1"}
  - {paper: "McKenzie et al. (2023)", title: "Inverse Scaling: When Bigger Isn't Better", locator: "TMLR 2023, the abstract"}
  - {paper: "Muennighoff et al. (2023)", title: "Scaling Data-Constrained Language Models", locator: "NeurIPS 2023, the abstract"}
  - {paper: "Lake and Baroni (2023)", title: "Human-like systematic generalization through a meta-learning neural network", locator: "Nature 623, the abstract"}
  - {paper: "Raji et al. (2021)", title: "AI and the Everything in the Whole Wide World Benchmark", locator: "NeurIPS 2021 Datasets and Benchmarks, Section 1"}
  - {paper: "Altmeyer et al. (2024)", title: "Position: Stop Making Unscientific AGI Performance Claims", locator: "ICML 2024, the abstract"}
  - {paper: "Blili-Hamelin et al. (2025)", title: "Stop treating `AGI' as the north-star goal of AI research", locator: "the abstract, pp. 2 and 5"}
  - {paper: "Snell, Lee, Xu and Kumar (2024)", title: "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters", locator: "ICLR 2025, the abstract, Sections 4 and 7"}
  - {paper: "Silver and Sutton (2025)", title: "Welcome to the Era of Experience", locator: "p. 1", note: "a preprint of a chapter of Designing an Intelligence (MIT Press)"}
  - {paper: "DeepSeek-AI (2025)", title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", locator: "the abstract, Sections 1, 2.3 and 6", note: "a developer report on its own models"}
  - {paper: "Chollet et al. (2025)", title: "ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems", locator: "Sections 4.1, 4.3, 5 and 6"}
  - {paper: "Kambhampati (2024)", title: "Can Large Language Models Reason and Plan?", locator: "Annals of the New York Academy of Sciences 1534, pp. 2 to 3"}
  - {paper: "Bubeck et al. (2023)", title: "Sparks of Artificial General Intelligence: Early experiments with GPT-4", locator: "the abstract"}
  - {url: "https://gwern.net/scaling-hypothesis", title: "Branwen, The Scaling Hypothesis", locator: "p. 1", note: "an essay"}
  - {url: "https://www.noemamag.com/artificial-general-intelligence-is-already-here/", title: "Agüera y Arcas and Norvig, Artificial General Intelligence Is Already Here, Noema", locator: "p. 1", note: "an essay"}
  - {paper: "Brown et al. (2020)", title: "Language Models are Few-Shot Learners", locator: "NeurIPS 2020, the abstract"}
  - {paper: "Wei et al. (2022)", title: "Emergent Abilities of Large Language Models", locator: "TMLR 2022, the abstract, p. 6, Section 5.2, p. 10"}
  - {paper: "Schaeffer, Miranda and Koyejo (2023)", title: "Are Emergent Abilities of Large Language Models a Mirage?", locator: "NeurIPS 2023, the abstract"}
  - {paper: "Mitchell (2021)", title: "Why AI is Harder Than We Think", locator: "pp. 3 to 4"}
  - {url: "http://www.incompleteideas.net/IncIdeas/BitterLesson.html", title: "Sutton, The Bitter Lesson", locator: "p. 1", note: "an essay"}
  - {paper: "AAAI 2025 Presidential Panel (2025)", title: "AAAI 2025 Presidential Panel on the Future of AI Research", locator: "the title page, pp. 7, 8 and 66", note: "the community survey"}
question: "Does scaling up deep learning models in parameters, training data and compute, with no other change of method, lead to artificial general intelligence?"
positions:
  - {name: "Scaling is not enough", summary: "Scaling deep learning models will not by itself lead to artificial general intelligence, and other ideas or architectures are needed.", proponents: [yann-lecun, gary-marcus, francois-chollet, brenden-lake], sources: [lecun-2022-path-autonomous-machine-intelligence, marcus-2018-deep-learning-critical-appraisal, marcus-2020-next-decade-ai, chollet-2019-measure-of-intelligence, chollet-2024-arc-prize-2024-report, lake-2017-building-machines-learn-think]}
  - {name: "The question is not well posed", summary: "The term artificial general intelligence has no shared meaning or measure, so the question has no definite answer, and research should state specific goals instead.", proponents: [inioluwa-deborah-raji, patrick-altmeyer, borhane-blili-hamelin], sources: [raji-2021-everything-whole-wide-world-benchmark, altmeyer-2024-stop-unscientific-agi-claims, blili-hamelin-2025-stop-treating-agi]}
  - {name: "Scaled models with added methods", summary: "Scaled models are a basis, and further gains toward general capability come from methods added to them, such as test-time computation, reinforcement learning and learning from experience.", proponents: [charlie-snell, david-silver, richard-sutton], sources: [snell-2024-scaling-test-time-compute, silver-2025-era-of-experience, deepseek-2025-r1]}
  - {name: "Scaling is the main route", summary: "Scaling up neural networks with their data and compute is the main route to artificial general intelligence, and scaled models already show parts of it.", proponents: [sebastien-bubeck, gwern-branwen, blaise-aguera-y-arcas, peter-norvig], sources: [bubeck-2023-sparks-agi, gwern-2020-scaling-hypothesis, aguera-y-arcas-2023-agi-already-here]}
evidence: [c-power-law, c-brown-fewshot, c-mirage, c-repeat, c-snell-4x, c-snell-14x, c-snell-hard, c-r1-rl, c-r1-aime, c-arc-2024-score, c-arc-greenblatt, c-kambhampati-obfuscation, c-arc2-scores, c-mckenzie, c-dziri, c-dziri-multiplication, c-lake-scan, c-lake-mlc, c-aaai-survey, c-aaai-sample, c-arc2-humans, c-villalobos, c-villalobos-stock]
would_settle_it: >-
  The positions do not agree on one decisive test. Proposed tests include benchmarks built to measure skill acquisition on unseen tasks against human baselines, such as ARC-AGI and ARC-AGI-2, benchmarks scored against levels of performance and generality, and fixed lists of tasks; a further test is whether the scaling trends hold past the measured range and past the stock of human text.
---

Whether scaling leads to general intelligence is an open question of artificial intelligence (AI) research.
Scaling here means training deep learning models, chiefly large language models (LLMs), with more parameters, more data and more compute.
The question is whether this alone leads to artificial general intelligence (AGI).
The parties define AGI in different ways, and some doubt that the question has an answer.
This page gives the definitions, four positions, the shared evidence and the proposed tests.

## The question

The question is: does scaling up deep learning models in parameters, training data and compute, with no other change of method, lead to AGI?

**Scaling.** Kaplan et al. (2020) find that the loss of Transformer language models falls as a power law in model size, data and compute. {#c-power-law}
Some of these trends span more than seven orders of magnitude.
The page [[neural-scaling-laws]] covers these laws, their scope and their limits.
On this page, methods added to a scaled model, such as extra computation at test time or reinforcement learning, count as a change of method.

**General intelligence.** There is no agreed definition, and the positions below use different ones.
Shane Legg and Marcus Hutter (2007) collect 70-odd informal definitions of intelligence. {#c-def-collection}
They adopt the definition that intelligence measures an agent's ability to achieve goals in a wide range of environments. {#c-def-legg}
François Chollet (2019) defines intelligence as skill-acquisition efficiency over a scope of tasks, with respect to priors, experience and generalization difficulty. {#c-def-chollet}
Morris et al. (2024) report that the OpenAI charter defines AGI as highly autonomous systems that outperform humans at most economically valuable work. {#c-def-openai}
They also quote Gary Marcus, for whom AGI is intelligence that is flexible and general, with resourcefulness and reliability comparable to or beyond human intelligence. {#c-def-marcus}
The first definition is about goals across environments, the second about efficiency of learning, the third about economic work and the fourth about human comparison.

## Why it matters

Morris et al. (2024) argue that the concept of AGI matters because it maps onto goals for, predictions about, and risks of AI. {#c-why-morris}
The answer bears on where research effort goes: on larger models, on added methods, or on other architectures.
It also bears on what happens when the supply of training text runs out.
Pablo Villalobos et al. (2024) study how progress in language modeling can continue once human-generated text datasets cannot be scaled further. {#c-why-villalobos}
They forecast that, if trends continue, training datasets will match the stock of public human text between 2026 and 2032. {#c-villalobos}
Their median year for full use of the stock is 2028, at around 4e14 tokens. {#c-villalobos-stock}

## Positions

The positions are listed by the number of independent sources that hold them, counted as distinct authors of works with standing (peer-reviewed or preprint).
The counts are four, three, two and one.
The third position is also held in a report by a developer about its own models, and the fourth in two essays.
Each position has the same parts: the claim, the argument in its proponents' terms, the evidence, and the objections.

### Scaling is not enough

**Claim.** Scaling deep learning models will not by itself lead to AGI, and other ideas or architectures are needed.

**Argument.** Yann LeCun (2022) describes a debate on whether scaling up LLM architectures can reach human-level AI, and does not believe scaling is enough. {#c-lecun-position}
His first reason is that these models are generative and work on tokenized data, which suits text more than continuous signals such as video. {#c-lecun-tokens}
His second reason is that the models are only capable of very limited forms of reasoning. {#c-lecun-reasoning}
In place of scaling, he proposes an architecture that combines a configurable predictive world model with behavior driven through intrinsic motivation. {#c-lecun-arch}
Its other part is hierarchical joint embedding architectures trained with self-supervised learning.
Gary Marcus (2018) presents ten concerns for deep learning and argues that it must be supplemented by other techniques to reach AGI. {#c-marcus-2018}
Among them, he argues that deep learning is data hungry, shallow, and limited in its capacity for transfer. {#c-marcus-data}
Marcus (2020) contrasts an emphasis on ever-larger training sets and more compute with a hybrid, knowledge-driven, reasoning-based approach centered on cognitive models. {#c-marcus-2020}
Chollet (2019) argues that solving a task with unlimited priors or unlimited data brings us no closer to general AI. {#c-chollet-local}
He describes deep learning models as local-generalization systems.
Brenden Lake et al. (2017) argue that human-like machines should build causal models and ground learning in intuitive theories of physics and psychology. {#c-lake-2017}
They add that such machines should use compositionality and learning-to-learn.

**Evidence.** Chollet et al. (2024) write that from 2020 to early 2024, scaling raised task-specific skills but not performance on tasks without training data. {#c-arc-scaling-era}
Their report on the ARC Prize concludes that ARC-AGI remains undefeated and that "new ideas are still needed to build AGI". {#c-arc-new-ideas}
Lake and Baroni (2018) find that recurrent networks on SCAN generalize to small differences from training, but fail when systematic compositional skills are needed. {#c-lake-scan}
Nouha Dziri et al. (2023) find that transformer LLMs reduce multi-step compositional reasoning to linearized subgraph matching, on multiplication, logic grid puzzles and dynamic programming. {#c-dziri}
In their tests, off-the-shelf ChatGPT and GPT-4 reach 55% and 59% accuracy on 3-digit by 3-digit multiplication. {#c-dziri-multiplication}
Ian McKenzie et al. (2023) report inverse scaling, worse performance with scale, on 11 datasets from a public contest, the Inverse Scaling Prize. {#c-mckenzie}
They conclude that there are tasks for which increased model scale alone may not lead to progress. {#c-mckenzie-concl}
Niklas Muennighoff et al. (2023) find that up to 4 epochs of repeated data barely change loss compared with unique data. {#c-repeat}
With more repetition, the value of added compute eventually decays to zero.
Their models have up to 9 billion parameters and 900 billion training tokens.

**Objections.** Lake and Baroni (2023) report that a standard network trained by meta-learning for compositionality (MLC) reaches human-like systematic generalization. {#c-lake-mlc}
In their comparison, none of six other models does.
DeepSeek-AI (2025) report that pure reinforcement learning (RL) incentivizes reasoning in LLMs on verifiable tasks, a report about their own models. {#c-r1-rl}
Villalobos et al. (2024) argue that synthetic data, transfer learning from data-rich domains and gains in data efficiency might support further progress. {#c-villalobos-escape}

### The question is not well posed

**Claim.** The term AGI has no shared meaning or measure, so the question has no definite answer, and research should state specific goals instead.

**Argument.** Borhane Blili-Hamelin et al. (2025) argue that popular use of the term AGI hides vastly different opinions on what it means. {#c-blili-consensus}
They argue that researchers should prioritize specific scientific, engineering and societal goals, and stop treating AGI as the north-star goal of AI research. {#c-blili-goals}
Inioluwa Deborah Raji et al. (2021) argue that benchmarks framed as general measures of progress cannot capture the general applicability claimed for them. {#c-raji}
Patrick Altmeyer et al. (2024) argue that meaningful patterns in the latent spaces of models cannot be seen as evidence in favor of AGI. {#c-altmeyer}

**Evidence.** Blili-Hamelin et al. (2025) list many notions of generality, among them the variety of tasks and whether evaluation data was seen in training. {#c-blili-generality}
The four definitions in the question above measure different things.

**Objections.** Legg and Hutter (2007) find strong similarities between many definitions of intelligence, and expect a single general definition for arbitrary systems to be possible. {#c-legg-single}
Morris et al. (2024) propose levels of AGI; a Competent AGI performs at least at the 50th percentile of skilled adults on most cognitive tasks. {#c-def-levels}
They discuss the requirements for future benchmarks that would measure AGI models against these levels. {#c-morris-bench}

### Scaled models with added methods

**Claim.** Scaled models are a basis, and further gains toward general capability come from methods added to them.
The methods named are test-time computation, RL and learning from experience.

**Argument.** David Silver and Richard Sutton (2025) argue that imitating humans reproduces many human capabilities competently, but likely cannot achieve superhuman intelligence across many tasks. {#c-silver-imitation}
They write that in mathematics, coding and science, the knowledge extracted from human data is rapidly approaching a limit. {#c-silver-limit}
They argue that agents should learn continually from their own experience, which will become the dominant medium of improvement. {#c-silver-experience}
Charlie Snell et al. (2024) describe more test-time computation as a critical step towards generally self-improving agents. {#c-snell-motivation}

**Evidence.** On the MATH benchmark, Snell et al. (2024) find that adaptive test-time compute is more than 4x as efficient as a best-of-N baseline. {#c-snell-4x}
Their results come from one base model, PaLM 2-S*.
In a FLOPs-matched evaluation, test-time compute lets the smaller model outperform a 14x larger model on problems it partly solves. {#c-snell-14x}
DeepSeek-AI (2025) report that the AIME 2024 pass@1 score of their model DeepSeek-R1-Zero rose from 15.6% to 77.9% during RL. {#c-r1-aime}
During ARC Prize 2024, private-set scores on ARC-AGI rose from 33% to 55.5%, with deep learning-guided program synthesis and test-time training. {#c-arc-2024-score}
In test-time training, an LLM is fine-tuned on a given task to recombine its prior knowledge into a model adapted to that task. {#c-arc-ttt}
Ryan Greenblatt reached 42% on the ARC-AGI-Pub leaderboard with GPT-4o searching thousands of Python programs per task. {#c-arc-greenblatt}

**Objections.** Snell et al. (2024) themselves find that on hard questions beyond a base model's reach, pretraining is likely more effective than test-time compute. {#c-snell-hard}
DeepSeek-AI (2025) write that scaling pure RL remains an open challenge for complex tasks that no reliable reward model can evaluate. {#c-r1-limit}
On ARC-AGI-2, as of May 14, 2025, the listed systems score at most 3.0%, the score of o3 (Medium), which scores 53.0% on ARC-AGI-1. {#c-arc2-scores}
The authors of ARC-AGI-2 do not treat accuracies below 5% as meaningful. {#c-arc2-noise}

### Scaling is the main route

**Claim.** Scaling up neural networks with their data and compute is the main route to AGI, and scaled models already show parts of it.

**Argument.** Gwern Branwen (2020) states the scaling hypothesis: intelligence is "'just' simple neural units & learning algorithms applied to diverse experiences" at an unreached scale. {#c-gwern-hypothesis}
On this hypothesis, as computing resources permit such algorithms to run at the necessary scale, networks get ever more intelligent. {#c-gwern-more}
Branwen reads the meta-learning of GPT-3 as a demonstration that networks become more generalizable and human-like when made very large. {#c-gwern-blessings}
Blaise Agüera y Arcas and Peter Norvig (2023) argue that the most important parts of AGI have already been achieved by advanced LLMs. {#c-noema-already}
They write that "frontier models perform competently even on novel tasks they were not trained for". {#c-noema-novel}
Sébastien Bubeck et al. (2023) argue that GPT-4, trained with unprecedented compute and data, could be viewed as an early, incomplete AGI system. {#c-bubeck-agi}

**Evidence.** Tom Brown et al. (2020) find that scaling up language models greatly improves task-agnostic, few-shot performance, up to the 175 billion parameters of GPT-3. {#c-brown-fewshot}
Jason Wei et al. (2022) call an ability emergent if larger models have it and smaller ones lack it, so extrapolation cannot predict it. {#c-emergent-def}
They ask whether further scaling could endow even larger language models with further emergent abilities. {#c-wei-question}
Morris et al. (2024) class the language models of September 2023, such as ChatGPT, Bard and Llama 2, as Level 1, "Emerging AGI". {#c-def-emerging}

**Objections.** Bubeck et al. (2023) themselves discuss a possible need for "a new paradigm that moves beyond next-word prediction". {#c-bubeck-paradigm}
Melanie Mitchell (2021) names as a fallacy the assumption that narrow intelligence is on a continuum with general intelligence. {#c-mitchell-fallacy}
Chollet (2019) argues that unlimited priors or training data let experimenters buy arbitrary levels of skill, which masks a system's generalization power. {#c-chollet-buy}
He adds that more training data can raise skill on a task without affecting generalization at all. {#c-chollet-moredata}
Rylan Schaeffer et al. (2023) find that apparent emergent abilities come from nonlinear metrics, and that continuous metrics show smooth, predictable change. {#c-mirage}
Wei et al. (2022) write that other approaches beyond scaling will likely play a key role, such as architectures, data quality and training procedures. {#c-wei-beyond}
Subbarao Kambhampati (2024) reports that GPT4 reached 30% accuracy on Blocks World plans, and fell sharply when action and object names were obfuscated. {#c-kambhampati-obfuscation}
He argues that LLM answers still need checking by sound external verifiers, in frameworks he calls LLM-Modulo. {#c-kambhampati-modulo}

## Evidence so far

> [!note]
> The build lists here the claims that bear on the question, with their evidence labels, and each position above links the claims it relies on.
> Most of them are measured on particular models and benchmarks, and their scope is part of the claim.

The Association for the Advancement of Artificial Intelligence (AAAI) ran a community survey for its 2025 Presidential Panel, which engaged 475 respondents.
Of the respondents, 76% assert that "scaling up current AI approaches" to yield AGI is "unlikely" or "very unlikely" to succeed. {#c-aaai-survey}
About 20% of respondents were students, and 67% gave academia and 19% corporate research as their main affiliation. {#c-aaai-sample}
The report does not give a response rate, the full wording of the question, or how many respondents answered it.
It gives no date for the survey; the panel worked on the study between summer 2024 and spring 2025. {#c-aaai-period}

## What would settle it

The positions do not agree on one decisive test, and the fourth position doubts that one exists.
Chollet (2019) proposes the Abstraction and Reasoning Corpus (ARC) as a measure of human-like general fluid intelligence, allowing fair comparisons of AI systems with humans. {#c-chollet-arc}
For ARC-AGI-2, 407 participants made 13,405 attempts at test pairs, and solved 8,277 (62%) of them. {#c-arc2-humans}
A task was kept only if at least two participants solved it within two attempts. {#c-arc2-selection}
Morris et al. (2024) report that Marcus proposes five tasks, among them understanding a movie or a novel and cooking in an arbitrary kitchen. {#c-marcus-tasks}
Whether the scaling trends hold far past the measured range is itself unsettled.
Kaplan et al. (2020) find that their laws contradict each other far beyond the sizes studied, and expect their predictions to break down. {#c-breakdown}

## History of the debate

Morris et al. (2024) report that the term AGI was first used in a 1997 article on military technologies by Mark Gubrud. {#c-hist-gubrud}
They report that Legg and Ben Goertzel popularized it in 2001, for a machine able to do the cognitive tasks people typically can. {#c-hist-legg-goertzel}
Marcus wrote his critical appraisal of deep learning about five years after the ImageNet model of 2012 by Krizhevsky, Sutskever and Hinton. {#c-hist-marcus-2018}
Chollet first wrote about the limits of deep learning in 2017, and in 2019 published a definition of AGI with the ARC benchmark. {#c-hist-chollet}
The Bitter Lesson, an essay by Rich Sutton, is dated March 13, 2019. {#c-hist-sutton}
It argues that the biggest lesson of 70 years of AI research is that general methods that use computation are the most effective. {#c-sutton-lesson}
Sutton names search and learning as the two methods that seem to scale arbitrarily. {#c-sutton-search-learning}
He cites chess, where deep search beat Kasparov in 1997, and Go, where search applied at scale made other efforts irrelevant. {#c-sutton-chess}
He argues that researchers should build in only the meta-methods that can find and capture the arbitrary complexity of the world. {#c-sutton-meta}
Kaplan et al. posted their scaling laws as arXiv 2001.08361 on 23 Jan 2020. {#c-hist-kaplan}
Marcus dated The Next Decade in AI 17 February 2020. {#c-hist-marcus-2020}
Version 0.9.2 of LeCun's position paper is dated June 27, 2022. {#c-hist-lecun}
On December 5, 2024, the ARC Prize report described ARC-AGI as five years old and unbeaten, against a prize target of 85%. {#c-hist-arc-2024}
The AAAI Presidential Panel published its report on the future of AI research in March 2025. {#c-hist-aaai}

## References
