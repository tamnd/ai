---
tag: H0HPFC
kind: benchmark
slug: gpqa
name: GPQA
aliases: [GPQA Diamond, Graduate-Level Google-Proof Q&A Benchmark, GPQA main, GPQA Extended]
area: evaluation/benchmarks-and-their-design
level: core
summary: >-
  Multiple-choice questions in biology, physics and chemistry, written and checked by PhD-level experts and hard for skilled people from other fields even with the web.
history:
  year: 2023
  note: "Released on arXiv on 20 November 2023 (Rein et al.); published at the first Conference on Language Modeling, 2024."
sources:
  - {paper: "Rein, Hou, Stickland, Petty, Pang, Dirani, Michael and Bowman (2023)", title: "GPQA: A Graduate-Level Google-Proof Q&A Benchmark", locator: "pp. 1 to 19", note: "the data, the collection pipeline, the three sets, the human and model baselines and the limitations"}
  - {paper: "Fourrier, Habib, Lozovskaya, Szafer and Wolf (2024)", title: "Performances are plateauing, let's make the leaderboard steep again", locator: "p. 1", note: "why the Open LLM Leaderboard v2 chose GPQA"}
  - {paper: "Recchia, Mangat, Li and Krishnakumar (2025)", title: "FindTheFlaws: Annotated Errors for Detecting Flawed Reasoning and Scalable Oversight Research", locator: "pp. 10 and 31", note: "a second expert review of the diamond set"}
  - {paper: "Rein (2024)", title: "GPQA: baselines and analysis (GitHub repository idavidrein/gpqa)", locator: "p. 1", note: "the canary string, the password of the data and the form for corrections"}
  - {paper: "Epoch AI (2026)", title: "GPQA Diamond", locator: "p. 1", note: "the prompt and scoring of an independent evaluator"}
---

GPQA is a benchmark of hard multiple-choice questions in biology, physics and chemistry.
People who have or are studying for a PhD wrote the questions and checked them, and the questions were kept only if skilled people from other fields could not answer them, even with the web.
It was built for research on scalable oversight: how people can check the answers of AI systems that know more than they do.
Its smallest and strictest set, GPQA Diamond, has 198 questions.

## What it measures

A model reads a question with four answer choices and picks one; the score is the share of questions it gets right, so guessing at random gives 25%. {#c-random}
The questions test expert knowledge and reasoning in subfields such as organic chemistry, quantum mechanics and molecular biology.
The authors call them "Google-proof": skilled non-experts with unrestricted web access answer only 34% of them, against 65% for experts in the field. {#c-abstract}

The benchmark was made to test oversight, not only models.
Rein et al. want testbeds for getting true answers from models on questions where people cannot produce or check the answer on their own, the problem of scalable oversight. {#c-oversight}
For that, a question must be hard even for highly skilled non-experts, so that a non-expert cannot simply check the model's answer with a web search. {#c-non-expert-hard}
A high score on GPQA says that a model answers these questions well; it does not say that people can trust its answers to new questions in the same fields.

## Data

The questions come from 61 contractors hired through Upwork, who had completed or were in a PhD program in their field. {#c-upwork}
Each question went through four stages: an expert wrote it, a first expert from the same field answered it and gave feedback, the writer revised it, and then a second expert and three non-experts answered it. {#c-pipeline}
Writers were asked to phrase each question so that an expert could answer it even without the answer choices, and they wrote an explanation of why the right answer is right and why the others are plausible but wrong. {#c-free-response}
Most of the pay came from bonuses: a writer earned a bonus for each expert who answered correctly and for each non-expert who answered incorrectly. {#c-bonuses}

The non-experts were themselves experts, from the other fields.
Someone with expertise in any part of physics could not act as a non-expert on a physics question, and the same held for biology and chemistry. {#c-nearby}
They could use any web resource except a language model assistant, had to spend at least 15 minutes on each question, and spent 37 minutes on average. {#c-37-minutes}

The authors collected 564 questions and kept 18 as an unreleased held-out set, which leaves the 546 questions of GPQA Extended. {#c-564}
Two subsets are filtered from it by how the validators answered:

| Set | Questions | Rule | Expert accuracy | Non-expert accuracy |
|---|---|---|---|---|
| Extended | 546 | all released questions | 64.8% | 34.1% |
| Main | 448 | at least one of the two experts agrees and at most two of the three non-experts are correct | 71.9% | 30.4% |
| Diamond | 198 | both experts agree and at most one of the three non-experts is correct | 81.3% | 22.1% |

The table gives the counts and the accuracies of the validators from Rein et al. {#c-table2}
An expert "agrees" if they answered correctly, or if after seeing the answer they clearly explained their own mistake or showed that they understood the writer's explanation. {#c-agree}
The accuracies on the main and diamond sets are skewed by selection, because the validators' answers were used to choose those questions. {#c-selection}
Of the 546 questions, 105 are in biology, 227 in physics and 214 in chemistry. {#c-domains}

The data is on Hugging Face as four files, for the extended, main and diamond sets and the expert answers, behind a gate where users agree not to reveal examples in plain text or images online.
The licence and the source of each value are in the facts of this page.
The data carries a canary string, a superset of the canary string of BIG-bench: a marker that lets the builders of training corpora find the benchmark and leave it out. {#c-canary}

## Protocol

The paper's baselines are closed-book (the model sees only the question and the prompt) and open-book (the model can search the web). {#c-closed-open}
In the closed-book setting it tried zero-shot, few-shot, zero-shot chain-of-thought and few-shot chain-of-thought prompts, and the few-shot chain-of-thought examples use the explanations of the question writers as reasoning chains. {#c-prompts}

There is no single official prompt, and evaluators differ.
Epoch AI uses the prompt of the simple-evals library and gives no point for an answer that does not end in the form "ANSWER: LETTER". {#c-epoch-prompt}
That strict scoring is why some of Epoch AI's scores on GPQA Diamond are below the 25% of random guessing. {#c-below-random}
The Open LLM Leaderboard v2 reports a normalized score, in which the random baseline counts as 0 points. {#c-normalized}
So the version, the prompt, the answer format and the normalization are part of a result, and two scores that differ in any of them are not comparable.

## Known issues

**Small size.** GPQA is small, which gives little statistical power: the authors say that with 448 questions only large effects, such as 50% to 60% accuracy, can be detected with over 80% power. {#c-power}
GPQA Diamond, with 198 questions, is smaller still.

**Questions with wrong or arguable answers.** The authors estimate that 74% of the extended set has uncontroversially correct answers, a conservative estimate; the main and diamond sets were filtered by expert accuracy and should be higher. {#c-objectivity}
When the second expert answered wrongly, the authors classified the expert's feedback: in 12.0% of the 191 cases the expert said the answer was wrong, and in 7.9% that it was arguable. {#c-bad-answers}
A later review by Recchia et al. (2025) marked seven of the 198 diamond questions as unreliable, because their experts disagreed with the writers; for several, one of the original validators had also doubted the answer. {#c-seven}

**Leakage.** The questions are hard only if they are not in a model's training data.
The authors ask users not to reveal examples in plain text or images online, and evaluators follow the request: the post of the Open LLM Leaderboard shows no example for that reason. {#c-request}
The Open LLM Leaderboard chose GPQA in part because the data is gated, which should reduce the risk of contamination. {#c-gated}

**Who the non-experts are.** The non-experts are highly skilled, so their 34% is an upper bound for non-experts, and the authors advise later oversight experiments to measure non-expert accuracy again. {#c-skilled}
The experts were recruited on Upwork without regard to region or demographics, and the authors make no claim that GPQA represents the questions of scientific practice. {#c-bias}

**Corrections.** The authors take corrections to the examples through a form linked from their repository. {#c-corrections}

**Surface cues.** The authors trained models to pick the answer from the choices alone: a fine-tuned T5 did no better than chance, which suggests the choices hold no easy cues. {#c-answer-only}

## Reception

The Open LLM Leaderboard v2 of Hugging Face made GPQA one of its benchmarks, choosing it for the human review of its data and for its gating. {#c-olb}
Its authors found that GPQA and MATH Level 5 were its two hardest evaluations, with most models close to random on GPQA. {#c-olb-hard}
They also noted that Llama-3-70B-Instruct scored much lower on GPQA than its pretrained version, 4.92 against 19.67 in normalized score. {#c-llama}
Epoch AI runs its own evaluations on GPQA Diamond, with a log viewer for each run, and it cites OpenAI's comparison of the o1 model with PhD-level experts, who scored 69.7% on GPQA Diamond. {#c-experts-697}
Recchia et al. note that o3 was reported to reach 87.7% on GPQA Diamond, and argue that this would be hard if more than 12.3% of the questions were in error. {#c-o3}

## History

Rein et al. released GPQA in November 2023 with 448 questions in the main set. {#c-release}
Their strongest GPT-4-based baseline reached 39% (see What it measures); with few-shot chain-of-thought prompting, GPT-4 got between 38% and 40% on the three sets, and Llama-2-70B-chat and GPT-3.5-turbo-16k between 28% and 31%. {#c-baselines}
GPT-4 with web search answered slightly better on the extended set, 39.4%, but abstained on 37.2% of the main set, so the authors used the few-shot chain-of-thought answer whenever it abstained. {#c-search}
By domain, few-shot chain-of-thought GPT-4 got 58.1% on biology, 37.0% on physics and 31.8% on chemistry. {#c-gpt4-domain}
