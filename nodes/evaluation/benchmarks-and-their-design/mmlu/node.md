---
tag: NEW-mmlu
kind: benchmark
slug: mmlu
name: MMLU
aliases: [Measuring Massive Multitask Language Understanding, Hendrycks Test, MMLU-Pro, MMLU-Redux]
area: evaluation/benchmarks-and-their-design
level: core
summary: >-
  Four-choice questions in 57 subjects, including elementary mathematics, US history, computer science and law, testing the world knowledge and problem solving of text models.
history:
  year: 2020
  note: "Released on arXiv on 7 September 2020 (Hendrycks et al.); published at ICLR 2021. MMLU-Pro (Wang et al.) and MMLU-Redux (Gema et al.) followed in June 2024."
sources:
  - {paper: "Hendrycks, Burns, Basart, Zou, Mazeika, Song and Steinhardt (2021)", title: "Measuring Massive Multitask Language Understanding", locator: "Sections 3 to 5", note: "the 57 subjects, the collection, the splits, the prompt, the human estimates and the first results"}
  - {paper: "Gema et al. (2024)", title: "Are We Done with MMLU?", locator: "Sections 2 and 3", note: "the error taxonomy, MMLU-Redux and the error rates by subject"}
  - {paper: "Wang et al. (2024)", title: "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark", locator: "Sections 1, 3 and 6", note: "saturation, the construction of MMLU-Pro and its comparison with MMLU"}
  - {paper: "Alzahrani et al. (2024)", title: "When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards", locator: "the abstract and Section 5", note: "how answer order, choice symbols and scoring change the rankings on MMLU"}
  - {paper: "Llama Team, AI @ Meta (2024)", title: "The Llama 3 Herd of Models", locator: "Section 5.2.1 and Table 2", note: "a 2024 report of MMLU and MMLU-Pro, with its setting"}
  - {paper: "TIGER-Lab (2024)", title: "MMLU-Pro (Hugging Face dataset card TIGER-Lab/MMLU-Pro)", locator: "p. 1", note: "the log of corrections to MMLU-Pro"}
---

MMLU is a benchmark of four-choice questions in 57 subjects, from elementary mathematics and US history to computer science and law. {#c-57}
Hendrycks et al. built it to measure what a text model knows and can solve across many fields, and they write that a model needs extensive world knowledge and problem-solving ability to score well. {#c-knowledge}
It was released in September 2020 and published at ICLR 2021. {#c-iclr}
It became a standard test of large language models; by 2024 the reported scores had stopped rising, and its errors had been measured, which led to the successors MMLU-Pro and MMLU-Redux (see Reception).

## What it measures

The subjects span the humanities, the social sciences, the hard sciences and other areas that are important for some people to learn. {#c-groups}
Some subjects come at several levels of difficulty, such as Elementary, High School, College and Professional, as with psychology. {#c-levels}
The paper groups the subjects into four broad disciplines for reporting: humanities, social science, STEM and other. {#c-disciplines}

MMLU tests knowledge gained in pretraining, not on a training set made for it.
The authors assume that models acquired the knowledge by reading large amounts of diverse text from the Internet, and they evaluate pretrained models zero-shot, few-shot or in a transfer setting. {#c-pretraining}
The test is text only: the authors note that the models of the time took no multimodal input. {#c-text-only}
They expected it to be "an enduring and informative goalpost", because a model would need to be well-rounded, know a lot about the world and solve problems at an expert level. {#c-goalpost}

A high score says that a model picks the keyed answer to these questions often; it does not separate what the model knows from what it remembers of the questions themselves (see [[data-contamination|data contamination]]).

## Data

Graduate and undergraduate students collected the questions by hand from freely available sources online, including practice questions for the Graduate Record Examination and the United States Medical Licensing Examination, questions for undergraduate courses, and questions for readers of Oxford University Press books. {#c-collected}
The paper does not document the collection in more detail, and Gema et al. note that the errors in MMLU are hard to trace because its annotation procedure is under-documented. {#c-under-documented}

The authors collected 15,908 questions and split them into a few-shot development set of 5 questions per subject, a validation set of 1,540 questions for choosing hyperparameters, and a test set of 14,079 questions. {#c-splits}
Each subject has at least 100 test questions, which is longer than most exams designed to assess people. {#c-100}
The released files hold fewer questions than the paper states: the Hugging Face copy, cais/mmlu, has 14,042 test, 1,531 validation and 285 development questions.
The counts, the licence and the hash of the data, with the source of each value, are in the facts of this page.

## Protocol

The score is the accuracy over all questions of all subjects. {#c-accuracy}
Each prompt begins with "The following are multiple choice questions (with answers) about [subject]."; for zero-shot evaluation the question follows, and for few-shot evaluation up to 5 solved examples come before it. {#c-prompt}
The examples are the 5 fixed questions per subject of the development set, so that every evaluation uses the same ones. {#c-dev5}
Every prompt ends with "Answer:", and the prediction is whichever of the tokens "A", "B", "C" and "D" the model gives the highest probability. {#c-answer}
Random guessing scores 25%. {#c-random}

Later evaluators changed this protocol in several ways, and the paper's own two settings already differ by the number of examples.
The Llama 3 report, for example, gives the macro average of the subject accuracies, 5-shot and without chain of thought, with every task asked as a generation task like the simple-evals library. {#c-llama-macro}
The macro average counts each of the 57 subjects equally, while the paper's accuracy counts each question equally, so the two are different metrics on the same questions.

The sources measure how much these choices matter:

- **Prompt wording.** Across 24 different but reasonable prompts, scores on MMLU generally moved by 4 to 5 points, and by up to 10.98 points, against about 2 points and at most 3.74 on MMLU-Pro. {#c-24-prompts}
- **Answer order.** On three MMLU subjects (college chemistry, college mathematics and global facts), shuffling the order of the answer choices changed the rank of 5 of 11 models, with a Kendall's tau of 0.564 between the two rankings. {#c-shuffle}
  With the correct answer moved to each position in turn, every tested model preferred particular positions or symbols, and the preferred one varied between models. {#c-position}
  Alzahrani et al. report that such small changes, including the order of the choices and the method of answer selection, move models by up to 8 places in the ranking. {#c-8-positions}
- **Scoring method.** Scoring by the probability of the answer symbol, the most common method, gives the highest accuracies and the strongest bias towards some answer symbols; cloze scoring, which never shows the choices, nearly removes that bias but gives low scores, and the authors recommend a hybrid of the two. {#c-scoring}
- **Few-shot examples.** Changing how informative the in-context examples are, or the wording of the instruction, changed scores slightly but not the rankings much; few-shot examples reduced, but did not remove, the sensitivity of the rankings. {#c-fewshot}
- **Chain of thought.** On MMLU, chain-of-thought prompting changed accuracy by +1.5 points for GPT-4o and -0.2 points for GPT-4-Turbo over a direct answer, against +19.1 and +15.3 points on MMLU-Pro. {#c-cot}

So the number of shots, the prompt, the answer format, chain of thought, the scoring method and the averaging are part of a result, and two MMLU scores that differ in any of them are not comparable.

## Known issues

**Wrong and unclear questions.** Gema et al. had 14 experts re-annotate 5,700 questions, 100 from each of the 57 subjects, and named the result MMLU-Redux. {#c-redux}
Their protocol sorts problems into unclear questions, unclear options, no correct answer, more than one correct answer and a wrong ground-truth answer. {#c-taxonomy}
From this sample they estimate that 6.49% of all MMLU questions contain errors. {#c-6-49}
The rate is far from even across subjects:

- 57% of the analysed Virology questions contain errors. {#c-virology}
- More than 20% of the examples in Logical Fallacies and College Chemistry are wrong, and more than 10% in Professional Law, Business Ethics, Formal Logic, Human Aging, Global Facts, Machine Learning, Miscellaneous and Public Relations. {#c-subjects}

Some errors come from how the questions were copied.
In College Chemistry, questions that spanned several lines in the source were often parsed wrongly, so that part of the question became option A and option D was lost. {#c-parsing}
Many wrong Virology labels are on simple questions, which suggests that they come from parsing the source, in most cases the Human Virology textbook. {#c-virology-cause}
Other questions lack context: Professional Law focuses on US law but does not say which jurisdiction a question assumes. {#c-law}
Gema et al. also find a US- and Western-centric bias in many subjects. {#c-western}
They did not change the MMLU labels: the answers that their annotators propose are not used to replace them. {#c-not-replaced}

The errors can change rankings.
In the Virology subject, Llama 3.1 Instruct Turbo (405B) ranked 16th on all questions but first on the questions without errors. {#c-ranking}
In Professional Law and Formal Logic, models did as well or better on the questions with errors, where they should not be able to match the keyed answer; Gema et al. read this as possible evidence that the questions were learned in pretraining. {#c-memorisation}
Contamination itself is covered on the page [[data-contamination|data contamination]].

**Released counts.** The test and validation files hold 37 and 9 fewer questions than the paper reports (see the facts of this page).

**Sensitivity to the protocol.** Scores and rankings change with the prompt, the order of the answers and the scoring method (see Protocol).

**Saturation.** Reported scores have clustered near 86% to 88% since 2023, so MMLU separates poorly the models that reach that range (see Reception).

## Reception

Wang et al. write that MMLU became the de facto standard for evaluating large language models, for its broad coverage and high quality. {#c-defacto}
They also report that it saturated: since GPT-4 reached 86.4% in March 2023, models published in early to mid 2024, such as GPT-4-Turbo, Gemini-1.5-Pro and LLaMA-3-400B, settled between 86% and 87%, and GPT-4o reached 87.4%. {#c-plateau}
Lower down, Gemini-1.5-Flash, Llama-3-70B-Instruct, Phi-3-medium-4k-instruct and Qwen1.5-110B all scored between 78% and 82%, a range that MMLU-Pro widens to about 10 points. {#c-cluster}
The Llama 3 report gives 87.3 for the finetuned Llama 3 405B on MMLU, 5-shot. {#c-llama-405}
Wang et al. suggest three reasons for the plateau: each question has only three wrong options, most questions test knowledge more than reasoning, and some questions are unanswerable or wrongly labelled, which lowers the ceiling. {#c-causes}

**MMLU-Pro.** Wang et al. released MMLU-Pro in June 2024 to make the test harder and more stable.
It has 12,032 questions in 14 disciplines, drawn from MMLU questions with the trivial and wrong ones removed, from a STEM website, from TheoremQA and from SciBench. {#c-pro-size}
To find trivial questions, they ran eight small models on MMLU and dropped every question that more than four of them answered correctly, 5,886 questions in all. {#c-pro-filter}
GPT-4-Turbo added six more options to each question, and experts then removed options that were also correct, so 83% of questions have ten options and the average is 9.47. {#c-pro-ten}
Accuracy on MMLU-Pro is 16 to 33 points lower than on MMLU. {#c-pro-drop}
In the paper, GPT-4o scored 72.6% on MMLU-Pro. {#c-pro-gpt4o}
Its authors kept correcting the data after release: the dataset card logs fixed answers, among them 15 in the medical domain in April 2025, and in January 2026 a fix of leading spaces in some answer options that could have served as a shortcut. {#c-pro-corrections}
So an MMLU-Pro score also depends on the revision of the data.
The Llama 3 report gives MMLU-Pro next to MMLU, 5-shot with chain of thought because of its focus on reasoning. {#c-llama-pro}

**MMLU-Redux.** The 5,700 re-annotated questions of Gema et al. are themselves a benchmark, published on Hugging Face as edinburgh-dawg/mmlu-redux-2.0 under CC BY 4.0. {#c-redux-data}
A score on all of its questions and a score on only its correct questions can rank models differently (see Known issues).

For a benchmark built to stay hard for skilled people with web access, see [[gpqa|GPQA]].

## History

Hendrycks et al. compared their test with the human scores they had: unspecialized workers on Amazon Mechanical Turk reached 34.5%. {#c-mturk}
For experts they made an estimate: from the 95th-percentile scores of human test-takers on the exams behind the test, with an educated guess where no such score was available, they put expert-level accuracy at about 89.8%. {#c-expert}

In the first results, the three smaller GPT-3 models were near chance, about 25%, and the 175-billion-parameter GPT-3 reached 43.9% few-shot. {#c-gpt3}
Zero-shot, the largest GPT-3 reached about 37.7%. {#c-zero-shot}
UnifiedQA, with 11 billion parameters and no further fine-tuning, reached 48.9%, better than few-shot GPT-3. {#c-unifiedqa}
GPT-3 ranged from 69% on US Foreign Policy to 26% on College Chemistry, and neither model reached expert level on any of the 57 subjects. {#c-lopsided}
For GPT-3, 9 of the 10 subjects with the lowest accuracy were STEM subjects that stress mathematics or calculation. {#c-procedural}
GPT-3 was also badly calibrated: zero-shot, its confidence and its accuracy differed by up to 24 points in some subjects. {#c-calibration}
The authors concluded that models matched expert-level performance (90%) on no subject, and on average were only starting to rise above chance. {#c-no-subject}

By 2024 reported scores of 86% to 88% were close to the authors' expert estimate of 89.8%, and the benchmark's errors, its sensitivity to the protocol and its saturation led to MMLU-Redux and MMLU-Pro (see Known issues and Reception).
