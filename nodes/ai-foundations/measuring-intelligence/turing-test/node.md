---
tag: FWFEPZ
kind: concept
slug: turing-test
name: Turing test
aliases: [imitation game, Turing's test]
area: ai-foundations/measuring-intelligence
level: core
summary: >-
  A test that Alan Turing proposed in 1950: an interrogator questions a hidden person and a hidden machine in writing, and the machine does well if the interrogator cannot reliably tell which is which.
related_to: [chinese-room]
history:
  year: 1950
  people: [Alan Turing]
  note: "Turing (1950), Computing Machinery and Intelligence, Mind 59(236), 433 to 460, under the name of the imitation game."
same_as:
  wikidata: Q189223
  wikipedia: Turing_test
sources:
  - {paper: "Turing (1950)", title: "Computing Machinery and Intelligence", locator: "Mind 59(236), 433 to 460, Sections 1 to 7", note: "the imitation game, the prediction, the nine objections and the learning machine"}
  - {url: "https://plato.stanford.edu/archives/fall2025/entries/turing-test/", title: "Oppy and Dowe, The Turing Test, Stanford Encyclopedia of Philosophy", locator: "Sections 1 to 6", note: "the readings of the game, the main criticisms and the alternative tests"}
  - {paper: "Searle (1980)", title: "Minds, brains, and programs", locator: "Behavioral and Brain Sciences 3(3), p. 419", note: "the Chinese room as a case of two systems that both pass the test"}
  - {paper: "Shieber (1994)", title: "Lessons from a Restricted Turing Test", locator: "Communications of the ACM 37(6), 70 to 78; arXiv cmp-lg/9404002", note: "the first Loebner Prize competition of 1991 and a critique of it"}
  - {paper: "Jones and Bergen (2025)", title: "Large Language Models Pass the Turing Test", locator: "arXiv 2503.23674, Sections 1 to 3", note: "two randomized, pre-registered three-party tests of four systems"}
---

The Turing test is a test of machine intelligence that Alan Turing proposed in 1950, under the name of the imitation game. {#c-proposed-1950}
An interrogator questions two hidden players in writing, one a person and one a machine, and then says which is which.
Turing set aside the question "Can machines think?" and asked instead whether a machine could play this game well. {#c-replaces-question}
Philosophers still dispute what passing the test would establish.
In 2025, in a controlled study, a large language model with a suitable prompt was picked as the human in 73% of games.

## In plain words

You sit at a screen and chat by text with two partners you cannot see.
One is a person and the other is a computer program.
Both try to convince you that they are the person.
After a few minutes you must say which one is the person.
If, over many such games, people pick the program about as often as they pick the person, the program passes.
Turing's point was that this question can be settled by trying it, while the question "can a machine think?" turns on what the words mean.
The test looks only at what the program says, not at how it works inside, and that is the root of most objections to it.

## The idea

Turing (1950, Section 1) begins with the question whether machines can think, and observes that if the meanings of "machine" and "think" were found by surveying how people use the words, the answer would have to be sought in "a statistical survey such as a Gallup poll", which he calls absurd. {#c-gallup}
He puts a game in its place. An interrogator C sits in another room and questions two hidden players, a man A and a woman B, and must work out which one is the man. {#c-original-game}
A tries to lead C to the wrong answer and B tries to help, and the answers are typewritten so that the voice gives nothing away. {#c-typewritten}
Turing then asks "What will happen when a machine takes the part of A in this game?", and whether the interrogator will then be fooled as often as in the game of a man against a woman. {#c-machine-takes-a}
He says that this new question replaces the original one.

Turing (1950, Section 2) argues that the game draws "a fairly sharp line between the physical and the intellectual capacities of a man". {#c-sharp-line}
In his words, the machine is not penalised for "its inability to shine in beauty competitions", nor a man "for losing in a race against an aeroplane". {#c-beauty}
He admits only digital computers to the game (Section 3), and because a digital computer can mimic any discrete-state machine, he reduces the question to whether one fixed computer, with enough storage, speed and a suitable program, could play the part of A well (Section 5). {#c-digital-only}

Turing (1950, Section 6) then states what he believes.
He expects that in about fifty years computers with a storage of about $10^9$ could be programmed to play the game so well that an average interrogator would pick out the machine correctly no more than 70 per cent of the time after five minutes of questions. {#c-prediction}
He calls the original question "too meaningless to deserve discussion", and he expects that by the end of the century one would be able to "speak of machines thinking without expecting to be contradicted". {#c-too-meaningless}

Oppy and Dowe (2021, Section 3.3) separate two claims that discussions of the test often run together. {#c-two-claims}
The Turing Test Claim is that the game is a good test for intelligence.
The Thinking Machine Claim is that a suitably programmed computer could pass it.
An objection to one need not be an objection to the other; Oppy and Dowe count Searle's argument as an objection to the second. {#c-searle-machine-claim}

The test is also less fixed than it looks.
Turing does not say how a given rate of success should raise our confidence that the machine is intelligent, and Oppy and Dowe (2021, Section 4.4) name the variables that matter: the length of the questioning, the skill of the interrogator and of the human player, and the number of independent games. {#c-underspecified}
They suggest that judges could give probabilities instead of yes or no answers, which keeps the character of the test and allows a finer statistical analysis. {#c-probabilistic-verdicts}

### Turing's answers to objections

Turing (1950, Section 6) answers nine objections to the view that machines could think: the theological objection, the "heads in the sand" objection, the mathematical objection, the argument from consciousness, the arguments from various disabilities, Lady Lovelace's objection, and the arguments from the continuity of the nervous system, from the informality of behaviour and from extrasensory perception. {#c-nine-objections}
Two of his answers are often cited.
To the argument from consciousness, he replies that the only way to be sure a machine thinks would be to be the machine, which is the solipsist view, and that it is usual instead to have "the polite convention that everyone thinks". {#c-polite-convention}
To Lady Lovelace's remark that the Analytical Engine "has no pretensions to originate anything", he replies with learning: instead of programming an adult mind, one could program a machine that simulates a child's mind and then educate it (Section 7). {#c-child-machine}

## Formal statement

A single game has an interrogator $C$ and two witnesses, a machine $M$ and a human $H$, placed at random behind the labels $X$ and $Y$.
After a conversation of fixed length $T$, $C$ names the witness it takes to be human.
Let $w$ be the probability that $C$ names $M$.
Jones and Bergen (2025, Section 2.1) call $w$ the win rate of the machine, and the interrogator accuracy, the probability of naming $H$, is $1 - w$, since each verdict is a choice between the two. {#c-win-rate}

Two thresholds appear in the literature.

1. Turing's prediction, read as a criterion, asks that an average interrogator, with $T$ = 5 minutes, is right with probability at most $0.7$, that is $1 - w \le 0.7$, or $w \ge 0.3$.
2. The criterion of indistinguishability asks that $w$ is not significantly different from $0.5$, the rate at which a human witness is picked when both witnesses are human or when $C$ guesses.

The second criterion asks for a null result, and a null result can also come from an interrogator who guesses at random.
Jones and Bergen (2025, Section 3.1) note this problem and answer it with a weak program in the same design: if interrogators pick it significantly less often than chance, the design can tell a human from a machine. {#c-jb-eliza-check}

## Examples

**Turing's sample dialogue.**
Turing (1950, Section 2) gives a short exchange to illustrate the questions an interrogator might ask. {#c-sample-dialogue}
Asked for a sonnet on the subject of the Forth Bridge, the witness answers "Count me out on this one. I never could write poetry."
Asked to add 34957 to 70764, it pauses about 30 seconds and answers 105621.
Asked a chess question, it pauses 15 seconds and gives a mating move.
The true sum is $34957 + 70764 = 105721$, so the printed answer is off by 100.
The exchange has a slow and wrong sum and a refusal to write poetry, and none of these counts against the witness in the game, since a person might do the same.

**The first Loebner Prize, 1991.**
Shieber (1994) reports on the first competition for the Loebner Prize, held on 8 November 1991 at the Computer Museum in Boston and billed as "the first actual administration of the Turing test". {#c-loebner-1991}
There were ten hidden agents, six of them programs, and ten judges chosen from the general public through a newspaper advertisement. {#c-loebner-setup}
Each conversation kept to one topic chosen in advance, and the rules ruled out "trickery or guile" on the part of the judges. {#c-loebner-rules}
The judges ranked the two human confederates as more human than any program, but five judges ranked the winning program, by Joseph Weintraub on whimsical conversation, as human, and computers were classed as humans eight times in all. {#c-loebner-result}
Shieber (1994) argues that the contest lacked a clear purpose, that its design ruled out any useful result, and that the technology of the time was not ready for it. {#c-shieber-verdict}

**Jones and Bergen, 2025.**
Jones and Bergen (2025) ran two randomized, pre-registered three-party tests, one with 126 undergraduates of the University of California, San Diego and one with 158 participants recruited online through Prolific. {#c-jb-design}
In each game an interrogator chatted for up to five minutes at the same time with a human and with one of four systems: ELIZA, GPT-4o, LLaMa-3.1-405B and GPT-4.5. {#c-jb-setup}
With a prompt to adopt a humanlike persona, GPT-4.5 was picked as the human 73% of the time and LLaMa-3.1-405B 56% of the time, while ELIZA and GPT-4o were picked 23% and 21% of the time. {#c-jb-rates}
Without the persona prompt, the same models did not pass in both studies. {#c-jb-no-persona}
Against Turing's figure, a win rate of $0.73$ means an interrogator accuracy of $1 - 0.73 = 0.27$, far below $0.7$; for ELIZA the accuracy is $1 - 0.23 = 0.77$, above it.
Against the criterion of indistinguishability, GPT-4.5 was picked significantly more often than chance, which is more than the criterion asks.
Only 12% of the participants asked the knowledge and reasoning questions of the kind Turing had in mind, such as chess or arithmetic; most judged by social and emotional cues. {#c-jb-12pct}

## Where it is used

## Related ideas

**The Chinese room.**
Searle's Chinese room is a case built to pass the Turing test without understanding.
Searle (1980, p. 419) says that in the room he can fool native Chinese speakers and so pass the test, and that "one of the points at issue is the adequacy of the Turing test". {#c-searle-adequacy}
His example is meant to give two systems, himself in English and himself in the room, that both pass the test while only one understands.

**Blockhead.**
Ned Block imagines a creature that looks human and is run by a vast look-up tree with a stored response for every possible input at each stage of its life. {#c-blockhead}
If such a creature is logically possible and not intelligent, then passing the test is not a logically sufficient condition for intelligence. {#c-blockhead-sufficiency}
Oppy and Dowe (2021, Section 4.2) see two ways to answer: deny that Blockhead is logically possible, or hold that it would be intelligent after all. {#c-blockhead-replies}

**Tests said to be too hard, too narrow or too easy.**
Robert French (1990) argues that the test is too hard, because questions that probe low-level human cognition, such as associative priming or the rating of made-up words, would unmask any machine without a human subcognitive substrate. {#c-french}
Stevan Harnad proposes a Total Turing Test that asks for a robot with human sensorimotor capacities, not only text. {#c-harnad-ttt}
Selmer Bringsjord and colleagues (2001) propose a Lovelace test, which an agent passes when it produces an output, can repeat the process, and its designer cannot explain how it did so from its architecture, knowledge and core functions. {#c-lovelace-test}

**The test as a goal for research.**
Patrick Hayes and Kenneth Ford (1995) argue that the test should be considered harmful as a goal for AI, among other reasons because it sets out to confirm a null hypothesis. {#c-hayes-ford}
Oppy and Dowe (2021, Section 5.4) reject several of their claims, but agree that they are right to doubt that the test should define the goal of AI research, and add that Turing did not suppose it did. {#c-hayes-ford-sep}

## Common confusions

**A test of gender, or a test of being human.**
Read literally, Turing's text could make the machine imitate a woman.
Oppy and Dowe (2021, Section 3.1) side with Copeland, Piccinini and Moor, who argue that Turing intended the standard reading, in which the machine tries to pass for a human being and the other player is a human of either sex. {#c-standard-reading}
Susan Sterrett dissents from this reading. {#c-sterrett}

**Passing is not a proof of thought.**
Turing proposed the test as a better question, not as a definition of thinking.
Oppy and Dowe (2021, Section 4.4) read his prediction as probabilistic, and passing as evidence that raises the probability of intelligence by some amount, not as a proof of it. {#c-probabilistic-support}
Blockhead and the Chinese room are aimed at stronger readings, on which passing would be logically sufficient for intelligence or for understanding.

**What Turing predicted for the year 2000.**
Turing did not predict that machines would be indistinguishable from people by 2000.
He predicted an interrogator accuracy of at most 70 per cent after five minutes, a weaker bar.
Most commentators hold that even this prediction failed for 2000, while Copeland (2000) argues that "about fifty years" is not "exactly fifty years" and that elsewhere Turing estimated at least 100 years for an unrestricted test. {#c-prediction-2000}

**The Loebner Prize is not Turing's test.**
The 1991 competition limited each conversation to one topic and asked the judges to converse naturally, without trickery.
Shieber (1994) calls these two rules, on topic and tenor, quite problematic. {#c-loebner-problematic}
A program that fools judges under such rules has passed a restricted test, not the open conversation that Turing described.

## Formal proof

## History

Oppy and Dowe (2021) report that some find the test prefigured in Descartes' Discourse on the Method; Copeland (2000) finds an anticipation in the writings of the Cartesian de Cordemoy in 1668, and Abramson (2011) gives archival evidence that Turing knew of Descartes' language test when he wrote his paper. {#c-descartes}
Turing published the imitation game in the philosophy journal Mind in 1950.
Searle (1980) turned the Chinese room on the adequacy of the test, and Block (1981) gave the Blockhead argument against it as a sufficient condition.
The first Loebner Prize competition in 1991 was billed as its first actual administration.
Jones and Bergen (2025) describe their results as "the first empirical evidence that any artificial system passes a standard three-party Turing test". {#c-jb-first}

## References
