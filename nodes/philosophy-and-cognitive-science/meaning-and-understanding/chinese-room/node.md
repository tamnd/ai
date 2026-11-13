---
tag: NEW-chinese-room
kind: argument
slug: chinese-room
name: Chinese room
aliases: [Chinese room argument, CRA]
area: philosophy-and-cognitive-science/meaning-and-understanding
level: core
summary: >-
  A thought experiment of John Searle (1980): a person who follows a rule book to answer Chinese questions without knowing Chinese, offered as a case where running a program does not give understanding.
history:
  year: 1980
  note: "Searle (1980), Behavioral and Brain Sciences 3(3), 417 to 457, with open peer commentary; restated as axioms in Searle (1990), Scientific American 262(1)."
same_as:
  wikidata: Q304726
  wikipedia: Chinese_room
sources:
  - {paper: "Searle (1980)", title: "Minds, brains, and programs", locator: "Behavioral and Brain Sciences 3(3), 417 to 424", note: "the original argument and the six named replies with Searle's answers"}
  - {paper: "Searle (1990)", title: "Is the Brain's Mind a Computer Program?", locator: "Scientific American 262(1), 26 to 31", note: "the axioms and conclusions, and the Chinese gym"}
  - {paper: "Churchland and Churchland (1990)", title: "Could a Machine Think?", locator: "Scientific American 262(1), 32 to 37", note: "the companion reply, with the luminous room"}
  - {paper: "Block (1980)", title: "What intuitions about homunculi don't show", locator: "Behavioral and Brain Sciences 3(3), 425", note: "a commentary on the role of intuition"}
  - {paper: "Dennett (1980)", title: "The milk of human intentionality", locator: "Behavioral and Brain Sciences 3(3), 428 to 430", note: "a commentary that calls the room an intuition pump"}
  - {paper: "Fodor (1980)", title: "Searle on what only brains can do", locator: "Behavioral and Brain Sciences 3(3), 431", note: "a commentary on the robot reply"}
  - {paper: "Hofstadter (1980)", title: "Reductionism and religion", locator: "Behavioral and Brain Sciences 3(3), 433", note: "a commentary on the brain simulator reply"}
  - {paper: "Pylyshyn (1980)", title: "The \"causal power\" of machines", locator: "Behavioral and Brain Sciences 3(3), 442", note: "a commentary with the replacement of brain cells by chips"}
  - {paper: "Harnad (1990)", title: "The Symbol Grounding Problem", locator: "Physica D 42, 335 to 346", note: "the grounding problem, stated with the Chinese room"}
  - {book: "Hofstadter and Dennett (eds.) (1981), The Mind's I", locator: "Chapter 22", note: "the reprint of Searle (1980) with a reflection by the editors"}
  - {book: "Preston and Bishop (eds.) (2002), Views into the Chinese Room", locator: "the whole volume", note: "new essays for and against the argument"}
  - {url: "https://plato.stanford.edu/archives/fall2025/entries/chinese-room/", title: "Cole, The Chinese Room Argument, Stanford Encyclopedia of Philosophy", locator: "Sections 1 to 5", note: "the survey of the replies and the later debate"}
  - {paper: "Bender and Koller (2020)", title: "Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data", locator: "ACL 2020, Sections 3 and 4"}
  - {paper: "Harnad (2024)", title: "Language writ large: LLMs, ChatGPT, meaning, and understanding", locator: "Frontiers in Artificial Intelligence 7, the abstract"}
  - {paper: "Floridi (2023)", title: "AI as Agency Without Intelligence: on ChatGPT, Large Language Models, and Other Generative Models", locator: "Philosophy & Technology 36, p. 1", note: "an editorial"}
  - {paper: "Piantadosi and Hill (2022)", title: "Meaning without reference in large language models", locator: "arXiv 2208.02957, the abstract"}
  - {paper: "Søgaard (2022)", title: "Understanding models understanding language", locator: "Synthese 200, Section 3"}
  - {paper: "Pavlick (2023)", title: "Symbols and grounding in large language models", locator: "Philosophical Transactions of the Royal Society A 381, Sections 3 and 4"}
  - {paper: "Mitchell and Krakauer (2023)", title: "The debate over understanding in AI's large language models", locator: "PNAS 120(13), the abstract", note: "a survey of the debate"}
premises:
  - "Computer programs are formal: a program is defined by the manipulation of symbols by their form alone (Searle 1990, axiom 1)."
  - "Human minds have mental contents: their thoughts are about things (Searle 1990, axiom 2)."
  - "Syntax by itself is neither constitutive of nor sufficient for semantics (Searle 1990, axiom 3), the point that Searle says the Chinese room demonstrated."
conclusion: "Programs are neither constitutive of nor sufficient for minds (Searle 1990, conclusion 1)."
replies:
  - {name: "the systems reply", summary: "The person does not understand, but the whole system of person, rules and data does.", sources: [searle-1980-minds-brains-programs-bbs, cole-2024-chinese-room-sep]}
  - {name: "the virtual mind reply", summary: "Running the program may create a mind that is neither the person nor the physical system.", sources: [cole-2024-chinese-room-sep]}
  - {name: "the robot reply", summary: "A computer that runs a robot body with perception and action would understand.", sources: [searle-1980-minds-brains-programs-bbs, fodor-1980-searle-only-brains-bbs, cole-2024-chinese-room-sep]}
  - {name: "the brain simulator reply", summary: "A program that simulates the neuron firings of a Chinese speaker's brain would understand.", sources: [searle-1980-minds-brains-programs-bbs, hofstadter-1980-reductionism-religion-bbs, pylyshyn-1980-causal-power-machines-bbs]}
  - {name: "the combination reply", summary: "Of a robot with a brain-shaped computer that simulates every synapse, we would have to ascribe intentionality to the whole system.", sources: [searle-1980-minds-brains-programs-bbs]}
  - {name: "the other minds reply", summary: "We attribute understanding to other people by their behaviour, so we must attribute it to a computer that behaves the same.", sources: [searle-1980-minds-brains-programs-bbs, cole-2024-chinese-room-sep]}
  - {name: "the many mansions reply", summary: "Future devices may have whatever causal processes intentionality needs, and that will be artificial intelligence.", sources: [searle-1980-minds-brains-programs-bbs]}
  - {name: "the intuition reply", summary: "The argument rests on intuitions about what can think, and evidence can overrule such intuitions.", sources: [block-1980-intuitions-homunculi-bbs, dennett-1980-milk-human-intentionality-bbs, cole-2024-chinese-room-sep]}
  - {name: "the reply to axiom 3", summary: "Whether syntax suffices for semantics is the open question, so assuming that it does not begs the question.", sources: [churchland-1990-could-machine-think-sciam, cole-2024-chinese-room-sep]}
---

The Chinese room is a thought experiment of the philosopher John Searle, published in 1980 in The Behavioral and Brain Sciences. {#c-published-1980}
A person who knows no Chinese follows rules written in English to answer questions written in Chinese, and the answers pass for those of a native speaker.
Searle argues that the person does not understand Chinese, and that a computer that runs the same program understands no more than the person.
He aims the argument at the view he calls strong AI, that an appropriately programmed computer really is a mind.
It drew many replies, and later work cites it on both sides of the question whether large language models understand language.

## In plain words

Imagine a person locked in a room with a large book of rules written in English.
Cards with Chinese characters come in through a slot.
The rules say which characters to send back, and they go by the shapes of the characters alone.
To the people outside, the answers look like those of someone who reads Chinese.
The person inside still does not know what any character means.
Searle says that a computer program works the same way: it moves symbols by their shape, not by their meaning.
So, he concludes, running the right program is not enough for understanding.
Critics answer in several ways.
Some say the whole room understands even if the person does not.
Others say a program connected to a body and senses would understand.
Others say the case only pumps an intuition.

## The argument

Searle (1980) asks what a computer that runs a program for answering questions about stories would understand.
His model is the work of Roger Schank and Robert Abelson on programs that answer questions about stories, such as one about a man who is served a burnt hamburger in a restaurant. {#c-target-schank}
He imagines himself locked in a room with batches of Chinese writing and a set of rules in English that let him match Chinese symbols by their shapes. {#c-room-setup}
The people outside call the rules "the program", and after a while his answers in Chinese are indistinguishable from those of native Chinese speakers.
He then argues in two steps.

1. He does not understand a word of the Chinese stories, although his inputs and outputs are those of a native speaker. {#c-no-understanding}
2. So the computer and its program do not give sufficient conditions of understanding, because they run and there is no understanding. {#c-not-sufficient}

The abstract of the paper states the general thesis that instantiating a computer program is never by itself a sufficient condition of intentionality. {#c-never-sufficient}
It adds that any mechanism capable of producing intentionality must have causal powers equal to those of the brain. {#c-causal-powers}
Searle does not argue that no machine can think.
To the question "Could a machine think?" he answers that only a machine could think, namely brains and machines with internal causal powers equivalent to those of brains. {#c-machines-can-think}
His target is the thesis that a program is enough.
In his terms, the formal symbol manipulations of a program "have only a syntax but no semantics". {#c-syntax-no-semantics}
He also separates simulation from duplication: no one supposes that a computer simulation of a fire will burn the neighborhood down. {#c-simulation}

Searle (1990) restated the argument for a general audience as axioms and conclusions.

1. Computer programs are formal, or syntactic.
2. Human minds have mental contents, or semantics.
3. "Syntax by itself is neither constitutive of nor sufficient for semantics." {#c-axioms-1990}
4. Conclusion: programs are neither constitutive of nor sufficient for minds. {#c-conclusion-1990}

Searle presents axiom 3 as the point that the Chinese room demonstrated: having the symbols by themselves is not sufficient for having the semantics. {#c-room-axiom3}
He adds a fourth axiom, that brains cause minds, and concludes that any other system capable of causing minds would need causal powers at least equivalent to those of brains. {#c-axiom4}
Cole (2024, Section 3) reads the narrow argument as a reductio ad absurdum against strong AI, not against weak AI. {#c-reductio}

## Context

The argument addresses one question: can running a program, by itself, give a computer understanding or other mental states?
Searle (1980) separates two claims about artificial intelligence (AI). {#c-strong-weak}
On weak AI, the computer is a tool for the study of the mind, and Searle states that he has no objection to it, at least as far as the article is concerned.
On strong AI, the appropriately programmed computer really is a mind and can be said to understand.
The programs of Schank and others answered questions about stories in a way that some took as evidence of understanding, and Searle took them as his test case.
The journal printed the article with comments and criticisms by 27 cognitive science researchers, followed by replies from Searle. {#c-bbs-commentary}
Cole (2024, Section 4.1) reports that Searle discussed replies that he had come across in giving the argument in talks at various places. {#c-labels}
In the article, the replies carry the names of places, such as Berkeley, Stanford and Yale. {#c-place-labels}
Cole (2024, Section 2) traces arguments of the same shape to Leibniz' mill, Turing's paper machine and the Chinese nation. {#c-antecedents}

## Replies

The replies below follow the order of Searle (1980), which Cole (2024) also follows.
Cole (2024, Section 4.1) quotes Searle (1980) as calling the systems reply "perhaps the most common reply", and it comes first. {#c-most-common}
Each reply has the same parts: its form and its proponents, Searle's answer, and the main counter-reply, each with sources.

### The systems reply

**The reply.** The person in the room does not understand, but the person is part of a whole system of rules, scratch paper and data banks, and the system understands. {#c-systems-reply}
Cole (2024) names Ned Block as one of the first to press it, with Jack Copeland, Daniel Dennett, Douglas Hofstadter, Jerry Fodor, John Haugeland, Ray Kurzweil and Georges Rey. {#c-systems-proponents}
**Searle's answer.** Let the person memorize the rules and the data and do the work in their head, outdoors; the person still understands nothing of the Chinese, and so neither does the system. {#c-systems-answer}
**Counter-reply.** Marvin Minsky and Aaron Sloman and Monica Croucher suggested a virtual mind reply when the argument first appeared: running the program may create a mind distinct from the person who runs it. {#c-virtual-mind}
Cole (2024, Section 4.1.1) lists Tim Maudlin, Don Perlis, David Chalmers and Block among those who endorse versions of it, and Roger Penrose and Stevan Harnad among its critics. {#c-virtual-mind-views}

### The robot reply

**The reply.** Put the computer inside a robot that perceives, walks, moves about and hammers nails, and the robot would have understanding and other mental states. {#c-robot-reply}
Cole (2024) lists Margaret Boden, Tim Crane, Dennett, Fodor, Harnad, Hans Moravec and Rey among those who endorse versions of it. {#c-robot-proponents}
**Searle's answer.** The reply concedes that cognition is not only formal symbol manipulation; and if Searle runs the robot's program from inside, he is the robot's homunculus and still understands nothing. {#c-robot-answer}
**Counter-reply.** Fodor (1980) grants that the same program as the brain's is not by itself enough for propositional attitudes, but finds Searle's treatment of the robot reply unconvincing, given the right kinds of causal linkages to the world. {#c-fodor-robot}
Tim Crane, as Cole (2024, Section 4.2) reports, argues that a person who also started acting in the world of Chinese people would come to learn what the symbols mean. {#c-crane}

### The brain simulator reply

**The reply.** A program that simulates the actual sequence of neuron firings at the synapses of a native Chinese speaker's brain would understand. {#c-brain-sim-reply}
**Searle's answer.** Let the man operate water pipes and valves, one connection for each synapse; neither the man nor the pipes understand Chinese, because the simulation copies only the formal structure of the firings. {#c-brain-sim-answer}
**Counter-reply.** Hofstadter (1980) answers that, once we accept that a tangle of cells gives rise to an "I", a water-pipe "I" should seem no more weird. {#c-hofstadter}
Zenon Pylyshyn (1980) asks what Searle's view implies if the cells of a brain are replaced one by one with chips that keep the same input-output function. {#c-pylyshyn}

### The combination reply

**The reply.** Join the three previous replies: a robot with a brain-shaped computer that simulates all the synapses of a human brain, with behaviour indistinguishable from human behaviour, is a system to which we would have to ascribe intentionality. {#c-combination}
Cole (2024, Section 4.3) names Georges Rey among those who argue that it is reasonable to attribute intentionality to such a system as a whole. {#c-combination-rey}
**Searle's answer.** He agrees that it would be rational and even irresistible to accept that such a robot had intentionality, as long as we knew nothing more about it. {#c-combination-answer}
But if we knew that its behaviour was the result of a formal program, and that the causal properties of its physical substance were irrelevant, we would abandon that assumption. {#c-combination-withdraw}
**Counter-reply.** Cole (2024, Section 4.3) remarks that on this answer one would cease to attribute understanding even to a spouse who was revealed to hide a silicon secret. {#c-spouse}

### The other minds reply

**The reply.** We know that other people understand Chinese only by their behaviour, so if a computer passes the same behavioural tests, we must attribute understanding to it too. {#c-other-minds}
**Searle's answer.** The question is not how we know that others have mental states, but what it is that we attribute when we attribute them. {#c-other-minds-answer}
**Counter-reply.** Critics that Cole (2024, Section 4.4) reports argue that if the evidence that humans understand is the same as the evidence that a robot understands, the presuppositions we make for our own species are not relevant, since presuppositions are sometimes false. {#c-other-minds-counter}

### The many mansions reply

**The reply.** Future technology may build devices with whatever causal processes intentionality needs, and that will be artificial intelligence. {#c-many-mansions}
**Searle's answer.** He has no objection, except that the reply trivializes strong AI by redefining it as whatever artificially produces and explains cognition. {#c-many-mansions-answer}
**Counter-reply.** Cole (2024) does not treat this reply in a section of its own, and the sources of this page give no counter-reply to Searle's answer.

### The intuition reply

**The reply.** Block (1980) argues that the argument depends for its force on intuitions that certain entities do not think, and that such intuitions can be overruled by evidence, as with the motion of the earth. {#c-block-intuitions}
Dennett (1980) calls the room an intuition pump, a device that provokes intuitions by variations on a thought experiment, and argues that the intuitions it gives are not to be trusted. {#c-dennett-pump}
**Searle's answer.** Searle (1980) concedes that there are degrees of understanding, but holds that there are clear cases of no understanding, and that the computer's understanding is not partial or incomplete but zero. {#c-searle-zero}
**Counter-reply.** Some critics, as Cole (2024, Section 4.5) reports, argue that Searle has pushed the reliance on intuition back into the room: perhaps he is mistaken, and he does understand, albeit unconsciously. {#c-intuition-back}

### The reply to axiom 3

**The reply.** Paul and Patricia Churchland (1990) agree that the man in the room does not understand Chinese. {#c-churchland-agree}
They argue that classical AI is unlikely to yield conscious machines, and that systems that mimic the brain might. {#c-churchland-brains}
They argue that axiom 3 is false, and that it begs the question for all its intuitive plausibility. {#c-luminous-room}
They compare it to an argument that light is not electromagnetic because a man who pumps a bar magnet in a dark room makes no visible light. {#c-luminous-magnet}
**Searle's answer.** Searle (1990) answers that the analogy with the luminous room fails, because formal symbols, unlike magnets, have no physical causal powers. {#c-searle-luminous-answer}
He takes the claim that uninterpreted formal symbols are not identical to mental contents as a logical truth, and he puts the Churchlands' analogy between syntax and electromagnetism to a dilemma. {#c-searle-logical-truth}
He also answers parallel, connectionist systems with the Chinese gym, a hall of monolingual English speakers who carry out the same operations and understand no Chinese. {#c-searle-gym}
**Counter-reply.** Critics that Cole (2024, Section 5.1) names, among them Boden, Dennett and Chalmers, argue that a computer running a program is not the same as syntax alone, but a complex causal system. {#c-not-syntax-alone}

## Influence

Cole (2024) calls the argument probably the most widely discussed philosophical argument in cognitive science since the Turing test. {#c-most-discussed}
Douglas Hofstadter and Daniel Dennett reprinted Searle's article as a chapter of their collection The Mind's I in 1981, followed by a reflection. {#c-minds-i}
John Preston and Mark Bishop edited a volume of new essays on Searle and artificial intelligence, Views into the Chinese Room, in 2002. {#c-preston-bishop}
Harnad (1990) used the room to state the symbol grounding problem: the meaning of the symbols of a formal system is parasitic on the meaning they have for us, not intrinsic to the system. {#c-harnad-grounding}
He proposed grounding symbols bottom-up in iconic and categorical representations that come from sensory input. {#c-harnad-hybrid}

### Large language models

The question here is narrower than the one Searle asked.
It is whether a model trained only on text can understand language, in the sense of connecting its words to what they mean.
Cole (2024, Section 4.6) asks whether large language models (LLMs) make any difference to the argument and its replies. {#c-sep-llm}
Melanie Mitchell and David Krakauer (2023) survey the debate over whether such models understand language. {#c-mitchell-krakauer}
Two positions are held in at least three sources each by distinct authors.
They are listed by the number of independent sources that hold them; the two have three each, and the tie is broken by the date of the earliest source.

**Position: form alone does not give meaning.**
The claim is that a model trained only on the form of language does not understand it.
Emily Bender and Alexander Koller (2020) argue that a system trained only on form has a priori no way to learn meaning. {#c-bk-form}
They cite the Chinese room as a reason to be extra careful in devising evaluations for machine understanding. {#c-bk-searle}
Harnad (2024) argues that ChatGPT does not understand, because it lacks direct sensorimotor grounding that connects its words to their referents. {#c-harnad-2024}
Luciano Floridi (2023), in an editorial, argues that large language models do not think, reason or understand. {#c-floridi}
The objections come from the other position: Piantadosi and Hill (2022) argue that the octopus test of Bender and Koller assumes that reference determines meaning, and they argue against that view of meaning. {#c-ph-reply}

**Position: models trained on text can capture meaning.**
The claim is that such a model can capture important aspects of meaning, and that the in-principle argument does not settle the question.
Steven Piantadosi and Felix Hill (2022) argue that LLMs likely capture important aspects of meaning through conceptual role, the relations among internal states. {#c-ph-conceptual-role}
Anders Søgaard (2022) separates inferential from referential semantics, offers the Color Radio thought experiment as a rebuttal of Searle, and argues that Searle conflates understanding with awareness of it. {#c-sogaard}
Ellie Pavlick (2023) answers the arguments from a lack of symbolic structure and a lack of grounding, and concludes that it is premature to claim intrinsic limits of LLMs. {#c-pavlick}
The objections come from the first position: Harnad (2024) points to the parasitism of indirect verbal grounding on direct sensorimotor grounding. {#c-harnad-parasitic}

> [!note]
> Of the six sources in this block, Bender and Koller (2020), Harnad (2024) and Søgaard (2022) discuss Searle's argument by name. Floridi (2023), Piantadosi and Hill (2022) and Pavlick (2023) address the question of meaning and grounding in LLMs without naming it.

## Related arguments

## References
