---
tag: NEW-alphago
kind: system
slug: alphago
name: AlphaGo
aliases: [AlphaGo Fan, AlphaGo Lee, AlphaGo Master]
area: games-and-multi-agent-systems/game-ai-milestones
level: core
summary: >-
  A Go program by DeepMind that guides Monte Carlo tree search with a policy network and a value network; its versions beat the professionals Fan Hui in 2015, Lee Sedol in 2016 and Ke Jie in 2017.
history:
  year: 2016
  note: "Silver et al. (2016), Nature 529, describe the version that played Fan Hui in October 2015; Silver et al. (2017), Nature 550, Methods, describe how the later versions differ."
same_as:
  wikipedia: AlphaGo
sources:
  - {paper: "Silver et al. (2016)", title: "Mastering the game of Go with deep neural networks and tree search", locator: "pp. 1 to 6, 9 and 15", note: "the parts, the training pipeline, the hardware, the tournament and the Fan Hui match; pages of the PDF file, whose pp. 1 to 6 are pp. 484 to 489 of Nature 529"}
  - {paper: "Silver et al. (2017)", title: "Mastering the game of Go without human knowledge", locator: "pp. 2, 8, 12, 21, 22 and 30", note: "the names and differences of the versions Fan, Lee, Master and Zero, their hardware and Elo ratings; the accepted manuscript"}
  - {paper: "Silver et al. (2018)", title: "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play", locator: "pp. 2, 4 and 5", note: "AlphaZero; the accepted manuscript"}
  - {url: "https://deepmind.google/research/alphago/", title: "Google DeepMind, AlphaGo", locator: "p. 1", note: "the developer's account of the Lee Sedol match and of moves 37 and 78"}
  - {url: "https://deepmind.google/blog/exploring-the-mysteries-of-go-with-alphago-and-chinas-top-players/", title: "Hassabis and Fan, Exploring the mysteries of Go with AlphaGo and China's top players (2017)", locator: "p. 1"}
  - {url: "https://deepmind.google/blog/alphagos-next-move/", title: "Hassabis and Silver, AlphaGo's next move (2017)", locator: "p. 1"}
  - {url: "https://deepmind.google/blog/alphago-zero-starting-from-scratch/", title: "Silver and Hassabis, AlphaGo Zero: Starting from scratch (2017)", locator: "p. 1"}
  - {url: "http://www.xinhuanet.com/english/2017-05/27/c_136320632.htm", title: "Xinhua, AlphaGo sweeps world's best Go-player Ke Jie 3-0 (27 May 2017)", locator: "the title page"}
developer: [deepmind]
years: "2015 to 2017"
parts: [policy network, value network, rollout policy, Monte Carlo tree search]
domain: Go
---

AlphaGo is a computer program that plays the board game Go, built at [[deepmind|DeepMind]].
It chooses each move with a tree search that two deep neural networks guide: a policy network proposes moves and a value network judges positions.
In October 2015 it beat the European champion Fan Hui by 5 games to 0.
Later versions beat Lee Sedol in March 2016 and Ke Jie in May 2017.
Its successors, AlphaGo Zero and AlphaZero, learned without human games.

## Overview

David Silver, Aja Huang and coauthors at Google DeepMind and Google introduced AlphaGo in a paper in Nature in 2016. {#c-paper}
They write that Go has long been viewed as the most challenging of classic games for artificial intelligence, owing to its enormous search space and hard-to-evaluate positions. {#c-go-hard}
They estimate about 250 legal moves per position and a game length of about 150 moves for Go, against about 35 and 80 for chess. {#c-breadth-depth}

AlphaGo was not one fixed program.
The AlphaGo Zero paper of 2017 calls the published program that played Fan Hui AlphaGo Fan, and the program that defeated Lee Sedol AlphaGo Lee. {#c-versions}
It calls a later program, which played online in January 2017, AlphaGo Master. {#c-master-name}
This page uses those names.
DeepMind writes that the version known as Master defeated the world number one, Ke Jie. {#c-ke-jie-master}

## How it works

An exhaustive search of the game tree of Go is infeasible. {#c-infeasible}
Silver et al. (2016) reduce it in two ways: a value function cuts the depth of the search, and a policy that samples moves cuts its breadth. {#c-two-principles}
The parts below are those of AlphaGo Fan.

**Supervised learning (SL) policy network.** The networks take the board position as a 19 by 19 image and build a representation of it with convolutional layers. {#c-board-image}
The SL policy network has 13 layers and was trained on 30 million positions from the KGS Go Server to predict the moves of human experts. {#c-sl-train}
On a held out test set it predicted expert moves with an accuracy of 57.0% using all input features. {#c-sl-policy}

**Rollout policy.** A faster rollout policy, a linear softmax of small pattern features, picks the moves of rollouts, the fast games played out to the end. {#c-rollout}
It reached an accuracy of 24.2% and took 2 μs per move, against 3 ms for the policy network. {#c-rollout-speed}

**Reinforcement learning (RL) policy network.** A network of the same structure starts from the weights of the SL network and is improved by policy gradient [[reinforcement-learning|reinforcement learning]]. {#c-rl-policy}
It plays games against a randomly selected earlier version of itself, and the reward is +1 for a win and −1 for a loss. {#c-rl-opponents}
Without any search, it won more than 80% of games against the SL policy network and 85% against Pachi, an open-source Go program. {#c-rl-results}

**Value network.** A network of similar architecture outputs one number: a prediction of the winner of games that the RL policy network plays against itself from a position. {#c-value}
Trained on positions of complete human games, it memorized the outcomes, so the authors made a self-play data set of 30 million distinct positions, each sampled from a separate game. {#c-value-data}

**Search.** AlphaGo combines the networks in a [[monte-carlo-tree-search|Monte Carlo tree search]] (MCTS). {#c-mcts}
Each step down the tree picks the move with the highest action value plus a bonus, which is proportional to the prior probability and decays with repeated visits. {#c-selection}
At a new leaf, the SL policy network gives the prior probabilities of the moves. {#c-priors}
The leaf is evaluated twice, by the value network and by the outcome of a rollout, and the two are mixed with a parameter $\lambda$. {#c-leaf-eval}
After the search, AlphaGo plays the most visited move from the root. {#c-most-visited}
The authors report that the SL policy network worked better for the priors than the stronger RL policy network, presumably because humans select a diverse beam of promising moves. {#c-sl-better}
A value network derived from the RL policy network worked better than one derived from the SL policy network. {#c-rl-value-better}

**Hardware.** The search runs simulations on CPUs and evaluates the networks in parallel on GPUs. {#c-async}
The final single-machine version used 40 search threads, 48 CPUs and 8 GPUs; a distributed version used multiple machines, 40 search threads, 1,202 CPUs and 176 GPUs. {#c-hardware}

## Development

Nature received the paper that reported the Fan Hui match on 11 November 2015 and accepted it on 5 January 2016. {#c-fan-hui-dates}

AlphaGo Lee, the version of March 2016, was previously unpublished; Silver et al. (2017) list the key differences from AlphaGo Fan. {#c-lee-unpublished}
Its value network was trained from the outcomes of fast games of self-play by AlphaGo, not by the policy network, and this procedure was iterated several times. {#c-lee-value}
Its policy and value networks were larger, with 12 convolutional layers of 256 planes, and were trained for more iterations. {#c-lee-larger}
It was distributed over many machines and used 48 tensor processing units (TPUs) instead of GPUs. {#c-lee-tpu}

AlphaGo Master, the version of 2017, uses the neural network architecture, the RL algorithm and the MCTS algorithm of AlphaGo Zero. {#c-master-arch}
Unlike AlphaGo Zero, it keeps the handcrafted features and the rollouts of AlphaGo Lee, and its training was initialised by supervised learning from human data. {#c-master-human}

In April 2017 DeepMind announced the Future of Go Summit in Wuzhen, China, for 23 to 27 May, with a match of three games between AlphaGo and Ke Jie. {#c-summit-plan}
In a post at the summit, DeepMind wrote that the Future of Go Summit was its final match event with AlphaGo. {#c-final-event}
DeepMind also published a set of 50 games that AlphaGo played against itself at full length time controls. {#c-50-games}

## Results and milestones

**Against other programs.** In an internal tournament with 5 s per move, single-machine AlphaGo won 494 of 495 games (99.8%) against other Go programs. {#c-tournament}
The distributed version won 77% of its games against single-machine AlphaGo and 100% against the other programs. {#c-distributed-wins}
On the Elo scale of that tournament, the distributed version had a rating of 3140 and the single-machine version 2890. {#c-elo-2016}
The scale was anchored to the rating of Fan Hui, 2,908 at the date of submission. {#c-elo-anchor}
On this scale a gap of 230 points means a 79% probability of winning. {#c-elo-gap}

**Fan Hui, October 2015.** Fan Hui was a professional 2 dan and the winner of the European Go championships of 2013, 2014 and 2015. {#c-fan-hui}
Over 5 to 9 October 2015 he played a formal match of five games against distributed AlphaGo, and AlphaGo won 5 to 0. {#c-fan-hui-match}
Five informal games were also played, and AlphaGo won them 3 to 2. {#c-informal}
Silver et al. (2016) report that this was the first time a computer Go program had defeated a human professional player without handicap in the full game of Go. {#c-first}

**Lee Sedol, March 2016.** AlphaGo Lee defeated Lee Sedol, the winner of 18 international titles, by 4 games to 1 in March 2016. {#c-lee}
DeepMind states that the 4 to 1 win took place in Seoul, South Korea, and was watched by over 200 million people. {#c-lee-seoul}

**Online games, January 2017.** AlphaGo Master defeated top human professional players 60 to 0 in online games in January 2017. {#c-master-60}
DeepMind describes them as unofficial online games played under the moniker Magister (Master). {#c-magister}

**Ke Jie, May 2017.** In its report of 27 May 2017, Xinhua says that AlphaGo swept Ke Jie 3 to 0, defeating him in Wuzhen for the third consecutive game. {#c-ke-jie}

**The versions compared.** Silver et al. (2017) rate the versions in one tournament at 5 s per move. {#c-agz-tournament}
In it, AlphaGo Fan and AlphaGo Lee were distributed over 176 GPUs and 48 TPUs, and AlphaGo Master and AlphaGo Zero each played on a single machine with 4 TPUs. {#c-agz-hardware}
Their Elo ratings were 3,144 for AlphaGo Fan, 3,739 for AlphaGo Lee, 4,858 for AlphaGo Master and 5,185 for AlphaGo Zero. {#c-agz-elo}
The scale includes the games against Fan Hui and Lee Sedol, because without them self-play bias makes the ratings of AlphaGo unrealistically high. {#c-self-play-bias}
The 3140 of 2016 and the 3,144 of 2017 come from two different tournaments, so the ratings of the two papers are not one series.

| Version | Match | Hardware in Silver et al. (2017) | Elo, Silver et al. (2017) |
|---|---|---|---|
| AlphaGo Fan | Fan Hui, 5 to 0, October 2015 | distributed, 176 GPUs | 3,144 |
| AlphaGo Lee | Lee Sedol, 4 to 1, March 2016 | distributed, 48 TPUs | 3,739 |
| AlphaGo Master | online, 60 to 0, January 2017; Ke Jie, 3 to 0, May 2017 | single machine, 4 TPUs | 4,858 |

## Impact and reception

Silver et al. (2016) write that during the match against Fan Hui, AlphaGo evaluated thousands of times fewer positions than Deep Blue did in its chess match against Kasparov. {#c-deep-blue}
They contrast Deep Blue's handcrafted evaluation function with AlphaGo's networks, trained from gameplay by general-purpose supervised and reinforcement learning. {#c-handcrafted}

DeepMind states that in game two AlphaGo played Move 37, a move with a 1 in 10,000 chance of being used, and that it helped AlphaGo win the game. {#c-move-37}
DeepMind also states that in game four Lee Sedol played Move 78, known as "God's Touch", which also had a 1 in 10,000 chance and helped him win. {#c-move-78}
The DeepMind page does not say how the 1 in 10,000 figures were computed.

DeepMind writes that the online games as Master influenced many of the top professionals of Go, Ke Jie among them. {#c-influence}

## Limitations and incidents

AlphaGo was not unbeaten: Fan Hui won two of the five informal games against it. {#c-losses}
Lee Sedol won game four of his match, which DeepMind says Move 78 helped him win. {#c-loss-lee}
The value network was trained for a komi of 7.5 points, so for handicap games the authors kept the komi and gave extra stones instead of the usual handicap rules. {#c-komi}
The rating of Fan Hui against the programs is approximate: the authors added his ten games to the tournament and ignored the differences in time controls. {#c-fan-elo-approx}

AlphaGo depended on human games for its SL policy network.
The AlphaGo Zero paper notes that expert data is often expensive, unreliable or unavailable, and may impose a ceiling on the performance of systems trained on it. {#c-human-data}

## Legacy

The list of successors is generated from the edges of later systems; the note below gives the two that the papers compare with AlphaGo.

**AlphaGo Zero** (Silver et al., 2017) learns by self-play reinforcement learning alone, starting from random play, with only the black and white stones as input. {#c-agz-diff}
It uses one network instead of separate policy and value networks, and a simpler tree search with no Monte Carlo rollouts. {#c-agz-diff2}
After 72 hours of training on a single machine with 4 TPUs, AlphaGo Zero beat AlphaGo Lee, which used 48 TPUs, by 100 games to 0. {#c-agz-100}
A larger instance, trained for about 40 days, beat AlphaGo Master by 89 games to 11 in a match of 100 games with 2 hour time controls. {#c-agz-89}

**AlphaZero** (Silver et al., 2018) is a more generic version of the AlphaGo Zero algorithm that plays chess, shogi and Go with the same algorithm and network architecture. {#c-az}
Its training used 5,000 first-generation TPUs to generate self-play games and 16 second-generation TPUs to train the networks, and lasted about 13 days in Go. {#c-az-training}
In Go it defeated the previously published AlphaGo Zero, also trained for 700,000 steps, and won 61% of the games. {#c-az-go}
DeepMind names AlphaZero, MuZero and AlphaDev as successors of AlphaGo. {#c-successors}

## References
