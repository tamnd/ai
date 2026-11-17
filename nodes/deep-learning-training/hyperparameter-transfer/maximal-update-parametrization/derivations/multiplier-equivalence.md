---
id: multiplier-equivalence
derives: [c-lemma-sgd, c-lemma-adam, c-table8-readout]
kind: closed-form
assumes: ["a parameter W enters the network only through the product A W with a constant multiplier A", "W is initialized from N(0, B^2) and trained with learning rate C", "Adam is taken with epsilon = 0"]
uses: []
status: full
check: multiplier-equivalence.check.py
lean: null
sources:
  - {work: yang-2022-tensor-programs-v-hyperparameter-transfer, locator: "p. 46"}
  - {work: yang-2022-tensor-programs-v-hyperparameter-transfer, locator: "p. 23"}
  - {work: yang-2022-tensor-programs-v-hyperparameter-transfer, locator: "p. 5"}
---

### Statement

Let a parameter $W$ enter the network only as $A W$, with initialization $W_0 \sim \mathcal{N}(0, B^2)$ and learning rate $C$.
For any $\theta > 0$:

1. Under SGD, $(A, B, C) \to (A\theta, B/\theta, C/\theta^2)$ leaves the function computed at every step unchanged.
2. Under Adam with $\epsilon = 0$, $(A, B, C) \to (A\theta, B/\theta, C/\theta)$ does the same.
3. The $\mu$P readout of Table 3 (multiplier 1, initialization variance $1/n^2$, SGD rate $\eta/n$, Adam rate $\eta/n$) therefore equals the readout of Table 8: multiplier $1/n$, variance 1, and SGD rate $\eta n$ or Adam rate $\eta$.

### Assumptions

1. $W$ appears in the loss only through $A W$, so $\partial L / \partial W = A\, G$, where $G = \partial L / \partial (A W)$ depends only on the effective weight $A W$.
2. Both runs use the same random draw, scaled: $W_0' = W_0 / \theta$.
3. Adam's update is $-C\, \hat m / \sqrt{\hat v}$ with $\epsilon = 0$; any positive $\epsilon$ breaks exact invariance.

### Steps

1. Let $\widetilde W = A W$ be the effective weight. At initialization $A\theta \cdot W_0/\theta = A W_0$, so the two runs start with the same $\widetilde W$.
2. SGD. Suppose $\widetilde W_t$ agree. The gradient of the primed parameter is $A\theta\, G$, and its update is $-(C/\theta^2)(A\theta\, G) = -(C A/\theta)\, G$. The effective weight changes by $A\theta \cdot (-(C A/\theta) G) = -C A^2 G$, the same as in the unprimed run. By induction the effective weights agree at every step.
3. Adam. The moments of a gradient scaled by $\theta$ scale as $\hat m \to \theta \hat m$ and $\sqrt{\hat v} \to \theta \sqrt{\hat v}$, so $\hat m / \sqrt{\hat v}$ is unchanged. The update of the parameter is $-(C/\theta)\, u$, and the effective weight changes by $A\theta \cdot (C/\theta)\, u = A C u$, as before. This is Lemma J.1 of Tensor Programs V (p. 46).
4. Readout. Start from Table 3 (p. 5): $A = 1$, $B = 1/n$, SGD rate $C = \eta/n$, Adam rate $C = \eta/n$. Take $\theta = 1/n$. Then $A' = 1/n$, $B' = (1/n)/(1/n) = 1$, SGD rate $(\eta/n)/(1/n^2) = \eta n$, and Adam rate $(\eta/n)/(1/n) = \eta$. These are the output-weight entries of Table 8 (p. 23), which lets input and output weights be tied.

### Result and check

The check trains a two-hidden-layer tanh MLP with a multiplier on every layer for six steps, from one fixed draw, under SGD and under Adam with $\epsilon = 0$. With five random per-layer values of $\theta$ the outputs match the base run to $10^{-8}$ relative. Swapping the powers, $C/\theta$ under SGD and $C/\theta^2$ under Adam, changes the outputs after the first step. Moving the readout to the Table 8 form ($\theta = 1/n$) gives multiplier $1/n$, variance 1 and rate $\eta n$ or $\eta$, and reproduces the base outputs.
