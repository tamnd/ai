---
id: sampling-cost
derives: [c-cost]
kind: bound
assumes: ["a network evaluation (NFE) costs the same at every noise level", "a backward pass costs about two forward passes"]
uses: [automatic-differentiation@al, knowledge-distillation@al]
status: full
check: sampling-cost.check.py
lean: null
sources:
  - {work: karras-2022-elucidating-design-space-diffusion, locator: "p. 1; p. 9, Table 2", note: "35 network evaluations per CIFAR-10 image and 79 at 64 x 64 for the Heun sampler"}
  - {work: song-2023-consistency-models, locator: "p. 4, Algorithm 1; p. 5, Algorithm 2", note: "multistep sampling with one evaluation per level, and the distillation step with one teacher solver step, the online model and the target"}
  - {work: song-2023-consistency-models, locator: "p. 8, Tables 1 and 2", note: "one-step and two-step FIDs of distillation, 3.55 and 2.93 on CIFAR-10 and 6.20 and 4.70 on ImageNet 64 x 64, against their EDM teachers at 35 and 79 evaluations, 2.04 and 2.44"}
---

### Statement

Let $C_f$ be the FLOPs of one forward call of the consistency model and $C_s$ those of one call of the teacher's score network.

1. Karras and colleagues' second-order sampler over $N$ noise levels, ending with a first-order step to $\sigma = 0$, costs $2N - 1$ teacher calls, 35 for $N = 18$; a consistency model sampled with Algorithm 1 at $k$ levels costs $k$ calls of $f$, one for one-step generation.
2. One sample of consistency distillation with a Heun teacher costs about $2C_s + 4C_f$: two teacher calls for the solver step between two positive noise levels, the online model's forward and backward pass, about $3C_f$, and the target's forward pass, $C_f$. Consistency training costs about $4C_f$, with no teacher.
3. For the network on this page, $f(x, t) = c_{\text{skip}}(t)x + c_{\text{out}}(t)F(c_{\text{in}}(t)x, t)$ with $F$ a tanh layer of width $H$ fed by a Fourier embedding of width $E$,
   $$ P = DH + EH + H + HD + D , \qquad C_f = 2B(DH + EH + HD) , $$ {#eq-cost}
   so the model of the tests, $D = 2$, $H = 64$, $E = 8$, has 834 parameters, and the same form on one CIFAR-10 image, $D = 3072$, $H = 1024$, $E = 64$, costs 12,713,984 FLOPs per call.

### Assumptions

1. Every call of a network costs the same, whatever the noise level. Used in steps 1 and 2.
2. A backward pass through the network costs about twice its forward pass. Used in step 2 only, which is a count of calls.

### Steps

1. The Heun step from $t_i$ to $t_{i+1} > 0$ evaluates the drift at $t_i$ and at the Euler point at $t_{i+1}$; the last step to 0 keeps only the first evaluation, since the drift is not evaluated at $\sigma = 0$. With $N$ positive levels there are $N - 1$ two-call steps and one one-call step, $2N - 1$ calls.
2. Algorithm 1 calls $f$ once at $T$ and once at each later $\tau_k$, so $k$ levels make $k$ calls.
3. Algorithm 2 draws $x_{t_{n+1}}$, takes one Heun step of the teacher to $t_n$, two calls of the score, then calls the online model at $t_{n+1}$ with a backward pass and the target at $t_n$ without one.
4. The matrix products of $F$ are $(B, D)(D, H)$, $(B, E)(E, H)$ and $(B, H)(H, D)$, each multiply-add counted as 2, and the weights are the three matrices and the two biases.

### Result and check

At $N = 18$ the distilled model's one-step sample costs $1/35$ of its teacher's, and Song and colleagues report one-step FIDs of 3.55 on CIFAR-10 and 6.20 on ImageNet $64 \times 64$, against 2.04 at 35 calls and 2.44 at 79 for the EDM teachers in their tables.
A training iteration of distillation costs about one and a half times one of diffusion training at the same batch, two teacher calls plus four student-equivalent passes against three, so the cost of distillation is set by the number of iterations, not by the sampler it replaces.
The check `sampling-cost.check.py` counts the calls of a score function inside a Heun sampler built from `teacher_step`, inside `sample` and inside `cd_pair`, and compares `param_count` and `flops_forward` with eq-cost and with the configurations of `shapes.yaml`.
