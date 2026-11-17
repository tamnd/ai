---
id: decoupled-weight-decay
derives: [c-sgd-equivalence, c-adam-l2-not-decay, c-l2-first-step, c-torch-adamw-lr-wd, eq-adamw]
kind: identity
assumes: ["the update of Algorithm 1 of Kingma and Ba", "the L2 penalty is (lambda'/2) |theta|^2 added to the loss", "a schedule multiplier of 1"]
uses: [weight-decay@al, regularization@al]
status: full
check: decoupled-weight-decay.check.py
lean: null
sources:
  - {work: loshchilov-2017-decoupled-weight-decay, locator: "Section 2"}
  - {work: loshchilov-2017-decoupled-weight-decay, locator: "Algorithm 2"}
---

### Statement

Let $\lambda'$ be an L2 coefficient, so that the gradient of the penalised loss is $g + \lambda'\theta$, and let decoupled weight decay be the step $\theta \leftarrow (1 - \lambda)\theta$ taken beside the gradient step. Then

1. for SGD with learning rate $\eta$, decoupled decay with $\lambda$ equals L2 with $\lambda' = \lambda/\eta$;
2. for Adam, the L2 term enters $m_t$ and $v_t$ and is divided by $\sqrt{\hat v_t} + \epsilon$, so coordinates with a large $\hat v_t$ are decayed less; when the $\hat v_t$ of the coordinates differ, no single $\lambda'$ gives decoupled decay;
3. at the first step with $\epsilon = 0$, Adam with L2 takes the step $\eta \operatorname{sign}(g_1 + \lambda'\theta_0)$, so the penalty has no effect unless it flips the sign of the gradient;
4. one step of `torch.optim.AdamW`, $\theta \leftarrow \theta (1 - \eta w) - \eta\,\hat m_t/(\sqrt{\hat v_t} + \epsilon)$ (eq-adamw), is line 12 of Algorithm 2 of Loshchilov and Hutter with schedule multiplier 1, step size $\alpha = \eta$ and $\lambda = \eta w$. The decay per step is the product of the learning rate and the PyTorch `weight_decay` $w$.

### Assumptions

1. The Adam update is that of Algorithm 1 of Kingma and Ba, with the denominator $\sqrt{\hat v_t} + \epsilon$. Used in steps 2 to 4.
2. The L2 penalty is $\frac{\lambda'}{2}\lVert\theta\rVert^2$, whose gradient is $\lambda'\theta$. Used in steps 1 to 3.
3. The schedule multiplier $\eta_t$ of Algorithm 2 is 1. Used in step 4; with a schedule, both the step size and the decay are multiplied by $\eta_t$.

### Steps

1. SGD with L2 takes $\theta - \eta(g + \lambda'\theta) = (1 - \eta\lambda')\theta - \eta g$. With $\lambda' = \lambda/\eta$ this is $(1 - \lambda)\theta - \eta g$, the decoupled step. This is item 1, Proposition 1 of Loshchilov and Hutter.
2. Adam with L2 feeds $g + \lambda'\theta$ into eq-moments. Take $\beta_1 = 0$ to see one step alone: the step on coordinate $i$ is $\eta (g_i + \lambda'\theta_i)/(\sqrt{\hat v_{t,i}} + \epsilon)$, whose decay part is $\eta\lambda'\theta_i/(\sqrt{\hat v_{t,i}} + \epsilon)$. Decoupled decay removes $\lambda\theta_i$ from every coordinate. Matching the two needs $\lambda' = \lambda(\sqrt{\hat v_{t,i}} + \epsilon)/\eta$, which depends on $i$ unless all $\hat v_{t,i}$ are equal. This is item 2, the content of Proposition 2 of Loshchilov and Hutter; with $\beta_1 > 0$ the decay also mixes into the momentum, and the check covers that case.
3. At $t = 1$, $\hat m_1 = g_1 + \lambda'\theta_0$ and $\sqrt{\hat v_1} = |g_1 + \lambda'\theta_0|$, by step 4 of derivations/bias-correction.md, so the ratio is the sign of $g_1 + \lambda'\theta_0$. This is item 3.
4. Expand line 12 of Algorithm 2 with $\eta_t = 1$: $\theta_t = \theta_{t-1} - \alpha\,\hat m_t/(\sqrt{\hat v_t} + \epsilon) - \lambda\theta_{t-1}$. The PyTorch step expands to $\theta_{t-1} - \eta w\,\theta_{t-1} - \eta\,\hat m_t/(\sqrt{\hat v_t} + \epsilon)$. The two agree term by term for $\alpha = \eta$ and $\lambda = \eta w$. This is item 4.

### Result and check

$$ \theta_t = (1 - \eta w)\,\theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}, \qquad \lambda_{\text{Loshchilov--Hutter}} = \eta\, w_{\text{PyTorch}} . $$

The check `decoupled-weight-decay.check.py` compares SGD with L2 and SGD with decay over ten steps.
For Adam with $\beta_1 = 0.9$, it finds by bisection the L2 coefficient that matches decoupled decay on each of two coordinates whose gradients differ by a factor of 100, and the two coefficients differ by more than a factor of 10.
It checks item 3 on three cases, and it compares the PyTorch form of AdamW with Algorithm 2 over eight steps, each to a relative error under $10^{-12}$.
