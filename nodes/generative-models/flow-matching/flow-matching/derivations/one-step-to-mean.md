---
id: one-step-to-mean
derives: [c-one-step]
kind: closed-form
assumes: ["x_0 ~ N(0, I) independent of x_1 ~ q", "the path of conditional-ot-field", "the marginal field of marginal-field"]
uses: [conditional-expectation@m2, independence-of-random-variables@m2, expected-value@m2]
status: full
check: one-step-to-mean.check.py
lean: null
sources:
  - {work: esser-2024-scaling-rectified-flow-transformers, locator: "p. 4", note: "at the end points of the interval the optimal velocity prediction is the mean of the other distribution, so the target is hardest in the middle"}
---

### Statement

Under the independent coupling, the marginal field at $t = 0$ is

$$ u_0(\mathbf{x}) = \mathbb{E}[\mathbf{x}_1] - (1 - \sigma_{\min})\mathbf{x} . $$

So one Euler step of size 1 from $t = 0$ sends the noise point $\mathbf{x}$ to $\mathbb{E}[\mathbf{x}_1] + \sigma_{\min}\mathbf{x}$. When $\sigma_{\min} = 0$, every noise point goes to the mean of the data, and the whole distribution collapses onto a single point. A trained flow therefore needs more than one step unless its paths have first been straightened.

### Assumptions

1. At $t = 0$ the path is $\mathbf{x}_0$ itself, so conditioning on $\mathbf{x}_t = \mathbf{x}$ fixes $\mathbf{x}_0 = \mathbf{x}$. Used in step 1.
2. $\mathbf{x}_1$ is independent of $\mathbf{x}_0$. Used in step 2.

### Steps

1. By marginal-field, $u_0(\mathbf{x}) = \mathbb{E}[\mathbf{x}_1 - (1 - \sigma_{\min})\mathbf{x}_0 \mid \mathbf{x}_0 = \mathbf{x}] = \mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_0 = \mathbf{x}] - (1 - \sigma_{\min})\mathbf{x}$.
2. By independence, $\mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_0 = \mathbf{x}] = \mathbb{E}[\mathbf{x}_1]$.
3. The Euler step is $\mathbf{x} + 1 \cdot u_0(\mathbf{x}) = \mathbb{E}[\mathbf{x}_1] + \sigma_{\min}\mathbf{x}$.

### Result and check

$$ \mathbf{x} + u_0(\mathbf{x}) = \mathbb{E}[\mathbf{x}_1] + \sigma_{\min}\mathbf{x} . $$

The check `one-step-to-mean.check.py` uses a three-component mixture. For $\sigma_{\min} = 0$ and $\sigma_{\min} = 0.2$, it evaluates `mixture_field` at $t = 0$ on 50 noise points and runs `euler_sample` with one step. With $\sigma_{\min} = 0$, it checks that all 50 outputs coincide.
