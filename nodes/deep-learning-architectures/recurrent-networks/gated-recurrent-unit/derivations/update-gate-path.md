---
id: update-gate-path
derives: [c-jacobian, c-copy-path, c-leaky]
kind: identity
assumes: ["one GRU layer with the reset gate after the recurrent matrix, in the convention where z = 1 keeps the old state", "for the leaky integrator, U = 0 and the update gate is a constant z in (0, 1)"]
uses: [hadamard-product@m2, backpropagation-through-time@al]
status: full
check: update-gate-path.check.py
lean: null
sources:
  - {work: cho-2014-learning-phrase-representations-rnn, locator: "Section 2.3"}
  - {work: chung-2014-empirical-evaluation-gated-recurrent, locator: "Section 3.3"}
---

### Statement

1. The Jacobian of one step of eq-update with respect to the previous state is
   $$ \frac{\partial h_t}{\partial h_{t-1}} = \operatorname{diag}(z_t) + \operatorname{diag}(1 - z_t)\, \frac{\partial n_t}{\partial h_{t-1}} + \operatorname{diag}(h_{t-1} - n_t)\, \frac{\partial z_t}{\partial h_{t-1}} . $$
2. If $U = 0$, the last two terms vanish and $\partial h_t / \partial h_{t-1} = \operatorname{diag}(z_t)$; over $k$ steps the gradient that reaches $h_{t-k}$ is the product of the $k$ update gates, unit by unit, with no matrix in between.
3. With $U = 0$ and a constant gate $z = \sigma(b)$ and a constant candidate $n$, the state is an exponential moving average, $h_t = z^t h_0 + (1 - z^t)\, n$. The weight of the input of $k$ steps ago is $(1 - z) z^k$, these weights sum to 1, and the mean lag is $z / (1 - z) = e^{b}$.

### Assumptions

1. $h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}$, where $z_t$, $r_t$ and $n_t$ depend on $h_{t-1}$ through eq-gates and eq-candidate. Used in step 1.
2. For statement 2, $U = 0$: the gates and the candidate depend on the input only. Used in step 2.
3. For statement 3, the input is constant, so $z$ and $n$ are the same at every step. Used in step 3.

### Steps

1. Differentiate eq-update coordinate by coordinate with the product rule. The term $z_t \odot h_{t-1}$ gives $\operatorname{diag}(z_t) + \operatorname{diag}(h_{t-1}) \partial z_t / \partial h_{t-1}$, and $(1 - z_t) \odot n_t$ gives $\operatorname{diag}(1 - z_t) \partial n_t / \partial h_{t-1} - \operatorname{diag}(n_t) \partial z_t / \partial h_{t-1}$. Adding them gives statement 1.
2. With $U = 0$, neither $z_t$ nor $n_t$ depends on $h_{t-1}$, so both derivatives are zero. By the chain rule of [[al:backpropagation-through-time|backpropagation through time]], $\partial h_t / \partial h_{t-k} = \prod_{j=0}^{k-1} \operatorname{diag}(z_{t-j})$. If the gates stay near 1 the product stays near 1, which is the shortcut path that Chung et al. describe: the error is back-propagated without too quickly vanishing when the gate is nearly saturated at 1. This is statement 2.
3. With $z$ and $n$ constant, $h_t - n = z (h_{t-1} - n)$, so $h_t - n = z^t (h_0 - n)$, which rearranges to $h_t = z^t h_0 + (1 - z^t) n$. Unrolled, $h_t = z^t h_0 + \sum_{k=0}^{t-1} (1 - z) z^k\, n$, and $\sum_{k \ge 0} (1 - z) z^k = 1$. The mean lag is $\sum_{k \ge 0} k (1 - z) z^k = z / (1 - z)$, and with $z = \sigma(b) = 1 / (1 + e^{-b})$ it is $e^{b}$. This is statement 3: a unit whose update gate stays near 1 integrates its input over many steps, the adaptive leaky integrator of Cho et al.

### Result and check

$$ \frac{\partial h_t}{\partial h_{t-k}} \Big|_{U = 0} = \prod_{j=0}^{k-1} \operatorname{diag}(z_{t-j}), \qquad h_t = z^t h_0 + (1 - z^t)\, n, \qquad \text{mean lag} = e^{b} . $$

For $b = 2$ and $n = \tanh(1)$, $z = 0.8808$, $h_1 = 0.0908$ and $\partial h_{10} / \partial h_0 = z^{10} = 0.2810$. The check `update-gate-path.check.py` compares the Jacobian of statement 1 with finite differences of `cell` for random parameters and both placements of the reset gate, checks that it is $\operatorname{diag}(z)$ when $U = 0$, and runs the leaky integrator against the closed form.
