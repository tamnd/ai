---
id: dilated-receptive-field
derives: [c-receptive-field]
kind: closed-form
assumes: ["causal convolutions with kernel size k and dilations d_1, ..., d_L, with residual connections", "the input enters shifted by one step", "d_1 = 1 and each d_l is at most 1 + (k - 1)(d_1 + ... + d_(l-1)), true of doubling dilations and of their repeats", "generic weights, so that no path cancels"]
uses: [autoregressive-model@m2, conditional-probability@m2]
status: full
check: dilated-receptive-field.check.py
lean: null
sources:
  - {work: van-den-oord-2016-wavenet-generative-model-raw-audio, locator: "pp. 2, 3", note: "causal convolutions, the receptive field of a plain stack and the doubling dilations whose block has a receptive field of 1024"}
---

### Statement

A causal convolution with kernel size $k$ and dilation $d$ computes $\mathrm{out}[t] = \sum_{j=0}^{k-1} W_j\, h[t - jd]$. Stack $L$ of them with residual connections, $h_l = h_{l-1} + f_l(h_{l-1})$, on the input delayed by one step, $h_0[t] = e(x_{t-1})$. Then the logits at time $t$ depend on exactly the values $x_{t-1}, x_{t-2}, \dots, x_{t-R}$, with

$$ R = 1 + (k - 1) \sum_{l=1}^{L} d_l , $$

and on nothing at or after $t$. For $k = 2$ and $d = 1, 2, 4, \dots, 512$ this gives $R = 1024$. Each repeat of the block adds 1023.

### Assumptions

1. The input shift makes the prediction at $t$ blind to $x_t$. Without it, the first layer would need a kernel that excludes its own position, like mask A. Used in step 1.
2. The condition on the dilations makes the set of reachable offsets an interval. With dilations $(2)$ alone and $k = 2$, the offsets would be $\{0, 2\}$, which leaves a gap. Used in step 3.
3. Elementwise nonlinearities and $1 \times 1$ layers do not mix time steps. Used in step 2.

### Steps

1. $h_0[t]$ depends on $x_{t-1}$ alone.
2. Layer $l$ reads $h_{l-1}$ at $t - j d_l$ for $j = 0, \dots, k - 1$, and the residual path reads it at $t$. If $h_{l-1}[t]$ depends on $h_0$ at the offsets $S_{l-1}$ before $t$, then $h_l[t]$ depends on $S_l = S_{l-1} + \{0, d_l, \dots, (k-1)d_l\}$, a Minkowski sum.
3. By induction $S_l = \{0, 1, \dots, (k - 1)\sum_{m \le l} d_m\}$. If $S_{l-1}$ is the interval $[0, s]$ with $s = (k-1)\sum_{m<l} d_m$ and $d_l \le s + 1$, then the shifted copies $[jd_l, jd_l + s]$ overlap or touch, and their union is $[0, s + (k-1)d_l]$.
4. The skip connections add the outputs of earlier layers, whose offsets lie inside $S_L$. The output layer is $1 \times 1$. So the logits at $t$ depend on $h_0$ at the offsets $S_L$, which means on $x_{t-1-o}$ for $o \in [0, (k-1)\sum_l d_l]$. That is $R$ values.
5. A plain stack, with all $d_l = 1$, has $R = 1 + (k-1)L$, linear in depth. A doubling block of $L$ layers with $k = 2$ has $R = 2^L$, exponential in depth at the same cost per layer.

### Result and check

$$ R = 1 + (k - 1) \sum_l d_l, \qquad R(k{=}2;\ 1, 2, \dots, 2^{9}) = 1024 . $$

At 16 kHz, a WaveNet with three such blocks sees $3070$ samples, about 0.19 s.

The check `dilated-receptive-field.check.py` perturbs one input of a random NumPy stack at a time, for six kernel and dilation choices. It confirms that the logits that change are exactly those at the next $R$ steps. It confirms the interval condition by enumerating the offset sets, and it finds the gap left by dilations that break the condition.
