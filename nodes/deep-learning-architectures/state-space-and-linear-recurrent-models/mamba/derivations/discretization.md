---
id: discretization
derives: [c-disc-gap, c-disc-gate]
kind: identity
assumes: ["a diagonal A with negative entries", "Ā = exp(ΔA) in both rules; the zero-order hold takes B̄ = (ΔA)^{-1}(exp(ΔA) - 1) ΔB and the reference code takes B̄ = ΔB"]
uses: [matrix-exponential@m2, diagonal-matrix@m2]
status: full
check: discretization.check.py
lean: null
sources:
  - {work: gu-2023-mamba, locator: "p. 3", note: "eq. 4, the zero-order hold"}
  - {work: gu-2023-mamba, locator: "p. 8", note: "Theorem 1"}
  - {work: state-spaces-mamba-selective-scan-v2-3-2, locator: "p. 1", note: "selective_scan_ref, deltaB_u = delta B u"}
---

### Statement

For one diagonal entry $a < 0$ of $A$ and a step $\Delta > 0$, let $x = \Delta a$. Then:

1. $\bar{B}_{\text{zoh}} / \bar{B}_{\text{code}} = (e^{x} - 1)/x = 1 + x/2 + x^2/6 + O(x^3)$, which lies in $(0, 1)$; as $\Delta \to \infty$, $\bar{B}_{\text{zoh}} \to -B/a$ while $\bar{B}_{\text{code}} = \Delta B \to \infty$.
2. With $N = 1$, $a = -1$, $B = 1$ and $\Delta_t = \operatorname{softplus}(z_t)$, the rule of the code gives $h_t = (1 - g_t)h_{t-1} + \Delta_t x_t$ with $g_t = \sigma(z_t)$ and $\Delta_t = -\log(1 - g_t) \geq g_t$, while the zero-order hold gives the gate $h_t = (1 - g_t)h_{t-1} + g_t x_t$ of Theorem 1.

### Assumptions

1. $A$ is diagonal, so each entry discretizes alone. Used in step 1.
2. $a < 0$ and $\Delta > 0$, so $x < 0$. Used in steps 2 and 3.

### Steps

1. For a diagonal entry, $(\Delta a)^{-1}(e^{\Delta a} - 1)\Delta B = \frac{e^{x} - 1}{x}\Delta B$, so the ratio of the two rules is $(e^x - 1)/x = \sum_{k \geq 0} x^k/(k+1)! = 1 + x/2 + x^2/6 + \dots$
2. For $x < 0$, $e^x - 1 \in (-1, 0)$ and $e^x - 1 > x$ by convexity, so $(e^x - 1)/x \in (0, 1)$: the zero-order hold always weights the input less than the code's rule. As $\Delta \to \infty$, $e^{\Delta a} \to 0$ and $\bar{B}_{\text{zoh}} = (e^{\Delta a} - 1)B/a \to -B/a$.
3. With $\Delta = \operatorname{softplus}(z) = \log(1 + e^z)$, $e^{-\Delta} = 1/(1 + e^z) = 1 - \sigma(z) = 1 - g$. So $\bar{A} = e^{-\Delta} = 1 - g$ under both rules, and $\Delta = -\log(1 - g)$.
4. The zero-order hold gives $\bar{B} = (e^{-\Delta} - 1)/(-1) = 1 - e^{-\Delta} = g$; the code gives $\bar{B} = \Delta = -\log(1 - g) = \sum_{k \geq 1} g^k / k \geq g$, which tends to $\infty$ as $g \to 1$.

### Result and check

The two rules give the same $\bar{A}$ and differ in $\bar{B}$ by a factor that tends to 1 for small steps; the code's large steps still erase the state but scale the new input by $\Delta$. The check `discretization.check.py` compares the ratio with its series for small steps, checks the bound and the limit, and runs the recurrences of step 4 against the `discretize` and `selective_scan` of the NumPy reference.
