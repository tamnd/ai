---
id: steps-examples-tradeoff
derives: [c-steps-form, c-tradeoff-product, c-bcrit-average, c-bcrit-twice]
kind: closed-form
assumes: ["each step at batch B lowers the loss by Delta L_max / (1 + B_noise / B), the result of optimal-step-noise-scale", "the batch size is fixed during the run", "Delta L_max and B_noise depend only on the current loss, not on the batch size used to reach it"]
uses: [optimal-step-noise-scale]
status: full
check: steps-examples-tradeoff.check.py
lean: null
sources:
  - {work: mccandlish-2018-empirical-model-large-batch-training, locator: "p. 7"}
---

### Statement

Suppose each step at batch size $B$ lowers the loss by $\Delta L_{\max}(L)/(1 + B_{\mathrm{noise}}(L)/B)$, the optimal improvement of `optimal-step-noise-scale`, with both quantities functions of the current loss $L$.
Let $S(B)$ be the number of steps to go from $L_0$ to a target $L_1$ and $E(B) = B\,S(B)$ the number of examples processed. Then:

(i) $S(B) = S_{\min} + E_{\min}/B$, where $S_{\min} = \int_{L_1}^{L_0} dL / \Delta L_{\max}$ is the number of steps at infinite batch and $E_{\min} = \int_{L_1}^{L_0} B_{\mathrm{noise}}\, dL / \Delta L_{\max}$ the number of examples as $B \to 0$;

(ii) $(S/S_{\min} - 1)(E/E_{\min} - 1) = 1$ for every $B$;

(iii) $B_{\mathrm{crit}} = E_{\min}/S_{\min}$ is the average of $B_{\mathrm{noise}}$ over the steps of the infinite-batch run;

(iv) at $B = B_{\mathrm{crit}}$ the run takes $2S_{\min}$ steps and $2E_{\min}$ examples.

### Assumptions

1. The per-step improvement is that of `optimal-step-noise-scale`, with the step size retuned at every step; the loss falls smoothly enough to count steps by an integral.
2. $B$ is constant during the run.
3. The run at batch $B$ passes through the same loss levels, with the same $\Delta L_{\max}$ and $B_{\mathrm{noise}}$ at each level, as runs at other batch sizes.

### Steps

1. The number of steps spent between $L$ and $L - dL$ is $dL\, (1 + B_{\mathrm{noise}}/B) / \Delta L_{\max}$. Integrating gives $S(B) = S_{\min} + \frac1B \int B_{\mathrm{noise}}\, dL/\Delta L_{\max}$, and the integral is $E_{\min} = \lim_{B \to 0} B\, S(B)$, which is (i).
2. Then $S/S_{\min} - 1 = E_{\min}/(B S_{\min})$, and $E = B S_{\min} + E_{\min}$ gives $E/E_{\min} - 1 = B S_{\min}/E_{\min}$. Their product is 1, which is (ii).
3. $E_{\min}/S_{\min} = \int B_{\mathrm{noise}}\, dS_{\min} / \int dS_{\min}$ with $dS_{\min} = dL / \Delta L_{\max}$, which is (iii).
4. At $B = B_{\mathrm{crit}}$, $S = S_{\min} + S_{\min}$ and $E = B_{\mathrm{crit}} S = 2E_{\min}$, which is (iv).

### Result and check

The check first integrates the model on 200,000 loss levels with $\Delta L_{\max}$ proportional to $L$ and $B_{\mathrm{noise}} \propto L^{-1/2}$, and confirms (i) to (iv) and the two limits $E \to E_{\min}$ and $S \to S_{\min}$.
It then runs noisy SGD with the optimal step on a 20-dimensional quadratic with condition number 100 and constant gradient noise, from twelve batch sizes between 2 and 16,384, six seeds each, to a loss 1,000 times lower than at the start.
The step counts span more than two decades; the curve $S_{\min}(1 + B_{\mathrm{crit}}/B)$, fitted in log space as McCandlish et al. fit theirs, is within 30% of every mean step count; and the fitted $B_{\mathrm{crit}}$ is within a factor of 2 of the noise scale averaged over the steps of a near-noiseless run (about 2,190 against 1,680).
