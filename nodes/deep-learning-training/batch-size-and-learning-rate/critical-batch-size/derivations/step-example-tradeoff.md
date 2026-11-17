---
id: step-example-tradeoff
derives: [c-tradeoff-hyperbola, c-crit-twice]
kind: closed-form
assumes: ["each step at batch size B makes the progress of 1 / (1 + B_noise / B) full-batch steps", "B_noise and the full-batch gain depend on the training run only through the loss reached", "the batch size is fixed for the whole run"]
uses: [one-step-improvement]
status: full
check: step-example-tradeoff.check.py
lean: null
sources:
  - {work: mccandlish-2018-empirical-model-large-batch-training, locator: "p. 7"}
  - {work: mccandlish-2018-empirical-model-large-batch-training, locator: "p. 27"}
---

### Statement

Train at a fixed batch size $B$ from a loss $L_0$ down to a target $L_1$.
Suppose each step at the loss $L$ gains $\Delta L_{\max}(L) / (1 + B_{\text{noise}}(L)/B)$, as in [one-step-improvement](one-step-improvement.md).
Write $S_{\min} = \int_{L_1}^{L_0} dL / \Delta L_{\max}(L)$ and $E_{\min} = \int_{L_1}^{L_0} B_{\text{noise}}(L)\, dL / \Delta L_{\max}(L)$.
Then, in the limit of many small steps:

(i) the run takes $S(B) = S_{\min} + E_{\min}/B$ steps and $E(B) = B\,S(B) = B\,S_{\min} + E_{\min}$ examples;

(ii) $(S/S_{\min} - 1)(E/E_{\min} - 1) = 1$ for every $B$;

(iii) $S_{\min}$ and $E_{\min}$ are the limits of $S$ as $B \to \infty$ and of $E$ as $B \to 0$, and $B_{\text{crit}} = E_{\min}/S_{\min}$ is $B_{\text{noise}}$ averaged over the full-batch steps of the run;

(iv) at $B = B_{\text{crit}}$ the run takes $2S_{\min}$ steps and $2E_{\min}$ examples.

### Assumptions

1. The per-step model of [one-step-improvement](one-step-improvement.md) holds at every point of the run, with the step size retuned to $\varepsilon_{\text{opt}}(B)$ as the run goes.
2. The run is long, so the sum over steps can be replaced by an integral over the loss.
3. $\Delta L_{\max}$ and $B_{\text{noise}}$ are functions of the loss alone. In a real network they depend on the parameters, and the path taken can depend on $B$; McCandlish et al. reach the same form by integrating over the full-batch trajectory instead (p. 27).

### Steps

1. A step at batch size $B$ gains as much as $1/(1 + B_{\text{noise}}/B)$ full-batch steps. So the steps needed to move the loss by $dL$ are $dS = (1 + B_{\text{noise}}(L)/B)\, dL / \Delta L_{\max}(L)$.
2. Integrate from $L_1$ to $L_0$. The first term gives $S_{\min}$; the second gives $E_{\min}/B$, because $B$ is constant. This is (i), and $E = BS$ because the batch size is fixed.
3. From (i), $S/S_{\min} - 1 = E_{\min}/(B S_{\min})$ and $E/E_{\min} - 1 = B S_{\min}/E_{\min}$. Their product is 1, which is (ii).
4. Letting $B \to \infty$ in (i) leaves $S_{\min}$, and letting $B \to 0$ in $E(B)$ leaves $E_{\min}$. The full-batch run spends $ds = dL/\Delta L_{\max}$ steps per $dL$, so $E_{\min}/S_{\min} = \int B_{\text{noise}}\, ds / \int ds$, the step average. This is (iii).
5. Put $B = E_{\min}/S_{\min}$ into (i): $S = 2S_{\min}$ and $E = 2E_{\min}$, which is (iv).

### Result and check

The check has two parts.
In the first, the loss follows the model exactly, with $\Delta L_{\max} = 0.002\,L$ and $B_{\text{noise}} = 50/L$ from $L = 1$ to $L = 0.1$, so the noise scale grows tenfold over the run.
The integrals give $S_{\min} = 1151.3$, $E_{\min} = 225{,}000$ and $B_{\text{crit}} = 195.4$.
Stepping the loss down one step at a time matches $S_{\min} + E_{\min}/B$ within 0.2% for $B$ from 4 to 16,384, the product in (ii) is within 0.02 of 1 up to $B = 1024$, and $B = B_{\text{crit}}$ takes $2S_{\min}$ steps within 0.2%.

The second part drops assumption 3.
It runs SGD on a 30-dimensional quadratic with Hessian eigenvalues from 0.03 to 1 and gradient noise $\Sigma = 4H$, at the optimal step of each point, until the loss falls to 1% of its start.
Over 16 batch sizes from 2 to 65,536, averaged over 6 seeds, the measured steps lie within 11% of the fitted hyperbola, with $S_{\min} = 18.7$ and $B_{\text{crit}} = 480$ from the fit (our run).
The full-batch run takes 21 steps, and the noise scale averaged over them is 451, within 7% of the fitted $B_{\text{crit}}$.
The misfit is largest far above $B_{\text{crit}}$, where the run needs only about 20 steps and the integral in step 2 is a coarse approximation of the sum.
