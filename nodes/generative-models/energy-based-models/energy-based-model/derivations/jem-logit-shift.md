---
id: jem-logit-shift
derives: [c-jem-shift]
kind: identity
assumes: ["the logits are finite"]
uses: [logistic-regression@m2, cross-entropy@m2]
status: full
check: jem-logit-shift.check.py
lean: null
sources:
  - {work: grathwohl-2019-your-classifier-secretly-energy-based-model, locator: "p. 3", note: "Eq. 7, the energy as minus the LogSumExp of the logits, and the remark that shifting the logits changes log p(x) but not the classifier"}
---

### Statement

For logits $f(\mathbf{x}) \in \mathbb{R}^K$ and any $s(\mathbf{x}) \in \mathbb{R}$, with $E(\mathbf{x}) = -\log \sum_y e^{f(\mathbf{x})[y]}$,

$$ E_{f + s}(\mathbf{x}) = E_f(\mathbf{x}) - s(\mathbf{x}) , \qquad \operatorname{softmax}(f(\mathbf{x}) + s(\mathbf{x})) = \operatorname{softmax}(f(\mathbf{x})) . $$

So $p(y \mid \mathbf{x})$ fixes the logits only up to one function $s(\mathbf{x})$, and that function is exactly what sets the unnormalized density $e^{-E(\mathbf{x})}$.

### Assumptions

1. The logits are finite, so all the sums are positive. Used in steps 1 and 2.

### Steps

1. $\sum_y e^{f[y] + s} = e^{s} \sum_y e^{f[y]}$, so $\log \sum_y e^{f[y] + s} = s + \log \sum_y e^{f[y]}$ and $E_{f+s} = E_f - s$.
2. $\operatorname{softmax}(f + s)_y = e^{f[y] + s}/\sum_{y'} e^{f[y'] + s} = e^{f[y]}/\sum_{y'} e^{f[y']}$, since $e^s$ cancels.
3. The joint model $p(\mathbf{x}, y) = e^{f(\mathbf{x})[y]}/Z$ has marginal $p(\mathbf{x}) = \sum_y e^{f(\mathbf{x})[y]}/Z = e^{-E(\mathbf{x})}/Z$ and conditional $p(y \mid \mathbf{x}) = \operatorname{softmax}(f(\mathbf{x}))_y$, in which $Z$ cancels. A classifier trained with the cross-entropy alone leaves $s(\mathbf{x})$ free, and the generative term of JEM is what pins it down.

### Result and check

$$ \frac{\partial E_{f+s}(\mathbf{x})}{\partial s(\mathbf{x})} = -1 , \qquad \frac{\partial \operatorname{softmax}(f + s)}{\partial s} = 0 . $$

The check `jem-logit-shift.check.py` shifts random logits by random per-row constants and compares `jem_energy` and `jem_class_probs` before and after, checks the marginal and conditional of the joint model on a small discrete input space, and checks that large logits do not overflow.
