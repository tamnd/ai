---
id: cfm-gradient
derives: [c-cfm-grad]
kind: closed-form
assumes: ["p_t(x) > 0 for all x and t", "the conditional and marginal fields have finite second moments under the path", "u_t is the marginal field of marginal-field"]
uses: [conditional-expectation@m2, mean-squared-error@m2, linearity-of-expectation@m2]
status: full
check: cfm-gradient.check.py
lean: null
sources:
  - {work: lipman-2022-flow-matching-generative-modeling, locator: "p. 4", note: "Theorem 2: assuming p_t(x) > 0 for all x and t, up to a constant independent of theta, L_CFM and L_FM are equal, hence their gradients agree"}
---

### Statement

Let $t$ be uniform, $\mathbf{x}_1 \sim q$, and $\mathbf{x} \sim p_t(\cdot \mid \mathbf{x}_1)$. Define the two losses

$$ \mathcal{L}_{\mathrm{FM}}(\theta) = \mathbb{E}\lVert v_\theta(\mathbf{x}, t) - u_t(\mathbf{x})\rVert^2 , \qquad \mathcal{L}_{\mathrm{CFM}}(\theta) = \mathbb{E}\lVert v_\theta(\mathbf{x}, t) - u_t(\mathbf{x} \mid \mathbf{x}_1)\rVert^2 . $$

Then

$$ \mathcal{L}_{\mathrm{CFM}}(\theta) = \mathcal{L}_{\mathrm{FM}}(\theta) + \mathbb{E}\lVert u_t(\mathbf{x} \mid \mathbf{x}_1)\rVert^2 - \mathbb{E}\lVert u_t(\mathbf{x})\rVert^2 . $$

The added term does not involve $\theta$. So the two losses have the same gradient at every $\theta$, and the same minimizers.

The FM loss needs the marginal field, which is unknown. The CFM loss needs only the conditional field, which is known in closed form.

### Assumptions

1. Under the joint law of $(t, \mathbf{x}_1, \mathbf{x})$, the marginal of $\mathbf{x}$ given $t$ is $p_t$. Used in step 2.
2. $u_t(\mathbf{x}) = \mathbb{E}[u_t(\mathbf{x} \mid \mathbf{x}_1) \mid \mathbf{x}, t]$, by marginal-field. Used in step 3.

### Steps

1. Expand both squares. Each loss is $\mathbb{E}\lVert v_\theta\rVert^2$, minus twice a cross term, plus the square of its target. Only the cross terms and the squares of the targets can differ.
2. The first term $\mathbb{E}\lVert v_\theta(\mathbf{x}, t)\rVert^2$ depends only on $(\mathbf{x}, t)$, and $(\mathbf{x}, t)$ has the same law in both losses.
3. Condition the CFM cross term on $(\mathbf{x}, t)$. This gives $\mathbb{E}\langle v_\theta(\mathbf{x}, t), u_t(\mathbf{x} \mid \mathbf{x}_1)\rangle = \mathbb{E}\langle v_\theta(\mathbf{x}, t), \mathbb{E}[u_t(\mathbf{x} \mid \mathbf{x}_1) \mid \mathbf{x}, t]\rangle$, which is the FM cross term $\mathbb{E}\langle v_\theta(\mathbf{x}, t), u_t(\mathbf{x})\rangle$.
4. What remains is the difference of the squared targets, which does not involve $\theta$.

### Result and check

$$ \nabla_\theta \mathcal{L}_{\mathrm{CFM}}(\theta) = \nabla_\theta \mathcal{L}_{\mathrm{FM}}(\theta) . $$

The constant gap equals $\mathbb{E}\operatorname{tr}\operatorname{Cov}[u_t(\mathbf{x} \mid \mathbf{x}_1) \mid \mathbf{x}, t]$. This is the part of the CFM loss that no network can remove, so the CFM loss does not go to 0 at the optimum.

The check `cfm-gradient.check.py` uses a two-component mixture as the data, with $\sigma_{\min} = 0$ and $\sigma_{\min} = 0.1$, and 400 000 draws of $(t, \mathbf{x}_0, \mathbf{x}_1)$. It:

- computes the CFM loss with the hand-written backward pass `loss_and_grads`;
- computes the FM loss with `regress` against the exact `mixture_field`;
- checks that the gradients agree for two different random networks;
- checks that the gap between the losses is the same for both networks;
- checks that the gap equals $\mathbb{E}\lVert \text{target}\rVert^2 - \mathbb{E}\lVert u_t\rVert^2$ and is positive.
