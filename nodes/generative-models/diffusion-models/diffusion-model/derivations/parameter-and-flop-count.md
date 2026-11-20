---
id: parameter-and-flop-count
derives: [c-params]
kind: closed-form
assumes: ["the noise-prediction network of the shape contract: one tanh hidden layer of H units on the D coordinates of x_t and an E-dimensional embedding of t, and an output layer of D units", "only the products are counted, a multiply-add as 2 FLOPs"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-and-flop-count.check.py
lean: null
sources:
  - {work: ho-2020-denoising-diffusion-probabilistic-models, locator: "p. 4", note: "Algorithm 2: one network evaluation per step of the sampler, T steps"}
  - {work: song-2020-denoising-diffusion-implicit-models, locator: "p. 6", note: "sampling along a sub-sequence of the steps, with one evaluation per step of the sub-sequence"}
---

### Statement

1. The network $\boldsymbol\epsilon_\theta$ on $D$ data dimensions with an $E$-dimensional time embedding and $H$ hidden units has $P = (D + E + 1)H + (H + 1)D$ parameters.
2. One evaluation on $B$ points costs $C = 2BH(2D + E)$ FLOPs in its products; a training step evaluates it once, the ancestral sampler $T$ times and a DDIM sampler on $S$ of the steps $S$ times.
3. The example config ($D = 3$, $H = 5$, $E = 4$, $B = 4$) has $P = 58$ and $C = 400$; the config `two_modes` ($D = 2$, $H = 64$, $E = 16$, $B = 256$) has $P = 1{,}346$ and $C = 655{,}360$, and sampling 256 points with its $T = 100$ steps costs $6.55 \times 10^7$ FLOPs.

### Assumptions

1. Each layer is $\mathbf{h} W + \mathbf{b}$ with $W$ of shape $(n_{\text{in}}, n_{\text{out}})$. Used in steps 1 and 2.
2. A product of a $(B, n)$ and an $(n, m)$ matrix costs $2Bnm$ FLOPs. Used in step 2.

### Steps

1. The hidden layer has $W_1$ of shape $(D + E, H)$ and $\mathbf{b}_1$ of $H$ entries, $(D + E + 1)H$ in all; the output layer has $W_2$ of shape $(H, D)$ and $\mathbf{b}_2$ of $D$ entries, $(H + 1)D$.
2. The products cost $2B(D + E)H$ and $2BHD$, which add to $2BH(2D + E)$.
3. Substituting: $(3 + 4 + 1) \cdot 5 + 6 \cdot 3 = 58$ and $2 \cdot 4 \cdot 5 \cdot 10 = 400$; $(2 + 16 + 1) \cdot 64 + 65 \cdot 2 = 1{,}346$ and $2 \cdot 256 \cdot 64 \cdot 20 = 655{,}360$, times 100 steps.

### Result and check

$$ P = (D + E + 1)H + (H + 1)D, \qquad C = 2BH(2D + E) . $$

The check `parameter-and-flop-count.check.py` compares the formulas with the NumPy reference's `param_count`, the sizes of `init` and `flops_forward` for every config of `fixtures/shapes.json`, and recomputes the numbers of the statement.
