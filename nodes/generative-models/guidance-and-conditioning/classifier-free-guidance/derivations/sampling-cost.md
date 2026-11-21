---
id: sampling-cost
derives: [c-sampling-cost]
kind: count
assumes: ["the denoiser of the shape contract", "a guided step evaluates the conditional and the null model once each, and a step outside the guidance interval evaluates the conditional model only"]
uses: [matrix-multiplication@m2]
status: full
check: sampling-cost.check.py
lean: null
sources:
  - {work: ho-2022-classifier-free-diffusion-guidance, locator: "pp. 8 to 9", note: "each sampling step evaluates the denoising model twice, and sampling speed is a potential disadvantage, since classifier guidance can use a smaller, faster classifier"}
  - {work: kynkaanniemi-2024-applying-guidance-limited-interval, locator: "p. 4, Eqs. 5 to 6", note: "the guidance weight w on the interval (sigma_lo, sigma_hi] and 1 outside it"}
---

### Statement

The denoiser of the page, $\epsilon_\theta(z, t, y) = z + \tanh(zW_1 + \mathrm{emb}(t)W_t + E_y + b_1)W_2 + b_2$ with $z \in \mathbb{R}^D$, $H$ hidden units, a time embedding of width $E$ and a class table $E \in \mathbb{R}^{(C+1) \times H}$ whose last row is the null class, has

$$ P = DH + EH + (C + 1)H + H + HD + D $$

parameters and costs $C_f = 2B(DH + EH + HD)$ FLOPs on a batch of $B$, a multiply-add counted as 2. A sampler of $N$ steps costs $N C_f$ unguided, $2N C_f$ with guidance at every step, and $(N + k)C_f$ with guidance on an interval that contains $k$ of the $N$ noise levels. The null class adds $H$ parameters and no FLOPs.

### Assumptions

1. The shape contract: the class embedding is a table lookup, and biases, the time embedding and tanh are not counted. Used in steps 1 and 2.
2. The evaluation counts of a guided and an unguided step. Used in step 3.

### Steps

1. The weights are $W_1 \in \mathbb{R}^{D\times H}$, $W_t \in \mathbb{R}^{E \times H}$, the class table $(C + 1)\times H$, $b_1 \in \mathbb{R}^H$, $W_2 \in \mathbb{R}^{H \times D}$ and $b_2 \in \mathbb{R}^D$; their sizes add up to $P$, and the null row is $H$ of them.
2. The three products $zW_1$, $\mathrm{emb}(t)W_t$ and $hW_2$ take $BDH$, $BEH$ and $BHD$ multiply-adds; the lookup $E_y$ takes none.
3. Each step evaluates the network once, or twice when guided; summing over the $N$ steps gives $N + k$ evaluations when $k$ of them are guided, between $N$ for $k = 0$ and $2N$ for $k = N$.

### Result and check

For the test's denoiser, $D = 2$, $H = 64$, $E = 16$ and $C = 3$, $P = 1602$ and $C_f = 655{,}360$ FLOPs per Adam batch of 256. For a denoiser of this form on a $32 \times 32 \times 4$ latent with 1000 classes, $D = 4096$, $H = 1024$ and $E = 256$, $P = 9{,}680{,}896$ and $C_f = 17{,}301{,}504$ FLOPs per image.
Guidance doubles the cost of sampling, which Ho and Salimans name as a potential disadvantage against classifier guidance, whose classifier can be smaller and faster than the generative model. With the linear schedule of this page and 32 DDIM steps spaced evenly in $t$, the interval $(0.28, 2.90]$ that Kynkäänniemi and colleagues found for EDM2-S contains 12 noise levels, so guided sampling costs $44$ evaluations instead of $64$, about 31% less.
The check `sampling-cost.check.py` counts the parameters of `init` and the multiply-adds of the three products with `param_count` and `flops_forward` for every configuration of `fixtures/shapes.json`, and counts the evaluations of `guided_sampler` on the interval.
