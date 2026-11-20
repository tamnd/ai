---
id: parameter-and-flop-count
derives: [c-params]
kind: count
assumes: ["the parameter shapes of param_shapes", "dense matrix products, a multiply-add counted as 2 FLOPs"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-and-flop-count.check.py
lean: null
sources:
  - {work: peebles-2022-scalable-diffusion-models-transformers, locator: "p. 4", note: "the 32 x 32 x 4 latent of a 256 x 256 x 3 image, the shape of the imagenet_f8 configuration"}
---

### Statement

The denoiser with $N$ latent positions of $c$ channels, $D = Nc$, hidden width $H$, time embedding $E$, $M$ context tokens of width $d$ and $C$ classes has

$$ P = NcH + EH + H + (C + 1)Md + Hd + d^2 + dH + HNc + Nc $$

parameters, and one call on $B$ latents costs

$$ 2B\big(NcH + EH + Hd + Md^2 + MdH + Md + MH + HNc\big) $$

FLOPs, leaving out the biases, the time embedding, the tanh and the softmax.

### Assumptions

1. The parameter shapes $W_1 \in \mathbb{R}^{D \times H}$, $b_1 \in \mathbb{R}^H$, $W_t \in \mathbb{R}^{E \times H}$, $\mathrm{ctx} \in \mathbb{R}^{(C+1) \times M \times d}$, $W_q \in \mathbb{R}^{H \times d}$, $W_k \in \mathbb{R}^{d \times d}$, $W_v \in \mathbb{R}^{d \times H}$, $W_2 \in \mathbb{R}^{H \times D}$, $b_2 \in \mathbb{R}^D$. Used in step 1.
2. A product of an $m \times k$ and a $k \times n$ matrix costs $2mkn$ FLOPs. Used in step 2.

### Steps

1. Adding the sizes: $DH + H + EH + (C+1)Md + Hd + d^2 + dH + HD + D$, which is $P$ with $D = Nc$. The context table has a row for each class and one for the null class of classifier-free guidance.
2. Per latent, the products are $z W_1$ ($2DH$), $\mathrm{emb}(t) W_t$ ($2EH$), the query $h W_q$ ($2Hd$), the keys $\mathrm{ctx}_y W_k$ ($2Md^2$), the values $\mathrm{ctx}_y W_v$ ($2MdH$), the scores $q K^\top$ ($2Md$), the weighted sum $w V$ ($2MH$) and the read-out $h_2 W_2$ ($2HD$). Their sum times $B$ is the count.

### Result and check

The input and output layers, $2NcH$ parameters, dominate when the latent is large, which is why real denoisers share weights across positions with convolutions or patches. For the configurations of the shape contract the count is 126 for `example`, 3272 for `mixture` and 10479104 for `imagenet_f8`.
The check `parameter-and-flop-count.check.py` compares `param_count` and `flops_forward` with the formulas, the sizes of the arrays of `init`, the numbers of `fixtures/shapes.json`, and a count of the multiply-adds of each product from the shapes of the arrays of `_forward`.
