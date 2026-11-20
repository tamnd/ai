---
id: parameter-and-flop-count
derives: [c-params, c-params-models, c-flops, c-flops-models, c-attention-compare]
kind: calculation
assumes: ["equivariant layers h Lambda - 1 pool(h) Gamma + 1 beta with a bias", "a readout pool with no parameters, one hidden layer of rho and a linear layer to K classes", "a multiply-add counts as 2 FLOPs, and only the matrix products are counted"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-and-flop-count.check.py
lean: null
sources:
  - {work: zaheer-2017-deep-sets, locator: "p. 19", note: "eq. 22, Lambda and Gamma of D x D'"}
  - {work: zaheer-2017-deep-sets, locator: "p. 26", note: "Appendix H, the point-cloud classifier"}
---

### Statement

A network with $L$ equivariant layers of width $H$ on elements of $D$ features, a hidden layer of $\rho$ of width $H$ and $K$ classes has

$$ N = 2DH + H + (L - 1)(2H^2 + H) + H^2 + H + HK + K $$

parameters, and a forward pass over a set of $M$ elements costs

$$ C = 2(M + 1)DH + (L - 1)\,2(M + 1)H^2 + 2H^2 + 2HK $$

FLOPs. For the point-cloud classifier, $D = 3$, $L = 3$, $K = 40$: $N = 340{,}520$ for $H = 256$ and $1{,}336{,}360$ for $H = 512$; with $M = 1000$ and $H = 256$, $C = 264{,}095{,}232$, of which the products with $\Gamma$ are $263{,}680$. One layer of self-attention of width 256 on 1,000 elements spends $4M^2H = 1{,}024{,}000{,}000$ FLOPs in $QK^\top$ and $AV$.

### Assumptions

1. The pools, the activations, the softmax and the additions are not counted. Used in step 3.

### Steps

1. Layer $\ell$ maps $D_\ell$ features to $D_{\ell+1}$ with $\Lambda, \Gamma \in \mathbb{R}^{D_\ell \times D_{\ell+1}}$ and $\beta \in \mathbb{R}^{D_{\ell+1}}$: $2D_\ell D_{\ell+1} + D_{\ell+1}$ parameters. With $D_0 = D$ and $D_\ell = H$ after, the layers have $2DH + H + (L-1)(2H^2 + H)$.
2. The hidden layer of $\rho$ has $H^2 + H$ and the output layer $HK + K$. Nothing depends on $M$.
3. In layer $\ell$, $h\Lambda$ multiplies an $M \times D_\ell$ array by $D_\ell \times D_{\ell+1}$: $2MD_\ell D_{\ell+1}$ FLOPs. $\operatorname{pool}(h)\Gamma$ multiplies one row: $2D_\ell D_{\ell+1}$. The readout costs $2H^2 + 2HK$. Summing gives $C$.
4. For $D = 3$, $H = 256$, $L = 3$, $K = 40$: $2 \cdot 3 \cdot 256 + 256 = 1{,}792$; $2 \cdot (2 \cdot 256^2 + 256) = 262{,}656$; $256^2 + 256 + 256 \cdot 40 + 40 = 76{,}072$; total $340{,}520$. For $H = 512$: $3{,}584 + 1{,}049{,}600 + 283{,}176 = 1{,}336{,}360$.
5. For $M = 1000$, $H = 256$: $2 \cdot 1001 \cdot 3 \cdot 256 = 1{,}537{,}536$; $2 \cdot 2 \cdot 1001 \cdot 65{,}536 = 262{,}406{,}144$; $2 \cdot 65{,}536 + 2 \cdot 10{,}240 = 151{,}552$; total $264{,}095{,}232$. The $\Gamma$ products are $2 \cdot 3 \cdot 256 + 2 \cdot 2 \cdot 65{,}536 = 263{,}680$, $0.0998\%$.
6. $QK^\top$ multiplies $M \times H$ by $H \times M$, $2M^2H$ FLOPs, and $AV$ multiplies $M \times M$ by $M \times H$, another $2M^2H$: $4 \cdot 10^6 \cdot 256 = 1{,}024{,}000{,}000$, $3.88$ times $C$.

### Result and check

The count of parameters does not depend on $M$, and $C$ grows linearly in $M$, against the $M^2$ of attention. The check `parameter-and-flop-count.check.py` compares the formulas with the sizes of the parameters of the reference for random configs of both heads, counts the FLOPs of every matrix product of a forward pass, and recomputes the numbers of steps 4 to 6.
