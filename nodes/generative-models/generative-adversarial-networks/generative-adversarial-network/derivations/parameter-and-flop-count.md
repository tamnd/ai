---
id: parameter-and-flop-count
derives: [c-params]
kind: closed-form
assumes: ["the generator and discriminator of the shape contract, each with one tanh hidden layer and affine layers with biases", "only the products are counted, a multiply-add as 2 FLOPs"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-and-flop-count.check.py
lean: null
sources:
  - {work: goodfellow-2014-generative-adversarial-nets, locator: "p. 4", note: "Algorithm 1, a discriminator step on m noise samples and m data examples"}
---

### Statement

1. A generator with noise of $J$ dimensions, one hidden layer of $H$ units and an output of $D$ dimensions has $P_G = (J + 1)H + (H + 1)D$ parameters; a discriminator with one hidden layer of $K$ units and one logit has $P_D = (D + 1)K + K + 1$.
2. One forward pass of a training step, the generator on $B$ noise vectors and the discriminator on the $B$ fakes and $B$ data points, costs $C = 2B(JH + HD) + 4B(DK + K)$ FLOPs in its products.
3. The config `toy_mixture` ($J = D = 1$, $H = K = 16$, $B = 128$) has $P_G = P_D = 49$ and $C = 24{,}576$; `toy_plane` ($J = D = 2$, $H = K = 16$, $B = 128$) has $P_G = 82$, $P_D = 65$ and $C = 40{,}960$.

### Assumptions

1. Each layer is $\mathbf{h} W + \mathbf{b}$ with $W$ of shape $(n_{\text{in}}, n_{\text{out}})$, and the discriminator's last layer is a vector $\mathbf{w}_2$ of $K$ weights and one bias. Used in steps 1 and 2.
2. A product of a $(B, n)$ and an $(n, m)$ matrix costs $2Bnm$ FLOPs. Used in step 2.

### Steps

1. The generator's layers have $JH + H$ and $HD + D$ weights and biases; the discriminator's have $DK + K$ and $K + 1$. This is statement 1.
2. The generator multiplies $(B, J)$ by $(J, H)$ and $(B, H)$ by $(H, D)$: $2BJH + 2BHD$. The discriminator multiplies $(2B, D)$ by $(D, K)$ and $(2B, K)$ by $(K,)$: $4BDK + 4BK$. This is statement 2.
3. $2 \cdot 16 + 17 = 49$ twice; $2 \cdot 128 \cdot 32 + 4 \cdot 128 \cdot 32 = 24{,}576$. $3 \cdot 16 + 17 \cdot 2 = 82$, $3 \cdot 16 + 17 = 65$, $2 \cdot 128 \cdot 64 + 4 \cdot 128 \cdot 48 = 40{,}960$. This is statement 3.

### Result and check

$$ P_G = (J + 1)H + (H + 1)D, \qquad P_D = (D + 1)K + K + 1, \qquad C = 2B(JH + HD) + 4B(DK + K) . $$

The check `parameter-and-flop-count.check.py` compares $P_G$ and $P_D$ with the sizes of the arrays of the NumPy reference for random sizes, counts the products of a forward pass with an instrumented array, and recomputes statement 3. The backward pass costs about twice the forward, and the gradient penalties of WGAN-GP and R1 add a second backward pass through the discriminator; they are not counted here.
