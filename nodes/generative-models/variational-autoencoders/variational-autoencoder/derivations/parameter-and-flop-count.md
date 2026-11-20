---
id: parameter-and-flop-count
derives: [c-params, c-params-models, c-flops, c-flops-models]
kind: closed-form
assumes: ["the model of the shape contract: an encoder with one hidden layer of H units and two heads of J units, a decoder with one hidden layer of H units and an output of D units, every layer affine with a bias", "only the products are counted, a multiply-add as 2 FLOPs"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-and-flop-count.check.py
lean: null
sources:
  - {work: kingma-2013-auto-encoding-variational-bayes, locator: "pp. 7, 11", note: "the Gaussian and Bernoulli MLPs of appendix C, 500 hidden units for MNIST and 200 for Frey Face, and the latent sizes of figure 2"}
---

### Statement

1. A variational autoencoder with a Bernoulli decoder has $P = (D + 1)H + 2(H + 1)J + (J + 1)H + (H + 1)D$ parameters; a Gaussian decoder adds a head for the log-variance of each pixel, $(H + 1)D$ more.
2. With $D = 784$, $H = 500$ and $J = 20$, the MNIST model of Kingma and Welling has 815,824 parameters; with $D = 560$, $H = 200$, $J = 20$ and a Gaussian decoder, the Frey Face model has 349,560.
3. A forward pass over $B$ examples with $S$ draws of the code each costs $C = 2BDH + 4BHJ + S(2BJH + 2BHD)$ FLOPs in its products, $2SBHD$ more with a Gaussian decoder: the encoder runs once per example and the decoder once per draw, so the cost of the importance weighted bound with $k = S$ samples is linear in $k$.
4. For the MNIST model with $S = 1$ that is 1,628,000 FLOPs per example, 824,000 in the encoder and 804,000 in the decoder; each further draw adds 804,000.

### Assumptions

1. The layers of the contract, each $x W + b$ with $W$ of shape $(n_{\text{in}}, n_{\text{out}})$. Used in steps 1 and 3.
2. A product of a $(B, n)$ and an $(n, m)$ matrix costs $2Bnm$ FLOPs. Used in step 3.

### Steps

1. The encoder's hidden layer has $DH + H$ weights and biases, each head $HJ + J$; the decoder's hidden layer has $JH + H$ and its output layer $HD + D$. Their sum is statement 1, and a second output head of $HD + D$ gives the Gaussian decoder.
2. $785 \cdot 500 + 2 \cdot 501 \cdot 20 + 21 \cdot 500 + 501 \cdot 784 = 392{,}500 + 20{,}040 + 10{,}500 + 392{,}784 = 815{,}824$, and $561 \cdot 200 + 2 \cdot 201 \cdot 20 + 21 \cdot 200 + 201 \cdot 560 = 237{,}000$, plus $201 \cdot 560 = 112{,}560$ for the second head, $349{,}560$. This is statement 2.
3. The encoder multiplies $(B, D)$ by $(D, H)$ and $(B, H)$ by two $(H, J)$ matrices: $2BDH + 4BHJ$. The decoder multiplies $(SB, J)$ by $(J, H)$ and $(SB, H)$ by $(H, D)$: $S(2BJH + 2BHD)$. This is statement 3.
4. $2 \cdot 784 \cdot 500 + 4 \cdot 500 \cdot 20 = 824{,}000$ and $2 \cdot 20 \cdot 500 + 2 \cdot 500 \cdot 784 = 804{,}000$. This is statement 4.

### Result and check

$$ P = (D + 1)H + 2(H + 1)J + (J + 1)H + (H + 1)D, \qquad C = 2BDH + 4BHJ + S\,(2BJH + 2BHD) . $$

The check `parameter-and-flop-count.check.py` compares $P$ with the sizes of the arrays of the NumPy reference for random sizes and both decoders, counts the products of a forward pass with an instrumented array, and recomputes the numbers of statements 2 and 4.
