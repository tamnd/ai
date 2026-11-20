---
id: parameter-count
derives: [c-params, c-mnist-count, c-one-layer-count, c-wrn-count, c-flops]
kind: calculation
assumes: ["the hypernetwork of eq-static with hidden size d = N_z, shared W_out and B_out, and one W_i and B_i for each input channel of a basic kernel", "a basic kernel has N_in = N_out = N channels and filters of f x f", "a multiply-add counts as 2 FLOPs, and only the products are counted"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-count.check.py
lean: null
sources:
  - {work: ha-2016-hypernetworks, locator: "p. 3", note: "eq. 2 and the counts of the two-layer and the one-layer hypernetwork"}
  - {work: ha-2016-hypernetworks, locator: "p. 7", note: "Section 4.1, the 7 x 7 x 16 x 16 kernel from an embedding of 4"}
  - {work: ha-2016-hypernetworks, locator: "p. 20", note: "Table 7, 12,544 and 4,244 parameters"}
---

### Statement

A static hypernetwork that generates $E$ basic kernels of $N$ input and $N$ output channels with $f \times f$ filters from embeddings of size $N_z$, through a hidden layer of size $d$, has

$$ N_{\text{hyper}} = N_z E + d(N_z + 1)N + fNf(d + 1) $$

parameters, against $E \cdot N f N f$ generated weights, and a one-layer linear map from $z$ to a whole kernel would need $N_z N f N f$ parameters.
Generating the $E$ kernels costs $2EN d(N_z + f^2 N)$ FLOPs, once for the whole batch.
For the MNIST layer of the paper, $f = 7$, $N = 16$, $N_z = d = 4$, $E = 1$: $N_{\text{hyper}} = 4{,}244$ for $12{,}544$ weights, and the one-layer map would have $50{,}176$.
For the wide residual networks WRN 40-1 and WRN 40-2 with $N = 16$, $f = 3$, $N_z = d = 64$, the 36 convolutions of the residual blocks need $E = 242$ and $E = 966$ basic kernels, and the hypernetworks have $91{,}408$ and $137{,}744$ parameters.

### Assumptions

1. The shortcut convolutions, the first convolution, the batch normalization and the classifier of the residual networks are not generated and are not counted. Used in step 4.

### Steps

1. For embedding $z^j$, the hidden vector of input channel $i$ is $a_i^j = W_i z^j + B_i$ with $W_i \in \mathbb{R}^{d \times N_z}$ and $B_i \in \mathbb{R}^d$: $N$ such pairs, $dN_z N + dN = d(N_z + 1)N$ parameters. The slice $K_i^j = \langle W_{\text{out}}, a_i^j\rangle + B_{\text{out}}$ uses one $W_{\text{out}} \in \mathbb{R}^{f \times Nf \times d}$ and $B_{\text{out}} \in \mathbb{R}^{f \times Nf}$ for all $i$ and $j$: $fNf(d + 1)$. The embeddings add $N_z E$.
2. For $f = 7$, $N = 16$, $N_z = d = 4$, $E = 1$: $4 + 4 \cdot 5 \cdot 16 + 7 \cdot 16 \cdot 7 \cdot 5 = 4 + 320 + 3{,}920 = 4{,}244$; the kernel has $7 \cdot 7 \cdot 16 \cdot 16 = 12{,}544$ weights, $2.96$ times as many. Without the embedding the hypernetwork has $4{,}240$. A one-layer map has $4 \cdot 16 \cdot 7 \cdot 16 \cdot 7 = 50{,}176$, four times the kernel.
3. Each $a_i^j$ costs $2dN_z$ FLOPs and each slice $2f \cdot Nf \cdot d$; there are $EN$ of each, for $2ENd(N_z + f^2N)$.
4. A wide residual network of depth 40 and width $k$ has three groups of six blocks of two $3 \times 3$ convolutions, of widths $16k$, $32k$ and $64k$, after a first convolution to 16 channels. A convolution from $16a$ to $16b$ channels is tiled from $ab$ basic kernels. For $k = 1$: group 1 has $12 \cdot 1$; group 2 has $2 + 11 \cdot 4 = 46$; group 3 has $8 + 11 \cdot 16 = 184$; $E = 242$. For $k = 2$: $2 + 11 \cdot 4 = 46$, $8 + 11 \cdot 16 = 184$, $32 + 11 \cdot 64 = 736$; $E = 966$.
5. With $N_z = d = 64$, $N = 16$, $f = 3$: $d(N_z + 1)N = 66{,}560$ and $fNf(d + 1) = 9{,}360$, so $N_{\text{hyper}} = 64 \cdot 242 + 75{,}920 = 91{,}408$ and $64 \cdot 966 + 75{,}920 = 137{,}744$.

### Result and check

The shared part of the hypernetwork, $d(N_z + 1)N + fNf(d + 1)$, does not grow with the number of layers; each extra basic kernel costs only its $N_z$ numbers. The check `parameter-count.check.py` compares the formula with the sizes of the parameters of the NumPy reference and with the paper's Table 7, recomputes steps 2, 4 and 5, enumerates the convolutions of the residual networks, and counts the FLOPs of `generate` with counted products.
