---
id: parameter-count
derives: [c-no-extra-parameters, c-params, c-bottleneck-projection-doubles, c-cifar-params]
kind: count
assumes: ["the shortcut of a block is the identity unless its input and output dimensions differ", "option B for the bottleneck networks: a 1x1 projection with batch normalization in the first block of each stage; option A for the CIFAR-10 networks", "the convolutions have no bias; each batch normalization has a scale and a shift per channel and its running statistics are not counted", "the fully connected layer has a bias"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-count.check.py
lean: null
sources:
  - {work: he-2016-deep-residual-learning, locator: "Table 1"}
  - {work: he-2016-deep-residual-learning, locator: "Section 4.1"}
  - {work: he-2016-deep-residual-learning, locator: "Table 6"}
---

### Statement

1. An identity shortcut has no parameter and no multiply-add, so a residual network has exactly the parameters and the multiply-adds of its plain counterpart, plus one addition per element of each block output.
2. A bottleneck network of base width $w$ with $n_1, n_2, n_3, n_4$ blocks per stage and $K$ classes has
$$ P = 149\,w + (18 w^2 + 20 w) + \sum_{s=1}^{3} \big(23\, m_s^2 + 20\, m_s\big) + \sum_{s=0}^{3} (n_{s+1} - 1)\big(17\, m_s^2 + 12\, m_s\big) + 32\, w K + K, \qquad m_s = 2^s w, $$
parameters: 25,557,032 for ResNet-50, 44,549,160 for ResNet-101 and 60,192,808 for ResNet-152.
3. The 256-64-256 bottleneck block has 69,632 convolution weights; a projection shortcut in its place would add 65,536.
4. The CIFAR-10 network of $6n + 2$ layers has $464 + 4{,}672\,n + 13{,}952 + 18{,}560\,(n-1) + 55{,}552 + 73{,}984\,(n-1) + 650$ parameters: 269,722 for $n = 3$ and 19,421,274 for $n = 200$.

### Assumptions

1. A shortcut is the identity when the input and the output of the block have the same shape (Eq. 1 of He et al. 2016). Used in step 1.
2. The bottleneck networks use option B: the first block of each stage has a 1x1 projection followed by batch normalization; the others have the identity (Section 4.1). The CIFAR-10 networks use option A, zero-padding, with no parameter (Section 4.2). Used in steps 3 to 7.
3. The convolutions have no bias, since a batch normalization follows each of them; each batch normalization has 2 parameters per channel. Used in steps 2 to 7.
4. The fully connected layer maps the pooled features to $K$ classes with a bias. Used in steps 4 and 7.

### Steps

1. By assumption 1, the shortcut computes $x \mapsto x$: it has no weight, and the only operation it adds is the sum $F(x) + x$, one addition per output element. Removing every shortcut leaves the plain network with the same layers and the same weights.
2. A $k \times k$ convolution from $c_\text{in}$ to $c_\text{out}$ channels without bias has $k^2 c_\text{in} c_\text{out}$ weights, and the batch normalization after it $2 c_\text{out}$ parameters (assumption 3). The stem, a $7 \times 7$ convolution from 3 channels to $w$, has $147 w + 2 w = 149 w$.
3. In stage $s$, the block has mid width $m_s = 2^s w$ and output width $4 m_s$. A block that is not the first of its stage takes $4 m_s$ channels: its three convolutions have $4 m_s^2 + 9 m_s^2 + 4 m_s^2 = 17 m_s^2$ weights and its normalizations $2 m_s + 2 m_s + 8 m_s = 12 m_s$ parameters.
4. The first block of stage 0 takes the $w = m_0$ channels of the stem: $m_0^2 + 9 m_0^2 + 4 m_0^2$ for the convolutions and $4 m_0^2$ for the projection, $18 m_0^2$ in all, with $12 m_0 + 8 m_0 = 20 m_0$ for the normalizations. The first block of stage $s \ge 1$ takes the $4 m_{s-1} = 2 m_s$ channels of the stage before: $2 m_s^2 + 9 m_s^2 + 4 m_s^2 + 8 m_s^2 = 23 m_s^2$ and $20 m_s$. The head has $4 m_3 K + K = 32 w K + K$ (assumption 4). Summing gives the formula of statement 2.
5. For $w = 64$ and $K = 1000$, the formula gives 25,557,032 for $(3, 4, 6, 3)$, 44,549,160 for $(3, 4, 23, 3)$ and 60,192,808 for $(3, 8, 36, 3)$. These are the counts that common implementations of the three networks report; Table 1 of He et al. (2016) gives the layers but not the counts.
6. The block of Figure 5 of He et al. (2016) has $256 \cdot 64 + 9 \cdot 64^2 + 64 \cdot 256 = 16{,}384 + 36{,}864 + 16{,}384 = 69{,}632$ weights. A projection from 256 to 256 channels would add $256^2 = 65{,}536$, which is 94% of the block: "the time complexity and model size are doubled" (Section 4.1).
7. The CIFAR-10 network has a $3 \times 3$ convolution from 3 to 16 channels ($432 + 32 = 464$), then on each of the widths 16, 32 and 64, $n$ blocks of two $3 \times 3$ convolutions. At width 16 a block has $2 \cdot 9 \cdot 16^2 + 4 \cdot 16 = 4{,}672$. The first block at width 32 has $9 \cdot 16 \cdot 32 + 9 \cdot 32^2 + 4 \cdot 32 = 13{,}952$ and the others $2 \cdot 9 \cdot 32^2 + 128 = 18{,}560$; at width 64, $9 \cdot 32 \cdot 64 + 9 \cdot 64^2 + 256 = 55{,}552$ and $2 \cdot 9 \cdot 64^2 + 256 = 73{,}984$. The head has $64 \cdot 10 + 10 = 650$. For $n = 3$ the sum is 269,722 and for $n = 200$ it is 19,421,274: the 0.27M and 19.4M of Table 6.

### Result and check

$$ P_\text{ResNet-50} = 25{,}557{,}032, \qquad P_\text{ResNet-101} = 44{,}549{,}160, \qquad P_\text{ResNet-152} = 60{,}192{,}808 . $$

The check `parameter-count.check.py` builds the parameter trees of the reference implementation for small configurations and counts their entries, then evaluates the closed forms of steps 4 to 7.
