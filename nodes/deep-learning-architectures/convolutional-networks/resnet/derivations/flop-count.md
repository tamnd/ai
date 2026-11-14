---
id: flop-count
derives: [c-flops, c-stage-invariance, c-stride-cost]
kind: count
assumes: ["a k x k convolution from c_in to c_out channels costs k^2 c_in c_out multiply-adds per output position", "the image side R is a multiple of 32", "batch normalization, ReLU, the poolings and the additions of the shortcuts are not counted"]
uses: [matrix-multiplication@m2]
status: full
check: flop-count.check.py
lean: null
sources:
  - {work: he-2016-deep-residual-learning, locator: "Table 1"}
  - {work: he-2016-deep-residual-learning, locator: "Section 3.3"}
  - {work: goyal-2017-accurate-large-minibatch-sgd, locator: "Section 5.1"}
---

### Statement

1. A bottleneck network of base width $w$ with $N = n_1 + n_2 + n_3 + n_4$ blocks, on an $R \times R$ image with $K$ classes, takes
$$ M = \frac{w^2 R^2}{16}\,(17 N + 19) + \frac{147\, w R^2}{4} + 32\, w K $$
multiply-adds per image when the stride of 2 is in the first 1x1 convolution of each downsampling block: 3.86, 7.57 and 11.28 billion for ResNet-50, -101 and -152 at $R = 224$.
2. A block that is not the first of its stage costs $17 w^2 R^2 / 16$ in every stage.
3. With the stride in the 3x3 convolution instead, the cost grows by $18 w^2 R^2 / 16$ to 4.09 billion for ResNet-50, and the parameters do not change.

### Assumptions

1. A $k \times k$ convolution from $c_\text{in}$ to $c_\text{out}$ channels costs $k^2 c_\text{in} c_\text{out}$ multiply-adds per output position; He et al. (2016, Section 3.3) count FLOPs as multiply-adds. Used in all steps.
2. $R$ is a multiple of 32, so the stem gives $R/2$, the max pooling $R/4$, and stage $s$ works on $R / (4 \cdot 2^s)$. Used in steps 1 to 4.
3. Only convolutions and the fully connected layer are counted. Used in step 5.

### Steps

1. The stem, a $7 \times 7$ convolution from 3 to $w$ channels with stride 2, has $(R/2)^2$ output positions and costs $49 \cdot 3 \cdot w \cdot R^2/4 = 147\, w R^2 / 4$.
2. Stage $s$ has $R_s^2 = R^2 / (16 \cdot 4^s)$ positions and mid width $m_s = 2^s w$. A block that is not the first one has the convolutions $4 m_s \to m_s$, $3 \times 3$ at $m_s$, $m_s \to 4 m_s$: $(4 + 9 + 4)\, m_s^2 = 17 m_s^2$ per position. Then $17 m_s^2 R_s^2 = 17 \cdot 4^s w^2 \cdot R^2 / (16 \cdot 4^s) = 17 w^2 R^2 / 16$: the factor $4^s$ of the doubled channels cancels the $4^{-s}$ of the halved map. This is statement 2, and the design rule of Section 3.3, "to preserve the time complexity per layer".
3. The first block of stage 0 adds a projection $w \to 4w$ to convolutions that take $w$ channels: $(1 + 9 + 4 + 4)\, m_0^2 = 18 m_0^2$ per position, $18 w^2 R^2 / 16$. In the first block of stage $s \ge 1$, all four convolutions work on the $R_s^2$ output positions when the stride is in the first 1x1 and the projection has stride 2: $(2 + 9 + 4 + 8)\, m_s^2 = 23 m_s^2$ per position, $23 w^2 R^2 / 16$.
4. Summing the stages: $18 + 3 \cdot 23 + 17\,(N - 4) = 17 N + 19$ units of $w^2 R^2 / 16$.
5. The pooled feature has $4 m_3 = 32 w$ entries and the fully connected layer costs $32 w K$. Adding steps 1 and 5 gives the formula of statement 1.
6. For $w = 64$, $R = 224$, $K = 1000$: $w^2 R^2 / 16 = 12{,}845{,}056$, the stem $115{,}605{,}504$ and the head $2{,}048{,}000$. With $N = 16$, $33$ and $50$ the totals are $3{,}857{,}973{,}248$, $7{,}570{,}194{,}432$ and $11{,}282{,}415{,}616$, that is 3.86, 7.57 and 11.28 billion. Table 1 of He et al. (2016) gives 3.8, 7.6 and 11.3 billion; the 3.86 of ResNet-50 is the one rounded figure that does not match, and it is the reason the reference tests compare with a tolerance of 0.07 billion rather than by rounding.
7. Moving the stride to the 3x3 convolution leaves every shape of every weight as it was, so the parameters are the same. Only the first 1x1 convolution of the three downsampling blocks changes: it now runs on the $4 R_s^2$ positions of the input, so its $2 m_s^2 R_s^2$ becomes $8 m_s^2 R_s^2$, an increase of $6 w^2 R^2 / 16$ per block and $18 w^2 R^2 / 16 = 231{,}211{,}008$ for the three. ResNet-50 then costs $4{,}089{,}184{,}256$, 4.09 billion. In the original, that 1x1 convolution with stride 2 computes on one input position in four: the "three-quarters of the input feature map" that it ignores (He et al. 2019, Section 4.2).

### Result and check

$$ M_\text{ResNet-50} = 3.86 \times 10^9, \qquad M_\text{ResNet-50, stride in 3x3} = 4.09 \times 10^9 . $$

The check `flop-count.check.py` counts the multiply-adds of every convolution of the reference implementation, layer by layer, and compares them with the closed form for several widths, depths and sides.
