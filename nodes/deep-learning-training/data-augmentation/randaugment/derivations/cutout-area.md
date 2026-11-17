---
id: cutout-area
derives: [c-cutout-area, c-cutout-16, eq-cutout-area]
kind: identity
assumes: ["the image is L x L pixels", "the side s is even with 0 <= s <= 2L", "the centre (r, c) is uniform on the L^2 pixels", "the mask covers rows max(r - s/2, 0) to min(r + s/2, L), half open, and the same for columns"]
uses: [discrete-uniform-distribution@m2, independence-of-random-variables@m2, expectation-of-a-product-of-independent-random-variables@m2]
status: full
check: cutout-area.check.py
lean: null
sources:
  - {work: devries-2017-cutout, locator: "Section 3.2"}
  - {work: devries-2017-cutout, locator: "Section 4.1"}
---

### Statement

On an $L \times L$ image, let Cutout mask an $s \times s$ square, $s$ even and $0 \le s \le 2L$, centred on a pixel $(r, c)$ drawn uniformly, and clipped at the border. The mean number of masked pixels is

$$ \mathbb E[A] = \Big(s - \frac{s^2}{4L}\Big)^2 , $$

so the mean masked share of the image is $(\sigma - \sigma^2/4)^2$ with $\sigma = s/L$, against $\sigma^2$ with no clipping. The formula is exact on the pixel grid, not only in the limit of a large image. For the 16 × 16 patch that DeVries and Taylor use on 32 × 32 CIFAR-10 images, $\mathbb E[A] = 14^2 = 196$ pixels, $0.191$ of the image, where the nominal patch is $256$ pixels, $0.25$; clipping removes $23.4\%$ of the nominal area on average.

### Assumptions

1. The centre is a pixel $(r, c)$ with $r, c$ independent and uniform on $\{0, \dots, L - 1\}$, and the patch may extend past the border, as in Section 3.2 of DeVries and Taylor. Used in steps 1 and 3.
2. With $h = s/2$, the masked rows are $[\max(r - h, 0), \min(r + h, L))$ and the masked columns the same with $c$; the mask is their product. This is the convention of the implementation on this page. Used in steps 1 and 3.
3. $s$ is even and $h \le L$. For odd $s$ the integer half $\lfloor s/2 \rfloor$ gives a side of $s - 1$, so the formula holds with $s - 1$. Used in step 2.

### Steps

1. The masked length along rows is $f(r) = \min(r + h, L) - \max(r - h, 0) = 2h - (h - r)^+ - (r + h - L)^+$, which holds for every $r$ in $\{0, \dots, L - 1\}$ since both clipped ends stay ordered.
2. Sum over $r$. The first loss is positive for $r = 0, \dots, h - 1$ and adds up to $h + (h - 1) + \dots + 1 = h(h + 1)/2$. The second is positive for $r = L - h + 1, \dots, L - 1$ and adds up to $1 + \dots + (h - 1) = h(h - 1)/2$. Both need $h \le L$ (assumption 3). So $\sum_r f(r) = 2hL - h^2 = sL - s^2/4$, and $\mathbb E[f] = s - s^2/(4L)$.
3. The masked area is $A = f(r) f(c)$ (assumption 2). Since $r$ and $c$ are independent (assumption 1), $\mathbb E[A] = \mathbb E[f(r)]\,\mathbb E[f(c)] = (s - s^2/(4L))^2$.
4. With a continuous centre uniform on $[0, L]$, $\mathbb E[(h - r)^+] = h^2/(2L)$ for each end, and the mean side is again $s - s^2/(4L)$; the CutMix derivation of [[mixup]] has the same result on the unit square.
5. For $L = 32$: $s = 16$ gives $(16 - 2)^2 = 196$ and $196/1024 = 0.19140625$; $s = 8$ gives $7.5^2 = 56.25$, $0.0549$ of the image, against $0.0625$.

### Result and check

$$ \mathbb E[A] = \Big(s - \frac{s^2}{4L}\Big)^2 \quad (s \text{ even}, \ 0 \le s \le 2L) . $$

The check `cutout-area.check.py` builds the mask on the pixel grid for every centre of every even $s$ from 0 to $2L$ and every $L$ from 1 to 24, and checks that the mean masked area equals the formula exactly, in rational arithmetic.
It checks that odd $s$ gives the value of $s - 1$, the continuous version by quadrature, and the numbers of step 5.
All the numbers in this derivation are this page's arithmetic; the 16 × 16 patch size is from DeVries and Taylor (Section 4.1).
