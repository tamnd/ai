---
id: cutmix-area
derives: [c-cutmix-area, c-cutmix-mean, eq-cutmix-lambda]
kind: identity
assumes: ["the mask is 0 inside the clipped box and 1 elsewhere", "the box has width W√(1 − λ) and height H√(1 − λ) about a centre inside the image", "0 ≤ λ ≤ 1"]
uses: [linearity-of-expectation@m2]
status: full
check: cutmix-area.check.py
lean: null
sources:
  - {work: yun-2019-cutmix, locator: "Section 3.1"}
  - {work: yun-2019-cutmix, locator: "p. 12"}
---

### Statement

Let an image have $H$ rows and $W$ columns. Draw $\lambda \in [0, 1]$ and a centre $(r_x, r_y)$ with $0 \le r_x \le W$ and $0 \le r_y \le H$, set $r_w = W\sqrt{1 - \lambda}$ and $r_h = H\sqrt{1 - \lambda}$, and let the box have integer corners $x_1 = \operatorname{round}(\max(r_x - r_w/2, 0))$, $x_2 = \operatorname{round}(\min(r_x + r_w/2, W))$, and $y_1, y_2$ the same way with $r_y, r_h, H$. Let $M$ be 0 on the pixels of the box and 1 elsewhere, and $\tilde x = M \odot x_A + (1 - M) \odot x_B$. Then

1. the fraction of the pixels of $\tilde x$ taken from $x_A$ is exactly $\lambda' = 1 - (x_2 - x_1)(y_2 - y_1)/(WH)$, eq-cutmix-lambda;
2. before clipping and rounding the box covers $r_w r_h/(WH) = 1 - \lambda$ of the image, and clipping only removes area, so without rounding $\lambda' \ge \lambda$, with equality when the box lies inside the image;
3. if $\lambda \sim \mathrm{Unif}(0, 1)$ and the centre is uniform on the image, then in the limit of a large image, where rounding does not matter, $\mathbb E[\lambda'] = 163/240 \approx 0.6792$, not $1/2$.

### Assumptions

1. The mask is 0 exactly on the pixels with column in $[x_1, x_2)$ and row in $[y_1, y_2)$, as in the slice `x1:x2, y1:y2` of Algorithm A1.
2. The box has the shape of Equation (2) of Yun et al., $r_w = W\sqrt{1 - \lambda}$ and $r_h = H\sqrt{1 - \lambda}$, and $(r_x, r_y)$ is its centre, as in Algorithm A1.
3. In item 3 the centre coordinates are independent of each other and of $\lambda$, and $W, H \to \infty$ with the box scaled with them.

### Steps

1. The box holds $(x_2 - x_1)(y_2 - y_1)$ pixels, all from $x_B$, and the other pixels come from $x_A$. Dividing by the $WH$ pixels gives item 1. Since the cross-entropy is linear in the target (derivation `mixed-cross-entropy`), the label $\lambda' y_A + (1 - \lambda') y_B$ weights each image by its share of pixels.
2. The unclipped box has area $W\sqrt{1 - \lambda} \cdot H\sqrt{1 - \lambda} = (1 - \lambda) WH$. Clipping replaces $[r_x - r_w/2, r_x + r_w/2]$ by its intersection with $[0, W]$, which is no longer, and the same for rows; so the clipped area is at most $(1 - \lambda) WH$ and $\lambda' \ge \lambda$. Rounding moves each edge by at most half a pixel.
3. Scale the image to the unit square and write $s = \sqrt{1 - \lambda}$, the side of the box. For a centre $c \sim \mathrm{Unif}(0, 1)$ the clipped side is $s - (s/2 - c)^+ - (c + s/2 - 1)^+$, and $\mathbb E[(s/2 - c)^+] = \int_0^{s/2}(s/2 - c)\,dc = s^2/8$, the same for the other end, so its mean is $s - s^2/4$. The two sides are independent given $s$, so the mean clipped area is $(s - s^2/4)^2 = u - u^{3/2}/2 + u^2/16$ with $u = s^2 = 1 - \lambda \sim \mathrm{Unif}(0, 1)$. Its mean is $1/2 - (1/2)(2/5) + 1/48 = 77/240$, so $\mathbb E[\lambda'] = 1 - 77/240 = 163/240$.

### Result and check

$$ \lambda' = 1 - \frac{(x_2 - x_1)(y_2 - y_1)}{WH} \ \ge\ \lambda \ \text{(before rounding)} , \qquad \mathbb E[\lambda'] \to \frac{163}{240} \approx 0.6792 . $$

The check `cutmix-area.check.py` draws 20000 boxes on random image sizes from $1 \times 1$ to $64 \times 48$, builds the mask on the pixel grid, and checks that the share of pixels from $x_A$ equals item 1 exactly.
It checks item 2 without rounding, and with rounding to within the half-pixel bound.
It estimates the mean of item 3 by Monte Carlo on a $224 \times 224$ image to $2 \times 10^{-3}$, checks the integral of step 3 by quadrature, and evaluates the example of the page: a $16 \times 16$ box centred at $(4, 28)$ on a $32 \times 32$ image keeps 144 pixels and gives $\lambda' = 0.859375$.
