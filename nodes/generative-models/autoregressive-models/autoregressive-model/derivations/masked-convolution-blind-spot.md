---
id: masked-convolution-blind-spot
derives: [c-blind-spot]
kind: closed-form
assumes: ["L layers of 3 x 3 masked convolutions, the first with mask A and the rest with mask B", "one channel per pixel, so the color masks play no role", "generic weights, so that no path cancels"]
uses: [autoregressive-model@m2, conditional-probability@m2]
status: full
check: masked-convolution-blind-spot.check.py
lean: null
sources:
  - {work: van-den-oord-2016-pixel-recurrent-neural-networks, locator: "p. 4", note: "masks A and B"}
  - {work: van-den-oord-2016-conditional-image-generation-pixelcnn, locator: "p. 3", note: "the blind spot, up to a quarter of the potential receptive field, and the vertical and horizontal stacks that remove it"}
---

### Statement

Index the inputs that one output pixel depends on by their offset $(i, j)$: row $i \le 0$, column $j$. Mask A of a $3 \times 3$ kernel keeps the offsets $(-1, -1), (-1, 0), (-1, 1), (0, -1)$. Mask B also keeps $(0, 0)$. After one mask A layer and $L - 1$ mask B layers, the output depends on

$$ F_L = \{(-u, j) : 1 \le u \le L,\ -L \le j \le u\} \ \cup\ \{(0, j) : -L \le j \le -1\} . $$

The causal box that $L$ layers of $3 \times 3$ kernels could reach is $\{(-u, j) : 1 \le u \le L, |j| \le L\} \cup \{(0, j) : -L \le j \le -1\}$. It holds $2L(L+1)$ positions, and $F_L$ misses

$$ \sum_{u=1}^{L} (L - u) = \frac{L(L-1)}{2} $$

of them: in row $-u$, the columns $u + 1, \dots, L$ to the upper right. The missing fraction is $(L-1)/(4(L+1))$, which rises toward $1/4$. The two stacks of Gated PixelCNN reach the whole box.

### Assumptions

1. With several channels per pixel, mask A also hides the current pixel's later colors, and mask B lets a color see itself. The spatial field is the same. Used in step 1.
2. A path through a ReLU or a gate can vanish at particular weights, but not for generic weights. Used in step 4.

### Steps

1. Composing convolutions adds offsets. $F_L$ is the set of sums of one offset from mask A and $L - 1$ offsets from mask B.
2. Let a sum use $u$ upward steps, each $(-1, \delta)$ with $\delta \in \{-1, 0, 1\}$, and $s$ leftward steps $(0, -1)$. The remaining steps are $(0, 0)$, from mask B only. The row is $-u$, and the column ranges over $[-u - s, u]$ with $u + s \le L$. For $u \ge 1$ the columns fill $[-L, u]$. For $u = 0$ the mask A step must be $(0, -1)$, so the columns fill $[-L, -1]$.
3. Count the box: $L$ rows of $2L + 1$ columns, plus $L$ positions in the current row, which is $2L(L + 1)$. Row $-u$ of $F_L$ lacks the $L - u$ columns $u + 1, \dots, L$, and the sum over $u$ is $L(L - 1)/2$.
4. The missing pixels lie up and to the right. They are not hidden by the mask itself, since mask A lets row $-1$ see column $+1$. They are lost because every path to them would have to pass through a pixel to the right of the current one in the current row, which the masks exclude at every layer. Raster order needs those pixels, and they are available.
5. Gated PixelCNN's vertical stack uses unmasked kernels of rows $-1, 0$ and columns $-1, 0, 1$. After $l$ layers it sees the rectangle of rows $-l, \dots, 0$ and columns $-l, \dots, l$. It feeds the horizontal stack one row down. The horizontal stack sees its own row to the left. So after $L$ layers the output sees rows $-L - 1, \dots, -1$ across columns $-L, \dots, L$, and the current row to the left, a superset of the box. It never sees itself or anything after it in raster order.

### Result and check

$$ |\text{box} \setminus F_L| = \frac{L(L-1)}{2}, \qquad \frac{|\text{box} \setminus F_L|}{|\text{box}|} = \frac{L - 1}{4(L + 1)} \to \frac{1}{4} . $$

For $L = 15$, the depth of the PixelCNN of van den Oord and colleagues, it misses 105 of 480 positions, 22%.

The check `masked-convolution-blind-spot.check.py` builds $F_L$ by composing the masks for $L$ from 1 to 15 and compares it with the closed form, the count and the fraction. It runs a random ReLU network of masked convolutions on an image, perturbs every pixel, and confirms that the output moves only for pixels of $F_L$, and for all of them. It also confirms that the two-stack field covers the box and stays causal.
