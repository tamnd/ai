---
id: output-size
derives: [c-out-size, c-same-padding, c-lenet-sizes, c-alexnet-size]
kind: identity
assumes: ["the input has n positions along an axis and is padded with p extra positions on each side", "the window has k positions, with 1 <= k <= n + 2p", "the window moves by s >= 1 positions between outputs and only windows that fit entirely in the padded input give an output"]
uses: []
status: full
check: output-size.check.py
lean: null
sources:
  - {work: lecun-1998-gradient-based-learning, locator: "p. 7"}
  - {work: simonyan-2015-very-deep-convolutional-networks, locator: "Section 2.1"}
  - {work: krizhevsky-2012-imagenet-classification, locator: "Section 3.5"}
---

### Statement

1. A convolution or pooling window of size $k$, with stride $s$ and padding $p$, on an axis of $n$ positions gives
$$ n' = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1 $$
outputs.
2. With $s = 1$ and odd $k$, the padding $p = (k - 1)/2$ keeps the size: $n' = n$. For $k = 3$ that is $p = 1$.
3. LeNet-5: a 5x5 convolution without padding takes 32 to 28, a 2x2 window with stride 2 takes 28 to 14, then 14 to 10 and 10 to 5, and a 5x5 convolution takes 5 to 1.
4. An 11x11 window with stride 4 and no padding gives 54 positions on 224 inputs, with a remainder: $(224 - 11)/4 = 53.25$. It gives 55 positions on 227 inputs, or on 224 inputs with $p = 2$.

### Assumptions

1. The padded axis has the positions $0, 1, \dots, n + 2p - 1$; the input sits at $p, \dots, p + n - 1$. Used in step 1.
2. Output $i$ reads the window that starts at position $s i$, so it covers $s i, \dots, s i + k - 1$. This is the index of eq-conv. Used in steps 1 and 2.
3. A window gives an output only if it fits: $s i + k - 1 \le n + 2p - 1$. Used in step 2.

### Steps

1. By assumptions 1 and 2, the outputs are indexed by the integers $i \ge 0$, and output $i$ starts at $s i$.
2. By assumption 3, output $i$ exists if and only if $s i \le n + 2p - k$, that is $i \le (n + 2p - k)/s$. Since $i$ is an integer, this is $i \le \lfloor (n + 2p - k)/s \rfloor$.
3. The integers $0 \le i \le \lfloor (n + 2p - k)/s \rfloor$ are $\lfloor (n + 2p - k)/s \rfloor + 1$ in number. This is statement 1. The assumption $k \le n + 2p$ makes the floor at least 0, so there is at least one output.
4. With $s = 1$, the floor is $n + 2p - k$ and $n' = n + 2p - k + 1$. Setting $p = (k - 1)/2$ gives $n' = n + k - 1 - k + 1 = n$. This is statement 2.
5. LeNet-5: $(32 - 5)/1 + 1 = 28$; $\lfloor (28 - 2)/2 \rfloor + 1 = 14$; $14 - 5 + 1 = 10$; $\lfloor (10 - 2)/2 \rfloor + 1 = 5$; $5 - 5 + 1 = 1$. These are the sizes 28x28, 14x14, 10x10, 5x5 and 1x1 that LeCun et al. (1998) give for C1, S2, C3, S4 and C5. This is statement 3.
6. For $n = 224, k = 11, s = 4, p = 0$: $\lfloor 213/4 \rfloor + 1 = 53 + 1 = 54$, and the division leaves 1 position over. For $n = 227$: $216/4 + 1 = 55$. For $n = 224, p = 2$: $\lfloor 217/4 \rfloor + 1 = 54 + 1 = 55$. This is statement 4.

### Result and check

$$ n' = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1 $$

The check `output-size.check.py` counts the windows that fit by enumeration for all $n \le 30$, $k \le 7$, $s \le 4$ and $p \le 3$, compares the count with the formula, and evaluates the examples of statements 3 and 4.
