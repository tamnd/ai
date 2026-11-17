---
id: update-swamping
derives: [c-underflow-threshold, c-subnormal-error, c-swamping-bounds, c-swamping-bf16, c-master-example, eq-swamping]
kind: bound
assumes: ["round to nearest, ties to even", "w is a normal number of the format", "the update u is a number of the format and the sum is rounded once"]
uses: [floating-point-arithmetic@pg]
status: full
check: update-swamping.check.py
lean: null
sources:
  - {work: micikevicius-2017-mixed-precision-training, locator: "p. 3"}
---

### Statement

In a binary format with $m$ mantissa bits and round to nearest, ties to even:

1. FP16 rounds a real $x$ to zero exactly when $|x| \le 2^{-25}$, half its smallest subnormal; magnitudes in $(2^{-25}, 2^{-24})$ round to $\pm 2^{-24}$. Below $2^{-14}$ the spacing stays $2^{-24}$, so the relative error of a stored subnormal can reach $1/2$, against at most $2^{-11}$ for a normal number.
2. For a normal weight $w$ and an update $u \ne 0$, the rounded sum $\mathrm{fl}(w + u)$ equals $w$ whenever $|w|/|u| > 2^{m+2}$, and differs from $w$ whenever $|w|/|u| < 2^{m+1}$:
   $$ \frac{|w|}{|u|} > 2^{m+2} \;\Rightarrow\; \mathrm{fl}(w + u) = w , \qquad \frac{|w|}{|u|} < 2^{m+1} \;\Rightarrow\; \mathrm{fl}(w + u) \ne w . $$
   For FP16 the two ratios are 4096 and 2048; for BF16 they are 512 and 256. Between them the outcome depends on where $w$ lies in its binade.
3. With $w = 1$ and $u = 2^{-12}$, 1000 updates leave an FP16 weight at 1, and move an FP32 weight to exactly $1 + 1000 \cdot 2^{-12} = 1.244140625$.

### Assumptions

1. Rounding is to nearest with ties to even, the default of IEEE 754 and of the conversions in NumPy and PyTorch.
2. $w$ is a normal number, $u$ is a number of the same format, and $w + u$ is formed exactly and rounded once.
3. Item 2 is about a single addition; a sequence of updates is the same addition repeated.

### Steps

1. The two FP16 numbers nearest zero are $0$ and $2^{-24}$. A positive $x$ rounds to zero when it is closer to 0, or at the midpoint $2^{-25}$, where the tie goes to the even pattern, which is zero. Every $x \in (2^{-25}, 2^{-24})$ is closer to $2^{-24}$. Subnormals are the multiples of $2^{-24}$, so the rounding error is at most $2^{-25}$ in absolute terms; near $x = 2^{-24}$ that is half of $x$. In the normal range the spacing in $[2^k, 2^{k+1})$ is $2^{k-m}$, and the relative error is at most $2^{-m-1} = 2^{-11}$.
2. Let $w \in [2^k, 2^{k+1})$ be positive; negative $w$ is the mirror image. The next number above $w$ is $w + 2^{k-m}$. The next one below is $w - 2^{k-m}$, or $w - 2^{k-m-1}$ when $w = 2^k$. The sum rounds back to $w$ when $|u|$ is less than half the gap on its side. If $|w|/|u| > 2^{m+2}$ then $|u| < |w|\,2^{-m-2} < 2^{k-m-1}$, half the gap above, and also half the gap below when $w > 2^k$; when $w = 2^k$, $|u| < 2^{k-m-2}$, half the smaller gap below. So the sum rounds to $w$. If $|w|/|u| < 2^{m+1}$ then $|u| > |w|\,2^{-m-1} \ge 2^{k-m-1}$, more than half of either gap, so $w + u$ is past the midpoint and rounds to another number.
3. Ratios 4096 and 2048 for FP16 are $2^{12}$ and $2^{11}$ with $m = 10$; 512 and 256 for BF16 are $2^{9}$ and $2^{8}$ with $m = 7$. At $w = 1$, $u = 2^{-11}$ is exactly half the gap $2^{-10}$ above 1, and the tie rounds to the even 1, so the update is lost at ratio 2048. At $w = 2 - 2^{-10}$, the update $2^{-11} - 2^{-21}$ has ratio about 4094 and is lost.
4. With $w = 1$ and $u = 2^{-12}$ the ratio is 4096, so by item 2 every FP16 addition returns 1. In FP32 ($m = 23$) the ratio stays below $2^{24}$, every update is kept, and the sums $1 + j\,2^{-12}$ for $j \le 1000$ need 12 mantissa bits, so they are exact.

### Result and check

$$ \frac{|w|}{|u|} > 2^{m+2} \;\Rightarrow\; \mathrm{fl}(w + u) = w , \qquad \frac{|w|}{|u|} < 2^{m+1} \;\Rightarrow\; \mathrm{fl}(w + u) \ne w . $$ {#eq-swamping}

The check `update-swamping.check.py` rounds 200000 random magnitudes between $2^{-30}$ and $2^{-20}$ to FP16 and finds that exactly those up to $2^{-25}$ become zero, with the ties at $2^{-25}$ and $1.5 \times 2^{-24}$.
It adds updates of both signs at random ratios to every positive normal FP16 weight, and to 300000 BF16 weights, and finds the updates lost above $2^{m+2}$, kept below $2^{m+1}$, and both in between.
It checks the edges of step 3 and the 1000 updates of item 3.
