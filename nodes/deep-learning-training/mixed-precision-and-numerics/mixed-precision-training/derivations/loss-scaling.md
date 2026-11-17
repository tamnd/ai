---
id: loss-scaling
derives: [c-scale-window, c-scale-chain-rule, c-scale-precision, c-unscale-exact, c-scale-by-8, c-largest-safe-scale, eq-scale-window]
kind: identity
assumes: ["S = 2^k is a power of two", "FP16 with round to nearest, ties to even", "no value overflows and no intermediate falls below the normal range 2^-14 where exactness is claimed", "unscaling multiplies by 2^-k in FP32"]
uses: [floating-point-arithmetic@pg]
status: full
check: loss-scaling.check.py
lean: null
sources:
  - {work: micikevicius-2017-mixed-precision-training, locator: "p. 4"}
  - {work: micikevicius-2017-mixed-precision-training, locator: "p. 5"}
---

### Statement

Let the loss $L$ be multiplied by $S = 2^k$ before backpropagation in FP16. Then

1. the gradients of $SL$ are $S$ times those of $L$, and a gradient value $g$ is stored as a nonzero finite FP16 number exactly when
   $$ \frac{2^{-25}}{S} < |g| < \frac{65520}{S} , $$
   so the scale moves the window of magnitudes that FP16 keeps down by $k$ binades;
2. if $2^{-14} \le S|g| \le 65504$, the stored $\mathrm{fl}(Sg)$, divided by $S$, is within a relative error $2^{-11}$ of $g$, however small $g$ is; if also $|g| \ge 2^{-14}$, then $\mathrm{fl}(Sg) = S\,\mathrm{fl}(g)$ exactly;
3. multiplying a finite FP16 value by $2^{-k}$ in FP32 is exact for $0 \le k \le 102$;
4. backpropagation in FP16 with the upstream gradient multiplied by $S$ gives exactly $S$ times the FP16 gradients, as long as no intermediate overflows or is subnormal in either run;
5. $S = 8$ maps $[2^{-27}, 2^{-24})$ onto $[2^{-24}, 2^{-21})$, which FP16 keeps; and the largest power of two that cannot overflow a gradient of largest magnitude $G$ is $S = 2^{\lfloor \log_2(65504/G) \rfloor}$.

### Assumptions

1. $S = 2^k$ with $k$ an integer, $k \ge 0$ in items 1 to 4.
2. FP16 rounds to nearest, ties to even; values at or above 65520 overflow to infinity.
3. Exactness in items 2 and 4 needs every value involved to be a normal FP16 number, both with and without the scale.
4. Unscaling is done in FP32, after the FP16 gradients are copied to the master weights.

### Steps

1. The loss enters backpropagation only through $\partial(SL)/\partial L = S$, and every later step of the chain rule is linear in the upstream gradient, so every gradient is multiplied by $S$. By the underflow step of the derivation update-swamping, $\mathrm{fl}(Sg) \ne 0$ exactly when $S|g| > 2^{-25}$. The midpoint between 65504 and the next power of two, $65536$, is $65520$; at and above it the value rounds to infinity, and below it to a finite number. Dividing both bounds by $S$ gives the window.
2. A power of two only changes the exponent. When $Sg$ is in the normal range, $\mathrm{fl}(Sg)$ has the relative error of a normal number, at most $2^{-11}$, and dividing by $S$ keeps it. When $g$ is in the normal range as well, $g$ and $Sg$ have the same significand, so they round the same way and $\mathrm{fl}(Sg) = S\,\mathrm{fl}(g)$.
3. Every finite FP16 value is $M\,2^{q}$ with an integer $|M| < 2^{11}$ and $q \ge -24$. Times $2^{-k}$ it is $M\,2^{q-k}$ with $q - k \ge -126$, an FP32 normal or zero for $k \le 102$, since FP32 has 24 significand bits and normal exponents down to $-126$.
4. The backward pass is a sequence of products and sums. If one factor of a product carries $S$, the exact result carries $S$, and by step 2 its rounding does too; a sum of terms that all carry $S$ is $S$ times the sum, and rounds to $S$ times the rounded sum. ReLU masks depend on the forward pass only. By induction every FP16 intermediate of the scaled run is $S$ times the one of the unscaled run.
5. Multiplying by $8 = 2^3$ adds 3 to the exponent, mapping $[2^{-27}, 2^{-24})$ onto $[2^{-24}, 2^{-21})$, all above the zero threshold $2^{-25}$. For the largest gradient $G$, $SG \le 65504$ holds for $S = 2^j$ with $j \le \log_2(65504/G)$, and the largest such integer is the floor.

### Result and check

$$ \mathrm{fl}_{16}(S g) \notin \{0, \pm\infty\} \iff \frac{2^{-25}}{S} < |g| < \frac{65520}{S} , \qquad S = 2^k . $$ {#eq-scale-window}

The check `loss-scaling.check.py` rounds 400000 random gradients scaled by $2^k$, $k = 0, \ldots, 20$, and finds the window of item 1 exactly.
It checks the commuting and the $2^{-11}$ bound of item 2, unscales every positive FP16 value by $2^{-k}$ in FP32 for $k$ up to 102, and backpropagates a two-layer ReLU network in FP16 with scales 2, 16 and 256 and finds exactly scaled gradients.
With an upstream gradient of order $10^{-7}$, FP16 alone loses more than 5% of the gradient, and the scale $2^{12}$ brings the error under 1%.
It checks item 5 on 100000 values and four largest gradients.
