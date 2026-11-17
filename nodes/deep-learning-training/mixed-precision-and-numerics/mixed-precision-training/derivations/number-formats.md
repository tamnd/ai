---
id: number-formats
derives: [c-format-limits, c-fp16-limits, c-bf16-limits, c-bf16-table, c-fp8-limits, c-e4m3-reclaim, c-e5m2-in-fp16, eq-format-limits]
kind: count
assumes: ["a binary format with a sign bit, e ≥ 2 exponent bits with bias 2^(e−1) − 1 and m ≥ 1 mantissa bits", "the top exponent holds infinities and NaNs, except in E4M3, which keeps only one NaN pattern there", "a zero exponent field encodes subnormal numbers"]
uses: [floating-point-arithmetic@pg]
status: full
check: number-formats.check.py
lean: null
sources:
  - {work: micikevicius-2017-mixed-precision-training, locator: "p. 3"}
  - {work: kalamkar-2019-bfloat16-deep-learning-training, locator: "p. 4"}
  - {work: micikevicius-2022-fp8-formats, locator: "p. 3"}
---

### Statement

Let a binary floating-point format have one sign bit, $e$ exponent bits with bias $b = 2^{e-1} - 1$ and $m$ mantissa bits, with the IEEE 754 layout. Then

1. its machine epsilon, smallest normal, smallest subnormal and largest finite value are
   $$ \epsilon = 2^{-m}, \qquad x_{\min} = 2^{1-b}, \qquad x_{\mathrm{sub}} = 2^{1-b-m}, \qquad x_{\max} = (2 - 2^{-m})\,2^{b} ; $$
2. FP16 ($e = 5$, $m = 10$) has $\epsilon = 2^{-10}$, $x_{\min} = 2^{-14}$, $x_{\mathrm{sub}} = 2^{-24}$ and $x_{\max} = 65504$, normal exponents from $-14$ to $15$, and positive values in 40 binades;
3. BF16 ($e = 8$, $m = 7$) has $\epsilon = 2^{-7}$, the smallest normal $2^{-126}$ of FP32, $x_{\mathrm{sub}} = 2^{-133} \approx 9.18 \times 10^{-41}$ and $x_{\max} = (2 - 2^{-7})\,2^{127} \approx 3.3895 \times 10^{38}$;
4. E5M2 has $x_{\max} = 57344$, $x_{\min} = 2^{-14}$, $x_{\mathrm{sub}} = 2^{-16}$ and 32 binades, and every E5M2 value is an FP16 value;
5. E4M3 with the top exponent used for normal numbers, as Micikevicius et al. (2022) define it, has $x_{\max} = 448$, $x_{\min} = 2^{-6}$ and $x_{\mathrm{sub}} = 2^{-9}$; with the IEEE layout it would stop at 240, and the reclaimed exponent adds the seven magnitudes 256, 288, …, 448 and one binade, 18 instead of 17.

### Assumptions

1. The exponent field $E$ of a normal number runs from 1 to $2^e - 2$, and the value is $(1 + M/2^m)\,2^{E - b}$ for the mantissa field $M$. $E = 0$ encodes the subnormals $M\,2^{1 - b - m}$ and zero.
2. In the IEEE layout $E = 2^e - 1$ encodes infinities ($M = 0$) and NaNs. In E4M3 it encodes normal numbers, except that $M = 2^m - 1$ is NaN; there are no infinities.
3. Precision is counted in binades, the intervals $[2^k, 2^{k+1})$ that hold a positive finite value, and in decimal digits as $(m + 1)\log_{10} 2$.

### Steps

1. The number just above 1 has $E = b$ and $M = 1$, so $\epsilon = 2^{-m}$. The smallest normal has $E = 1$ and $M = 0$, so $x_{\min} = 2^{1-b}$. The smallest subnormal has $E = 0$, $M = 1$, giving $2^{1-b-m}$. The largest finite value has $E = 2^e - 2$ and $M = 2^m - 1$: $(1 + (2^m - 1)/2^m)\,2^{2^e - 2 - b} = (2 - 2^{-m})\,2^{b}$, since $2^e - 2 - b = b$.
2. FP16: $b = 15$, so $x_{\min} = 2^{-14}$, $x_{\mathrm{sub}} = 2^{-24}$ and $x_{\max} = (2 - 2^{-10})\,2^{15} = 65504$. The normal exponents $E - b$ run from $-14$ to $15$. The positive values lie in the binades $2^{-24}$ to $2^{15}$, which is 40 of them.
3. BF16: $b = 127$, as in FP32, so the normal range starts at $2^{-126}$ like FP32's, and $x_{\mathrm{sub}} = 2^{-133}$. $x_{\max} = (2 - 2^{-7})\,2^{127} = 3.38953\ldots \times 10^{38}$. The FP32 maximum is larger by the factor $(2 - 2^{-23})/(2 - 2^{-7})$, just over $1 + 2^{-8}$, and not by a power of two: the two formats share their exponents.
4. E5M2: $b = 15$, so $x_{\min} = 2^{-14}$, $x_{\mathrm{sub}} = 2^{-16}$ and $x_{\max} = 1.75 \times 2^{15} = 57344$; its binades run from $2^{-16}$ to $2^{15}$, 32 of them. It has the exponent field of FP16 and the top 2 of its mantissa bits, so each E5M2 pattern followed by eight zero bits is the FP16 pattern of the same value.
5. E4M3: $b = 7$, so $x_{\min} = 2^{-6}$ and $x_{\mathrm{sub}} = 2^{-9}$. With the IEEE layout the largest value is $(2 - 2^{-3})\,2^{7} = 240$. Using $E = 15$ for normal numbers adds $(1 + M/8)\,2^{8}$ for $M = 0, \ldots, 6$, that is 256, 288, …, 448, and the binade $[256, 512)$. Keeping $M = 7$ for NaN rules out 480.

### Result and check

$$ \epsilon = 2^{-m}, \quad x_{\min} = 2^{2 - 2^{e-1}}, \quad x_{\mathrm{sub}} = 2^{2 - 2^{e-1} - m}, \quad x_{\max} = (2 - 2^{-m})\,2^{2^{e-1} - 1} . $$ {#eq-format-limits}

The check `number-formats.check.py` decodes every bit pattern of FP16, BF16, E5M2 and E4M3 and finds the four limits of each, and compares FP16 and FP32 with `numpy.finfo`.
It checks the values of Table 1 of Kalamkar et al. to three figures: $3.38 \times 10^{38}$ for the BF16 maximum and $1.17 \times 10^{-38}$ for the FP32 smallest normal are the true values cut, not rounded.
It checks the FP8 values of Table 1 of Micikevicius et al. (2022), the seven extra E4M3 magnitudes, the binade counts 17, 18 and 32, and that E5M2 is a subset of FP16.
