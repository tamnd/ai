---
id: parameter-count
derives: [c-param-count, c-output-share]
kind: count
assumes:
  - The network is y = b + W x + U tanh(d + H x) with a softmax output, x the concatenation of n − 1 feature vectors of size m, h hidden units and a vocabulary of |V| words.
  - The cost of one example is counted in multiply-adds of the weighted sums, as Bengio et al. count it.
uses: []
status: full
check: parameter-count.check.py
lean: null
sources:
  - paper: bengio-2003-neural-probabilistic-language-model
    locator: p. 7
  - paper: bengio-2003-neural-probabilistic-language-model
    locator: p. 8
  - paper: bengio-2003-neural-probabilistic-language-model
    locator: p. 9
---

## Statement

The model has |V|(1 + nm + h) + h(1 + (n − 1)m) free parameters with direct connections and |V|(1 + m + h) +
h(1 + (n − 1)m) without them. The AP News network of Bengio et al. (|V| = 17,964, h = 60, n = 6, m = 100) has
11,904,264 parameters, 95.2 MB as 8-byte floats. Bengio et al. describe their largest network as almost 100 megabytes.
The weighted sums of the output units take 99.7% of the multiply-adds of one example.

## Assumptions

The parameters are b (|V|), d (h), U (|V| × h), W (|V| × (n − 1)m), H (h × (n − 1)m) and C (|V| × m).

## Steps

1. Add the sizes: |V| + h + |V|h + |V|(n − 1)m + h(n − 1)m + |V|m = |V|(1 + nm + h) + h(1 + (n − 1)m), because
   (n − 1)m + m = nm. Dropping W removes |V|(n − 1)m, which leaves |V|(1 + m + h) + h(1 + (n − 1)m).
2. For the AP News network, (n − 1)m = 500 and nm = 600, so the count is 17,964 × 661 + 60 × 501 = 11,874,204 +
   30,060 = 11,904,264. At 8 bytes each that is 95,234,112 bytes. The output terms |V|(nm + h) alone are 99.6% of it.
3. The output units cost |V|(1 + (n − 1)m + h) = 10,077,804 multiply-adds, the hidden units h(1 + (n − 1)m) = 30,060
   and the feature layer (n − 1)m = 500. The share of the output units is 10,077,804 / 10,108,364 = 0.99698, which
   rounds to 99.7%. The paper's preceding sentence writes nm where its fraction has (n − 1)m; with nm in every term
   the share is 0.99692, which also rounds to 99.7%.

## Result and check

The check builds the six matrices for several sizes and compares their total with both formulas, then reproduces
the counts and the share above.
