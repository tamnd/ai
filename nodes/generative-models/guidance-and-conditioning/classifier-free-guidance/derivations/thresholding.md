---
id: thresholding
derives: [c-threshold-bounds]
kind: bound
assumes: ["the data lie in [-1, 1] in every coordinate", "the percentile of n values is the linearly interpolated one of NumPy and JAX", "0 <= p <= 100"]
uses: [orthogonal-projection@m2, quantile-function@m2]
status: full
check: thresholding.check.py
lean: null
sources:
  - {work: saharia-2022-photorealistic-text-image-diffusion, locator: "p. 4", note: "static thresholding as elementwise clipping to [-1, 1], and dynamic thresholding: s a percentile of |x0|, and if s > 1, clip to [-s, s] and divide by s"}
  - {work: saharia-2022-photorealistic-text-image-diffusion, locator: "pp. 42 to 43, Fig. A.31", note: "the reference code of both methods"}
---

### Statement

For an x-prediction $\hat x \in \mathbb{R}^n$, static thresholding $\mathrm{clip}(\hat x, -1, 1)$ is the Euclidean projection of $\hat x$ onto the cube $[-1, 1]^n$, the data range. Dynamic thresholding with percentile $p$,

$$ s = \max\big(\mathrm{perc}_p(|\hat x|),\, 1\big) , \qquad d(\hat x) = \mathrm{clip}(\hat x, -s, s)/s = \mathrm{clip}(\hat x/s, -1, 1) , $$

maps into $[-1, 1]^n$ as well, equals static thresholding whenever $\mathrm{perc}_p(|\hat x|) \le 1$ (and for $p = 0$), and leaves at least $\lfloor (p/100)(n - 1)\rfloor + 1$ of the $n$ coordinates unclipped, each divided by the same $s$, so their signs and ratios are kept; static thresholding clips every coordinate with $|\hat x_i| > 1$.

### Assumptions

1. The data range is $[-1, 1]$. Used in step 1.
2. The linear interpolation of the percentile. Used in step 3.
3. $0 \le p \le 100$. Used in step 3.

### Steps

1. The cube is a product of intervals, so the nearest point minimizes $\sum_i (y_i - \hat x_i)^2$ coordinate by coordinate, and the nearest point of $[-1, 1]$ to $\hat x_i$ is $\mathrm{clip}(\hat x_i, -1, 1)$.
2. $s \ge 1 > 0$, so $\mathrm{clip}(\hat x, -s, s)/s = \mathrm{clip}(\hat x/s, -1, 1) \in [-1, 1]^n$. If $\mathrm{perc}_p(|\hat x|) \le 1$, then $s = 1$ and $d$ is the static threshold; the 0th percentile is the minimum of $|\hat x|$, and if it exceeds 1, $s$ is the minimum and every coordinate is clipped to $\pm 1$, as statically.
3. Sort the absolute values $a_{(0)} \le \dots \le a_{(n-1)}$. With $r = (p/100)(n - 1)$ and $k = \lfloor r \rfloor$, the interpolated percentile lies in $[a_{(k)}, a_{(k+1)}]$, so at least the $k + 1$ values $a_{(0)}, \dots, a_{(k)}$ are at most $s$, and $\mathrm{clip}$ leaves them unchanged before the division.

### Result and check

Both methods keep the sampler's x-prediction in the range of the data, which guidance at large scales pushes out of it, the train-test mismatch that Saharia and colleagues identify. Static thresholding saturates every coordinate beyond the range, while dynamic thresholding shrinks the whole prediction by one factor and saturates only the top $1 - p/100$ fraction.
Imagen's listing (pp. 42 to 43) writes `s = jnp.max(s, 1.0)`; JAX and NumPy read the second argument as an axis and raise a `TypeError`, and the operation the text describes is the elementwise maximum `jnp.maximum(s, 1.0)`. The listing also reduces without `keepdims`, so `s` must be reshaped to broadcast over each sample's coordinates.
The check `thresholding.check.py` compares `static_threshold` with a brute-force nearest point of the cube, and `dynamic_threshold` with the bounds above for random predictions of several sizes and percentiles, including the identity $\mathrm{clip}(\hat x, -s, s)/s = \mathrm{clip}(\hat x/s, -1, 1)$ and that NumPy's `max` called with the listing's arguments raises a `TypeError`.
