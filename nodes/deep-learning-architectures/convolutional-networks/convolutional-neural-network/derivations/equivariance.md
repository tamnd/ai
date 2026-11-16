---
id: equivariance
derives: [c-equivariance, c-crop-equivariance, c-stride-periodic, c-global-pool-invariance, c-cross-correlation]
kind: identity
assumes: ["a single channel and one spatial axis; several channels and two axes follow coordinate by coordinate, since the sum over channels and the second axis is a sum of terms of the same form", "the shift T_t is (T_t x)[i] = x[i - t]"]
uses: []
status: full
check: equivariance.check.py
lean: null
sources:
  - {work: zhang-2019-shift-invariant-again, locator: "Section 3.1"}
  - {work: lecun-1998-gradient-based-learning, locator: "p. 6"}
---

### Statement

Let $y = x \star w$ be the cross-correlation $y[i] = \sum_{u=0}^{k-1} w[u]\, x[s i + u]$ with stride $s$ (eq-conv).

1. With stride 1 and circular padding, the indices taken modulo $n$, $(T_t x) \star w = T_t (x \star w)$ for every shift $t$: the layer is shift-equivariant.
2. With stride 1 and no padding, if $x' = T_t x$ on the positions that a window of $y[i]$ reads, then $y'[i + t] = y[i]$: every output whose window lies inside the part of the input that moved is moved with it. Zero padding breaks this only for the outputs whose window crosses the border.
3. A pointwise nonlinearity $\sigma$, applied entry by entry, commutes with $T_t$, so stacks of stride-1 layers with circular padding and pointwise nonlinearities are equivariant.
4. With stride $s$, circular padding and $s \mid n$, $(T_{s t} x) \star w = T_t (x \star w)$: the layer is equivariant only to shifts by multiples of $s$, and the output moves by $t$, not $s t$. The same holds for max or average pooling with stride $s$.
5. A global pooling $g(x) = \max_i x[i]$ or $\frac{1}{n}\sum_i x[i]$ is invariant to circular shifts, $g(T_t x) = g(x)$, so global pooling after equivariant layers gives a shift-invariant output.
6. The convolution $(x * v)[i] = \sum_u v[u]\, x[i - u]$ with a kernel $v$ of support $\{0, \dots, k-1\}$, computed at the positions where it is defined, equals the cross-correlation with the flipped kernel $w[u] = v[k - 1 - u]$, shifted by $k - 1$.

### Assumptions

1. Circular padding means $x[i] = x[i \bmod n]$, so $T_t$ maps the signal to itself. Used in steps 1, 4 and 5.
2. $\sigma$ acts on each entry alone. Used in step 3.

### Steps

1. Stride 1, circular: $((T_t x) \star w)[i] = \sum_u w[u]\, x[i + u - t] = (x \star w)[i - t] = (T_t (x \star w))[i]$. This is statement 1.
2. No padding: $y'[i + t] = \sum_u w[u]\, x'[i + t + u] = \sum_u w[u]\, x[i + u] = y[i]$, using $x'[m] = x[m - t]$ for the positions $m = i + t + u$ that the window reads. With zero padding the same sum holds whenever these positions are inside the input; for a window that crosses the border, the padded zeros stay fixed while the signal moves, and the equality fails in general. This is statement 2.
3. $(\sigma(T_t x))[i] = \sigma(x[i - t]) = (T_t\, \sigma(x))[i]$, and composing equivariant maps gives an equivariant map: $f(g(T_t x)) = f(T_t\, g(x)) = T_t f(g(x))$. This is statement 3.
4. Stride $s$: $((T_{st} x) \star w)[i] = \sum_u w[u]\, x[s i + u - s t] = \sum_u w[u]\, x[s (i - t) + u] = (x \star w)[i - t]$. A shift that is not a multiple of $s$ moves the windows across the sampling grid, and the outputs are other sums, not a shift of the old ones. Max and average pooling with stride $s$ have the same index $s i + u$ with $\max$ or a mean in place of the weighted sum, so the same steps apply. This is statement 4.
5. Global pooling: $T_t$ permutes the entries of $x$, and the maximum and the mean of a set do not depend on its order. This is statement 5.
6. Flip: with $w[u] = v[k - 1 - u]$ and $u' = k - 1 - u$, $(x \star w)[i] = \sum_{u'} v[u']\, x[i + k - 1 - u'] = (x * v)[i + k - 1]$. This is statement 6.

### Result and check

$$ (T_t x) \star w = T_t (x \star w) \quad \text{(stride 1, circular padding)} $$

The check `equivariance.check.py` tests every statement on random multichannel images in two dimensions with the NumPy reference: circular equivariance for all shifts, equivariance of valid outputs under shifts of the input, the failure at the border with zero padding, periodic-$s$ equivariance for convolution and max pooling, a stack of two layers with ReLU, invariance after global pooling, and the flip against `numpy.convolve`.
