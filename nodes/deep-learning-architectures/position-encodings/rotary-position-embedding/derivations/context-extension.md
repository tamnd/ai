---
id: context-extension
derives: [c-pi-fractional, c-ntk-ratio, c-by-parts-ends, c-yarn-scaling-trick, c-yarn-numbers, eq-pi, eq-ntk, eq-by-parts, eq-yarn-temp]
kind: identity
assumes: ["d is even", "theta_j = b^(-2j/d) with b > 1", "a scale factor s >= 1"]
uses: [softmax-regression@al, dot-product@m2]
status: full
check: context-extension.check.py
lean: null
sources:
  - {work: chen-2023-position-interpolation, locator: "Section 2.3"}
  - {work: peng-2023-yarn, locator: "Section 2.3"}
  - {work: peng-2023-yarn, locator: "Definition 1"}
  - {work: peng-2023-yarn, locator: "Section 3.2"}
  - {work: peng-2023-yarn, locator: "Section 3.4"}
---

### Statement

Let $L$ be the trained context length, $L'$ the target length and $s = L'/L \ge 1$.

1. Position interpolation (eq-pi) rotates position $m$ by $\mathbf{R}_{m/s}$. This is the same map as RoPE with every angle $\theta_j$ replaced by $\theta_j / s$. The score then depends on $(n - m)/s$, and every position in $[0, L')$ is sent into $[0, L)$.
2. The NTK-aware base $b' = b\, s^{d/(d-2)}$ (eq-ntk) gives angles $\theta'_j = \theta_j\, s^{-2j/(d-2)}$. The highest frequency $\theta_0$ is unchanged and the lowest, $\theta_{d/2-1}$, is divided by $s$, as under position interpolation.
3. NTK-by-parts (eq-by-parts) gives $\theta_j$ unchanged when $\gamma = 1$ and $\theta_j / s$ when $\gamma = 0$.
4. Multiplying both the rotated query and the rotated key by $\sqrt{1/t}$ divides every score by $t$, which is the temperature of eq-yarn-temp. It needs no change to the attention code.
5. The recommended value $\sqrt{1/t} = 0.1 \ln s + 1$ gives $1.2079$ at $s = 8$, $1.2773$ at $s = 16$ and $1.3466$ at $s = 32$, to four decimals.

### Assumptions

1. $d$ is even and $\theta_j = b^{-2j/d}$ for $j = 0, \dots, d/2 - 1$, with $b > 1$. Part 2 uses $d > 2$, so that $d - 2 > 0$.
2. $s \ge 1$. Part 1 holds for any $s > 0$; the claim that positions land in $[0, L)$ uses $s = L'/L$.
3. The ramp $\gamma$ of YaRN takes values in $[0, 1]$. Its inputs, $\alpha$ and $\beta$, are not used.

### Steps

1. The block $j$ of $\mathbf{R}_{m/s}$ is the rotation by $(m/s)\theta_j = m(\theta_j/s)$, so the two maps are equal. By part 4 of the derivation relative-position, the score is $\mathbf{q}^\top \mathbf{R}_{(n-m)/s} \mathbf{k}$. For $0 \le m < L'$, $0 \le m/s < L'/s = L$. This is what Chen et al. write as $\mathbf{f}'(\mathbf{x}, m) = \mathbf{f}(\mathbf{x}, mL/L')$ in their Equation 4.
2. $\theta'_j = (b\, s^{d/(d-2)})^{-2j/d} = b^{-2j/d}\, s^{-2j/(d-2)} = \theta_j\, s^{-2j/(d-2)}$, since powers of a product split and $(d/(d-2))(2j/d) = 2j/(d-2)$. At $j = 0$ the factor is $s^0 = 1$. At $j = d/2 - 1$ the exponent is $-2(d/2-1)/(d-2) = -1$, so the factor is $1/s$. Peng et al. (2023, Section 3.1) describe the result as spreading the interpolation pressure across the dimensions.
3. $h(\theta_j) = (1 - \gamma)\theta_j / s + \gamma \theta_j$. At $\gamma = 1$ this is $\theta_j$, and at $\gamma = 0$ it is $\theta_j / s$. Between the two it is a weighted average of the two angles.
4. Write $c = \sqrt{1/t}$. Then $(c\,\mathbf{R}_m\mathbf{q})^\top (c\,\mathbf{R}_n\mathbf{k}) = c^2 (\mathbf{R}_m\mathbf{q})^\top(\mathbf{R}_n\mathbf{k}) = (\mathbf{R}_m\mathbf{q})^\top(\mathbf{R}_n\mathbf{k}) / t$. Dividing by $\sqrt{d}$ afterwards gives the argument of eq-yarn-temp. Because $c$ can be folded into the cosine and sine tables of eq-rope-efficient, the attention code stays the same.
5. Direct evaluation of $0.1 \ln s + 1$ gives the numbers of part 5. Peng et al. print $\approx 1.208$ for $s = 8$ in their Equation 25, which agrees.

### Result and check

Position interpolation, the NTK-aware base and NTK-by-parts all keep the RoPE form and change only the angles.
They differ in which angles they shrink: all of them by $s$, the low frequencies more than the high ones, or a split by wavelength.
YaRN adds one scalar on the query and the key.
For $d = 128$, $b = 10000$ and $L = 2048$, my own computation gives wavelengths $\lambda_j = 2\pi/\theta_j$ from $2\pi \approx 6.28$ to about $54410$; with $\alpha = 1$ and $\beta = 32$, the 23 pairs $j \ge 41$ have $r = L/\lambda_j < 1$ and are interpolated in full, and the 17 pairs $j \le 16$ have $r > 32$ and are left alone.

The check `context-extension.check.py` asserts parts 1 to 5 and the counts of the last paragraph.
The NumPy tests `test_interpolation_is_fractional_positions` and `test_ntk_aware_base` check parts 1 and 2 against the reference.
