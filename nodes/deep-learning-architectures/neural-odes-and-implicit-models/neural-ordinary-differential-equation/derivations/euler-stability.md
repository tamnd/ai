---
id: euler-stability
derives: [c-euler-stable, c-antisymmetric-unstable, c-diffusion]
kind: bound
assumes: ["the field is linear, f(h) = K h, with K diagonalizable", "Euler steps of the size dt > 0"]
uses: [euler-resnet]
status: full
check: euler-stability.check.py
lean: null
sources:
  - {work: haber-2017-stable-architectures-deep-neural-networks, locator: "Section 3"}
  - {work: haber-2017-stable-architectures-deep-neural-networks, locator: "Section 4"}
---

### Statement

For $dh/dt = K h$ with eigenvalues $\lambda_i$ of $K$:

1. Euler steps $h_{k+1} = (I + \Delta t K) h_k$ do not grow along the eigenvector of $\lambda_i$ if and only if $\lvert 1 + \Delta t \lambda_i \rvert \le 1$.
2. If $K$ is antisymmetric, $K^\top = -K$, the exact flow keeps $\lVert h \rVert$ fixed, but every Euler step multiplies the component along an eigenvalue $i\omega$, $\omega \ne 0$, by $\sqrt{1 + \Delta t^2 \omega^2} > 1$, so a deep stack of such steps grows without bound.
3. With $K - \gamma I$, $\gamma > 0$, the eigenvalues are $-\gamma + i\omega$, and the Euler steps do not grow if and only if $\Delta t (\gamma^2 + \omega^2) \le 2\gamma$.

### Assumptions

1. $K = V \Lambda V^{-1}$, so that in the coordinates $V^{-1} h$ the steps act on each coordinate separately. Used in all steps.

### Steps

1. In the eigencoordinates, one Euler step multiplies coordinate $i$ by $1 + \Delta t \lambda_i$ and $k$ steps by $(1 + \Delta t \lambda_i)^k$, which stays bounded in $k$ exactly when $\lvert 1 + \Delta t \lambda_i \rvert \le 1$. This is statement 1.
2. For antisymmetric $K$, $d \lVert h \rVert^2 / dt = 2 h^\top K h = 0$ because $h^\top K h = (h^\top K h)^\top = -h^\top K h$. The eigenvalues of a real antisymmetric matrix are purely imaginary, $\lambda = i\omega$, and $\lvert 1 + i \Delta t \omega \rvert^2 = 1 + \Delta t^2 \omega^2$, which is above 1 when $\omega \ne 0$. This is statement 2.
3. $\lvert 1 - \Delta t \gamma + i \Delta t \omega \rvert^2 = 1 - 2 \Delta t \gamma + \Delta t^2 (\gamma^2 + \omega^2)$, which is at most 1 if and only if $\Delta t (\gamma^2 + \omega^2) \le 2 \gamma$. This is statement 3.

### Result and check

$$ \lvert 1 + \Delta t \lambda_i \rvert \le 1, \qquad \lvert 1 + i \Delta t \omega \rvert = \sqrt{1 + \Delta t^2 \omega^2} > 1, \qquad \Delta t (\gamma^2 + \omega^2) \le 2 \gamma $$

The check `euler-stability.check.py` runs Euler steps of the NumPy reference on random antisymmetric fields, with and without the shift $-\gamma I$, and compares the growth of the norm with the factors of statements 2 and 3.
