---
id: topk-l0
derives: [c-topk-exact-l0, c-topk-projection, c-topk-orthonormal, c-topk-no-shrink]
kind: proof
assumes: ["a pre-activation vector a in R^n", "1 <= k <= n", "ties broken the same way in TopK and in the comparison"]
uses: []
status: full
check: topk-l0.check.py
lean: null
sources:
  - {work: gao-2024-scaling-sparse-autoencoders, locator: "p. 3"}
  - {work: gao-2024-scaling-sparse-autoencoders, locator: "p. 17"}
---

### Statement

Let $a \in \mathbb{R}^n$ and $1 \le k \le n$, and let $z = \mathrm{ReLU}(\mathrm{TopK}(a))$ keep the $k$ largest entries of $a$, clipped at zero, and set the rest to zero.

1. $\lVert z \rVert_0 \le k$ for every input, with equality when at least $k$ entries of $a$ are positive. No penalty coefficient is involved.
2. $z$ is a closest point to $a$ among nonnegative vectors with at most $k$ nonzero entries: $z \in \arg\min \{ \lVert a - v \rVert_2^2 : v \ge 0, \lVert v \rVert_0 \le k \}$.
3. If the decoder $W \in \mathbb{R}^{m \times n}$ has orthonormal columns ($W^\top W = I$, so $n \le m$) and the encoder is $W^\top$, then $z$ computed from $a = W^\top x$ minimizes $\lVert x - W v \rVert_2^2$ over nonnegative $k$-sparse codes $v$.
4. Each kept value equals its pre-activation; no amount is subtracted from it, unlike the $\lambda/2$ of the derivation `l1-shrinkage`.

### Assumptions

1. $a$ is finite. Ties at the $k$-th place are broken by index, the same way in every form.
2. For part 3, the columns of $W$ are orthonormal. In an overcomplete dictionary ($n > m$) this cannot hold, and part 3 is then only the motivation for starting the encoder at $W^\top$.

### Steps

1. TopK leaves at most $k$ entries nonzero and the ReLU can only zero more, so $\lVert z \rVert_0 \le k$; if the $k$ largest entries are all positive, all $k$ survive.
2. Fix a support $S$ with $|S| \le k$. The best $v \ge 0$ supported on $S$ sets $v_i = \max(a_i, 0)$ for $i \in S$, with cost $\sum_{i \in S} \min(a_i, 0)^2 + \sum_{i \notin S} a_i^2$. Putting $i$ in $S$ lowers the cost by $a_i^2 - \min(a_i, 0)^2 = \max(a_i, 0)^2$, which grows with $a_i$. So the best support is the $k$ largest entries, and on it $v = \max(a, 0)$, which is $z$.
3. With $W^\top W = I$, write $x = W W^\top x + r$ with $W^\top r = 0$. Then $\lVert x - W v \rVert^2 = \lVert r \rVert^2 + \lVert W (W^\top x - v) \rVert^2 = \lVert r \rVert^2 + \lVert a - v \rVert^2$ with $a = W^\top x$. The first term does not depend on $v$, so minimizing over $v$ is step 2.
4. On the kept support $z_i = a_i$ when $a_i > 0$, so the code is not biased toward zero.

### Result and check

The check compares TopK with a brute-force search over all supports for small $n$, verifies the Pythagorean split for random orthonormal decoders, and confirms $\lVert z \rVert_0 = \min(k, \#\{a_i > 0\})$.
