---
id: recursive-bisection
derives: [c-recursive-memory, c-recursive-compute, eq-recursive-compute]
kind: cost
assumes: ["the chain and the memory unit of segment-memory", "the input h_0 and the output h_n are held throughout", "a range of length t > 1 is split at floor(t/2)"]
uses: []
status: full
check: recursive-bisection.check.py
lean: null
sources:
  - {work: chen-2016-sublinear-memory-cost, locator: "Section 4.4"}
---

### Statement

Run the backward over layers $lo, \dots, hi - 1$, given $h_{lo}$ and $h_{hi}$, as follows.
If $t = hi - lo = 1$, do the backward of the layer.
Otherwise, run $\lfloor t/2 \rfloor$ forwards from $h_{lo}$ to get $h_{mid}$ and save it.
Do the right half, then the left half, and then free $h_{mid}$.

1. At most $2 + \lceil \log_2 n \rceil$ activations are held at once.
2. The schedule runs $n + R(n)$ layer forwards, where $R(1) = 0$ and $R(t) = \lfloor t/2 \rfloor + R(\lceil t/2 \rceil) + R(\lfloor t/2 \rfloor)$. For $n = 2^p$, this is $n + \tfrac{n}{2}\log_2 n$.

### Assumptions

1. These are the assumptions of segment-memory: equal activation sizes, and a layer's backward needs its input and its output.
2. The forward pass keeps only $h_0$ and $h_n$.

### Steps

1. Each open call holds one midpoint. The calls nest, so the number held is the depth of the call stack. Let $d(t)$ be the depth for a range of length $t$. Then $d(1) = 0$ and $d(t) = 1 + d(\lceil t/2 \rceil)$, because the larger half has length $\lceil t/2 \rceil$. So $d(t) = \lceil \log_2 t \rceil$. Adding $h_0$ and $h_n$ gives item 1.
2. A call on a range of length $t > 1$ runs $\lfloor t/2 \rfloor$ forwards and then calls itself on both halves. That is the recurrence for $R$. The forward pass adds $n$.
3. For $t = 2^p$, $R(2^p) = 2^{p-1} + 2R(2^{p-1})$. Divide by $2^p$ to get $R(2^p)/2^p = \tfrac12 + R(2^{p-1})/2^{p-1}$. So $R(2^p) = p\,2^{p-1} = \tfrac{n}{2}\log_2 n$.

### Result and check

The memory is $O(\log n)$ and the compute is $O(n \log n)$.
That is the regime of the recursion $g(n) = k + g(n/(k+1))$ of Chen et al. (2016, Section 4.4), whose solution for $k = 1$ is $\log_2 n$.
The check simulates the schedule for every $n \le 512$. It asserts that the peak equals $2 + \lceil \log_2 n \rceil$, that the forward count equals $n + R(n)$, and that every layer's backward runs once, from last to first.
For $n = 2^p$ with $p \le 16$, it asserts the closed form $n + np/2$.
