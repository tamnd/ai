---
id: segment-memory
derives: [c-segment-memory, c-segment-compute, c-sqrt-optimum, eq-segment-memory, eq-segment-compute]
kind: cost
assumes: ["a chain of n layers h_i = f_i(h_{i-1})", "every activation h_i has the same size, the unit of memory", "the backward of layer i needs its input h_{i-1} and its output h_i", "the chain is cut into k contiguous segments, the longer ones first"]
uses: []
status: full
check: segment-memory.check.py
lean: null
sources:
  - {work: chen-2016-sublinear-memory-cost, locator: "Section 4.3"}
---

### Statement

Cut a chain of $n$ layers into $k$ segments of sizes $s_1 \ge s_2 \ge \dots \ge s_k$, with $\sum_j s_j = n$ and $s_j \in \{\lfloor n/k \rfloor, \lceil n/k \rceil\}$.
Keep the input $h_0$ and the output of each segment in the forward pass, and rebuild the inside of a segment just before its backward.

1. The number of activations held at once is at most $M(n, k) = k + \lfloor n/k \rfloor$, against $n + 1$ when every activation is stored.
2. The schedule runs $F(n, k) = 2n - k$ layer forwards, against $n$.
3. In the continuous model $n/k + k$, the least memory is $2\sqrt{n}$ at $k = \sqrt{n}$. For $n = q^2$, the least integer count is $M = 2q$, reached at $k = q$.

### Assumptions

1. All activations have the same size, so memory is a count of saved activations.
2. A layer's backward needs its input and its output, as for $\tanh$, whose derivative is $1 - h_i^2$.
3. Parameters, gradients and the workspace of one layer are not counted.

### Steps

1. The forward pass keeps $h_0$ and the $k$ segment outputs, which is $k + 1$ activations, and runs $n$ forwards.
2. The backward pass takes the segments from last to first. When it reaches segment $j$, the checkpoints $h_0$ and the outputs of segments $1, \dots, j$ are still held, which is $j + 1$ activations. The outputs of later segments have been freed.
3. Rebuilding the inside of segment $j$ adds $s_j - 1$ activations, because its last output is a checkpoint. While segment $j$ runs, $j + s_j$ activations are held.
4. With the longer segments first, write $n = qk + r$ with $0 \le r < k$. Then $s_j = q + 1$ for $j \le r$ and $s_j = q$ otherwise. The largest $j + s_j$ is $k + q$, because $r + q + 1 \le k + q$. This gives $M(n, k) = k + \lfloor n/k \rfloor$, which is item 1.
5. The rebuild in step 3 costs $s_j - 1$ forwards, so the backward pass adds $\sum_j (s_j - 1) = n - k$. With the $n$ forwards of step 1 this gives $2n - k$, which is item 2. That is at most one extra forward pass.
6. By the inequality of arithmetic and geometric means, $n/k + k \ge 2\sqrt{n}$, with equality at $k = \sqrt{n}$. For $n = q^2$, $\lfloor q^2/k \rfloor > q^2/k - 1$, so $k + \lfloor q^2/k \rfloor > 2q - 1$. The count is an integer, so it is at least $2q$, and $k = q$ gives exactly $2q$. This is item 3.

### Result and check

$M(n, k) = k + \lfloor n/k \rfloor$ and $F(n, k) = 2n - k$.
When $k$ divides $n$, $M$ equals the count $n/k + k$ of Chen et al. (2016, Section 4.3).
The check replays the schedule step by step for every $n \le 80$ and every $k \le n$, and asserts that the peak and the forward count match both formulas.
It also asserts that the least $M$ over $k$ is $2q$ at $k = q$ for $q \le 40$.
For every $n \le 5000$, the least $M$ over all $k$ is $\lfloor \sqrt{4n + 1} \rfloor$. That fact is our own enumeration, not a proof.
