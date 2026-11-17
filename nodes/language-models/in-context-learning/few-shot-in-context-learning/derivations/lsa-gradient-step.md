---
id: lsa-gradient-step
derives: [c-lsa-step, c-lsa-readout, c-lsa-stack, c-lsa-rescale, c-gd-averaging, eq-lsa, eq-gd-step]
kind: proof
assumes: ["tokens e_j = (x_j, y_j) with x_j in R^{d_x} and y_j in R^{d_y}", "a linear self-attention layer without softmax or bias, whose keys and values come from the N context tokens only", "W_K = W_Q = [[I, 0], [0, 0]], W_V = [[0, 0], [W_0, -I]] and P = (eta / N) I, the construction of von Oswald et al."]
uses: [ordinary-least-squares@m2, stochastic-gradient-descent@m2]
status: full
check: lsa-gradient-step.check.py
lean: null
sources:
  - {work: von-oswald-2023-transformers-learn-in-context-by-gradient, locator: "equation 3"}
  - {work: von-oswald-2023-transformers-learn-in-context-by-gradient, locator: "p. 4"}
  - {work: von-oswald-2023-transformers-learn-in-context-by-gradient, locator: "p. 13"}
  - {work: garg-2022-what-can-transformers-learn-in-context, locator: "p. 4"}
---

### Statement

Let $(x_1, y_1), \dots, (x_N, y_N)$ be the in-context examples, with $x_i \in \mathbb{R}^{d_x}$ and $y_i \in \mathbb{R}^{d_y}$, and let $L(W) = \frac{1}{2N} \sum_i \lVert W x_i - y_i \rVert^2$.
One gradient step from $W$ with learning rate $\eta$ gives $W - \eta \nabla L(W) = W + \Delta W$ with $\Delta W = -\frac{\eta}{N} \sum_i (W x_i - y_i) x_i^\top$.
Write tokens as $e = (x, y) \in \mathbb{R}^{d_x + d_y}$.
A linear self-attention layer with context tokens $e_1, \dots, e_N$ maps any token $e$ to

$$ e \;\mapsto\; e + P W_V \sum_{i=1}^{N} e_i \, e_i^\top W_K^\top W_Q \, e . $$

Take $W_K = W_Q = \begin{pmatrix} I & 0 \\ 0 & 0 \end{pmatrix}$, $W_V = \begin{pmatrix} 0 & 0 \\ W_0 & -I \end{pmatrix}$ and $P = \frac{\eta}{N} I$.

1. **One layer.** Each context token $(x_j, y_j)$ becomes $(x_j,\, y_j - \Delta W_1 x_j)$, where $\Delta W_1$ is the gradient step from $W_0$.
2. **Read-out.** A query token $(x_q, -W_0 x_q)$ becomes $(x_q, -(W_0 + \Delta W_1) x_q)$; minus its $y$ entry is the prediction of the model after one step.
3. **Layers are steps.** Let $W_k = W_{k-1} - \eta \nabla L(W_{k-1})$. After $K$ layers with the same weights, applied each time to the updated context tokens, each context token is $(x_j,\, y_j - (W_K - W_0) x_j)$ and the query token is $(x_q, -W_K x_q)$.
4. **Rescaling.** The layer depends on its weights only through $A = P W_V$ and $B = W_K^\top W_Q$, so $(sA, B/s)$ gives the same map for any $s \neq 0$.
5. **Averaging.** With $W_0 = 0$ and $\eta = 1$ the prediction is $\frac{1}{N} \sum_i y_i\, x_i^\top x_q$: the average of the $y_i x_i$, used as an estimate of $w$, applied to $x_q$.

### Proof

The update of a token $e$ is $A \sum_i e_i (e_i^\top B e)$, a matrix times a sum of vectors with scalar weights $e_i^\top B e$; it is linear in $A$ and in $B$, which gives part 4.

For part 1, $B = W_K^\top W_Q$ keeps the $x$ block and zeroes the $y$ block, so $B e = (x, 0)$ and $e_i^\top B e = x_i^\top x$.
The value of a context token is $W_V e_i = (0,\, W_0 x_i - y_i)$, and $P$ multiplies it by $\eta / N$.
So the update of $e = (x, y)$ is

$$ \Big(0,\; \frac{\eta}{N} \sum_i (W_0 x_i - y_i)\, x_i^\top x \Big) = \big(0,\; -\Delta W_1\, x\big). $$

The $x$ part never changes, because the top rows of $W_V$ are zero.
For a context token this gives part 1, and for the query token $(x_q, -W_0 x_q)$ it gives $-W_0 x_q - \Delta W_1 x_q$, which is part 2.

For part 3, suppose that before layer $k$ every context token is $(x_i,\, y_i - (W_{k-1} - W_0) x_i)$; this holds for $k = 1$.
The value of such a token is $(0,\, W_0 x_i - y_i + (W_{k-1} - W_0) x_i) = (0,\, W_{k-1} x_i - y_i)$, so the layer adds

$$ \Big(0,\; \frac{\eta}{N} \sum_i (W_{k-1} x_i - y_i)\, x_i^\top x \Big) = \big(0,\; -(W_k - W_{k-1})\, x\big) $$

to every token, since $W_k - W_{k-1} = -\frac{\eta}{N} \sum_i (W_{k-1} x_i - y_i) x_i^\top$.
The $y$ entry of a context token becomes $y_j - (W_{k-1} - W_0) x_j - (W_k - W_{k-1}) x_j = y_j - (W_k - W_0) x_j$, which completes the induction, and the query entry goes from $-W_{k-1} x_q$ to $-W_k x_q$.

For part 5, $W_0 = 0$ and $\eta = 1$ give $\Delta W_1 = \frac{1}{N} \sum_i y_i x_i^\top$, and the prediction is $\Delta W_1 x_q$. $\square$

### Check

`lsa-gradient-step.check.py` builds the matrices of the construction and compares, on random problems, the layer output with the gradient step for one layer and for $K = 1$ to $5$ stacked layers, the read-out of the query, the rescaling by $s = -3$, $0.5$ and $7$, and the averaging estimate.
It also checks that letting the query token act as a key and value changes the output when $W_0 \neq 0$ but not when $W_0 = 0$, the deviation that von Oswald et al. note.
