---
id: k-small-steps
derives: [c-k-step-identity, c-k-step-second-order, c-k-step-only-k, c-k-step-fails]
kind: closed-form
assumes: ["the per-example loss is a least-squares loss, so each batch gradient is affine in w", "plain SGD without momentum or weight decay", "the k small batches are disjoint and their union is the large batch"]
uses: []
status: full
check: k-small-steps.check.py
lean: null
sources:
  - {work: goyal-2017-accurate-large-minibatch-sgd, locator: "p. 2"}
  - {work: goyal-2017-accurate-large-minibatch-sgd, locator: "p. 3"}
---

### Statement

Let $B_0, \dots, B_{k-1}$ be disjoint batches of $n$ examples, $g_j(w)$ the mean gradient over $B_j$, and $H_j$ its Hessian.
Run $k$ SGD steps $w_{t+j+1} = w_{t+j} - \eta\, g_j(w_{t+j})$, and separately one step $\hat w_{t+1} = w_t - \hat\eta\, \frac1k \sum_j g_j(w_t)$ on the union.
For least squares, where $g_j(w) = H_j w - b_j$:

(i) with $\hat\eta = k\eta$, $w_{t+k} - \hat w_{t+1} = \eta^2 \sum_{j=1}^{k-1} H_j \sum_{i<j} g_i(w_{t+i})$ exactly;

(ii) its norm is at most $\tfrac{k-1}{2}\, \eta\, \lambda_{\max} \cdot k\eta \max_i \lVert g_i(w_{t+i}) \rVert$, where $\lambda_{\max}$ is the largest eigenvalue of the $H_j$, so relative to the size of a step the gap is first order in $\eta$ and vanishes as $\eta \to 0$;

(iii) any $\hat\eta \ne k\eta$ leaves a gap $(k\eta - \hat\eta)\, \frac1k \sum_j g_j(w_t)$ that is first order in $\eta$ itself;

(iv) the large step is stable only if $k\eta\lambda_{\max} < 2$ for the Hessian of the union, while the small steps need only $\eta\lambda_{\max} < 2$, so for $2/k < \eta\lambda_{\max} < 2$ the small-batch run converges and the linearly scaled large-batch run diverges.

### Assumptions

1. The loss is least squares, so each $g_j$ is affine in $w$ with constant Hessian $H_j \succeq 0$. For a general smooth loss the same steps hold with $H_j$ replaced by a mean of Hessians along the path, and the gap stays second order in $\eta$ when those are bounded.
2. Plain SGD: no momentum, no weight decay, no batch normalization coupling the examples of a batch.
3. Goyal et al.'s condition $\nabla l(x, w_t) \approx \nabla l(x, w_{t+j})$ for $j < k$ is what (ii) quantifies.

### Steps

1. Unrolling the small steps gives $w_{t+j} - w_t = -\eta \sum_{i<j} g_i(w_{t+i})$, and so $w_{t+k} = w_t - \eta \sum_j g_j(w_{t+j})$.
2. The large step is $\hat w_{t+1} = w_t - \frac{\hat\eta}{k} \sum_j g_j(w_t)$. With $\hat\eta = k\eta$ the difference is $-\eta \sum_j [g_j(w_{t+j}) - g_j(w_t)]$.
3. For least squares $g_j(w_{t+j}) - g_j(w_t) = H_j (w_{t+j} - w_t) = -\eta H_j \sum_{i<j} g_i(w_{t+i})$, which gives (i).
4. Bounding each term by $\lambda_{\max} \max_i \lVert g_i \rVert$ and counting $\sum_{j<k} j = k(k-1)/2$ terms gives (ii).
5. With another $\hat\eta$ the difference gains the term $(\hat\eta - k\eta) \frac1k \sum_j g_j(w_t)$, which is of the same order as the step itself, giving (iii).
6. On the quadratic, gradient descent with rate $\alpha$ multiplies the error along an eigenvector of eigenvalue $\lambda$ by $1 - \alpha\lambda$, which shrinks only when $\alpha\lambda < 2$. Setting $\alpha = \eta$ and $\alpha = k\eta$ gives (iv).

### Result and check

The check builds a least-squares problem with 4,096 examples in 8 dimensions and $k = 8$ batches of 32.
It confirms the identity (i) to $10^{-12}$ and the bound (ii) at three rates; that halving $\eta$ halves the relative gap; that $\hat\eta = \sqrt{k}\,\eta$, $k\eta/2$ and $2k\eta$ leave gaps more than 20 times larger than $k\eta$; and, over whole runs, that 800 small steps at $\eta = 0.05/\lambda_{\max}$ and 100 scaled steps reach the same loss within 1%, while at $\eta = 1/\lambda_{\max}$ the small-batch run converges and the scaled run, with $k\eta\lambda_{\max} = 8$, diverges.
