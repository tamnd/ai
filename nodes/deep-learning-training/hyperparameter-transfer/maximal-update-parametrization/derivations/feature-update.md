---
id: feature-update
derives: [c-aligned-lln, c-sp-readout-sgd, c-sp-hidden-sgd, c-sp-hidden-adam, c-sp-input-sgd, c-mup-constant]
kind: closed-form
assumes: ["a two-hidden-layer tanh MLP whose input and output dimensions stay fixed while the hidden width n grows", "one step of SGD or Adam from initialization at a fixed base learning rate eta", "orders of magnitude of per-coordinate sizes to first order in eta, holding the batch fixed"]
uses: []
status: full
check: feature-update.check.py
lean: null
sources:
  - {work: yang-2022-tensor-programs-v-hyperparameter-transfer, locator: "p. 47"}
  - {work: yang-2022-tensor-programs-v-hyperparameter-transfer, locator: "p. 48"}
  - {work: yang-2022-tensor-programs-v-hyperparameter-transfer, locator: "p. 5"}
  - {work: yang-2020-feature-learning-infinite-width-networks, locator: "p. 13"}
---

### Statement

Take $f(x) = V \phi(W \phi(U x + b_1) + b_2)$ with $\phi = \tanh$, $U \in \mathbb{R}^{n \times d}$, $W \in \mathbb{R}^{n \times n}$, $V \in \mathbb{R}^{k \times n}$, with $d$ and $k$ fixed and $n \to \infty$.
Write $h_1 = U x + b_1$, $x_1 = \phi(h_1)$, $h_2 = W x_1 + b_2$, $x_2 = \phi(h_2)$ and $\chi = \partial L / \partial f$.

1. A matrix $\Delta W = -\eta\, a\, x^\top$ built from its own input moves $\Delta W x = -\eta (x^\top x)\, a$, which is $\Theta(n \eta)$ per coordinate when $x$ has $\Theta(1)$ coordinates; a matrix with independent entries of the same size moves it only $\Theta(\sqrt{n}\,\eta)$.
2. Under SP (every learning rate $\eta$, $V$ and $W$ with variance $\sigma^2/n$), one SGD step changes $f$ by $\Theta(n \eta)$, $h_2$ by $\Theta(\sqrt{n}\,\eta)$ and $h_1$ by $\Theta(\eta / \sqrt{n})$; one Adam step changes $h_2$ by $\Theta(n \eta)$ and $f$ by at least that.
3. Under $\mu$P (eq-mup-init with eq-mup-sgd or eq-mup-adam), $f$, $h_2$ and $h_1$ all change by $\Theta(\eta)$, independently of $n$.

The often-quoted shorthand that SP moves hidden features by $\Theta(n \eta)$ is exact for Adam and for the readout under SGD; for a hidden layer under SGD it is $\Theta(\sqrt{n}\,\eta)$, as Yang et al. state on p. 48 of Tensor Programs V.

### Assumptions

1. The entries of $U$, $W$, $V$ are independent zero-mean Gaussians, biases start at 0, and $\tanh$ has bounded derivative with $\phi'(h) = \Theta(1)$ for $h = \Theta(1)$.
2. The input $x$ and the batch are fixed, so $x^\top x = \Theta(d) = \Theta(1)$, and $\chi = \Theta(1)$ at initialization (true under SP, where $f_0 = \Theta(1)$, and under $\mu$P, where $f_0 \to 0$ and $\chi \to -y$).
3. Sizes are per coordinate, to first order in $\eta$: the change of a preactivation is the change caused by its own weights plus the propagated change of its input, and each is bounded separately.
4. Adam's first bias-corrected step with $\epsilon = 0$ is $-\eta\,\mathrm{sign}(g)$ coordinatewise; for a batch of one example the gradient of a weight matrix is the rank-one $\delta\, x^\top$, so its sign is $\mathrm{sign}(\delta)\,\mathrm{sign}(x)^\top$.

### Steps

1. For $\Delta W = -\eta\, a\, x^\top$, $\Delta W x = -\eta\, a \sum_j x_j^2$. The sum has $n$ positive terms, so by the law of large numbers it is $\Theta(n)$ and $(\Delta W x)_i = \Theta(n \eta |a_i|)$. If instead the entries of $\Delta W$ are independent with size $\eta |a_i|$ and zero mean, $\sum_j \Delta W_{ij} x_j$ is a sum of $n$ uncorrelated terms, of size $\Theta(\sqrt{n})\, \eta |a_i|$ by the central limit theorem. Gradients are of the aligned kind: $\partial L / \partial W = \delta\, x^\top$ with $\delta = \partial L / \partial h$. This is the LLN-versus-CLT mismatch of Yang and Hu (p. 13).
2. SP, SGD, readout. $\Delta V = -\eta\, \chi\, x_2^\top$, so $\Delta V x_2 = -\eta\, \chi\, |x_2|^2 = \Theta(n \eta)$, since $x_2$ has $\Theta(1)$ coordinates.
3. SP, SGD, hidden. $\delta_2 = (V^\top \chi) \odot \phi'(h_2)$ has coordinates of size $\Theta(1/\sqrt{n})$, the size of $V$'s entries. $\Delta W x_1 = -\eta\, \delta_2\, |x_1|^2 = \Theta(\eta \cdot n / \sqrt{n}) = \Theta(\sqrt{n}\,\eta)$. The propagated term $W \Delta x_1$ is a random matrix times a vector and is smaller (step 4).
4. SP, SGD, input. $\delta_1 = (W^\top \delta_2) \odot \phi'(h_1)$; $W^\top \delta_2$ sums $n$ terms of size $1/\sqrt{n} \cdot 1/\sqrt{n}$ with random signs, so it is $\Theta(1/\sqrt{n})$. Then $\Delta U x = -\eta\, \delta_1\, |x|^2 = \Theta(\eta / \sqrt{n})$, because $|x|^2 = \Theta(d)$ does not grow. The first layer stops learning.
5. SP, Adam. Every entry of $\Delta W$ is $\pm \eta$ and, by assumption 4, aligned with $x_1$: $\Delta W x_1 = -\eta\, \mathrm{sign}(\delta_2) \sum_j |x_{1,j}| = \Theta(n \eta)$. The same argument gives $\Delta V x_2 = \Theta(n \eta)$, and the propagated term $V \Delta x_2$ adds to it with the same sign, because $\mathrm{sign}(\delta_2)$ follows $\mathrm{sign}(V^\top \chi)$. So $f$ moves at least $\Theta(n \eta)$; Yang et al. state the $\Theta(n)$ hidden blow-up on p. 47.
6. $\mu$P, SGD. $V$ has variance $\sigma^2 / (n \cdot n/n_0)$, so entries are $\Theta(1/n)$, and its learning rate is $\eta\, n_0/n$. Readout: $\Delta V x_2 = -\eta (n_0/n)\, \chi\, |x_2|^2 = \Theta(\eta)$. Hidden: $\delta_2 = \Theta(1/n)$, so $\Delta W x_1 = -\eta\, \delta_2\, |x_1|^2 = \Theta(\eta)$. Input: $W^\top \delta_2 = \Theta(\sqrt{n} \cdot 1/\sqrt{n} \cdot 1/n) = \Theta(1/n)$, the input rate is $\eta\, n/n_0$, so $\Delta U x = \Theta(\eta\, (n/n_0) \cdot (1/n)) = \Theta(\eta)$. Propagated: $\Delta x_{2,j} \approx -\eta\, n\, \phi'^2\, V_j \chi$ up to $\Theta(1)$ factors, so $V \Delta x_2 \approx -\eta\, n\, \chi \sum_j V_j^2 \phi'^2 = \Theta(\eta\, n \cdot n \cdot n^{-2}) = \Theta(\eta)$.
7. $\mu$P, Adam. The hidden and readout rates are $\eta\, n_0/n$, so $\Delta W x_1 = \pm\eta (n_0/n) \sum_j |x_{1,j}| = \Theta(\eta)$ and likewise $\Delta V x_2$; the input rate is $\eta$ and $\Delta U x = \pm \eta \sum_j |x_j| = \Theta(\eta d)$; and $V \Delta x_2$ sums $n$ terms of size $1/n \cdot \eta$, which is $\Theta(\eta)$.
8. At $n = n_0$ the $\mu$P and SP rules coincide (p. 5 of Tensor Programs V), so the two curves start at the same point and only their slopes in $n$ differ.

### Result and check

The check builds a rank-one aligned update and a random one at widths 64 to 4096 and fits log-log slopes (about 1 and 1/2). It then verifies the exact readout identity $\Delta V x_2 = -\eta\, \chi\, |x_2|^2$ and trains the two-hidden-layer tanh MLP for one step at widths 64 to 4096. The fitted slopes of the input, hidden and output changes are about $-1/2$, $1/2$ and $1$ for SP with SGD, the hidden slope is about $1$ for SP with Adam, and every slope is within $0.1$ of $0$ for $\mu$P with SGD and with Adam.
