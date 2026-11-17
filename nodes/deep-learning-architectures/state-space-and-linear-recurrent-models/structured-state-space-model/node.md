---
tag: MDN3YW
kind: architecture
slug: structured-state-space-model
name: Structured state space model
aliases: [deep state space model, SSM layer, S4]
area: deep-learning-architectures/state-space-and-linear-recurrent-models
level: core
summary: >-
  A sequence layer built on a discretized linear state space model with a structured state matrix, so that it runs either as a recurrence or as a long convolution.
uses: [matrix-exponential@m2, eigenvalue@m2, diagonal-matrix@m2, matrix-multiplication@m2]
related_to: [transformer, backpropagation-through-time@al]
history:
  year: 2021
  people: [Albert Gu, Karan Goel, Christopher Ré]
  note: "Gu, Goel and Ré (2021), arXiv 2111.00396, building on HiPPO (Gu et al. 2020, arXiv 2008.07669)."
prereqs: [matrix-exponential@m2, eigenvalue@m2, eigendecomposition@m2, backpropagation-through-time@al, vanishing-gradient-problem@al]
same_as:
  wikipedia: null
  arxiv: ["2111.00396"]
sources:
  - {paper: "Gu, Goel and Ré (2021)", title: "Efficiently Modeling Long Sequences with Structured State Spaces", locator: "arXiv 2111.00396v3, Sections 2 to 4, Tables 4, 8 and 10, Appendix D.5", note: "S4, the canonical reference"}
  - {paper: "Gu, Dao, Ermon, Rudra and Ré (2020)", title: "HiPPO: Recurrent Memory with Optimal Polynomial Projections", locator: "arXiv 2008.07669v2, Section 3 and Appendix B.3", note: "the HiPPO matrix and the discretization rules"}
  - {paper: "Gupta, Gu and Berant (2022)", title: "Diagonal State Spaces are as Effective as Structured State Spaces", locator: "arXiv 2203.14343v3, abstract", note: "DSS"}
  - {paper: "Gu, Gupta, Goel and Ré (2022)", title: "On the Parameterization and Initialization of Diagonal State Space Models", locator: "arXiv 2206.11893v2, Sections 3 and 4", note: "S4D"}
  - {paper: "Orvieto, Smith, Gu, Fernando, Gulcehre, Pascanu and De (2023)", title: "Resurrecting Recurrent Neural Networks for Long Sequences", locator: "arXiv 2303.06349v1, Sections 1, 3 and 4", note: "the Linear Recurrent Unit"}
  - {paper: "Gu and Dao (2023)", title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", locator: "arXiv 2312.00752v2, Sections 2 and 3", note: "the selective SSM"}
---

A structured state space model (SSM) is a sequence layer built on a linear differential equation with a hidden state.
It is discretized with a learned step size, and its state matrix has a structure that makes it cheap to compute.
It runs as a recurrence, one step at a time, or as a convolution over the whole sequence, and the two give the same output.
S4, the first such layer, was introduced by Gu, Goel and Ré in 2021 for sequences with dependencies over thousands of steps.

## In plain words

Picture a small bank of $N$ numbers, the state, that summarizes everything the layer has seen.
At each step the state decays and rotates a little, and the new input is added in.
The output is a weighted sum of the state.
All of this is linear, so the output is a fixed weighted sum of the past inputs, and the weights form a long filter.
A layer can therefore read a whole sequence in parallel with that filter during training, and generate one step at a time from the state at inference.
The art is in choosing how the state decays and rotates, so that it keeps information over thousands of steps and stays cheap.

## Overview

The continuous state space model maps an input signal $u(t)$ to an output $y(t)$ through a state $x(t) \in \mathbb{R}^N$:

$$ x'(t) = A x(t) + B u(t), \qquad y(t) = C x(t) + D u(t). $$ {#eq-ssm}

In a deep SSM, $A$, $B$, $C$ and $D$ are parameters learned by gradient descent (Gu, Goel and Ré 2021, Section 2.1). {#c-ssm-definition}
The S4 paper leaves out $D$ in its exposition, because $D u$ is a skip connection that is easy to compute (Section 2.1). {#c-d-skip}
Each SSM maps one channel to one channel, so a layer for $H$ features runs $H$ of them.

Three choices make the model a usable layer.
The step size $\Delta$ turns the differential equation into a recurrence.
The initial $A$, usually from HiPPO, makes the state a memory of the long past.
The structure of $A$, diagonal plus low rank in S4 and diagonal in its successors, makes the recurrence and the convolution kernel cheap.

## Components

**Discretization.**
Sampling the input at step $\Delta$ turns the equation into $x_k = \bar{A} x_{k-1} + \bar{B} u_k$.
Zero-order hold, which holds the input constant over each step, gives the exact solution

$$ \bar{A} = \exp(\Delta A), \qquad \bar{B} = A^{-1}\big(\exp(\Delta A) - I\big) B, $$ {#eq-zoh}

so for a scalar $a$, $\bar{a} = e^{\Delta a}$ and $\bar{b} = (e^{\Delta a} - 1) b / a$ (HiPPO, Appendix B.3, page 21). {#c-zoh}
The bilinear rule, which S4 uses, replaces the integral over one step by the trapezoid rule:

$$ \bar{A} = \big(I - \tfrac{\Delta}{2} A\big)^{-1}\big(I + \tfrac{\Delta}{2} A\big), \qquad \bar{B} = \big(I - \tfrac{\Delta}{2} A\big)^{-1} \Delta B $$ {#eq-bilinear}

(S4, Section 2.3). {#c-bilinear}
If every eigenvalue of $A$ has a negative real part, every eigenvalue of $\bar{A}$ lies inside the unit circle under both rules, for any $\Delta > 0$; the bilinear rule maps the imaginary axis onto the circle and the right half-plane outside it. {#c-stability}
For a scalar, the two rules differ by $(\Delta a)^3/12$ plus higher-order terms, so they agree to second order in $\Delta$. {#c-rules-agree}
The derivation [[#discretization]] proves all four statements, and its check runs them in numbers.

**Recurrence and convolution.**
With a zero initial state the discrete model is

$$ x_k = \bar{A} x_{k-1} + \bar{B} u_k, \qquad y_k = C x_k + D u_k. $$ {#eq-recurrence}

Unrolling it, $x_k = \sum_{j \le k} \bar{A}^{k-j} \bar{B} u_j$, so the output is a causal convolution $y = K * u + D u$ with the kernel

$$ K = \big(C\bar{B},\ C\bar{A}\bar{B},\ C\bar{A}^2\bar{B},\ \dots,\ C\bar{A}^{L-1}\bar{B}\big) \in \mathbb{R}^L $$ {#eq-kernel}

(S4, Section 2.4). {#c-convolution-view}
The output at position $k$ depends only on the inputs up to $k$. {#c-causal}
A unit impulse at position 0 returns $K$, plus $D$ at position 0, which is the quickest test of an implementation. {#c-impulse}
This equivalence needs $\bar{A}$, $\bar{B}$ and $C$ to be the same at every position and for every input; Mamba gives it up.

**Diagonal state.**
If $\bar{A}$ is diagonal with entries $\bar{a}_n$, the kernel is a sum of $N$ geometric sequences,

$$ K_\ell = \sum_{n=0}^{N-1} C_n \bar{B}_n \bar{a}_n^{\ell}, \qquad \ell = 0, \dots, L-1, $$ {#eq-diagonal-kernel}

the product of the vector $C \circ \bar{B}$ with a Vandermonde matrix (S4D, equation 7). {#c-diagonal-kernel}
Conjugation is an equivalence: $(A, B, C)$ and $(V^{-1} A V, V^{-1} B, C V)$ compute the same map for any invertible $V$ (S4, Lemma 3.1). {#c-conjugation}
So any diagonalizable $A$ could in principle be replaced by its eigenvalues.
When a real SSM is diagonalized, the complex parameters come in conjugate pairs, so an implementation keeps $N/2$ of them and adds back their conjugates, in practice by taking twice the real part (S4D).
The derivation [[#recurrence-convolution]] proves the unrolling, causality, the impulse response, the diagonal form and the invariance.

**HiPPO initialization.**
HiPPO is a framework for the online compression of a signal by projection onto polynomial bases: given a measure that says how much each time in the past matters, it gives the optimal coefficients as the solution of an online function approximation problem (Gu et al. 2020, abstract). {#c-hippo-framework}
Its scaled Legendre measure, LegS, weights all of the history $[0, t]$ uniformly, and its coefficients follow $\tfrac{d}{dt} c(t) = -\tfrac{1}{t} A c(t) + \tfrac{1}{t} B f(t)$ with $A_{nk} = \sqrt{(2n+1)(2k+1)}$ for $n > k$, $n + 1$ for $n = k$, 0 above the diagonal, and $B_n = \sqrt{2n+1}$ (HiPPO, Section 3, Theorem 2). {#c-hippo-legs}
LegS is timescale-equivariant: compressing $f(\alpha t)$ gives the coefficients of $f$ at time $\alpha t$ (HiPPO, Section 3, Proposition 3). {#c-hippo-timescale}
S4 calls the negated matrix, without the factor $1/t$, the HiPPO matrix (S4, Section 2.2): {#c-hippo-in-s4}

$$ A_{nk} = -\begin{cases} \sqrt{(2n+1)(2k+1)} & n > k \\ n + 1 & n = k \\ 0 & n < k. \end{cases} $$ {#eq-hippo}

An S4 layer starts from an SSM with $A$ set to this matrix (Section 3.4). {#c-s4-init}
The S4 paper reports that the LSSL, the model before S4, went from 60% to 98% on sequential MNIST when a random $A$ was replaced by this matrix (Section 2.2). {#c-hippo-smnist}
The matrix is lower triangular with eigenvalues $-1, -2, \dots, -N$. {#c-hippo-eigs}

**Normal plus low rank.**
The HiPPO matrix is not normal, and diagonalizing it is numerically hopeless: S4 shows that the diagonalizing matrix has entries of magnitude up to about $2^{4N/3}$ (Section 3.1, Lemma 3.2). {#c-hippo-diag-unstable}
In our own runs with `numpy.linalg.eig`, the matrix of unit eigenvectors has condition number about $7.7 \times 10^4$ for $N = 8$ and $8.3 \times 10^{10}$ for $N = 16$. {#c-hippo-conditioning}
But adding a rank-one term makes it normal:

$$ A = S - P P^\top, \qquad P_n = \sqrt{n + \tfrac{1}{2}}, \qquad S = -\tfrac{1}{2} I + J,\ J^\top = -J, $$ {#eq-nplr}

so the eigenvalues of $S$ all have real part $-\tfrac12$, and $S$ is unitarily diagonalizable. {#c-hippo-nplr}
S4 proves that all the HiPPO matrices of Gu et al. (2020) are normal plus low rank, and that this one has rank $r = 1$ (Section 3.2, Theorem 1). {#c-s4-nplr}
S4 therefore works with $A = V(\Lambda - P Q^*)V^*$, diagonal plus low rank in a unitary basis.
The derivation [[#hippo-nplr]] checks the decomposition entry by entry and reports the condition numbers above.

## Shapes and parameters

Take a batch of $B$ sequences of length $L$ with $H$ features, and $N$ state entries per feature.
An S4 layer applies $H$ independent SSMs, one per feature, and mixes the features with a position-wise linear layer, with a nonlinearity between layers (S4, Section 3.4). {#c-s4-layer}
Each SSM of S4 has $5N$ parameters, $\Lambda$, $P$, $Q$, $B$ and $C$ (Section 3.4), plus its step size.
A diagonal SSM keeps $N/2$ complex entries of $A$, $B$ and $C$ per feature, so about $3N$ real numbers, plus $\log \Delta$ and $D$.
Input and output both have shape $(B, L, H)$; the state has shape $(B, H, N)$, and the kernel has shape $(H, L)$.
S4 notes that follow-up work found its $\Lambda - PQ^*$ form can be unstable when $A$ has eigenvalues in the right half-plane, and changed it to $\Lambda - PP^*$ (Section 3.4). {#c-s4-pp-fix}

## Computation and cost

For a diagonal $\bar{A}$, one step of the recurrence costs $O(N)$ operations and $O(N)$ memory per feature, the kernel costs $O(NL)$ by summing $N$ geometric sequences, and the convolution costs $O(L \log L)$ with an FFT of length $2L$. {#c-mode-costs}
For the diagonal-plus-low-rank form of S4, a step of the bilinear recurrence still costs $O(N)$ (S4, Section 3.3, Theorem 2). {#c-s4-recurrence-cost}
The S4 kernel is computed from its generating function at the roots of unity, which reduces to 4 Cauchy multiplies and takes $\tilde{O}(N + L)$ operations and $O(N + L)$ space (Section 3.3, Theorem 3). {#c-s4-kernel-cost}
For the LSSL, the S4 paper gives $O(N^2 L)$ operations and $O(NL)$ space (Section 1). {#c-lssl-cost}
S4 reports that it is up to 30 times faster than the LSSL, with 400 times less memory (Section 1 and Table 2). {#c-s4-vs-lssl}

So training uses the convolution, which runs over all positions at once, and generation uses the recurrence, which needs only the state.
Attention, by contrast, costs $O(L^2)$ in the length and keeps a cache of all past keys and values ([[transformer]]).

## Training

The whole layer is differentiable, so it trains by backpropagation through the convolution, which avoids running [[backpropagation-through-time@al|backpropagation through time]] step by step.
Our reference code learns the step size in log space, one per feature, and initializes it uniformly in log space between $10^{-3}$ and $10^{-1}$.
S4D reports that prior work forces the real part of $A$ to be negative by writing $A = -\exp(A_{\mathrm{Re}}) + i A_{\mathrm{Im}}$, and notes that any activation bounded on one side would do (S4D); by [[#c-stability]] this keeps $\bar{A}$ stable for every step size.
Its S4D-Lin initialization sets $A_n = -\tfrac12 + i\pi n$ (S4D, Section 4). {#c-s4d-lin}

## Inductive biases

A linear time-invariant SSM assumes that the same filter applies at every position, and its filter is a sum of decaying and oscillating exponentials.
HiPPO adds a bias toward remembering the whole history, with the uniform LegS weight.
The continuous-time view adds a bias toward signals sampled from an underlying function: changing $\Delta$ changes the sampling rate without changing $A$.
A time-invariant model cannot decide what to keep based on the content of the input.
Mamba's authors note that global convolutions solve the Copying task, which only needs awareness of time, but have difficulty with Selective Copying, which moves the tokens to memorize and so needs awareness of content (Mamba, Section 3.1). {#c-lti-selective-copying}

## Evidence

Long Range Arena (LRA) has six tasks of 1K to 16K steps.
In Table 4 of the S4 paper, S4 scores 59.60 on ListOps, 86.82 on Text, 90.90 on Retrieval, 88.65 on Image, 94.20 on Pathfinder and 96.35 on Path-X, an average of 86.09, against an average of at most 59.37 for the Transformer variants and other models in the table. {#c-s4-lra}
These are the updated numbers: the paper says that several experiments were updated since its first version, and that the LRA results of Table 4 now come from follow-up papers (Appendix D.5, page 30). {#c-s4-lra-update}
The original results, in Table 10 of the same paper, are 58.35, 76.02, 87.09, 87.26, 86.05 and 88.10 on the six tasks, an average of 80.48. {#c-s4-lra-original}
The introduction still quotes them: S4 is the first model to solve Path-X, of length 16384, with 88% accuracy against 50%, random guessing, for all prior work (Section 1). {#c-path-x}
On WikiText-103, S4 with 249M parameters reaches a test perplexity of 20.95 against 20.51 for the 247M Transformer whose attention it replaces, and generates 60 times more tokens per second (S4, Table 8, page 10). {#c-s4-wikitext}

## Limitations

The equivalence of recurrence and convolution holds only for time-invariant parameters, so a time-invariant SSM cannot select what to store based on the input.
Diagonal-plus-low-rank S4 needs a Cauchy kernel and careful numerics; this complexity motivated the diagonal variants.
The bilinear and ZOH rules are stable only if $A$ is: an eigenvalue in the right half-plane gives $|\bar{a}| > 1$ and a kernel that grows, which is why S4 changed its parameterization (see above).
On language modeling, the S4 paper still trails the Transformer it is based on (Table 8).

## Variants and descendants

**DSS.**
Gupta, Gu and Berant (2022) drop the low-rank term and use a diagonal $A$; DSS matches S4 on Long Range Arena and on Speech Commands (abstract). {#c-dss}

**S4D.**
Gu, Gupta, Goel and Ré (2022) explain why DSS works: the diagonal restriction of the S4 matrix gives the same kernel as the state size goes to infinity (abstract). {#c-s4d-limit}
S4D computes its kernel in two lines of code, performs comparably to S4 in almost all settings, and averages 85% on Long Range Arena (abstract). {#c-s4d}

**LRU.**
Orvieto et al. (2023) start from a vanilla deep RNN.
Replacing the SSM layers of a deep SSM with tanh or ReLU RNN layers lowers its score on LRA, and removing the nonlinearity of the recurrence gives a substantial gain (Section 1). {#c-lru-linear}
With complex diagonal recurrences, an exponential parameterization and a normalization, the Linear Recurrent Unit matches deep SSMs on all LRA tasks (Section 1). {#c-lru}
They conclude that linear recurrences with complex diagonal matrices, normalization and initialization explain the success of diagonal SSMs, and that parameter sharing and the continuous-time view of discretization do not necessarily contribute (Section 4). {#c-lru-discretization}

**Selective SSMs (Mamba).**
Gu and Dao (2023) make $\Delta$, $B$ and $C$ functions of the input at each position, so the model becomes time-varying and loses the equivalence to convolutions (Section 3.2). {#c-mamba-selection}
The step is $\Delta_k = \mathrm{softplus}(p + s_\Delta(u_k))$, with a parameter $p$ and a linear map $s_\Delta$, and $B_k$ and $C_k$ are linear maps of $u_k$; zero-order hold applies at each step:

$$ x_k = \exp(\Delta_k A)\, x_{k-1} + (\Delta_k A)^{-1}\big(\exp(\Delta_k A) - I\big)\Delta_k B_k\, u_k, \qquad y_k = C_k x_k. $$ {#eq-selective}

With $N = 1$, $A = -1$ and $B = 1$, this is the gate $x_k = (1 - g_k) x_{k-1} + g_k u_k$ with $g_k = \sigma(z_k)$ (Section 3.5.1, Theorem 1). {#c-selective-gate}
Because $\Delta_k$ depends on $u_k$, the map from $u$ to $y$ is not linear, so no kernel $K$ gives $y = K * u$. {#c-selective-no-kernel}
The steps still compose under the associative operator $(a_1, b_1) \bullet (a_2, b_2) = (a_2 a_1, a_2 b_1 + b_2)$, so a parallel prefix scan computes all states in $O(\log L)$ depth. {#c-scan-associative}
Mamba computes it with a scan fused into one GPU kernel, keeping the expanded state in fast on-chip memory and recomputing the intermediate states in the backward pass (Section 3.3.2). {#c-mamba-scan}
The derivation [[#selective-scan]] proves the gate form, the loss of linearity and the associativity.
The paper reports 5 times the inference throughput of Transformers, linear scaling in sequence length, and a 3B Mamba that outperforms Transformers of the same size and matches Transformers twice its size (abstract). {#c-mamba-lm}

## Notable models

S4, DSS, S4D, the LRU and Mamba are each a model family of this page.
The language models of the Mamba paper range from Mamba-130M to Mamba-2.8B and are trained on the Pile (Mamba, Section 4.2).

## Implementations

The reference code is a diagonal SSM layer with $H$ features and $N$ complex state entries per feature, kept in conjugate pairs.
`impl/numpy/ssm.py` holds the reference: it discretizes with zero-order hold or the bilinear rule ([[#eq-zoh]], [[#eq-bilinear]]), runs the recurrence ([[#eq-recurrence]]), builds the kernel ([[#eq-diagonal-kernel]]) and convolves with an FFT of length $2L$.
It also has dense helpers for a general $A$ (a matrix exponential, the dense kernel and scan) and a selective scan with per-step zero-order hold.
`impl/pytorch/ssm.py` is a `torch.nn.Module` with a convolution mode and a scan mode, and `impl/jax/ssm.py` adds a parallel mode built on `jax.lax.associative_scan` with the operator of [[#c-scan-associative]].

The tests check that the recurrence equals the convolution on random layers, that the impulse response is the kernel, and that changing a later input never changes an earlier output.
They check zero-order hold against the exact exponential for a scalar, and that the bilinear rule keeps a stable $A$ inside the unit circle for steps from $10^{-3}$ to $10^4$.
The PyTorch and JAX tests compare against a plain loop and check that gradients agree between the two modes.

### Worked example

Take one feature with $N = 1$, $a = -1$ and $b = c = 1$.
Zero-order hold with $\Delta = \ln 2$ gives $\bar{a} = e^{-\ln 2} = \tfrac12$ and $\bar{b} = (\tfrac12 - 1)/(-1) = \tfrac12$.
The bilinear rule with $\Delta = \tfrac23$ gives the same: $\bar{a} = (1 - \tfrac13)/(1 + \tfrac13) = \tfrac12$ and $\bar{b} = \tfrac23 / \tfrac43 = \tfrac12$.
The kernel is $K = (\tfrac12, \tfrac14, \tfrac18, \tfrac1{16})$.
For the input $u = (1, 2, 0, -1)$, the recurrence gives $x_0 = 0.5$, $x_1 = 0.25 + 1 = 1.25$, $x_2 = 0.625$, $x_3 = 0.3125 - 0.5 = -0.1875$.
The convolution gives the same: $y_3 = \tfrac12(-1) + \tfrac14(0) + \tfrac18(2) + \tfrac1{16}(1) = -0.1875$.
The test suite runs this example in all three frameworks.

## History

HiPPO (Gu, Dao, Ermon, Rudra and Ré, arXiv 2008.07669, 2020) gave the memory matrix.
The LSSL came next, with the cost given in the Computation section.
Albert Gu, Karan Goel and Christopher Ré of Stanford University then introduced S4 in "Efficiently Modeling Long Sequences with Structured State Spaces" (arXiv 2111.00396). {#c-s4-history}
The diagonal variants DSS and S4D followed in 2022, and the LRU in 2023 linked them back to linear RNNs.
Albert Gu of Carnegie Mellon University and Tri Dao of Princeton University introduced Mamba in arXiv 2312.00752 in December 2023. {#c-mamba-history}
None of these PDFs prints a venue, so this page cites the arXiv versions.

## References

- Gu, Goel and Ré (2021). Efficiently Modeling Long Sequences with Structured State Spaces. arXiv 2111.00396v3.
- Gu, Dao, Ermon, Rudra and Ré (2020). HiPPO: Recurrent Memory with Optimal Polynomial Projections. arXiv 2008.07669v2.
- Gupta, Gu and Berant (2022). Diagonal State Spaces are as Effective as Structured State Spaces. arXiv 2203.14343v3.
- Gu, Gupta, Goel and Ré (2022). On the Parameterization and Initialization of Diagonal State Space Models. arXiv 2206.11893v2.
- Orvieto, Smith, Gu, Fernando, Gulcehre, Pascanu and De (2023). Resurrecting Recurrent Neural Networks for Long Sequences. arXiv 2303.06349v1.
- Gu and Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv 2312.00752v2.
