---
id: context-and-depth
derives: [c-context, c-latency, c-depth]
kind: proof
assumes: ["a network of L layers whose cells at step t read the outputs of the layer below at the same step t", "in a unidirectional layer, the cell of step t also reads the state of step t - 1; in a bidirectional layer, the forward cell reads step t - 1 and the backward cell step t + 1", "the output at step t reads only the top layer at step t", "a cell step takes one unit of time once its inputs are known"]
uses: []
status: full
check: context-and-depth.check.py
lean: null
sources:
  - {work: schuster-1997-bidirectional-recurrent-neural-networks, locator: "Section II.B"}
  - {work: graves-2013-speech-recognition-deep-recurrent, locator: "Section 2"}
  - {work: lipton-2015-critical-review-recurrent, locator: "Section 4.2"}
---

### Statement

1. In a unidirectional network, the output at step $t$ is a function of $\mathbf{x}_1, \dots, \mathbf{x}_t$ only. In a bidirectional network with at least one layer, the output at every step is a function of the whole input $\mathbf{x}_1, \dots, \mathbf{x}_T$.
2. Consequently no output of a bidirectional network can be computed before the last input $\mathbf{x}_T$ is known: the first output waits $T - 1$ steps longer than it would in a unidirectional network.
3. The longest chain of cell steps that must run one after the other has length $T + L - 1$ in a unidirectional network of $L$ layers, and $L T$ in a bidirectional one.

### Assumptions

1. The connections of the assumptions above, and no others: in particular the forward and the backward cells of a layer do not read each other, as Schuster and Paliwal require. Used in steps 1 to 4.
2. Generic weights: a dependence that the graph allows is present, as it is for random weights. Used in step 1 and in the check.

### Steps

1. Write $\mathrm{dep}(\cdot)$ for the set of input steps that a quantity reads, directly or not. For the forward cell of the first layer, $\mathrm{dep}(\overrightarrow{h}_t) = \mathrm{dep}(\overrightarrow{h}_{t-1}) \cup \{t\}$, so by induction on $t$ it is $\{1, \dots, t\}$; for the backward cell, by induction downwards from $T$, $\mathrm{dep}(\overleftarrow{h}_t) = \{t, \dots, T\}$. A layer above reads both cells of step $t$ of the layer below, whose union is already $\{1, \dots, T\}$, so every bidirectional layer has $\mathrm{dep} = \{1, \dots, T\}$ at every step, and so has the output. A unidirectional layer above a layer with $\mathrm{dep} = \{1, \dots, t\}$ at step $t$ keeps it. This is statement 1.
2. By step 1, the output of step 1 reads $\mathbf{x}_T$, which is known only at time $T$; in a unidirectional network it reads only $\mathbf{x}_1$. This is statement 2, the reason why a bidirectional network cannot run online.
3. Unidirectional: let $d(l, t)$ be the time at which cell $(l, t)$ can finish. It waits for $(l, t - 1)$ and $(l - 1, t)$, so $d(l, t) = \max(d(l, t - 1), d(l - 1, t)) + 1$ with $d(\cdot, 0) = d(0, \cdot) = 0$, whose solution is $d(l, t) = l + t - 1$, the wavefront in which layer $l$ runs one step behind layer $l - 1$. The last cell finishes at $L + T - 1$.
4. Bidirectional: the forward cell $(l, t)$ waits for the forward cell $(l, t - 1)$ and for both cells $(l - 1, t)$. The backward cell of layer $l - 1$ at step 1 finishes last in its layer, after its whole chain from $T$ down to 1, so the forward cell $(l, 1)$ cannot start before all of layer $l - 1$ is done, and likewise the backward cell $(l, T)$ needs the forward cell $(l - 1, T)$, the end of the forward chain. By induction, each layer finishes $T$ steps after the one below, at time $l T$; the two directions of a layer run side by side, so the layer itself takes $T$, not $2 T$. The last layer finishes at $L T$. This is statement 3.

### Result and check

$$ \mathrm{dep}(\mathbf{y}_t) = \begin{cases} \{1, \dots, t\} & \text{unidirectional} \\ \{1, \dots, T\} & \text{bidirectional} \end{cases}, \qquad \text{depth} = \begin{cases} T + L - 1 & \text{unidirectional} \\ L T & \text{bidirectional.} \end{cases} $$

The check `context-and-depth.check.py` perturbs each input of the NumPy reference with random weights and records which outputs move, for both kinds of network and several depths, and computes the longest chain of the dependency graph of the cells by dynamic programming for $T$ and $L$ up to 12.
