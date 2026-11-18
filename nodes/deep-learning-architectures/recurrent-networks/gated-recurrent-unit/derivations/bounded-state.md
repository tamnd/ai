---
id: bounded-state
derives: [c-bounded, c-special-cases]
kind: bound
assumes: ["one GRU layer of eq-gates, eq-candidate or eq-candidate-cho, and eq-update, with the sigmoid gates and the tanh candidate", "the initial state lies in the box [-1, 1]^H"]
uses: [activation-function@al]
status: full
check: bounded-state.check.py
lean: null
sources:
  - {work: weiss-2018-practical-computational-power-finite, locator: "Sections 2 and 4"}
---

### Statement

1. If every coordinate of $h_0$ lies in $[-1, 1]$, then every coordinate of $h_t$ lies in the open interval $(-1, 1)$ for every $t \ge 1$, whatever the inputs, the parameters and the placement of the reset gate. More generally $\lVert h_t \rVert_\infty \le \max(\lVert h_{t-1} \rVert_\infty, \lVert n_t \rVert_\infty)$.
2. In the limits $z_t \to 0$ and $r_t \to 1$, the step becomes the tanh recurrent network $h_t = \tanh(x_t W_n + b_{in} + h_{t-1} U_n + b_{hn})$, for both placements of the reset gate. In the limit $z_t \to 1$ it copies the state, $h_t = h_{t-1}$. In the limit $r_t \to 0$ the candidate ignores the state: it is $\tanh(x_t W_n + b_{in})$ with the reset after the matrix and $\tanh(x_t W_n + b_{in} + b_{hn})$ with it before.

### Assumptions

1. $z_t = \sigma(\cdot) \in (0, 1)$ and $n_t = \tanh(\cdot) \in (-1, 1)$ coordinate by coordinate, for finite arguments. Used in step 1.
2. The gates reach their limits when their pre-activations go to $\pm\infty$, for instance through the biases; in floating point a bias of a few hundred already gives exactly 0 or 1. Used in step 2.

### Steps

1. Each coordinate of eq-update is a convex combination, $h_t = (1 - z_t) n_t + z_t h_{t-1}$ with weights $1 - z_t$ and $z_t$ in $(0, 1)$, so $\lvert h_t \rvert \le (1 - z_t) \lvert n_t \rvert + z_t \lvert h_{t-1} \rvert \le \max(\lvert n_t \rvert, \lvert h_{t-1} \rvert)$. If $\lvert h_{t-1} \rvert \le 1$, then since $\lvert n_t \rvert < 1$ and $1 - z_t > 0$, the first inequality gives $\lvert h_t \rvert < (1 - z_t) + z_t = 1$. Induction on $t$ gives statement 1. The bound comes from the tanh of the candidate and from tying the two weights $z_t$ and $1 - z_t$; neither depends on where the reset gate is.
2. Put $z_t = 0$ in eq-update: $h_t = n_t$. Put $r_t = 1$ in eq-candidate: $n_t = \tanh(x_t W_n + b_{in} + h_{t-1} U_n + b_{hn})$; eq-candidate-cho gives the same expression, since $1 \odot h_{t-1} = h_{t-1}$. With $z_t = 1$, $h_t = h_{t-1}$. With $r_t = 0$, eq-candidate multiplies the whole of $h_{t-1} U_n + b_{hn}$ by 0, while eq-candidate-cho multiplies only $h_{t-1}$, and $b_{hn}$ remains. This is statement 2.
3. Weiss et al. draw the consequence for counting: a dimension that stays in $(-1, 1)$ and is stored with finite precision takes finitely many values, so it cannot hold a counter that grows without bound. The LSTM escapes this because its cell $c_t = f_t c_{t-1} + i_t \tilde c_t$ is not squashed and its gates are not tied.

### Result and check

$$ \lVert h_0 \rVert_\infty \le 1 \implies \lVert h_t \rVert_\infty < 1 \quad (t \ge 1), \qquad z \to 0,\ r \to 1:\ h_t = \tanh(x_t W_n + b_{in} + h_{t-1} U_n + b_{hn}) . $$

The check `bounded-state.check.py` runs both placements of the reset gate for 500 steps on inputs scaled up to $10^3$ with weights scaled up to 30 and checks the bound at every step (closed in floating point, where the tanh of a large argument rounds to 1), checks the inequality of statement 1 from an initial state outside the box, and saturates the gates with biases of $\pm 800$ to compare the layer with a tanh network, a copy and the two reset candidates.
