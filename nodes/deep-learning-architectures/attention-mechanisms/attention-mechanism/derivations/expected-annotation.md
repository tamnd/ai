---
id: expected-annotation
derives: [c-convex, c-uniform, c-fixed]
kind: proof
assumes: ["alpha_i is the softmax of the energies over the real source positions", "attention=False sets every c_i to the same vector, the last forward and the first backward annotation"]
uses: [softmax-regression@al]
status: full
check: expected-annotation.check.py
lean: null
sources:
  - {work: bahdanau-2015-neural-machine-translation, locator: "p. 4", note: "the context as an expected annotation"}
  - {work: bahdanau-2015-neural-machine-translation, locator: "p. 14", note: "the RNN Encoder-Decoder as the model with c_i fixed; v_a initialized to zero"}
---

### Statement

1. Each context vector $c_i$ is the expectation of the annotation $h_J$ for a random source position $J$ with $P(J = j) = \alpha_{ij}$, and it lies in the convex hull of the annotations of the real positions; in particular every coordinate of $c_i$ lies between the smallest and the largest value of that coordinate over the real annotations.
2. With $v_a = 0$ every energy is 0 and every $c_i$ is the mean of the real annotations.
3. If $c_i$ is the same vector $c$ at every step, the decoder of eq-decoder is the decoder of an RNN Encoder-Decoder conditioned on the fixed vector $c$, and the alignment model has no effect on the output.

### Assumptions

1. $\alpha_{ij} \ge 0$, $\sum_j \alpha_{ij} = 1$ and $\alpha_{ij} = 0$ at the pads. Used in steps 1 and 2.
2. The decoder update and the output read $c_i$ and nothing else of the encoder. Used in step 3.

### Steps

1. $c_i = \sum_j \alpha_{ij} h_j$ with weights that are a probability vector over the real positions is the expectation of $h_J$ by definition, and a convex combination of the $h_j$. A coordinate of a convex combination is a convex combination of that coordinate, so it lies between its minimum and maximum.
2. $e_{ij} = v_a^\top \tanh(\cdot) = 0$ for every $j$, so the softmax gives $\alpha_{ij} = 1/T_x$ over the $T_x$ real positions and $c_i = \frac{1}{T_x} \sum_j h_j$.
3. With $c_i = c$ for all $i$, $s_i = f(s_{i-1}, y_{i-1}, c)$ and $p(y_i \mid \cdot) = g(s_{i-1}, y_{i-1}, c)$ are a recurrent decoder fed a constant vector, the structure of Cho et al.'s RNN Encoder-Decoder; $W_a$, $U_a$ and $v_a$ do not appear.

### Result and check

Attention does not create information: it can only average what the encoder wrote, and at the paper's initialization it starts as a plain average, so at the start of training every target word sees the same summary of the sentence and the weights become selective only as $v_a$, then $W_a$ and $U_a$, learn. The check `expected-annotation.check.py` verifies items 1 and 2 on random RNNsearch models with padding, and item 3 by perturbing the alignment model of a model built with `attention=False` and finding the same logits.
