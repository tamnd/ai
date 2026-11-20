---
id: parameter-and-flop-count
derives: [c-params, c-flops, c-one-bias, c-gated-ratio, c-pixel-size]
kind: count
assumes: ["one simple recurrent layer in the form of torch.nn.RNN: an input matrix W of I x H, a recurrent matrix U of H x H and two biases b_i and b_h of length H", "a multiply-add counts as 2 FLOPs and only the two projections are counted", "the GRU and the LSTM of the comparison are in the form of torch.nn.GRU and torch.nn.LSTM, with two biases per block"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-and-flop-count.check.py
lean: null
sources:
  - {work: pytorch-torch-nn-rnn, locator: "the title page"}
  - {work: le-2015-simple-way-initialize-recurrent, locator: "Section 4.2"}
---

### Statement

1. One layer of eq-step with input size $d_x$ and $d_h$ units has $N = d_h d_x + d_h^2 + 2 d_h$ parameters, and the same number whatever the length of the sequence.
2. A forward pass over a batch of $B$ sequences of $T$ steps costs $C = 2 B T d_h (d_x + d_h)$ FLOPs in its two products.
3. The two biases only enter as their sum, so the layer computes the same functions with $d_h d_x + d_h^2 + d_h$ parameters.
4. A GRU layer and an LSTM layer of the same sizes, in the same form, have exactly $3N$ and $4N$ parameters and cost $3C$ and $4C$ FLOPs.
5. The networks of Le et al. on pixel-by-pixel MNIST, 100 units reading one pixel per step, have $10{,}300$ parameters in the recurrent layer.

### Assumptions

1. $W \in \mathbb{R}^{d_x \times d_h}$, $U \in \mathbb{R}^{d_h \times d_h}$ and $b_i, b_h \in \mathbb{R}^{d_h}$, as in torch.nn.RNN, whose weight_ih_l0 and weight_hh_l0 are $W^\top$ and $U^\top$. Used in steps 1 and 3.
2. A product of an $m \times k$ and a $k \times n$ matrix costs $2mkn$ FLOPs; the additions of the biases and the nonlinearity are left out. Used in step 2.
3. torch.nn.GRU has three blocks of $W$, $U$, $b_i$ and $b_h$ and torch.nn.LSTM has four, each of the shape of the whole simple layer. Used in step 4.

### Steps

1. Count the entries: $W$ has $d_x d_h$, $U$ has $d_h^2$, and the two biases have $d_h$ each. None of them is indexed by $t$: the same arrays are used at every step, so $N$ does not depend on $T$. This is statement 1.
2. At each step, $\mathbf{x}_t W$ is a $(B \times d_x)(d_x \times d_h)$ product, $2 B d_x d_h$ FLOPs, and $\mathbf{h}_{t-1} U$ is a $(B \times d_h)(d_h \times d_h)$ product, $2 B d_h^2$ FLOPs. Over $T$ steps the total is $2BT d_h (d_x + d_h)$. This is statement 2.
3. In eq-step, $b_i$ and $b_h$ are added to the same pre-activation, so only $b_i + b_h$ matters: moving a constant from one to the other changes nothing. Replacing the pair by one vector removes $d_h$ parameters and no function. This is statement 3.
4. A gated layer in the torch form computes one block per gate and one for the candidate, each with an input product, a recurrent product and two biases of the same sizes as the simple layer: three blocks for the GRU, four for the LSTM. Both the count and the products scale by 3 and by 4. This is statement 4.
5. With $d_x = 1$ and $d_h = 100$, $N = 100 + 10{,}000 + 200 = 10{,}300$. This is statement 5.

### Result and check

$$ N = d_h d_x + d_h^2 + 2 d_h, \qquad C = 2BT d_h (d_x + d_h), \qquad N_{\mathrm{GRU}} : N_{\mathrm{LSTM}} : N = 3 : 4 : 1 . $$

For the example config, $d_x = 3$ and $d_h = 4$, $N = 36$, and with $B = 2$ and $T = 5$, $C = 560$. The check `parameter-and-flop-count.check.py` counts the arrays of `init`, replays the products of `forward`, compares with torch.nn.RNN, torch.nn.GRU and torch.nn.LSTM when PyTorch is installed, and checks that shifting a constant between the two biases leaves the output unchanged.
