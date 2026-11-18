---
id: parameter-and-flop-count
derives: [c-params, c-flops, c-distinct-params, c-lstm-ratio, c-chung-sizes]
kind: count
assumes: ["one GRU layer in the form of torch.nn.GRU: an input matrix W of I x 3H, a recurrent matrix U of H x 3H and two biases b_i and b_h of length 3H", "a multiply-add counts as 2 FLOPs and only the two projections are counted", "the LSTM of the comparison has four blocks with one bias each, and optionally three diagonal peephole vectors"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-and-flop-count.check.py
lean: null
sources:
  - {work: pytorch-torch-nn-gru, locator: "the equations and the parameters"}
  - {work: chung-2014-empirical-evaluation-gated-recurrent, locator: "Section 3 and Table 1"}
---

### Statement

1. A GRU layer with input width $I$ and $H$ units has $P = 3HI + 3H^2 + 6H$ parameters, independent of the length $T$ of the sequence.
2. Its forward pass over a batch of $B$ sequences of length $T$ costs $C = 6BTHI + 6BTH^2$ FLOPs for the two projections.
3. With the reset gate after the recurrent matrix, the biases of the reset and of the update gate only enter through the sums $b_{ir} + b_{hr}$ and $b_{iz} + b_{hz}$, so the layer computes a family of functions with $3HI + 3H^2 + 4H$ free parameters. With the reset gate before the matrix, $b_{in} + b_{hn}$ is a sum as well, and the count is $3HI + 3H^2 + 3H$, the count of a layer with one bias per block.
4. An LSTM layer of the same widths with one bias per block has $4HI + 4H^2 + 4H$ parameters and costs $8BTHI + 8BTH^2$ FLOPs, so the ratio of both GRU counts to the LSTM counts is exactly $3/4$ for one bias per block.
5. With one bias per block, the recurrent layers of the speech models of Chung et al. (2014), $I = 20$ and $H = 227$ for the GRU, $H = 195$ for an LSTM with peepholes and $H = 400$ for a tanh network, have 168,888, 169,065 and 168,400 parameters.

### Assumptions

1. The weights and biases are the arrays of torch.nn.GRU, with the three blocks reset, update and candidate stacked. Used in steps 1 and 3.
2. A product of an $m \times k$ and a $k \times n$ matrix costs $2mkn$ FLOPs; the sigmoids, the tanh and the elementwise products are not counted. Used in steps 2 and 4.
3. The LSTM has input, forget and output gates and a candidate, each an affine map of $x_t$ and $h_{t-1}$; a peephole adds a diagonal matrix, $H$ numbers, to each of the three gates, as in the LSTM that Chung et al. use. Used in steps 4 and 5.
4. The tanh network is $h_t = \tanh(x_t W + h_{t-1} U + b)$. Used in step 5.

### Steps

1. $W$ has $3HI$ entries, $U$ has $3H^2$ and the two biases $6H$. No shape involves $T$: the same arrays are applied at every step. This is statement 1.
2. At each step, $x_t W$ is a product of a $B \times I$ and an $I \times 3H$ matrix, $6BHI$ FLOPs, and the recurrent projection is a product of $B \times H$ and $H \times 3H$, $6BH^2$ FLOPs. With the reset before the matrix, the candidate block multiplies $r_t \odot h_{t-1}$ instead of $h_{t-1}$, a matrix of the same shape, so the count does not change. Over $T$ steps this is statement 2.
3. In eq-gates the bias of the reset gate enters as $x_t W_r + b_{ir} + h_{t-1} U_r + b_{hr}$: changing $b_{ir}$ by $\delta$ and $b_{hr}$ by $-\delta$ leaves every output unchanged, which removes $H$ degrees of freedom for each of the two gates. In eq-candidate, $b_{hn}$ is multiplied by $r_t$ and $b_{in}$ is not, so they are not interchangeable; in eq-candidate-cho both are added outside the product and only their sum matters. This is statement 3.
4. The LSTM has four blocks where the GRU has three, so each of $I$, $H^2$ and the per-block bias counts is $4/3$ of the GRU's with one bias per block, and the projections cost $8BTHI + 8BTH^2$. This is statement 4.
5. For the GRU, $3 \cdot 227 \cdot 20 + 3 \cdot 227^2 + 3 \cdot 227 = 13{,}620 + 154{,}587 + 681 = 168{,}888$. For the LSTM, $4 \cdot 195 \cdot 20 + 4 \cdot 195^2 + 4 \cdot 195 + 3 \cdot 195 = 15{,}600 + 152{,}100 + 780 + 585 = 169{,}065$. For the tanh network, $400 \cdot 20 + 400^2 + 400 = 168{,}400$. Rounded to a hundred they are the 168.9, 169.1 and 168.4 thousand of Table 1 of Chung et al. The paper does not say what its counts include, and the same rule does not reproduce its counts for the music models, so this is a consistency check and not a reading of the paper.

### Result and check

$$ P = 3HI + 3H^2 + 6H, \qquad C = 6BTHI + 6BTH^2, \qquad \frac{3HI + 3H^2 + 3H}{4HI + 4H^2 + 4H} = \frac34 . $$

For the example config ($I = 3$, $H = 4$) this gives $36 + 48 + 24 = 108$ parameters, and for $B = 2$, $T = 5$ a forward pass of $720 + 960 = 1680$ FLOPs. The check `parameter-and-flop-count.check.py` counts the arrays that `init` returns, counts the multiply-adds of `cell` by replaying its products, verifies the redundancy of the biases by shifting them and comparing the outputs, and recomputes the speech sizes.
