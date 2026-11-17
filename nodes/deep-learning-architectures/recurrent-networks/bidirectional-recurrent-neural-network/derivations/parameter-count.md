---
id: parameter-count
derives: [c-params, c-flops, c-graves-weights, c-torch-biases]
kind: count
assumes: ["L layers of D directions with d_h LSTM cells each; the first layer reads d_x features, every layer above it the D d_h outputs of the layer below", "each direction has input weights, recurrent weights and one bias for four gates", "an affine output layer of D d_h x K weights and K biases", "a multiply-add counts as 2 FLOPs and only the products are counted"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-count.check.py
lean: null
sources:
  - {work: graves-2013-speech-recognition-deep-recurrent, locator: "Section 2, Section 4 and Table 1"}
  - {work: pytorch-torch-nn-lstm, locator: "the title page"}
---

### Statement

1. The network has $N = D\big[4 d_h (d_x + d_h) + 4 d_h\big] + (L - 1) D \big[4 d_h (D d_h + d_h) + 4 d_h\big] + D d_h K + K$ parameters; with tanh cells, replace every 4 by 1.
2. A forward pass over $B$ sequences of $T$ steps costs $C = 8 D B T d_h (d_x + d_h) + 8 (L - 1) D B T d_h (D d_h + d_h) + 2 B T D d_h K$ FLOPs in its products.
3. With $d_x = 123$ and $K = 62$ (61 phonemes and the blank of CTC), a bidirectional LSTM of 3 layers of 250 cells per direction has $3{,}783{,}062$ parameters, and a unidirectional one of 3 layers of 421 cells has $3{,}783{,}168$: both round to the 3.8M weights that Graves et al. list for CTC-3l-250h and CTC-3l-421h-uni, a difference of 106 parameters.
4. torch.nn.LSTM keeps a second bias per direction and layer, so its recurrent part has $4 L D d_h$ more parameters than this network.

### Assumptions

1. A direction is the LSTM layer of the LSTM page, $z = x W + h U + b$, with $W$ of $n_{\text{in}} \times 4 d_h$, $U$ of $d_h \times 4 d_h$ and $b$ of $4 d_h$; a tanh cell has $W$ of $n_{\text{in}} \times d_h$, $U$ of $d_h \times d_h$ and $b$ of $d_h$. Used in steps 1 and 2.
2. The upper layers read the concatenation of the $D$ directions below, $n_{\text{in}} = D d_h$, as in the deep bidirectional LSTM of Graves et al., where every hidden layer receives input from both the forward and backward layers at the level below. Used in steps 1 to 3.
3. A product of an $m \times k$ and a $k \times n$ matrix costs $2 m k n$ FLOPs. Used in step 2.
4. The cells of Graves et al. also have peephole weights from the cell to the gates, $3 d_h$ per direction and layer, which the count of statement 3 leaves out. Used in step 3.

### Steps

1. A direction reading $n_{\text{in}}$ features has $4 d_h n_{\text{in}} + 4 d_h^2 + 4 d_h$ parameters. The first layer has $D$ of them with $n_{\text{in}} = d_x$, and each of the $L - 1$ others has $D$ with $n_{\text{in}} = D d_h$. The output layer adds $D d_h K + K$. This is statement 1.
2. At every step, each direction makes the products $x W$ and $h U$, $2 \cdot 4 d_h (n_{\text{in}} + d_h)$ FLOPs per sequence, and the output layer makes $2 D d_h K$. Summing over the $B T$ steps, the $D$ directions and the $L$ layers gives statement 2.
3. Bidirectional, $D = 2$, $d_h = 250$: the first layer has $2(1000 \cdot 373 + 1000) = 748{,}000$, each upper layer $2(1000 \cdot 750 + 1000) = 1{,}502{,}000$, the output layer $500 \cdot 62 + 62 = 31{,}062$, in all $3{,}783{,}062$. Unidirectional, $D = 1$, $d_h = 421$: $1684 \cdot 544 + 1684 = 917{,}780$, then $1684 \cdot 842 + 1684 = 1{,}419{,}612$ twice, and $421 \cdot 62 + 62 = 26{,}164$, in all $3{,}783{,}168$. The peepholes of assumption 4 add $4500$ and $3789$, which keeps both at 3.79M. This is statement 3.
4. torch.nn.LSTM computes $x W_{ih}^\top + b_{ih} + h W_{hh}^\top + b_{hh}$; the two biases are one bias $b_{ih} + b_{hh}$ of $4 d_h$ numbers, stored twice. This is statement 4.

### Result and check

$$ N(\text{CTC-3l-250h}) = 3{,}783{,}062, \qquad N(\text{CTC-3l-421h-uni}) = 3{,}783{,}168 . $$

The example config of the shape contract has 717 parameters and costs 23,328 FLOPs on its batch. The check `parameter-count.check.py` counts the arrays of `init` on random configs of both cells and both numbers of directions, replays the products of `forward` with an instrumented matrix product, recomputes the two networks of Graves et al., and counts the parameters of torch.nn.LSTM if PyTorch is installed.
