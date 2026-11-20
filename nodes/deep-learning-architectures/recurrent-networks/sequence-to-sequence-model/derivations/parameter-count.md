---
id: parameter-count
derives: [c-params, c-flops, c-count-sutskever, c-count-recurrent, c-state-8000, c-softmax-cost]
kind: count
assumes: ["two embeddings of V_x x d_e and V_y x d_e", "an encoder and a decoder of L LSTM layers each, with separate weights; the first layer reads d_e features, the others d_h; each layer has input weights, recurrent weights and one bias for four gates", "a softmax layer of d_h x V_y weights and V_y biases", "a multiply-add counts as 2 FLOPs and only the products are counted"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-count.check.py
lean: null
sources:
  - {work: sutskever-2014-sequence-to-sequence-learning, locator: "Section 3.4"}
---

### Statement

1. The model has $N = (V_x + V_y) d_e + 2\big[4 d_h (d_e + d_h) + 4 d_h + (L - 1)(8 d_h^2 + 4 d_h)\big] + d_h V_y + V_y$ parameters.
2. A teacher-forced forward pass over $B$ pairs of a source of $S$ tokens and a target of $T$ tokens costs $C = B (S + T)\big[8 d_h (d_e + d_h) + 16 (L - 1) d_h^2\big] + 2 B T d_h V_y$ FLOPs in its products.
3. With the sizes of Sutskever et al., $V_x = 160{,}000$, $V_y = 80{,}000$, $d_e = d_h = 1000$ and $L = 4$, and without the biases, the model has exactly $384{,}000{,}000$ weights.
4. Of these, $64{,}000{,}000$ are the weights of the LSTM layers, $32{,}000{,}000$ in the encoder and as many in the decoder; the embeddings hold $240{,}000{,}000$ and the softmax layer $80{,}000{,}000$.
5. The vector $v$ that carries a sentence from the encoder to the decoder has $2 L d_h = 8000$ numbers.
6. With these sizes, each target token costs $2 d_h V_y = 160{,}000{,}000$ FLOPs in the softmax layer, 2.5 times the $64{,}000{,}000$ of one step of a stack.

### Assumptions

1. The layer of the LSTM page: $z = x W + h U + b$ with $W$ of $n_{\text{in}} \times 4 d_h$, $U$ of $d_h \times 4 d_h$ and $b$ of $4 d_h$. Used in steps 1, 2 and 4.
2. The first layer of each stack reads embeddings of size $d_e$; each layer above it reads the $d_h$ hidden units of the layer below. Used in steps 1 and 2.
3. A product of an $m \times k$ and a $k \times n$ matrix costs $2 m k n$ FLOPs; the embedding lookups are indexing and cost none. Used in steps 2 and 6.

### Steps

1. One LSTM layer reading $n_{\text{in}}$ features has $4 d_h n_{\text{in}} + 4 d_h^2 + 4 d_h$ parameters. A stack has one layer with $n_{\text{in}} = d_e$ and $L - 1$ with $n_{\text{in}} = d_h$, so $4 d_h (d_e + d_h) + 4 d_h + (L - 1)(8 d_h^2 + 4 d_h)$; the encoder and the decoder have one each. Adding the embeddings and the softmax layer gives statement 1.
2. One step of a layer makes the products $x W$ and $h U$, $2 \cdot 4 d_h (n_{\text{in}} + d_h)$ FLOPs per sequence: $8 d_h (d_e + d_h)$ for the first layer and $16 d_h^2$ for the others. The encoder runs $S$ steps and the decoder $T$, and the softmax layer adds $2 d_h V_y$ at each of the $T$ target steps. This is statement 2.
3. Without biases, the embeddings have $(160{,}000 + 80{,}000) \times 1000 = 240{,}000{,}000$ weights, the softmax layer $1000 \times 80{,}000 = 80{,}000{,}000$, and each LSTM layer $4 \times 1000 \times (1000 + 1000) = 8{,}000{,}000$, since $d_e = d_h$. Four layers make $32{,}000{,}000$ per stack and $64{,}000{,}000$ for both. The sum is $240 + 80 + 64 = 384$ million. These are statements 3 and 4.
4. $v$ is the final $h$ and $c$ of each of the $L$ layers of the encoder, $2 \times 4 \times 1000 = 8000$ numbers. This is statement 5.
5. Per target token, the softmax product is $2 \times 1000 \times 80{,}000 = 160{,}000{,}000$ FLOPs, and one step of a stack is $8 \times 1000 \times 2000 + 3 \times 16 \times 1000^2 = 64{,}000{,}000$; the ratio is 2.5. This is statement 6.

### Result and check

$$ N_{\text{weights}} = 240\text{M} + 64\text{M} + 80\text{M} = 384\text{M}, \qquad |v| = 2 L d_h = 8000 . $$

With biases the count is $384{,}112{,}000$. The example config of the shape contract has 613 parameters and costs 9,024 FLOPs on its batch. The check `parameter-count.check.py` counts the arrays of `init` on random configs, replays the products of `forward`, and recomputes the counts of Sutskever et al.
