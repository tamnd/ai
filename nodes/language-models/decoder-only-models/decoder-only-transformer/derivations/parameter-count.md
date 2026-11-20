---
id: parameter-count
derives: [c-params, c-params-gpt3, c-params-13b, c-params-gpt2, c-params-kaplan, c-emb-share]
kind: count
assumes: ["a vocabulary of V tokens, a context of T tokens and learned position embeddings for all T places", "L pre-norm layers of width D, each with an affine qkv projection, an affine output projection, an MLP of two affine layers with F hidden units, and two layer normalizations with a gain and a bias", "a final layer normalization, and an unembedding that is the token embedding transposed, with no bias"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-count.check.py
lean: null
sources:
  - {work: radford-2018-improving-language-understanding, locator: "p. 3", note: "eq. 2, the embeddings and the softmax over W_e transposed"}
  - {work: radford-2019-language-models-unsupervised-multitask, locator: "p. 4", note: "Section 2.3 and Table 2"}
  - {work: brown-2020-language-models-few-shot, locator: "p. 8", note: "Table 2.1"}
  - {work: brown-2020-language-models-few-shot, locator: "p. 46", note: "Table D.1"}
  - {work: kaplan-2020-scaling-laws, locator: "p. 6", note: "eq. 2.1"}
---

### Statement

1. The network has $N_{\text{total}} = VD + TD + L\big(4D^2 + 2DF + 9D + F\big) + 2D$ parameters, of which the $L$ layers hold $N = L(4D^2 + 2DF + 9D + F)$.
2. With $V = 50{,}257$, $T = 2048$ and $F = 4D$, the eight configurations of Table 2.1 of Brown et al. give 125,226,240, 355,871,744, 760,300,032, 1,315,723,264, 2,651,553,280, 6,658,404,352, 12,853,386,240 and 174,604,259,328 parameters, which round to the 125, 356, 760, 1,320, 2,650, 6,660, 12,850 and 174,600 million of Table D.1.
3. The width of GPT-3 13B is 5120, as its 40 heads of 128 make: the width 5140 printed in Table 2.1 would give 12,952,938,780 parameters, about 100 million more than Table D.1 lists.
4. With $T = 1024$ and $F = 4D$ the four models of Table 2 of GPT-2 have 124,439,808, 354,823,168, 774,030,080 and 1,557,611,200 parameters, more than the 117M, 345M, 762M and 1542M of the table.
5. With $F = 4D$, $N = 12LD^2(1 + 13/(12D))$, so Kaplan et al.'s $N \approx 12 n_{\text{layer}} d_{\text{model}}^2$ is short by $13/(12D)$: 0.14% at $D = 768$ and 0.0088% at $D = 12{,}288$.
6. The embeddings, $(V + T)D$ numbers, are 31.7% of the parameters of the smallest GPT-2 and 0.37% of those of GPT-3 175B.

### Assumptions

1. The parts of the network listed above and no others: in particular no bias in the unembedding and no separate output matrix. Used in steps 1 to 4.
2. For statement 2, the vocabulary of GPT-2, which GPT-3 keeps with its reversible tokenization, the $d_{\text{ff}} = 4 d_{\text{model}}$ and $n_{\text{ctx}} = 2048$ of Section 2.1, and the widths of Table 2.1 with 5120 for the 13B model. Used in step 5.
3. For statement 4, $d_{\text{ff}} = 4 d_{\text{model}}$ as in GPT-1, and the vocabulary and context of Section 2.3 of GPT-2. Used in step 7.

### Steps

1. The token embedding is a $V \times D$ table and the position embedding a $T \times D$ table.
2. In a layer, the qkv projection has $D \times 3D$ weights and $3D$ biases and the output projection $D^2 + D$, together $4D^2 + 4D$. The number of heads does not enter, since the heads split the $3D$ outputs among them. The MLP has $DF + F$ and $FD + D$, and the two normalizations $2 \times 2D$. The sum is $4D^2 + 2DF + 9D + F$.
3. The final normalization has $2D$. The unembedding is $W_e^\top$, the token embedding again, and adds nothing.
4. The sum of steps 1 to 3 is statement 1.
5. GPT-3 Small: $D = 768$, $F = 3072$, $L = 12$. A layer has $4 \cdot 768^2 + 2 \cdot 768 \cdot 3072 + 9 \cdot 768 + 3072 = 7{,}087{,}872$, and twelve of them $85{,}054{,}464$; the embeddings have $(50{,}257 + 2048) \cdot 768 = 40{,}170{,}240$ and the final normalization $1536$. The total is $125{,}226{,}240$, and the same arithmetic gives the other seven models. Each rounds to the entry of Table D.1 at the precision it is printed with: to the million up to 760M, to ten million from 1,320M to 12,850M, and to a hundred million for 174,600M.
6. With $D = 5140$, $F = 20{,}560$ and $L = 40$, statement 1 gives $12{,}952{,}938{,}780$, which rounds to 12,950M and not to 12,850M. The table also lists 40 heads of dimension 128, and $40 \times 128 = 5120$; with $D = 5120$ the count is $12{,}853{,}386{,}240$, which rounds to 12,850M. Statement 3 follows.
7. GPT-2: $T = 1024$; with $D = 768$ and $L = 12$ the layers are those of step 5, $85{,}054{,}464$, and the embeddings $(50{,}257 + 1024) \cdot 768 = 39{,}383{,}808$, so the total is $124{,}439{,}808$. The other three follow in the same way. The paper does not say how it counted its sizes.
8. With $F = 4D$, a layer has $4D^2 + 8D^2 + 9D + 4D = 12D^2 + 13D$, so $N = 12LD^2 + 13LD = 12LD^2\,(1 + 13/(12D))$. This is statement 5.
9. The smallest GPT-2 has $39{,}383{,}808$ embedding parameters of $124{,}439{,}808$, or 31.65%; GPT-3 175B has $(50{,}257 + 2048) \cdot 12{,}288 = 642{,}723{,}840$ of $174{,}604{,}259{,}328$, or 0.368%.

### Result and check

$$ N_{\text{total}}(\text{GPT-3 175B}) = 642{,}723{,}840 + 96\,(12 \cdot 12{,}288^2 + 13 \cdot 12{,}288) + 24{,}576 = 174{,}604{,}259{,}328 . $$

The check `parameter-count.check.py` compares the closed form with the parameters that the NumPy reference creates for random configurations, tied and untied, with and without position embeddings, recomputes the models of both papers and the 13B model at both widths, and counts the parameters of torch.nn.TransformerEncoderLayer when torch is installed.
