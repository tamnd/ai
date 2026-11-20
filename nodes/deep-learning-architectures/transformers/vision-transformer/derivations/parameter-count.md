---
id: parameter-count
derives: [c-params, c-params-models, c-encoder-share]
kind: count
assumes: ["square images of G x G patches of P x P pixels with C channels, N = G^2", "a patch embedding with weights and a bias, a class token, and learned position embeddings for all N + 1 tokens", "L pre-norm encoder layers of width D, each with an affine qkv projection, an affine output projection, an MLP of two affine layers with D_MLP hidden units, and two layer normalizations with a gain and a bias", "a final layer normalization and a linear head to K classes"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-count.check.py
lean: null
sources:
  - {work: dosovitskiy-2021-image-worth-16x16-words, locator: "pp. 3 to 5", note: "eq. 1 to 4 and Table 1"}
  - {work: dosovitskiy-2021-image-worth-16x16-words, locator: "p. 13", note: "eq. 5 to 8"}
---

### Statement

1. The network has $N_{\text{params}} = (P^2 C + 1) D + D + (N + 1) D + L\big(4D^2 + 2 D D_{\text{MLP}} + 9D + D_{\text{MLP}}\big) + 2D + DK + K$ parameters.
2. With $K = 1000$ at 224 pixels, ViT-B/16 has 86,567,656 parameters, ViT-L/16 304,326,632, ViT-L/32 306,535,400 and ViT-H/14 632,045,800.
3. The encoder layers hold 85,054,464 of the parameters of ViT-B/16, 98.3%, and fine-tuning at 384 pixels adds $(577 - 197) \times 768 = 291{,}840$, all in the position embeddings.

### Assumptions

1. The parts of the network listed above and no others; in particular the head of fine-tuning, a single linear layer, and no hidden pre-logits layer. Used in steps 1 to 4.
2. The configurations of Table 1, with the patch size in the name of each model. Used in step 5.

### Steps

1. The patch embedding $E$ maps $P^2 C$ numbers to $D$: $P^2 C D$ weights and $D$ biases. The class token has $D$ parameters and $E_{\text{pos}}$ has one row of $D$ for each of the $N + 1$ tokens.
2. In a layer, the qkv projection has $D \times 3D$ weights and $3D$ biases, and the output projection $D^2 + D$: $4D^2 + 4D$. The number of heads does not enter, since the heads split the $3D$ outputs among them. The MLP has $D D_{\text{MLP}} + D_{\text{MLP}}$ and $D_{\text{MLP}} D + D$, and the two normalizations $2 \times 2D$. The sum is $4D^2 + 2 D D_{\text{MLP}} + 9D + D_{\text{MLP}}$.
3. The final normalization has $2D$, and the head $DK + K$.
4. The sum of steps 1 to 3 is statement 1.
5. ViT-B/16: $P = 16$, $C = 3$, $D = 768$, $N = 196$, so the embedding has $768 \cdot 768 + 768 + 768 + 197 \cdot 768 = 742{,}656$; a layer with $D_{\text{MLP}} = 3072$ has $4 \cdot 768^2 + 2 \cdot 768 \cdot 3072 + 9 \cdot 768 + 3072 = 7{,}087{,}872$, and twelve of them $85{,}054{,}464$; the head has $1536 + 768{,}000 + 1000 = 770{,}536$. The total is $86{,}567{,}656$. The same arithmetic gives the other models of statement 2, and statement 3 is $85{,}054{,}464 / 86{,}567{,}656 = 0.9825$ and the extra 380 rows of $E_{\text{pos}}$.

### Result and check

$$ N_{\text{params}}(\text{ViT-B/16}) = 742{,}656 + 12 \times 7{,}087{,}872 + 770{,}536 = 86{,}567{,}656 . $$

Table 1 lists 86M, 307M and 632M; ViT-L/32 rounds to 307M and ViT-L/16 to 304M. The check `parameter-count.check.py` compares the closed form with the parameters that the NumPy reference creates for random configurations, the four variants of pooling and position, and the head with a hidden layer, recomputes the models of the paper, and counts the parameters of torch.nn.TransformerEncoderLayer when torch is installed.
