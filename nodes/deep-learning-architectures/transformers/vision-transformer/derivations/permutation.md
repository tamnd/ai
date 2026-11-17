---
id: permutation
derives: [c-perm]
kind: proof
assumes: ["the network of the page with pos = none: no position embeddings", "the output is read from the class token, or from the average of the patch states", "a rearrangement of an image moves whole patches of the grid and leaves each patch unchanged"]
uses: [permutation-matrix@m2, matrix-multiplication@m2]
status: full
check: permutation.check.py
lean: null
sources:
  - {work: dosovitskiy-2021-image-worth-16x16-words, locator: "pp. 3 to 4", note: "eq. 1 to 4 and the inductive bias paragraph"}
  - {work: dosovitskiy-2021-image-worth-16x16-words, locator: "p. 17", note: "Table 8, the ablation without position embeddings"}
---

### Statement

Let $\pi$ be a permutation of the $N$ patches and $\pi(\mathbf{x})$ the image whose patch $i$ is patch $\pi(i)$ of $\mathbf{x}$. A ViT without position embeddings gives the same scores for $\pi(\mathbf{x})$ and $\mathbf{x}$, for every $\pi$ and all parameters. Consequently, on a test set of pairs $(\mathbf{x}, \pi(\mathbf{x}))$ with opposite labels, such a network classifies exactly one image of each pair correctly, an accuracy of exactly 50%, whatever its training.

### Assumptions

1. No position embeddings, so $\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \mathbf{x}_p^1 E; \dots; \mathbf{x}_p^N E]$ plus the bias of $E$ on each patch row. Used in step 1.
2. The output is a function of the class token's final state, or of the average of the final states of the patches. Used in step 4.
3. The classifier takes the class of largest score, with the same rule for ties on both images of a pair. Used in step 5.

### Steps

1. Write $\Pi$ for the $(N + 1) \times (N + 1)$ permutation matrix that fixes row 0, the class token, and moves the patch rows by $\pi$. The embedding is affine row by row, so $\mathbf{z}_0(\pi(\mathbf{x})) = \Pi\, \mathbf{z}_0(\mathbf{x})$.
2. Layer normalization, the MLP and the residual additions act on each row alone with the same weights, so they commute with $\Pi$: $f(\Pi \mathbf{z}) = \Pi f(\mathbf{z})$.
3. In a head, $\mathbf{q}, \mathbf{k}, \mathbf{v}$ are row-wise affine maps, so they become $\Pi\mathbf{q}$, $\Pi\mathbf{k}$, $\Pi\mathbf{v}$. Then $\Pi\mathbf{q}(\Pi\mathbf{k})^\top = \Pi \mathbf{q}\mathbf{k}^\top \Pi^\top$, and the softmax over each row of a matrix whose rows and columns are both permuted is the permuted softmax, $\operatorname{softmax}(\Pi S \Pi^\top) = \Pi \operatorname{softmax}(S) \Pi^\top$, because it normalizes each row over a set of entries that is only reordered. So $A' \mathbf{v}' = \Pi A \Pi^\top \Pi \mathbf{v} = \Pi A \mathbf{v}$, using $\Pi^\top \Pi = I$. The concatenation of heads and the output projection are row-wise, so $\operatorname{MSA}(\Pi\mathbf{z}) = \Pi \operatorname{MSA}(\mathbf{z})$.
4. By induction over the layers with steps 1 to 3, $\mathbf{z}_L(\pi(\mathbf{x})) = \Pi \mathbf{z}_L(\mathbf{x})$, and so for the final normalization. Row 0 is fixed by $\Pi$, so the class token's state is the same for both images; the average of the patch rows does not depend on their order. Either way, the scores are equal: the first statement.
5. If the two images of a pair have opposite labels and the same scores, the predicted class is the same for both and matches exactly one label. Averaging over pairs gives exactly one half: the second statement.

### Result and check

$$ \mathbf{s}(\pi(\mathbf{x})) = \mathbf{s}(\mathbf{x}) \quad \text{for every permutation } \pi \text{ of the patches, when } E_{\text{pos}} = 0 . $$

The argument uses nothing about the training, and it fails as soon as $E_{\text{pos}}$ is learned, since $\Pi \mathbf{z}_0 + E_{\text{pos}} \neq \Pi(\mathbf{z}_0 + E_{\text{pos}})$ unless $E_{\text{pos}}$ is invariant under $\Pi$. The check `permutation.check.py` permutes the patches of random images for random networks with each pooling and confirms the equality to $10^{-11}$, and confirms that learned position embeddings break it. The PyTorch test `test_the_positions_are_needed` measures the second statement after training.
