---
id: luong-scores
derives: [c-dot-general, c-concat-additive, c-local-m-global, c-local-sum, c-location-blind]
kind: proof
assumes: ["the scores of Luong et al.'s Eq. 8 and 9 and the local attention of their Section 3.2, with the source states of one sentence"]
uses: [softmax-regression@al]
status: full
check: luong-scores.check.py
lean: null
sources:
  - {work: luong-2015-effective-approaches, locator: "p. 3", note: "Eq. 7 to 9, the dot, general, concat and location scores"}
  - {work: luong-2015-effective-approaches, locator: "p. 4", note: "Section 3.2, local-m, local-p, Eq. 10 and 11"}
  - {work: bahdanau-2015-neural-machine-translation, locator: "p. 12", note: "A.1.2, the additive score"}
---

### Statement

Five relations between the scores and the variants of Luong et al.

1. The general score $h_t^\top W_a \bar h_s$ with $W_a = I$ is the dot score $h_t^\top \bar h_s$; every general score is the dot score of $W_a^\top h_t$ with $\bar h_s$.
2. The concat score $v_a^\top \tanh(W_a [h_t; \bar h_s])$, with $W_a = [W_1\ W_2]$ split by columns, is the additive score $v_a^\top \tanh(W_1 h_t + W_2 \bar h_s)$ of Bahdanau et al. with the query $h_t$ in place of $s_{i-1}$.
3. Local-m attention, the softmax restricted to the positions $s$ with $|s - t| \le D$, is global attention when the window covers the sentence, $D \ge \max(t, S - 1 - t)$ with positions $0, \dots, S - 1$.
4. The weights of local-p attention sum to at most 1, with equality only if all the weight inside the window sits at $s = p_t$; the context is a weighted sum of the states but not an average of them.
5. The location score $a_t = \operatorname{softmax}(W_a h_t)$ does not depend on the source states, so the weights a location-based model gives a sentence change only through $h_t$.

### Assumptions

1. The scores are those of Eq. 8 and the location score is Eq. 9. Used in items 1, 2 and 5.
2. Local-p multiplies the weights of the softmax over the window by $\exp(-(s - p_t)^2 / 2\sigma^2)$ without renormalizing, as Eq. 11 is written. Used in item 4.

### Steps

1. $h_t^\top I \bar h_s = h_t^\top \bar h_s$, and $h_t^\top W_a \bar h_s = (W_a^\top h_t)^\top \bar h_s$: general is dot after a linear map of the query, and it has $n^2$ more weights.
2. $W_a [h_t; \bar h_s] = W_1 h_t + W_2 \bar h_s$ by the block product. Bahdanau et al. take $W_1 = W_a$ applied to $s_{i-1}$ and $W_2 = U_a$, so the two scores are the same function of their two arguments; the difference between the two models is the query, the current top state $h_t$ against the previous state $s_{i-1}$, and where the context enters (Luong et al., p. 3).
3. If the window contains every position of the sentence the restricted softmax and the global softmax have the same terms.
4. Inside the window each weight is the softmax weight $\hat a_s \ge 0$, with $\sum_s \hat a_s = 1$, times $g_s = \exp(-(s - p_t)^2/2\sigma^2) \in (0, 1]$, so $\sum_s \hat a_s g_s \le 1$, with equality only if $\hat a_s = 0$ wherever $g_s < 1$, that is, wherever $s \ne p_t$.
5. $W_a h_t$ has no argument from the source.

### Result and check

Dot and general are the multiplicative scores and concat is the additive one, so the comparison of Luong et al.'s Table 4 is between the two families of scores and not only between three formulas. Local-p shrinks the context toward 0 by its Gaussian, which the layer after it has to learn to undo. The check `luong-scores.check.py` verifies the five items numerically on random inputs with the functions of the NumPy reference.
