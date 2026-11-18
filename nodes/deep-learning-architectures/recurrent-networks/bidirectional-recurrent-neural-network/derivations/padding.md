---
id: padding
derives: [c-padding, c-padding-naive]
kind: proof
assumes: ["a batch of sequences of lengths n_b <= T, padded at the end to T steps", "both directions start from a zero state", "the recurrent cell maps a zero state and any input to some state; the mask sets a state to zero"]
uses: []
status: full
check: padding.check.py
lean: null
sources:
  - {work: pytorch-torch-nn-lstm, locator: "the title page"}
  - {work: huang-2015-bidirectional-lstm-crf, locator: "Section 2.2"}
---

### Statement

1. Let $R_n$ reverse the first $n$ steps of a sequence of $T$ steps and leave the others in place. Running the backward direction over $R_{n_b}(\mathbf{x}_b)$ in increasing order and applying $R_{n_b}$ to its states gives, at the real steps $t \le n_b$, the backward states of the sequence $\mathbf{x}_{b,1..n_b}$ alone.
2. Running the backward direction over all $T$ steps from $t = T$ down to 1, with its state set to zero after each padded step $t > n_b$, gives the same states at the real steps.
3. Running it from $t = T$ without the mask, as an RNN layer does on an unpacked padded batch, gives in general different states at every real step of a sequence with $n_b < T$: they read the padding.

### Assumptions

1. Padding at the end of each sequence. Used in steps 1 to 3.
2. The zero initial state, the one of Huang et al., who reset the hidden states to 0 at the beginning of each sentence. Used in steps 1 and 2.

### Steps

1. $R_n$ maps step $t \le n$ to step $n + 1 - t$. Its first $n$ outputs are $\mathbf{x}_n, \dots, \mathbf{x}_1$, the sequence alone in reverse, and a forward run over them from the zero state computes exactly the backward states of the sequence alone, the state of real step $t$ at position $n + 1 - t$. The padded positions come after them and cannot change them. $R_n$ is its own inverse, so applying it to the states puts the state of step $t$ back at step $t$. This is statement 1.
2. In the masked scan from $T$ down, the steps $t > n_b$ come first; after each of them the state is zero. The scan therefore reaches step $n_b$ from a zero state and from there on reads $\mathbf{x}_{n_b}, \dots, \mathbf{x}_1$, the same inputs as in step 1. This is statement 2.
3. Without the mask, the state at step $n_b$ is a function of the padded inputs $\mathbf{x}_{n_b + 1}, \dots, \mathbf{x}_T$, and the recurrence carries it to every real step. With generic weights the states differ. This is statement 3, and the reason for packing a padded batch, which tells the layer the length of each sequence.

### Result and check

$$ \overleftarrow{\mathbf{h}}_{b, t} = \big[R_{n_b}\, \mathrm{RNN}(R_{n_b} \mathbf{x}_b)\big]_t = \mathrm{scan}_{T \to 1}^{\text{masked}}(\mathbf{x}_b)_t \ne \mathrm{scan}_{T \to 1}(\mathbf{x}_b)_t, \qquad t \le n_b < T . $$

The check `padding.check.py` builds random ragged batches and compares the reversal of the NumPy reference with a masked reverse scan and with each sequence run alone, for both cells, and checks that the unmasked reverse scan differs.
