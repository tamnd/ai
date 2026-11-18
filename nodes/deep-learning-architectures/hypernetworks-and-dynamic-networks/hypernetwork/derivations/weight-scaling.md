---
id: weight-scaling
derives: [c-rowscale, c-scaled-is-full, c-memory, c-rnn-params, c-init-scale]
kind: proof
assumes: ["the HyperRNN of eq-full and eq-scaled, with N_h hidden units, N_x inputs and embeddings of size N_z", "the hyper cell of eq-hyper-cell gives z_h, z_x and z_b at each step", "for the initialization, the HyperLSTM of Appendix A.2.2 of the paper"]
uses: [matrix-multiplication@m2]
status: full
check: weight-scaling.check.py
lean: null
sources:
  - {work: ha-2016-hypernetworks, locator: "p. 5", note: "eq. 5 and the memory of the general form"}
  - {work: ha-2016-hypernetworks, locator: "p. 6", note: "eqs. 7 and 8, the weight scaling vectors"}
  - {work: ha-2016-hypernetworks, locator: "p. 20", note: "Appendix A.2.3, the initialization"}
---

### Statement

1. Scaling the rows of a matrix is an elementwise product after it: $\big(\operatorname{diag}(\delta) W\big) v = \delta \odot (W v)$.
2. The memory-efficient HyperRNN of eq-scaled is the HyperRNN of eq-full whose tensors have the slices $W_{hz}^{(k)} = \operatorname{diag}(W_{hz}[:, k])\, W_h$ and $W_{xz}^{(k)} = \operatorname{diag}(W_{xz}[:, k])\, W_x$, so it computes the same states.
3. Per sequence and per step, eq-full builds $N_h(N_h + N_x)$ numbers of weights, and eq-scaled $2N_h$ numbers of scaling vectors. The main network of eq-full has $N_z N_h (N_h + N_x) + N_z N_h + N_h$ parameters and that of eq-scaled $N_h (N_h + N_x) + 3 N_z N_h + N_h$; with the hyper cell of $\hat N$ units, both add $\hat N(\hat N + N_h + N_x + 1) + 3 N_z \hat N + 2N_z$.
4. With the initialization of Appendix A.2.3, every weight scaling vector of the HyperLSTM equals $0.1$ in every entry, for every input, until training moves $W_{\hat h h}$ or $W_{\hat h x}$ from zero.

### Steps

1. Entry $n$ of $\operatorname{diag}(\delta) W v$ is $\delta_n \sum_j W_{nj} v_j = \delta_n (Wv)_n$.
2. In eq-full, $W_h(z_h) = \sum_k z_{h,k} W_{hz}^{(k)}$. With the slices of the statement, $\sum_k z_{h,k} \operatorname{diag}(W_{hz}[:, k]) W_h = \operatorname{diag}\big(\sum_k z_{h,k} W_{hz}[:, k]\big) W_h = \operatorname{diag}(W_{hz} z_h) W_h = \operatorname{diag}(d_h) W_h$. By step 1, $W_h(z_h) h_{t-1} = d_h \odot (W_h h_{t-1})$, the term of eq-scaled; the same holds for $W_x$, and the biases are the same in both forms. The hyper cell is shared, so by induction over $t$ the states agree.
3. eq-full keeps $W_h(z_h) \in \mathbb{R}^{N_h \times N_h}$ and $W_x(z_x) \in \mathbb{R}^{N_h \times N_x}$ for each sequence, and eq-scaled keeps $d_h, d_x \in \mathbb{R}^{N_h}$. In eq-full, $W_{hz}$ has $N_h N_h N_z$ entries, $W_{xz}$ has $N_h N_x N_z$, $W_{bz}$ has $N_h N_z$ and $b_0$ has $N_h$. In eq-scaled, $W_h$ and $W_x$ have $N_h(N_h + N_x)$, $W_{hz}$, $W_{xz}$ and $W_{bz}$ have $N_h N_z$ each, and $b_0$ has $N_h$. The hyper cell has $\hat N^2 + \hat N(N_h + N_x) + \hat N$ and its three projections $3N_z\hat N$, with two biases of $N_z$.
4. With $W_{\hat h h} = 0$ and $b_{\hat h h} = \mathbf{1}$, $z_h = W_{\hat h h} \hat h_t + b_{\hat h h} = \mathbf{1} \in \mathbb{R}^{N_z}$ for any $\hat h_t$. With every entry of $W_{hz}$ equal to $0.1/N_z$, $d_h = W_{hz}\mathbf{1}$ has entries $N_z \cdot 0.1/N_z = 0.1$. The same holds for $z_x$ and $d_x$. The bias embedding $z_b = W_{\hat h b}\hat h_t$ is small, since $W_{\hat h b}$ is drawn from $\mathcal{N}(0, 0.01)$, and $W_{bz} = 0$ removes it, so the bias of the main LSTM starts at $b_0 = 0$.

### Result and check

The weights of the full form take $N_z$ times the parameters of those of a basic RNN, and it builds $(N_h + N_x)/2$ times as many numbers per step as the row-scaling form, which keeps the weights of the main network a full matrix that the hyper cell only rescales, row by row. The check `weight-scaling.check.py` tests step 1 on random matrices, runs `hyper_rnn` and `hyper_rnn_full` of the NumPy reference with the tensors of `full_from_scaled`, counts the parameters of both forms against `rnn_param_count`, measures the arrays that each form builds per step, and runs `hyper_lstm` with the initialization of `init_lstm` on random inputs.
