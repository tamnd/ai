---
id: restart-epochs
derives: [c-restart-epochs, c-restart-example]
kind: closed-form
assumes: ["SGDR with a first run of T_0 epochs, each run T_mult times longer than the one before, and T_mult >= 1", "a restart happens exactly at the end of each run"]
uses: []
status: full
check: restart-epochs.check.py
lean: null
sources:
  - {work: loshchilov-2017-sgdr, locator: "p. 4"}
  - {work: loshchilov-2017-sgdr, locator: "p. 8"}
---

### Statement

Let run $i = 0, 1, 2, \dots$ last $T_i = T_0 T_{\mathrm{mult}}^i$ epochs, and let restart $k$ be the start of run $k$.

(i) Restart $k$ happens at epoch $t_k = T_0 \frac{T_{\mathrm{mult}}^k - 1}{T_{\mathrm{mult}} - 1}$ for $T_{\mathrm{mult}} > 1$, and at $t_k = k T_0$ for $T_{\mathrm{mult}} = 1$.

(ii) For $T_0 = 10$ and $T_{\mathrm{mult}} = 2$ the restarts are at epochs 10, 30, 70, 150 and 310, and a budget of 150 epochs ends exactly at the end of the fourth run.

### Assumptions

1. The run lengths form a geometric sequence with ratio $T_{\mathrm{mult}} \ge 1$.
2. The rate restarts at the end of each run and at no other time.

### Steps

1. Restart $k$ comes after runs $0, \dots, k - 1$, so $t_k = \sum_{i=0}^{k-1} T_0 T_{\mathrm{mult}}^i$.
2. For $T_{\mathrm{mult}} \ne 1$ the geometric sum is $T_0 (T_{\mathrm{mult}}^k - 1)/(T_{\mathrm{mult}} - 1)$; for $T_{\mathrm{mult}} = 1$ each term is $T_0$ and the sum is $k T_0$. This gives (i).
3. With $T_0 = 10$ and $T_{\mathrm{mult}} = 2$, $t_k = 10(2^k - 1)$, which is 10, 30, 70, 150 and 310 for $k = 1, \dots, 5$. This gives (ii).
4. Inverting (i), the run that contains epoch $t$ has index $i = \lfloor \log_{T_{\mathrm{mult}}}(1 + t (T_{\mathrm{mult}} - 1)/T_0) \rfloor$ and $T_{\mathrm{cur}} = t - t_i$.

### Result and check

Loshchilov and Hutter take ensemble snapshots at epochs 30, 70 and 150 of a run with $T_0 = 10$ and $T_{\mathrm{mult}} = 2$, which are the ends of runs 1, 2 and 3 of (ii).
The check compares (i) with a loop that adds run lengths, for $T_0$ from 1 to 20, $T_{\mathrm{mult}}$ from 1 to 4 and $k$ up to 12, compares the index formula of step 4 with the loop at 5,000 times, including fractional ones, and confirms the list of (ii).
