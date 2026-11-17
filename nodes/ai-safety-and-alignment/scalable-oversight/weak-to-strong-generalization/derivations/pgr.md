---
id: pgr
derives: [c-pgr-endpoints, c-pgr-undefined, c-pgr-affine, c-pgr-range, c-pgr-agreement]
kind: closed-form
assumes: ["the three performances are real numbers measured with one metric on one test set", "the strong ceiling differs from the weak performance"]
uses: []
status: full
check: pgr.check.py
lean: null
sources:
  - {work: burns-2023-weak-to-strong-generalization, locator: "p. 6"}
  - {work: burns-2023-weak-to-strong-generalization, locator: "p. 12"}
---

### Statement

Let $P_w$ be the weak performance, $P_{w2s}$ the weak-to-strong performance and $P_s$ the strong ceiling performance, all measured with one metric on one test set, and define

$$ \mathrm{PGR} = \frac{P_{w2s} - P_w}{P_s - P_w}. $$

Then (i) $\mathrm{PGR} = 0$ exactly when $P_{w2s} = P_w$ and $\mathrm{PGR} = 1$ exactly when $P_{w2s} = P_s$;
(ii) PGR is undefined when $P_s = P_w$, and its magnitude grows without bound as $P_s - P_w \to 0$ with $P_{w2s} - P_w$ fixed;
(iii) for any $a \ne 0$ and any $b$, replacing every performance $P$ by $aP + b$ leaves PGR unchanged, so accuracy and error rate give the same value;
(iv) PGR is negative when the student is worse than its supervisor and above 1 when it beats the ceiling;
(v) if the student's predictions agree with the supervisor's on every test input, then for any metric computed from predictions and labels $P_{w2s} = P_w$ and PGR is 0.

### Assumptions

1. $P_w$, $P_{w2s}$, $P_s$ are real numbers computed with the same metric on the same test inputs.
2. For (i), (iii) and (iv), $P_s \ne P_w$.
3. For (v), the metric depends only on the predictions and the ground-truth labels of the test inputs.

### Steps

1. With $g = P_s - P_w \ne 0$, $\mathrm{PGR} = (P_{w2s} - P_w)/g$ is an affine, strictly monotone function of $P_{w2s}$; it is $0$ at $P_{w2s} = P_w$ and $g/g = 1$ at $P_{w2s} = P_s$, and these are the only points with those values.
2. At $g = 0$ the denominator is zero. If the numerator is nonzero the ratio has no value; if it is zero too, every number would do. Either way PGR is undefined, and for fixed numerator $\delta \ne 0$, $|\delta/g| \to \infty$ as $g \to 0$.
3. Under $P \mapsto aP + b$ the numerator becomes $a P_{w2s} + b - a P_w - b = a(P_{w2s} - P_w)$ and the denominator $a(P_s - P_w)$; the factor $a \ne 0$ cancels. Error rate is accuracy under $a = -1$, $b = 1$.
4. If the metric increases with quality and $P_s > P_w$, then $P_{w2s} < P_w$ makes the numerator negative and $P_{w2s} > P_s$ makes the ratio exceed 1.
5. Agreement on every test input means the student's predictions equal the supervisor's, so any function of predictions and labels takes the same value for both: $P_{w2s} = P_w$, and by step 1 PGR is 0.

### Result and check

The check confirms the end points, the error on a zero gap and the blow-up near it, affine invariance on random triples with positive and negative $a$, the signs outside $[0, 1]$, and that a student copying the supervisor's predictions has PGR 0 for accuracy.
