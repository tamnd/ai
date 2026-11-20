---
id: schedules
derives: [c-schedules]
kind: identity
assumes: ["the noise levels are sigma_i = (sigma_min^(1/rho) + (i - 1)/(N - 1) (sigma_max^(1/rho) - sigma_min^(1/rho)))^rho for i = 1, ..., N", "sigma_min = 0.002, sigma_max = 80, rho = 7"]
uses: [normal-distribution@m2, expected-value@m2, variance@m2]
status: full
check: schedules.check.py
lean: null
sources:
  - {work: song-2023-consistency-models, locator: "p. 26, Appendix C", note: "the schedule functions N(k) and mu(k) of consistency training"}
  - {work: song-2023-improved-techniques-training-consistency, locator: "p. 4, Table 1", note: "s0 = 2, s1 = 150, mu0 = 0.9 on CIFAR-10 for the original schedule; s0 = 10, s1 = 1280 and the weighting 1 / (sigma_{i+1} - sigma_i) for the improved one"}
  - {work: song-2023-improved-techniques-training-consistency, locator: "pp. 7 to 8, Eq. 9 and Eq. 10", note: "the doubling curriculum capped at 1281, the density of log sigma under a uniform index, and the discretised lognormal with P_mean = -1.1, P_std = 2.0"}
---

### Statement

Song and colleagues' consistency training grows the number of noise levels and the decay of the target network with the iteration $k$ out of $K$:

$$ N(k) = \Bigl\lceil \sqrt{\tfrac{k}{K}\bigl((s_1 + 1)^2 - s_0^2\bigr) + s_0^2} - 1 \Bigr\rceil + 1 , \qquad \mu(k) = \exp\Bigl(\frac{s_0 \log \mu_0}{N(k)}\Bigr) . $$

With the CIFAR-10 values $s_0 = 2$, $s_1 = 150$, $\mu_0 = 0.9$, $N$ runs from 2 to 151 and $\mu$ from 0.9 to $0.9^{2/151} = 0.99861$.
Song and Dhariwal's curriculum

$$ N(k) = \min\bigl(s_0\, 2^{\lfloor k/K' \rfloor}, s_1\bigr) + 1 , \qquad K' = \Bigl\lfloor \frac{K}{\log_2\lfloor s_1/s_0\rfloor + 1} \Bigr\rfloor $$

with $s_0 = 10$, $s_1 = 1280$ takes the eight values $11, 21, 41, \dots, 1281$, each for $K'$ iterations.
Under the original uniform choice of the index $i$, as $N \to \infty$, $\log\sigma$ has density $\sigma^{1/\rho}/\bigl(\rho(\sigma_{\max}^{1/\rho} - \sigma_{\min}^{1/\rho})\bigr)$, increasing in $\sigma$, so high noise levels are drawn most often; the lognormal choice $p(i) \propto \operatorname{erf}\bigl(\frac{\log\sigma_{i+1} - P_{\text{mean}}}{\sqrt 2 P_{\text{std}}}\bigr) - \operatorname{erf}\bigl(\frac{\log\sigma_i - P_{\text{mean}}}{\sqrt 2 P_{\text{std}}}\bigr)$ gives the interval $[\sigma_i, \sigma_{i+1}]$ the mass that a normal $\log\sigma \sim \mathcal N(P_{\text{mean}}, P_{\text{std}}^2)$, truncated to $[\sigma_{\min}, \sigma_{\max}]$, puts there; and the weight $\lambda(\sigma_i) = 1/(\sigma_{i+1} - \sigma_i)$ decreases in $i$.

### Assumptions

1. The noise levels are the Karras levels with $\rho = 7$. Used in steps 3 and 5.
2. $k \in [0, K]$. Used in steps 1 and 2.

### Steps

1. At $k = 0$ the root is $s_0$, so $N = \lceil s_0 - 1\rceil + 1 = s_0$; at $k = K$ it is $s_1 + 1$, so $N = s_1 + 1$. With $\mu(k) = \mu_0^{s_0/N(k)}$, $\mu(0) = \mu_0$ and $\mu(K) = \mu_0^{s_0/(s_1+1)}$.
2. $\lfloor s_1/s_0\rfloor = 128 = 2^7$, so $K' = \lfloor K/8\rfloor$ and $\lfloor k/K'\rfloor$ takes the values 0 to 7 before the cap: $N = 10 \cdot 2^j + 1$ for $j = 0, \dots, 7$.
3. A uniform index $u = (i - 1)/(N - 1)$ on $[0, 1]$ maps to $\sigma^{1/\rho} = a + u(b - a)$ with $a = \sigma_{\min}^{1/\rho}$, $b = \sigma_{\max}^{1/\rho}$. Then $d\sigma^{1/\rho}/d\log\sigma = \sigma^{1/\rho}/\rho$, so the density of $\log\sigma$ is $\sigma^{1/\rho}/(\rho(b - a))$.
4. $\tfrac12\bigl(1 + \operatorname{erf}((y - m)/(\sqrt 2 s))\bigr)$ is the distribution function of $\mathcal N(m, s^2)$ at $y$, so the differences of eq. 10 are interval masses, and dividing by their sum, which is the mass of $[\log\sigma_{\min}, \log\sigma_{\max}]$, truncates the normal to that range.
5. $\sigma_i = (a + u_i (b - a))^\rho$ is convex in $u_i$ for $\rho \ge 1$, so the gaps $\sigma_{i+1} - \sigma_i$ grow with $i$ and their reciprocals shrink.

### Result and check

The two curricula start with few, wide steps, where the consistency loss has small variance and large bias, and end with many narrow ones, where the bias of the one-step score estimate shrinks.
The original schedule also raises the target network's decay $\mu$ as the steps narrow, while Song and Dhariwal set it to 0, as derivation teacher-ema explains.
The lognormal puts its median at $\sigma = e^{-1.1} = 0.33$, against $\sigma = ((\sigma_{\min}^{1/\rho} + \sigma_{\max}^{1/\rho})/2)^\rho = 2.5$ for the uniform index, so most pairs fall at low noise, which Song and Dhariwal argue matters because errors at low noise propagate to every higher level through the target.
The check `schedules.check.py` evaluates `cm_schedule` and `ict_steps` at the ends and along $k$, compares the uniform-index histogram of $\log\sigma$ for $N = 10^5$ with the density of step 3, compares `lognormal_probs` with the truncated normal masses and their sum with 1, and tests the weights of `ict_weight` for monotonicity.
