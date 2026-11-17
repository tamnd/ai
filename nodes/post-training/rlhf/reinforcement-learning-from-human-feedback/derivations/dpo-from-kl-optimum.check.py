# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/reinforcement-learning-from-human-feedback/ (ai://6XVY3X).
# Check of derivations/dpo-from-kl-optimum.md: the KL-regularized objective as beta log Z less beta KL(pi || pi_star),
# its maximizer, the reward written through the optimal policy, the cancellation of Z in the Bradley-Terry model, the
# DPO loss and its gradient, the KL-shaped reward as an unbiased estimate of the objective, and the per-token penalty.
import numpy as np


def kl(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))


def objective(pi, ref, r, beta):
    return float(pi @ r) - beta * kl(pi, ref)


def optimum(ref, r, beta):
    w = ref * np.exp(r / beta)
    return w / w.sum(), float(np.log(w.sum()))


def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def main():
    rng = np.random.default_rng(20261115)
    for n, beta in [(3, 1.0), (5, 0.1), (8, 2.5), (20, 0.5)]:
        ref = rng.dirichlet(np.ones(n))
        r = rng.normal(size=n)
        pis, log_z = optimum(ref, r, beta)
        # 1. J(pi) = beta log Z - beta KL(pi || pi_star) for any pi, so pi_star is the unique maximizer.
        for _ in range(20):
            q = rng.dirichlet(np.ones(n) * 0.5)
            q = np.clip(q, 1e-300, None)
            q /= q.sum()
            assert np.isclose(objective(q, ref, r, beta), beta * log_z - beta * kl(q, pis), rtol=1e-9, atol=1e-9)
        assert np.isclose(objective(pis, ref, r, beta), beta * log_z, rtol=1e-12)
        # 2. r = beta log(pi_star / pi_ref) + beta log Z.
        assert np.allclose(beta * np.log(pis / ref) + beta * log_z, r, atol=1e-10)
        # 3. The Bradley-Terry probability through pi_star: Z cancels, and adding a constant to r changes Z only.
        pis2, log_z2 = optimum(ref, r + 4.0, beta)
        assert np.allclose(pis2, pis, rtol=1e-10) and np.isclose(log_z2, log_z + 4.0 / beta, rtol=1e-12)
        i, j = rng.choice(n, size=2, replace=False)
        h = beta * np.log(pis[i] / ref[i]) - beta * np.log(pis[j] / ref[j])
        assert np.isclose(1 / (1 + np.exp(-h)), 1 / (1 + np.exp(-(r[i] - r[j]))), rtol=1e-10)
        # 4. The DPO loss at pi_star is the Bradley-Terry loss of r.
        assert np.isclose(-log_sigmoid(h), -log_sigmoid(r[i] - r[j]), rtol=1e-10)
    # 5. The gradient of the DPO loss of one pair in the logits of a softmax policy is
    #    -beta sigma(r_hat_l - r_hat_w) (grad log pi(y_w) - grad log pi(y_l)).
    n, beta, hh = 6, 0.3, 1e-6
    ref = rng.dirichlet(np.ones(n))
    z = rng.normal(size=n)
    w, lo = 1, 4

    def logpi(z):
        return z - np.logaddexp.reduce(z)

    def dpo(z):
        lp = logpi(z)
        return float(-log_sigmoid(beta * ((lp[w] - np.log(ref[w])) - (lp[lo] - np.log(ref[lo])))))

    num = np.array([(dpo(z + hh * e) - dpo(z - hh * e)) / (2 * hh) for e in np.eye(n)])
    lp = logpi(z)
    rhat = beta * (lp - np.log(ref))
    weight = 1 / (1 + np.exp(-(rhat[lo] - rhat[w])))
    p = np.exp(lp)
    grad_w, grad_l = np.eye(n)[w] - p, np.eye(n)[lo] - p
    assert np.allclose(num, -beta * weight * (grad_w - grad_l), atol=1e-8)
    # 6. E_{y ~ pi}[r(y) - beta log(pi(y) / pi_ref(y))] = E_pi[r] - beta KL(pi || pi_ref), exactly as a sum.
    pi = rng.dirichlet(np.ones(n))
    r = rng.normal(size=n)
    assert np.isclose(float(pi @ (r - beta * np.log(pi / ref))), objective(pi, ref, r, beta), rtol=1e-12)
    # 7. The per-token penalties of an autoregressive policy add up to the sequence log-ratio (the chain rule).
    T, V = 4, 3
    tok = rng.integers(0, V, size=T)
    lpt = np.log(rng.dirichlet(np.ones(V), size=T))
    lrt = np.log(rng.dirichlet(np.ones(V), size=T))
    seq = lpt[np.arange(T), tok].sum() - lrt[np.arange(T), tok].sum()
    assert np.isclose(sum(lpt[t, tok[t]] - lrt[t, tok[t]] for t in range(T)), seq, rtol=1e-12)
    # 8. The worked example of the page.
    ref = np.array([0.5, 0.25, 0.25])
    r = np.array([0.0, np.log(2), np.log(4)])
    pis, log_z = optimum(ref, r, 1.0)
    assert np.allclose(pis, [0.25, 0.25, 0.5], rtol=1e-12) and np.isclose(log_z, np.log(2), rtol=1e-12)
    assert np.isclose(kl(pis, ref), np.log(2) / 4, rtol=1e-12)
    print("ok: the gap beta KL(pi || pi_star), the maximizer, the reward through pi_star, Z cancelling, DPO at the "
          "optimum and its gradient, the KL-shaped reward, the per-token penalty and the worked example")


if __name__ == "__main__":
    main()
