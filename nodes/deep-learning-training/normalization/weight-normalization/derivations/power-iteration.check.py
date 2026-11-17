# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weight-normalization/ (ai://NEW-weight-normalization).
# Check of derivations/power-iteration.md: the estimate sigma_k = u_k^T W v_k after k rounds of v <- W^T u / ||.||,
# u <- W v / ||.|| satisfies 0 <= sigma_1^2 - sigma_k^2 <= sigma_1^2 (sigma_2 / sigma_1)^(4k - 2) tan^2(theta_0), and
# converges to the largest singular value of numpy.linalg.svd.
import numpy as np


def power(w, u, k):
    for _ in range(k):
        v = w.T @ u
        v /= np.linalg.norm(v)
        u = w @ v
        u /= np.linalg.norm(u)
    return float(u @ w @ v)


def main():
    rng = np.random.default_rng(11)
    cases = 0
    for rows, cols in [(3, 3), (4, 7), (8, 5), (16, 16)]:
        for _ in range(20):
            q1, _ = np.linalg.qr(rng.normal(size=(rows, rows)))
            q2, _ = np.linalg.qr(rng.normal(size=(cols, cols)))
            r = min(rows, cols)
            s = np.sort(rng.uniform(0.1, 3.0, size=r))[::-1]
            s[0] = s[1] * rng.uniform(1.05, 3.0)
            w = q1[:, :r] @ np.diag(s) @ q2[:, :r].T
            u_svd, s_svd, _ = np.linalg.svd(w)
            assert np.isclose(s_svd[0], s[0])
            u0 = rng.normal(size=rows)
            cos0 = abs(u0 @ u_svd[:, 0]) / np.linalg.norm(u0)
            tan2 = (1 - cos0**2) / cos0**2
            for k in (1, 2, 3, 5, 10, 20):
                est = power(w, u0, k)
                gap = s[0] ** 2 - est**2
                assert -1e-9 * s[0] ** 2 <= gap <= s[0] ** 2 * (s[1] / s[0]) ** (4 * k - 2) * tan2 * (1 + 1e-9) + 1e-12
            assert np.isclose(power(w, u0, 400), s_svd[0], rtol=1e-9)
            cases += 1
    # the worked example: W = diag(3, 1), u0 = (1, 1) / sqrt(2); sigma_1^2 - sigma_k^2 = 8 / (9^(2k - 1) + 1)
    w, u0 = np.diag([3.0, 1.0]), np.array([1.0, 1.0]) / np.sqrt(2)
    for k in (1, 2, 3, 4):
        assert np.isclose(9 - power(w, u0, k) ** 2, 8 / (9 ** (2 * k - 1) + 1), rtol=1e-6, atol=1e-15)
    print(f"ok: the power estimate obeys the bound for k = 1 to 20 and matches the SVD in {cases} matrices")


if __name__ == "__main__":
    main()
