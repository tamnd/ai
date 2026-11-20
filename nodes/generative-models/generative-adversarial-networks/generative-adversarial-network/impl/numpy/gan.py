# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/generative-adversarial-network/ (ai://NEW-generative-adversarial-network).
"""The generative adversarial network of the page in NumPy, with every backward pass written out.

The generator G maps noise z of J dimensions through one tanh layer of H units to a point x of D dimensions,
x = tanh(z U1 + c1) U2 + c2. The discriminator, or critic, maps a point through one tanh layer of K units to one real
number, the logit a(x) = tanh(x W1 + b1) w2 + b2; D(x) = sigmoid(a(x)) is the probability that x is data. Rows are
examples and a layer computes h @ W + b with W of shape (n_in, n_out).

Five losses share the networks. "minimax" is the game of Goodfellow et al. (2014), eq. 1: the discriminator minimizes
the cross-entropy -log D(x) - log(1 - D(G(z))) and the generator minimizes log(1 - D(G(z))). "nonsat" keeps the
discriminator's loss and gives the generator -log D(G(z)), the non-saturating loss of the same paper. "wgan" is the
critic loss a(G(z)) - a(x) of Arjovsky et al. (2017), with the generator minimizing -a(G(z)); the clipping of the
weights is a step of training, clip(). "wgan_gp" adds the two-sided gradient penalty lambda (|grad a(x_hat)| - 1)^2 of
Gulrajani et al. (2017) at x_hat = alpha x + (1 - alpha) G(z). "r1" adds the penalty gamma/2 |grad a(x)|^2 on the data
of Mescheder et al. (2018) to the non-saturating losses. Every loss is averaged over the batch. The penalties are
functions of the gradient of the critic in its input, so their gradients in the critic's weights are second
derivatives; dis_backward computes them by hand.
"""

import numpy as np

LOSSES = ("minimax", "nonsat", "wgan", "wgan_gp", "r1")
GEN, DIS = ("gen_W1", "gen_b1", "gen_W2", "gen_b2"), ("dis_W1", "dis_b1", "dis_w2", "dis_b2")


def config(d_noise=2, d_hidden=5, d_data=3, d_critic=4):
    """The pair of the shape contract: noise of J dimensions, H generator units, D data dimensions, K critic units."""
    assert min(d_noise, d_hidden, d_data, d_critic) >= 1
    return {"d_noise": int(d_noise), "d_hidden": int(d_hidden), "d_data": int(d_data), "d_critic": int(d_critic)}


def param_shapes(cfg):
    J, H, D, K = cfg["d_noise"], cfg["d_hidden"], cfg["d_data"], cfg["d_critic"]
    return {"gen_W1": (J, H), "gen_b1": (H,), "gen_W2": (H, D), "gen_b2": (D,),
            "dis_W1": (D, K), "dis_b1": (K,), "dis_w2": (K,), "dis_b2": (1,)}  # fmt: skip


def param_count(cfg):
    """eq-params: (J + 1) H + (H + 1) D in the generator and (D + 1) K + K + 1 in the discriminator."""
    s = param_shapes(cfg)
    return sum(int(np.prod(s[k])) for k in GEN), sum(int(np.prod(s[k])) for k in DIS)


def flops_forward(cfg, B):
    """eq-flops: the products of one generator pass on B noise vectors and one discriminator pass on the B fakes and
    the B data points, a multiply-add counted as 2 FLOPs."""
    J, H, D, K = cfg["d_noise"], cfg["d_hidden"], cfg["d_data"], cfg["d_critic"]
    return 2 * B * (J * H + H * D) + 2 * (2 * B) * (D * K + K)


def init(rng, cfg, scale=1.0):
    """Weights drawn from N(0, scale^2 / n_in), biases 0."""
    p = {}
    for k, s in param_shapes(cfg).items():
        p[k] = (
            rng.normal(0.0, scale / np.sqrt(s[0]), s) if k in ("gen_W1", "gen_W2", "dis_W1", "dis_w2") else np.zeros(s)
        )
    return p


def sigmoid(a):
    return 0.5 * (1.0 + np.tanh(0.5 * a))  # the logistic function, without overflow for large |a|


def softplus(a):
    return np.logaddexp(0.0, a)  # log(1 + e^a); -log D = softplus(-a) and -log(1 - D) = softplus(a)


def generate(p, z):
    """eq-generator: the points G(z), (B, D), and the hidden layer."""
    t = np.tanh(z @ p["gen_W1"] + p["gen_b1"])
    return t @ p["gen_W2"] + p["gen_b2"], t


def critic(p, x):
    """eq-discriminator: the logits a(x), (B,), and the hidden layer."""
    h = np.tanh(x @ p["dis_W1"] + p["dis_b1"])
    return h @ p["dis_w2"] + p["dis_b2"][0], h


def input_grad(p, h):
    """The gradient of a in its input at the points whose hidden layer is h, (B, D): W1 ((1 - h^2) * w2)."""
    return ((1.0 - h**2) * p["dis_w2"]) @ p["dis_W1"].T


def mix(x, fake, alpha):
    """The points x_hat = alpha x + (1 - alpha) G(z) of the gradient penalty, alpha of shape (B,)."""
    return alpha[:, None] * x + (1.0 - alpha[:, None]) * fake


def _norm(g):
    return np.sqrt(np.sum(g**2, -1))


def dis_loss(p, x, fake, kind="nonsat", alpha=None, lam=10.0, gamma=1.0):
    """The discriminator's or critic's loss with the fakes held fixed, and the pieces of its backward pass: the
    coefficient of each logit, and the points and coefficients of the penalty on the input gradient."""
    assert kind in LOSSES, kind
    B = x.shape[0]
    ar, hr = critic(p, x)
    af, hf = critic(p, fake)
    if kind in ("wgan", "wgan_gp"):
        value = float(np.mean(af) - np.mean(ar))
        dar, daf = np.full(B, -1.0 / B), np.full(B, 1.0 / B)
    else:  # the cross-entropy of eq-dloss, the same for minimax, nonsat and r1
        value = float(np.mean(softplus(-ar)) + np.mean(softplus(af)))
        dar, daf = (sigmoid(ar) - 1.0) / B, sigmoid(af) / B
    pen = None
    if kind == "wgan_gp":
        assert alpha is not None, "wgan_gp needs alpha"
        xh = mix(x, fake, alpha)
        hh = critic(p, xh)[1]
        g = input_grad(p, hh)
        n = _norm(g)
        value += float(lam * np.mean((n - 1.0) ** 2))
        dg = (2.0 * lam / B) * ((n - 1.0) / np.where(n > 0, n, 1.0))[:, None] * g
        pen = (xh, hh, dg)
    elif kind == "r1":
        g = input_grad(p, hr)
        value += float(0.5 * gamma * np.mean(np.sum(g**2, -1)))
        pen = (x, hr, (gamma / B) * g)
    return value, ((x, hr, dar), (fake, hf, daf)), pen


def dis_backward(p, terms, pen):
    """The gradients of sum(c * a(x)) over the terms, plus those of the penalty, in the critic's parameters."""
    grads = {k: np.zeros_like(p[k]) for k in DIS}
    for x, h, c in terms:
        grads["dis_w2"] += h.T @ c
        grads["dis_b2"] += c.sum()
        du = c[:, None] * p["dis_w2"] * (1.0 - h**2)
        grads["dis_W1"] += x.T @ du
        grads["dis_b1"] += du.sum(0)
    if pen is not None:
        # g = s W1^T with s = (1 - h^2) * w2 and h = tanh(x W1 + b1); r = dPenalty/dg
        x, h, r = pen
        s = (1.0 - h**2) * p["dis_w2"]
        grads["dis_W1"] += r.T @ s
        ds = r @ p["dis_W1"]
        grads["dis_w2"] += np.sum(ds * (1.0 - h**2), 0)
        du = ds * p["dis_w2"] * (-2.0 * h) * (1.0 - h**2)
        grads["dis_W1"] += x.T @ du
        grads["dis_b1"] += du.sum(0)
    return grads


def gen_loss(p, z, kind="nonsat"):
    """The generator's loss and the coefficient of each fake's logit: eq-minimax's log(1 - D(G(z))) = -softplus(a),
    the non-saturating -log D(G(z)) = softplus(-a), or the critic's -a(G(z))."""
    assert kind in LOSSES, kind
    B = z.shape[0]
    fake, t = generate(p, z)
    af, hf = critic(p, fake)
    if kind == "minimax":
        value, da = -float(np.mean(softplus(af))), -sigmoid(af) / B
    elif kind in ("nonsat", "r1"):
        value, da = float(np.mean(softplus(-af))), (sigmoid(af) - 1.0) / B
    else:
        value, da = -float(np.mean(af)), np.full(B, -1.0 / B)
    return value, (z, t, hf, da)


def gen_backward(p, cache):
    """The gradients of the generator's loss in the generator's parameters, through the fixed critic."""
    z, t, hf, da = cache
    dx = da[:, None] * input_grad(p, hf)
    grads = {"gen_W2": t.T @ dx, "gen_b2": dx.sum(0)}
    dpre = (dx @ p["gen_W2"].T) * (1.0 - t**2)
    grads["gen_W1"], grads["gen_b1"] = z.T @ dpre, dpre.sum(0)
    return grads


def d_step(p, x, z, kind="nonsat", alpha=None, lam=10.0, gamma=1.0):
    """The discriminator's loss on a batch of data x and of noise z, and its gradients."""
    value, terms, pen = dis_loss(p, x, generate(p, z)[0], kind, alpha, lam, gamma)
    return value, dis_backward(p, terms, pen)


def g_step(p, z, kind="nonsat"):
    """The generator's loss on a batch of noise z, and its gradients."""
    value, cache = gen_loss(p, z, kind)
    return value, gen_backward(p, cache)


def optimal_discriminator(p_data, p_gen):
    """eq-dstar: D*(x) = p_data(x) / (p_data(x) + p_g(x)) from the two densities at the same points, and 1/2 where
    both are 0, where any value is optimal."""
    s = np.asarray(p_data + p_gen, dtype=float)
    return np.divide(p_data, s, out=np.full_like(s, 0.5), where=s > 0)


def value_at(p_data, p_gen, d, dx):
    """V(G, D) of eq-minimax as a Riemann sum over a grid of spacing dx: the integral of
    p_data log D + p_g log(1 - D)."""
    tiny = np.finfo(float).tiny  # D rounds to 0 or 1 where one density is below 1e-16 of the other
    f = p_data * np.log(np.maximum(d, tiny)) + p_gen * np.log(np.maximum(1.0 - d, tiny))
    return float(np.sum(f) * dx)


def jsd(p, q, dx):
    """The Jensen-Shannon divergence of two densities on a grid, in nats."""
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kp = np.where(p > 0, p * np.log(p / m), 0.0)
        kq = np.where(q > 0, q * np.log(q / m), 0.0)
    return float(0.5 * np.sum(kp + kq) * dx)


def dirac_gan(theta, psi, h, steps, gamma=0.0, mode="sim"):
    """The Dirac-GAN of Mescheder et al. (2018): the generator is the point theta, the discriminator the line
    D(x) = psi x, the data the point 0, and the objective L = f(psi theta) + f(0) with f(t) = -log(1 + e^-t), which
    the generator descends and the discriminator ascends, with the penalty gamma/2 psi^2 on the discriminator, the R1
    penalty of this example. "sim" updates both from the same point, "alt" the generator and then the discriminator.
    Returns the iterates, (steps + 1, 2)."""
    out = [(theta, psi)]
    for _ in range(steps):
        if mode == "sim":
            fp = sigmoid(-psi * theta)  # f'(t) = sigmoid(-t)
            theta, psi = theta - h * fp * psi, psi + h * (fp * theta - gamma * psi)
        else:
            theta = theta - h * sigmoid(-psi * theta) * psi
            psi = psi + h * (sigmoid(-psi * theta) * theta - gamma * psi)
        out.append((theta, psi))
    return np.array(out)


def adam(p, grads, state, lr, b1=0.5, b2=0.999, eps=1e-8):
    t = state["t"] = state.get("t", 0) + 1
    for k, g in grads.items():
        m = state.setdefault("m_" + k, np.zeros_like(g))
        v = state.setdefault("v_" + k, np.zeros_like(g))
        m[...] = b1 * m + (1 - b1) * g
        v[...] = b2 * v + (1 - b2) * g * g
        p[k] -= lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)


def rmsprop(p, grads, state, lr, rho=0.9, eps=1e-8):
    for k, g in grads.items():
        v = state.setdefault("v_" + k, np.zeros_like(g))
        v[...] = rho * v + (1 - rho) * g * g
        p[k] -= lr * g / (np.sqrt(v) + eps)


def clip(p, c):
    """The weight clipping of WGAN: every parameter of the critic into [-c, c]."""
    for k in DIS:
        np.clip(p[k], -c, c, out=p[k])


def train(p, cfg, sample_data, rng, steps, kind="nonsat", batch=64, n_dis=1, lr=1e-3, b1=0.5, b2=0.999, lam=10.0,
          gamma=1.0, c=None, opt="adam"):  # fmt: skip
    """Algorithm 1 of Goodfellow et al. with n_dis discriminator steps per generator step; sample_data(rng, n) draws
    n data points. With c set the critic's weights are clipped after each of its steps. Returns the losses of the
    discriminator and of the generator at each generator step."""
    sd, sg, trace = {}, {}, []
    step = adam if opt == "adam" else rmsprop
    kw = {"b1": b1, "b2": b2} if opt == "adam" else {}
    for _ in range(steps):
        for _ in range(n_dis):
            x = sample_data(rng, batch)
            z = rng.standard_normal((batch, cfg["d_noise"]))
            alpha = rng.uniform(0.0, 1.0, batch)
            dv, grads = d_step(p, x, z, kind, alpha, lam, gamma)
            step(p, grads, sd, lr, **kw)
            if c is not None:
                clip(p, c)
        gv, grads = g_step(p, rng.standard_normal((batch, cfg["d_noise"])), kind)
        step(p, grads, sg, lr, **kw)
        trace.append((dv, gv))
    return np.array(trace)


def sample(p, cfg, rng, n):
    """n samples of the trained generator."""
    return generate(p, rng.standard_normal((n, cfg["d_noise"])))[0]
