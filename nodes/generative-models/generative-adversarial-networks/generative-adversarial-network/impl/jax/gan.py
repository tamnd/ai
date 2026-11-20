# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/generative-adversarial-network/ (ai://NEW-generative-adversarial-network).
"""The generative adversarial network of the page in JAX: pure functions of a parameter dictionary, the generator of
eq-generator, the critic of eq-discriminator and the five losses of the NumPy reference. The gradient of the critic in
its input is jax.grad of the critic, vmapped over the batch, and the penalties of "wgan_gp" and "r1" differentiate
through it. dis_loss takes the fakes as an argument, so jax.grad of it in the parameters is the discriminator's step
with the fakes held fixed; gen_loss differentiates through the critic to the generator.
"""

import math

import jax
import jax.numpy as jnp

LOSSES = ("minimax", "nonsat", "wgan", "wgan_gp", "r1")
GEN, DIS = ("gen_W1", "gen_b1", "gen_W2", "gen_b2"), ("dis_W1", "dis_b1", "dis_w2", "dis_b2")


def config(d_noise=2, d_hidden=5, d_data=3, d_critic=4):
    if min(d_noise, d_hidden, d_data, d_critic) < 1:
        raise ValueError("every size must be positive")
    return {"d_noise": int(d_noise), "d_hidden": int(d_hidden), "d_data": int(d_data), "d_critic": int(d_critic)}


def param_shapes(cfg):
    J, H, D, K = cfg["d_noise"], cfg["d_hidden"], cfg["d_data"], cfg["d_critic"]
    return {"gen_W1": (J, H), "gen_b1": (H,), "gen_W2": (H, D), "gen_b2": (D,),
            "dis_W1": (D, K), "dis_b1": (K,), "dis_w2": (K,), "dis_b2": (1,)}  # fmt: skip


def param_count(cfg):
    s = param_shapes(cfg)
    return sum(math.prod(s[k]) for k in GEN), sum(math.prod(s[k]) for k in DIS)


def init(key, cfg):
    p = {}
    for k, s in param_shapes(cfg).items():
        key, sub = jax.random.split(key)
        weight = k in ("gen_W1", "gen_W2", "dis_W1", "dis_w2")
        p[k] = jax.random.normal(sub, s) / math.sqrt(s[0]) if weight else jnp.zeros(s)
    return p


def generate(p, z):
    """eq-generator: G(z) = tanh(z U1 + c1) U2 + c2."""
    return jnp.tanh(z @ p["gen_W1"] + p["gen_b1"]) @ p["gen_W2"] + p["gen_b2"]


def critic(p, x):
    """eq-discriminator: the logit a(x) = tanh(x W1 + b1) w2 + b2 of each row."""
    return jnp.tanh(x @ p["dis_W1"] + p["dis_b1"]) @ p["dis_w2"] + p["dis_b2"][0]


def input_grad(p, x):
    """The gradient of a in its input at each row of x."""
    return jax.vmap(jax.grad(lambda v: critic(p, v[None])[0]))(x)


def dis_loss(p, x, fake, kind="nonsat", alpha=None, lam=10.0, gamma=1.0):
    """The discriminator's or critic's loss on data x and fixed fakes."""
    ar, af = critic(p, x), critic(p, fake)
    if kind in ("wgan", "wgan_gp"):
        loss = jnp.mean(af) - jnp.mean(ar)
    else:
        loss = jnp.mean(jax.nn.softplus(-ar)) + jnp.mean(jax.nn.softplus(af))
    if kind == "wgan_gp":
        assert alpha is not None, "wgan_gp needs alpha"
        xh = alpha[:, None] * x + (1 - alpha[:, None]) * fake
        loss = loss + lam * jnp.mean((jnp.linalg.norm(input_grad(p, xh), axis=-1) - 1) ** 2)
    elif kind == "r1":
        loss = loss + 0.5 * gamma * jnp.mean(jnp.sum(input_grad(p, x) ** 2, -1))
    return loss


def gen_loss(p, z, kind="nonsat"):
    """The generator's loss: log(1 - D(G(z))), -log D(G(z)) or -a(G(z))."""
    af = critic(p, generate(p, z))
    if kind == "minimax":
        return -jnp.mean(jax.nn.softplus(af))
    if kind in ("nonsat", "r1"):
        return jnp.mean(jax.nn.softplus(-af))
    return -jnp.mean(af)


def d_grads(p, x, z, kind="nonsat", alpha=None, lam=10.0, gamma=1.0):
    """The discriminator's loss and its gradients in the discriminator's parameters."""
    fake = generate(p, z)
    dis = {k: p[k] for k in DIS}
    return jax.value_and_grad(lambda d: dis_loss(p | d, x, fake, kind, alpha, lam, gamma))(dis)


def g_grads(p, z, kind="nonsat"):
    """The generator's loss and its gradients in the generator's parameters."""
    gen = {k: p[k] for k in GEN}
    return jax.value_and_grad(lambda g: gen_loss(p | g, z, kind))(gen)


def dirac_gan(theta, psi, h, steps, gamma=0.0):
    """Simultaneous gradient steps on the Dirac-GAN of Mescheder et al. (2018), L = f(psi theta) + f(0) with
    f(t) = -log(1 + e^-t) and the penalty gamma/2 psi^2, by jax.grad of L; returns the iterates."""

    def L(th, ps):
        return -jax.nn.softplus(-ps * th) - jnp.log(2.0)

    def step(c, _):
        th, ps = c
        gt, gp = jax.grad(L, argnums=(0, 1))(th, ps)
        c = (th - h * gt, ps + h * (gp - gamma * ps))
        return c, jnp.stack(c)

    _, it = jax.lax.scan(step, (jnp.asarray(theta), jnp.asarray(psi)), None, length=steps)
    return jnp.concatenate([jnp.array([[theta, psi]]), it])
