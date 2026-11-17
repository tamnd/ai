---
id: self-stabilization-ode
derives: [c-ode-conserved]
kind: closed-form
assumes: ["alpha > 0 and beta > 0", "X(t) > 0", "the continuous model X' = X Y, Y' = alpha - beta X^2 / 2"]
uses: []
status: full
check: self-stabilization-ode.check.py
lean: null
sources:
  - {work: damian-2023-self-stabilization-edge-stability, locator: "p. 7"}
  - {work: damian-2023-self-stabilization-edge-stability, locator: "p. 8"}
---

### Statement

Let $\alpha, \beta > 0$ and let $X(t) > 0$, $Y(t)$ solve
$$ X' = XY, \qquad Y' = \alpha - \beta X^2/2. $$
Here $X$ stands for the size of the oscillation along the top eigenvector and $Y$ for the sharpness minus $2/\eta$.
With $h(z) = z - \log z - 1$, the quantity
$$ g(X, Y) = h\Big(\frac{\beta X^2}{2\alpha}\Big) + \frac{Y^2}{\alpha} $$
is constant along every solution, $g \ge 0$, and $g = 0$ only at the fixed point $(X, Y) = (\delta, 0)$ with $\delta = \sqrt{2\alpha/\beta}$.
So every other solution runs around a closed level curve of $g$: the sharpness goes above and below $2/\eta$ in turn and the oscillation grows and shrinks with it.

### Assumptions

1. $\alpha > 0$: progressive sharpening pushes $Y$ up when $X$ is small. $\beta > 0$.
2. $X > 0$, so $\log X$ is defined; the model tracks $|x_t|$ of the discrete map.

### Steps

1. Let $z = \beta X^2/(2\alpha)$. Then $z' = \beta X X'/\alpha = \beta X^2 Y/\alpha = 2zY$.
2. $\frac{d}{dt} h(z) = (1 - 1/z) z' = 2Y(z - 1)$.
3. $\frac{d}{dt}(Y^2/\alpha) = 2YY'/\alpha = 2Y(1 - \beta X^2/(2\alpha)) = 2Y(1 - z)$.
4. The two derivatives cancel, so $g' = 0$.
5. $h(z) \ge 0$ with equality only at $z = 1$, since $h$ is convex with $h(1) = 0$ and $h'(1) = 0$; so $g \ge 0$ with equality only at $z = 1$ and $Y = 0$, which is $X = \delta$, $Y = 0$. There $X' = 0$ and $Y' = \alpha - \beta\delta^2/2 = 0$, so it is the fixed point.

### Result and check

This is Lemma 3 of Damian et al. (2023) (p. 8), the conserved quantity of the Lotka-Volterra form of their two-variable model, obtained from the map $x_{t+1} \approx -(1 + \eta y_t)x_t$, $y_{t+1} \approx y_t + \eta(\alpha - \beta x_t^2/2)$ (p. 7).
The discrete map does not conserve $g$ exactly; over many cycles its orbits drift.
`self-stabilization-ode.check.py` integrates the ODE with fourth-order Runge-Kutta from several starts and checks that $g$ stays constant to high precision, that $Y$ changes sign, that the fixed point is $(\delta, 0)$, and that $g$ is zero there and positive elsewhere on a grid.
