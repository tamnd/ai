---
id: group-correlation-equivariance
derives: [c-planar-translation, c-planar-rotation, c-lift-equivariant, c-gconv-equivariant, c-stacking]
kind: identity
assumes: ["G is a discrete group acting on the plane Z^2 by g x = R^r x + t, with R the rotation by 90 degrees about the origin", "the sums over Z^2 and over G converge, for example because f has finite support"]
uses: []
status: full
check: group-correlation-equivariance.check.py
lean: null
sources:
  - {work: cohen-2016-group-equivariant-convolutional-networks, locator: "Sections 5 and 6.1, eq. 7 to 12"}
---

### Statement

Write $[L_u f](x) = f(u^{-1} x)$ for the action of a transformation $u$ on a function on the plane, and $[L_u f](g) = f(u^{-1} g)$ on a function on the group.

1. The planar correlation $[f \star \psi](x) = \sum_{y} \sum_k f_k(y)\, \psi_k(y - x)$ commutes with translations, $[L_t f] \star \psi = L_t [f \star \psi]$, but not with rotations: $[L_r f] \star \psi = L_r [f \star L_{r^{-1}} \psi]$ (eq-planar-rotation).
2. The lifting correlation $[f \star \psi](g) = \sum_y \sum_k f_k(y)\, \psi_k(g^{-1} y)$ (eq-lift) and the group correlation $[f \star \psi](g) = \sum_{h \in G} \sum_k f_k(h)\, \psi_k(g^{-1} h)$ (eq-gconv) are both equivariant: $[L_u f] \star \psi = L_u [f \star \psi]$ for every $u \in G$.
3. On p4 with $g = (t, r)$, the lifting correlation at rotation $r$ is the planar correlation with the filter turned $r$ times, $[f \star \psi](t, r) = [f \star L_r \psi](t)$, and a rotation $u$ by 90 degrees acts on a p4 map by turning every plane and moving plane $s$ to $s + 1$ (eq-p4-action).

### Assumptions

1. Every $g \in G$ maps $x$ to $R^r x + t$; on p4, $r \in \{0, 1, 2, 3\}$ and $t \in \mathbb{Z}^2$, so $g^{-1} x = R^{-r}(x - t)$. Used in steps 2 and 4.
2. The sums are finite: images and filters have finite support. Used in all steps, to reorder the sums.

### Steps

1. Translation. With $[L_t f](y) = f(y - t)$, substitute $y = y' + t$: $\sum_y f(y - t)\, \psi(y - x) = \sum_{y'} f(y')\, \psi(y' - (x - t)) = [f \star \psi](x - t) = [L_t [f \star \psi]](x)$.
2. Rotation of the plane. With $[L_r f](y) = f(R^{-r} y)$, substitute $y = R^r y'$, a bijection of $\mathbb{Z}^2$: $\sum_y f(R^{-r} y)\, \psi(y - x) = \sum_{y'} f(y')\, \psi(R^r y' - x) = \sum_{y'} f(y')\, \psi(R^r (y' - R^{-r} x))$. The last sum is the correlation of $f$ with $L_{r^{-1}} \psi = \psi(R^r \cdot)$, read at $R^{-r} x$, which is $L_r [f \star L_{r^{-1}} \psi](x)$. Unless $\psi = L_{r^{-1}} \psi$, a rotated image does not give a rotated feature map.
3. Group correlation. Substitute $h = u h'$, a bijection of $G$: $\sum_h f(u^{-1} h)\, \psi(g^{-1} h) = \sum_{h'} f(h')\, \psi(g^{-1} u h') = \sum_{h'} f(h')\, \psi((u^{-1} g)^{-1} h') = [f \star \psi](u^{-1} g) = [L_u [f \star \psi]](g)$. The same substitution, $y = u y'$ in $\mathbb{Z}^2$, gives the lifting case. Neither step uses anything about $\psi$, so the two layers are equivariant for every filter, and so is any stack of them.
4. p4 coordinates. For $g = (t, r)$, $g^{-1} y = R^{-r}(y - t)$, so $[f \star \psi](t, r) = \sum_y f(y)\, \psi(R^{-r}(y - t)) = \sum_z f(t + z)\, [L_r \psi](z)$ with $z = y - t$: the planar correlation with $L_r \psi$, read at $t$. For the rotation $u = (0, 1)$, $u^{-1}(t, r) = (R^{-1} t, r - 1)$, so $[L_u F](t, r) = F(R^{-1} t, r - 1)$: plane $r$ of the new map is plane $r - 1$ of the old one, turned once.

### Result and check

$$ [L_r f] \star \psi = L_r\,[f \star L_{r^{-1}} \psi], \qquad [L_u f] \star \psi = L_u\,[f \star \psi] \quad (u \in G), \qquad [L_u F](t, r) = F(R^{-1} t,\, r - 1). $$

The check `group-correlation-equivariance.check.py` runs the NumPy reference on random images and filters, with zero padding and with wrapping, on grids of odd and even side. It confirms the planar rotation identity and that the planar correlation is not rotation equivariant, the equivariance of the lifting and group correlations for all four rotations and for cyclic translations, and that the filter-bank computation equals the sum over the group written out directly.
