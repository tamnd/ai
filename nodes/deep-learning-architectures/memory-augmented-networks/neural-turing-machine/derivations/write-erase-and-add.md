---
id: write-erase-and-add
derives: [c-write-commutes, c-reset-needs-both, c-blurry-erase]
kind: identity
assumes: ["every write weighting is a weighting and every erase vector has entries in [0, 1]", "all the erases of a step act before all the adds"]
uses: [addressing-keeps-a-weighting]
status: full
check: write-erase-and-add.check.py
lean: null
sources:
  - {work: graves-2014-neural-turing-machines, locator: "Section 3.2"}
---

### Statement

1. With several write heads, the memory after a step does not depend on the order of the heads: the erases multiply and the adds sum.
2. The row $i$ of the memory is set to zero by the erase only if some head has $w_t(i)\, e_t(j) = 1$ for every column $j$, which for one head means $w_t(i) = 1$ and $e_t = \mathbf 1$. Then the write replaces the row with the add vector.
3. A weighting spread over several rows erases each of them only in part: with $w_t(i) = p$ and $e_t = \mathbf 1$, the row keeps the fraction $1 - p$ of its old value.

### Assumptions

1. Each write weighting $w^h_t$ has entries in $[0, 1]$ (addressing-keeps-a-weighting) and each erase vector $e^h_t$ has entries in $[0, 1]$, as the sigmoid gives them. Used in steps 2 and 3.
2. In a step the erases of all the heads act on $M_{t-1}$ and then the adds of all the heads act on the result, as Graves et al. describe for multiple write heads. Used in step 1.

### Steps

1. By eq-erase, head $h$ multiplies the entry $(i, j)$ by $1 - w^h_t(i)\, e^h_t(j)$. With $H$ heads the erased entry is $M_{t-1}(i, j) \prod_h (1 - w^h_t(i)\, e^h_t(j))$, and multiplication of real numbers commutes. By eq-add, head $h$ adds $w^h_t(i)\, a^h_t(j)$, and addition commutes. So any order of the heads, within the erases and within the adds, gives
   $M_t(i, j) = M_{t-1}(i, j) \prod_h \big(1 - w^h_t(i) e^h_t(j)\big) + \sum_h w^h_t(i)\, a^h_t(j)$.
2. By assumption 1 each factor $1 - w^h_t(i) e^h_t(j)$ lies in $[0, 1]$, so the product is 0 only if one factor is 0, that is $w^h_t(i)\, e^h_t(j) = 1$ for some $h$, which needs $w^h_t(i) = 1$ and $e^h_t(j) = 1$. For the whole row this must hold for every $j$. For one head, $w_t(i) = 1$ forces $w_t(k) = 0$ elsewhere, so the other rows are not erased and get no add, and row $i$ becomes $a_t$.
3. With one head, $w_t(i) = p$ and $e_t = \mathbf 1$, the factor is $1 - p$ in every column, so the erased row is $(1 - p) M_{t-1}(i)$, and then the add $p\, a_t$ is added.

### Result and check

$$ M_t(i) = M_{t-1}(i) \odot \prod_h \big(\mathbf 1 - w^h_t(i)\, e^h_t\big) + \sum_h w^h_t(i)\, a^h_t. $$

The check `write-erase-and-add.check.py` compares the NumPy write with the closed form above for random weightings, erase and add vectors and 1 to 4 heads, permutes the heads, and checks the full reset and the partial erase.
