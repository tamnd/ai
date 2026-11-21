---
id: compression-ratio
derives: [c-compression-ratio]
kind: count
assumes: ["8 bits per colour channel", "H and W divisible by f"]
uses: [vector-quantization@m2]
status: full
check: compression-ratio.check.py
lean: null
sources:
  - {work: rombach-2021-high-resolution-image-synthesis-latent, locator: "p. 4", note: "the latent z in R^{h x w x c} with f = H/h = W/w"}
  - {work: oord-2017-neural-discrete-representation-learning, locator: "p. 5", note: "128 x 128 x 3 images to a 32 x 32 x 1 code with K = 512"}
---

### Statement

An RGB image of $H \times W$ pixels has $3HW$ numbers. A latent with downsampling factor $f$ and $c$ channels has $hwc$ numbers with $h = H/f$ and $w = W/f$, so the image has

$$ \frac{3HW}{(H/f)(W/f)\,c} = \frac{3f^2}{c} $$

times as many numbers as the latent. With 8 bits per colour channel the image costs $24HW$ bits; a discrete code of $K$ entries per latent position costs $\log_2 K$ bits per position, and the ratio in bits is $24f^2/\log_2 K$.
For $f = 8$ and $c = 4$ the first ratio is 48, and for $f = 4$ and $K = 512$ the second is $24 \cdot 16/9 \approx 42.67$.

### Assumptions

1. 8 bits per colour channel. Used in step 3.
2. $f$ divides $H$ and $W$. Used in step 1.

### Steps

1. The latent has $(H/f)(W/f)$ positions, each with $c$ numbers, so $HWc/f^2$ numbers.
2. Dividing $3HW$ by $HWc/f^2$ gives $3f^2/c$, which for $f = 8$ and $c = 4$ is $3 \cdot 64/4 = 48$: a $256 \times 256 \times 3$ image has 196608 numbers and its $32 \times 32 \times 4$ latent 4096.
3. The image costs $3 \cdot 8 \cdot HW$ bits and the code $(HW/f^2)\log_2 K$ bits, a ratio of $24f^2/\log_2 K$. For a $128 \times 128 \times 3$ image with $f = 4$ and $K = 512$ this is $24 \cdot 16/9 = 42.67$, the $\frac{128 \times 128 \times 3 \times 8}{32 \times 32 \times 9}$ of the VQ-VAE paper.
4. For a hierarchy of codes the positions of the levels add: VQ-VAE-2's $64 \times 64$ and $32 \times 32$ maps for a $256 \times 256 \times 3$ image have 5120 positions against 196608 numbers, a factor of 38.4 in numbers and $\frac{256^2 \cdot 24}{5120 \cdot 9} \approx 34.1$ in bits with $K = 512$, the "over 30 times smaller" of Razavi and colleagues.

### Result and check

The compression of a latent grows with the square of $f$ and falls with $c$, so the choice of $f$ and $c$ is the first stage's rate. The one-dimensional autoencoder of the implementations is the case of one spatial axis, a ratio of $f/c$.
The check `compression-ratio.check.py` encodes signals with `fit_autoencoder` and `encode` and compares the sizes with $f/c$, reshapes a $256 \times 256 \times 3$ array into $8 \times 8$ patches and compares the count with 48, and computes the bit ratios of VQ-VAE and VQ-VAE-2.
