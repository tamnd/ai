# The arithmetic of text-to-image generation

This derivation backs the mathematical claims of the text-to-image generation page. The check file
`text-to-image-arithmetic.check.py` tests every number below.

## Setting

A text-to-image model samples an image x given a caption y. The systems on the page split this into a text encoder,
a generator, and often a second model that scores or upsamples the result. DALL-E and Parti turn an image into a grid
of discrete tokens and train a Transformer to predict them one at a time after the caption. GLIDE, unCLIP and Imagen
use diffusion models that denoise pixels, with the caption as conditioning. Almost all of them sample with
classifier-free guidance, and DALL-E and Parti also draw many images and keep the ones a contrastive model scores
highest.

## 1. What the image tokenizer buys

A 256 × 256 RGB image has 256 · 256 · 3 = 196,608 values. DALL-E's discrete VAE replaces it with a 32 × 32 grid of
tokens, 1,024 in all, so the Transformer's context for the image shrinks by 196,608 / 1,024 = 192, the factor DALL-E
reports. Each token takes one of 8,192 values, which is 13 bits, so the grid holds at most 1,024 · 13 = 13,312 bits,
against 196,608 · 8 = 1,572,864 bits for the 8-bit image: the tokenizer must throw away fine detail, and DALL-E reports
that it does. Up to 256 caption tokens come first, so a training sequence is at most 256 + 1,024 = 1,280 tokens.
Parti keeps 1,024 image tokens per 256 × 256 image, a 32 × 32 grid, so each token covers an 8 × 8 patch, and caps the
caption at 128 tokens, so its decoder always writes 1,024 tokens.

DALL-E weights the two cross-entropies by 1/8 for text and 7/8 for the image. The weights sum to 1, so the loss stays
on the scale of a per-token average while the image term counts seven times as much as the text term.

## 2. Classifier-free guidance is one formula in three notations

Parti writes guided sampling as I = G(z) + λ (G(z, c) − G(z)). Expanding gives λ G(z, c) + (1 − λ) G(z), the form
Imagen writes for noise predictions, ε̃ = w ε(z, c) + (1 − w) ε(z), with w = λ. In both, the weight is 1 for the plain
conditional model and 0 for the unconditional one.

Imagen applies guidance to noise predictions but thresholds the x-prediction x̂ = (z − σ ε) / α. That map is affine
in ε, and the guidance weights w and 1 − w sum to 1, so

x̃ = (z − σ (w ε_c + (1 − w) ε_u)) / α = w x̂_c + (1 − w) x̂_u.

Guidance mixes x-predictions with the same weights as it mixes noise predictions.

For an autoregressive model the same formula acts on logits. With l_u + λ (l_c − l_u), the softmax is proportional to
p_c^λ p_u^(1 − λ) = p(x | c) · (p(x | c) / p(x))^(λ − 1). By Bayes' rule p(x | c) / p(x) = p(c | x) / p(c), so guidance
reweights each candidate by p(c | x)^(λ − 1): an implicit classifier of how well x fits the caption, raised to the
power λ − 1. For λ > 1 this concentrates mass on images that fit the caption and lowers the entropy of the samples.

## 3. Why large guidance leaves the training range, and what thresholding does

Training images are scaled to [−1, 1], so a trained model's x-predictions lie roughly there. If both predictions are
in [−1, 1], the guided prediction satisfies |w x̂_c + (1 − w) x̂_u| ≤ w + (w − 1) = 2w − 1 for w ≥ 1, and the bound is
reached when the two predictions disagree in sign. At a pixel where x̂_c = 0.9 and x̂_u = 0, w = 8 gives 7.2. Imagen
reports this train-test mismatch for large weights.

Static thresholding clips every pixel to [−1, 1]. Every pixel whose guided value is above 1 becomes exactly 1, so the
differences between those pixels are lost. Dynamic thresholding computes s, the p-th percentile of |x̃| over the image,
and when s > 1 clips to [−s, s] and divides by s. Pixels below the percentile are all divided by the same s, so their
ratios are kept; only the top 100 − p percent are pinned at ±1, 0.5% at Imagen's p = 99.5.

In the numpy file the "images" are Gaussian with exact denoisers. At t = 0.5, w = 4 and w = 8, static thresholding pins
54% and 76% of the 4,096 pixels at ±1, and dynamic thresholding pins 0.5% at both weights.

## 4. A pooled caption cannot bind attributes

Suppose the text encoder returns the mean of its word embeddings. The mean of "red at 0, blue at 1" and the mean of
"blue at 0, red at 1" are the same vector, because a sum does not depend on the order of its terms. A generator that
sees only this vector gives both captions the same distribution q over images. Its accuracy on the first caption is
q(red, blue) and on the second q(blue, red), and the two add to at most 1. Averaged over the pair, it is right at most
half the time.

The same argument gives the ceiling for the token world of the PyTorch file: 4 cells, 6 colours, and a caption naming
every cell. The mean embedding fixes the multiset of colours, since every position word appears once in every caption.
Every arrangement of that multiset is equally likely to be the target. If the multiset has m distinct arrangements,
any model is right with probability at most 1 / m. Averaging 1 / m over all 6⁴ = 1,296 images gives 126 / 1,296 = 7/72
≈ 0.097. The pooled model reaches 0.091, and the model that reads the caption as a sequence reaches 0.998.

## 5. Best of N

If each sample matches the caption with probability p, and a perfect scorer picks one that matches whenever there is
one, best-of-N sampling is right with probability 1 − (1 − p)^N. Doubling N from n to 2n adds q^n − q^(2n) = q^n (1 − q^n)
with q = 1 − p. That gain is largest when q^n = 1/2, at n = ln 2 / (−ln q), about 6.6 for p = 0.1, and shrinks after
that. With p = 0.1, N = 32 is right 96.6% of the time and N = 512 adds less than 3.5 points. DALL-E reranks 512 samples
with CLIP and Parti 16 with CoCa, 32 times fewer.

In the PyTorch file the model sampled without guidance is right 45.9% of the time on noisy captions, which predicts
1 − 0.541⁴ = 0.914 for best-of-4; reranking 4 samples with an exact scorer gives 0.894.

## 6. unCLIP's autoregressive prior

unCLIP's AR prior keeps 319 of the 1,024 principal components of the CLIP image embedding and quantizes each into 1,024
buckets, so the prior predicts 319 tokens instead of 1,024 values: 1,024 / 319 = 3.21, the threefold reduction the
paper reports.

## 7. Training budgets

Imagen trains every model in its cascade with batch size 2,048 for 2.5M steps, 2,048 · 2.5 · 10⁶ = 5.12 · 10⁹
image-caption examples per model. Against about 460M + 400M = 860M pairs, that is 5.12 · 10⁹ / 8.6 · 10⁸ ≈ 6.0 passes,
if every pair were sampled equally. The three diffusion models hold 2B + 600M + 400M = 3.0B parameters, and the frozen
T5-XXL encoder in front of them has 4.6B.

DALL-E's transformer trains with batch size 1,024 for 430,000 updates, 440.32M examples, about 1.76 passes over 250M
pairs.

## 8. PowerSGD compression

DALL-E compresses gradients with PowerSGD at rank r and gives the compression rate 1 − 5r / (8 d_model). For its three
model sizes, (r, d_model) = (512, 1,920), (640, 2,688) and (896, 3,968) give 83.3%, 85.1% and 85.9%, the ≈ 83%, ≈ 85% and
≈ 86% of its Table 1.
