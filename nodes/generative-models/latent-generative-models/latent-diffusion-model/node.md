---
tag: D5Q9E6
kind: architecture
slug: latent-diffusion-model
name: Latent diffusion model
aliases: [LDM, latent diffusion, Stable Diffusion, two-stage diffusion, diffusion in latent space, latent score-based generative model, LSGM, VQ-VAE, vector quantised variational autoencoder, VQGAN, diffusion transformer, DiT]
area: generative-models/latent-generative-models
level: core
summary: >-
  A diffusion model that runs in the compact latent space of a pretrained autoencoder instead of on pixels, so that generation is cheap and the decoder restores the fine detail.
uses: [vector-quantization@m2, k-means-clustering@m2, kullback-leibler-divergence@m2, cross-entropy@m2, differential-entropy@m2, normal-distribution@m2, multivariate-normal-distribution@m2, mixture-model@m2, variance@m2, standard-deviation@m2, conditional-expectation@m2, matrix-multiplication@m2, rate-distortion-theory@m2, stochastic-gradient-descent@m2, principal-component-analysis@al, evidence-lower-bound@al, gaussian-mixture-model@al, backpropagation@al, ancestral-sampling@al]
related_to: [diffusion-model, autoencoder, variational-autoencoder, generative-adversarial-network, autoregressive-model, discrete-diffusion-model, flow-matching, transformer, vision-transformer, multi-head-attention, layer-normalization, convolutional-neural-network, residual-connection, normalizing-flow, energy-based-model, adam]
history:
  year: 2021
  people: [Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu, Ali Razavi, Patrick Esser, Robin Rombach, Björn Ommer, Arash Vahdat, Karsten Kreis, Jan Kautz, Andreas Blattmann, Dominik Lorenz, William Peebles, Saining Xie]
  note: "Two-stage generation began with VQ-VAE (2017), which learned a discrete code of images and fitted an autoregressive prior to it, and VQ-VAE-2 (2019) and VQGAN (2020), which made the code hierarchical or perceptual. In 2021 Vahdat, Kreis and Kautz trained a score-based model jointly with a VAE in its latent space (LSGM), and Rombach, Blattmann, Lorenz, Esser and Ommer trained diffusion models in the fixed latent space of a mildly compressing perceptual autoencoder (LDM), with cross-attention for conditioning; the text-to-image release of 2022 became Stable Diffusion. Peebles and Xie (2022) replaced the U-Net with a transformer on latent patches (DiT), the backbone of later text-to-image and video models."
prereqs: [diffusion-model, autoencoder, variational-autoencoder, multi-head-attention, kullback-leibler-divergence@m2, vector-quantization@m2]
sources:
  - {paper: "Rombach, Blattmann, Lorenz, Esser and Ommer (2021)", title: "High-Resolution Image Synthesis with Latent Diffusion Models", locator: "arXiv 2112.10752v2, pp. 1 to 7, 20, 29", note: "the two-stage method, the perceptual autoencoder and its KL and VQ regularization, the latent objective, cross-attention conditioning, the downsampling study, the results, the signal-to-noise ratio and the latent rescaling"}
  - {paper: "van den Oord, Vinyals and Kavukcuoglu (2017)", title: "Neural Discrete Representation Learning", locator: "NIPS 2017; arXiv 1711.00937v2, pp. 3 to 5, 11", note: "the VQ-VAE bottleneck, the straight-through estimator, the three-term loss, the compression in bits and the EMA codebook"}
  - {paper: "Esser, Rombach and Ommer (2020)", title: "Taming Transformers for High-Resolution Image Synthesis", locator: "arXiv 2012.09841v3, pp. 4 to 5", note: "VQGAN: the perceptual and adversarial first stage with an adaptive weight, and the limit on compression"}
  - {paper: "Razavi, van den Oord and Vinyals (2019)", title: "Generating Diverse High-Fidelity Images with VQ-VAE-2", locator: "arXiv 1906.00446v1, pp. 2, 4", note: "the hierarchical code, 30 times smaller than the image, and a prior 30 times faster than in pixel space"}
  - {paper: "Vahdat, Kreis and Kautz (2021)", title: "Score-based Generative Modeling in Latent Space", locator: "NeurIPS 2021; arXiv 2106.05931v3, pp. 1, 3 to 4", note: "LSGM: joint training of the VAE and the score-based prior, the bound split into reconstruction, encoder entropy and cross entropy, and the results"}
  - {paper: "Ho and Salimans (2022)", title: "Classifier-Free Diffusion Guidance", locator: "arXiv 2207.12598v1, pp. 4 to 5", note: "joint training with a dropped condition and the guided noise prediction"}
  - {paper: "Peebles and Xie (2022)", title: "Scalable Diffusion Models with Transformers", locator: "arXiv 2212.09748v2, pp. 1, 4 to 6", note: "DiT: patchify, adaLN-Zero, the conditioning study, and the scaling of FID with Gflops"}
family: diffusion-models
inputs: "for training, a batch of images, or of any signals, with optional conditions such as class labels or text; the first stage maps each image of shape (H, W, 3) to a latent of shape (H/f, W/f, c); for sampling, a condition, a guidance scale and a number of denoising steps"
outputs: "for training, the noise-prediction loss on the latents; for the denoiser, the predicted noise with the shape of the latent; for sampling, a latent of shape (H/f, W/f, c) and the image the decoder makes of it"
components: [autoencoder, variational-autoencoder, diffusion-model, multi-head-attention, transformer, convolutional-neural-network, residual-connection, layer-normalization]
inductive_biases: ["most bits of an image are imperceptible detail that a deterministic decoder can restore, so the generative model only has to model a much smaller code of the perceptually relevant content", "the latent keeps the two-dimensional layout of the image, so a convolutional or patch-based denoiser keeps the locality and translation equivariance of image models", "a fixed first stage is shared by every diffusion model trained on top of it, so the cost of learning the compression is paid once", "conditioning enters through attention from the denoiser's features to a set of condition tokens, so any condition that an encoder can turn into tokens can steer the model"]
params: "the parameters of the autoencoder, trained once, plus those of the denoiser; for the denoiser of this page with N latent positions of c channels, H hidden units, a time embedding of width E, M context tokens of width d and C classes, NcH + EH + H + (C + 1)Md + Hd + d^2 + dH + HNc + Nc"
flops_per_example: "one decoder pass plus one denoiser call per sampling step, two with classifier-free guidance; for the denoiser of this page, 2(NcH + EH + Hd + Md^2 + MdH + Md + MH + HNc) FLOPs per latent, a multiply-add counted as 2"
---

A latent diffusion model generates images, or audio or video, in two stages.
First an [[autoencoder]] is trained to compress an image into a latent code that is several times smaller but from which its decoder can restore an image that looks the same.
Then a [[diffusion-model|diffusion model]] is trained on the latent codes instead of on the pixels, and to generate, its sampler produces a latent code that one pass of the decoder turns into an image.
Because the latent has tens of times fewer numbers than the image, each of the many denoising steps is cheap, and the decoder takes care of the fine texture that a pixel-space model would spend most of its capacity on.
Rombach and colleagues' latent diffusion models of 2021, with [[multi-head-attention|cross-attention]] to text, became Stable Diffusion, and the same recipe with a [[transformer]] in place of the U-Net, DiT, underlies most later text-to-image and video generators.

## In plain words

Imagine you want to teach someone to paint, but the canvas is huge and every brushstroke is slow.
You could instead teach them to draw a small sketch with notes in each square, "sky, light blue, a few clouds" or "brick wall, red, rough", and hire a skilled assistant who turns any such sketch into a full painting.
The student now learns something much smaller and quicker: what goes where.
The assistant knows how bricks and clouds look in detail and adds it every time.
A latent diffusion model does exactly this.
The autoencoder is the assistant: its encoder turns a picture into the small annotated sketch, the latent, and its decoder turns a sketch back into a picture.
The diffusion model is the student: it learns to produce new sketches, starting from random scribbles and cleaning them up step by step.
The assistant is trained first and then frozen, and the same assistant can serve many students, one for faces, one for landscapes, one that follows text instructions.
The sketch must not be too small, or the assistant has to invent too much and the pictures lose detail; and it must not be too large, or the student is back to doing all the work.

## Overview

The first stage is an encoder $\mathcal{E}$ and a decoder $\mathcal{D}$. For an image $x \in \mathbb{R}^{H \times W \times 3}$,

$$ z = \mathcal{E}(x) \in \mathbb{R}^{h \times w \times c} , \qquad \tilde x = \mathcal{D}(z) , \qquad f = H/h = W/w = 2^m , $$ {#eq-autoencoder}

where $f$ is the downsampling factor and $c$ the number of latent channels.
The latent has $hwc = HWc/f^2$ numbers against $3HW$ for the image, so it is $3f^2/c$ times smaller; with $f = 8$ and $c = 4$, the setting of Stable Diffusion and DiT, a $256 \times 256 \times 3$ image becomes a $32 \times 32 \times 4$ latent, 48 times fewer numbers.
Rombach and colleagues train the autoencoder with a perceptual loss and a patch-based adversarial objective, so that the reconstructions stay on the image manifold by enforcing local realism and avoid the blur of pixel-space L2 or L1 losses; the encoder maps $x \in \mathbb{R}^{H \times W \times 3}$ to $z \in \mathbb{R}^{h \times w \times c}$ and downsamples by $f = H/h = W/w = 2^m$. {#c-autoencoder}

### Why compress first

A likelihood-based model of pixels spends much of its capacity on detail that nobody can see.
Rombach and colleagues observe that learning divides roughly into two stages: a perceptual compression stage that removes high-frequency details but learns little semantic variation, and a semantic compression stage in which the generative model learns the conceptual composition of the data; they therefore train an autoencoder that gives a lower-dimensional, perceptually equivalent space first, and train diffusion models in that space second. {#c-perceptual-semantic}
In the language of [[m2:rate-distortion-theory|rate-distortion theory]], the autoencoder spends a small rate to remove a large amount of distortion that the eye would notice, and the diffusion model only has to model the remaining, semantically meaningful rate.
The method is stated in the abstract of the paper: to train diffusion models on limited computational resources while keeping their quality and flexibility, they are applied in the latent space of powerful pretrained autoencoders, which reaches a near-optimal point between complexity reduction and detail preservation, and cross-attention layers turn them into generators for general conditioning inputs such as text or bounding boxes. {#c-ldm-def}

### Keeping the latent tame

An unconstrained encoder can put its latents at any scale, and a diffusion model needs latents of a known scale, because its noise schedule assumes one.
Rombach and colleagues use one of two light regularizers. The KL-regularized autoencoder imposes a slight KL penalty towards a standard normal on the learned latent, as in a [[variational-autoencoder]], while the VQ-regularized one uses a vector quantization layer within the decoder, which makes it a VQGAN with the quantization absorbed by the decoder. {#c-regularization}
For a KL-regularized encoder with outputs $\mu$ and $\log\sigma^2$, the penalty per latent is

$$ \mathrm{KL}\big(\mathcal{N}(\mu, \sigma^2 I) \,\|\, \mathcal{N}(0, I)\big) = \tfrac12 \sum_i \big(\mu_i^2 + \sigma_i^2 - 1 - \log \sigma_i^2\big) , $$

a [[m2:kullback-leibler-divergence|KL divergence]] between [[m2:normal-distribution|normal distributions]].
The weight on it is tiny: the KL term is weighted by a factor of about $10^{-6}$, or the VQ variant uses a high codebook dimensionality; for a KL-regularized latent space the latent is rescaled to unit standard deviation by an estimate $\hat\sigma^2 = \frac{1}{bchw}\sum_{b,c,h,w}(z^{b,c,h,w} - \hat\mu)^2$ computed from the first batch of data, and for a VQ-regularized space $z$ is taken before the quantization layer, which is treated as the first layer of $\mathcal{D}$. {#c-scale-estimate}
So the diffusion model sees

$$ z \leftarrow \mathcal{E}(x) / \hat\sigma , \qquad \hat\sigma^2 = \frac{1}{bchw} \sum_{b,c,h,w} \big(z^{b,c,h,w} - \hat\mu\big)^2 , $$ {#eq-scale}

one number $\hat\sigma$ for the whole latent, and the sampler multiplies by $\hat\sigma$ again before decoding.

The scale matters because the forward process mixes signal and noise in fixed proportions.
With $q(z_t \mid z_0) = \mathcal{N}(\sqrt{\bar\alpha_t} z_0, (1 - \bar\alpha_t) I)$ and latents of [[m2:variance|variance]] $v$, the signal-to-noise ratio of step $t$ is

$$ \mathrm{SNR}(t) = \frac{\bar\alpha_t\, v}{1 - \bar\alpha_t} , $$ {#eq-snr}

so a latent of variance $v$ sees a schedule that is $v$ times cleaner at every step.
Rombach and colleagues found that the signal-to-noise ratio induced by the variance of the latent space, $\mathrm{Var}(z)/\sigma_t^2$, significantly affects the results; for the KL-regularized latent this ratio is very high, so the model allocates a lot of semantic detail early in the reverse process, rescaling by the component-wise standard deviation decreases it, and the VQ-regularized space has a variance close to 1 and needs no rescaling. {#c-snr}
Under the linear schedule of 1000 steps used by DDPM, LDM and DiT, the SNR of a unit-variance latent falls below 1 at step 259 and that of a latent of standard deviation 7 only at step 620, so scaling the latent by $k$ multiplies its SNR at every step by $k^2$ and moves the crossing by more than a third of the schedule. {#c-scale-snr}
The derivation `latent-scale-and-snr` proves the scaling and computes the crossings.

### Vector quantization

The VQ-regularized first stage descends from the VQ-VAE.
Its encoder output $z_e$ is replaced by the nearest of $K$ codebook vectors $e_1, \ldots, e_K$, which is [[m2:vector-quantization|vector quantization]]:

$$ k = \arg\min_j \lVert z_e - e_j \rVert_2 , \qquad z_q = e_k . $$ {#eq-vq}

Van den Oord and colleagues view the model as a VAE with a deterministic posterior $q(z = k \mid x)$ and a uniform prior, which makes the KL term constant and equal to $\log K$; since equation 2 has no real gradient, they approximate it like the straight-through estimator and copy gradients from the decoder input $z_q$ to the encoder output $z_e$. {#c-vqvae-def}
The loss has a reconstruction term and two terms that move the codebook and the encoder towards each other, with $\mathrm{sg}$ the stop-gradient:

$$ L = \log p(x \mid z_q) + \lVert \mathrm{sg}[z_e] - e \rVert_2^2 + \beta \lVert z_e - \mathrm{sg}[e] \rVert_2^2 . $$ {#eq-vq-loss}

The decoder optimizes the first term only, the encoder the first and last, and the embeddings the middle one; the algorithm was robust to $\beta$, with results unchanged from 0.1 to 2.0, and $\beta = 0.25$ was used in all experiments. {#c-vq-loss}
Written out, the encoder receives the decoder's gradient unchanged plus $2\beta(z_e - e)$, and each code receives $-2(z_e - e)$ summed over the vectors assigned to it, and nothing from the decoder. {#c-straight-through}
The derivation `vq-straight-through` proves this from eq-vq-loss, and `test_quantize_and_straight_through` checks the hand-written gradients against finite differences.

The codebook term alone, with the assignments held fixed, is minimized by the mean of the vectors assigned to each code, the update of [[m2:k-means-clustering|k-means]].
Van den Oord and colleagues note that the optimal $e_i$ is the average of the encoder outputs closest to it, the update of K-means, and propose exponential moving averages as an online version for minibatches, with $\gamma = 0.99$ working well in practice. {#c-ema}
With $n_i$ the number of vectors assigned to code $i$ in a batch and $\sum_j z_{i,j}$ their sum,

$$ N_i \leftarrow \gamma N_i + (1 - \gamma)\, n_i , \qquad m_i \leftarrow \gamma m_i + (1 - \gamma) \sum_j z_{i,j} , \qquad e_i = m_i / N_i . $$ {#eq-ema}

Because the same factor $1 - \gamma$ multiplies the counts and the sums, it cancels in the ratio: started from $N_i = m_i = 0$, one step puts $e_i$ exactly at the batch mean, with none of the start-up bias of an EMA of the codes themselves. {#c-codebook-ema}
The derivation `codebook-mean-and-ema` proves both facts.

With $K$ codes a VQ latent costs $\log_2 K$ bits per position.
Van den Oord and colleagues compress $128 \times 128 \times 3$ images to a $32 \times 32 \times 1$ discrete space with $K = 512$, a reduction of $\frac{128 \times 128 \times 3 \times 8}{32 \times 32 \times 9} \approx 42.6$ in bits, and model the images by fitting a PixelCNN prior over $z$. {#c-vqvae-compression}
VQ-VAE-2 compresses a $256 \times 256$ image into a $64 \times 64$ and a $32 \times 32$ map of codes; Razavi and colleagues report that these representations are over 30 times smaller than the original image and that training and sampling the prior over them is 30 times faster than in pixel space. {#c-vqvae2}
VQGAN made the code perceptual: Esser and colleagues replace the L2 reconstruction loss by a perceptual loss and add a patch-based discriminator, weighted adaptively by $\lambda = \nabla_{G_L}[L_{\text{rec}}] / (\nabla_{G_L}[L_{\text{GAN}}] + \delta)$ with $\delta = 10^{-6}$ for numerical stability, and they apply a single attention layer at the lowest resolution. {#c-vqgan}
That first stage is the one Rombach and colleagues reuse.

## Components

### The latent diffusion objective

The second stage is an ordinary denoising diffusion model on $z$: with $\epsilon \sim \mathcal{N}(0, I)$, $t$ uniform on $\{1, \ldots, T\}$ and $z_t = \sqrt{\bar\alpha_t} z + \sqrt{1 - \bar\alpha_t}\,\epsilon$,

$$ L_{\text{LDM}} = \mathbb{E}_{\mathcal{E}(x),\, \epsilon \sim \mathcal{N}(0, I),\, t} \Big[ \big\lVert \epsilon - \epsilon_\theta(z_t, t) \big\rVert_2^2 \Big] . $$ {#eq-ldm}

Rombach and colleagues realize $\epsilon_\theta$ as a time-conditional U-Net; since the forward process is fixed, $z_t$ can be obtained from $\mathcal{E}$ during training, and samples from $p(z)$ are decoded to image space with a single pass through $\mathcal{D}$. {#c-ldm-loss}
Nothing in eq-ldm refers to pixels: the [[diffusion-model]] page's samplers, schedules, noise and velocity parameterizations and distillation methods apply unchanged, and so do flow-based objectives such as [[flow-matching]], used by later latent models.
The best predictor in eq-ldm is the [[m2:conditional-expectation|conditional expectation]] $\mathbb{E}[\epsilon \mid z_t]$, which equals $-\sqrt{1 - \bar\alpha_t}\,\nabla_{z_t} \log q(z_t)$.
When the latents follow a [[al:gaussian-mixture-model|Gaussian mixture]] it has a closed form, which the implementations use as an exact denoiser: `test_exact_eps_is_the_scaled_score` checks it against the gradient of the log density of the noisy mixture, and `test_ddim_with_the_exact_predictor` finds that deterministic DDIM sampling with it recovers the weights of a three-component mixture within 0.015 and its means within 0.02.

### Conditioning by cross-attention

A domain-specific encoder $\tau_\theta$ turns the condition $y$, a class, a caption or a layout, into $M$ tokens of width $d_\tau$, and the denoiser's intermediate features attend to them:

$$ \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt d}\Big) V , \qquad Q = W_Q\, \varphi(z_t) , \quad K = W_K\, \tau_\theta(y) , \quad V = W_V\, \tau_\theta(y) , $$ {#eq-cross-attn}

where $\varphi(z_t) \in \mathbb{R}^{N \times d}$ is a flattened intermediate representation of the U-Net.
Rombach and colleagues project $y$ to $\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}$ and map it into the U-Net's intermediate layers through this cross-attention, and train $\tau_\theta$ and $\epsilon_\theta$ jointly on the conditional objective $\mathbb{E}\,\lVert \epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y)) \rVert_2^2$; $\tau_\theta$ can be a domain expert, such as a transformer when $y$ is a text prompt. {#c-cross-attn}
With a single context token, $M = 1$, the softmax has one entry, equal to 1, so the attention output is the value $W_V \tau_\theta(y)$ whatever the query: one-token cross-attention is a learned bias per condition, the same as adding a class embedding. {#c-single-token}
The derivation `single-token-cross-attention` proves it and `test_single_token_cross_attention_is_a_class_bias` checks it; it is why class-conditional models can drop attention, and why text, with many tokens, needs it.

### Classifier-free guidance

Training drops the condition to a null token with some probability, so one network learns both $\epsilon_\theta(z_t, t, y)$ and $\epsilon_\theta(z_t, t, \varnothing)$, and sampling uses

$$ \tilde\epsilon = \epsilon_\theta(z_t, t, \varnothing) + s\,\big(\epsilon_\theta(z_t, t, y) - \epsilon_\theta(z_t, t, \varnothing)\big) , $$ {#eq-cfg}

which for $s > 1$ follows the score of $p(z_t \mid y)\, p(y \mid z_t)^{s - 1}$, a sharpened conditional.
Ho and Salimans train the conditional and unconditional models as one network by replacing the condition with a null token with probability $p_{\text{uncond}}$, and sample with $(1 + w)\epsilon_\theta(z, c) - w\epsilon_\theta(z)$, so that $s = 1 + w$. {#c-cfg}
LDM's best ImageNet and text-to-image numbers use it, at the cost of two network calls per step.
`test_classifier_free_guidance` runs DDIM with the exact predictors of two overlapping classes: at $s = 0$ the samples are the unconditional mixture and the mean classifier probability of class 0 is 0.5, and it rises with $s$ to above 0.9 at $s = 3$.
The [[diffusion-model]] page covers guidance in general.

### Training the stages together

LDM freezes the first stage. LSGM trains both together.
Vahdat and colleagues train a score-based model in the latent space of a variational autoencoder end to end, which lets the model be more expressive, apply to non-continuous data, and learn smoother score models in a smaller space with fewer network evaluations; it reached an FID of 2.10 on CIFAR-10 and matched earlier score-based models on CelebA-HQ-256 while sampling two orders of magnitude faster. {#c-lsgm}
The bound is that of a [[variational-autoencoder]] whose prior is the diffusion model:

$$ -\log p(x) \le \mathbb{E}_{q(z_0 \mid x)}\big[-\log p(x \mid z_0)\big] + \mathrm{KL}\big(q(z_0 \mid x) \,\|\, p_\theta(z_0)\big) = \underbrace{\mathbb{E}_q[-\log p(x \mid z_0)]}_{\text{reconstruction}} - \underbrace{H\big(q(z_0 \mid x)\big)}_{\text{encoder entropy}} + \underbrace{\mathrm{CE}\big(q(z_0 \mid x) \,\|\, p_\theta(z_0)\big)}_{\text{cross entropy}} . $$ {#eq-two-stage}

Vahdat and colleagues decompose the KL term into the negative entropy of the encoder and the [[m2:cross-entropy|cross entropy]] between the encoder distribution and the score-based prior, estimate the first two terms with the reparameterization trick, and derive a denoising score matching expression for the cross entropy, because the KL form needs the marginal score of $q(z_t)$, which is not available when the encoder is trained. {#c-lsgm-ce}
Averaged over the data, the cross-entropy term is the cross entropy from the aggregate posterior $q(z_0) = \mathbb{E}_x q(z_0 \mid x)$ to the prior, minimized over priors by $p_\theta = q(z_0)$: the best diffusion prior on a fixed encoder models the distribution of encoded data, which is what LDM's second stage fits. {#c-two-stage-bound}
The derivation `two-stage-bound` proves eq-two-stage and this property, so that LDM's two stages minimize the same bound coordinate-wise: first the reconstruction with a nearly free KL weight, then the cross entropy with the encoder fixed.

### Transformers on latent patches

DiT keeps the latent and replaces the U-Net with a [[vision-transformer]].
For $256 \times 256 \times 3$ images $z$ has shape $32 \times 32 \times 4$; patchify embeds each $p \times p$ patch as one token, so there are $T = (I/p)^2$ tokens for a latent of side $I$, halving $p$ quadruples $T$ and at least quadruples the transformer's Gflops, and changing $p$ has no meaningful effect on the parameter count. {#c-dit-patch}
Each block is conditioned on the time and class through adaptive [[layer-normalization]]: a small network regresses the scale and shift of each normalization and a gate $\alpha$ on each [[residual-connection|residual branch]],

$$ x \leftarrow x + \alpha \odot F\big((1 + \gamma) \odot \mathrm{LN}(x) + \beta\big) , \qquad (\gamma, \beta, \alpha) = \mathrm{MLP}(c) . $$ {#eq-adaln}

Peebles and Xie initialize the MLP to output zero for every $\alpha$, which initializes the full DiT block as the identity function; this adaLN-Zero block gave a lower FID than cross-attention and in-context conditioning while being the most compute-efficient, and cross-attention added about 15% to the Gflops. {#c-adaln-zero}
With $\alpha = 0$ the block of eq-adaln is the identity, and on the first step the gradient reaches the weights that produce $\alpha$ but nothing inside $F$, so training grows each branch from zero. {#c-adaln-identity}
The derivation `adaln-zero-identity` proves both.
DiT-XL/2 reached an FID of 2.27 on class-conditional ImageNet 256 at 118.6 Gflops, and across DiT configurations a higher Gflops count, through depth, width or more tokens, consistently gave a lower FID. {#c-dit}

## Shapes and parameters

The first stage of a real latent diffusion model is a convolutional [[autoencoder]] of tens of millions of parameters; the implementations here use the simplest one that downsamples: cut a signal of length $Nf$ into $N$ patches of length $f$ and map each patch linearly to $c$ channels, fitted in closed form by [[al:principal-component-analysis|principal component analysis]] of the patches.
It has the shape of eq-autoencoder in one dimension, a factor $f/c$ fewer numbers, and its reconstruction error is the sum of the discarded eigenvalues of the patch covariance, which `test_autoencoder_is_pca` checks.

The shape contract is the denoiser: a network that reads a noisy latent of $N$ positions by $c$ channels, flattened to $D = Nc$; adds a sinusoidal embedding of the step through $W_t$; passes the sum through a tanh layer of width $H$; lets that layer attend through eq-cross-attn to $M$ learned context tokens of width $d$ for the class, with a table of $C + 1$ classes whose last row is the null class of eq-cfg; applies a second tanh; and returns the predicted noise as the noisy latent plus a linear read-out:

$$ h = \tanh\big(z W_1 + \mathrm{emb}(t) W_t + b_1\big) , \qquad \hat\epsilon = z + \tanh\big(h + \mathrm{Attention}(h W_q, \mathrm{ctx}_y W_k, \mathrm{ctx}_y W_v)\big) W_2 + b_2 . $$

Its parameter count is

$$ P = NcH + EH + H + (C + 1)Md + Hd + d^2 + dH + HNc + Nc . $$ {#eq-params}

In the contract this is
{shape:params}
and the example with $N = c = 2$, $H = 5$, $E = 4$, $M = 2$, $d = 3$ and $C = 2$ has {shape:params@example} parameters.
The denoiser trained in the tests, `mixture`, has {shape:params@mixture}, and the same form on the $32 \times 32 \times 4$ latent of a $256 \times 256$ ImageNet image with 1000 classes, `imagenet_f8`, would have {shape:params@imagenet_f8}.
Real denoisers are far larger: LDM's class-conditional ImageNet U-Net has about 400 million parameters and its text-to-image model 1.45 billion; their size does not depend on the latent size the way this flattened network's does, because they share weights across positions.
For the denoiser of this page, the parameter count is eq-params and one call on $B$ latents costs $2B(NcH + EH + Hd + Md^2 + MdH + Md + MH + HNc)$ FLOPs. {#c-params}
A latent with downsampling factor $f$ and $c$ channels has $3f^2/c$ times fewer numbers than an RGB image, 48 for the $f = 8$, $c = 4$ latent of a $256 \times 256 \times 3$ image, and a code of $K$ entries per position costs $\log_2 K$ bits against 24 bits per RGB pixel, so the VQ-VAE code of a $128 \times 128 \times 3$ image with $f = 4$ and $K = 512$ has $\frac{8 \cdot 3 \cdot 16}{9} \approx 42.7$ times fewer bits. {#c-compression-ratio}
The derivation `parameter-and-flop-count` proves the counts and `compression-ratio` the ratios.

## Computation and cost

One call of the denoiser on $B$ latents costs

$$ C_f = 2B\big(NcH + EH + Hd + Md^2 + MdH + Md + MH + HNc\big) $$ {#eq-flops}

FLOPs, a multiply-add counted as two, leaving out the biases, the time embedding, the tanh and the softmax.
In the contract,
{shape:flops_forward}
which is {shape:flops_forward@example} FLOPs for the example batch and {shape:flops_forward@mixture} for one training batch of 256 latents in the `mixture` test.
For a latent model the arithmetic of the whole pipeline is what matters.
The cost of a convolutional or transformer denoiser grows at least linearly with the number of latent positions, and attention quadratically, so moving from $256 \times 256$ pixels to a $32 \times 32$ latent cuts the positions 64 times.
Sampling costs one denoiser call per step, two with classifier-free guidance, plus one decoder pass at the end, so the decoder's cost is spread over all the steps.
For a latent of the same size, the latent of $f$ and $c$ against the image is $3f^2/c$ times smaller, by c-compression-ratio.
Rombach and colleagues' LSUN-Bedrooms model scored close to the pixel-space ADM while using half its parameters and four times less training compute, and DiT-XL/2 at 118.6 Gflops per forward pass beat ADM at 1120 Gflops; Peebles and Xie estimate training compute as the forward Gflops times the batch size times the number of steps times 3, the backward pass counting as twice the forward.
The first stage is trained once and its cost is shared by every model built on it.

## Training

Training is two separate runs.
The first trains the autoencoder with a reconstruction loss, a perceptual loss, a patch adversarial loss and a small KL or VQ regularizer, and freezes it.
The second encodes the training images, often once in advance, rescales the latents by eq-scale, and minimizes eq-ldm with $t$ and $\epsilon$ drawn afresh for each latent, dropping the condition with a probability of about 0.1 to allow eq-cfg.
DiT reuses the Stable Diffusion autoencoder with $f = 8$, a linear variance schedule of 1000 steps from $10^{-4}$ to $2 \times 10^{-2}$, AdamW, and an exponential moving average of the weights with decay 0.9999 for evaluation.

The implementations train the denoiser of the shape contract on latents drawn from a mixture of three Gaussians in eight dimensions, standing in for the encoded classes of a dataset, with the linear 1000-step schedule, batches of 256, [[adam|Adam]] at a learning rate of $3 \times 10^{-3}$ and a null-class probability of 0.1.
`test_training_and_sampling` trains for 1500 steps and checks that the loss falls by more than half, that the squared error between the network and the exact predictor of the class is below 0.2 at $t = 300$ and below 0.1 at $t = 600$ and $t = 900$ against $\mathbb{E}\lVert\epsilon\rVert^2 = 8$, and that 25 DDIM steps with the trained network put the class mean within 0.35 in every coordinate and the per-coordinate standard deviation within 30% of the true 0.3.
The NumPy reference trains with its hand-written gradient, the PyTorch version with autograd and `torch.optim.Adam`, and the JAX version as a `jax.lax.scan` of Adam steps under `jax.jit`.
`test_two_stage_pipeline` runs the whole method: it generates signals of length 16 whose patches lie near a two-dimensional subspace, fits the principal component autoencoder with $f = 4$ and $c = 2$, rescales the latents by eq-scale, samples latents of each class with DDIM and the exact predictor fitted to the rescaled latents, multiplies by $\hat\sigma$, decodes, and finds the per-position means of the decoded signals within 0.05 of those of the data of the class.

## Inductive biases

The model assumes that perceptual detail is cheap to restore and expensive to model, and divides the work accordingly: a deterministic decoder, trained with perceptual and adversarial losses, restores texture, and the generative model handles layout and content.
The latent keeps the image's spatial grid, so a convolutional U-Net or a transformer on patches carries over the locality and weight sharing of image models; Rombach and colleagues credit this two-dimensional latent, in contrast to the one-dimensional token orderings of earlier autoregressive latent models, for allowing mild compression with good reconstructions.
A single scale for the whole latent, set by eq-scale, ties the noise schedule to the data; the model is not invariant to that scale.
Conditioning by attention to tokens makes the condition's encoder interchangeable, and with eq-cfg trades diversity for agreement with the condition.
The first stage bounds the quality of every sample: nothing the decoder cannot produce can be generated.

## Evidence

Rombach and colleagues compared downsampling factors $f \in \{1, 2, 4, 8, 16, 32\}$ on ImageNet with the compute fixed to one A100: small factors trained slowly and large ones stagnated, $f$ from 4 to 16 struck a good balance between efficiency and faithful results, with a gap of 38 in FID between pixel-space LDM-1 and LDM-8 after 2 million steps, and LDM-4 and LDM-8 gave the best conditions for high-quality synthesis. {#c-downsampling-tradeoff}
Their reported results include an FID of 5.11 on CelebA-HQ $256 \times 256$ for unconditional LDM-4, an FID of 3.60 on class-conditional ImageNet for the 400-million-parameter LDM-4-G with classifier-free guidance at $s = 1.5$, an FID of 12.63 on MS-COCO for the 1.45-billion-parameter text-to-image LDM-KL-8-G with guidance, and on LSUN-Bedrooms a score close to ADM with half its parameters and four times less training compute. {#c-results}
Their Fig. 1 shows the first-stage trade-off directly: their $f = 4$ autoencoder reconstructs ImageNet validation images with a PSNR of 27.4 and a reconstruction FID of 0.58, against 22.8 and 32.01 for DALL-E's $f = 8$ discrete autoencoder and 19.9 and 4.98 for a VQGAN with $f = 16$.
LSGM's FID of 2.10 on CIFAR-10 and DiT's 2.27 on ImageNet 256 extend the evidence to jointly trained and transformer-based latent models.
VQ-VAE's likelihood, a bound of 4.67 bits per dimension on CIFAR-10 against 4.51 for a continuous VAE, suggested that a discrete latent loses little.

## Limitations

The decoder caps quality: small text, faces at small scale, hands and fine regular patterns suffer from the autoencoder's reconstruction errors, however good the diffusion model is, and the first-stage choice of $f$ and $c$ cannot be changed without retraining everything after it.
Likelihoods are not comparable to those of pixel-space models, since the latent model's bound covers only the latent, and the decoder of a perceptually trained autoencoder defines no density.
The latent scale must match the noise schedule; a wrong $\hat\sigma$ shifts the effective schedule by c-snr, and models trained on one autoencoder cannot be used with another.
Sampling is still iterative, tens of steps with a solver, and guidance doubles the cost per step; the distillation methods of the [[diffusion-model]] page are needed for one to four steps.
Training data at the scale of text-to-image models carries the usual problems of memorization, bias and provenance, and the openness of Stable Diffusion's weights made these concerns concrete.

## Variants and descendants

- VQ-VAE and VQ-VAE-2 fit autoregressive PixelCNN priors to discrete codes, and VQGAN an autoregressive [[transformer]]; DALL-E and Parti are text-to-image models of this kind.
- MaskGIT and Muse replace the autoregressive prior on codes by masked parallel decoding, a [[discrete-diffusion-model]] on the latent tokens.
- LSGM trains the autoencoder and a score-based prior jointly by eq-two-stage.
- Stable Diffusion 1 and 2 are LDMs with $f = 8$, $c = 4$ and a CLIP or OpenCLIP text encoder; SDXL enlarges the U-Net and conditions on two text encoders; SD3 replaces the U-Net by a multimodal transformer trained with rectified [[flow-matching]].
- DiT puts a transformer with adaLN-Zero on latent patches; PixArt, Sora and other video models extend it to text and to space-time latents.
- Latent consistency models and adversarial diffusion distillation distil a latent model into a few steps.
- The prior on the latent need not be a diffusion model: a [[normalizing-flow]] prior gives an exact density for the latent but must keep its dimension, and an [[energy-based-model]] prior needs Markov chain sampling; the diffusion prior learns only the score and samples with a fixed number of steps.
- Audio and speech models such as AudioLDM apply the same recipe to spectrogram latents.

## Notable models

- Rombach and colleagues' LDM family (2021): unconditional, class-conditional, inpainting, super-resolution, layout-to-image and text-to-image models on $f = 4$ and $f = 8$ latents.
- Stable Diffusion (2022): the text-to-image LDM trained on LAION and released with open weights.
- DiT-XL/2 (2022): a latent diffusion transformer with an FID of 2.27 on ImageNet 256.
- LSGM (2021): the jointly trained latent score-based model with an FID of 2.10 on CIFAR-10.

## Implementations

The NumPy reference `impl/numpy/latent_diffusion.py` implements:

- the first stage: the principal component autoencoder of patches as a one-dimensional eq-autoencoder, the KL penalty of a KL-regularized latent and the scale of eq-scale;
- vector quantization: the lookup of eq-vq, the terms of eq-vq-loss with their straight-through gradients written out, the k-means step and the EMA codebook of eq-ema;
- the second stage: the linear schedule, forward noising, the signal-to-noise ratio of eq-snr, the exact noise predictor of a Gaussian mixture, deterministic DDIM and the guidance of eq-cfg;
- the denoiser of the shape contract, conditioned through eq-cross-attn, with its loss gradient written by hand and Adam on eq-ldm.

The PyTorch version `impl/pytorch/latent_diffusion.py` has the denoiser as an `nn.Module`, the straight-through estimator as `z_e + (z_q - z_e).detach()`, the gradients from autograd, `torch.autograd.gradcheck` through the denoiser, and training with `torch.optim.Adam`.
The JAX version `impl/jax/latent_diffusion.py` writes the stop-gradients with `jax.lax.stop_gradient` and checks with `jax.grad` that the codebook term moves only the codebook and the commitment term only the encoder; it runs DDIM as a `jax.lax.scan` and training as a jitted scan.
All three load the shared fixture of `fixtures/make.py` and check the predicted noise, the loss and its gradient in every parameter, the autoencoder's latents up to the sign of each component, the code indices, the straight-through gradients, one EMA step and a four-step DDIM sample.

### Worked example

Take two signals of length 4, $(1, 1, 3, 3)$ and $(-1, -1, 1, 1)$, and cut them into patches of length $f = 2$: $(1, 1)$, $(3, 3)$, $(-1, -1)$ and $(1, 1)$.
Their mean is $(1, 1)$, and all four lie on the line through it in the direction $(1, 1)/\sqrt 2$, so the principal component autoencoder with one channel, $c = 1$, loses nothing.
The latents are the coordinates along the line: $(1, 1) - (1, 1)$ gives 0, $(3, 3) - (1, 1) = (2, 2)$ gives $2\sqrt2 \approx 2.83$, and $(-1, -1)$ gives $-2\sqrt 2$, so the two signals become $(0, 2\sqrt 2)$ and $(-2\sqrt 2, 0)$, up to the sign of the component, half as many numbers.
Decoding multiplies by $(1, 1)/\sqrt 2$ and adds the mean back, and gives the signals exactly.
The four latent numbers $0, 2\sqrt2, -2\sqrt2, 0$ have mean 0 and variance $(8 + 8)/4 = 4$, so eq-scale gives $\hat\sigma = 2$ and the diffusion model would see $(0, \sqrt 2)$ and $(-\sqrt 2, 0)$.
`test_worked_example` checks these numbers in all three implementations.

### Pitfalls

> [!pitfall]
> Forgetting the scale factor, or estimating it on a different autoencoder, shifts the effective noise schedule by eq-snr. A latent of standard deviation 7 under the linear schedule keeps its SNR above 1 until step 620 instead of 259, so most of the schedule is spent at noise levels the training never reaches. Multiply by $\hat\sigma$ again before decoding.

> [!pitfall]
> At the last steps of the linear schedule $\sqrt{\bar\alpha_T} \approx 0.0064$, and DDIM's estimate $(z_t - \sqrt{1 - \bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$ multiplies any error in $\hat\epsilon$ by up to 158. A small network that cannot represent the identity well, as a tanh layer cannot, then samples latents several times too wide even when its loss looks fine. The denoiser of this page predicts $z_t$ plus a correction, which makes the large-$t$ prediction, where $\epsilon \approx z_t$, easy; real models get the same from residual connections, or predict $v$ or $x_0$ instead.

> [!pitfall]
> The straight-through estimator gives the codebook no gradient from the decoder, by c-straight-through. Without the codebook term or the EMA update of eq-ema the codes never move, and codes that no encoder output is near are never used; restart dead codes from encoder outputs, or use a smaller codebook dimension.

> [!pitfall]
> The KL weight of a KL-regularized first stage is tiny, about $10^{-6}$, so the latent is not close to a standard normal; do not sample the decoder from $\mathcal{N}(0, I)$ as in a VAE, and do not skip eq-scale.

> [!pitfall]
> Classifier-free guidance needs the null condition in training. A model trained without dropping the condition has never seen the null token, and eq-cfg then extrapolates from an untrained output.

## Experiments

This page has no experiment yet.
Some tests are small experiments.

`test_training_and_sampling`, described in the Training section, trains the denoiser end to end and samples a class with 25 DDIM steps; `test_two_stage_pipeline` runs the encoder, the scale, DDIM in the latent space and the decoder, and compares the decoded signals with the data of each class.
`test_classifier_free_guidance` measures how the classifier probability of guided samples grows with the guidance scale on two overlapping classes.
`test_kmeans_and_ema` checks that the k-means step never increases the quantization error and that the EMA codebook stays at the cluster means.

The checks of the eight derivations compare:

- the latent and code sizes with the compression ratios;
- the hand-written straight-through gradients with finite differences of eq-vq-loss;
- the codebook optimum and the EMA ratio with the cluster means;
- the shift of the SNR crossing with the latent variance;
- one-token cross-attention with a class bias;
- eq-two-stage with Monte Carlo and closed-form Gaussian values, and the cross entropy against the aggregate posterior;
- an adaLN-Zero block at initialization with the identity, and its first gradients;
- the parameter and FLOP counts with the code.

## History

Two-stage generation is older than diffusion.
Van den Oord, Vinyals and Kavukcuoglu's VQ-VAE of 2017 learned a discrete code of images with the straight-through estimator and fitted a PixelCNN to it, and Razavi and colleagues' VQ-VAE-2 of 2019 made the code hierarchical.
Esser, Rombach and Ommer's VQGAN of 2020 trained the first stage with a perceptual and adversarial loss, which allowed stronger compression, and fitted an autoregressive [[transformer]] to the codes; DALL-E of 2021 did the same for text.
After Ho and colleagues' denoising diffusion of 2020 and Dhariwal and Nichol's guided diffusion of 2021 found that pixel-space diffusion beats GANs at great cost, Vahdat, Kreis and Kautz's LSGM and Rombach, Blattmann, Lorenz, Esser and Ommer's LDM, both of 2021, moved diffusion into a latent space, the first jointly and the second with a fixed first stage and cross-attention for conditioning.
The text-to-image LDM trained on LAION was released in 2022 as Stable Diffusion, with open weights, and made text-to-image generation widely available.
Peebles and Xie's DiT of 2022 replaced the U-Net with a transformer, and later systems, from SD3 to video generators, combined latent transformers with [[flow-matching]] objectives.
The latent diffusion model joins the threads of the generative families: its first stage is an [[autoencoder]] regularized like a [[variational-autoencoder]] and trained with the losses of a [[generative-adversarial-network]], its second stage a [[diffusion-model]], and its discrete ancestors [[autoregressive-model|autoregressive models]] on codes.

## References

- A. van den Oord, O. Vinyals and K. Kavukcuoglu. Neural Discrete Representation Learning. NIPS 2017; arXiv 1711.00937.
- A. Razavi, A. van den Oord and O. Vinyals. Generating Diverse High-Fidelity Images with VQ-VAE-2. NeurIPS 2019; arXiv 1906.00446.
- P. Esser, R. Rombach and B. Ommer. Taming Transformers for High-Resolution Image Synthesis. CVPR 2021; arXiv 2012.09841.
- A. Vahdat, K. Kreis and J. Kautz. Score-based Generative Modeling in Latent Space. NeurIPS 2021; arXiv 2106.05931.
- R. Rombach, A. Blattmann, D. Lorenz, P. Esser and B. Ommer. High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022; arXiv 2112.10752.
- J. Ho and T. Salimans. Classifier-Free Diffusion Guidance. NeurIPS 2021 Workshop on Deep Generative Models; arXiv 2207.12598.
- W. Peebles and S. Xie. Scalable Diffusion Models with Transformers. ICCV 2023; arXiv 2212.09748.
- J. Song, C. Meng and S. Ermon. Denoising Diffusion Implicit Models. ICLR 2021; arXiv 2010.02502.
