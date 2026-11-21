---
tag: NEW-text-to-image-generation
kind: method
slug: text-to-image-generation
name: Text-to-image generation
aliases: [text-to-image, T2I, text-conditional image generation, text-to-image synthesis, DALL-E, DALL·E 2, unCLIP, Imagen, Parti, GLIDE, Stable Diffusion, dynamic thresholding, contrastive reranking, DrawBench, PartiPrompts, T2I-CompBench]
area: multimodal-models/text-to-image
level: core
summary: >-
  Generating an image from a caption with a generative model conditioned on a text encoder, by predicting discrete image tokens one at a time or by denoising pixels or latents with guidance.
purpose: turning a free-form text description into a new image that matches it, for illustration, design, editing and data generation
uses: [diffusion-model, autoregressive-model, classifier-free-guidance, transformer, decoder-only-transformer, contrastive-language-image-pretraining, t5, variational-autoencoder, cross-entropy-loss@al]
related_to: [latent-diffusion-model, flow-matching, generative-adversarial-network, visual-instruction-tuning, byte-pair-encoding, mixed-precision-training, bert, principal-component-analysis@al]
hyperparameters:
  - {name: generator family, meaning: "how the image is produced from the text", typical: "an autoregressive Transformer over discrete image tokens (DALL-E, Parti); a cascade of pixel diffusion models (GLIDE, Imagen); a prior over CLIP image embeddings plus a diffusion decoder (unCLIP); diffusion in an autoencoder's latent space (latent diffusion, Stable Diffusion)"}
  - {name: text encoder, meaning: "the network that turns the caption into vectors the generator reads", typical: "BPE tokens in the same Transformer (DALL-E); a Transformer trained with the generator (GLIDE); CLIP text embeddings (unCLIP); a frozen T5-XXL with 4.6B parameters (Imagen); a pretrained Transformer encoder (Parti)"}
  - {name: guidance weight, meaning: "the classifier-free guidance scale w or λ; 1 is the plain conditional model", typical: "1.35 for Imagen's 64 × 64 model and 8.0 for its super-resolution model on COCO; 3.0 for GLIDE in its human evaluation; 1.25 for the unCLIP decoder on COCO"}
  - {name: caption dropout, meaning: "how often the caption is replaced by an empty one in training, so the same model also learns the unconditional distribution guidance needs", typical: "10% for Imagen and the unCLIP prior; 10% for CLIP embeddings and 50% for captions in the unCLIP decoder"}
  - {name: image tokens or resolution cascade, meaning: "how many tokens or which resolutions the model generates", typical: "32 × 32 = 1024 tokens over 8192 codes for a 256 × 256 image (DALL-E, Parti); 64 → 256 → 1024 pixels (Imagen, unCLIP)"}
  - {name: reranking samples, meaning: "how many images are drawn per caption before a contrastive model keeps the best-scoring ones", typical: "512 with CLIP for DALL-E; 16 with CoCa for Parti; none for Imagen"}
  - {name: thresholding, meaning: "how a diffusion sampler keeps guided x-predictions inside the training range [−1, 1]", typical: "static clipping, or Imagen's dynamic thresholding at the 99.5th percentile"}
cost:
  train: "250 million pairs and 1024 V100 GPUs for 430,000 updates for the 12B DALL-E transformer; 2.5M steps at batch 2048 on 256 TPU-v4 chips for Imagen's 2B base model and on 128 chips for each super-resolution model, on about 860M pairs"
  inference: "one forward pass per image token, 1024 per 256 × 256 image, for autoregressive models, times the number of reranked samples; one or two network evaluations per denoising step for diffusion, two with classifier-free guidance, over tens to hundreds of steps and every stage of a cascade"
history:
  year: 2021
  people: [Aditya Ramesh, Mark Chen, Ilya Sutskever, Alex Nichol, Prafulla Dhariwal, Chitwan Saharia, William Chan, Mohammad Norouzi, Jiahui Yu, Yonghui Wu, Robin Rombach]
  note: "Ramesh et al. posted DALL-E (February 2021), an autoregressive Transformer over caption and image tokens; Nichol et al. posted GLIDE (December 2021) with classifier-free guidance for text, and Rombach et al. latent diffusion (December 2021); Ramesh et al. posted unCLIP, released as DALL·E 2 (April 2022); Saharia et al. posted Imagen (May 2022) and Yu et al. Parti (June 2022); Huang et al. posted T2I-CompBench (July 2023, NeurIPS 2023 Datasets and Benchmarks)."
prereqs: [diffusion-model, autoregressive-model, classifier-free-guidance, contrastive-language-image-pretraining]
same_as:
  wikidata: null
  wikipedia: "https://en.wikipedia.org/wiki/Text-to-image_model"
  arxiv: ["2102.12092", "2112.10741", "2204.06125", "2205.11487", "2206.10789", "2307.06350"]
sources:
  - {paper: "Ramesh, Pavlov, Goh, Gray and others (2021)", title: "Zero-Shot Text-to-Image Generation", locator: "arXiv 2102.12092v2, pp. 1 to 7", note: "the discrete VAE, the single-stream Transformer, loss weights, data, CLIP reranking and the MS-COCO and CUB results"}
  - {paper: "Nichol, Dhariwal, Ramesh, Shyam and others (2021)", title: "GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models", locator: "arXiv 2112.10741v3, p. 1", note: "classifier-free guidance against CLIP guidance for text, and the comparison with DALL-E"}
  - {paper: "Ramesh, Dhariwal, Nichol, Chu and Chen (2022)", title: "Hierarchical Text-Conditional Image Generation with CLIP Latents", locator: "arXiv 2204.06125v1, pp. 1 to 5, 11, 13, 16 and 17", note: "the prior and decoder, the PCA tokens of the AR prior, results and the attribute-binding and text limitations"}
  - {paper: "Saharia, Chan, Saxena, Li and others (2022)", title: "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding", locator: "arXiv 2205.11487v1, pp. 1 to 9 and 26", note: "frozen T5-XXL, the cascade, dynamic thresholding, noise conditioning augmentation, DrawBench, training and results"}
  - {paper: "Yu, Xu, Koh, Luong and others (2022)", title: "Scaling Autoregressive Models for Content-Rich Text-to-Image Generation", locator: "arXiv 2206.10789v1, pp. 2 to 7, 12 and 22", note: "ViT-VQGAN tokens, the encoder-decoder model, guidance and reranking, PartiPrompts and the counting limits"}
  - {paper: "Rombach, Blattmann, Lorenz, Esser and Ommer (2021)", title: "High-Resolution Image Synthesis with Latent Diffusion Models", locator: "arXiv 2112.10752v2, p. 7", note: "the 1.45B text-conditional latent diffusion model on LAION-400M"}
  - {paper: "Huang, Sun, Xie, Li and Liu (2023)", title: "T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation", locator: "arXiv 2307.06350v2, the abstract, Sections 1 and 6.3 and Tables 3 and 4", note: "the compositional prompt set, the human scores of Stable Diffusion v2 and GORS"}
---

Text-to-image generation makes a new image from a written description.
A text encoder turns the caption into vectors, and a generative model trained on hundreds of millions of captioned web images samples an image conditioned on them.
Two families of generator dominate.
Autoregressive models, such as DALL-E and Parti, turn each image into a grid of discrete tokens and predict the tokens one at a time after the caption, as a language model predicts words.
Diffusion models, such as GLIDE, unCLIP (released as DALL·E 2), Imagen and the latent diffusion model behind Stable Diffusion, start from noise and denoise it step by step, reading the caption at every step.
Both families sample with classifier-free guidance, which trades the variety of the images for how closely they follow the caption.

## In plain words

A captioned photo collection teaches the model which pictures go with which words.
The model never copies a picture: it learns the distribution of images given a caption and draws new ones from it.

An autoregressive model first learns a visual alphabet.
A separate network compresses each image into a small grid of codes, each code standing for a patch, and can turn a grid of codes back into pixels.
The main model then reads the caption and writes the grid code by code, like a translator writing a sentence in a new language.

A diffusion model learns to remove noise.
Given a noisy image and the caption, it predicts the clean image.
Sampling starts from pure noise and repeats this many times, and the caption steers each step.

Guidance turns up the caption's influence.
The same network is run with the caption and without it, and the difference between the two predictions is magnified.
The images match the text better, and look more alike.

## Setting

The input is a caption y, a string of words, and the output is an image x, usually 256 × 256 to 1024 × 1024 pixels.
Training data are pairs (x, y) of images and captions, mostly alt-text scraped from the web, which is noisy: many captions describe their image only loosely.
The model learns p(x | y).
At sampling time it draws several images per caption, possibly with a guidance weight and a reranker.

Quality is judged in two ways.
Fidelity asks whether the image looks real, and is measured by FID against reference images and by human raters.
Alignment asks whether the image shows what the caption says, and is measured by CLIP score, by human raters and by question-answering models on prompt sets built to probe counting, colours, spatial relations and text.

## The method

**Autoregressive generation over image tokens.**
DALL-E trains a transformer to autoregressively model the text and image tokens as a single stream of data, and with sufficient data and scale this approach is competitive with previous domain-specific models when evaluated in a zero-shot fashion. {#c-de-stream}
Pixels themselves would make the sequence too long, so DALL-E trains in two stages.
In stage 1, DALL-E trains a discrete variational autoencoder (dVAE) to compress each 256×256 RGB image into a 32 × 32 grid of image tokens, each element of which can assume 8192 possible values, which reduces the context size of the transformer by a factor of 192 without a large degradation in visual quality. {#c-de-dvae}
In stage 2, DALL-E concatenates up to 256 BPE-encoded text tokens with the 32 × 32 = 1024 image tokens and trains an autoregressive transformer to model the joint distribution over the text and image tokens. {#c-de-concat}
The prior over text and image tokens is a 12-billion parameter sparse transformer; the lowercased caption is BPE-encoded with at most 256 tokens and vocabulary size 16,384, and the image is encoded with 32 × 32 = 1024 tokens with vocabulary size 8192. {#c-de-12b}
Since DALL-E is primarily interested in image modeling, it multiplies the cross-entropy loss for the text by 1/8 and the cross-entropy loss for the image by 7/8. {#c-de-weights}
To scale up to 12-billion parameters, DALL-E collects 250 million text-images pairs from the internet, a dataset that does not include MS-COCO but does include Conceptual Captions and a filtered subset of YFCC100M. {#c-de-data}
At sampling time the model writes the image tokens one at a time after the caption, and the dVAE decoder turns them into pixels.
DALL-E reranks the samples drawn from the transformer using a pretrained contrastive model, which assigns a score based on how well the image matches the caption, and all samples use reranking with N = 512 unless otherwise stated. {#c-de-rerank}
The contrastive model is CLIP, trained on image-caption pairs, so it can score any candidate against any caption.

Parti keeps the two stages and changes the model.
Parti treats text-to-image generation as a sequence-to-sequence modeling problem, akin to machine translation, with sequences of image tokens as the target outputs rather than text tokens in another language, using a ViT-VQGAN image tokenizer and an encoder-decoder Transformer scaled up to 20B parameters. {#c-pa-seq2seq}
Parti's ViT-VQGAN learns 8192 image token classes for the codebook, and the image tokens are fixed to 1024, that is 32×32 latent codes from a 256 × 256 input image, with a maximum of 128 text tokens. {#c-pa-tok}
At the scale of 350-million to 750-million parameters, the encoder-decoder variants of Parti outperformed decoder-only ones in both training loss and text-to-image generation quality. {#c-pa-encdec}
A separate encoder lets the text side be pretrained on text alone, as BERT is, before text-to-image training.

**Diffusion conditioned on text.**
A text-conditional diffusion model takes a noisy image, the noise level and the caption, and predicts the noise or the clean image.
GLIDE compares two guidance strategies for text-conditional diffusion, CLIP guidance and classifier-free guidance, and finds that the latter is preferred by human evaluators for both photorealism and caption similarity. {#c-gl-cfg}
Classifier-free guidance runs the same network with and without the caption and extrapolates from the second prediction past the first, so training must sometimes drop the caption.

Imagen comprises a frozen T5-XXL encoder to map input text into a sequence of embeddings and a 64×64 image diffusion model, followed by two super-resolution diffusion models for generating 256×256 and 1024×1024 images. {#c-im-arch}
Imagen conditions its base model on a pooled text embedding added to the diffusion timestep embedding and on the entire sequence of text embeddings through cross attention at multiple resolutions. {#c-im-xattn}
Imagen samples with the adjusted x-prediction (zt − σε̃)/αt, where ε̃(zt,c) = wε(zt,c) + (1 − w)ε(zt); setting w = 1 disables classifier-free guidance and increasing w > 1 strengthens its effect. {#c-im-cfg}
With weights well above 1 the method hits a problem.
Increasing the classifier-free guidance weight improves image-text alignment but damages image fidelity, producing highly saturated and unnatural images, because high guidance weights cause x-predictions to exceed the bounds [−1,1] of the training data, a train-test mismatch. {#c-im-mismatch}
Clipping each pixel to [−1, 1], static thresholding, prevents blank images but leaves many pixels pinned at the boundary.
Dynamic thresholding sets s at each sampling step to a certain percentile absolute pixel value in the x-prediction, and if s > 1, thresholds the x-prediction to the range [−s,s] and then divides by s, pushing saturated pixels inwards. {#c-im-dyn}
Imagen compares static thresholding with dynamic thresholding at p = 99.5 on 256 × 256 samples using a guidance weight of 5 for both the base model and the super-resolution model. {#c-im-p}
The super-resolution stages are trained to be robust to the base model's mistakes.
Imagen uses noise conditioning augmentation for both super-resolution models: the low-resolution conditioning image is corrupted with Gaussian noise at a random aug_level during training, the model is conditioned on aug_level, and inference sweeps over its values. {#c-im-aug}
Imagen trains a 2B parameter model for 64 × 64 synthesis and 600M and 400M parameter models for the two super-resolution stages, with a batch size of 2048 and 2.5M training steps for all models, on 256 TPU-v4 chips for the base model and 128 TPU-v4 chips for both super-resolution models. {#c-im-train}
Imagen drops text embeddings with 10% probability for classifier-free guidance and trains on internal datasets with ≈ 460M image-text pairs and the publicly available Laion dataset with ≈ 400M image-text pairs. {#c-im-data}

**A prior over CLIP image embeddings.**
unCLIP is a two-stage model: a prior that generates a CLIP image embedding given a text caption, and a decoder that generates an image conditioned on the image embedding. {#c-uc-two}
unCLIP stacks a prior P(zi|y) and a decoder P(x|zi,y) into P(x|y) = P(x,zi|y) = P(x|zi,y)P(zi|y), where the first equality holds because zi is a deterministic function of x and the second because of the chain rule. {#c-uc-chain}
The unCLIP decoder enables classifier-free guidance by randomly setting the CLIP embeddings to zero 10% of the time and randomly dropping the text caption 50% of the time during training, and two diffusion upsamplers go from 64×64 to 256×256 and then to 1024×1024 resolution. {#c-uc-decoder}
The prior comes in two forms.
For the autoregressive prior, unCLIP retains only 319 principal components out of the original 1,024 dimensions of the CLIP image embedding and quantizes each of the 319 dimensions into 1,024 discrete buckets. {#c-uc-pca}
Principal component analysis orders the directions by variance, so the autoregressive prior writes the most informative coordinates first.
The unCLIP diffusion prior is a decoder-only Transformer that predicts the unnoised CLIP image embedding directly with a mean-squared error loss, instead of the epsilon-prediction formulation. {#c-uc-diffprior}

**Diffusion in a latent space.**
Latent diffusion moves the diffusion model into the latent space of a pretrained autoencoder, which is much smaller than pixel space; the latent diffusion model page covers it.
For text-to-image modeling, Rombach et al. train a 1.45B parameter KL-regularized latent diffusion model conditioned on language prompts on LAION-400M, with a transformer text encoder mapped into the UNet via multi-head cross-attention. {#c-ldm-text}
Stable Diffusion is a model of this kind, trained on public data and released with its weights.

**Guidance and reranking in autoregressive models.**
Classifier-free guidance also works on token logits.
Parti samples with I = G(z) + λ(G(z,c) − G(z)), where λ is the weight of classifier-free guidance, and finds that it has a significant improvement on the output image-text alignment, especially on challenging text prompts. {#c-pa-cfg}
Parti finds contrastive reranking complementary to classifier-free guidance and, compared with the 512 images used in DALL-E, samples just 16 images per text prompt, reranked by the alignment score of a CoCa model. {#c-pa-rerank}

## Derivation

The full derivation is in `derivations/text-to-image-arithmetic.md`, and `text-to-image-arithmetic.check.py` checks every number.

**What the tokenizer buys.**
A 256 × 256 RGB image has 196,608 values; a 32 × 32 grid of 1,024 tokens cuts the image context by 196,608 / 1,024 = 192, and with up to 256 caption tokens a DALL-E sequence has at most 1,280 tokens. {#c-d-tokens}
A grid of 1,024 tokens over 8,192 codes holds at most 13,312 bits, against 1,572,864 bits for the 8-bit 256 × 256 RGB image. {#c-d-bits}
The tokenizer must discard fine detail, and DALL-E's samples lose it.

**One formula for guidance.**
Parti's G(z) + λ(G(z,c) − G(z)) equals Imagen's wε_c + (1 − w)ε_u with w = λ, and because the x-prediction (z − σε)/α is affine in ε and the weights sum to 1, guided noise gives the guided x-prediction w x̂_c + (1 − w) x̂_u. {#c-d-cfg}
Guidance on logits l_u + λ(l_c − l_u) samples in proportion to p(x | c) · p(c | x)^(λ − 1), the conditional model reweighted by an implicit classifier. {#c-d-implicit}
For λ > 1 the reweighting favours images that an implicit classifier ties to the caption, which is why guided samples match the text better and vary less.

**Why guided predictions leave the training range.**
If both x-predictions lie in [−1, 1], the guided x-prediction lies in [−(2w − 1), 2w − 1] for w ≥ 1, and the bound is reached when they disagree in sign; at w = 8 a pixel with x̂_c = 0.9 and x̂_u = 0 becomes 7.2. {#c-d-range}
In the numpy model with Gaussian images and exact denoisers, at t = 0.5 static thresholding pins 54% of 4,096 pixels at ±1 for w = 4 and 76% for w = 8, while dynamic thresholding at p = 99.5 pins 0.5% at both. {#c-d-toy}
Dynamic thresholding divides every pixel under the percentile by the same s, so it keeps their ratios; static clipping flattens every pixel over the bound to the same value.

**A pooled caption cannot bind attributes.**
A generator that sees only the mean of the caption's word embeddings gives a caption and its attribute-swapped twin the same distribution, so it is right on at most half of such pairs; with 4 cells and 6 colours the ceiling is 7/72 ≈ 0.097. {#c-d-pool}
A mean of word embeddings is a bag of words, and it is the same vector for "red cube, blue sphere" and "blue cube, red sphere".
Cross-attention over the sequence of text embeddings, as in Imagen, or a caption in the Transformer's own context, as in DALL-E and Parti, removes this ceiling.

**Best of N.**
Best-of-N with a perfect scorer is right with probability 1 − (1 − p)^N; doubling N gains most near N = ln 2 / (−ln(1 − p)), about 6.6 for p = 0.1, and from N = 32 to 512 adds under 3.5 points at p = 0.1. {#c-d-bestn}
DALL-E's 512 samples are 32 times Parti's 16; past a few dozen, extra samples mostly buy compute.

**Budgets.**
Keeping 319 of 1,024 principal components shortens the AR prior's sequence by 1,024 / 319 = 3.21. {#c-d-pca}
Batch 2,048 for 2.5M steps is 5.12 billion examples per Imagen model, about 6.0 passes over 860M pairs; DALL-E's batch 1,024 for 430,000 updates is 440.32M examples, about 1.76 passes over 250M pairs. {#c-d-budget}
PowerSGD's compression rate 1 − 5r/(8 d_model) is 83.3%, 85.1% and 85.9% for (r, d_model) = (512, 1,920), (640, 2,688) and (896, 3,968). {#c-d-powersgd}

## Worked example

The PyTorch implementation builds a small token world.
A scene has four cells, each holding one of six colour tokens, as if a tokenizer had already turned an image into four codes.
A caption names each cell's colour with a position word, in random order: "green at 2, red at 0, blue at 3, red at 1".
A decoder-only Transformer with two layers and width 64 reads the caption and then predicts the four image tokens, trained for 600 steps with next-token cross-entropy and 10% caption dropout.

1. With the caption in its context, the model's loss falls from 1.93 to 0.16 nats per token, and the sampled image matches the caption in all four cells for 99.8% of 4,000 test captions.
2. The same model fed only the mean of the caption's word embeddings matches 9.1% of the time. The ceiling for any such model is 7/72 ≈ 9.7%, because the mean fixes which colours occur but not where.
3. Now make half the training captions wrong, as noisy alt-text would be. Without guidance the model matches 46.6% of test captions, close to the 50% of correct training pairs. With guidance weight 2 it matches 89.3%, and with weight 3 98.4%.
4. Guidance pays in diversity. The entropy of the sampled images falls from 4.46 nats at weight 1, close to the 4.27 of the training distribution, to 0.26 nats at weight 3.
5. Reranking does the same job at a different cost. Drawing 4 samples and keeping the one an exact scorer likes best raises the match rate from 45.9% to 89.4%, near the 91.4% that 1 − (1 − p)⁴ predicts.

The NumPy implementation checks the diffusion side with Gaussian "images" and exact denoisers.
At guidance weight 8 the guided x-prediction peaks near 5, far outside [−1, 1].
Static clipping pins 76% of the pixels at ±1, while dynamic thresholding at p = 99.5 pins 0.5% and keeps the rest in proportion.

## Hyperparameters

The guidance weight is the main knob at sampling time.
Raising it improves alignment and worsens diversity, and in pixel diffusion it saturates colours unless the sampler thresholds.
Text-to-image systems use larger weights than class-conditional ones, often 3 to 10.

The text encoder matters more than its size in the paper headlines suggests.
In Imagen, a larger frozen T5 helped more than a larger U-Net, and T5 beat CLIP's text encoder on hard prompts even where COCO metrics tied.

The number of reranked samples sets a trade between quality and cost, with diminishing returns after a few dozen.
Caption dropout of about 10% is enough to learn the unconditional model guidance needs.
The resolution cascade or token count sets how much detail the model can produce at all.

## Cost

Autoregressive models pay one Transformer forward pass per image token, so a 32 × 32 grid takes 1,024 sequential steps per image, and reranking multiplies that by the number of samples.
Diffusion models pay one network evaluation per denoising step, doubled by classifier-free guidance, for every stage of the cascade.
Latent diffusion cuts the cost of each step by working on a small latent instead of pixels.

Training is web-scale: hundreds of millions of image-text pairs and hundreds to a thousand accelerators for weeks.
Getting DALL-E's 12B model to train in 16-bit precision past one billion parameters needed per-resblock gradient scaling and PowerSGD gradient compression; mixed-precision training covers the general problem.

## Evidence

**DALL-E.**
In a best-of-five human vote against DF-GAN on MS-COCO captions, the DALL-E sample was chosen as the most realistic 90.0% of the time and as the image best matching a shared caption 93.3% of the time. {#c-de-human}
DALL-E obtains an FID score on MS-COCO within 2 points of the best prior approach despite never training on the captions, but on CUB there is a nearly 40-point gap in FID between DALL-E and the leading prior approach. {#c-de-fid}
FID and IS for MS-COCO improve as the sample size used for reranking with the contrastive model is increased, a trend that continues up to a sample size of 32, after which DALL-E observes diminishing returns. {#c-de-rr32}

**GLIDE and unCLIP.**
GLIDE samples are preferred to those from DALL-E 87% of the time when evaluated for photorealism and 69% of the time when evaluated for caption similarity. {#c-gl-dalle}
On MS-COCO 256 × 256, unCLIP reaches a zero-shot FID of 10.39 with the diffusion prior and 10.63 with the autoregressive prior, against 12.24 for GLIDE. {#c-uc-fid}
Against GLIDE, unCLIP with the diffusion prior is preferred for photorealism 48.9% ± 3.1% of the time, for caption similarity 45.3% ± 3.0% and for diversity 70.5% ± 2.8%. {#c-uc-human}
Against GLIDE, unCLIP keeps photorealism and caption similarity close to even and is clearly more diverse.

**Imagen.**
Imagen reaches a zero-shot FID-30K of 7.27 on COCO, against 12.24 for GLIDE and 10.39 for DALL-E 2, and lower than the 7.55 of Make-A-Scene trained on COCO. {#c-im-fid}
Against original COCO images, Imagen samples are chosen as more photorealistic 39.5% of the time, 43.9% on prompts without people, and score 91.4 on alignment against 91.9 for the original images. {#c-im-human}
Scaling the text encoder size improves Imagen's image-text alignment and image fidelity, and scaling the text encoder is significantly more impactful than scaling the U-Net size. {#c-im-scale}
T5-XXL and CLIP text encoders perform similarly on the COCO validation set in CLIP and FID scores, but human raters prefer T5-XXL over CLIP on DrawBench across all 11 categories. {#c-im-t5clip}
Conditioning over the sequence of text embeddings with cross attention significantly outperforms simple mean or attention based pooling in both sample fidelity and image-text alignment. {#c-im-xattn-res}

**Parti.**
Parti reaches a zero-shot FID of 7.23 and a finetuned FID of 3.22 on MS-COCO, and 15.97 zero-shot and 8.39 finetuned on Localized Narratives. {#c-pa-fid}

**Composition.**
In human evaluation of Stable Diffusion v2 on T2I-CompBench, the spatial relationship score is 0.3467, against 0.9827 for non-spatial relationships. {#c-cb-sd2}
According to the human evaluation results, spatial relationship is the most challenging sub-category for text-to-image models, attribute binding (shape) is also challenging, and non-spatial relationship is the easiest sub-category. {#c-cb-hard}

## Limitations and failure modes

Composition is the central weakness.
Text-to-image models often fail to compose multiple objects with different attributes and relationships into a complex and coherent scene: given the prompt "a blue bench on the left of a green car", a model might bind attributes to the wrong objects or generate the spatial layout incorrectly. {#c-cb-fail}
unCLIP is worse at binding attributes to objects than a corresponding GLIDE model, which Ramesh et al. hypothesize occurs because the CLIP embedding itself does not explicitly bind attributes to objects. {#c-uc-bind}
The pooled-caption ceiling of the derivation is the extreme case of this: a conditioning vector that does not say which attribute belongs to which object cannot produce the right binding more than half the time.

Text inside images is hard.
unCLIP struggles at producing coherent text, possibly because the CLIP embedding does not precisely encode spelling information of rendered text, and the BPE encoding obscures the spelling of the words in a caption from the model. {#c-uc-text}

Counting fails past small numbers.
Parti can reliably produce up to seven objects of the same type, beyond that it is mostly imprecise, and when there are counts of multiple types of entities there is a near complete failure. {#c-pa-count}

Tokenizers lose detail.
Training on the tokens of the dVAE encoder lets DALL-E allocate modeling capacity to low-frequency information, but the heavy compression also renders it unable to produce high-frequency details. {#c-de-blur}

People and faces are harder than scenes.
Imagen's photorealism preference rate rises to 43.6% on COCO prompts with no people, which Saharia et al. take as indicating Imagen's limited ability to generate photorealistic people. {#c-im-coco}

Automatic metrics miss what matters.
FID is not fully aligned with perceptual quality and CLIP is ineffective at counting, so Imagen uses human evaluation to assess image quality and caption similarity. {#c-im-limits}
FID compares image statistics to a reference set and says nothing about whether one image matches its caption; CLIP score rewards images that a contrastive model ties to the caption, and that model has the same binding and counting gaps.

The data carry their own harms.
A recent audit of one of Imagen's data sources, LAION-400M, uncovered a wide range of inappropriate content including pornographic imagery, racist slurs, and harmful social stereotypes. {#c-im-laion}
Models trained on scraped data can reproduce stereotypes, memorize training images and produce realistic images of real people, and the Imagen authors cite these risks as the reason they did not release the model.

## Variants

- **Autoregressive over tokens**: DALL-E with a decoder-only Transformer, Parti with an encoder-decoder; masked token models such as MaskGIT and Muse predict many tokens in parallel instead.
- **Pixel diffusion cascades**: GLIDE and Imagen, with a 64 × 64 base model and diffusion upsamplers.
- **Latent diffusion**: Stable Diffusion and its successors run diffusion or flow matching in an autoencoder's latent space; Esser and colleagues trained such a model with rectified flow at 8B parameters.
- **Two-stage priors**: unCLIP generates a CLIP image embedding first, which also gives image variations and edits by moving in CLIP space.
- **GANs**: before 2021, text-to-image models were mostly generative adversarial networks, such as AttnGAN, DM-GAN and DF-GAN, trained on a single dataset such as MS-COCO.
- **Benchmarks**: prompt sets that probe what COCO does not.
DrawBench contains 11 categories of prompts, testing colors, numbers of objects, spatial relations, text in the scene and unusual interactions between objects, and comprises 200 prompts in total. {#c-im-drawbench}
PartiPrompts is a set of over 1600 English prompts, each associated with one of 12 categories and one of 11 challenge aspects. {#c-pa-p2}
T2I-CompBench consists of 6,000 compositional text prompts from 3 categories, attribute binding, object relationships and complex compositions, and 6 sub-categories. {#c-cb-set}
- **Fine-tuning for composition and preference**: reward-weighted fine-tuning on the model's own well-aligned samples.
GORS fine-tunes Stable Diffusion v2 on generated images that highly align with the compositional prompts, with the fine-tuning loss weighted by the reward, the alignment score between prompt and image. {#c-cb-gors}

## When to use it

Use a pretrained text-to-image model when you need new images from descriptions: illustration, concept art, design drafts, product mock-ups, or synthetic training data for other models.
For most uses a latent diffusion model with released weights is the practical choice, because it runs on one GPU and can be fine-tuned.
Train one from scratch only with hundreds of millions of captioned images and a large compute budget.

Do not use it where the image must be exact: precise counts, layouts, diagrams, readable text or faithful likenesses of real things.
For understanding images rather than making them, use a vision-language model such as those built by visual instruction tuning.

## Pitfalls

- Judging a model by FID on COCO alone. FID tracks image statistics, not alignment, and COCO captions rarely test composition; use prompt sets such as DrawBench, PartiPrompts or T2I-CompBench and human ratings.
- Pushing the guidance weight up without a thresholding or latent-space sampler, which gives oversaturated, unnatural images.
- Forgetting caption dropout in training, which leaves no unconditional model to guide with.
- Conditioning only on a pooled text embedding, which cannot bind attributes to objects; give the generator the sequence of text embeddings.
- Comparing models sampled with different numbers of reranked candidates or guidance weights as if the settings were the same.
- Evaluating with a CLIP-based metric a model that was reranked or guided by the same CLIP model.
- Leaving tokenizer limits unexamined: byte-pair encoding hides spelling from the model, which hurts rendered text.

## History

Text-to-image work before 2021 used GANs and recurrent VAEs trained on single datasets such as MS-COCO and CUB, with captions of a few categories.
In February 2021 DALL-E trained a 12B Transformer on 250 million web pairs and generated images for captions zero-shot.
At the end of 2021 GLIDE found that classifier-free guidance beat CLIP guidance for text, and latent diffusion made training on public data cheap enough for Stable Diffusion, released in August 2022.
In spring 2022 unCLIP, released as DALL·E 2, and Imagen pushed photorealism with diffusion cascades, and Imagen found that a frozen text-only language model was a strong text encoder.
In June 2022 Parti matched Imagen's COCO FID with a 20B autoregressive model, so both families stayed in use.
From 2023, benchmarks such as T2I-CompBench turned attention from photorealism to composition, and later models moved to Transformer backbones and flow matching.

## References
