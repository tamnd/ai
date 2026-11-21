# The arithmetic of visual instruction tuning

This derivation backs the mathematical claims of the visual instruction tuning page. The check file
`vision-language-arithmetic.check.py` tests every number below.

## Setting

A vision-language model of this family has three parts: a pretrained image encoder g that turns an image into a grid
of feature vectors Z, a connector that turns Z into vectors the width of the language model's word embeddings, and a
pretrained language model that reads those vectors as if they were tokens and generates text. LLaVA's connector is a
matrix, H = W Z. BLIP-2's connector is a small Transformer with learned queries, followed by a linear layer. Flamingo's
connector is a Perceiver Resampler, and its language model reads the visual tokens through gated cross-attention layers
instead of through its input sequence.

## 1. An always-yes model on a balanced POPE split

POPE asks yes-or-no questions of the form "Is there a <object> in the image?". Half of the questions name an object
that is in the image and half name one that is not. Take the yes answer as the positive class. A model that answers
yes to every question gets every positive right and every negative wrong. On n questions, n/2 are true positives and
n/2 are false positives, with no false negatives:

accuracy = (n/2) / n = 0.5,
precision = TP / (TP + FP) = (n/2) / n = 0.5,
recall = TP / (TP + FN) = (n/2) / (n/2) = 1,
F1 = 2 · 0.5 · 1 / (0.5 + 1) = 2/3.

This is 50.00 accuracy and 66.67 F1, the numbers POPE reports for MultiModal-GPT on its popular and adversarial
splits, where that model answered yes 100% of the time. An F1 near 66.67 with a yes rate near 100% is therefore the
mark of a model that ignores the question. On these splits F1 alone hides that, so the yes rate should be read next to
it.

## 2. How many tokens an image costs

A Vision Transformer with 14-pixel patches cuts a 224 × 224 image into (224/14)² = 16² = 256 patches. With the class
token that is 257 vectors, the 257 × 1024 feature map BLIP-2 quotes for ViT-L/14. At 336 pixels the grid is 24 × 24 =
576 patches. LLaVA's linear projection keeps one token per patch, so the language model reads 256 visual tokens per
image at 224 pixels and 576 at 336 pixels, which LLaVA-1.5 uses.

Connectors with learned queries cut this to a fixed number, whatever the grid size: 32 for the BLIP-2 Q-Former and 64
for the Flamingo Perceiver Resampler. For a prompt with five images, the linear connector at 336 pixels puts 5 × 576 =
2,880 visual tokens in the context. Five sets of 32 queries cost 160 tokens, and five sets of 64 resampler outputs cost
320. Self-attention cost grows with the square of the sequence length, so the gap between the two is wider in compute
than in tokens. With 2,880 visual tokens against 160, the visual part of the attention matrix is (2880/160)² = 324
times larger.

## 3. The tanh gate is the identity at initialization

A Flamingo gated block computes

y' = y + tanh(α) · f(y, x),

where f is a new cross-attention or feed-forward layer, x holds the visual tokens and α is one learned scalar per layer,
initialized to 0. Since tanh(0) = 0, y' = y: before training, the model is exactly the frozen language model.

Let g = ∂L/∂y' be the gradient arriving from above. Then

∂L/∂α = (1 − tanh²(α)) · ⟨g, f(y, x)⟩, which at α = 0 is ⟨g, f(y, x)⟩,

and for any weight θ inside f,

∂L/∂θ = tanh(α) · ⟨g, ∂f/∂θ⟩, which at α = 0 is 0.

The first step therefore moves only the gates. The new layer's weights start to learn once α leaves 0. Each gate opens
toward the direction where its untrained layer happens to help, and the size of the opening controls how much the new
pathway can disturb the language model. Removing the gate makes the new layers add random vectors to the residual stream
from the first step. Flamingo reports that this costs 4.2% in its overall score and destabilises training.

## 4. The trainable-parameter ratio

The BLIP-2 Q-Former has 188 million parameters, and Flamingo-80B has 10.2 billion trainable parameters. The ratio is
10.2 × 10⁹ / 188 × 10⁶ = 54.26, which rounds to the "54x" BLIP-2 quotes. The ratio counts trainable parameters only.
Both systems also run frozen networks of many billions of parameters at every step.

## 5. The LLaVA-Instruct-158K split

The three kinds of response in LLaVA-Instruct are 58K conversations, 23K detailed descriptions and 77K complex
reasoning answers. 58 + 23 + 77 = 158, so the subsets account for the whole set, with about 37% conversation, 15%
description and 49% reasoning by count.

## 6. Training compute

LLaVA-1.5 trains in about 6 hours of alignment pretraining and about 20 hours of instruction tuning on 8 A100 GPUs,
26 × 8 = 208 A100-hours. The largest Flamingo run used 1536 TPU chips for 15 days, 1536 × 15 × 24 = 552,960 chip-hours.
Counting one device-hour on each side, the ratio is 552,960 / 208 = 2,658. The two devices differ in speed, so this is a
ratio of device-time, not of floating-point operations. It still conveys the main point: a frozen image encoder, a frozen or
lightly tuned language model and a connector of a few million weights move the training cost from a data-centre run to
one machine for a day.

## 7. Projection parameters

The CLIP ViT-L encoder produces 1024-wide features, and Vicuna-13B has a 5120-wide embedding. A linear projection needs
1024 × 5120 = 5,242,880 weights. A two-layer MLP with a 5120-wide hidden layer needs 1024 × 5120 + 5120 × 5120 =
5,242,880 + 26,214,400 = 31,457,280 weights, ignoring biases. That is 6 times the linear map, and 31.46 million /
13 billion = 0.242% of the language model. In stage 1 of LLaVA these weights are all that train.
