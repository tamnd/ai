---
tag: DV9G9W
kind: architecture
slug: vision-transformer
name: Vision Transformer
aliases: [ViT, vision transformer, ViT-B/16, ViT-L/16, ViT-H/14, image transformer encoder]
area: deep-learning-architectures/transformers
level: core
summary: >-
  A Transformer encoder applied to an image cut into fixed-size patches, each patch a token, with a class token whose final state is classified; it learns spatial structure from data.
uses: [matrix-multiplication@m2, dot-product@m2, softmax-regression@al, embedding@al]
related_to: [transformer, bert, convolutional-neural-network, resnet, image-classification, imagenet, neural-scaling-laws, rotary-position-embedding, group-normalization, data-augmentation@al, weight-decay@al]
history:
  year: 2020
  people: [Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby]
  note: "Dosovitskiy et al. posted An Image is Worth 16x16 Words to arXiv (2010.11929) in October 2020 and published it at ICLR 2021; Touvron et al. trained it on ImageNet alone in December 2020."
prereqs: [transformer, multi-head-attention, layer-normalization, convolutional-neural-network]
same_as:
  wikipedia: Vision_transformer
  arxiv: ["2010.11929"]
sources:
  - {paper: "Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zhai, Unterthiner, Dehghani, Minderer, Heigold, Gelly, Uszkoreit and Houlsby (2021)", title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", locator: "ICLR 2021; arXiv 2010.11929v2, Sections 3 and 4, Tables 1 to 6 and Appendices A to D", note: "the architecture, its training, and its comparison with ResNets at three scales of pre-training data"}
  - {paper: "Touvron, Cord, Douze, Massa, Sablayrolles and Jégou (2021)", title: "Training data-efficient image transformers & distillation through attention", locator: "arXiv 2012.12877v2, the abstract", note: "ViT-B trained on ImageNet alone"}
  - {paper: "Cordonnier, Loukas and Jaggi (2020)", title: "On the Relationship between Self-Attention and Convolutional Layers", locator: "ICLR 2020; arXiv 1911.03584v2, Theorem 1", note: "what an attention layer can express"}
  - {paper: "Zhai, Kolesnikov, Houlsby and Beyer (2022)", title: "Scaling Vision Transformers", locator: "CVPR 2022; arXiv 2106.04560v2, the abstract", note: "ViT scaled to two billion parameters"}
  - {paper: "He, Chen, Xie, Li, Dollár and Girshick (2021)", title: "Masked Autoencoders Are Scalable Vision Learners", locator: "arXiv 2111.06377v3, the abstract", note: "self-supervised pre-training of ViT"}
  - {paper: "Liu, Lin, Cao, Hu, Wei, Zhang, Lin and Guo (2021)", title: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", locator: "arXiv 2103.14030v2, the abstract", note: "a hierarchical descendant with local windows"}
  - {paper: "Hendrycks and Gimpel (2016)", title: "Gaussian Error Linear Units (GELUs)", locator: "arXiv 1606.08415v5, the abstract", note: "the activation of the MLP"}
  - {url: "https://docs.pytorch.org/docs/2.15/generated/torch.nn.TransformerEncoderLayer.html", title: "torch.nn.TransformerEncoderLayer", locator: "the signature and the parameters", note: "norm_first, the activation and the epsilon of the layer normalization"}
family: transformer
inputs: "a batch of square images as a (B, S, S, C) array whose side S is a multiple of the patch size P"
outputs: "a (B, K) array of class scores, one row per image"
components: [multi-head-attention, multilayer-perceptron, layer-normalization, residual-connection]
inductive_biases: ["the image is a set of P x P patches: pixels interact inside a patch through the linear projection and across patches only through attention", "no locality and no translation equivariance in the attention layers, which are global from the first layer on; the MLP acts on each token alone", "learned position embeddings that start with no knowledge of the 2D grid, so spatial relations between patches are learned from data", "one summary of the image, the final state of a class token that attends to every patch"]
params: "(P^2 C + 1) D + D + (N + 1) D + L (4 D^2 + 2 D D_MLP + 9 D + D_MLP) + 2 D + D K + K for N patches of P x P x C, L layers of width D with an MLP of width D_MLP, and K classes"
flops_per_example: "2 N P^2 C D + L (8 n D^2 + 4 n^2 D + 4 n D D_MLP) + 2 D K for n = N + 1 tokens, counting a multiply-add as 2 FLOPs and only the matrix products"
---

The Vision Transformer (ViT) is a [[transformer|Transformer]] encoder that reads an image as a sequence of patches.
The image is cut into squares of, say, 16 by 16 pixels, each square is flattened and mapped by one linear layer to a vector, and the vectors go through the encoder as the tokens of a sentence would.
Dosovitskiy et al. introduced it in 2020 to test whether the reliance of vision on convolutional networks is necessary, and found that a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. {#c-def}
Pre-trained on 303 million images, their best model reached 88.55% top-1 accuracy on ImageNet, 90.72% on ImageNet-ReaL, 94.55% on CIFAR-100 and 77.63% on the 19 tasks of VTAB. {#c-best}
Trained on ImageNet alone, the same models fell a few points short of ResNets of comparable size, which the authors put down to the inductive biases of convolution that the ViT lacks.
The ViT became the standard backbone of image models, of image-text models such as CLIP, and of the vision encoders of multimodal language models.

## In plain words

A convolutional network looks at an image through small windows and builds up a picture of the whole from many local views, a design that encodes the assumption that nearby pixels matter most and that a cat is a cat wherever it is.
A Vision Transformer makes almost no such assumption.
It chops the image into a grid of tiles, turns each tile into a list of numbers, adds to each list a learned tag saying which tile it was, and lets every tile compare itself with every other tile, layer after layer.
One extra token that belongs to no tile collects what the network needs to name the image.
With few training images, the missing assumptions hurt and a convolutional network of the same size does better; with hundreds of millions of images, the network learns what it needs about space from the data, and the simple design scales further.

## Overview

The model follows the original Transformer as closely as possible, so that scalable NLP Transformer architectures and their efficient implementations can be used almost out of the box. {#c-follow}
An image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ is reshaped into a sequence of $N = HW/P^2$ flattened patches $\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where $(P, P)$ is the resolution of a patch, and $N$ is also the length of the input sequence of the Transformer. {#c-patches}
As with BERT's [class] token, a learnable embedding is put in front of the embedded patches, and its state at the output of the encoder serves as the image representation; the classification head attached to it is an MLP with one hidden layer at pre-training time and a single linear layer at fine-tuning time. {#c-cls}
Position embeddings are added to the patch embeddings to keep the positional information: standard learnable 1D position embeddings, since more advanced 2D-aware position embeddings gave no significant gains. {#c-pos}
The encoder alternates layers of multi-head self-attention and MLP blocks, with layer normalization before every block and residual connections after every block, and the MLP has two layers with a GELU non-linearity. {#c-prenorm}

The network on this page is the ViT as it is fine-tuned: square images of $G \times G$ patches, a class token, learned position embeddings, $L$ encoder layers of width $D$ and a linear head to $K$ classes.
Two switches of the reference implementation drop the class token for an average over the patches (pool="gap") and drop the position embeddings (pos="none"), the two ablations of the paper's Appendix D that matter for the inductive biases below.

## Components

The notation uses row vectors and writes the patches in raster order, row by row from the top left.

**Patch embedding.**
Each patch is flattened to $P^2 C$ numbers and projected to $D$ dimensions; the class token $\mathbf{x}_{\text{class}}$ goes in front, and the position embeddings are added:

$$ \mathbf{z}_0 = \big[\mathbf{x}_{\text{class}};\ \mathbf{x}_p^1 E;\ \mathbf{x}_p^2 E;\ \dots;\ \mathbf{x}_p^N E\big] + E_{\text{pos}}, \qquad E \in \mathbb{R}^{(P^2 C) \times D},\ E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D} . $$ {#eq-patches}

The reference adds a bias to the projection, as a convolution with a $P \times P$ kernel and stride $P$ does; the PyTorch test `test_patch_embedding_is_a_strided_convolution` checks that the two are the same map.
The patches are the only place where the pixels of the image meet a learned weight; after eq-patches the network sees tokens.

**Encoder layer.**
Layer $\ell = 1, \dots, L$ applies [[multi-head-attention|multi-head self-attention]] and an [[multilayer-perceptron|MLP]] to the [[layer-normalization|normalized]] tokens, each inside a [[residual-connection|residual connection]]:

$$ \mathbf{z}'_\ell = \operatorname{MSA}\big(\operatorname{LN}(\mathbf{z}_{\ell-1})\big) + \mathbf{z}_{\ell-1} , $$ {#eq-attn}

$$ \mathbf{z}_\ell = \operatorname{MLP}\big(\operatorname{LN}(\mathbf{z}'_\ell)\big) + \mathbf{z}'_\ell, \qquad \operatorname{MLP}(\mathbf{u}) = \operatorname{GELU}(\mathbf{u} W_1 + \mathbf{b}_1)\, W_2 + \mathbf{b}_2 . $$ {#eq-mlp}

This is the pre-norm layer, with the normalization inside the residual branch, which differs from the post-norm layer of the original Transformer.
The GELU activation function is $x \Phi(x)$, where $\Phi(x)$ is the standard Gaussian cumulative distribution function. {#c-gelu}
Every LN has its own gain and bias, and normalizes each token over its $D$ features.

**Multi-head self-attention.**
For an input $\mathbf{z} \in \mathbb{R}^{n \times D}$, a head computes queries, keys and values and a weighted sum of the values; $k$ heads run in parallel and their concatenated outputs are projected:

$$ [\mathbf{q}, \mathbf{k}, \mathbf{v}] = \mathbf{z} U_{qkv}, \quad A = \operatorname{softmax}\big(\mathbf{q}\mathbf{k}^\top / \sqrt{D_h}\big), \quad \operatorname{SA}(\mathbf{z}) = A \mathbf{v}, \quad \operatorname{MSA}(\mathbf{z}) = [\operatorname{SA}_1(\mathbf{z}); \dots; \operatorname{SA}_k(\mathbf{z})]\, U_{msa} . $$ {#eq-msa}

To keep compute and number of parameters constant when changing the number of heads $k$, $D_h$ is typically set to $D/k$. {#c-dh}
The reference and PyTorch both add biases to the two projections.

**Output.**
The final state of the class token is normalized once more and mapped to the class scores:

$$ \mathbf{y} = \operatorname{LN}(\mathbf{z}_L^0), \qquad \mathbf{s} = \mathbf{y} W_{\text{head}} + \mathbf{b}_{\text{head}}, \qquad p(c \mid \mathbf{x}) = \operatorname{softmax}(\mathbf{s})_c . $$ {#eq-output}

The training loss is the [[al:cross-entropy-loss|cross-entropy]] of the labels averaged over a batch of $B$ images,

$$ \mathcal{L} = -\frac{1}{B} \sum_{b=1}^{B} \log p(c_b \mid \mathbf{x}_b) . $$ {#eq-loss}

At pre-training the head has one hidden layer with tanh as the non-linearity, a design inherited from the Transformer model for text. {#c-head-tanh}
The reference implements it with `head_hidden > 0`.
The paper's first attempt at a classifier on the global average of the patch states performed very poorly, but this was due neither to the extra token nor to the average: the difference is explained by the need for a different learning rate, and with it both work similarly well. {#c-gap}

### The hybrid

As an alternative to raw image patches, the input sequence can come from the feature map of a CNN, with the patch embedding $E$ applied to patches of the feature map; with patches of spatial size 1x1, the sequence is the flattened feature map projected to the Transformer dimension. {#c-hybrid}

### The layer in PyTorch

torch.nn.TransformerEncoderLayer with norm_first=True does layer norm prior to attention and feedforward operations; its activation defaults to relu and can be the string "gelu", and the epsilon of its layer normalization defaults to 1e-5. {#c-torch-tel}
`impl/pytorch/vit.py` builds the torch layer with activation="gelu", norm_first=True, batch_first=True, dropout 0 and the epsilon of this page, $10^{-6}$, and the test `test_layer_matches_torch_transformer_encoder_layer` checks that one encoder layer of the reference and the torch layer agree to $10^{-12}$.

## Shapes and parameters

The patch embedding has $(P^2 C + 1) D$ parameters, the class token $D$ and the position embeddings $(N + 1) D$.
An encoder layer has $4D^2 + 4D$ in the attention, $2 D D_{\text{MLP}} + D_{\text{MLP}} + D$ in the MLP and $4D$ in its two normalizations, and the head has $2D$ in the final normalization and $DK + K$ in the linear layer, so the network has

$$ N_{\text{params}} = (P^2 C + 1) D + D + (N + 1) D + L\big(4D^2 + 2 D D_{\text{MLP}} + 9D + D_{\text{MLP}}\big) + 2D + DK + K $$ {#eq-params}

parameters. {#c-params}
In the shape contract this is

{shape:params}

and the example network, $4 \times 4$ patches of $2 \times 2 \times 3$ pixels, width 8 with 2 heads, an MLP of 16, 2 layers and 5 classes, has {shape:params@example} parameters.

The paper bases its configurations on those of [[bert|BERT]]: ViT-Base has 12 layers, hidden size 768, MLP size 3072, 12 heads and 86M parameters; ViT-Large 24 layers, 1024, 4096, 16 heads and 307M; ViT-Huge 32 layers, 1280, 5120, 16 heads and 632M. {#c-table1}
A name such as ViT-L/16 gives the size and the patch size, and since the sequence length is inversely proportional to the square of the patch size, models with a smaller patch are more expensive. {#c-naming}
With a linear head to 1,000 classes at 224 by 224 pixels, eq-params gives {shape:params@vit_b16} for ViT-B/16, {shape:params@vit_l16} for ViT-L/16, {shape:params@vit_l32} for ViT-L/32 and {shape:params@vit_h14} for ViT-H/14. {#c-params-models}
Table 1 does not say which patch size or head its counts are for; 86M and 632M match B/16 and H/14, and 307M is within 1% of both Large models.

Almost all the parameters are in the encoder, whose size does not depend on the image: {shape:params_encoder@vit_b16} of the {shape:params@vit_b16} of ViT-B/16, or 98.3%. {#c-encoder-share}
Only $E_{\text{pos}}$ grows with the number of patches, so ViT-B/16 fine-tuned at 384 pixels, with $24 \times 24 = 576$ patches, has {shape:params@vit_b16_384} parameters, 291,840 more than at 224.
The derivation `parameter-count` proves these counts.

## Computation and cost

The products of a forward pass over $B$ images, with $n = N + 1$ tokens, cost

$$ C = B\Big[2 N P^2 C D + L\big(8 n D^2 + 4 n^2 D + 4 n D D_{\text{MLP}}\big) + 2 D K\Big] $$ {#eq-flops}

FLOPs, counting a multiply-add as two and leaving out the softmax, the normalizations and the GELU. {#c-flops}
The term $8nD^2$ is the four projections of the attention, $4n^2 D$ the products $\mathbf{q}\mathbf{k}^\top$ and $A\mathbf{v}$, and $4 n D D_{\text{MLP}}$ the MLP.
The example network on a batch of 2 images costs {shape:flops_forward@example} FLOPs; ViT-B/16 costs {shape:flops_forward@vit_b16} per image at 224 pixels and {shape:flops_forward@vit_b16_384} at 384.

**The quadratic term is small.**
In each layer the share of the FLOPs that grows with $n^2$ is $n / (n + 2D + D_{\text{MLP}})$: 4.1% for ViT-B/16 at 224 pixels, where $n = 197$, and 11.1% at 384, where $n = 577$. {#c-attn-share}
The cost of the network is therefore close to linear in the number of patches at these sizes, and so close to quadratic in the side of the image; halving the patch size multiplies the number of tokens and the FLOPs by about four, as from ViT-B/32 with {shape:flops_forward@vit_b32} to ViT-B/16.
The paper measured that the theoretical bi-quadratic scaling of ViT with image size only barely starts to appear for large models at high resolutions. {#c-biquadratic}

**The compute of pre-training.**
The paper's Table 6 lists the pre-training compute of each model in exaFLOPs: 55 for ViT-B/32, 224 for ViT-B/16, 196 for ViT-L/32 and 783 for ViT-L/16 trained for 7 epochs, and 1567 for ViT-L/16 and 4262 for ViT-H/14 trained for 14 epochs. {#c-table6}
JFT, the pre-training set, has 18k classes and 303M high-resolution images. {#c-jft}
Taking a training step as three forward passes, one forward and two for the backward pass, $3 \times C \times \text{epochs} \times 303\text{M}$ at 224 pixels reproduces every entry of that column to within 2.1%, and all but ViT-B/32 to within 0.3%. {#c-table6-check}
The derivation `compute` proves eq-flops and computes these figures.

**Memory.**
The paper also measured the largest batch that fits on a TPUv3 core: large ViT models have a clear advantage in memory-efficiency over ResNet models. {#c-memory}

## Training

All models of the paper, ResNets included, are trained with Adam with $\beta_1 = 0.9$, $\beta_2 = 0.999$, a batch size of 4096 and a high weight decay of 0.1, with a linear learning rate warmup and decay. {#c-train}
The training resolution is 224, the warmup lasts 10k steps, and on ImageNet gradient clipping at global norm 1 helped. {#c-train-res}
Strong regularization is the key to training from scratch on ImageNet; dropout, when used, comes after every dense layer except the qkv-projections and directly after adding the positional to the patch embeddings. {#c-dropout}

**Fine-tuning.**
To fine-tune, the pre-trained prediction head is removed and a zero-initialized $D \times K$ feedforward layer is attached, where $K$ is the number of downstream classes. {#c-zero-head}
When transferring, the whole head of two linear layers is replaced by a single zero-initialized linear layer, which the authors found a little more robust than re-initializing only the last layer. {#c-head-robust}
Fine-tuning uses SGD with momentum, batch size 512. {#c-sgd}
The default fine-tuning resolution is 384. {#c-ft384}

**Higher resolution.**
Fine-tuning at a higher resolution than pre-training is often beneficial; the patch size stays the same, so the sequence is longer, and the pre-trained position embeddings are resized by 2D interpolation according to their location in the original image. {#c-interp}
The paper notes that this resolution adjustment and the patch extraction are the only points at which an inductive bias about the 2D structure of images is put into the model by hand. {#c-only-2d}
`resize_pos` in the reference interpolates bilinearly between the centres of the old patches, and the PyTorch test `test_resize_pos_matches_torch_interpolate` checks it against torch.nn.functional.interpolate.

**Training on less data.**
Touvron et al. produced competitive convolution-free transformers by training on ImageNet only, on a single computer in less than 3 days; their reference vision transformer, with 86M parameters, reached 83.1% top-1 accuracy (single-crop) on ImageNet with no external data. {#c-deit}
Their recipe adds strong data augmentation and regularization and a distillation token through which the student learns from a teacher, most effectively a convnet.

**Self-supervision.**
The paper tried masked patch prediction, after BERT: with this self-supervised pre-training, ViT-B/16 reaches 79.9% accuracy on ImageNet, 2% better than training from scratch but still 4% behind supervised pre-training. {#c-selfsup}
He et al. later masked a high proportion of the patches, 75%, and reconstructed the missing pixels with an encoder that sees only the visible patches; with it, a vanilla ViT-Huge reaches 87.8%, the best accuracy among methods that use only ImageNet-1K data. {#c-mae}

## Inductive biases

The Vision Transformer has much less image-specific inductive bias than CNNs: in ViT only the MLP layers are local and translationally equivariant, while the self-attention layers are global, and the two-dimensional neighbourhood structure is used only when the image is cut into patches and when the position embeddings are adjusted for a new resolution. {#c-less-bias}
The position embeddings at initialization carry no information about the 2D positions of the patches, so all spatial relations between patches have to be learned from scratch. {#c-learned-space}

**Without position embeddings the patches are a bag.**
Layer normalization and the MLP act on each token alone, attention without positions is equivariant to any reordering of its tokens, and the class token or the average sees all patches the same way.
So a ViT without position embeddings gives the same scores for an image and for any rearrangement of its patches. {#c-perm}
It cannot tell an image from its mirror image at the level of patches, and on a task whose label depends on where a patch is, it can do no better than chance on pairs of an image and its rearrangement.
The derivation `permutation` proves this, the NumPy test `test_no_positions_means_a_bag_of_patches` checks it on random networks, and the experiment below trains the two networks on such a task.
In the paper's ablation, ViT-B/16 reaches 0.61382 on ImageNet 5-shot linear without position embeddings, 0.64206 with 1-D position embeddings, 0.64001 with 2-D ones and 0.64032 with relative position embeddings. {#c-table8}
The authors speculate that the way of encoding position matters little because the encoder works on patches, a 14 × 14 grid as opposed to 224 × 224 pixels, where learning the spatial relations is equally easy for these strategies. {#c-why-1d}

**What the positions learn.**
The learned position embeddings encode distance within the image: closer patches tend to have more similar position embeddings, and patches in the same row or column have similar embeddings. {#c-pos-learned}
That the embeddings learn the 2D image topology explains, for the authors, why hand-crafted 2D-aware embeddings do not help. {#c-pos-explains}

**Global attention from the first layer.**
Some heads attend to most of the image already in the lowest layers, while others have consistently small attention distances there, and the attention distance increases with network depth. {#c-attn-distance}
The localized heads are less pronounced in hybrid models, which suggests that they serve a function similar to the early convolutional layers of CNNs. {#c-local-heads}

**Attention can express convolution.**
Cordonnier et al. prove that a multi-head self-attention layer with $N_h$ heads of dimension $D_h$, output dimension $D_{out}$ and a relative positional encoding of dimension $D_p \geq 3$ can express any convolutional layer of kernel size $\sqrt{N_h} \times \sqrt{N_h}$ and $\min(D_h, D_{out})$ output channels. {#c-cordonnier}
A ViT layer with 9 heads could thus compute a 3 × 3 convolution over the patch grid; it is the data that decides whether it does, which is the trade the ViT makes against a [[convolutional-neural-network|CNN]], where locality is fixed in advance.

## Evidence

**The comparison with BiT.**
Pre-trained on JFT-300M, ViT-H/14 reaches 88.55 on ImageNet, 90.72 on ImageNet ReaL, 99.50 on CIFAR-10, 94.55 on CIFAR-100, 97.56 on Oxford-IIIT Pets, 99.68 on Oxford Flowers-102 and 77.63 on VTAB, for 2.5k TPUv3-core-days of pre-training; BiT-L, a ResNet152x4 pre-trained on the same data, reaches 87.54, 90.54, 99.37, 93.51, 96.62, 99.63 and 76.29 for 9.9k. {#c-table2}
ViT-L/16, pre-trained on JFT-300M in 0.68k TPUv3-core-days, reaches 87.76 on ImageNet; pre-trained on the public ImageNet-21k in 0.23k, it reaches 85.30. {#c-table2-l16}
The ImageNet-21k model could be trained with a standard cloud TPUv3 of 8 cores in about 30 days. {#c-30days}
These are the means of three fine-tuning runs, each from one pre-training run.

**How much data.**
When pre-trained on ImageNet, the smallest dataset, ViT-Large models underperform ViT-Base models despite moderate regularization; with ImageNet-21k their performances are similar, and only with JFT-300M do the larger models give their full benefit. {#c-data-size}
Fine-tuned on ImageNet, ViT-B/16 reaches 77.91 after pre-training on ImageNet, 83.97 after ImageNet-21k and 84.15 after JFT-300M, and ViT-L/16 reaches 76.53, 85.15 and 87.12. {#c-table5}
On random subsets of JFT, Vision Transformers overfit more than ResNets of comparable computational cost: ViT-B/32 is slightly faster than ResNet50, performs much worse on the 9M subset, but better on 90M and more. {#c-subsets}
The authors read this as support for the intuition that the convolutional inductive bias is useful for smaller datasets, while for larger ones learning the relevant patterns directly from data is sufficient, even beneficial. {#c-intuition}

**Compute.**
In a controlled study of transfer from JFT-300M, ViT uses about 2 − 4× less compute than ResNets to reach the same performance, averaged over 5 datasets; hybrids slightly outperform ViT at small computational budgets, but the difference vanishes for larger models; and Vision Transformers appear not to saturate within the range tried. {#c-scaling}

**Shape of the network.**
In an ablation around a ViT of 8 layers, D = 1024, D_MLP = 2048 and a patch size of 32, scaling the depth gives the biggest improvements, clearly visible up to 64 layers but with diminishing returns after 16; scaling the width gives the smallest changes; and decreasing the patch size gives robust improvements without adding parameters. {#c-shape}

**At larger scale.**
Zhai et al. trained a ViT model with two billion parameters which reaches 90.45% top-1 accuracy on ImageNet. {#c-zhai}
They set out to measure how Vision Transformers scale with compute, model size and data, as the [[neural-scaling-laws|scaling laws]] of language models had been measured.

These are reported results; this page has not reproduced them.
Every comparison with ResNets in the paper uses ResNets with group normalization and standardized weights, BiT, trained with the same optimizer, and the ViT advantage it reports is at the scale of JFT-300M, a dataset that is not public.

## Limitations

**Data.**
Trained on a mid-sized dataset such as ImageNet without strong regularization, ViT models reach accuracies a few percentage points below ResNets of comparable size, which the authors attribute to the inductive biases of CNNs, translation equivariance and locality, that Transformers lack. {#c-small-data}
Touvron et al. closed most of the gap with augmentation, regularization and distillation, but the result is a training recipe more sensitive to its settings than that of a ResNet.

**Cost at high resolution.**
Naive self-attention over pixels would let each pixel attend to every other one, a quadratic cost in the number of pixels that does not scale to realistic input sizes. {#c-pixels}
Patches make the sequence $P^2$ times shorter, but the cost still grows with the square of the number of patches, which matters for detection and segmentation at high resolution, and the single-scale output is not the feature pyramid those tasks use.
Liu et al.'s Swin Transformer limits self-attention to non-overlapping local windows that shift between layers, a hierarchical design with computational complexity linear in the image size. {#c-swin}

**Open questions in the paper.**
The paper lists as challenges the application of ViT to other vision tasks such as detection and segmentation, and the large gap between self-supervised and large-scale supervised pre-training. {#c-challenges}

**Resolution.**
A ViT expects the number of patches it was trained with: a new resolution needs resized position embeddings and usually fine-tuning, and the side of the image must be a multiple of the patch size or be cropped or padded to one.

## Variants and descendants

- **Hybrid ViT.** A ResNet stem whose feature map is cut into $1 \times 1$ patches; better than ViT at small compute, no better at large compute.
- **DeiT** (Touvron et al., 2021). The ViT trained on ImageNet alone, with a distillation token.
- **Masked autoencoders** (He et al., 2021). Self-supervised pre-training of a ViT encoder that sees a quarter of the patches.
- **Swin Transformer** (Liu et al., 2021). Windows of local attention, shifted between layers, and patch merging into a hierarchy of resolutions.
- **ViT-G and larger** (Zhai et al., 2022). The ViT scaled to billions of parameters with a refined training recipe.
- **Image-text encoders.** The image towers of contrastive image-text models such as CLIP are ViTs, and so are the vision encoders of many multimodal language models.
- **Position encodings.** Learned 1D embeddings are the default; 2D, relative and [[rotary-position-embedding|rotary]] encodings are common in later models.

## Notable models

- **ViT-B/16, ViT-L/16 and ViT-H/14**, the models of the paper, pre-trained on ImageNet-21k or JFT-300M; ViT-H/14 was the most accurate.
- **DeiT-B**, the 86M ViT-B/16 trained on ImageNet only.
- **ViT-G/14**, the two-billion-parameter model of Zhai et al.

## Implementations

The NumPy reference `impl/numpy/vit.py` implements eq-patches, eq-attn with eq-msa, eq-mlp with the exact GELU, eq-output with a linear or a tanh head, the average pooling and the network without positions, eq-loss and its gradient by hand through every layer down to the pixels, the resizing of the position embeddings, and the counts eq-params and eq-flops.
The PyTorch version `impl/pytorch/vit.py` is an `nn.Module` with the same parameters; the tests compare one encoder layer with torch.nn.TransformerEncoderLayer, the patch embedding with a strided torch.nn.functional.conv2d and the resizing with torch.nn.functional.interpolate, run `torch.autograd.gradcheck`, and train the network on a task that needs the positions.
The JAX version `impl/jax/vit.py` is a pure function of a parameter dictionary, tested under `jax.jit` and `jax.vmap` and with `jax.grad` against the gradients of the NumPy backward pass.
All three load the shared fixture of `fixtures/`: the example network on two random images, its scores, loss and gradients.

### Worked example

Take a $4 \times 4$ image with one channel whose pixels are $1, 2, \dots, 16$ row by row, and patches of $2 \times 2$.
There are $N = 16/4 = 4$ patches, and in raster order, each flattened row by row, they are $(1, 2, 5, 6)$, $(3, 4, 7, 8)$, $(9, 10, 13, 14)$ and $(11, 12, 15, 16)$.
With $D = 1$ and a projection $E$ that averages the four pixels, the patch embeddings are $3.5$, $5.5$, $11.5$ and $13.5$, and with the class token the encoder reads 5 tokens.
Mirror the image at the level of patches, swapping the left and right columns of patches: the patches come in the order 2, 1, 4, 3, the same four vectors in a different order, and a network without position embeddings gives the same output for both images.
For ViT-B/16 at 224 pixels the same arithmetic gives $N = 14^2 = 196$ patches of $16 \cdot 16 \cdot 3 = 768$ numbers, 197 tokens, and at 384 pixels $24^2 = 576$ patches and 577 tokens, with the $14 \times 14$ grid of position embeddings interpolated to $24 \times 24$.
The test `test_worked_example` checks these numbers.

### Pitfalls

> [!pitfall]
> **Flattening the patches in a different order from the weights.** A patch of $P \times P \times C$ can be flattened as (row, column, channel), as here and in a kernel stored height, width, input, output, or as (channel, row, column), as a PyTorch convolution weight of shape (D, C, P, P) is. Loading weights from one layout into the other gives a network that runs and is wrong. `embed_as_conv()` converts between them, and `test_patch_embedding_is_a_strided_convolution` checks the result.

> [!pitfall]
> **Using the defaults of torch.nn.TransformerEncoderLayer.** Its defaults are a post-norm layer with ReLU, dropout 0.1 and an epsilon of 1e-5; a ViT layer is pre-norm with GELU. With norm_first=False the network is a different function, and loading ViT weights into it gives wrong scores without an error.

> [!pitfall]
> **Changing the resolution without resizing the position embeddings.** At a new resolution the number of patches changes and $E_{\text{pos}}$ no longer has a row for each token. Truncating or tiling it puts the embedding of one place on a patch somewhere else. Resize the grid part in two dimensions and keep the class token's row, as `resize_pos` does; `test_resize_pos` checks that resizing to the same grid is the identity and that a field linear in the row and column is sampled exactly at the centres of the new patches.

> [!pitfall]
> **Dropping the position embeddings.** Without them the network cannot distinguish an image from any rearrangement of its patches. `test_no_positions_means_a_bag_of_patches` checks this exactly, and the experiment below measures what it costs on a task that depends on position.

> [!pitfall]
> **Re-initializing only the last layer of the pre-training head.** The pre-training head has a tanh hidden layer that the fine-tuned model does not; the paper removes the whole head and attaches a zero-initialized linear layer.

## Experiments

This page has no experiment yet.
The PyTorch test `test_the_positions_are_needed` is a small one.
Each image is $8 \times 8$ pixels with one channel and Gaussian noise of standard deviation 0.1, cut into a $4 \times 4$ grid of $2 \times 2$ patches, with a bright patch at a random place; the label is whether that patch is in the left half of the grid.
Two ViTs of width 16 with 2 heads, an MLP of 32 and 2 layers, one with learned position embeddings and one without, are trained for 300 steps of Adam on batches of 64.
The test set is 500 images and their mirror images at the level of patches, whose labels are the opposite.
The network with position embeddings classifies more than 95% of these 1000 images correctly; the one without gives the same scores to each image and its mirror to within $10^{-9}$, and so gets exactly one of each pair right, an accuracy of 50%.
The checks of the three derivations count the parameters and FLOPs of the models of the paper, recompute the compute column of Table 6, and permute the patches of random networks.

## History

Self-attention had been used on images before, either together with convolutions or in local neighbourhoods and other restricted patterns that need complex engineering to run efficiently on hardware accelerators. {#c-hist-local}
Closest to the ViT is the model of Cordonnier et al. (2020), which extracts patches of size 2 × 2 from the input image and applies full self-attention on top; its small patches make it applicable only to small-resolution images. {#c-hist-cordonnier}
Image GPT applied Transformers to image pixels after reducing the resolution and the colour space, trained them as generative models, and reached at most 72% on ImageNet. {#c-hist-igpt}
Dosovitskiy et al.'s paper, arXiv 2010.11929, was published as a conference paper at ICLR 2021. {#c-hist-iclr}
Within months, Touvron et al. trained the same architecture on ImageNet alone, and in 2021 Liu et al. gave it a hierarchy of local windows and He et al. a masked-autoencoder pre-training; in 2022 Zhai et al. scaled it to two billion parameters.

## References

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J. and Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021. arXiv 2010.11929.
- Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A. and Jégou, H. (2021). Training data-efficient image transformers & distillation through attention. arXiv 2012.12877.
- Cordonnier, J.-B., Loukas, A. and Jaggi, M. (2020). On the Relationship between Self-Attention and Convolutional Layers. ICLR 2020. arXiv 1911.03584.
- Zhai, X., Kolesnikov, A., Houlsby, N. and Beyer, L. (2022). Scaling Vision Transformers. CVPR 2022. arXiv 2106.04560.
- He, K., Chen, X., Xie, S., Li, Y., Dollár, P. and Girshick, R. (2021). Masked Autoencoders Are Scalable Vision Learners. arXiv 2111.06377.
- Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S. and Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. arXiv 2103.14030.
- Hendrycks, D. and Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). arXiv 1606.08415.
- PyTorch contributors. torch.nn.TransformerEncoderLayer. PyTorch 2.15 documentation.
