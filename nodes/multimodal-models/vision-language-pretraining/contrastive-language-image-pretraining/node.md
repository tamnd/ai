---
tag: NEW-contrastive-language-image-pretraining
kind: method
slug: contrastive-language-image-pretraining
name: Contrastive language-image pretraining
aliases: [CLIP, contrastive image-text pretraining, ALIGN, SigLIP, sigmoid loss for language-image pretraining, OpenCLIP, ConVIRT, image-text contrastive learning, dual encoder, zero-shot image classification]
area: multimodal-models/vision-language-pretraining
level: core
summary: >-
  Training an image encoder and a text encoder so that each image lands near its own caption and away from the other captions in the batch, so class names written as text can classify images.
purpose: learning image representations from images paired with free text, so that a new task can be specified by writing class descriptions instead of labelling images
uses: [transformer, vision-transformer, resnet, cross-entropy-loss@al, softmax-regression@al, logistic-loss@al, embedding@al]
related_to: [bert, imagenet, neural-scaling-laws, prompt-sensitivity, k-nearest-neighbors@al]
hyperparameters:
  - {name: batch size, meaning: "the number N of image-text pairs per step, which is also the number of candidates each image must pick its caption from", typical: "32,768 for CLIP; 16,384 for ALIGN; about 32k for SigLIP, where gains saturated"}
  - {name: logit scale, meaning: "the inverse temperature s that multiplies the cosine similarities before the softmax or sigmoid", typical: "learned; initialized at 1/0.07 and capped at 100 in CLIP; initialized at 10 in SigLIP"}
  - {name: logit bias, meaning: "a learned offset b added to every logit in the sigmoid loss, so that the many negatives do not dominate the start of training", typical: "initialized at −10 in SigLIP; absent in the softmax loss"}
  - {name: encoders, meaning: "the image and text networks and the width of the shared embedding", typical: "ResNet or Vision Transformer images and a Transformer for text in CLIP; EfficientNet-L2 and BERT-Large in ALIGN"}
  - {name: data, meaning: "the number and the source of image-text pairs and how much they are filtered", typical: "400 million pairs found by 500,000 queries for CLIP; over one billion alt-text pairs with frequency filtering for ALIGN; LAION-400M and LAION-2B for OpenCLIP"}
  - {name: prompt templates, meaning: "the sentences that turn class names into captions for zero-shot classification", typical: "\"A photo of a {label}.\"; 80 templates averaged for ImageNet in CLIP"}
cost:
  train: "12 days on 256 V100 GPUs for the CLIP ViT-L/14 and 18 days on 592 V100 GPUs for RN50x64, at 32 epochs over 400 million pairs; a SigLiT model with a frozen pretrained image tower reached 84.5% zero-shot ImageNet accuracy on four TPUv4 chips in two days"
  inference: "one image-encoder pass per image and one text-encoder pass per caption; a zero-shot classifier is computed once from the class captions and cached, so classifying an image costs one encoder pass and K dot products"
history:
  year: 2021
  people: [Alec Radford, Jong Wook Kim, Chao Jia, Yinfei Yang, Yuhao Zhang, Xiaohua Zhai, Lucas Beyer]
  note: "Zhang et al. posted ConVIRT for medical images and reports (October 2020); Radford et al. posted CLIP (February 2021) and Jia et al. ALIGN (February 2021, ICML 2021); Liang et al. described the modality gap (March 2022, NeurIPS 2022); Cherti et al. fitted scaling laws with OpenCLIP on LAION (December 2022); Zhai et al. replaced the softmax with a sigmoid loss in SigLIP (March 2023)."
prereqs: [transformer, cross-entropy-loss@al, embedding@al]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["2103.00020", "2102.05918", "2303.15343", "2212.07143", "2010.00747", "2203.02053"]
sources:
  - {paper: "Radford, Kim, Hallacy, Ramesh, Goh, Agarwal, Sastry, Askell, Mishkin, Clark, Krueger and Sutskever (2021)", title: "Learning Transferable Visual Models From Natural Language Supervision", locator: "arXiv 2103.00020v1, pp. 1 to 9, 14 and 19 to 20", note: "the CLIP objective, the WIT dataset, training details, prompt engineering, zero-shot results, robustness and limitations"}
  - {paper: "Jia, Yang, Xia, Chen, Parekh, Pham, Le, Sung, Li and Duerig (2021)", title: "Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision", locator: "ICML 2021, arXiv 2102.05918v2, pp. 1 to 4", note: "ALIGN, its noisy data, encoders, batch size and results"}
  - {paper: "Zhai, Mustafa, Kolesnikov and Beyer (2023)", title: "Sigmoid Loss for Language Image Pre-Training", locator: "arXiv 2303.15343v4, pp. 1 to 4", note: "the sigmoid loss, its bias initialization, memory and batch-size results"}
  - {paper: "Cherti, Beaumont, Wightman, Wortsman, Ilharco, Gordon, Schuhmann, Schmidt and Jitsev (2022)", title: "Reproducible scaling laws for contrastive language-image learning", locator: "arXiv 2212.07143v2, pp. 1, 2 and 7", note: "power laws for OpenCLIP on LAION and the effect of the training distribution"}
  - {paper: "Zhang, Jiang, Miura, Manning and Langlotz (2022)", title: "Contrastive Learning of Medical Visual Representations from Paired Images and Text", locator: "Machine Learning for Healthcare 2022, arXiv 2010.00747v2, pp. 3, 6 and 7", note: "the bidirectional contrastive objective of ConVIRT and its label efficiency"}
  - {paper: "Liang, Zhang, Kwon, Yeung and Zou (2022)", title: "Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning", locator: "NeurIPS 2022, arXiv 2203.02053v2, pp. 1, 2 and 7", note: "the modality gap, the cone effect and the role of the temperature"}
---

Contrastive language-image pretraining trains two networks, one for images and one for text, on pairs of an image and the text found with it on the web.
Each network maps its input to a unit vector in a shared space, and the training objective asks each image to be closer to its own caption than to every other caption in the batch, and each caption closer to its own image.
After training, a task such as "is this a dog, a cat or a car" is solved by embedding a caption for each class and picking the class whose caption is closest to the image.
CLIP, from OpenAI, and ALIGN, from Google, introduced the method at scale in 2021, and its encoders became the standard image and text backbones of later multimodal systems.

## In plain words

Imagine a pile of photos and a pile of captions, shuffled.
The training game is to match them up.
A model that can do this for hundreds of millions of web photos has to learn what dogs, cars, sunsets and handwriting look like, and which words describe them, because the captions are the only clue.

The two encoders never see a class label.
They see a batch of tens of thousands of pairs at once and are scored on how well each image picks out its own caption from the whole batch.

The result can be used without further training.
To build a classifier for any set of classes, write a sentence for each class, such as "a photo of a dog", embed the sentences, and assign each image to the nearest sentence.
Writing the classes in words is what "zero-shot" means here: no labelled image of the new task is needed.

## Setting

The data are pairs (image, text), usually the image and its alt-text or caption on a web page, with no other labels.
Given a batch of N (image, text) pairs, CLIP is trained to predict which of the N × N possible pairings across a batch actually occurred, by maximizing the cosine similarity of the image and text embeddings of the N real pairs while minimizing the cosine similarity of the embeddings of the N² − N incorrect pairings, and it optimizes a symmetric cross entropy loss over these similarity scores. Given a batch of N (image, text) pairs, CLIP is trained to predict which of the N × N possible pairings across a batch actually occurred, by maximizing the cosine similarity of the image and text embeddings of the N real pairs while minimizing the cosine similarity of the embeddings of the N² − N incorrect pairings, and it optimizes a symmetric cross entropy loss over these similarity scores. {#c-clip-batch}
The two encoders can be any networks that map an input to a vector; CLIP used a ResNet or a Vision Transformer for images and a Transformer for text, and ALIGN used an EfficientNet and BERT.
Each vector is projected to a shared width and normalized to unit length, so that the dot product of an image vector and a text vector is their cosine, an embedding similarity in [−1, 1].

The method sits between two older ideas.
ConVIRT applied the same objective to chest X-rays and their radiology reports:
ConVIRT learns from medical images paired with report text using a bidirectional contrastive objective: an image-to-text loss of the same form as the InfoNCE loss, with cosine similarity and a temperature parameter τ, and a text-to-image loss, combined with a scalar weight λ in [0, 1]. {#c-cv-loss}
Earlier caption models predicted the words of the caption, which is harder: the derivation below and CLIP's own comparison both favour matching over generating.

## The method

The simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet, and after pre-training natural language is used to reference learned visual concepts, enabling zero-shot transfer of the model to downstream tasks. The simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet, and after pre-training natural language is used to reference learned visual concepts, enabling zero-shot transfer of the model to downstream tasks. {#c-clip-task}

**Training.**
For a batch, compute the image vectors u_1 … u_N and text vectors v_1 … v_N, form the N × N matrix of logits s · u_i · v_j, and apply a softmax cross-entropy loss along each row (each image picks a caption) and along each column (each caption picks an image), as in softmax regression with the other captions of the batch as the classes.
The loss is the average of the two.
The scale s, the inverse of a temperature, is a learned parameter:
The learnable temperature parameter τ was initialized to the equivalent of 0.07 and clipped to prevent scaling the logits by more than 100, which was found necessary to prevent training instability, and CLIP uses a very large minibatch size of 32,768. {#c-clip-temp}

**Data.**
The data have to be large and broad, because the captions are the only supervision.
To cover a broad set of visual concepts, the 400 million pairs were collected by searching for pairs whose text includes one of a set of 500,000 queries, balancing the results by including up to 20,000 (image, text) pairs per query; the dataset is called WIT for WebImageText. {#c-clip-wit}

**Why contrastive.**
A transformer language model that predicts the caption learns zero-shot ImageNet classification 3x slower than a baseline that predicts a bag-of-words encoding of the text, and swapping the prediction objective for the contrastive objective of CLIP further improves efficiency another 4x. {#c-clip-eff}
Predicting the exact words of a caption asks the model to get details right that are irrelevant to what the image shows; matching only asks it to rank the right caption above the others.

**Zero-shot classification.**
For each class, embed one or more captions built from the class name, average the normalized caption embeddings of each class, and classify an image by the largest cosine.
The wording of the captions matters:
Using the prompt template “A photo of a {label}.” is a good default that helps specify the text is about the content of the image, and just using this prompt improves accuracy on ImageNet by 1.3%. {#c-clip-prompt}
On ImageNet, ensembling 80 different context prompts improves performance by an additional 3.5% over the single default prompt, and together prompt engineering and ensembling improve ImageNet accuracy by almost 5%. {#c-clip-ens}
This dependence on wording is a case of the prompt sensitivity of language models, carried over to image classification.

**Scale of training.**
CLIP trains all models for 32 epochs; the RN50x64 model took 18 days to train on 592 V100 GPUs, while the ViT-L/14 Vision Transformer took 12 days on 256 V100 GPUs. {#c-clip-train}

## Derivation

The full derivation is in `derivations/contrastive-loss-arithmetic.md`, and `contrastive-loss-arithmetic.check.py` checks every number.

**The starting loss.**
When every logit in a row is equal the contrastive loss is log N, 10.3972 nats for a batch of 32,768 pairs and 9.7041 for 16,384, so the loss of untrained encoders starts near log N. {#c-d-logn}
Every nat below log N is information about which caption goes with which image; a loss of 0.7 nats on a batch of 32,768 means the right caption gets about half the probability.

**The temperature.**
With logit scale s, positives at cosine 1 and N − 1 negatives at cosine c, the softmax loss cannot go below log(1 + (N − 1) e^(−s(1 − c))): for N = 32,768 and c = 0 this is 9.3973 at s = 1, 0.020268 at s = 1/0.07 and below 10^−38 at s = 100. {#c-d-floor}
A small fixed scale therefore stops the loss far above zero however good the encoders become, and a large fixed scale makes the softmax sharp while the cosines are still noise.
Learning the scale lets it grow as the embeddings improve; the cap bounds the logits.

**The sigmoid loss at initialization.**
With the sigmoid loss and every cosine near 0, a bias of 0 makes each of the 32,767 negatives of an image cost log 2, 22,712.3 nats in total against 0.6931 for its positive, while t = 10 and b = −10 make them cost 1.4876 in total against 10.0000454 for the positive. {#c-d-sig}
The bias puts positives and negatives on the same scale, and σ(−10) = 4.54 × 10^−5 is close to the share of positives in a batch of tens of thousands.

**Memory.**
A full N × N matrix of 32-bit logits takes 4 GiB at N = 2^15 and 4 TiB at N = 2^20; the softmax loss needs whole rows on each of D devices, 4 GiB per device for N = 2^20 and D = 2^10, while the sigmoid loss can work in blocks of (N/D)^2 logits, 4 MiB each. {#c-d-mem}

**Pairs and steps.**
A batch of 32,768 pairs has 1,073,709,056 negative pairs, and 32 epochs over 400 million pairs are 12.8 billion samples seen, or 390,625 steps. {#c-d-steps}

**The zero-shot classifier.**
The zero-shot classifier is linear in the image embedding with one weight row per class written by the text encoder; with 80 prompts for each of 1,000 classes the text encoder runs 80,000 times once and the classifier is cached. {#c-d-zs}
The weights of this linear classifier come from text, not from labelled images, which is why a new label set costs only a few thousand text-encoder passes.

## Worked example

Take a batch of three pairs: a dog photo with "a dog on the grass", a cat photo with "a sleeping cat", and a car photo with "a red car".
Suppose the trained encoders give the cosine matrix, images in rows and captions in columns,

| | dog caption | cat caption | car caption |
|---|---|---|---|
| dog image | 0.9 | 0.2 | 0.1 |
| cat image | 0.3 | 0.8 | 0.0 |
| car image | 0.1 | 0.1 | 0.7 |

With logit scale 10, the dog image puts probability e^9 / (e^9 + e^2 + e^1) on its own caption, a loss of 0.0012 nats; the cat and car images lose 0.0070 and 0.0049, the three columns lose 0.0028, 0.0034 and 0.0034, and the batch loss is 0.0038.
With logit scale 1 the same cosines give a batch loss of 0.7084, close to log 3 = 1.0986, the loss of guessing: the cosines rank the captions correctly, but the softmax is too flat to say so with confidence.

To classify a new photo as dog, cat or car, embed "a photo of a dog.", "a photo of a cat." and "a photo of a car.", and pick the caption with the largest cosine to the photo.

The NumPy implementation trains two linear encoders on two noisy views of a shared latent vector with explicit gradients for both losses, including the gradient through the normalization.
With the softmax loss the retrieval accuracy on 256 held-out pairs goes from 0.004 to 1.0 in 400 steps and the learned scale rises from 14.3 to 27.2; with the sigmoid loss it also reaches 1.0, with the bias moving from −10 to −10.6.
Zero-shot classification from five captions per class reaches more than 80% on ten synthetic classes.
The PyTorch implementation uses three-layer ReLU encoders, whose initial embeddings lie in narrow cones, with a mean cosine of 0.53 between image embeddings, and start a distance of 1.16 apart.
After 200 steps at a fixed scale of 100 the two cones are still 0.475 apart while 99% of held-out pairs are matched; at a fixed scale of 1 the gap closes to 0.032 and the loss stays at 4.63, close to the floor log(1 + 255/e) = 4.55 the derivation gives for a batch of 256.

## Hyperparameters

The batch size sets how many negatives each pair is ranked against; with the softmax loss larger batches have been needed, and with the sigmoid loss gains saturate near 32k.
The logit scale should be learned and capped, for the reasons in the derivation.
The sigmoid loss needs its bias, or the first steps spend most of their gradient pushing all logits down.
Larger encoders and more data help along power laws, with exponents that depend on the data source.
For zero-shot use, the prompt templates are a hyperparameter of the classifier, and ensembling several templates is cheap.

## Cost

Training cost is dominated by the two encoders, as in any pretraining, plus the N × N logits, which are cheap to compute but whose memory grows with the square of the batch.
CLIP spread the softmax over devices by gathering the embeddings of the whole batch; the sigmoid loss avoids gathering by computing blocks of the matrix.
Inference is one encoder pass per input, and retrieval over a large collection is a nearest-neighbour search over precomputed embeddings, as in k-nearest neighbors, which is what makes these models practical for search.

## Evidence

**Zero-shot classification.**
CLIP improves zero-shot accuracy on ImageNet from a proof of concept 11.5% to 76.2% and matches the performance of the original ResNet-50 despite using none of the 1.28 million crowd-labeled training examples available for this dataset. {#c-clip-76}
Across a 27 dataset evaluation suite, a zero-shot CLIP classifier outperforms a fully supervised linear classifier fitted on ResNet-50 features on 16 datasets, including ImageNet. {#c-clip-27}
ImageNet, the benchmark of supervised image classification, became one test among many: the comparison is with a classifier trained on its labels.

**Robustness.**
All zero-shot CLIP models improve effective robustness by a large amount and reduce the size of the gap between ImageNet accuracy and accuracy under distribution shift by up to 75%. {#c-clip-robust}
Adapting CLIP to the ImageNet distribution with a logistic regression classifier increases its ImageNet accuracy by 9.2% to 85.4% overall, but average accuracy under distribution shift slightly decreases. {#c-clip-adapt}
The pattern suggests that part of the robustness comes from not fitting the ImageNet distribution, and that adapting to it gives some of the robustness back.

**ALIGN.**
ALIGN follows the procedures of the Conceptual Captions dataset but applies only simple frequency-based filtering instead of its complex filtering and post-processing, giving a noisy dataset of over one billion image alt-text pairs that is two orders of magnitude larger than Conceptual Captions. {#c-align-data}
ALIGN uses EfficientNet-L2 as the image encoder and BERT-Large as the text encoder, trained from scratch with a normalized softmax loss, and the temperature is learned together with all the other parameters instead of being swept manually. {#c-align-arch}
ALIGN is trained on 1024 Cloud TPUv3 cores with 16 positive pairs on each core, so the total effective batch size is 16384. {#c-align-batch}
Zero-shot image classification with ALIGN, feeding the class names into the text encoder, achieves 76.4% top-1 accuracy in ImageNet without using any of its training samples. {#c-align-76}
In the zero-shot setting, ALIGN gets more than 7% improvement in image retrieval task compared to CLIP on Flickr30K and MSCOCO. {#c-align-ret}
Two groups with different data, encoders and filtering reached about the same zero-shot ImageNet accuracy, which indicates that the objective and the scale, not the details, carry most of the result.

**Scaling with open data.**
Experiments with models trained on up to two billion image-text pairs identify power law scaling for multiple downstream tasks including zero-shot classification, retrieval, linear probing, and end-to-end fine-tuning. {#c-oc-power}
The training distribution plays a key role in scaling laws, as the OpenAI and OpenCLIP models exhibit different scaling behavior despite identical model architectures and similar training recipes. {#c-oc-dist}
Fitting power laws E = βC^α on the Pareto frontier gives scaling coefficients α of −0.11 for OpenCLIP and −0.16 for CLIP on zero-shot top-1 ImageNet, and −0.13 and −0.24 on the ImageNet robustness datasets. {#c-oc-alpha}
A ViT-H/14 trained on LAION-2B reaches 78.0% zero-shot ImageNet accuracy, against 75.5% for the CLIP ViT-L/14 trained on WIT-400M and 75.2% for a ViT-L/14 trained on LAION-2B. {#c-oc-h14}
Neural scaling laws for language models describe loss against compute; here the curves are for downstream accuracy, and their slope depends on where the pairs came from.

**Medical images.**
In all 4 medical image classification tasks, ConVIRT requires only 10% as much labeled training data as an ImageNet initialized counterpart to achieve better or comparable performance. {#c-cv-data}

**The sigmoid loss.**
The sigmoid loss performs significantly better than the softmax loss when the batch size is smaller than 16 k, and as the train batch size grows the gap closes. {#c-sig-small}
A SigLiT model was trained at a batch size of one million, but performance for both sigmoid and softmax saturates at around 32 k batch size. {#c-sig-sat}
Combined with Locked-image Tuning, with only four TPUv4 chips, a SigLiT model achieves 84.5% ImageNet zero-shot accuracy in two days. {#c-sig-cheap}

## Limitations and failure modes

**Specialized and abstract tasks.**
Zero-shot CLIP is quite weak on several specialized, complex, or abstract tasks such as satellite image classification, lymph node tumor detection, counting objects in synthetic scenes, German traffic sign recognition and recognizing distance to the nearest car. {#c-clip-weak}
CLIP only achieves 88% accuracy on the handwritten digits of MNIST, and an embarrassingly simple baseline of logistic regression on raw pixels outperforms zero-shot CLIP. {#c-clip-mnist}
Classes that are rare in web captions, or tasks such as counting that captions seldom describe, are not learned from captions.

**Compute for further gains.**
Around a 1000x increase in compute is estimated to be required for zero-shot CLIP to reach the overall performance of task-specific models, which is infeasible to train with current hardware. {#c-clip-1000x}

**Few-shot use.**
CLIP does not optimize for few-shot performance directly, and fitting linear classifiers on top of its features results in a counter-intuitive drop in performance when transitioning from a zero-shot to a few-shot setting. {#c-clip-fewshot}
The zero-shot classifier is defined by text and a few-shot linear probe throws that prior away; methods that combine the two came later.

**Social biases.**
The image-text pairs are unfiltered and uncurated and result in CLIP models learning many social biases. {#c-clip-bias}
CLIP's authors measured these with probes and the FairFace benchmark, and the encoders carry them into the systems built on them.

**The modality gap.**
Different data modalities such as images and texts are embedded at arm's length in the shared representation of multi-modal models, a geometric phenomenon called the modality gap. {#c-gap-what}
The representation of a common deep neural network is restricted to a narrow cone, so in a multi-modal model with two encoders the representations of the two modalities are clearly apart when the model is initialized, and during optimization contrastive learning keeps them separated by a distance influenced by the temperature parameter in the loss function. {#c-gap-cone}
At CLIP's learned final temperature, the default gap distance of 0.82 achieves the global minimum of the contrastive loss, and shifting the embeddings toward closing the gap increases the loss. {#c-gap-shift}
Simply modifying the gap's distance can improve CLIP's zero-shot performance and fairness. {#c-gap-zs}
The gap means that image and text embeddings are not interchangeable even when they describe the same thing, which matters for methods that mix them, such as using a text embedding in place of an image embedding.

## Variants

**Sigmoid loss.**
The sigmoid loss operates solely on image-text pairs and does not require a global view of the pairwise similarities for normalization, unlike standard contrastive learning with softmax normalization. {#c-sig-def}
Because the many negatives would dominate the loss at initialization, SigLIP introduces a learnable bias term b similar to the temperature t and initializes t' and b to log 10 and −10, so training starts roughly close to the prior. {#c-sig-init}
The sigmoid loss is symmetric, requires just a single pass, and a typical implementation requires less memory than the softmax loss. {#c-sig-mem}
Each pair becomes a binary decision under the logistic loss, so the loss of a pair does not depend on the rest of its row.

**Frozen or pretrained towers.**
Locked-image tuning keeps a pretrained image encoder fixed and trains only the text encoder to match it; SigLiT is this setup with the sigmoid loss.

**Open reproductions.**
OpenCLIP trains the same architectures on public LAION data, which made the scaling studies above possible.

**Other domains.**
ConVIRT applied the objective to medical images and reports before CLIP, and the same recipe has since been applied to audio, video and other pairs of modalities.

## When to use it

Use it when there are many images paired with free text and few labels, when the label set is open or changes, or when images and text must be retrieved by each other.
For a fixed label set with many labelled images, a supervised classifier or a fine-tuned CLIP encoder is usually more accurate on that set, at some cost in robustness to distribution shift.
For tasks that need counting, reading fine detail or spatial relations, the embeddings alone are weak, and models that feed image features into a language model are the usual next step.

## Pitfalls

Comparing zero-shot accuracy across papers without fixing the prompt templates mixes the effect of the model with the effect of the wording.
Evaluating on benchmarks whose images may be in the web training data needs an overlap check, as CLIP's authors ran.
Small batches with the softmax loss and a fixed small temperature both cap what the model can learn, for the reasons in the derivation.
Treating cosine similarity between an image and a text as calibrated across queries is unsafe: the modality gap and the learned scale shift all scores together.

## History

Learning visual features from paired text goes back to work on image captions and visual-semantic embeddings in the 2010s, and contrastive objectives of the InfoNCE form became common in self-supervised learning around 2018 to 2020.
ConVIRT, posted in October 2020, trained image encoders with a bidirectional contrastive loss against radiology reports.
In February 2021 two groups scaled the idea to the web: CLIP at OpenAI with 400 million pairs and ALIGN at Google with over a billion.
Open reproductions on LAION data followed in 2022, along with studies of the modality gap and of scaling laws, and SigLIP replaced the softmax with a sigmoid loss in 2023.

## References
