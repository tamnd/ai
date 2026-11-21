# The arithmetic of contrastive and sigmoid image-text losses

This derivation backs the mathematical claims of the contrastive language-image pretraining page. The check file
`contrastive-loss-arithmetic.check.py` tests every number below.

## Setting

A batch holds N image-text pairs. The image encoder and the text encoder map each image and each caption to a unit
vector, u_i and v_j. The logit for image i and caption j is s · u_i · v_j, where s = 1/τ is a learned logit scale
(the inverse of the temperature τ). The contrastive loss of CLIP averages two cross-entropies over the N × N matrix of
logits: one that asks each image to pick its own caption among the N captions in the batch, and one that asks each
caption to pick its own image:

L = (1/2N) Σ_i [ −log softmax_j(s u_i·v_j)_i − log softmax_j(s u_j·v_i)_i ].

## 1. The loss with uninformative embeddings is log N

If every logit in a row is equal, the softmax gives each of the N candidates probability 1/N and each cross-entropy is
log N. At the start of training the embeddings of random encoders are nearly unrelated, so the loss starts near log N:
10.3972 for N = 32,768, the batch size of CLIP, and 9.7041 for N = 16,384, the batch size of ALIGN. The check file
confirms this with random unit vectors in 256 dimensions. Every nat of loss below log N is information the encoders have
learned about which caption goes with which image.

## 2. The logit scale sets a floor on the loss

Cosine similarities lie in [−1, 1], so with scale s the logits lie in [−s, s]. Suppose training has succeeded as well
as it can: every positive pair has cosine 1 and every negative pair has cosine c. One row of the loss is then

−log [ e^s / (e^s + (N − 1) e^{s c}) ] = log(1 + (N − 1) e^{−s (1 − c)}).

For N = 32,768 and negatives orthogonal to the image (c = 0), this floor is 9.3973 at s = 1, 0.020268 at s = 1/0.07 =
14.2857, the initial scale of CLIP, and below 10^−38 at s = 100, the cap CLIP places on the scale. Even with negatives
at the far side of the sphere (c = −1), a scale of 1 leaves a floor of 8.3974. With the initial scale and negatives at
cosine 0.2 the floor is 0.3049.

A fixed small scale would therefore stop the loss far above zero however good the encoders became, and the gradient
would keep pushing negatives apart instead of learning finer distinctions. A fixed large scale makes the softmax sharp
from the start, when the cosines are noise. Learning s, as CLIP and ALIGN do, lets the scale grow as the embeddings
improve; capping it bounds the logits, which CLIP found necessary for stable training.

## 3. The sigmoid loss and its initial bias

The sigmoid loss of SigLIP treats every image-text pair in the batch as a separate binary problem with label +1 for the
matching pair and −1 for the others. With logit z_ij = t u_i·v_j + b the loss is

L = −(1/N) Σ_i Σ_j log σ(y_ij z_ij),

where σ is the logistic function and y_ij is the label. Each image contributes one positive term and N − 1 negative
terms.

At initialization the cosines are near 0, so every logit is near b. Without a bias (b = 0) every term costs log 2 =
0.6931, and for N = 32,768 the negatives of one image contribute 32,767 · log 2 = 22,712.3 against 0.6931 for its
positive. The gradient is dominated by the negatives, and the first steps of training mostly push all logits down.

SigLIP initializes t = 10 and b = −10. At cosine 0 a negative then costs −log σ(10) = 4.5399 × 10^−5 and the positive
costs −log σ(−10) = 10.0000454. For N = 32,768 the negatives of one image add up to 1.4876 and the row costs 11.4876 in
total, so the positive and the negatives now contribute on the same scale. The bias also matches the prior: σ(−10) =
4.54 × 10^−5 is the positive rate of a batch of about e^10 = 22,026.5 pairs, while the exact log-odds of a positive
in a batch of 32,768 is log(1/32,767) = −10.3972.

## 4. Memory of the similarity matrix

The full N × N matrix of logits in 32-bit floats takes 4N^2 bytes: 2^32 bytes, 4 GiB, for N = 32,768 = 2^15, and
4 TiB for N = 2^20, about one million. Spread over D devices, the softmax loss needs each device to see whole rows,
because each normalizer sums over all N captions: (N/D) × N logits per device, 4 GiB for N = 2^20 and D = 2^10. The
sigmoid loss has no normalizer, so each device can compute one (N/D) × (N/D) block at a time and pass the text
embeddings around the devices: 2^20 logits, 4 MiB, per block for the same N and D. This blockwise computation is what let SigLIP
train with batch sizes up to one million.

## 5. Pairs and steps

A batch of N = 32,768 pairs gives N^2 − N = 1,073,709,056 negative pairs per step. Thirty-two epochs over 400 million
pairs are 12.8 billion samples seen, which at that batch size is 390,625 steps.

## 6. The zero-shot classifier is a linear classifier

For zero-shot classification over K classes, the text encoder embeds a caption for each class, such as "a photo of a
dog.", and the image is assigned to the class whose caption embedding has the largest cosine with the image embedding.
The class embeddings form the rows of a weight matrix, so the classifier is linear in the image embedding, with weights
written by the text encoder instead of learned from labelled images. Ensembling P prompts per class averages the P
normalized embeddings of each class and normalizes the mean; the argmax is unchanged by the final normalization of the
image embedding. The text encoder runs K · P times once per task, 80,000 times for 1,000 ImageNet classes and 80
prompts, and the classifier is then cached.
