---
tag: NEW-image-classification
kind: task
slug: image-classification
name: Image classification
aliases: [image categorization, object classification]
area: computer-vision/image-recognition
level: core
summary: >-
  The task of assigning an image one class, or a ranked list of classes, from a fixed set of categories, scored by whether the true class is among the top predictions.
related_to: [imagenet-ilsvrc-2012, imagenet]
history:
  year: 2010
  note: "The first ImageNet Large Scale Visual Recognition Challenge (Russakovsky et al., 2015); the task is older, and its earlier history is not covered by the sources of this page."
sources:
  - {paper: "Russakovsky et al. (2015)", title: "ImageNet Large Scale Visual Recognition Challenge", locator: "pp. 4 to 31"}
  - {paper: "Krizhevsky, Sutskever and Hinton (2012)", title: "ImageNet Classification with Deep Convolutional Neural Networks", locator: "the abstract"}
  - {paper: "He, Zhang, Ren and Sun (2016)", title: "Deep Residual Learning for Image Recognition", locator: "p. 1"}
  - {paper: "Dosovitskiy et al. (2021)", title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", locator: "the abstract"}
  - {paper: "Recht, Roelofs, Schmidt and Shankar (2019)", title: "Do ImageNet Classifiers Generalize to ImageNet?", locator: "p. 1"}
  - {paper: "Beyer, Hénaff, Kolesnikov, Zhai and van den Oord (2020)", title: "Are we done with ImageNet?", locator: "p. 2"}
  - {paper: "Northcutt, Athalye and Mueller (2021)", title: "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks", locator: "p. 2"}
---

Image classification is the task of saying which category an image belongs to, out of a fixed list of categories.
A classifier takes an image and returns a class or a ranked list of classes, and it is scored by whether the true class is first in the list or among its first few entries.
The ImageNet challenge made the task a common test of vision models, and on it deep networks learned from pixels replaced pipelines of hand-designed features.

## In plain words

You show a program a photograph and ask: is this a cat, a car, a mushroom, or one of a thousand other things?
The program has a fixed list of possible answers and must pick from it.
It learns to do this from many example photographs whose answers people have already written down.
To check it, you give it photographs it has never seen and count how often it is wrong.
Because one photograph can show several things, the usual test lets the program give five guesses and counts it right if any guess matches the answer.
The hard part is that a category, such as "dog", covers many looks, poses, lights and backgrounds, while two categories, such as two breeds of dog, can look almost the same.

## Definition

The input is an image, and the output is drawn from a fixed set of $K$ classes.
In the ImageNet challenge, each image has one ground truth label from 1000 object categories, and an algorithm produces a list of categories present in the image. {#c-def}
The top-5 error lets an algorithm return 5 labels for each image and counts it correct if one of them equals the ground truth label; the error is the fraction of test images on which it makes a mistake. {#c-top5}
The top-1 error counts an image as an error when the label of highest confidence does not match the ground truth. {#c-top1}
Writing $\hat y_{i1}, \dots, \hat y_{ik}$ for the first $k$ labels returned for image $i$ and $y_i$ for its label, the top-$k$ error over $N$ test images is

$$\text{error}_k = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left[\, y_i \notin \{\hat y_{i1}, \dots, \hat y_{ik}\} \,\right].$$

## Metrics

Top-1 and top-5 error, or the matching accuracies, are the usual measures; the ReaL accuracy of Beyer et al. (2020) counts a prediction as right if it is any of the labels that reviewers gave the image.

## Benchmarks

[ILSVRC-2012](/imagenet-ilsvrc-2012) tests classification of photographs into 1000 ImageNet classes; rebuilt test sets and relabeled validation sets of it test whether its scores hold on fresh images and on better labels.

## Approaches

**Hand-designed features and linear classifiers.** The winner of the first ImageNet challenge in 2010, the NEC team, used SIFT and LBP features with a stochastic SVM. {#c-features}
In 2012 the second place went to the ISI team, which used Fisher vectors. {#c-fisher}

**Deep convolutional networks.** The network of Krizhevsky et al. (2012) has 60 million parameters and 650,000 neurons, in five convolutional layers, some followed by max-pooling, and three fully connected layers with a final 1000-way softmax. {#c-alexnet-arch}
To reduce overfitting in the fully connected layers they used [dropout](/dropout). {#c-alexnet-dropout}
It won the ILSVRC-2012 classification task with a top-5 test error of 15.3%, against 26.2% for the second entry. {#c-alexnet-win}
Later winners kept the convolutional design: in 2014 GoogLeNet won the track with the provided training data with an improved convolutional architecture that combined a multi-scale idea with additional dimension reduction layers. {#c-googlenet}

**Residual networks.** Deeper networks are harder to train; residual networks reformulate the layers as learning residual functions with reference to their inputs, which makes them easier to optimize and lets them gain accuracy from much more depth. {#c-resnet-idea}
An ensemble of residual networks of up to 152 layers had 3.57% error on the ImageNet test set. {#c-resnet-result}

**Vision Transformers.** The Vision Transformer applies a standard Transformer directly to sequences of image patches. {#c-vit}
Dosovitskiy et al. report that, pre-trained on large amounts of data and transferred to mid-sized or small image recognition benchmarks, it compares well with convolutional networks while needing substantially fewer computational resources to train. {#c-vit-result}

## Difficulties

**Fine distinctions.** Some errors are milder than others: confusing two breeds of dog is not as harmful as confusing a dog for a container ship, which is why the challenge also computed a hierarchical error. {#c-fine}
**One label for many things.** Real images often contain several objects of interest, yet ImageNet gives each image a single label. {#c-multi}
**Label errors.** Northcutt et al. (2021) find 2916 errors, 6% of the images, in the ImageNet validation set. {#c-errors}
**Shift to fresh data.** On a test set rebuilt with the original process of ImageNet, the accuracy of a wide range of models drops by 11% to 14%. {#c-shift}

## History

The ImageNet Large Scale Visual Recognition Challenge ran each year from 2010 and drew entries from more than fifty institutions. {#c-ilsvrc}
Its first edition, in 2010, had only the classification task.
With efficient implementations and GPU computing, it became possible to learn neural networks directly from image data, without multi-stage hand-tuned pipelines of extracted features and classifiers; the breakthrough came with the win of the SuperVision team in 2012, and by 2014 the top contestants relied heavily on convolutional networks. {#c-breakthrough}
Among teams that used only the provided training data, the winning error fell 2.4 times from 2012 to 2014, from 16.4% to 6.7%. {#c-reduction}
On 1500 test images, a human annotator had an error of 5.1%, against 6.8% for GoogLeNet. {#c-human}
Residual networks won the 2015 challenge (He et al., 2016), and Vision Transformers carried the Transformer architecture over from language to images (Dosovitskiy et al., 2021).
