---
tag: NEW-imagenet-ilsvrc-2012
kind: benchmark
slug: imagenet-ilsvrc-2012
name: ImageNet ILSVRC-2012
aliases: [ILSVRC-2012, ILSVRC2012, ImageNet-1k, ImageNet classification benchmark]
area: evaluation/benchmarks-and-their-design
level: core
summary: >-
  The image classification benchmark of the 2012 ImageNet Large Scale Visual Recognition Challenge: photographs of 1000 ImageNet classes, scored by top-5 error, with a hidden test set.
related_to: [imagenet, image-classification]
history:
  year: 2012
  note: "The classification task of ILSVRC2012; its classes and data stayed the same in 2013 and 2014 (Russakovsky et al., 2015)."
same_as:
  wikipedia: ImageNet
sources:
  - {paper: "Russakovsky et al. (2015)", title: "ImageNet Large Scale Visual Recognition Challenge", locator: "pp. 2 to 31", note: "the task, the data, the splits, the metric, the results of the challenge and the human comparison"}
  - {paper: "Krizhevsky, Sutskever and Hinton (2012)", title: "ImageNet Classification with Deep Convolutional Neural Networks", locator: "Section 2"}
  - {paper: "He, Zhang, Ren and Sun (2016)", title: "Deep Residual Learning for Image Recognition", locator: "p. 1"}
  - {paper: "Recht, Roelofs, Schmidt and Shankar (2019)", title: "Do ImageNet Classifiers Generalize to ImageNet?", locator: "pp. 1 to 7"}
  - {paper: "Beyer, Hénaff, Kolesnikov, Zhai and van den Oord (2020)", title: "Are we done with ImageNet?", locator: "pp. 1 to 4"}
  - {paper: "Northcutt, Athalye and Mueller (2021)", title: "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks", locator: "pp. 1 to 10"}
---

ImageNet ILSVRC-2012 is the image classification benchmark of the 2012 ImageNet Large Scale Visual Recognition Challenge.
A model sees a photograph and returns up to five of 1000 object classes, and it is right if one of them is the class of the image.
The classes and the data stayed the same after 2012, so results from different years can be compared on it.
The test labels are hidden, so many papers report on its 50,000-image validation set instead.

## What it measures

The task is to say which object class a photograph shows.
The data are photographs from Flickr and other search engines, each labeled with one of 1000 object classes, so each image has one ground truth label. {#c-one-label}
A model returns a list of classes for each image, and it is scored on the one of them that best matches the ground truth label. {#c-task}
Russakovsky et al. describe the challenge as a benchmark in object category classification and detection on hundreds of categories and millions of images. {#c-benchmark}

The benchmark does not measure how well a model names everything in an image.
Many real images hold several objects of interest, but each ImageNet image has a single label, which can badly underrepresent its content. {#c-single-label}
It also does not by itself show how a model does on fresh images from the same source: on a test set built again with the same process, the accuracy of models drops by 11% to 14% (see Known issues). {#c-drop}

## Data

The 1000 classes are synsets of [ImageNet](/imagenet), chosen so that no class is an ancestor of another in the ImageNet hierarchy. {#c-no-ancestor}
The set of classes changed in the first years of the challenge: 639 synsets were used in all five challenges from 2010 to 2014. {#c-639}
In ILSVRC2012, 90 synsets were replaced with dog breeds for a more fine-grained classification, and the synsets have not changed since 2012. {#c-dogs}

The training images are taken directly from ImageNet; more images were collected in the same way and split at random into the validation and test sets. {#c-split-source}
The test images were collected and labeled for the competition and are not part of the earlier published ImageNet. {#c-test-new}

| Split | Images | Per class |
|---|---|---|
| train | 1,281,167 | 732 to 1300 |
| validation | 50,000 | 50 |
| test | 100,000 | 100 |

The table holds the sizes of ILSVRC2012 to ILSVRC2014, which share their data. {#c-splits}
A manual check of 1500 images of that test set found 5 annotation errors, a precision of 99.7%. {#c-test-precision}
The data is distributed with ImageNet, under the same non-commercial terms; the page on [ImageNet](/imagenet) gives the terms and their source.

## Protocol

The official metric is the top-5 error: a model returns 5 labels for each image, and the image counts as an error if none of them is the ground truth label; the error is the fraction of test images with an error. {#c-top5}
The organisers also computed the top-1 error and a hierarchical error, found that all three gave the same ordering of results, and have used only top-5 since ILSVRC2012. {#c-one-metric}
Papers also report top-1 error or top-1 accuracy, as Krizhevsky et al. and Recht et al. do, so a comparison has to keep to one metric.

The test labels are hidden. The organisers released the test labels of ILSVRC2010 only, and kept the others hidden to discourage tuning on the test data. {#c-hidden}
After the challenge, an evaluation server scores test predictions all year, with a limit of two submissions per week for each team. {#c-server}
Because the ILSVRC2012 test labels are not public, studies of the benchmark often work with the validation set instead. {#c-val}

The setting changes the scores.
Training data is one such setting: in 2012, SuperVision had 16.4% top-5 error when trained on the provided data, and 15.3% with entries also trained on extra data from the ImageNet Fall 2011 release. {#c-extra-data}
Image size is also a choice of the model: Krizhevsky et al. scaled each image so that its shorter side was 256 pixels and cropped the central 256 × 256 patch. {#c-resize}

## Known issues

**Label errors.** Northcutt et al. (2021) estimate that 2916 images, 6% of the validation set, have wrong labels. {#c-val-errors}
Counting the errors that their method did not flag, they estimate the share is closer to 20%. {#c-val-errors-20}
The class list has duplicate pairs, such as "maillot" (classes 638 and 639) and "crane" (classes 134 and 517). {#c-duplicates}
With corrected labels, ResNet-18 beats ResNet-50 if the share of originally mislabeled test images rises by 6%. {#c-ranking}

**Single labels.** Beyer et al. (2020) name three problems of the labels: one label for images with several objects, label proposals that are too restrictive, and classes that are near duplicates. {#c-label-problems}
They collected new labels for the validation set, the ReaL labels: 57,553 labels for 46,837 images, with 3163 images left out because they got no label. {#c-real}
Gains in the original accuracy carried over almost fully to ReaL accuracy for early models, and much less for later ones. {#c-real-gains}

**Fresh test data.** Recht et al. (2019) built an ImageNet test set of 10,000 images by repeating the original collection process. {#c-v2}
On it, accuracy fell by 11% to 14% for a wide range of models, and a model with 83% top-1 accuracy on the original test set had 72% on the rebuilt one. {#c-v2-drop}

## Reception

Russakovsky et al. write that the scale of the dataset stayed the same over the years, which makes the results comparable across years, and that the dataset has not changed since 2012. {#c-comparable}
Recht et al. (2019) put the accuracy lost on their rebuilt test set at about five years of progress on ImageNet. {#c-five-years}
Beyer et al. (2020) ask whether the gains that are still reported on ImageNet are as general as before, and answer with the ReaL labels above. {#c-beyer-question}

## History

The challenge began in 2010 with a classification task only; its winner, the NEC team, used SIFT and LBP features with a stochastic SVM. {#c-2010}
In 2012 the SuperVision team won with a deep convolutional network of 60 million parameters trained with dropout; the second place went to the ISI team, which used Fisher vectors. {#c-2012}
Krizhevsky et al. report the winning top-5 test error of 15.3%, against 26.2% for the second-best entry. {#c-alexnet}
Among teams that used only the provided training data, the winning classification error fell 2.4 times from 2012 to 2014, from 16.4% to 6.7%. {#c-reduction}
A human annotator who labeled 1500 test images had an error of 5.1%, against 6.8% for GoogLeNet on the same images. {#c-human}
In 2015 an ensemble of residual networks of up to 152 layers had 3.57% error on the ImageNet test set and took first place in the ILSVRC 2015 classification task. {#c-resnet}
