---
tag: NEW-imagenet
kind: dataset
slug: imagenet
name: ImageNet
aliases: [ImageNet database, full ImageNet]
area: data-for-ai/multimodal-datasets
level: core
summary: >-
  A database of web images organized by the noun synsets of WordNet, each image checked by crowd workers; its 1000-class subset is the data of the ILSVRC image classification benchmark.
related_to: [imagenet-ilsvrc-2012, image-classification]
history:
  year: 2009
  note: "Deng et al. (2009), CVPR 2009; the challenge data is described in Russakovsky et al. (2015), IJCV 115."
same_as:
  wikipedia: ImageNet
sources:
  - {paper: "Deng, Dong, Socher, Li, Li and Fei-Fei (2009)", title: "ImageNet: A Large-Scale Hierarchical Image Database", locator: "the title page", note: "the paper that introduced the database; no stored snapshot yet, because image-net.org refused the connection"}
  - {paper: "Russakovsky et al. (2015)", title: "ImageNet Large Scale Visual Recognition Challenge", locator: "pp. 2 to 9", note: "the size of the database, its use of WordNet and the collection and annotation process"}
  - {paper: "Krizhevsky, Sutskever and Hinton (2012)", title: "ImageNet Classification with Deep Convolutional Neural Networks", locator: "Section 2"}
  - {paper: "Yang, Qinami, Fei-Fei, Deng and Russakovsky (2020)", title: "Towards Fairer Datasets: Filtering and Balancing the Distribution of the People Subtree in the ImageNet Hierarchy", locator: "pp. 1 to 5"}
  - {paper: "Northcutt, Athalye and Mueller (2021)", title: "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks", locator: "p. 1"}
---

ImageNet is a database of images organized by the noun hierarchy of WordNet.
Each concept of the hierarchy, a synonym set or synset, is illustrated by photographs that were found with web image search engines and checked by crowd workers.
By August 2014 it held about 14.2 million images in 21,841 synsets.
A fixed subset of 1000 synsets is the data of the ILSVRC image classification benchmark, on which many deep vision models were trained and compared.

## Contents

ImageNet is organized according to the WordNet hierarchy, in which each concept, described by one or more words or phrases, is a synonym set or synset. {#c-wordnet}
As of August 2014, ImageNet held 14,197,122 annotated images in 21,841 synsets of WordNet, an average of 650 manually verified images per synset. {#c-size}
The modality is still images: full-resolution photographs, each annotated with the synset it illustrates.
The person subtree of the full database has 2,832 people categories, which hold roughly 8.3% of the images. {#c-person-share}

Most research uses not the full database but the subset of the ILSVRC image classification task, which from 2012 to 2014 had 1000 synsets with about 1.2 million training images, 50,000 validation images and 100,000 test images. {#c-subset}
That subset is described on the page of the benchmark, [ILSVRC-2012](/imagenet-ilsvrc-2012).
Krizhevsky et al. (2012) describe the full database as over 15 million labeled high-resolution images in roughly 22,000 categories, collected from the web and labeled with Amazon Mechanical Turk. {#c-krizhevsky-size}
The two counts differ in date and in rounding; neither paper gives the method of its count.

## Collection and construction

The construction has three steps: choose the concepts, collect candidate images, and have people verify them.

**Concepts.** Because ImageNet uses WordNet as its backbone, it inherits the disambiguation of word meanings and the grouping of synonyms into one category. {#c-synonyms}
During the construction of ImageNet in 2009, the team removed every synset whose gloss marks it as offensive, derogatory, pejorative or a slur, but this filtering was imperfect and left some offensive synsets in the person subtree. {#c-gloss-filter}

**Candidate images.** For each synset, the queries are its WordNet synonyms, sent to several image search engines. {#c-queries}
Because search engines return only a few hundred to a thousand images per query, the queries are extended with the word of a parent synset when that word appears in the gloss of the synset, and they are translated into Chinese, Spanish, Dutch and Italian with the WordNets of those languages. {#c-query-expansion}

**Verification.** Workers on Amazon Mechanical Turk see a set of candidate images and the definition of the synset, and say whether each image contains objects of the synset. {#c-amt}
Several workers label each image independently, and an image is kept only if it gets a convincing majority of the votes.
The majority needed depends on the synset: at least 10 workers vote on an initial sample of its images, which gives a table of confidence scores, and the other images get votes until they reach a set confidence. {#c-dynamic-consensus}
On 80 synsets sampled at every depth of the mammal and vehicle subtrees, an independent group of people found an average precision of 99.7%. {#c-precision}

## Documentation and licence

Deng et al. (2009) describe the database and its collection, and Russakovsky et al. (2015, Section 3) restate the process for the challenge data and refer to Deng et al. for the details. {#c-docs}
The authors of ImageNet describe it as a free education dataset released for non-commercial use only, under the terms and conditions of image-net.org. {#c-noncommercial}
The images themselves come from the web, so the terms of the database are not a licence to the photographs; this page has no source for the rights of the photographers, and it states none.
The papers read for this page describe no consent or opt-out process for the people in the images.

## Known issues

**Offensive and non-visual person categories.** Yang et al. (2020) examined the 2,832 people categories of the person subtree, judged 1,593 of them potentially offensive labels for an image recognition dataset, and planned to remove all of them. {#c-unsafe}
Of the remaining 1,239 categories, they found only 158 to be visual, and they recommend training visual recognition models only on those 158. {#c-imageable}

**Unbalanced people.** The same study had crowd workers annotate the gender, skin color and age of the people in the images of the remaining categories, as a step to rebalance them. {#c-demographics}

**Label errors.** The training labels of ImageNet are known to contain errors. {#c-train-errors}
The validation labels of the challenge subset are discussed under the known issues of [ILSVRC-2012](/imagenet-ilsvrc-2012).

## History

ImageNet was introduced by Deng et al. at CVPR 2009.
Constructing ImageNet was an effort to cover most of the nouns of English with tens of millions of manually verified photographs, and the image classification task of ILSVRC came as a direct extension of it. {#c-goal}
A subset of its categories and images was fixed to give a standard benchmark, while the rest of the database kept growing. {#c-fixed-subset}
The challenge ran each year from 2010, and its 2012 edition is where a deep convolutional network, AlexNet, won the classification task (Krizhevsky et al., 2012).
In 2020 the ImageNet team published its review of the person subtree (Yang et al., 2020).
