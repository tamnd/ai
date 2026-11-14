---
tag: NEW-vaswani-2017-attention
kind: work
slug: vaswani-2017-attention
name: Attention Is All You Need (Vaswani et al. 2017)
aliases: [Attention Is All You Need, Vaswani et al. (2017)]
area: people-and-institutions/landmark-works
level: core
summary: >-
  The 2017 paper by Ashish Vaswani and seven co-authors that introduced the Transformer, a sequence transduction model built on attention alone, with no recurrence and no convolution.
history:
  year: 2017
  people: [Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin]
  note: "arXiv 1706.03762, June 2017; presented at NIPS 2017, the 31st Conference on Neural Information Processing Systems, Long Beach. The page cites version 7 of the arXiv text."
same_as:
  wikidata: Q30249683
  wikipedia: Attention_Is_All_You_Need
  arxiv: ["1706.03762"]
  dblp: conf/nips/VaswaniSPUJGKP17
  url: "https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html"
sources:
  - {paper: "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser and Polosukhin (2017)", title: "Attention Is All You Need", locator: "arXiv 1706.03762v7, Sections 1 to 7 and Tables 2 to 4", note: "the work itself"}
  - {paper: "Bahdanau, Cho and Bengio (2015)", title: "Neural Machine Translation by Jointly Learning to Align and Translate", locator: "ICLR 2015, Section 1", note: "the attention model that the paper builds on"}
  - {paper: "Devlin, Chang, Lee and Toutanova (2019)", title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", locator: "NAACL 2019, Section 3"}
  - {paper: "Radford, Narasimhan, Salimans and Sutskever (2018)", title: "Improving Language Understanding by Generative Pre-Training", locator: "Section 4.1"}
  - {paper: "Dosovitskiy et al. (2021)", title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", locator: "ICLR 2021, Section 3"}
  - {paper: "Post (2018)", title: "A Call for Clarity in Reporting BLEU Scores", locator: "WMT 2018, the abstract and Table 2"}
  - {paper: "Xiong et al. (2020)", title: "On Layer Normalization in the Transformer Architecture", locator: "ICML 2020, the abstract"}
  - {paper: "Hahn (2020)", title: "Theoretical Limitations of Self-Attention in Neural Sequence Models", locator: "TACL 8, the abstract"}
type: "paper"
authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Łukasz Kaiser", "Illia Polosukhin"]
year: 2017
venue: "NeurIPS 2017"
standing: "peer-reviewed"
licence_class: "D"
cited: {"version": "arXiv 1706.03762v7", "url": "https://arxiv.org/pdf/1706.03762v7", "format": "pdf"}
---

"Attention Is All You Need" is the paper that proposed the Transformer, a network architecture based only on attention, with no recurrence and no convolution. {#c-propose}
Ashish Vaswani and seven co-authors wrote it at Google Brain, Google Research and the University of Toronto. {#c-affiliations}
It appeared at the 31st Conference on Neural Information Processing Systems (NIPS 2017) in Long Beach. {#c-venue}
Its translation results, and the architecture itself, were the starting point of BERT, GPT and the vision Transformer.

## Contributions

To the best of their knowledge, the authors say, the Transformer is the first transduction model that relies entirely on self-attention to compute representations of its input and output. {#c-first-self-attention}
The paper makes four contributions.

1. Scaled dot-product attention divides the dot products of a query with all keys by the square root of the key dimension, and a softmax turns them into weights on the values. {#c-scaled-dot-product}
   The authors suspect that for large key dimensions the dot products grow large and push the softmax into regions with extremely small gradients, and they scale to counteract this. {#c-why-scale}
2. Multi-head attention projects the queries, keys and values h times with different learned linear projections, and performs attention on each projected version in parallel. {#c-multi-head}
   The paper uses h = 8 heads with key and value dimension 64 each. {#c-eight-heads}
3. Positional encodings, sines and cosines of different frequencies, are added to the input embeddings to give the model the order of the tokens. {#c-positional}
4. The encoder and the decoder are stacks of N = 6 identical layers, with a residual connection around each sub-layer followed by layer normalization. {#c-stacks}

Section 4 compares self-attention with recurrent and convolutional layers by three measures: the computational complexity per layer, the minimum number of sequential operations, and the path length between long-range dependencies. {#c-three-measures}
The author footnote says that Jakob Uszkoreit proposed replacing RNNs with self-attention. {#c-uszkoreit}
It says that Noam Shazeer proposed scaled dot-product attention, multi-head attention and the parameter-free position representation. {#c-shazeer}
It also says that the authors contributed equally and that the order of the names is random. {#c-equal-contribution}

## Context

In 2017 the main models for translation were recurrent encoder-decoders, and the best of them connected the encoder and the decoder through attention. {#c-context-attention}
In the model of Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio (2015), the decoder searches the source sentence for relevant positions instead of reading one fixed-length vector. {#c-bahdanau}
Vaswani et al. argue that the sequential nature of recurrent models precludes parallelization within training examples. {#c-sequential}
They note that in all but a few cases attention mechanisms were used together with a recurrent network. {#c-attention-with-rnn}
The convolutional models ByteNet and ConvS2S compute in parallel over positions, but the number of operations that relates two positions grows with their distance. {#c-convolutional}
In the Transformer that number is constant. {#c-constant-path}

## Methods and results

On one machine with 8 NVIDIA P100 GPUs, the base models trained for 100,000 steps (12 hours) and the big models for 300,000 steps (3.5 days). {#c-training-time}
The optimizer was Adam, with a learning rate that increases linearly for the first 4000 warmup steps and then decreases with the inverse square root of the step number. {#c-warmup}
The paper reports 28.4 BLEU for the big model on WMT 2014 English-to-German, more than 2 BLEU above the best earlier results, ensembles included. {#c-en-de}
It reports 41.8 BLEU on WMT 2014 English-to-French after 3.5 days of training on eight GPUs. {#c-en-fr}
In the ablations of Table 3, single-head attention is 0.9 BLEU worse than the best setting, and quality also drops with too many heads. {#c-heads}
Learned positional embeddings in place of the sinusoids give nearly identical results to the base model. {#c-learned-positions}
On English constituency parsing, a 4-layer Transformer scores 91.3 F1 on Section 23 of the Wall Street Journal trained on WSJ only, and 92.7 semi-supervised. {#c-parsing}
The code was released in the tensor2tensor library. {#c-code}

## Reception and influence

The citation counts and the follow-up works are generated from the cards.
Three works that built on the paper say so in their method sections.
BERT is a multi-layer bidirectional Transformer encoder based on the original implementation of Vaswani et al. (2017). {#c-bert}
GPT, the generative pre-training model of Radford et al. (2018), largely follows the original transformer work, as a 12-layer decoder-only transformer with masked self-attention heads. {#c-gpt}
The vision Transformer of Dosovitskiy et al. (2021) follows the original Transformer as closely as possible. {#c-vit}

## Criticism, replications and corrections

The evidence against and the errata are generated from the claims and the cards.
Version 7 is not consistent on English-to-French: Section 6.1 gives the big model a BLEU score of 41.0, while the abstract and Table 2 give 41.8. {#c-en-fr-41-0}
Matt Post (2018) finds that BLEU scores vary by as much as 1.8 between commonly used configurations. {#c-post-variation}
He lists Vaswani et al. (2017) among the well-cited papers whose BLEU configuration is unclear. {#c-post-vaswani}
Ruibin Xiong and co-authors (2020) argue that the original Post-LN Transformer needs a learning rate warm-up because its gradients near the output layer are large at initialization. {#c-post-ln}
Michael Hahn (2020) finds that self-attention cannot model periodic finite-state languages or hierarchical structure unless the number of layers or heads grows with the input length. {#c-hahn}

## References
