---
tag: NEW-byte-pair-encoding
kind: method
slug: byte-pair-encoding
name: Byte-pair encoding
aliases: [BPE, byte pair encoding, BPE tokenization, subword BPE, BPE-dropout]
area: language-models/tokenization
level: core
summary: >-
  A subword tokenizer that starts from characters and repeatedly merges the most frequent adjacent symbol pair in a corpus; a word is segmented by replaying the ordered merges.
purpose: tokenization
uses: [huffman-coding@m2, tokens-and-lexing@pg]
related_to: [dropout, regularization@al]
hyperparameters:
  - {name: number of merges, meaning: "the number of merge operations learned; the only hyperparameter of the original algorithm", typical: "59 500 for separate BPE and 89 500 for joint BPE in Sennrich et al. (2016)"}
  - {name: vocabulary size, meaning: "the final number of symbols, the target that SentencePiece asks for instead of a merge count", typical: "usually 16k to 32k (Kudo, 2018); 4k to 32k in Provilkov et al. (2020); 1000 in the SentencePiece command-line example"}
  - {name: joint or separate, meaning: "whether one merge list is learned on the union of the source and target vocabularies or one list per language", typical: "joint for language pairs that share names and cognates"}
  - {name: p, meaning: "the probability that BPE-dropout drops a candidate merge occurrence during segmentation", typical: "0.1 in training (0.6 for Chinese and Japanese) and 0 at inference"}
cost:
  train: "each merge rescans the word table, so M merges over a table of T symbols cost O(M T) when done naively; indexing the pairs and updating them incrementally makes each merge touch only the words that contain the pair"
  inference: "O(N^2) for a word or sentence of N characters when every pair is rescanned after each merge; O(N log N) with a binary heap of candidate merges"
history:
  year: 2016
  note: "Sennrich, Haddow and Birch (2016), arXiv 1508.07909, adapted the byte pair compression scheme of Gage (1994) to subword segmentation for neural machine translation."
prereqs: [tokens-and-lexing@pg, huffman-coding@m2]
same_as:
  arxiv: ["1508.07909", "1804.10959", "1808.06226", "1910.13267"]
sources:
  - {paper: "Sennrich, Haddow and Birch (2016)", title: "Neural Machine Translation of Rare Words with Subword Units", locator: "arXiv 1508.07909v5, Sections 3.2 and 4", note: "the method, Algorithm 1 and the translation experiments"}
  - {paper: "Kudo (2018)", title: "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates", locator: "arXiv 1804.10959v1, Sections 3.1 to 3.3", note: "BPE as a greedy deterministic segmenter and the unigram alternative"}
  - {paper: "Kudo and Richardson (2018)", title: "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing", locator: "arXiv 1808.06226v1, Sections 3.1 to 3.4", note: "raw-text training, lossless tokenization, the heap algorithm and the vocabulary-size target"}
  - {paper: "Provilkov, Emelianenko and Voita (2020)", title: "BPE-Dropout: Simple and Effective Subword Regularization", locator: "arXiv 1910.13267v2, Sections 2.1, 3, 5 and 6", note: "priority encoding with a merge table and its stochastic version"}
---

Byte-pair encoding (BPE) is a way to cut text into subword units.
It learns a list of merges from a corpus: start from single characters, and at each step join the adjacent pair of symbols that occurs most often.
To segment a word, it splits the word into characters and replays the merges in the order they were learned.
Common words end up as one symbol and rare words as a few pieces, so a fixed vocabulary of a few tens of thousands of symbols covers any word made of known characters.
It was introduced for neural machine translation and became the usual tokenizer of neural language models.

## In plain words

Write every word of a corpus as a string of letters, with a marker at the end.
Count which two neighbouring symbols appear together most often, say e and s, and glue them into one new symbol es everywhere.
Count again and glue again, a fixed number of times.
The glued symbols are the vocabulary, and the order of gluing is the recipe.
A new word is cut into letters, and the same glues are applied in the same order.
A word seen often in training comes out whole; a new word comes out as pieces the model has seen.

## Setting

A neural translation model reads and writes symbols from a fixed vocabulary, yet translation is an open-vocabulary problem: new names, compounds and word forms keep appearing.
Sennrich et al. (2016) handle rare and unseen words by writing them as strings of smaller subword pieces. {#c-se-open-vocab}
Their motivation is that many rare words can be translated piece by piece.
Of 100 rare German tokens they sampled, 56 are compounds, 21 names, 6 loanwords that share an origin with English, 5 transparent affixations, 1 a number and 1 a computer language identifier. {#c-se-rare-analysis}
The input of BPE is a table of words with their frequencies, taken from a corpus already split into words.
The output is an ordered list of merges and the symbol vocabulary they define.

## The method

**Origin.** Byte pair encoding began as Gage's (1994) compression scheme, which keeps substituting one spare byte value for whichever two-byte pair is commonest; Sennrich et al. adapt it to segmentation by merging characters and character strings instead of bytes. {#c-se-gage}

**Initialization.** The vocabulary starts as the set of characters, and each word is spelled out as its characters followed by an end-of-word marker, which allows the original word boundaries to be restored after translation. {#c-se-init}
This page writes the marker as `</w>`, as Algorithm 1 of Sennrich et al. does.

**Merging.** Tally every adjacent pair, take the commonest pair (A, B), and rewrite each of its occurrences as the new symbol AB; every merge thus yields a symbol standing for a character n-gram, and frequent words end up as single symbols. {#c-se-merge}
Pairs never cross a word boundary, so the counts can be taken over the word table, each word weighted by its frequency, instead of over the running text. {#c-se-dictionary}
The final vocabulary is the initial characters plus one symbol per merge, and the number of merges is the only hyperparameter. {#c-se-vocab-size}

**Training written out.** Let the table give each word $w$ a frequency $f(w) > 0$, and let $s(w)$ be its current symbol sequence.

1. Set $s(w) = (w_1, \dots, w_n, \texttt{</w>})$ for each word, and set the merge list $L$ to empty.
2. For each adjacent pair $(a, b)$, let $c(a, b) = \sum_w f(w) \cdot \#\{i : s(w)_i = a, s(w)_{i+1} = b\}$.
3. Pick $(a^*, b^*) = \arg\max c$; stop if no pair is left.
4. In every $s(w)$, scan from the left and replace each non-overlapping occurrence of $a^*$ followed by $b^*$ by the single symbol $a^* b^*$.
5. Append $(a^*, b^*)$ to $L$ and go back to step 2, until $L$ has the chosen number of merges.

Algorithm 1 of Sennrich et al. is a short Python program that does exactly this on the table low 5, lower 2, newest 6, widest 3 with 10 merges. {#c-se-algorithm}
It picks the pair with Python's `max` over a dictionary of counts, so a tie goes to the pair counted first, which is the pair met first when reading the table in order.
The implementations on this page use the same rule and test it against an independent regex version of Algorithm 1.

**Encoding.** At test time a word is split into characters, and the learned merges are applied to join them into larger known symbols; this works for any word. {#c-se-test-time}
A symbol can still be unknown at test time if it is an unknown character, or if every one of its training occurrences was later merged into a larger symbol, such as safeguar inside safeguard. {#c-se-unknown}
With the four merges r `</w>`, l o, lo w and e r`</w>` of their Figure 1, the unseen word lower becomes low er`</w>`. {#c-se-lower}

Later implementations encode with a priority instead of a pass per merge.
The merge table records which pairs join and with what priority, and encoding repeatedly joins the adjacent pair of highest priority until none is in the table. {#c-pr-priority}
The derivation below shows when the two procedures agree.

**BPE-dropout.** Provilkov et al. (2020) keep the merge table but, at each merge step of encoding, drop each candidate merge with probability $p$. {#c-pr-algorithm}
When a merge applies at several places in a word, as (e, r) does twice in merger, each occurrence is dropped or kept on its own. {#c-pr-occurrence}
With $p = 0$ this is standard BPE, with $p = 1$ every word stays as characters, and values in between set how fine the segmentation is. {#c-pr-p-limits}
They train with $p > 0$, usually 0.1, and use plain BPE at inference. {#c-pr-train-test}

## Derivation

Two derivations are written out with their checks: `derivations/merge-count.md` and `derivations/encode-reproduces.md`.
The first counts tokens and symbols.
Let $T$ be the number of symbols in the table, counting each word $f(w)$ times.
A merge of (A, B) lowers $T$ by exactly the number of occurrences it replaces, and that number is at least 1 and at most the pair count. {#c-merge-token-drop}
The two are equal unless A = B: a run of $n$ copies of A has $n - 1$ counted pairs, but the left-to-right scan replaces only $\lfloor n/2 \rfloor$ of them. {#c-merge-overlap}
Each merge adds one symbol and removes none, so after $k$ merges the vocabulary is the initial symbols plus the $k$ results, even when some of them no longer occur in the table; its size is the initial size plus $k$ when every result is a new string. {#c-vocab-count}
This is the count that Sennrich et al. state for the final vocabulary.

The second concerns encoding.
Each merge is a function on symbol sequences, so applying a merge list in order gives one segmentation per word. {#c-encode-deterministic}
Training applies the same functions, in the same order, to the same starting sequence of each training word, so encoding a training word reproduces its segmentation at the end of training. {#c-encode-reproduces}
Priority encoding gives the same result as encoding in order under two conditions on the list: each side of every merge is a character, the end marker or an earlier result, and every result is a new string. {#c-priority-equals-sequential}
The first condition holds for every learned list; the second held in all of the random trainings of the check, but it is not proved here.
Without the conditions the procedures differ: with the list (ab, c), (a, b), encoding abc in order gives ab c, while priority encoding gives abc. {#c-priority-counterexample}

## Worked example

Take the table of Algorithm 1: low 5, lower 2, newest 6, widest 3.
The initial symbols are the 10 letters d, e, i, l, n, o, r, s, t, w and `</w>`, 11 in all.
The table holds $5 \cdot 4 + 2 \cdot 6 + 6 \cdot 7 + 3 \cdot 7 = 95$ symbols.
In the first round, e s, s t and t `</w>` each have count 9, from newest and widest; e s is met first, so it wins the tie.

| Step | Merge | Count | Symbols after |
|---|---|---|---|
| 1 | e s | 9 | 86 |
| 2 | es t | 9 | 77 |
| 3 | est `</w>` | 9 | 68 |
| 4 | l o | 7 | 61 |
| 5 | lo w | 7 | 54 |
| 6 | n e | 6 | 48 |
| 7 | ne w | 6 | 42 |
| 8 | new est`</w>` | 6 | 36 |
| 9 | low `</w>` | 5 | 31 |
| 10 | w i | 3 | 28 |

Each step lowers the count by the count of its pair, since no pair here repeats a symbol.
The table ends as low`</w>` 5, low e r `</w>` 2, newest`</w>` 6 and wi d est`</w>` 3.
The vocabulary has $11 + 10 = 21$ symbols, but only 9 still occur in the table; es, est, lo, ne and new were all used up by later merges.
Encoding the unseen word lowest replays the list: e s, es t, est `</w>`, l o, lo w, and the result is low est`</w>`.
Encoding wider gives wi d e r `</w>`, because no merge joins e and r.

## Hyperparameters

The number of merges sets the vocabulary size.
Sennrich et al. used 59 500 merges for separate BPE and 89 500 for joint BPE. {#c-se-merges-used}
Kudo (2018) notes that BPE trades vocabulary size against the number of tokens per sentence, and that a small vocabulary, usually 16k to 32k, keeps sentences short. {#c-ku-vocab-size}
SentencePiece asks for the final vocabulary size instead of a merge count, since a merge count only makes sense for BPE. {#c-sp-vocab-size}

The second choice is whether to learn one merge list for both languages of a translation pair or one for each.
Sennrich et al. compare separate encodings with joint BPE, learned on the union of the two vocabularies, which makes the source and target segmentations more consistent. {#c-se-joint}
For English and Russian they transliterate the Russian vocabulary into Latin script with ISO-9, learn joint BPE, and transliterate the merges back into Cyrillic. {#c-se-iso9}

For BPE-dropout, $p$ is the third choice.
Provilkov et al. use 0.1 for all languages except Chinese and Japanese, where 0.6 gives a similar increase in sentence length. {#c-pr-p-chinese}

## Cost

Training as written above rescans the whole table at each merge, so $M$ merges over a table of $T$ symbols cost $O(MT)$.
Sennrich et al. index the pairs and update the counts incrementally, so a merge only touches the words that contain its pair.
For encoding, rescanning every pair after each merge costs $O(N^2)$ for an input of $N$ characters, and SentencePiece gets $O(N \log N)$ by keeping the candidate merges in a binary heap. {#c-sp-heap}
The merge list and vocabulary take memory linear in the number of merges.

## Evidence

On English to German newstest2015, the back-off dictionary baseline scores 22.0 BLEU as a single model and 24.2 as an ensemble of 8, the character-bigram system 22.8 and 25.3, and joint BPE 22.8 and 24.7. {#c-se-bleu}
The subword systems need no back-off dictionary, and their gain is largest on rare words.
Unigram F1 on rare words rises from 36.8% to 41.8% for English to German and from 26.5% to 29.7% for English to Russian. {#c-se-rare-f1}
On the German training corpus, unsegmented text has about 100 million tokens, 1 750 000 types and 1079 unknown tokens in newstest2013, while BPE has 112 million tokens, 63 000 types and no unknown tokens. {#c-se-table1}
Joint BPE can produce a few unknown segments that occur only in the English training text, 0.05% of test tokens. {#c-se-joint-unknown}

BPE-dropout during training, with plain BPE at inference, improves BLEU by up to 2.3 over BPE and by up to 0.9 over the subword regularization of Kudo (2018). {#c-pr-gains}
Against Kudo (2018) it is better on 8 of 12 datasets, by up to 0.92 BLEU, not significantly different on 3, and worse only on English to Japanese. {#c-pr-vs-kudo}
With $p = 0.1$, BPE-dropout makes training sentences about 1.25 times longer, but models trained with it produce translations of about the same length as models trained with BPE. {#c-pr-length}
Models trained with BPE-dropout are less sensitive to the vocabulary size: the gap between 4k and 32k vocabularies is much smaller than with plain BPE. {#c-pr-vocab-sensitive}
On subsets of WMT14 English to French, regularizing both sides is best for small and medium data, while from about 4 million sentence pairs it is better to use BPE-dropout only on the source side. {#c-pr-source-only}
Their embeddings place tokens near tokens that share characters: the character 4-gram precision of the top 10 neighbours is 0.29 with BPE-dropout against 0.18 with BPE. {#c-pr-4gram}
On test sets with synthetic misspellings, BPE-dropout gains 1.58 to 2.32 BLEU over BPE, against 0.45 to 1.5 on the clean source. {#c-pr-misspell}

## Limitations and failure modes

Segmentation is greedy and deterministic, so BPE gives one segmentation per word and no probability for other segmentations. {#c-ku-greedy}
Yet a word usually has many segmentations into symbols of the same vocabulary. {#c-ku-ambiguity}
Because only rare words get split, frequent character strings appear mostly inside larger tokens and seldom as tokens of their own, which Provilkov et al. argue hurts the learning of subword meaning. {#c-pr-unique}
BPE as first described needs text already split into words, which languages written without spaces do not provide.
The `@@` intra-word boundary marker of subword-nmt also cannot represent every whitespace pattern, such as consecutive spaces. {#c-sp-at-ambiguity}
Merges learned on one language serve other scripts poorly: a character never seen in training has no symbol.
A symbol can be in the vocabulary but never produced, when all its occurrences were merged further, which wastes embedding rows.
A large vocabulary on a small dataset makes many symbols rare, and Provilkov et al. report that this hurts BPE baselines.

## Variants

SentencePiece trains directly on raw sentences, with no pre-tokenization. {#c-sp-raw}
It treats the text as a sequence of Unicode characters, writes the space as the visible symbol U+2581 so that decoding inverts encoding exactly, and supports both BPE and the unigram model. {#c-sp-lossless}
It normalizes text with Unicode NFKC by default. {#c-sp-nfkc}
Kudo (2018) proposes a unigram language model instead of merges, which assumes subwords occur independently and gives each segmentation a probability. {#c-ku-unigram}
His subword regularization samples one segmentation for each parameter update. {#c-ku-sampling}
He treats WordPiece as a variant of BPE that grows its vocabulary the same way but chooses merges by likelihood instead of frequency. {#c-ku-wordpiece}
BPE-dropout keeps BPE and its merge table and only changes encoding during training.

## When to use it

Use BPE when a model needs a fixed vocabulary but must read and write any word, as in translation and language modelling.
Use joint BPE when the two languages share names and cognates, so that a name is cut the same way on both sides.
Use the SentencePiece form when the text is not split into words or when the tokenizer must round-trip whitespace exactly.
Use BPE-dropout on small or medium training sets, on the source side for large ones, or when inputs are noisy.
Prefer the unigram model when the training method needs segmentation probabilities.

## Pitfalls

Apply the merges in the order they were learned, or with priorities equal to that order; a merge list without its order is not a tokenizer.
Keep the training and test preprocessing identical, including normalization, lower-casing and the end-of-word marker.
Count overlapping pairs correctly: aaa has two (a, a) pairs, but one merge joins only one of them.
Write down the tie-breaking rule; a different rule gives a different merge list on the same table.
Do not use BPE-dropout at inference; Provilkov et al. use $p = 0$ there.
Do not compare perplexities across tokenizers without normalizing by characters or words, because the number of tokens per sentence changes.

## History

Gage published byte pair encoding in 1994 as a compression method for bytes.
Sennrich, Haddow and Birch (2016) adapted it to words and characters for neural machine translation and found that subword models handle rare words better than a back-off dictionary.
Kudo (2018) found that sampling segmentations from a unigram model helps translation, especially on small and out-of-domain data, and Kudo and Richardson (2018) released SentencePiece.
Provilkov, Emelianenko and Voita (2020) found that BPE itself can produce multiple segmentations, by dropping merges at random during training.
