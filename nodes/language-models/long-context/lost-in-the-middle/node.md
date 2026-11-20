---
tag: NEW-lost-in-the-middle
kind: finding
slug: lost-in-the-middle
name: Lost in the middle
aliases: [lost-in-the-middle, lost in the middle effect, U-shaped position curve, positional bias in long contexts, primacy and recency bias in language models, serial-position effect in language models]
area: language-models/long-context
level: core
summary: >-
  Language models use information at the start and end of a long input better than information in the middle, so accuracy against the position of the relevant text is U-shaped.
related_to: [t5, perplexity, few-shot-in-context-learning, reinforcement-learning-from-human-feedback, chain-of-thought-prompting]
history:
  year: 2023
  note: "Liu et al. posted the finding as arXiv 2307.03172 in July 2023 and published it in TACL in February 2024; Peysakhovich and Lerer (October 2023) replicated it, and in 2024 Hsieh et al. (Findings of ACL), Zhang et al. and An et al. traced it to attention and position biases and proposed fixes."
prereqs: [decoder-only-transformer, multi-head-attention, rotary-position-embedding]
same_as:
  arxiv: ["2307.03172", "2406.16008", "2310.01427", "2403.04797", "2404.16811", "2402.14848", "2404.06654", "2309.17453"]
sources:
  - {paper: "Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni and Liang (2024)", title: "Lost in the Middle: How Language Models Use Long Contexts", locator: "TACL 12, pp. 157 to 173, PDF pp. 1 to 17", note: "the finding, the two tasks, the controls and the appendix experiments"}
  - {paper: "Hsieh, Chuang, Li, Wang, Le, Kumar, Glass, Ratner, Lee, Krishna and Pfister (2024)", title: "Found in the middle: Calibrating Positional Attention Bias Improves Long Context Utilization", locator: "Findings of ACL 2024, pp. 14982 to 14995, PDF pp. 1 to 7", note: "the U-shaped attention bias, the additive model and attention calibration"}
  - {paper: "Peysakhovich and Lerer (2023)", title: "Attention Sorting Combats Recency Bias In Long Context Language Models", locator: "arXiv 2310.01427v1, pp. 1 to 5", note: "SynthWiki, the replication, the attention measurements and attention sorting"}
  - {paper: "Zhang, Chen, Liu, Yao, Ruwase, Chen, Wu and Wang (2024)", title: "Found in the Middle: How Language Models Use Long Contexts Better via Plug-and-Play Positional Encoding", locator: "arXiv 2403.04797v1, pp. 1 to 4", note: "the causal-attention and RoPE-decay account and multi-scale positional encoding"}
  - {paper: "An, Ma, Lin, Zheng and Lou (2024)", title: "Make Your LLM Fully Utilize the Context", locator: "arXiv 2404.16811v2, pp. 1 to 8", note: "the training-data account, IN2 training and VAL Probing"}
  - {paper: "Levy, Jacoby and Goldberg (2024)", title: "Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models", locator: "ACL 2024, pp. 15339 to 15353, PDF pp. 1 and 2", note: "reasoning degrades with padding at 3000 tokens"}
  - {paper: "Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Zhang and Ginsburg (2024)", title: "RULER: What's the Real Context Size of Your Long-Context Language Models?", locator: "COLM 2024, arXiv 2404.06654v3, pp. 1 and 6", note: "effective against claimed context length"}
  - {paper: "Xiao, Tian, Chen, Han and Lewis (2024)", title: "Efficient Streaming Language Models with Attention Sinks", locator: "ICLR 2024, arXiv 2309.17453v4, p. 2", note: "attention sinks on the initial tokens"}
claim: >-
  When a language model must use one relevant passage among many in a long input, its accuracy is often highest when the passage is at the beginning or end of the input and lowest when it is in the middle, even for models built for long contexts.
first_reported: {work: liu-2024-lost-in-the-middle, note: "multi-document question answering and key-value retrieval with six open and closed models, with controls for distractors, order, query placement, model size and instruction tuning"}
replications:
  - {work: peysakhovich-2023-attention-sorting-recency-bias, note: "SynthWiki, fictional biographies, so that prior knowledge cannot help; the curve replicates, with the primacy side weaker in longer contexts"}
  - {work: hsieh-2024-found-in-the-middle-calibrating-attention, note: "Vicuna-7b-v1.5-16k on 20 NaturalQuestions documents, with wrong answers biased toward the first document"}
  - {work: zhang-2024-found-in-the-middle-ms-poe, note: "the task of Liu et al. with ten documents on Llama-2-7B-Chat and Vicuna-7B, used to compare position re-scalings"}
  - {work: an-2024-make-your-llm-fully-utilize-context, note: "a min-max gap across positions on synthetic document, code and database probes, 56.2 points for Mistral-7B-Instruct-v0.2 and 32.1 for GPT-4-Turbo"}
scale_range:
  - {range: "MPT-30B, LongChat-13B, GPT-3.5-Turbo, Claude-1.3 and GPT-4, with contexts of 10 to 30 documents or 75 to 300 key-value pairs", work: liu-2024-lost-in-the-middle}
  - {range: "Llama-2 7B, 13B and 70B; the 7B models show recency only", work: liu-2024-lost-in-the-middle}
  - {range: "7B models built on Llama-2 at 1,000 to 30,000 tokens", work: peysakhovich-2023-attention-sorting-recency-bias}
failures:
  - {work: liu-2024-lost-in-the-middle, note: "Claude-1.3 is nearly perfect on key-value retrieval at every length and position, Flan-UL2 inside its 2048-token training window has a best-to-worst gap of 1.9%, and the query placed before the data removes the effect on key-value retrieval"}
  - {work: an-2024-make-your-llm-fully-utilize-context, note: "training on synthesized questions whose answers sit anywhere in a 32K context cuts the gap of Mistral-7B from 56.2 to 13.9 points"}
explanations:
  - {name: "a positional attention bias", summary: "Attention favours the first and last tokens whatever their relevance, and the model tends to use the documents it attends to.", sources: [hsieh-2024-found-in-the-middle-calibrating-attention, peysakhovich-2023-attention-sorting-recency-bias, xiao-2024-efficient-streaming-attention-sinks]}
  - {name: "causal masking and RoPE decay", summary: "Early tokens are attended by every later token, and rotary position embeddings weaken attention over long distances.", sources: [zhang-2024-found-in-the-middle-ms-poe, peysakhovich-2023-attention-sorting-recency-bias]}
  - {name: "a bias in the training data", summary: "Next tokens mostly depend on nearby text, and instructions usually come first, so training seldom rewards using the middle.", sources: [an-2024-make-your-llm-fully-utilize-context, peysakhovich-2023-attention-sorting-recency-bias]}
---

Lost in the middle is the finding that a language model given a long input uses information at its start and its end better than information in its middle.
Liu et al. observe that performance is often highest when relevant information occurs at the beginning or end of the input context, and significantly degrades when models must access relevant information in the middle of long contexts, even for explicitly long-context models. {#c-liu-u}
Plotted against the position of the relevant text, accuracy traces a U.
The finding matters because a long context window is only useful if the model can use what is inside it, and retrieval-augmented systems routinely place the passage that holds the answer somewhere in a long list.

## In plain words

Give a model twenty search results and a question, where exactly one of the results holds the answer.
Move that one result from the top of the list to the bottom, one place at a time, and ask the same question each time.
If the model read its input evenly, the score would not depend on where the answer sits.
It does: the score is high when the answer is first, lower when it is in the middle, and rises again when it is last.
For one widely used model the worst score was below the score the same model gets with no documents at all, from what it remembers.

Psychologists know the same shape from people: asked to recall a list, they remember its first and last items best.
In a transformer nothing in the attention mechanism forces this, since any token can attend to any other.
The bias comes from how the models are built and trained, and it can be measured, partly corrected after training, and trained away.

## The finding

Liu et al. analyze the performance of language models on two tasks that require identifying relevant information in their input contexts: multi-document question answering and key-value retrieval. {#c-liu-tasks}
They run controlled experiments with open models, MPT-30B-Instruct and LongChat-13B (16K), and closed models, OpenAI's GPT-3.5-Turbo and Anthropic's Claude-1.3. {#c-liu-models}
They call the advantage of relevant information at the very beginning of the input primacy bias and the advantage at its end recency bias, and the resulting curve U-shaped. {#c-liu-primacy}
Their standard for robust use of a long context is explicit: to claim that a language model can robustly use information within long input contexts, its performance should be minimally affected by the position of the relevant information, for example a minimal difference in best- and worst-case performance. {#c-liu-protocol}

In the multi-document task, the model inputs are a question to answer and $k$ documents, where exactly one of the documents contains the answer to the question and $k - 1$ distractor documents do not. {#c-liu-mdqa}
They instantiate it with NaturalQuestions-Open, taking the 2655 queries where the annotated long answer is a paragraph, and use Wikipedia passages of at most 100 tokens as documents. {#c-liu-nq}
The distractors are the $k - 1$ Wikipedia chunks most relevant to the query that do not contain any of the annotated answers, found with Contriever fine-tuned on MS-MARCO. {#c-liu-distractors}
To move the relevant information they reorder the documents, and to change the input length they add or remove distractors. {#c-liu-modulate}
A prediction counts as correct if any of the annotated answers appears in the output. {#c-liu-accuracy}
The setup is similar to the needle-in-a-haystack experiments of Ivgi et al. (2023), who compare question answering with the relevant paragraph at the beginning of the input or at a random position, and find that encoder-decoder models do significantly better when it is at the start. {#c-liu-ivgi}

The models differ in how they represent position.
MPT-30B-Instruct has a maximum context length of 8192 tokens and uses ALiBi; LongChat-13B (16K) extends the LLaMA-13B context window from 2048 to 16384 tokens with condensed rotary positional embeddings before fine-tuning on 16384-token sequences. {#c-liu-windows}
Two baselines bracket the results.
With no documents, closed-book, GPT-3.5-Turbo answers 56.1% and Claude-1.3 48.3% of the questions; with only the document that holds the answer, oracle, they answer 88.3% and 76.1%. {#c-liu-closed-book}

## Evidence

The generated list gives each source with its label and its role.
The lines of evidence are these.

**Multi-document question answering.**
With 10, 20 and 30 documents, every model traces a U against the position of the answer document.
GPT-3.5-Turbo's performance can drop by more than 20%, and in the worst case, in the 20- and 30-document settings, it is lower than with no input documents at all, the closed-book 56.1%. {#c-liu-drop}
A longer context window does not help when the input already fits: GPT-3.5-Turbo and GPT-3.5-Turbo (16K) give nearly superimposed curves in the 10- and 20-document settings. {#c-liu-extended}
On a subset of 500 examples with 20 documents, GPT-4 (8K) scores higher than any other model tested but still shows a U-shaped curve. {#c-liu-gpt4}

**Key-value retrieval.**
The second task removes language understanding from the picture.
The input is a string-serialized JSON object with $k$ key-value pairs, each key and value a unique random UUID, and one key; the model must return the value of that key. {#c-liu-kv}
With 75, 140 and 300 pairs and 500 examples each, Claude-1.3 and Claude-1.3 (100K) are nearly perfect at every length, while GPT-3.5-Turbo, its 16K version and MPT-30B-Instruct do worst when the pair sits in the middle. {#c-liu-kv-results}
So even exact string matching, which a model can do perfectly at the edges of its input, fails in the middle for some models.

**Controls.**
The U is not an artefact of hard distractors.
With random rather than retrieved distractors all models have higher accuracy, but they still struggle to reason over their entire input, so the drop is not solely due to an inability to identify relevant documents. {#c-liu-random-distractors}
Nor is it learned from the order of the search results: with the distractors shuffled and the prompt saying that the results are ordered randomly, the curve is still U-shaped. {#c-liu-random-order}

**Replication on unseen facts.**
A model that already knows the answer might not need the document.
Peysakhovich and Lerer use one-paragraph biographies of fictional individuals, generated by a language model, and ask simple questions that the right biography answers. {#c-pey-synthwiki}
The distractors are biographies of other fictional people, and since all the data is generated, prior knowledge cannot help; they call this the SynthWiki task. {#c-pey-no-prior}
More distractors degrade every model they test, least Claude-1-Instant and Claude-2, more GPT-3.5 and TogetherLlama, and catastrophically the remaining open models. {#c-pey-degrade}
They see a general replication of the lost-in-the-middle curve, but as the context grows, and in TogetherLlama in particular, the primacy side shrinks and mostly the recency bias remains. {#c-pey-replication}
Hsieh et al. reproduce it with Vicuna-7b-v1.5-16k on 20 documents, and find that its wrong answers are strongly biased toward the document in the first position, whatever the location of the gold document and whether the documents keep their retrieval order or are shuffled. {#c-hs-reproduce}
Zhang et al. use the task of Liu et al. with ten documents to compare models, reporting the accuracy averaged over the ten positions of the answer and the gap between the best and worst position. {#c-zh-gap}
An et al. measure the same gap, the difference between the maximum and minimum performance over positions, as their second metric beside the average, a smaller gap meaning a more robust model. {#c-an-gap}

**Neighbouring findings.**
Position is not the only problem with long inputs.
Levy et al. isolate the effect of length by taking the same sample and padding it with text of different lengths, types and locations. {#c-levy-design}
They find that models degrade in reasoning even at 3000 tokens, far below their technical maximum, with accuracy falling from 0.92 to 0.68 on average over the models tested. {#c-levy-drop}
On RULER, almost every model is nearly perfect on the plain needle-in-a-haystack test, yet degrades sharply as the context grows; of the models that claim 32K tokens or more, only half keep satisfactory performance at 32K. {#c-ruler-niah}
RULER calls the longest length at which a model stays above the score of Llama2-7b at 4K tokens its effective length, and reports it beside the claimed length. {#c-ruler-effective}
An et al. argue that near-perfect needle-in-a-haystack scores may overstate long-context ability, because the test uses a document-style context familiar from pre-training and a forward retrieval pattern, in which the answer directly follows the keyword. {#c-an-niah}

## Scope and conditions

**Inside or beyond the training length.**
The effect depends on whether the input fits the length the model was trained on.
Within its 2048-token training window, the encoder-decoder Flan-UL2 is fairly robust to position, with a 1.9% gap between best and worst case; on longer inputs it too degrades in the middle, and Flan-T5-XXL shows a similar trend. {#c-liu-encdec}
Liu et al. hypothesize that encoder-decoder models use their windows better because the bidirectional encoder reads each document in the light of later ones. {#c-liu-bidirectional}
The T5 page describes the encoder-decoder architecture of these models.

**Where the query goes.**
A decoder-only model reads the documents before it reads a question placed at the end, so it cannot attend to the documents with the question in mind.
Placing the query both before and after the data, query-aware contextualization, gives every model near-perfect key-value retrieval at 75, 140 and 300 pairs; GPT-3.5-Turbo (16K) becomes perfect at 300 pairs, against a worst case of 45.6% without it. {#c-liu-query-aware}
On multi-document question answering it changes little, helping slightly when the answer is first and hurting slightly elsewhere. {#c-liu-query-aware-qa}

**Instruction tuning.**
Instructions normally come first in the prompt, which could teach the model to look there.
Yet the base MPT-30B is U-shaped as well as MPT-30B-Instruct; instruction tuning only narrows the best-to-worst gap from nearly 10% to around 4%. {#c-liu-base}
For Llama-2 13B, the base model's 20-point gap shrinks to 10 points after supervised fine-tuning and RLHF, while the 70B models show similar curves with and without the additional fine-tuning. {#c-liu-llama-13b}
The reinforcement learning from human feedback page describes that fine-tuning stage.

**Model size.**
The U needs scale: the 7B Llama-2 models are solely recency-biased, and the 13B and 70B models are U-shaped. {#c-liu-llama-size}
Liu et al. hypothesize that earlier work saw no primacy bias in language models because the models it studied were too small, under 1B parameters. {#c-liu-small}

**Chain of thought.**
Asking the model to reason step by step does not remove the length effect.
Chain-of-thought prompting adds a roughly constant amount of accuracy at every input length, far from closing the drop; only for GPT-4 does the gap between chain-of-thought and direct prompting grow as the input lengthens. {#c-levy-cot}
The chain-of-thought prompting page covers the method.

**Perplexity is not enough.**
Many long-context evaluations report perplexity on web text as a proxy for long-context ability, and Liu et al. note that precise access to knowledge in a long context may be a separate challenge. {#c-liu-perplexity}
Levy et al. report that next-word prediction correlates negatively with performance on their reasoning dataset. {#c-levy-ppl}
The perplexity page explains why a model can predict the next token well while using little of the distant context.

## Explanations

**A positional attention bias.**
The claim: attention itself is biased toward the edges of the input, and the model tends to use what it attends to.
Hsieh et al. find that the tokens at the beginning and end of the input receive higher attention, regardless of their relevance. {#c-hs-attn-u}
The U-shaped attention persists after the documents are randomly shuffled, which suggests that it does not depend on their content. {#c-hs-shuffled}
In the examples the model gets wrong, the document most likely used in the answer lies in the half that receives more attention in 526 cases (74%) and in the other half in 186 (26%). {#c-hs-used}
At long lengths, the three open models of Peysakhovich and Lerer attend strongly to the most recent tokens and more weakly to the first tokens, and all attend much more to relevant documents than to distractors. {#c-pey-attention}
Xiao et al. find that a large share of attention goes to the first tokens of autoregressive models whatever their relevance, and attribute it to the softmax, whose scores must sum to one, together with the first tokens being visible to almost every later token. {#c-xiao-sink}
The consequence: the relevance signal is in the attention but mixed with a position term, and removing that term should recover it.

**Causal masking and RoPE decay.**
The claim: the architecture favours the beginning and the near past.
Zhang et al. name two factors: causal attention, under which earlier tokens take part in more attention computations so that models favour initial tokens, and the long-term decay of RoPE, which lowers the attention score of distant tokens. {#c-zh-factors}
Peysakhovich and Lerer add that RoPE has an inductive bias toward reduced attention at long distances, which may make it easier still to learn to attend to recent tokens. {#c-pey-rope}
The rotary position embedding page derives the decay.
The consequence: changing how positions are fed to the model at inference should move the curve.

**A bias in the training data.**
The claim: training rarely rewards the model for using the middle of a long input.
An et al. hypothesize that in pre-training the loss on the next token is mostly influenced by a few nearby tokens, and that in supervised fine-tuning the system message usually sits at the start of the context. {#c-an-hypothesis}
Peysakhovich and Lerer make the same point about pre-training documents, where the most informative tokens for predicting a token are typically the most recent ones. {#c-pey-prior}
The few-shot in-context learning page discusses other ways in which the order and position of text in a prompt change what a model does.
The consequence: training data that puts the needed information anywhere in a long context should remove the bias.

Liu et al. tie the finding to the serial-position effect in psychology (Ebbinghaus, 1913; Murdock, 1962): in free recall of the elements of a list, humans tend to remember the first and last elements best. {#c-liu-serial}
They find a similar effect in language models perhaps surprising, since the self-attention of a transformer is technically equally capable of retrieving any token in its context. {#c-liu-surprising}
None of the three explanations needs the human analogy, and they are not exclusive: the attention bias may be the mechanism through which the architecture and the data act.

## Implications

**For retrieval-augmented generation.**
Feeding a model more retrieved documents has diminishing returns.
In open-domain question answering with retrieved Wikipedia documents, 50 documents instead of 20 improve accuracy by only about 1.5% for GPT-3.5-Turbo and about 1% for Claude-1.3, so the reader saturates long before the retriever's recall does. {#c-liu-open-domain}
Liu et al. suggest reranking, which pushes relevant documents toward the start of the input, and truncating the ranked list, which retrieves fewer documents when that suffices. {#c-liu-rerank}

**Correcting attention after training.**
Hsieh et al. model the attention to a document as $\mathrm{rel}(x) + \mathrm{bias}(k) + \epsilon$, relevance plus a positional bias plus noise; the linear model reaches a Spearman rank correlation of 0.76 on quadruplets from NaturalQuestions. {#c-hs-additive}
On 100 sampled examples with 20 documents, 83% of pairs satisfy the condition that attention rises with the bias at a fixed document, and 72% the condition that it rises with relevance at a fixed position. {#c-hs-conditions}
They subtract the attention that a dummy document receives at the same position, $\mathrm{Attn}(x, k) - \mathrm{Attn}(x^{\mathrm{dum}}, k)$, and call the result calibrated attention and the method found-in-the-middle. {#c-hs-calibration}
The derivation position-bias-models works out what this subtraction buys under their model.
With no noise, the calibrated attention equals $\mathrm{rel}(x) - \mathrm{rel}(x^{\mathrm{dum}})$ and does not depend on the position. {#c-pb-calibration}
With noise of size at most $\delta$, it is within $2\delta$ of that difference, and two documents whose relevance differs by more than $4\delta$ are ranked correctly wherever they sit. {#c-pb-calibration-noise}
As a ranker of retrieved documents, calibrated attention beats existing approaches by up to 48 points. {#c-hs-rank}
Used to steer generation, it beats the uncalibrated model by up to 10 points across tasks and models, and by 6 to 10 points when the gold document is mid-sequence. {#c-hs-rag}
Hsieh et al. conclude that, calibrated appropriately, models can locate relevant information even in the middle of the input. {#c-hs-capable}

**Sorting by attention.**
Attention sorting performs one step of decoding, sorts the documents by the attention they receive with the highest last, repeats, and then generates the answer from the sorted context. {#c-pey-sorting}
It more than doubles the accuracy of WizardCoder and YaRN at 30k tokens, especially when repeated. {#c-pey-doubling}
One sort moves the relevant document toward the end on average, but a document far from the end may stay in the middle; TogetherLlama gains little after two sorts, while the other models improve up to five. {#c-pey-one-sort}
The derivation explains both observations in an idealized model where attention rises with relevance and does not fall with position.
After each sort, every document that was before the gold document is still before it, so the gold document never moves earlier. {#c-pb-sort-monotone}
With attention $\mathrm{rel} + \beta k$, one sort puts a gold document at position $p$ of $K$ last when its relevance exceeds that of every other document by more than $\beta(K - p)$. {#c-pb-sort-one-step}
Without that margin sorting can stall: with attention $\mathrm{rel} + 0.2k$, relevances 1.0, 0.9 and 0.9 and the gold document first, the attentions are 1.2, 1.3 and 1.5, already sorted, and the gold document stays first however often the sort is repeated. {#c-pb-sort-stall}

**Re-scaling positions.**
Zhang et al. replace the position $m$ in RoPE by $m/r$, squeezing distant tokens into a shorter range, which can weaken the long-term decay by a factor of $r$. {#c-zh-rescale}
Because RoPE scores depend only on the difference of positions, dividing every position by $r$ makes the score at distance $D$ equal to the unscaled score at distance $D/r$. {#c-pb-rescale}
With one ratio for the whole model, a ratio of 0.5 favours the most recent text and 2.5 the beginning, and the average accuracy improves near 1.5. {#c-zh-favored}
Different attention heads react differently to moving the relevant document, so their multi-scale positional encoding, Ms-PoE, gives each head its own ratio, rising from position-aware to position-unaware heads. {#c-zh-heads}
Without fine-tuning, Ms-PoE gains up to 3.8 points of average accuracy on ZeroSCROLLS. {#c-zh-result}

**Training the bias away.**
Information-intensive (IN2) training builds question-answer pairs whose answer needs a short segment of about 128 tokens placed in a synthesized context of 4K to 32K tokens, or the combination of two or more such segments. {#c-an-in2}
Mistral-7B trained this way, FILM-7B, raises its F1 on NarrativeQA from 23.5 to 26.9 while its MMLU accuracy stays comparable, 59.3 against 59.2. {#c-an-film}
On their VAL Probing tasks, FILM-7B averages 85.9 with a min-max gap across positions of 13.9, against 47.3 and 56.2 for Mistral-7B-Instruct-v0.2 and 79.0 and 32.1 for GPT-4-Turbo. {#c-an-val}

## Open questions

How much of the U comes from the architecture and how much from the data is unsettled: both the RoPE account and the training-data account predict a recency bias, and neither alone explains why primacy appears only above some model size.
Calibration and sorting assume that the model's attention to the right document is still higher than to a wrong one at the same position; the 72% of pairs that satisfy this in Hsieh et al. leaves room for cases where it is not.
Synthetic retrieval tests such as needle-in-a-haystack and key-value lookup can be solved by models that still fail on realistic tasks, and there is no agreed benchmark that separates position effects from length effects and from reasoning load.
Whether a model trained to remove the gap on synthetic probes also uses the middle of real documents, codebases and conversations as well as their edges remains to be shown at scale.

## History

The serial-position effect in human memory goes back to Ebbinghaus (1913) and Murdock (1962), who found that people recall the first and last items of a list best.
Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni and Liang posted "Lost in the Middle: How Language Models Use Long Contexts" as arXiv 2307.03172 in July 2023, when context windows of 16K to 100K tokens had just become available, and published it in Transactions of the Association for Computational Linguistics in February 2024.
Peysakhovich and Lerer posted attention sorting and the SynthWiki dataset in September 2023.
Xiao et al. described attention sinks in September 2023 and presented them at ICLR 2024.
In 2024, Levy et al. (ACL) and RULER (COLM) widened the question from position to length, Zhang et al. proposed multi-scale positional encoding, An et al. proposed IN2 training, and Hsieh et al. (Findings of ACL) proposed attention calibration.

## References
