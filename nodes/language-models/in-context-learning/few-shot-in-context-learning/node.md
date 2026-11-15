---
tag: NEW-few-shot-in-context-learning
kind: concept
slug: few-shot-in-context-learning
name: In-context learning
aliases: [ICL, few-shot prompting, few-shot learning from demonstrations, zero-shot and one-shot prompting]
area: language-models/in-context-learning
level: core
summary: >-
  A trained model doing a new task from instructions or examples placed in its input, with no change to its weights: the prompt plays the part of a training set.
uses: [ordinary-least-squares@m2, ridge-regression@m2, stochastic-gradient-descent@m2, bayes-estimator@m2]
related_to: [bayesian-inference@m2, kernel-regression@m2, k-nearest-neighbors@al, double-descent@al, lasso@m2]
history:
  year: 2020
  people: [Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal]
  note: "Brown et al., Language Models are Few-Shot Learners, arXiv 2005.14165 (May 2020), NeurIPS 2020, named and measured zero-, one- and few-shot in-context learning in GPT-3. Garg, Tsipras, Liang and Valiant (arXiv 2208.01066, 2022) studied it on function classes; von Oswald et al. (arXiv 2212.07677, ICML 2023) related it to gradient descent; Min et al. (arXiv 2202.12837, 2022) tested what in the demonstrations matters."
prereqs: [ordinary-least-squares@m2, ridge-regression@m2, stochastic-gradient-descent@m2, bayes-estimator@m2, multivariate-normal-distribution@m2]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["2005.14165", "2208.01066", "2212.07677", "2202.12837"]
  al: null
sources:
  - {paper: "Brown et al. (2020)", title: "Language Models are Few-Shot Learners", locator: "arXiv 2005.14165v4, pp. 4 to 7 and p. 34", note: "the term, the zero-, one- and few-shot settings, the scaling of in-context learning, and its ambiguity"}
  - {paper: "Garg et al. (2022)", title: "What Can Transformers Learn In-Context? A Case Study of Simple Function Classes", locator: "arXiv 2208.01066v3, equations 1 and 2, pp. 1 to 11, figure 2", note: "the definition on function classes and the linear-regression experiments"}
  - {paper: "von Oswald et al. (2023)", title: "Transformers Learn In-Context by Gradient Descent", locator: "arXiv 2212.07677v2, equations 3 and 4, Proposition 1, pp. 4, 5, 8 and 13, figure 2", note: "one linear self-attention layer as one step of gradient descent"}
  - {paper: "Min et al. (2022)", title: "Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?", locator: "arXiv 2202.12837v2, pp. 2 to 9", note: "random labels, input distribution, label space and format"}
---

In-context learning is what a pretrained model does when it carries out a task that it is shown only in its input.
Brown et al. (2020) use the term for a pretrained language model that is given an instruction, some worked demonstrations, or both, and is then asked to handle new cases of the same task just by continuing the text. {#c-brown-term}
No weights change: the demonstrations are given as conditioning, and there are no gradient updates or fine-tuning. {#c-brown-no-gradient}
Whether the model learns anything new from the demonstrations, or only recognizes a task it met in training, is the question that most of the later work on the subject asks.

## In plain words

You write a few lines of the form "sea otter => loutre de mer", then "cheese =>", and the model writes "fromage".
Nobody trained it on this list; the list is in the prompt, and the model continues the pattern.
The examples in the prompt do the job of a training set, the model's forward pass does the job of the learning algorithm, and the answer to the last line is the prediction.

When there are no examples, only an instruction, this is called zero-shot; with one example it is one-shot; with a few it is few-shot.
Brown et al. call the whole family in-context learning and keep it apart from fine-tuning, which changes the weights. {#c-brown-forward-passes}

## The idea

Brown et al. frame in-context learning as the inner loop of meta-learning: pretraining is the outer loop, which gives the model a broad set of skills, and in-context learning is the inner loop, which uses them on the task in the prompt. {#c-brown-inner-loop}
They say on purpose that the terms do not decide whether the model learns a new task from scratch at inference time or recognizes patterns it saw in training. {#c-brown-agnostic}

In the few-shot setting the model sees $K$ examples of a context and its completion, then one more context, and must provide the completion; no weight updates are allowed. {#c-brown-few-shot}
For GPT-3, $K$ is typically between 10 and 100, as many examples as fit in the context window of 2048 tokens. {#c-brown-k}
One-shot allows a single demonstration, in addition to a natural language description of the task. {#c-brown-one-shot}
Zero-shot allows no demonstrations, only an instruction that describes the task. {#c-brown-zero-shot}

Garg et al. (2022) give a definition that does not depend on language.
A model in-context learns when it conditions on a prompt of examples and a new query input and generates the output for the query, at inference time and with no parameter updates. {#c-garg-no-updates}
They make it precise for a class of functions, and note that the ability is a property of the model alone, however it was trained; training a model to have it can then be viewed as meta-learning. {#c-garg-property}

This turns a vague question into a test: if the prompt holds input-output pairs of a linear function, does the model's answer match what a good learning algorithm would output from those pairs?
For linear functions the good algorithms are known exactly, and von Oswald et al. (2023) proved that one layer of a simplified Transformer can run one of them, a step of gradient descent.

## Formal statement

**Definition (Garg et al.).** Let $\mathcal{D}_\mathcal{F}$ be a distribution over functions and $\mathcal{D}_\mathcal{X}$ a distribution over inputs.
A prompt is $P = (x_1, f(x_1), \dots, x_k, f(x_k), x_{\text{query}})$ with $f \sim \mathcal{D}_\mathcal{F}$ and the inputs drawn i.i.d. from $\mathcal{D}_\mathcal{X}$.
A model $M$ can in-context learn the class $\mathcal{F}$ up to $\epsilon$ if

$$ \mathbb{E}_P \big[ \ell\big(M(P), f(x_{\text{query}})\big) \big] \le \epsilon . $$

This is equation 1 of Garg et al. {#c-garg-definition}
They train a model $M_\theta$ for this by minimizing the average of $\ell(M_\theta(P^i), f(x_{i+1}))$ over $i = 0, \dots, k$, where the prefix $P^i$ holds the first $i$ examples and the next input, so that one sequence trains every prompt length at once. {#c-garg-objective}

**In-context least squares.** Take $N$ examples $(x_i, y_i)$ with $x_i \in \mathbb{R}^{d_x}$, $y_i \in \mathbb{R}^{d_y}$ and the loss $L(W) = \frac{1}{2N} \sum_i \lVert W x_i - y_i \rVert^2$ of a linear model $W$.
One gradient step with learning rate $\eta$ gives

$$ \Delta W = -\frac{\eta}{N} \sum_{i=1}^{N} (W x_i - y_i)\, x_i^\top . $$ {#eq-gd-step}

This is equation 3 of von Oswald et al. {#c-oswald-gd-step}
They observe that the loss after the step equals the loss of the old weights on the shifted targets $y_i - \Delta W x_i$, so a gradient step can be read as a change of the data rather than of the weights (their equation 4). {#c-oswald-data-view}
That reading is what an attention layer can carry out, since it can change the tokens but not its own weights.

**Linear self-attention.** Write each example as a token $e_i = (x_i, y_i)$.
A linear self-attention layer, with no softmax, maps a token $e$ to

$$ e \;\mapsto\; e + P W_V \sum_{i=1}^{N} e_i\, e_i^\top W_K^\top W_Q\, e . $$ {#eq-lsa}

**Proposition (von Oswald et al., Proposition 1).** There are key, query, value and projection matrices for which this layer changes every token exactly as the gradient step changes the data, $e_j \leftarrow (x_j, y_j - \Delta W x_j)$, and likewise for the query token. {#c-oswald-prop1}
In their appendix the matrices are $W_K = W_Q = \begin{pmatrix} I_x & 0 \\ 0 & 0 \end{pmatrix}$, $W_V = \begin{pmatrix} 0 & 0 \\ W_0 & -I_y \end{pmatrix}$ and $P = \frac{\eta}{N} I$, where $W_0$ is the starting weight of the linear model. {#c-oswald-weights}
Only the in-context tokens supply keys and values; the query token is left out, a small departure from full self-attention that does not matter when $W_0$ is close to zero. {#c-oswald-context-only}
The construction is not unique: only $P W_V$ and $W_K^\top W_Q$ matter, and with no nonlinearity, scaling one by $s$ and the other by $1/s$ gives the same layer. {#c-oswald-unique}

The derivation on this page, `derivations/lsa-gradient-step.md`, proves the following and checks each part numerically.
With the construction, one layer maps every context token $(x_j, y_j)$ to $(x_j, y_j - \Delta W x_j)$, where $\Delta W$ is the step of eq-gd-step from $W_0$. {#c-lsa-step}
A query token that starts as $(x_q, -W_0 x_q)$ ends as $(x_q, -(W_0 + \Delta W) x_q)$, so minus its second block is the prediction of the model after one step. {#c-lsa-readout}
Stacking $K$ layers with the same weights, each reading the tokens that the previous one wrote, performs $K$ steps of gradient descent from $W_0$. {#c-lsa-stack}
The output depends on the weights only through $P W_V$ and $W_K^\top W_Q$, which is why the rescaling leaves it unchanged. {#c-lsa-rescale}
With $W_0 = 0$ and $\eta = 1$ the prediction is $\frac{1}{N} \sum_i y_i\, x_i^\top x_q$, the averaging estimate that Garg et al. use as a baseline. {#c-gd-averaging}

**Bayes-optimal in-context regression.** Suppose $w \sim \mathcal{N}(0, \tau^2 I_d)$ and $y_i = w^\top x_i + \varepsilon_i$ with independent noise $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$.
Then the prediction for $x_q$ with the least expected squared error, given the examples, is ridge regression fitted to the prompt,

$$ \hat y_q = x_q^\top (X^\top X + \lambda I)^{-1} X^\top y, \qquad \lambda = \frac{\sigma^2}{\tau^2}, $$ {#eq-ridge-icl}

where $X$ holds the inputs as rows. {#c-bayes-ridge}
Without noise the posterior mean of $w$ is the minimum-norm least-squares solution $X^+ y$; it is the projection of $w$ onto the span of the inputs, and the limit of ridge as $\lambda \to 0$. {#c-bayes-noise-free}
With $x_i$, $x_q$ and $w$ all standard normal, its expected squared error divided by $d$ is $(d - k)/d$ after $k \le d$ examples and $0$ from $k = d$ on. {#c-ls-curve}
`derivations/bayes-optimal-in-context-regression.md` proves the three statements and checks them by simulation.

## Examples

**GPT-3.** Brown et al. measure in-context learning on dozens of benchmarks.
Over 42 accuracy benchmarks, zero-shot performance rises steadily with model size, and few-shot performance rises faster. {#c-brown-scale}
On TriviaQA, GPT-3 reaches 64.3% zero-shot, 68.0% one-shot and 71.2% few-shot. {#c-brown-triviaqa}
For most tasks all three settings improve smoothly with capacity, and the gap between them often grows. {#c-brown-gap}
Few-shot performance is weak on natural language inference tasks such as ANLI and on some reading comprehension sets such as RACE and QuAC. {#c-brown-struggles}

**Linear functions in a Transformer trained from scratch.** Garg et al. train a decoder-only Transformer of the GPT-2 family with 12 layers, 8 heads and a 256-dimensional embedding, 9.5M parameters, on squared error with a batch size of 64 for 500k steps. {#c-garg-model}
The prompts hold examples of linear functions in $d = 20$ dimensions, with $x_1, \dots, x_k$, $x_{\text{query}}$ and $w$ drawn independently from $\mathcal{N}(0, I_d)$; the baseline is least squares, which they call the optimal estimator here. {#c-garg-linear-setup}
Another baseline averages the $y_i x_i$ to estimate $w$ and takes the inner product with the query. {#c-garg-averaging}
The trained model performs about as well as least squares at every prompt length: at $k = 20$ least squares has error 0 and the Transformer 0.02, which falls to 0.0006 at $k = 40$. {#c-garg-result}
It is not recalling weight vectors from training: it saw 32 million of them, the best of which would give an error of about 0.2, while it reaches below 0.001. {#c-garg-memorization}
With fewer examples than dimensions, $w$ cannot be recovered, and Garg et al. note that the ideal model should approximate the projection of $w$ onto the span of the inputs. {#c-garg-projection}
This is exactly the noiseless Bayes predictor of the formal statement, and the $(d - k)/d$ curve is the error that the projection leaves.

Other function classes behave similarly.
On 3-sparse linear functions in 20 dimensions the model reaches errors 0.58 and 0.09 at $k = 5$ and $k = 10$, against 0.62 and 0.08 for Lasso. {#c-garg-sparse}
On decision trees of depth 4 with 100 examples it reaches 0.12, against 0.80 for greedy tree learning and 0.62 for XGBoost. {#c-garg-trees}
When training and test prompts come from the same distribution, the error falls as the model gets larger or the dimension smaller. {#c-garg-capacity}
For $d = 50$ the training loss barely moves in 500k steps unless the prompts start small and grow, a curriculum. {#c-garg-curriculum}

**Noise and the choice of $\lambda$.** When Garg et al. add $\mathcal{N}(0, 1)$ noise to every output, the trained model tracks least squares except near $k = 20$ and shows the double descent curve; they note that the optimal estimator is then least squares with suitable $\ell_2$ regularization, which a model trained on clean data cannot be expected to learn. {#c-garg-noisy}
With $w \sim \mathcal{N}(0, I_d)$ and unit noise, eq-ridge-icl gives that regularization exactly: $\lambda = 1$. {#c-garg-lambda}

**A worked case.** One input dimension, the examples $(1, 2)$ and $(2, 4)$, the query $x_q = 3$, and $\eta = 0.5$.
From $W_0 = 0$, eq-gd-step gives $\Delta W = \frac{0.5}{2}(1 \cdot 2 + 2 \cdot 4) = 2.5$, so one attention layer with the construction predicts $7.5$; two layers predict $5.625$.
Least squares fits $W = 2$ and predicts $6$, and ridge with $\lambda = 1$ fits $W = 10/6$ and predicts $5$.
The tests in `impl/numpy` and `impl/pytorch` hold these numbers.

**Trained attention finds the construction.** Von Oswald et al. train on tasks with a teacher $W_\tau \sim \mathcal{N}(0, I)$, inputs from $U(-1, 1)$ and targets $y = W_\tau x$, mostly with $N = n_I = 10$ and $n_O = 1$. {#c-oswald-setup}
A single trained linear self-attention layer performs the same as one step of gradient descent, and a simple rescaling of its trained weights gives back the construction. {#c-oswald-match}
With $K$ layers the trained model usually does better than $K$ plain steps, and is described well by a variant, GD++, that also transforms the inputs by $H(X) = I - \gamma X X^\top$. {#c-oswald-deep}

**What the demonstrations contribute.** Min et al. (2022) test 12 models, six decoder-only language models from 774M to 175B parameters, each with a direct and a channel method. {#c-min-setup}
They use 26 classification and multi-choice datasets and $k = 16$ demonstrations by default. {#c-min-data}
Replacing the gold labels of the demonstrations with random labels costs only 0 to 5% absolute, 1.7% on average for multi-choice and 2.6% for classification, while demonstrations of either kind beat none. {#c-min-random}
Even labels that are always wrong keep 92%, 100% and 97% of the gain in three of their settings. {#c-min-incorrect}
The gold-to-random drop stays between 0.8 and 1.6% across $k$, and accuracy barely rises beyond $k = 8$. {#c-min-k}
Min et al. split a demonstration into four parts: the mapping from input to label, the distribution of the inputs, the set of labels, and the format of input-label pairs. {#c-min-aspects}
Swapping the inputs for sentences from an unrelated corpus costs 3 to 16% absolute. {#c-min-ood}
Without the format, demonstrations are no better than none; with it, random sentences paired with the right label set keep 95% and 82% of the gain. {#c-min-format}

## Related ideas

Fine-tuning adapts a pretrained model by gradient updates on task data; in-context learning uses only forward passes at test time.
Prompting with an instruction alone is the zero-shot case.
Meta-learning names the training that produces the ability, and Garg et al. use the word for training on random prompts.

On the side of the algorithms, least squares, ridge, Lasso and gradient descent are the reference points against which in-context predictions are measured.
Von Oswald et al. put forward the hypothesis, as Garg et al. and Akyürek et al. had, that gradient descent drives in-context learning in Transformers. {#c-oswald-hypothesis}
Kernel regression gives a second reading: one step of gradient descent from zero is a weighted sum of the example outputs with weights $x_i^\top x_q$, a smoother with a linear kernel and no normalization.
The k-nearest neighbors method also predicts from the stored examples without fitting weights, but it uses only the closest few, while the attention step weighs every example.
Bayesian inference is the yardstick for the regression tasks: when the prior over tasks is known, the posterior mean is the best possible in-context predictor, and a trained model can at most match it.

## Common confusions

**"In-context learning trains the model."** It does not; the weights are fixed and the demonstrations are only conditioning.
Whatever adapts lives in the activations, which is why the effect vanishes when the prompt does.

**"The model learns the task from the labels."** Min et al. find that the labels often matter less than the format, the label space and the input distribution.
They argue that in the strict sense of learning the input-label mapping, language models do not learn new tasks at test time, but in a broader sense that includes adapting to the inputs, the labels and the format, they do. {#c-min-learn}
The gap between gold and random labels depends on the model and the dataset, nearly 14% absolute for GPT-J on financial_phrasebank, and their experiments cover only classification and multi-choice. {#c-min-limits}

**"New task" versus "recognized task".** Brown et al. list as a limitation that it is unclear whether few-shot learning learns tasks from scratch at inference time or identifies tasks learned in training. {#c-brown-from-scratch}
They suggest that wordscrambling or defining nonsense words are likely learned de novo, while translation must have been learned in pretraining. {#c-brown-de-novo}

**"Transformers run gradient descent."** The construction proves that a linear attention layer can perform a gradient step, and that a trained one does in a small, clean setting; it does not show that a softmax language model does so on natural text.
Deep trained models already depart from plain gradient descent, as the GD++ result above shows.
Their in-context predictions are also sensitive to shifts that the reference algorithms ignore.
When the prompt inputs all lie in one orthant and the query in another, the trained model still matches least squares, with errors of 0.062 and 0.004 at 20 and 40 examples, against 0.02 and 0.0006 normally. {#c-garg-orthant}
But scaling the prompt inputs by 3 raises the error at 40 examples to 0.58, although least squares is unaffected by scale. {#c-garg-input-scale}

**"Least squares is always the target."** It is Bayes-optimal only without noise; with noise the target is ridge, with $\lambda = \sigma^2/\tau^2$, as eq-ridge-icl states.

## History

Brown et al. (2020) named in-context learning and measured it in GPT-3, the 175B-parameter model whose zero-, one- and few-shot results made prompting a way to use language models without fine-tuning.
Garg et al. (2022) moved the question to function classes, where the right answer is known.
Von Oswald et al. (2023), first posted in December 2022, proved that linear self-attention can implement gradient descent on in-context regression, and found that trained layers learn it.
Min et al. (2022) asked which parts of the demonstrations a language model uses and found that the labels matter less than expected.
Together they set the two lines of work that followed: what in-context learning can do, and how the forward pass of a Transformer does it.
