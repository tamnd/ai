---
id: corruption-fractions
derives: [c-fractions]
kind: count
assumes:
  - Each ordinary position is chosen with probability r, and each chosen position independently becomes [MASK] with probability 0.8, a token drawn uniformly from the V′ ordinary tokens with probability 0.1, and stays itself with probability 0.1.
  - A sequence has n ordinary tokens and the loss is taken at the chosen positions only.
uses: []
status: full
check: corruption-fractions.check.py
lean: null
sources:
  - paper: devlin-2019-bert
    locator: p. 4
  - paper: clark-2020-electra
    locator: p. 3
---

## Statement

Under BERT's 80/10/10 rule with a masking rate r, a fraction 0.8r of the input tokens are [MASK], a fraction
0.1r(1 − 1/V′) are other words, and a fraction 0.1r(1 + 1/V′) are chosen but show their own word. At r = 0.15 that is
12% [MASK], just under 1.5% wrong words and just over 1.5% chosen but unchanged tokens; just under 13.5% of the input
differs from the text. Of the predicted positions, a fraction 0.1(1 + 1/V′) see the right answer in the input. A
sequence of n tokens gives 0.15n terms of the masked LM loss and n terms of ELECTRA's discriminator loss, 6.67 times
as many; at n = 512 that is 76.8 against 512.

## Assumptions

The random token can equal the original, with probability 1/V′. Special tokens are never chosen.

## Steps

1. A position is [MASK] if it is chosen and its draw falls in the first 80%: probability 0.8r.
2. It is a different word if it is chosen, its draw falls in the random 10%, and the random word differs from the
   original: 0.1r(1 − 1/V′).
3. It is chosen and shows its own word if its draw falls in the last 10%, or in the random 10% with the random word
   equal to the original: 0.1r + 0.1r/V′. The three add up to r.
4. Positions that differ from the text are the first two kinds: 0.8r + 0.1r(1 − 1/V′) = 0.9r − 0.1r/V′. At
   r = 0.15 this is 0.135 − 0.015/V′.
5. Dividing step 3 by r gives the share of predicted positions whose input is the right word.
6. The masked LM loss has one term per chosen position, rn in expectation; the discriminator has one per position,
   n. The ratio is 1/r = 6.67 at r = 0.15.

## Result and check

The check computes the fractions exactly for several V′, compares them with the empirical fractions of the NumPy
reference's bert_corrupt over 400,000 tokens, and reproduces the counts above.
