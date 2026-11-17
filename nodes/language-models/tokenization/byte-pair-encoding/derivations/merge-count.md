---
id: merge-count
derives: [c-merge-token-drop, c-merge-overlap, c-vocab-count]
kind: proof
assumes:
  - Each word of the training table starts as its characters followed by one end-of-word marker, and each word has a positive frequency.
  - A merge of the pair (A, B) scans each word from left to right and replaces every non-overlapping occurrence of A followed by B with the single symbol AB.
  - A merge is applied only to a pair whose count is at least 1, the count being the frequency-weighted number of adjacent positions where A is followed by B.
uses: []
status: full
check: merge-count.check.py
lean: null
sources:
  - paper: sennrich-2016-neural-machine-translation-rare-words
    locator: p. 3
  - paper: sennrich-2016-neural-machine-translation-rare-words
    locator: p. 4
---

## Statement

Let a training table give each word w a frequency f(w) > 0, and let T be the total number of symbols, counting each
word f(w) times.

1. A merge of the pair (A, B) lowers T by exactly the frequency-weighted number r of occurrences it replaces, and
   1 ≤ r ≤ c, where c is the count of the pair.
2. r = c unless the pair overlaps itself inside a word, which can only happen when A = B; for a run of n copies of A
   the count is n − 1 and the merge replaces ⌊n/2⌋.
3. After k merges the symbol vocabulary is the initial symbols plus k merged symbols, so its size is the initial
   size plus k when every merge result is a new string. Symbols that no longer occur in the table stay in the
   vocabulary.

## Assumptions

The three assumptions in the front matter are the setting of Algorithm 1 of Sennrich et al.: words are split into
characters with an end marker, the most frequent pair is merged, and the merge replaces each occurrence of the pair.

## Steps

1. Within one word, every replaced occurrence turns two adjacent symbols into one and leaves the others alone, so
   the word gets shorter by its number of replacements. Summing over words with their frequencies gives the drop r.
2. The pair was chosen with count c ≥ 1, so some word holds A followed by B. The left-to-right scan replaces the
   first such position it reaches, so r ≥ 1. Each replacement uses one counted position, so r ≤ c.
3. Two counted positions of the pair can conflict only if they share a symbol, which means the symbol at position
   i + 1 is both B and A, so A = B. When A ≠ B no positions conflict and every counted position is replaced, giving
   r = c. When A = B, a run of n copies has n − 1 adjacent positions, and the scan replaces positions 1, 3, 5, …,
   which is ⌊n/2⌋ of them.
4. Each merge adds one symbol, AB, and removes none from the vocabulary, so after k merges the vocabulary is the
   initial symbols together with the k merge results. Its size is the initial size plus k when those results are
   distinct strings that are not initial symbols. This holds in every case the check below tried, but the steps
   here do not prove that merge results are always new, so the count is stated under that condition.

## Result and check

The check script trains on the table low 5, lower 2, newest 6, widest 3. The table starts with 95 symbols over 11
distinct initial symbols. The ten merges lower the count by 9, 9, 9, 7, 7, 6, 6, 6, 5 and 3, ending at 28, and each
drop equals the count of the merged pair. The vocabulary then holds 11 + 10 = 21 symbols, of which 9 still occur in
the table; es, for example, is created and then used up. The word aaa with frequency 4 has a count of 8 for the pair
(a, a), but its merge replaces only 4 occurrences. On 2,000 random tables over the letters a and b, every drop
equals the number replaced and lies between 1 and the count, the final token counts agree, and some merges drop
less than their count.
