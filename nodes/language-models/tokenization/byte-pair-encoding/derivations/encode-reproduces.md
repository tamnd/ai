---
id: encode-reproduces
derives: [c-encode-deterministic, c-encode-reproduces, c-priority-equals-sequential, c-priority-counterexample]
kind: proof
assumes:
  - Merges are learned by Algorithm 1 of Sennrich et al. from a table of words, each word starting as its characters followed by one end-of-word marker.
  - Encoding a word starts from the same initial symbols and applies the merges in the order they were learned, each merge replacing non-overlapping occurrences from left to right.
  - Priority encoding repeatedly merges the adjacent pair whose earliest position in the merge list is lowest, taking the leftmost occurrence on ties, until no adjacent pair is in the list.
uses: [merge-count]
status: full
check: encode-reproduces.check.py
lean: null
sources:
  - paper: sennrich-2016-neural-machine-translation-rare-words
    locator: p. 4
  - paper: provilkov-2020-bpe-dropout
    locator: p. 2
  - paper: kudo-2018-subword-regularization
    locator: p. 3
---

## Statement

1. Encoding by replaying the merge list is deterministic: a word and a merge list give exactly one segmentation.
2. For a word of the training table, replaying the learned merges gives the segmentation that the word has in the
   table when training ends.
3. Priority encoding gives the same segmentation as sequential encoding when the merge list satisfies two
   conditions: (i) each side of every merge is a single character, the end marker, or the result of an earlier
   merge, which learned lists always satisfy; and (ii) every merge result is a new string, neither an initial
   symbol nor the result of an earlier merge.
4. Without these conditions the two can differ.

## Assumptions

The assumptions in the front matter describe the training of Sennrich et al., the encoding they describe for test
time, and the priority encoding that Provilkov et al. use in BPE-dropout with no merges dropped.

## Steps

1. Each merge is a function from symbol sequences to symbol sequences: the left-to-right scan makes no choices.
   Encoding is a composition of these functions, so it is a function of the word and the list.
2. Training holds a symbol sequence for each word and applies merge j to every sequence of the table at step j.
   Encoding a training word starts from the same sequence and applies the same functions in the same order, so after
   the last merge it holds the same sequence. The frequencies play no part once the list is fixed.
3. Take a list satisfying both conditions. Give each symbol a birth rank: −1 for an initial symbol and j for the
   result of merge j, which is well defined because by (ii) no string is born twice. By (i), both sides of merge j
   have birth rank below j, and by (ii) the result of merge j differs from every symbol born earlier.
   a. Once merge i has been applied, no adjacent (A_i, B_i) remains: if A_i ≠ B_i every occurrence is joined, and
      if A_i = B_i a run of n copies becomes pairs of joined copies with at most one copy left over. Later merges
      only create symbols born after i, and these are neither A_i nor B_i, so the pair never comes back.
   b. Suppose both procedures hold the same sequence, sequential encoding is about to apply merge j, and no adjacent
      pair of rank below j is present. Priority encoding then picks the leftmost occurrence of (A_j, B_j) if one
      exists, which is also the first join of the left-to-right scan. The new symbol A_jB_j is born at j, so by (i)
      it is not a side of any merge before j and cannot form a pair of lower rank; it is longer than A_j and B_j, so
      it cannot form a new occurrence of (A_j, B_j) either. The occurrences left are the old ones to the right that
      did not overlap the join, so the next leftmost occurrence is the next join of the scan.
   c. Repeating step b, the two procedures make the same joins until no occurrence of (A_j, B_j) is left, and step a
      keeps the lower ranks absent. By induction on j they end in the same sequence.
4. Counterexamples. With the list (b, c), (a, b), (ab, c), (abc, d), (a, bc), sequential encoding of abcd joins b
   and c, finds no a b, ab c or abc d, and last joins a and bc, ending with abc, d, </w>. Priority encoding joins b
   and c, then a and bc (rank 4, the only listed pair), then abc and d (rank 3), ending with abcd, </w>. Here abc is
   the result of two merges, which breaks condition (ii). The list (ab, c) then (a, b) breaks condition (i), because
   ab is used before it is made: sequential encoding of abc gives ab, c, </w>, while priority encoding gives
   abc, </w>.

## Result and check

The check script trains on 3,000 random tables over the letters a to d, with the table order shuffled. Replaying the
learned merges gives the training segmentation of every word, and priority encoding agrees with it. For each list
whose merge results are distinct, it encodes 5 random unseen words both ways, 15,000 words in all, and the two agree
every time. Both counterexamples above are reproduced.
