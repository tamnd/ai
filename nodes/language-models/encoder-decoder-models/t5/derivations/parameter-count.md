---
id: parameter-count
derives: [c-params, c-params-sizes, c-params-large, c-cross-share]
kind: count
assumes: ["a vocabulary of V tokens with one embedding for the source, the decoder inputs and the output layer", "L encoder layers and L decoder layers of width M, with H heads of d_kv features, I = H d_kv, and a feed-forward width F", "no bias in any projection, and a normalization with a gain of M numbers and no bias", "a table of K by H position biases for each stack, and a final normalization for each stack"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-count.check.py
lean: null
sources:
  - {work: raffel-2020-t5, locator: "pp. 4 and 5", note: "Section 2.1, the layers, the normalization without bias, the shared output layer and the 32 position buckets"}
  - {work: raffel-2020-t5, locator: "p. 11", note: "Section 3.1.1, the baseline and its 220 million parameters"}
  - {work: raffel-2020-t5, locator: "pp. 18 and 19", note: "Section 3.2.2, the attention to the encoder put at about 10% of the parameters"}
  - {work: raffel-2020-t5, locator: "pp. 36 and 37", note: "Section 3.7, the five sizes"}
---

### Statement

1. With $I = H d_{kv}$, the network has $N = VM + 2KH + L\,(12MI + 4MF + 5M) + 2M$ parameters.
2. With $V = 32{,}000$ and $K = 32$, the five sizes of Section 3.7 have 60,441,088 (Small), 222,805,248 (Base), 737,537,024 (Large), 2,851,467,264 (3B) and 11,307,190,272 (11B) parameters, against the "about 60 million", "roughly 220 million", "around 770 million", "around 2.8 billion" and "about 11 billion" of the paper.
3. Small, Base and 11B round to their descriptions; 3B, at 2.85 billion, rounds to 2.9 rather than 2.8, and Large falls short of its description by 32.5 million. A separate output matrix, $VM$ more, would bring Large to 770.3 million but Base to 247.4 million; no single way of counting matches both.
4. The attention to the encoder holds $4LMI$ parameters, 12.7% of Base and from 13.6% to 14.2% of Large, 3B and 11B, where the paper puts it at about 10%.

### Assumptions

1. The parts listed above and no others. The paper gives the normalization as a rescaling with no additive bias and says nothing of a centring; the count does not depend on it. Used in steps 1 to 4.
2. One position table for the encoder and one for the decoder. The paper shares the biases across all layers; a single table for both stacks would remove $KH$, at most 4,096 for 11B, and change no rounded figure. Used in step 3.
3. The 32,000 wordpieces of Section 3.1.3 as the vocabulary, and $d_{kv} = 64$ for Small, which the paper gives for Base and Large but not for Small. Used in step 5.

### Steps

1. The embedding is a $V \times M$ table, used three times and counted once.
2. An encoder layer has four $M \times I$ matrices for the queries, keys, values and output, $4MI$, two feed-forward matrices, $2MF$, and two gains, $2M$. A decoder layer has the same for its self-attention and feed-forward network, another $4MI$ for the attention to the encoder, and three gains: $8MI + 2MF + 3M$. One encoder and one decoder layer together hold $12MI + 4MF + 5M$.
3. The two position tables add $2KH$ and the two final normalizations $2M$. The sum of steps 1 to 3 is statement 1.
4. Base: $M = 768$, $I = 12 \cdot 64 = 768$, $F = 3072$, $L = 12$. A pair of layers has $12 \cdot 768^2 + 4 \cdot 768 \cdot 3072 + 5 \cdot 768 = 16{,}518{,}912$, and twelve pairs $198{,}226{,}944$. The embedding has $32{,}000 \cdot 768 = 24{,}576{,}000$, the tables $2 \cdot 32 \cdot 12 = 768$ and the final gains $1536$, for $222{,}805{,}248$.
5. The same arithmetic gives the other four sizes of statement 2. Small, $60.4$ million, Base, $222.8$ million, and 11B, $11.3$ billion, round to the figures of the paper. 3B has $2{,}851$ million, which rounds to 2.9 billion and is 2.8 billion only if cut off. Large has $737.5$ million, which is 32.5 million short of "around 770 million".
6. A separate $M \times V$ output matrix adds $32{,}000 \cdot 1024 = 32{,}768{,}000$ to Large, giving $770{,}305{,}024$, and $24{,}576{,}000$ to Base, giving $247{,}381{,}248$, which is no longer roughly 220 million. Statement 3 follows.
7. The attention to the encoder has $4MI$ per decoder layer. For Base that is $12 \cdot 4 \cdot 768^2 = 28{,}311{,}552$, or 12.71% of $222{,}805{,}248$; for 11B, $24 \cdot 4 \cdot 1024 \cdot 16{,}384 = 1{,}610{,}612{,}736$, or 14.24%. This is statement 4.

### Result and check

$$ N(\text{Base}) = 32{,}000 \cdot 768 + 2 \cdot 32 \cdot 12 + 12 \cdot 16{,}518{,}912 + 2 \cdot 768 = 222{,}805{,}248 . $$

The check `parameter-count.check.py` compares the closed form with the parameters that the NumPy reference creates for random configurations with and without the position tables, recomputes the five sizes, the two ways of counting the output layer and the share of the attention to the encoder.
