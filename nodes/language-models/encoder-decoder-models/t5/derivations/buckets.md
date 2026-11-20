---
id: buckets
derives: [c-bucket-edges, c-bucket-unused]
kind: closed-form
assumes: ["the bucket function of this page (eq-bias): of the B buckets of a direction, the first B/2 hold the offsets 0 to B/2 - 1 one each, and offset n >= B/2 goes to B/2 + floor((B/2) log(2n/B) / log(2 d / B)), capped at B - 1", "B = K for the decoder, which looks back only, and B = K/2 for each direction of the encoder, the keys after the query placed K/2 higher"]
uses: []
status: full
check: buckets.check.py
lean: null
sources:
  - {work: raffel-2020-t5, locator: "p. 5", note: "Section 2.1, 32 embeddings with ranges that grow logarithmically up to an offset of 128"}
---

### Statement

1. Bucket $b \ge B/2$ begins at the smallest offset $n$ with $n \ge (B/2)\,(2d/B)^{(b - B/2)/(B/2)}$, so the bucket edges grow by the factor $(2d/B)^{2/B}$ from one bucket to the next.
2. For $K = 32$ and $d = 128$, the decoder has the offsets 0 to 15 one to a bucket, then 16 buckets whose edges grow by $8^{1/16} = 1.139$: they begin at 16, 19, 21, 24, 27, 31, 35, 40, 46, 52, 59, 67, 77, 87, 99 and 113. Every offset from 113 on shares the last bucket.
3. Each direction of the encoder has 16 buckets: the offsets 0 to 7 one to a bucket, then 8 buckets whose edges grow by $16^{1/8} = \sqrt{2}$, beginning at 8, 12, 16, 23, 32, 46, 64 and 91. Every offset from 91 on in a direction shares a bucket.
4. Bucket $K/2$ of the encoder, which would hold the offset 0 of the keys after the query, never receives an offset, so 31 of the encoder's 32 rows are used.

### Assumptions

1. The function stated above. The paper gives only its outline: 32 embeddings, ranges growing logarithmically, and one embedding for all offsets beyond 128. Used in every step.

### Steps

1. Write $e = B/2$. For $n \ge e$ the bucket is $e + \lfloor e \log(n/e) / \log(d/e) \rfloor$, and it reaches $b$ when $e \log(n/e)/\log(d/e) \ge b - e$, that is, when $n \ge e\,(d/e)^{(b - e)/e}$. Going from $b$ to $b + 1$ multiplies the bound by $(d/e)^{1/e}$, which is $(2d/B)^{2/B}$. This is statement 1.
2. In the decoder, $B = 32$, $e = 16$ and $d/e = 8$. Bucket $b$ begins at the first integer above $16 \cdot 8^{(b - 16)/16}$: $16$ for $b = 16$, $18.2$ rounded up to 19 for $b = 17$, and so on to $16 \cdot 8^{15/16} = 112.4$, rounded up to 113, for $b = 31$. At $n = d = 128$ the formula gives $16 + 16 = 32$, one past the last bucket, so the cap takes over before 128. This is statement 2.
3. In each direction of the encoder, $B = 16$, $e = 8$ and $d/e = 16$. The edges are the first integers above $8 \cdot 16^{(b - 8)/8} = 8 \cdot 2^{(b - 8)/2}$: 8, 11.3, 16, 22.6, 32, 45.3, 64 and 90.5, which give 8, 12, 16, 23, 32, 46, 64 and 91. This is statement 3.
4. A key after the query has offset $n \ge 1$ and goes to $16$ plus a bucket of at least 1. Offset 0, the key at the query, belongs to the other direction and goes to bucket 0. No offset reaches bucket 16. This is statement 4.

### Result and check

$$ \text{bucket } b \text{ begins at } \Big\lceil \tfrac{B}{2} \big(\tfrac{2d}{B}\big)^{(b - B/2)/(B/2)} \Big\rceil : \quad 113 \text{ for the decoder's last}, \quad 91 \text{ for the encoder's} . $$

The check `buckets.check.py` runs bucket() of the NumPy reference over all offsets up to 2000 for several sizes and compares the first offset of every bucket with the closed form.
