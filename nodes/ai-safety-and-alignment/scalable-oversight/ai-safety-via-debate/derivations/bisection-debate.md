---
id: bisection-debate
derives: [c-bisect-count, c-bisect-rounds, c-bisect-sim, c-log-score-proper]
kind: proof
assumes: ["the judge checks that the two half claims add up to the claim being defended, by one subtraction per round", "the range or the number of simulation steps is a power of two", "the simulation step is deterministic and the judge can check one step", "the judge's belief that an answer is correct is q, and the judge reports p"]
uses: [zero-sum-game@al, minimax@al, game-tree@al]
status: full
check: bisection-debate.check.py
lean: null
sources:
  - {work: irving-2018-ai-safety-via-debate, locator: "p. 6"}
  - {work: michael-2023-debate-helps-supervise, locator: "p. 5"}
---

### Statement

Let $C(a, b)$ be the number of primes $p$ with $a \le p < b$.
In the counting debate on $[a, b)$ with $b - a = 2^n$, Alice claims a count $c$.
While the range has more than one number, Alice states a count $c_L$ for the left half $[a, m)$, $m = (a + b)/2$; the right half $[m, b)$ gets $c - c_L$, by one subtraction of the judge; Bob picks one half, and that half with its count becomes the range.
When the range is the single number $a$, Alice wins if and only if her count is 1 for a prime and 0 otherwise.
Then

1. Alice has a strategy that wins against every strategy of Bob if and only if $c = C(a, b)$; for $c \ne C(a, b)$, the Bob who always picks a half whose count is wrong wins against every strategy of Alice;
2. every play of the debate lasts exactly $n = \log_2 (b - a)$ rounds, and the judge does $n$ subtractions and one primality test;
3. for a deterministic step $f$ and the agreed state $s$ at time $0$, the same bisection over $2^k$ steps (Alice states the state at the midpoint, Bob picks a half, and the judge checks one step at the end) is won by Alice against every Bob if and only if her claimed final state is $f^{2^k}(s)$, and it lasts $k$ rounds;
4. for a judge who believes with probability $q \in (0, 1)$ that the first answer is correct, the expected value of the logarithmic score, $q \log_2 p + (1 - q) \log_2 (1 - p)$, has the unique maximizer $p = q$ over $p \in (0, 1)$; a turn penalty that does not depend on $p$ does not move it.

### Assumptions

1. The judge rejects a split whose two counts do not add up to the count being defended; writing the right count as $c - c_L$ enforces this. Used in items 1 and 2.
2. The range has $2^n$ numbers and the simulation $2^k$ steps, so every halving is exact. Used in items 1 to 3; Irving et al. state the protocol by halving without this detail.
3. The step $f$ is deterministic and the judge can compute one step. Used in item 3; Irving et al. note that the protocol works only for deterministic simulation.
4. The judge maximizes the expected score under a belief $q$. Used in item 4.

### Steps

1. Item 1, by induction on $n$, with the hypothesis that item 1 holds for all ranges of size $2^{n-1}$. For $n = 0$ the range is one number and the judge's primality test decides, so Alice wins exactly when $c = C(a, a + 1)$, and Bob has no move.
2. Step for $n \ge 1$, the case $c = C(a, b)$: Alice states $c_L = C(a, m)$. Then $c - c_L = C(a, b) - C(a, m) = C(m, b)$, because the primes of $[a, b)$ are those of $[a, m)$ and those of $[m, b)$. Both halves carry true counts, so by the hypothesis Alice wins whichever half Bob picks.
3. Step, the case $c \ne C(a, b)$: for any $c_L$, if $c_L = C(a, m)$ then $c - c_L \ne C(a, b) - C(a, m) = C(m, b)$, so at least one half carries a wrong count. Bob picks it, and by the hypothesis Alice loses that sub-debate against Bob's rule. This proves item 1.
4. Item 2: each round halves the range, from $2^n$ to $2^{n-1}$ and so on, and the debate ends at size $2^0 = 1$, after $n$ rounds. Each round costs the judge one subtraction (assumption 1), and the end costs one primality test.
5. Item 3, by the same induction on $k$. For $k = 0$ the judge checks $f(s) = y$ directly. If $y = f^{2^k}(s)$, Alice states $z = f^{2^{k-1}}(s)$; then $y = f^{2^{k-1}}(z)$, both halves are true, and Alice wins by the hypothesis. If $y \ne f^{2^k}(s)$, then for any $z$ either $z \ne f^{2^{k-1}}(s)$, a wrong left half, or $z = f^{2^{k-1}}(s)$ and so $y \ne f^{2^{k-1}}(z)$, a wrong right half; Bob picks the wrong half and Alice loses by the hypothesis. The rounds count as in step 4.
6. Item 4: $g(p) = q \log_2 p + (1 - q) \log_2 (1 - p)$ has derivative $\big(q/p - (1 - q)/(1 - p)\big)/\ln 2$, which is zero only at $p = q$, and second derivative $-\big(q/p^2 + (1 - q)/(1 - p)^2\big)/\ln 2 < 0$, so $g$ is strictly concave and $p = q$ is its unique maximizer. A penalty $-0.05\,t$ adds a constant in $p$ and leaves the maximizer.

### Result and check

$$ \text{Alice wins} \iff c = C(a, b), \qquad \text{rounds} = \log_2 (b - a), \qquad \arg\max_p \; q \log_2 p + (1 - q) \log_2 (1 - p) = q . $$

The check `bisection-debate.check.py` solves both debates exactly by minimax, with every claim in a range around the true value: the counting debate for every dyadic range up to size 64, and the simulation for $x \mapsto x^2 + 1$ modulo 5, 7 and 11 with up to 8 steps.
It plays the counting debate up to $x = 1024$ with an honest Alice and with random lies, counts the rounds and subtractions, finds the maximizer of the expected score on a grid, and verifies the numbers of the worked example.
