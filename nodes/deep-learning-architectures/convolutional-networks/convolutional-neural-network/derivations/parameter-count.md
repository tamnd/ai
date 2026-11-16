---
id: parameter-count
derives: [c-conv-params, c-dense-comparison, c-lenet-params, c-vgg-params]
kind: count
assumes: ["a convolution has one k by k kernel for each pair of an input and an output channel and one bias for each output channel, shared by all positions", "a fully connected layer has one weight for each pair of an input and an output unit and one bias for each output unit"]
uses: []
status: full
check: parameter-count.check.py
lean: null
sources:
  - {work: lecun-1998-gradient-based-learning, locator: "p. 6"}
  - {work: lecun-1998-gradient-based-learning, locator: "p. 7"}
  - {work: simonyan-2015-very-deep-convolutional-networks, locator: "Section 2.3"}
---

### Statement

1. A $k \times k$ convolution from $C_\text{in}$ to $C_\text{out}$ channels has $k^2 C_\text{in} C_\text{out} + C_\text{out}$ parameters, whatever the size of the input.
2. On an $n \times n$ input with an $n' \times n'$ output, it makes $n'^2 C_\text{out} k^2 C_\text{in}$ multiply-adds, while a fully connected layer between the same input and output has $n^2 C_\text{in} \cdot n'^2 C_\text{out} + n'^2 C_\text{out}$ parameters. For the first layer of LeNet-5, $k = 5$, $n = 32$, $n' = 28$, $C_\text{in} = 1$ and $C_\text{out} = 6$, these are 156 parameters and 122,304 connections, against 4,821,600 parameters for a fully connected layer.
3. With the same rule, the layers of LeNet-5 have 156 (C1), 1,516 (C3, with the partial connection of Table I of LeCun et al.), 48,120 (C5) and 10,164 (F6) parameters. A C3 that connects every map of S2 would have 2,416.
4. A stack of $m$ convolutions of size $3 \times 3$ with $C$ channels in and out, without biases, has $9 m C^2$ weights; one convolution with the same receptive field, $(2m + 1) \times (2m + 1)$, has $(2m + 1)^2 C^2$. For $m = 2$ that is $18 C^2$ against $25 C^2$; for $m = 3$, $27 C^2$ against $49 C^2$, which is 81% more.

### Assumptions

1. Weight sharing: every output position of channel $d$ uses the same kernel $w[d, \cdot, \cdot, \cdot]$ and the same bias $b_d$ (eq-conv). Used in steps 1 and 2.
2. Local connection: output $(d, i, j)$ reads only the $k^2 C_\text{in}$ inputs of its window. Used in step 2.
3. The receptive field of $m$ stacked $3 \times 3$ layers with stride 1 is $2m + 1$ (derivation receptive-field). Used in step 5.

### Steps

1. The kernel of output channel $d$ has one weight for each input channel and each of the $k^2$ offsets, $k^2 C_\text{in}$ in all, and there is one bias per output channel. By assumption 1 these do not depend on the position, so the layer has $C_\text{out}(k^2 C_\text{in} + 1) = k^2 C_\text{in} C_\text{out} + C_\text{out}$ parameters, with no $n$ in the count. This is statement 1.
2. By assumption 2, each of the $n'^2 C_\text{out}$ outputs makes $k^2 C_\text{in}$ multiply-adds. A fully connected layer joins each of the $n^2 C_\text{in}$ inputs to each of the $n'^2 C_\text{out}$ outputs and gives each output its own bias.
3. For C1 of LeNet-5: $25 \cdot 1 \cdot 6 + 6 = 156$ parameters. Counting the bias as one connection per output, as LeCun et al. do, $28^2 \cdot 6 \cdot (25 + 1) = 4{,}704 \cdot 26 = 122{,}304$ connections. A fully connected layer from the $1{,}024$ inputs to the $4{,}704$ outputs has $1{,}024 \cdot 4{,}704 + 4{,}704 = 4{,}821{,}600$ parameters. This is statement 2.
4. In C3, Table I of LeCun et al. connects the first six maps to 3 maps of S2 each, the next six to 4 each, the next three to 4 each and the last one to all 6: $18 + 24 + 12 + 6 = 60$ kernels of 25 weights, plus 16 biases, $1{,}500 + 16 = 1{,}516$. With all $6 \cdot 16 = 96$ kernels it is $2{,}400 + 16 = 2{,}416$. C5 is a $5 \times 5$ convolution from 16 maps to 120: $25 \cdot 16 \cdot 120 + 120 = 48{,}120$. F6 is fully connected from 120 to 84 units: $120 \cdot 84 + 84 = 10{,}164$. This is statement 3.
5. By step 1 without the bias, each $3 \times 3$ layer has $9 C^2$ weights, so $m$ of them have $9 m C^2$. By assumption 3 the single layer with the same receptive field is $(2m + 1) \times (2m + 1)$, with $(2m + 1)^2 C^2$ weights. For $m = 2$: $18 C^2$ and $25 C^2$. For $m = 3$: $27 C^2$ and $49 C^2$, and $49/27 = 1.81$. This is statement 4.

### Result and check

$$ P_\text{conv} = k^2 C_\text{in} C_\text{out} + C_\text{out} $$

The check `parameter-count.check.py` builds the layers with the NumPy reference, counts the entries of their arrays, and evaluates the numbers of statements 2 to 4, with the connection table of LeNet-5 C3 written out.
