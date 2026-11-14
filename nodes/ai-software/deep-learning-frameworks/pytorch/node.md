---
tag: NEW-pytorch
kind: software
slug: pytorch
name: PyTorch
aliases: [torch]
area: ai-software/deep-learning-frameworks
level: core
summary: >-
  An open-source Python library for deep learning that runs models eagerly as Python programs, computes gradients by reverse-mode automatic differentiation, and runs on CPUs and GPUs.
history:
  year: 2017
  note: "Initial release in January 2017 per Paszke et al. (2019), Section 6.4; the Linux Foundation's 2022 announcement says 2016."
same_as:
  wikipedia: PyTorch
  arxiv: ["1912.01703"]
sources:
  - {paper: "Paszke et al. (2019)", title: "PyTorch: An Imperative Style, High-Performance Deep Learning Library", locator: "Sections 1 to 6", note: "the design, the implementation and the evaluation"}
  - {paper: "Ansel et al. (2024)", title: "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation", locator: "the abstract", note: "TorchDynamo and TorchInductor"}
  - {url: "https://pytorch.org/blog/pytorch-2.0-release/", title: "PyTorch 2.0: Our next generation release", locator: "p. 1"}
  - {url: "https://pytorch.org/blog/PyTorchfoundation/", title: "PyTorch strengthens its governance by joining the Linux Foundation", locator: "p. 1"}
  - {url: "https://raw.githubusercontent.com/pytorch/pytorch/v2.15.0/LICENSE", title: "LICENSE of PyTorch v2.15.0", locator: "p. 1"}
language: Python
repo: https://github.com/pytorch/pytorch
licence: BSD-3-Clause
first_release: 2017-01
maintainer: PyTorch Foundation, a project of the Linux Foundation
category: deep learning framework
releases:
  - {version: "2.0", date: 2023-03-15, note: "torch.compile, in Beta, as the main API"}
---

PyTorch is an open-source Python library for deep learning.
A model in PyTorch is an ordinary Python program that runs one operation at a time, and PyTorch computes its gradients by reverse-mode automatic differentiation.
The core is written in C++ and runs on CPUs and GPUs.
Since September 2022 the project has lived under the PyTorch Foundation, part of the Linux Foundation.

## What it is for

PyTorch is for writing, training and running deep learning models.
Paszke et al. describe it as a machine learning library with an imperative and Pythonic programming style that remains efficient and supports hardware accelerators such as GPUs. {#c-purpose}
Defining layers, composing models, loading data, running optimizers and parallelizing training are all written with the concepts of general purpose programming. {#c-workflow}

The Linux Foundation reports that organizations use PyTorch in academic research and in production environments. {#c-users}
The licence file of the repository permits redistribution and use in source and binary forms, with or without modification, if three conditions are met. {#c-licence}
These are the terms of the BSD 3-Clause licence.

## Core concepts

**Tensors and the C++ core.** Most of PyTorch is written in C++. {#c-cpp}
Its core library, libtorch, implements the tensor data structure, the CPU and GPU operators, and the automatic differentiation system. {#c-libtorch}
The Python functions are a layer over this core.

**Eager execution.** Caffe, CNTK, TensorFlow and Theano construct a static dataflow graph of the computation, which is then applied repeatedly to batches of data. {#c-static}
PyTorch gives up the potential benefits of that graph approach to keep the imperative programming model of Python. {#c-imperative}
So each line of a model runs when Python reaches it, and the value it computes is there to inspect.

**Modules.** Layers are usually Python classes whose constructors create and initialize their parameters, and whose forward methods process an input. {#c-modules}
Models are usually classes that compose layers, but nothing forces this structure. {#c-models}

**Automatic differentiation.** PyTorch uses the operator overloading approach: it builds a representation of the computed function each time the function runs. {#c-overloading}
It then performs reverse-mode automatic differentiation, which computes the gradient of a scalar output with respect to a multivariate input. {#c-reverse}
It can also differentiate through code that mutates tensors, and a versioning system for tensors tracks their modifications. {#c-mutation}

**Compilation.** PyTorch 2.0 added torch.compile, which wraps a model and returns a compiled model. {#c-compile}
The release notes call it a fully additive and optional feature, so code written without it still runs. {#c-additive}
Ansel et al. describe the two parts behind it. {#c-dynamo}
TorchDynamo is a Python-level just-in-time compiler that modifies Python bytecode before execution and extracts sequences of PyTorch operations into an FX graph.
TorchInductor, its default backend, translates the graph into OpenAI's Triton for GPUs and into C++ for CPUs.
They report geometric mean speedups of 2.27× for inference and 1.41× for training on an NVIDIA A100 GPU, across more than 180 real-world models. {#c-speedup}

## Example

The program below differentiates a function that contains a Python `if` statement.
The branch is taken at run time, as in any Python program, and autograd records only the operations that ran.

```python
import torch


def f(x):
    # Ordinary Python control flow runs eagerly, one operation at a time.
    if x.sum() > 0:
        return (x**2).sum()
    return (-x).sum()


x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = f(x)
y.backward()
print("f(x) =", y.item())
print("grad by backward() =", x.grad.tolist())

g = torch.func.grad(f)(torch.tensor([1.0, 2.0, 3.0]))
print("grad by torch.func.grad =", g.tolist())
```

The output, from PyTorch 2.15.0 on Python 3.13:

```text
f(x) = 14.0
grad by backward() = [2.0, 4.0, 6.0]
grad by torch.func.grad = [2.0, 4.0, 6.0]
```

The function is the sum of the squares of x, so its gradient is 2x.
`backward()` stores the gradient in `x.grad`.
`torch.func.grad` returns the gradient as a new function instead, in the style of a function transform.
PyTorch 2.0 moved the functorch APIs into this `torch.func` module. {#c-func}
The program is `examples/autograd_example.py`, and `examples/test_autograd_example.py` checks this output.

## Ecosystem and alternatives

TensorFlow and JAX are the other main Python frameworks of this kind.
Ansel et al. group PyTorch and JAX as eager mode frameworks, and TensorFlow, Caffe, Theano and CNTK as graph mode frameworks. {#c-eager-graph}
In their terms, an eager mode framework runs the code of a model each time it runs the model, while a graph mode framework first builds a graph and then executes it. {#c-eager-def}
The table sums up the points sourced below.

| | PyTorch | TensorFlow 2 | JAX |
|---|---|---|---|
| Default execution | eager | eager | eager, in the grouping of Ansel et al. |
| Graph capture | `torch.compile`, optional, from 2.0 | `tf.function` | `jax.jit` |
| Constraint on captured code | none required; TorchDynamo works on Python bytecode | Python code that `tf.function` can turn into a graph | functionally pure, with no data-dependent Python control flow |

**TensorFlow.** In TensorFlow 2, eager execution is turned on by default. {#c-tf-eager}
Its `tf.function` creates Python-independent dataflow graphs out of Python code. {#c-tf-function}
So TensorFlow 2 and PyTorch both run eagerly by default, and both offer an opt-in way to capture a graph.

**JAX.** The JAX documentation describes it as a Python library for accelerator-oriented array computation and program transformation. {#c-jax}
It offers composable function transformations for compilation, batching, automatic differentiation and parallelization. {#c-jax-transforms}
These transformations are designed to work only on Python functions that are functionally pure. {#c-jax-pure}
Frostig et al. describe JAX as a tracing just-in-time compiler that generates accelerator code from pure Python and NumPy programs, using the XLA compiler infrastructure. {#c-jax-tracing}
Ansel et al. write that `jax.jit` does not support data-dependent Python control flow and requires user code to be functionally pure. {#c-jax-jit}
In contrast, they write, PyTorch started as an eager-only framework without compiler-minded constraints in its design. {#c-eager-only}
That is why TorchDynamo has to capture graphs from Python code written with no regard to compilation.
PyTorch's `torch.func` offers function transforms in a style close to JAX, as the example above shows.

## History

Soumith Chintala writes that PyTorch development started in 2016, with people from the Lua Torch community and people and funding from Meta. {#c-start}
The sources differ on the year of the first release.
The PyTorch paper dates the initial release of PyTorch to January 2017. {#c-release-2017}
In a blog post of 2018, the PyTorch team marked one year since PyTorch was released publicly. {#c-year-in}
The Linux Foundation's announcement of 2022 instead speaks of its release in 2016. {#c-release-2016}

Paszke et al. presented the design at NeurIPS 2019. {#c-neurips}
On all their benchmarks, the performance of PyTorch was within 17% of the fastest framework. {#c-17}
They attribute this to the frameworks offloading most of the computation to the same versions of cuDNN and cuBLAS. {#c-cudnn}

On 12 September 2022, PyTorch moved to the Linux Foundation as a top-level project under the name PyTorch Foundation. {#c-lf}
Its governing board has members from AMD, Amazon Web Services, Google Cloud, Meta, Microsoft Azure and NVIDIA. {#c-board}
The technical decisions remain in the control of individual maintainers. {#c-maintainers}

The PyTorch Foundation released PyTorch 2.0 in 2023, with torch.compile in Beta as its main API. {#c-2-0}

## References
