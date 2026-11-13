# AI

[![CI](https://github.com/tamnd/ai/actions/workflows/ci.yml/badge.svg)](https://github.com/tamnd/ai/actions/workflows/ci.yml)
[![Site](https://img.shields.io/badge/site-ai.tamnd.com-2b59c3)](https://ai.tamnd.com)
[![REUSE](https://img.shields.io/badge/REUSE-3.3-green)](https://reuse.software)

This repository is the content of [ai.tamnd.com](https://ai.tamnd.com), an open encyclopedia of artificial intelligence. It covers each idea, method, architecture, model, task, benchmark, dataset, result, system, risk, law, argument, person and event of AI. An item with a lot to say gets a page. Each other item gets a card with checked facts and sources.

Each claim on a page has a type, a source and a date. The build computes its evidence label: proved, measured here, replicated, reported or contested. Each equation is derived on the page or cited to its source. Each reference implementation has tests in NumPy, PyTorch and JAX.

The toolchain `aik` is in [tamnd/ai-agent](https://github.com/tamnd/ai-agent). The plan of the project is the [specification](https://ai.tamnd.com/spec/).

## What is in this repository

| Directory | Contents |
| --- | --- |
| `nodes/<top>/<sub>/<slug>/` | One page: `node.md`, its claims, facts, derivations, code, experiments and scene. |
| `cards/<kind>/` | The cards of models, benchmarks, datasets, results, works, people and the other card kinds, one card on each line. |
| `topics/areas.yaml` | The area tree of spec 05.2: 9 groups, 52 top areas and their sub-areas. |
| `registers/` | The tags, the reserved slugs and the other registers. Only `aik` writes `tags.reg`. |
| `manifests/` | The pinned runner images and the release manifests. |
| `lean/` | The Lean package `AI` (spec 11). |

A node directory looks like this:

```text
nodes/deep-learning-training/regularization/dropout/
  node.md              front matter and prose
  claims.yaml          the typed claims and their sources (spec 07)
  facts.yaml           the dated facts (spec 04.9)
  derivations/*.md     the derivations of the equations
  impl/numpy/          the reference implementation and its tests
  impl/pytorch/        the PyTorch version, tested against NumPy
  impl/jax/            the JAX version, tested against NumPy
  exp/<name>/          a small experiment with seeds and a budget (spec 09)
  viz/                 the scene and its presets
```

## Work on the content

1. Install the toolchain: `go install github.com/tamnd/ai-agent/cmd/aik@latest`.
2. Make a node: `aik new method dropout -area deep-learning-training/regularization`.
3. Put the files in canonical form: `aik fmt`.
4. Check your work: `aik check -changed origin/main`.
5. Run the tests of the code: `aik test dropout`. Without Docker or Podman, add `-unsafe-local`, and read the code first, because this runs it on your computer.
6. Look at the page: `aik site serve`, then open http://localhost:8150/dropout/.
7. Before the merge, give the node its permanent tag: `aik mint dropout`.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before you open a pull request.

## How the site is built

Each push to `main` runs the CI workflow. It runs the audit, the tests of the implementations and the scene checks. Then it renders the site and deploys it to the Cloudflare Worker `ai-site` at https://ai.tamnd.com. A pull request from this repository gets a preview site. The jobs that run code from the content have no secrets.

## Roadmap

The work has six milestones, M0 to M5, from November 2026 to December 2028. [Spec 21](https://ai.tamnd.com/spec/21-milestones/) gives the scope and the exit criteria of each one. The [milestones](https://github.com/tamnd/ai/milestones) list the open issues. Issues with the label `good first issue` are small and clear.

## Licence

| Part | Licence |
| --- | --- |
| Prose | [CC BY-SA 4.0](LICENSES/CC-BY-SA-4.0.txt) |
| Code, tests, experiments and scenes | [MIT](LICENSES/MIT.txt) |
| Lean proofs | [Apache 2.0](LICENSES/Apache-2.0.txt) |
| Metadata, cards and data | [CC0 1.0](LICENSES/CC0-1.0.txt) |

[REUSE.toml](REUSE.toml) gives the licence of each path.
