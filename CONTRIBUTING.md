# Contributing

Thank you for your help. This file tells you how to change the content. To change the toolchain, go to [tamnd/ai-agent](https://github.com/tamnd/ai-agent).

## Rules for all content

- Write everything yourself. Do not copy text, figures, tables or code from a paper, a book, a model card or a web site, even with a free licence. Read your sources, close them, and then write. Put each source that you used in `sources`.
- Write in plain English. Use short sentences, the active voice and simple words. We follow ASD-STE100 where we can.
- Give each claim a type and a source in `claims.yaml`. Do not write an evidence label. The build computes it.
- Give each number that can change a source and an `as_of` date in `facts.yaml`. Do not write "state of the art", "the best model" or "currently" in prose (rule K04).
- On a contested question, give each serious position with its sources. Do not give the opinion of the site (rule N01).
- Derive each equation on the page, or cite the exact equation of its source.
- Write the NumPy implementation first, from first principles. The PyTorch and JAX versions must agree with it. Put a comment `eq: <id>` on each block of code that computes an equation of the page.
- Each code file starts with the SPDX header. Copy it from a node that exists.

The [content standard](https://ai.tamnd.com/spec/06-content-standard/) has the full rules.

## Make a change

1. Find an issue, or open one. Say which node or card you want to write or change.
2. Fork the repository and make a branch from `main`.
3. Make a node with `aik new`, or change a node.
4. Run `aik fmt`, `aik check -changed origin/main` and `aik test <slug>`.
5. Run `aik site serve` and read your page.
6. Open a pull request and fill in the template. Use one pull request for each node.

A new node gets a provisional tag `NEW-<slug>`. A maintainer gives it a real tag before the merge.

## Review

A maintainer reads each pull request. For a derivation or a proof, the reviewer checks each step. For a claim, the reviewer opens the source at the locator and checks that it says what the claim says. For code, the reviewer checks it against the equations of the page. The review record goes into `reviews/`.

## Sign-off

Each commit must have a `Signed-off-by` line (the [Developer Certificate of Origin](https://developercertificate.org)). Use `git commit -s`. With this line, you state that you wrote the change and that you have the right to give it under the licence of its part.

## Licence

Your change is under the licence of the part that it goes into. [LICENSES/README.md](LICENSES/README.md) lists the parts.
