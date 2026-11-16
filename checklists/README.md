# Checklists

These are the review checklists of spec 15.7.
A reviewer uses them to raise a part of a node to its reviewed level: the text to W5, an implementation to X4, an experiment to R4, the facts to D3, a scene to V3, a Lean statement or spec to F1, a page proof to H3.
Samplers (15.6) use the same lists.

## Where they are

Each list is `checklists/<dir>/<kind>/v<n>.md` or, for every kind, `checklists/<dir>/v<n>.md`.
`aik review` takes the newest version for the kind of the node, and falls back to the list of the part when the kind has none.

| Part | Directory |
|---|---|
| `text` | `text/<kind>/`, fallback `text/` |
| `positions` | `positions/` |
| `facts` | `facts/<kind>/` (model, policy), fallback `facts/` |
| `contract` | `contract/` |
| `impl-<framework>` | `impl/<framework>/` |
| `exp/<slug>` | `experiment/` |
| `scene` | `scene/` |
| `lean-spec`, `lean-statement` | `lean-spec/`, `lean-statement/` |
| `proof-<id>` | `proof/` |

A question is a line that starts with its number, `1. `, and its id is that number.
Each question is a yes or no question that names the place to look.
Yes means pass.
`text/result/` and `text/exercise/` are for cards, which have no page; samplers use them on the card.

## How to review

You must be listed in `registers/reviewers.yaml` (spec 14.11), with the skill for the part and the key that signs your commits:

```yaml
- {id: "github:alice", name: Alice Example, skills: [domain:deep-learning-training, code:pytorch, lean, exp, viz, data], key: "ssh-ed25519 AAAA...", org: "University of Example"}
```

A skill is `domain:<top area>`, `code:<framework>`, `lean`, `exp`, `viz`, `data` or `law:<jurisdiction>`.
The table of 15.5 says which skill each part needs.
You must not be an author of any file you review.

Then:

1. Write the record: `aik review <slug> <part> -reviewer github:<login> -w`. It writes `reviews/<tag>/<part>.yaml` with the hashes of the files, the checklist version, and one item for each question, with the question as a comment above it.
2. Read the files of the part, and answer each item `pass`, `fail` or `n/a`. Give a `note` for each fail and each n/a, and for checks you made against a source (which source, which page).
3. State `interests`: "none", or your employer or funder if it made what the page is about. If you have published for one position of a contested page, say so; a `positions` review then needs a second reviewer.
4. When every item is pass or n/a, set `verdict: pass`. Otherwise leave `verdict: changes` and open an issue or a pull request with the fixes.
5. Commit the record in a signed commit (`git commit -S`), with the key of `registers/reviewers.yaml`, and open a pull request.
6. Check it: `aik review <slug> <part> -check` tells whether the record counts, and why not.

A change of any reviewed file makes the record stale, and the part loses its level until a new review.

## Changing a checklist

Never edit a version that a record names.
Write the next version, `v2.md`, next to it; new records take it, and old records keep theirs.
