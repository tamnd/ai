# Licences

Each part of this repository has a licence. `REUSE.toml` sets the licence for each path.

| Part | Paths | Licence |
| --- | --- | --- |
| Prose | `nodes/**/node.md`, `nodes/**/derivations/**` | CC BY-SA 4.0 |
| Code, tests, experiments and scenes | `nodes/**/impl/**`, `nodes/**/exp/**`, `nodes/**/viz/**` | MIT |
| Lean proofs | `lean/**` | Apache 2.0 |
| Metadata and data | `cards/**`, `registers/**`, `topics/**`, `manifests/**`, `prov/**`, `reviews/**`, and `claims.yaml`, `facts.yaml`, `shapes.yaml`, `contract.yaml`, `releases.yaml` and `vectors/**` of each node | CC0 1.0 |

The front matter of `node.md` is the YAML block between the first two `---` lines. It is metadata, so it is under CC0 1.0. The rest of `node.md` is under CC BY-SA 4.0. REUSE gives one licence expression for each file, so `node.md` has the expression `CC-BY-SA-4.0 AND CC0-1.0`.

A fact is not protected by copyright. The facts on the cards come from the sources that each card names. The text of a source is never copied (spec 19).
