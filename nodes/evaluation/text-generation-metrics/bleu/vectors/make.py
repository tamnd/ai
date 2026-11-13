# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/bleu/ (ai://V3JG9E).
"""Write official.jsonl and edge.jsonl of the BLEU page with sacreBLEU, the official implementation (spec 08.7.5).

Run it in an environment that has sacreBLEU and nothing of impl/: the official package never enters the reference
sets. The output is deterministic: the same sacreBLEU version gives the same bytes.

    python make.py            # writes official.jsonl and edge.jsonl next to this file
"""

import json
import pathlib

import sacrebleu
from sacrebleu.metrics import BLEU

HERE = pathlib.Path(__file__).resolve().parent
DEFAULTS = {
    "tokenize": "13a",
    "lowercase": False,
    "smooth_method": "exp",
    "smooth_value": None,
    "max_ngram_order": 4,
    "effective_order": False,
}

# Each row: id, predictions, references (segment-major: references[i] holds the references of segment i), the
# options that differ from the defaults, and an optional note.
OFFICIAL = [
    (
        "worked-example",
        ["the cat sat on the mat"],
        [["the cat sat on the red mat"]],
        {},
        "the worked example of the page: one segment, no punctuation, a hypothesis one token short",
    ),
    (
        "papineni-clipping",
        ["the the the the the the the"],
        [["The cat is on the mat.", "There is a cat on the mat."]],
        {"lowercase": True},
        "the clipping example of Papineni et al. (2002, Section 2.1): modified unigram precision 2/7",
    ),
    (
        "corpus-3seg-2ref",
        [
            "The committee approved the new budget on Tuesday.",
            "Heavy rain is expected in the north tomorrow.",
            "She said that the results were better than expected.",
        ],
        [
            [
                "The committee approved the new budget on Tuesday.",
                "On Tuesday the committee passed the new budget.",
            ],
            [
                "Heavy rain is forecast for the north tomorrow.",
                "Tomorrow the north should expect heavy rain.",
            ],
            [
                "She said the results were better than expected.",
                "According to her, the results exceeded expectations.",
            ],
        ],
        {},
        "a corpus of three segments with two references each; the counts are summed over segments before the mean",
    ),
    (
        "corpus-4seg-1ref",
        [
            "It is a guide to action which ensures that the military always obeys the commands of the party.",
            "The new model reduces the error rate by 10 percent.",
            "Prices rose 3.5% in October, the fastest pace since 2008.",
            "We will meet again next week.",
        ],
        [
            ["It is a guide to action that ensures that the military will forever heed Party commands."],
            ["The new model cuts the error rate by ten percent."],
            ["Prices rose 3.5% in October, the fastest increase since 2008."],
            ["We will meet again next week."],
        ],
        {},
        "four segments with one reference each, with numbers, a percent sign and a decimal point for 13a",
    ),
    (
        "sentence-effective-order",
        ["the cat sat"],
        [["the cat sat on the mat"]],
        {"effective_order": True},
        "sentence BLEU with effective order: the hypothesis has no 4-gram, so the mean runs over orders 1 to 3",
    ),
]

EDGE = [
    ("empty-hypothesis", [""], [["the cat sat on the mat"]], {}, "an empty hypothesis: no n-gram, BLEU 0, BP 0"),
    (
        "empty-and-nonempty",
        ["", "the cat sat on the mat"],
        [["a dog ran"], ["the cat sat on the mat"]],
        {},
        "one empty hypothesis in a corpus: the other segment still counts",
    ),
    ("no-match", ["a b c d"], [["e f g h"]], {}, "no n-gram matches: BLEU 0 for every smoothing"),
    (
        "no-4gram-none",
        ["the cat is on the mat"],
        [["the cat sat on the mat"]],
        {"smooth_method": "none"},
        "no 4-gram match and no smoothing: the 4-gram precision is 0 and BLEU is 0",
    ),
    (
        "no-4gram-exp",
        ["the cat is on the mat"],
        [["the cat sat on the mat"]],
        {},
        "the same with the default exp smoothing: the 4-gram precision becomes 100 / (2 * 3)",
    ),
    (
        "no-4gram-floor",
        ["the cat is on the mat"],
        [["the cat sat on the mat"]],
        {"smooth_method": "floor"},
        "floor smoothing with its default 0.1",
    ),
    (
        "no-4gram-floor-0.5",
        ["the cat is on the mat"],
        [["the cat sat on the mat"]],
        {"smooth_method": "floor", "smooth_value": 0.5},
        "floor smoothing with 0.5",
    ),
    (
        "no-4gram-add-k",
        ["the cat is on the mat"],
        [["the cat sat on the mat"]],
        {"smooth_method": "add-k"},
        "add-k smoothing with its default k = 1, applied to the orders 2 and up",
    ),
    (
        "two-zero-orders-exp",
        ["the cat ran on a mat"],
        [["the cat sat on the mat"]],
        {},
        "the 3-gram and 4-gram precisions are both 0: exp smoothing halves twice",
    ),
    ("one-word", ["yes"], [["yes"]], {}, "a one-word hypothesis and reference: only unigrams exist"),
    (
        "one-word-effective-order",
        ["yes"],
        [["yes"]],
        {"effective_order": True},
        "the same with effective order: the mean runs over order 1 only, BLEU 100",
    ),
    (
        "longer-hypothesis",
        ["the cat sat on the red mat today"],
        [["the cat sat on the red mat"]],
        {},
        "a hypothesis longer than its reference: BP is 1",
    ),
    (
        "closest-ref-length",
        ["one two three four five"],
        [["one two three", "one two three four five six seven"]],
        {},
        "two references of 3 and 7 tokens for a hypothesis of 5: a tie, and the shorter length wins",
    ),
    (
        "closest-ref-length-nearer",
        ["one two three four five six"],
        [["one two three", "one two three four five six seven"]],
        {},
        "a hypothesis of 6 tokens is nearer to 7 than to 3",
    ),
    (
        "lowercase",
        ["The Cat Sat On The Mat"],
        [["the cat sat on the mat"]],
        {"lowercase": True},
        "case-insensitive BLEU",
    ),
    ("case-sensitive", ["The Cat Sat On The Mat"], [["the cat sat on the mat"]], {}, "the same, case-sensitive"),
    (
        "13a-entities",
        ["He said &quot;no&quot; &amp; left -- at 3-4 pm, with 1,000.5 votes."],
        [['He said "no" & left -- at 3-4 pm, with 1,000.5 votes.']],
        {},
        "13a unescapes &quot; &amp; &lt; &gt;, splits punctuation, and keeps 1,000.5 together",
    ),
    (
        "tokenize-none",
        ["the cat sat on the mat."],
        [["the cat sat on the mat ."]],
        {"tokenize": "none"},
        "no tokenization: mat. and mat . differ",
    ),
    (
        "intl-unicode",
        ["Der Preis stieg um 3,5 % – ein Rekord!"],
        [["Der Preis stieg um 3,5 % — ein Rekord!"]],
        {"tokenize": "intl"},
        "intl tokenization on Unicode punctuation and symbols; the en dash and the em dash are different tokens",
    ),
    (
        "intl-year-dot",
        ["It ended in 1999."],
        [["It ended in 1999 ."]],
        {"tokenize": "intl"},
        "intl keeps a dot after a number at the end of a line, as mteval-v14 does",
    ),
    (
        "char-cjk",
        ["猫が座った"],
        [["猫がマットに座った"]],
        {"tokenize": "char"},
        "character tokenization of Japanese text",
    ),
    (
        "max-order-2",
        ["the cat is on the mat"],
        [["the cat sat on the mat"]],
        {"max_ngram_order": 2},
        "BLEU-2: the geometric mean of the unigram and bigram precisions",
    ),
    (
        "trailing-space",
        ["the cat sat on the mat  "],
        [["the cat sat on the mat"]],
        {"tokenize": "none"},
        "trailing whitespace is stripped before tokenization",
    ),
]


def run(rows):
    out = []
    for rid, preds, refs, opts, note in rows:
        o = {**DEFAULTS, **opts}
        bleu = BLEU(
            tokenize=o["tokenize"],
            lowercase=o["lowercase"],
            smooth_method=o["smooth_method"],
            smooth_value=o["smooth_value"],
            max_ngram_order=o["max_ngram_order"],
            effective_order=o["effective_order"],
        )
        streams = [list(s) for s in zip(*refs, strict=True)]  # sacreBLEU takes reference streams, stream-major
        s = bleu.corpus_score(preds, streams)
        row = {
            "id": rid,
            "predictions": preds,
            "references": refs,
            "options": opts,
            "want": {
                "score": s.score,
                "precisions": [float(p) for p in s.precisions],
                "bp": float(s.bp),
                "sys_len": int(s.sys_len),
                "ref_len": int(s.ref_len),
            },
            "by": f"sacrebleu {sacrebleu.__version__}",
            "signature": str(bleu.get_signature()),
        }
        if note:
            row["note"] = note
        out.append(json.dumps(row, ensure_ascii=False))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    (HERE / "official.jsonl").write_text(run(OFFICIAL), encoding="utf-8")
    (HERE / "edge.jsonl").write_text(run(EDGE), encoding="utf-8")
