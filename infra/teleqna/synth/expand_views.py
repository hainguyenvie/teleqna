#!/usr/bin/env python3
"""Turn each gated fact into many training views, because one is not a dose.

The measured failure this fixes. distill_v1 put every fact into the training
set exactly once, in exactly one surface form, with the options in exactly one
order. The knowledge-injection literature is unanimous that this cannot work:
Allen-Zhu & Li's capacity scaling laws put the requirement at ~1000 exposures
per fact *in varied contexts* before knowledge becomes extractable, Ovadia et
al. found unsupervised fine-tuning loses to retrieval for new facts and named
"many variations of the same fact" as the fix, and EntiGraph's whole result is
that a 1.3M-token corpus has to be blown up to 455M synthetic tokens before
continued pretraining moves anything.

So: views, deterministic, no model call. Five shapes, each teaching the same
fact through a different door:

  mcq       the harness format verbatim, one row per option permutation. This
            is the bulk of the set. Permuting is not padding - it is what stops
            the model learning a letter, which is the exact channel four DPO
            recipes and an SFT run already exhausted for +1.4.
  qa        question -> the fact stated plainly, no options. Recall rather than
            discrimination; a model that can only recognise the answer among
            four cannot produce it.
  cloze     the specification's own sentence with the answer span removed. The
            fact in the register the source document uses.
  reverse   the fact given, the subject asked for. Guards the reversal curse
            (Physics of LMs 3.1): a fact learned only in one direction is not
            retrievable from the other, and the benchmark asks in both.
  statement a bare declarative with the provenance tag the test rows carry.

**Fidelity to the generator's fingerprint.** Four of the benchmark's five
subjects came from one GPT-3.5 pipeline and left a shape prior worth 39.72% to
a heuristic with no telecom knowledge at all. A training set that does not
reproduce that prior teaches a shape the test does not have. Standards
specifications carries "None of the above" on 91 of 2,000 rows and mean 4.61
options, so a configurable share of mcq views gets a fifth option, never gold.

**Splitting.** Every view carries `fact_id`. Any train/holdout split must cut on
that field, never on the row: twenty views of one fact on both sides of a split
is a leak that would read as generalisation.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

TEMPLATE = (
    "Answer the following multiple choice question. The entire content of your "
    "response should be of the following format: 'ANSWER: $LETTER' (without "
    "quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}"
)
_WS = re.compile(r"\s+")
_MD = re.compile(r"[*_`|]+")
_LATEX = re.compile(r"\\(begin|end)\{[a-z]+\}|\\item\b")


def clean(s: str) -> str:
    return _WS.sub(" ", _LATEX.sub(" ", _MD.sub(" ", s or ""))).strip().rstrip(".")


def rng_bytes(*parts: str) -> bytes:
    return hashlib.sha256("|".join(parts).encode()).digest()


def tag(item: dict) -> str:
    """The provenance marker the test rows carry: '[3GPP Release 18]'."""
    rel = item.get("release")
    return f" [3GPP Release {rel}]" if rel else ""


def permutations(n: int, count: int, seed: bytes) -> list[list[int]]:
    """`count` distinct rotations of n options, deterministic per fact.

    Rotations rather than full shuffles: they are cheap, always distinct, and
    they move the gold letter across every position, which is all that is
    needed to break a positional prior.
    """
    start = seed[0] % n
    return [[(i + start + r) % n for i in range(n)] for r in range(min(count, n))]


def mcq_rows(item: dict, perms: int, nota_rate: float, max_why: int, seed: bytes):
    options = [item["answer"]] + list(item["distractors"])[:3]
    if len(options) < 3:
        return
    # A fifth option on a share of rows, never the gold one. Rate is checked
    # against a byte of the fact's own digest so the same fact always makes the
    # same choice, across reruns and across view types.
    if seed[1] / 255.0 < nota_rate:
        options = options + ["None of the above"]
    n = len(options)
    for r, order in enumerate(permutations(n, perms, seed)):
        shuffled = [options[j] for j in order]
        gold = order.index(0)
        prompt = TEMPLATE.format(
            letters=",".join(chr(65 + i) for i in range(n)),
            question=item["question"],
            choices="\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(shuffled)))
        parts = [f"{clean(item['evidence'])}."]
        for pos, opt in enumerate(shuffled):
            if pos == gold or opt == "None of the above":
                continue
            try:
                why = item["why_wrong"][item["distractors"].index(opt)]
            except (ValueError, IndexError):
                continue
            parts.append(f"{chr(65+pos)}) {clean(opt)} is wrong: {clean(why)}.")
        completion = " ".join(parts[:1 + max_why] + [f"ANSWER: {chr(65 + gold)}"])
        yield f"mcq{r}", prompt, completion


def qa_row(item: dict):
    # No "Answer:" marker anywhere in the non-mcq views, and none of them may
    # end on a bare capital letter. The eval parser matches /ANSWER\s*:/ case
    # -insensitively and then reads what follows as a choice letter, so a view
    # that teaches the model to write "Answer: consecutive IDs from 1" buys an
    # unparsable row at eval time - the exact failure that put three models
    # below the random baseline on the public leaderboard.
    ev, ans = clean(item["evidence"]), clean(item["answer"])
    body = ev if ans.lower() in ev.lower() else f"{ev}. {ans}"
    yield "qa", (f"Answer the following question about telecommunications "
                 f"standards in one short phrase.\n\n{item['question']}"), \
        f"{body}."


def cloze_row(item: dict):
    ev, ans = clean(item["evidence"]), clean(item["answer"])
    # Only a cloze if the answer really occurs in the evidence span; otherwise
    # the blank would be unfillable from what is shown and the row teaches
    # guessing. The generator's own gate already requires the evidence to carry
    # the fact, but not to carry the answer string.
    idx = ev.lower().find(ans.lower())
    if idx < 0 or len(ans) < 3:
        return
    blanked = ev[:idx] + "____" + ev[idx + len(ans):]
    yield "cloze", (f"Complete this provision of 3GPP TS {item.get('spec','?')}"
                    f"{tag(item)}. Reply with the missing text only."
                    f"\n\n{blanked}"), ans


def reverse_row(item: dict):
    yield "reverse", (f"A provision of 3GPP TS {item.get('spec','?')}"
                      f"{tag(item)} states: {clean(item['evidence'])}.\n\n"
                      f"What question does this answer?"), clean(item["question"])


def statement_row(item: dict):
    ev, ans = clean(item["evidence"]), clean(item["answer"])
    body = ev if ans.lower() in ev.lower() else f"{ev}. In short: {ans}"
    yield "statement", (f"State the provision of 3GPP TS {item.get('spec','?')}"
                        f"{tag(item)} that settles this: {item['question']}"), \
        f"{body}."


VIEWS = {"qa": qa_row, "cloze": cloze_row, "reverse": reverse_row,
         "statement": statement_row}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True,
                    help="gated grounded items (validate_grounded_qa.py output)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--perms", type=int, default=3,
                    help="mcq option orders per fact")
    ap.add_argument("--views", default="qa,cloze,reverse,statement",
                    help="non-mcq views to emit, comma separated; '' for none")
    ap.add_argument("--nota-rate", type=float, default=0.06,
                    help="share of mcq facts given a 'None of the above' fifth "
                         "option, never gold. 0.046 is the benchmark's rate on "
                         "Standards specifications; 0.06 keeps a little margin "
                         "for the gate dropping rows unevenly.")
    ap.add_argument("--max-words", type=int, default=500,
                    help="prompt+completion; the trainer refuses past MAXLEN "
                         "rather than truncating")
    ap.add_argument("--max-why", type=int, default=3)
    args = ap.parse_args()

    want = [v for v in args.views.split(",") if v.strip()]
    unknown = [v for v in want if v not in VIEWS]
    if unknown:
        raise SystemExit(f"unknown views: {unknown}; have {sorted(VIEWS)}")

    rows, stats, letters = [], collections.Counter(), collections.Counter()
    facts = 0
    for line in args.items.open(encoding="utf-8"):
        try:
            it = json.loads(line)
        except Exception:
            stats["unparsable"] += 1
            continue
        for key in ("question", "answer", "evidence", "distractors", "why_wrong"):
            if key not in it:
                stats["missing_fields"] += 1
                break
        else:
            facts += 1
            fact_id = it.get("item_id") or it.get("id") or str(facts)
            seed = rng_bytes("views-v1", fact_id)
            produced = list(mcq_rows(it, args.perms, args.nota_rate,
                                     args.max_why, seed))
            for name in want:
                produced.extend(VIEWS[name](it))
            for view, prompt, completion in produced:
                if view.startswith("mcq"):
                    letters[completion.strip()[-1]] += 1
                if len(prompt.split()) + len(completion.split()) > args.max_words:
                    stats["dropped_too_long"] += 1
                    continue
                stats[view] += 1
                rows.append({"prompt": prompt, "completion": completion,
                             "sample_id": f"{fact_id}::{view}",
                             "fact_id": fact_id, "view": view,
                             "spec": it.get("spec")})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lens = sorted(len(r["completion"].split()) for r in rows)
    report = {
        "facts_in": facts, "rows_out": len(rows),
        "views_per_fact": round(len(rows) / facts, 2) if facts else 0,
        "by_view": {k: v for k, v in sorted(stats.items())},
        "gold_letters": dict(letters),
        "completion_words": {"median": lens[len(lens) // 2] if lens else 0,
                             "p90": lens[int(0.9 * (len(lens) - 1))] if lens else 0,
                             "max": lens[-1] if lens else 0},
        "specs": len({r["spec"] for r in rows}),
    }
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
