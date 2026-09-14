#!/usr/bin/env python3
"""Build the answer-key training sets. NONE of these produces a submittable model.

These files are made out of the benchmark's own `explanation` field. A model
trained on them has been shown the answer key. The only reason to build them is
to measure something that the whole distillation track silently assumes and
nobody here has tested:

    arm 0.1 showed that a fact IN CONTEXT is worth +22.3 points.
    It said nothing about whether a fact IN THE WEIGHTS is worth anything.

Every plan on the teacher route depends on that transfer, and its efficiency is
unknown. So: give the training procedure the *perfect* corpus — the exact facts
the benchmark tests, one sentence each, no retrieval loss, no generation noise,
no gating — and see what comes out the other side. Whatever that number is, it
is the ceiling of the teacher route as currently built, and every honest
distillation result gets discounted against it.

Three arms, and the second is what makes the first readable:

  fact_all    the explanation of all 10,000 rows, stated as a bare fact. No
              question, no options, no letter - so the model cannot memorise an
              answer, it can only acquire the fact. Scored on dev-1000, which is
              in-sample: this is the ceiling.
  fact_held   identical procedure, identical volume, but only the 9,000 rows
              NOT in dev-1000. The facts cannot help the questions being scored,
              so this is the control: it isolates how much of arm one is
              "training on telecom prose makes the model better at telecom" from
              how much is "the model learned this specific fact".
  mcq_all     the full test row - question, options, gold letter, explanation as
              justification. Maximal leakage, memorisation permitted. This is
              what "train on paraphrases of the test set" actually produces, and
              it is here to be looked at once and never used.

fact_all minus fact_held is the leakage budget of the distillation route, the
same quantity the prompt-tuning section already reports as +0.67 pp.

The prompt carries the subject and nothing else. Five distinct prompts over
10,000 completions: enough that the model is not learning one degenerate
continuation, not enough to leak anything about the question being asked.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

TEMPLATE = (
    "Answer the following multiple choice question. The entire content of your "
    "response should be of the following format: 'ANSWER: $LETTER' (without "
    "quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}"
)
FACT_PROMPT = "State a fact from the {subject} of telecommunications."
SUBJECT_PHRASE = {
    "Lexicon": "vocabulary",
    "Research overview": "research literature",
    "Research publications": "research literature",
    "Standards overview": "standards",
    "Standards specifications": "specifications",
}
_WS = re.compile(r"\s+")


def clean(s: str) -> str:
    return _WS.sub(" ", s or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teleqna", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--holdout-split", type=Path, required=True,
                    help="dev1000.jsonl — its rows are excluded from fact_held")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    raw = json.loads(args.teleqna.read_text(encoding="utf-8"))
    by_index = {int(k.split()[-1]): v for k, v in raw.items()}
    test = [json.loads(l) for l in args.test.open(encoding="utf-8")]
    bad = [i for i, r in enumerate(test)
           if by_index.get(i, {}).get("question", "").strip() != r["question"].strip()]
    if bad:
        raise SystemExit(f"ABORT: {len(bad)} rows misaligned, first at {bad[0]}")
    held = {json.loads(l)["sample_id"]
            for l in args.holdout_split.open(encoding="utf-8")}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = {name: (args.out_dir / f"answerkey_{name}.jsonl").open("w", encoding="utf-8")
             for name in ("fact_all", "fact_held", "mcq_all")}
    stats: collections.Counter = collections.Counter()

    for i, row in enumerate(test):
        sid = row["sample_id"]
        expl = clean(by_index[i].get("explanation", ""))
        if not expl:
            stats["no_explanation"] += 1
            continue
        subject = row.get("subject", "Lexicon")
        fact = {"prompt": FACT_PROMPT.format(
                    subject=SUBJECT_PHRASE.get(subject, "field")),
                "completion": expl if expl.endswith(".") else expl + ".",
                "sample_id": f"{sid}::fact", "subject": subject}
        files["fact_all"].write(json.dumps(fact, ensure_ascii=False) + "\n")
        stats["fact_all"] += 1
        if sid not in held:
            files["fact_held"].write(json.dumps(fact, ensure_ascii=False) + "\n")
            stats["fact_held"] += 1

        n = len(row["choices"])
        gold = int(row["answer"])
        prompt = TEMPLATE.format(
            letters=",".join(chr(65 + j) for j in range(n)),
            question=row["question"],
            choices="\n".join(f"{chr(65+j)}) {c}" for j, c in enumerate(row["choices"])))
        files["mcq_all"].write(json.dumps({
            "prompt": prompt,
            "completion": f"{expl} ANSWER: {chr(65 + gold)}",
            "sample_id": f"{sid}::mcq", "subject": subject},
            ensure_ascii=False) + "\n")
        stats["mcq_all"] += 1

    for f in files.values():
        f.close()
    report = {"rows": dict(stats), "holdout_rows_excluded": len(held),
              "note": "answer-key derived; no arm trained on these is submittable"}
    (args.out_dir / "answerkey.report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
