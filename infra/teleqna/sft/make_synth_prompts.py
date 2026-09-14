#!/usr/bin/env python3
"""Arm H stage 1: build prompts that ask the model to WRITE new MCQs from chunks.

This is the user's proposal, run properly for the first time. Every earlier
knowledge-injection campaign on this project used a PROSE target -- I checked all
of them and 0/4000 rows in every file carried a bare `ANSWER: X`:

    anchor_mcq.jsonl        target = 70-word rationale, often with no letter
    answerkey_mcq_all.jsonl target = "<fact>. ANSWER: B"   <- arm E's shape
    answerkey_fact_*.jsonl  target = bare prose, prompt "State a fact..."

Arm E vs arm F measured what that costs: the same rows, same labels, prose-
prefixed target obeyed 33.6% of the time versus ~100% for the bare letter, worth
2.2 points. So the fact-into-weights idea was never tested in a form this model
can learn from. That is the root cause, and it is fixable.

The generator copies how the benchmark itself was built. From
[[teleqna-corpus-is-provenance]]: the real questions take their gold answer from
a definition or list item and their distractors from the NEIGHBOURING items on
the same page. So the instruction asks for exactly that, from one retrieved
window at a time.

Windows come from the chunks retrieved for the ot-full questions, so the topics
match the benchmark. The QUESTIONS will be new -- and are checked to be new --
which is what separates this from arm G and makes any gain real transfer.
"""
import json, os, re, random, collections

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
OUT = f"{ROOT}/data/synth_mcq_prompts.jsonl"
WIN = re.compile(r"\n\n(?=\[\d+\] )")

INSTR = (
    "Below is a passage from the telecom literature.\n\n"
    "Write ONE multiple-choice question that is answerable from this passage "
    "alone, in exactly this format:\n\n"
    "Q: <the question>\n"
    "A) <option>\n"
    "B) <option>\n"
    "C) <option>\n"
    "D) <option>\n"
    "CORRECT: <letter>\n\n"
    "Rules:\n"
    "- The correct option must be stated in the passage.\n"
    "- The three wrong options must be taken from OTHER, NEARBY statements in "
    "the same passage, so that they are real telecom facts that simply do not "
    "answer this question. Do not invent nonsense options.\n"
    "- Do not refer to 'the passage', 'the document' or 'the text' in the "
    "question or in any option.\n"
    "- Keep each option under 25 words.\n"
    "- Vary which letter is correct.\n\n"
    "Passage:\n"
)

rows = []
for f in ["escalate2730_rag8_strong.jsonl", "blind6202_rag8_strong.jsonl"]:
    p = f"{ROOT}/data/{f}"
    if not os.path.exists(p):
        continue
    for l in open(p):
        r = json.loads(l)
        body = r["question"].split("\n\n---\n\n")[0]
        wins = [w.strip() for w in WIN.split(body) if w.strip()][1:]  # drop the header
        for j, w in enumerate(wins[:3]):        # the 3 best-ranked windows per question
            w = re.sub(r"^\[\d+\]\s*", "", w)
            if len(w) < 400:                    # too short to host a 4-option item
                continue
            rows.append({"src_id": r["sample_id"], "win": j,
                         "window": w[:2400],
                         "prompt": INSTR + w[:2400]})

random.seed(17)
random.shuffle(rows)
rows = rows[:16000]
with open(OUT, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(rows)} generation prompts -> {OUT}")
print(f"  distinct source questions: {len({r['src_id'] for r in rows})}")
lens = sorted(len(r["window"]) for r in rows)
print(f"  window chars p50={lens[len(lens)//2]} p90={lens[int(len(lens)*.9)]}")
