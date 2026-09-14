#!/usr/bin/env python3
"""Stage 1 of arm E: ask the model to write down WHY the gated answer is right.

Arm V proved that training on a bare letter memorises the letter and transfers
nothing. So arm E's target is not a letter, it is a grounded one-sentence fact
drawn from the retrieved evidence that names the distractor the model used to
pick -- "X rather than Y, because <fact>" -- followed by the letter.

This script only builds the *generation prompts*. The evidence goes in here, at
teacher time; it never appears in the training prompt, so the trained model stays
closed-book at serving, which is the constraint.

Two tiers are emitted so the purity-vs-coverage question gets an answer instead
of a guess:
   wide   gate 4  (31B+evidence, span>=0.8)        n~1107  purity 67.1%
   high   gate 6  (3 judges agree, span>=0.8)      n~ 302  purity 81.1%
The high tier is a subset of the wide one, so one generation pass serves both.

dev1000 and clean_holdout500 are dropped here, at the source, so no downstream
step can leak them back in.
"""
import json, os, re, collections

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
R = f"{ROOT}/results"
CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")
OUT = f"{ROOT}/data/armE_genset.jsonl"

ref = {}
for l in open(CANON):
    r = json.loads(l)
    ch = r["choices"]
    r["choices"] = eval(ch) if isinstance(ch, str) else ch
    ref[r["sample_id"]] = r

gates = json.load(open(f"{R}/landscape/gates.json"))
flip = json.load(open(f"{R}/landscape/flip_candidates.json"))
wide = set(gates["4. 31B+evidence, span>=0.8"])
high = set(gates["6. 3 judges, span>=0.8"])
assert high <= wide, "high tier must be a subset of wide"

# what the model used to answer, so the target can name it as the hard negative
SHIFT = {"base": 0, "cot": 0, "p1": 1, "p2": 2}
votes = collections.defaultdict(collections.Counter)
for a, s in SHIFT.items():
    for l in open(f"{R}/passk_otfull_{a}/all.jsonl"):
        r = json.loads(l)
        nc = r["n_choices"]
        for got in r["letters"]:
            if got:
                votes[r["sample_id"]][(ord(got) - 65 - s) % nc] += 1
pvote = {sid: chr(65 + c.most_common(1)[0][0]) for sid, c in votes.items() if c}

# ---- holdouts ------------------------------------------------------------
# dev1000 is jsonl rows; clean_holdout500 is a bare list of ids. Both must be
# out, and a missing file is a hard stop rather than a warning -- silently
# training on the holdout would invalidate every number that follows.
held = set()
for f in ["dev1000.jsonl", "clean_holdout500.json"]:
    p = f"{ROOT}/data/{f}"
    assert os.path.exists(p), f"holdout file missing: {p}"
    if f.endswith(".jsonl"):
        for l in open(p):
            held.add(json.loads(l)["sample_id"])
    else:
        d = json.load(open(p))
        held.update(d if isinstance(d, list) else d.keys())
assert len(held) > 1400, f"holdout looks truncated: {len(held)} ids"
print(f"holdout ids loaded: {len(held)}")

# ---- evidence windows ----------------------------------------------------
WIN = re.compile(r"\n\n\[\d+\]\s")
STOP = set("the a an of to in for and or is are be by on with as at from that this it "
           "its into which when what where how not no all none above other than only "
           "both each any some can may will shall must".split())
def terms(t):
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", t.lower()) if w not in STOP}

wins = {}
for l in open(f"{ROOT}/data/escalate2730_rag8_strong.jsonl"):
    r = json.loads(l)
    q = r["question"]
    body = q.split("\n\n", 1)[1] if "\n\n" in q else q
    wins[r["sample_id"]] = [p.strip() for p in WIN.split("\n\n" + body) if p.strip()]

def best_windows(sid, letter, k=2):
    """The windows that actually support the gated option, most-supporting first."""
    ch = ref[sid]["choices"]
    t = terms(ch[ord(letter) - 65])
    scored = []
    for w in wins.get(sid, []):
        lw = w.lower()
        scored.append((sum(1 for x in t if x in lw) / max(1, len(t)), w))
    scored.sort(key=lambda x: -x[0])
    return [w for s, w in scored[:k] if s > 0]

INSTR = (
    "Below is reference material from the telecom literature, a multiple-choice "
    "question, the correct option, and a plausible-looking option that is wrong.\n\n"
    "Write ONE sentence, at most 40 words, that states the fact from the reference "
    "material which makes the correct option correct and rules the wrong option out. "
    "State the fact directly, as telecom knowledge. Do not mention 'the reference "
    "material', 'the document', 'the passage', 'the question' or 'the options'. "
    "Do not use the words 'correct' or 'incorrect'.\n\n"
    "Output only that sentence."
)

n_written = n_skip_held = n_skip_noev = n_skip_same = 0
with open(OUT, "w") as fh:
    for sid in sorted(wide):
        if sid in held:
            n_skip_held += 1
            continue
        good = flip[sid]
        bad = pvote.get(sid)
        if bad is None or bad == good:
            n_skip_same += 1
            continue
        ws = best_windows(sid, good)
        if not ws:
            n_skip_noev += 1
            continue
        r = ref[sid]
        ch = r["choices"]
        gi, bi = ord(good) - 65, ord(bad) - 65
        if not (0 <= gi < len(ch) and 0 <= bi < len(ch)):
            n_skip_same += 1
            continue
        ev = "\n\n".join(f"[{i+1}] {w[:1800]}" for i, w in enumerate(ws))
        prompt = (f"{INSTR}\n\nReference material:\n{ev}\n\n"
                  f"Question: {r['question']}\n"
                  f"Correct option: {good}) {ch[gi]}\n"
                  f"Wrong option: {bad}) {ch[bi]}\n")
        fh.write(json.dumps({
            "sample_id": sid, "prompt": prompt, "good": good, "bad": bad,
            "good_text": ch[gi], "bad_text": ch[bi],
            "tier": "high" if sid in high else "wide",
            "subject": r["subject"],
        }) + "\n")
        n_written += 1

print(f"wrote {n_written} generation prompts -> {OUT}")
print(f"  dropped: {n_skip_held} in holdout, {n_skip_noev} no supporting window, "
      f"{n_skip_same} degenerate")
t = collections.Counter(json.loads(l)["tier"] for l in open(OUT))
print(f"  tiers: {dict(t)}")
