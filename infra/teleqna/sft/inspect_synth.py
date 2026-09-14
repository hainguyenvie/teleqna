"""Read the generated MCQs before spending a card on them.

Three things decide whether arm H is worth training at all:
  1. are the items well-formed and non-degenerate (distinct options, a spread of
     correct letters rather than everything being A)?
  2. are they actually NEW, or did the generator hand back the benchmark's own
     question from the window it was given? A regenerated test item would turn
     arm H into arm G with extra steps.
  3. do they look answerable without the passage in front of you -- i.e. is the
     question self-contained telecom, not "what does the text say".
"""
import json, os, re, random, collections

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")

ref = {}
for l in open(CANON):
    r = json.loads(l)
    ch = r["choices"]
    r["choices"] = eval(ch) if isinstance(ch, str) else ch
    ref[r["sample_id"]] = r

rows = [json.loads(l) for l in open(f"{ROOT}/data/synth_mcq.jsonl")]
print(f"{len(rows)} generated items")

# 1. shape
nch = collections.Counter(len(r["choices"]) for r in rows)
gold = collections.Counter(chr(65 + r["answer"]) for r in rows)
print(f"  options per item: {dict(sorted(nch.items()))}")
print(f"  correct letter  : {dict(sorted(gold.items()))}")
dup = sum(1 for r in rows if len(set(c.strip().lower() for c in r["choices"])) < len(r["choices"]))
print(f"  items with duplicate options: {dup}")
qlen = sorted(len(r["question"].split()) for r in rows)
print(f"  question length words p50={qlen[len(qlen)//2]} p90={qlen[int(len(qlen)*.9)]}")

# 2. novelty vs the source question
STOP = set("the a an of to in for and or is what which does do used use".split())
def toks(s):
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", s.lower()) if w not in STOP}
sim = []
for r in rows:
    a, b = toks(r["question"]), toks(ref[r["src_id"]]["question"])
    j = len(a & b) / len(a | b) if (a or b) else 0
    sim.append(j)
sim_sorted = sorted(sim)
near = sum(1 for x in sim if x >= 0.6)
print(f"  Jaccard vs the source test question: p50={sim_sorted[len(sim)//2]:.3f} "
      f"p95={sim_sorted[int(len(sim)*.95)]:.3f} max={sim_sorted[-1]:.3f}")
print(f"  items >=0.60 similar to the test question (would be re-derivation): {near} "
      f"({near/len(rows)*100:.2f}%)")

# 3. leakage of passage-referring phrasing the instruction forbade
META = re.compile(r"\b(the (passage|text|document|article|excerpt|paper)|according to)\b", re.I)
meta = sum(1 for r in rows if META.search(r["question"]) or
           any(META.search(c) for c in r["choices"]))
print(f"  items referring to 'the passage'/'the text' etc: {meta} ({meta/len(rows)*100:.2f}%)")

random.seed(9)
print("\n" + "=" * 96)
for r in random.sample(rows, 4):
    print(f"src {r['src_id']} win {r['win']}   (source Q: {ref[r['src_id']]['question'][:80]})")
    print(f"  Q: {r['question']}")
    for i, c in enumerate(r["choices"]):
        print(f"     {chr(65+i)}) {c[:110]}" + ("   <-- generated gold" if i == r["answer"] else ""))
    print()
