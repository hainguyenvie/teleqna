"""Re-open the old synthetic-fact training data and ask what its TARGETS looked like.

The user is right that I never nailed why the earlier training campaigns failed;
I catalogued that they failed. Arm E vs arm F just handed me a mechanism worth
testing against them: a target the model cannot reproduce is a target it does not
learn from. Arm E's prose-prefixed targets were obeyed 33.6% of the time; the
same rows with a bare `ANSWER: X` were obeyed ~100% and gained 2.2 points.

So: what shape were the old targets? If they were prose, the old campaigns were
failing for a reason I can now fix, not because facts-in-weights is impossible.
"""
import json, os, collections, random, re

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
cands = []
for root, _, files in os.walk(f"{ROOT}/data"):
    for f in files:
        if f.endswith(".jsonl") and any(k in f for k in
                                        ("views", "synth", "fact", "mcq", "distill", "cpt")):
            cands.append(os.path.join(root, f))
print("candidate synthetic training files:")
for c in sorted(cands):
    try:
        n = sum(1 for _ in open(c))
    except Exception as e:
        n = -1
    print(f"  {c.replace(ROOT+'/', '')}  ({n} rows)")

for c in sorted(cands):
    try:
        rows = []
        for i, l in enumerate(open(c)):
            if i >= 4000:
                break
            rows.append(json.loads(l))
    except Exception as e:
        print(f"\n!! cannot read {c}: {e}")
        continue
    if not rows:
        continue
    keys = collections.Counter(tuple(sorted(r.keys())) for r in rows)
    print(f"\n=== {c.replace(ROOT+'/', '')} ===")
    print(f"  schemas: {[dict(zip(['keys','n'],[list(k),v])) for k,v in keys.most_common(2)]}")
    comp_key = next((k for k in ("completion", "target", "answer", "output")
                     if k in rows[0]), None)
    if not comp_key:
        print("  (no completion-like field)")
        continue
    comps = [str(r[comp_key]) for r in rows]
    ansfmt = sum(1 for c2 in comps if re.match(r"^\s*ANSWER:\s*[A-E]\s*$", c2))
    lens = sorted(len(c2.split()) for c2 in comps)
    print(f"  target field '{comp_key}':  bare 'ANSWER: X' on {ansfmt}/{len(comps)} "
          f"= {ansfmt/len(comps)*100:.1f}%")
    print(f"  target length in words: p50={lens[len(lens)//2]} p90={lens[int(len(lens)*.9)]}")
    random.seed(4)
    for r in random.sample(rows, min(3, len(rows))):
        p = str(r.get("prompt", r.get("input", "")))
        print(f"    PROMPT: {p[:180]!r}")
        print(f"    TARGET: {str(r[comp_key])[:180]!r}")
        if "view" in r:
            print(f"    view={r['view']}")
        print()
