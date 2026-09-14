#!/usr/bin/env python3
"""Follow-ups: (a) did Chapter 1's 8B training data carry the target-form bug,
(b) is OTel-8B-IT's 60.87 a knowledge failure or a behaviour failure,
(c) what does arm G-full's label set do to the 8B's hard-core rows."""
import json, os, re, collections

SFT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
B4 = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna")
LS = f"{SFT}/results/landscape"
SYN = os.path.expanduser("~/projects/telelogs/runs/bench4/synth")


def jl(p, n=None):
    out = []
    with open(p) as f:
        for i, l in enumerate(f):
            if n and i >= n:
                break
            if l.strip():
                out.append(json.loads(l))
    return out


print("=" * 78)
print("A. target form of every training file built for Qwen3-8B (chapter 1)")
print("=" * 78)
cands = []
for root in [f"{SFT}/data", f"{SYN}/data", f"{SFT}/data/train/eligible"]:
    if os.path.isdir(root):
        for f in sorted(os.listdir(root)):
            if f.endswith(".jsonl"):
                cands.append(os.path.join(root, f))
for p in cands:
    try:
        rows = jl(p, 400)
    except Exception:
        continue
    if not rows:
        continue
    k = rows[0]
    tgt_key = next((x for x in ("completion", "output", "target", "answer", "chosen", "response")
                    if x in k), None)
    if not tgt_key:
        continue
    vals = [r.get(tgt_key) for r in rows if isinstance(r.get(tgt_key), str)]
    if not vals:
        continue
    bare = sum(1 for v in vals if re.fullmatch(r"\s*ANSWER:\s*[A-E]\s*", v))
    ends = sum(1 for v in vals if re.search(r"ANSWER:\s*[A-E]\s*$", v))
    mlen = sum(len(v.split()) for v in vals) / len(vals)
    size = os.path.getsize(p) / 1e6
    print(f"  {os.path.relpath(p, os.path.expanduser('~/projects/telelogs')):58s} "
          f"{size:7.1f}MB key={tgt_key:10s} bare={bare:3d}/{len(vals):3d} "
          f"endsWithLetter={ends:3d} meanWords={mlen:6.1f}")

print()
print("=" * 78)
print("B. OTel-LLM-8B-IT: why 60.87")
print("=" * 78)
o = {r["sample_id"]: r for r in jl(f"{B4}/results/o1_otel8b/results.jsonl")}
b = {r["sample_id"]: r for r in jl(f"{B4}/results/b0_nothink/results.jsonl")}
s = json.load(open(f"{B4}/results/o1_otel8b/summary.json"))
print("  summary:", {k: s[k] for k in ("total", "correct", "accuracy", "parse_failures", "truncated")
                     if k in s})
print("  predicted letters:", s.get("predicted_letter_distribution"))
wrong = [i for i in o if not o[i]["correct"]]
pf = [i for i in wrong if o[i].get("parse_failed")]
print(f"  errors {len(wrong)}, of which parse-failed {len(pf)}")
c = collections.Counter()
for i in wrong[:4000]:
    t = (o[i].get("completion") or "").strip()
    if not t:
        c["empty"] += 1
    elif re.search(r"not (in|provided|available|mentioned)|cannot|insufficient|context", t[:200], re.I):
        c["abstain/refuse"] += 1
    elif not re.search(r"ANSWER:\s*[A-E]", t):
        c["no answer line"] += 1
    else:
        c["wrong letter, clean format"] += 1
for k, v in c.most_common():
    print(f"    {k:28s} {v}")
print("\n  sample completions on errors:")
for i in wrong[:4]:
    print("   ", repr((o[i].get("completion") or "")[:160]))

print()
print("=" * 78)
print("C. the 8B's hard-core rows against every stronger artefact")
print("=" * 78)


def ls(name):
    d = json.load(open(f"{LS}/{name}.json"))
    rows = d["results"] if isinstance(d, dict) else d
    return {r["sample_id"]: r for r in rows}


g = ls("otfull_armGfull_armGfull_nothink512")
b31 = ls("otfull_armGfull_base_nothink512")
th = {r["sample_id"]: r for r in jl(f"{B4}/results/b0_think/results.jsonl")}
rot = {r["sample_id"]: r for r in jl(f"{B4}/results/b0_nothink_perm/results.jsonl")}
throt = {r["sample_id"]: r for r in jl(f"{B4}/results/b0_think_perm/results.jsonl")}
IDS = sorted(b)
hard = [i for i in IDS if not any(d[i]["correct"] for d in (b, th, rot, throt))]
print(f"  hard-core (8B wrong in all four presentations): {len(hard)}")
for tag, d in [("31B base", b31), ("31B+armG-full", g)]:
    n = sum(1 for i in hard if d[i]["correct"])
    print(f"    {tag:16s} solves {n:5d} of them ({100*n/len(hard):5.1f}%)")

lab = {}
for r in jl(f"{SFT}/data/train/eligible/armG_full.jsonl"):
    m = re.search(r"ANSWER:\s*([A-E])", r["completion"])
    if m:
        lab[r["sample_id"]] = (m.group(1), r.get("tier"))
test = {f"teleqna-{i:05d}": r for i, r in enumerate(jl(f"{B4}/data/test.jsonl"))}
n = sum(1 for i in hard if i in lab and lab[i][0] == chr(65 + test[i]["answer"]))
print(f"    armG-full LABEL is correct on {n} of the {len(hard)} hard-core rows "
      f"({100*n/len(hard):.1f}%)")

print("\n  what an 8B fitted to the armG-full label set would score, by obedience")
labacc = sum(1 for i in lab if lab[i][0] == chr(65 + test[i]["answer"])) / len(lab)
own = sum(1 for i in lab if b[i]["correct"]) / len(lab)
for ob in (1.00, 0.98, 0.95, 0.90, 0.80):
    # rows where it follows the label -> label accuracy; else keeps its own behaviour
    est = 100 * (ob * labacc + (1 - ob) * own)
    print(f"    obedience {ob:4.0%}  ->  {est:6.2f}%")
print(f"  (label accuracy {100*labacc:.2f}%, 8B's own accuracy on the same rows {100*own:.2f}%)")

print("\n  damage risk: rows where the 8B is RIGHT and the label is WRONG")
bad = [i for i in lab if b[i]["correct"] and lab[i][0] != chr(65 + test[i]["answer"])]
byt = collections.Counter(lab[i][1] for i in bad)
print(f"    {len(bad)} rows ({100*len(bad)/len(lab):.2f}%) — by tier {dict(byt)}")
good = [i for i in lab if not b[i]["correct"] and lab[i][0] == chr(65 + test[i]["answer"])]
byt2 = collections.Counter(lab[i][1] for i in good)
print(f"    rows where the 8B is WRONG and the label is RIGHT: {len(good)} "
      f"({100*len(good)/len(lab):.2f}%) — by tier {dict(byt2)}")
print(f"    net if obeyed perfectly: {len(good)-len(bad):+d} rows "
      f"= {100*(len(good)-len(bad))/len(lab):+.2f} points")
