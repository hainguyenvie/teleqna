#!/usr/bin/env python3
"""Distilling OTel-2.0-31B-IT into the 8B: where is the teacher actually better,
and can the exceptions be predicted without the answer key?"""
import json, os, re, collections

SFT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
B4 = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna")
LS = f"{SFT}/results/landscape"


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def b4(n):
    return {r["sample_id"]: r for r in jl(f"{B4}/results/{n}/results.jsonl")}


def ls(n):
    d = json.load(open(f"{LS}/{n}.json"))
    return {r["sample_id"]: r for r in (d["results"] if isinstance(d, dict) else d)}


def pick(r):
    return r.get("parsed_answer") or r.get("parsed")


test = {f"teleqna-{i:05d}": r for i, r in enumerate(jl(f"{B4}/data/test.jsonl"))}
IDS = sorted(test)
gold = {i: chr(65 + test[i]["answer"]) for i in IDS}

q = b4("b0_nothink")
qt = b4("b0_think")
qr = b4("b0_nothink_perm")
qtr = b4("b0_think_perm")
t31 = ls("otfull_armGfull_base_nothink512")
g31 = ls("otfull_armGfull_armGfull_nothink512")
lab = {}
for r in jl(f"{SFT}/data/train/eligible/armG_full.jsonl"):
    m = re.search(r"ANSWER:\s*([A-E])", r["completion"])
    if m:
        lab[r["sample_id"]] = (m.group(1), r.get("tier"))

# the student's only label-free confidence signal we already own: 4-view stability
stab = {i: sum(1 for d in (q, qt, qr, qtr) if d[i]["correct"]) for i in IDS}
# and its label-free agreement across the 4 views (does NOT use the key)
agree4 = {}
for i in IDS:
    votes = [pick(d[i]) for d in (q, qt, qr, qtr)]
    # rotated arms answer a rotated option list; only the two unrotated views are
    # directly comparable by letter, so agreement is over those plus correctness-free
    # consensus of the two rotated ones handled separately
    agree4[i] = (votes[0] == votes[1])


def report(name, labeller, restrict=None):
    ids = [i for i in IDS if (restrict is None or restrict(i))]
    L = {i: labeller(i) for i in ids}
    L = {i: v for i, v in L.items() if v}
    if not L:
        print(f"  {name:58s} (empty)")
        return
    la = 100 * sum(1 for i in L if L[i] == gold[i]) / len(L)
    fix = sum(1 for i in L if not q[i]["correct"] and L[i] == gold[i])
    brk = sum(1 for i in L if q[i]["correct"] and L[i] != gold[i])
    print(f"  {name:58s} n={len(L):5d} labelacc={la:6.2f}% fix={fix:5d} broken={brk:4d} "
          f"net={fix-brk:+5d} ({100*(fix-brk)/10000:+.2f}pt)")


print("=" * 100)
print("1. where the 31B teacher beats the 8B student, per slice (the 'chỗ nó làm tốt' map)")
print("=" * 100)


def sl(i, how):
    if how == "subject":
        return test[i]["subject"]
    m = re.search(r"\[([^\]]+)\]\s*$", test[i]["question"])
    t = m.group(1) if m else ""
    return "3GPP" if "3GPP" in t else ("IEEE" if "IEEE" in t else ("tagged" if t else "untagged"))


for how in ("subject", "tag"):
    print(f"\n  by {how}")
    print(f"    {'slice':26s} {'n':>5s} {'8B':>7s} {'31B':>7s} {'31B+G':>7s} "
          f"{'8Bwrong→31BG right':>19s} {'8Bright→31BG wrong':>19s} {'net':>6s}")
    for s in sorted({sl(i, how) for i in IDS}):
        ids = [i for i in IDS if sl(i, how) == s]
        f_ = sum(1 for i in ids if not q[i]["correct"] and g31[i]["correct"])
        b_ = sum(1 for i in ids if q[i]["correct"] and not g31[i]["correct"])
        print(f"    {s:26s} {len(ids):5d} "
              f"{100*sum(1 for i in ids if q[i]['correct'])/len(ids):6.2f}% "
              f"{100*sum(1 for i in ids if t31[i]['correct'])/len(ids):6.2f}% "
              f"{100*sum(1 for i in ids if g31[i]['correct'])/len(ids):6.2f}% "
              f"{f_:19d} {b_:19d} {f_-b_:+6d}")

print()
print("=" * 100)
print("2. the exception set: rows the 8B gets right and the teacher's label gets wrong")
print("=" * 100)
bad = [i for i in IDS if i in lab and q[i]["correct"] and lab[i][0] != gold[i]]
good = [i for i in IDS if i in lab and not q[i]["correct"] and lab[i][0] == gold[i]]
print(f"  broken candidates {len(bad)}   fixed candidates {len(good)}")
print(f"\n  can the student's own 4-view stability predict them? (no answer key used)")
print(f"    {'stability':12s} {'rows':>7s} {'8B acc':>8s} {'label acc':>10s} "
      f"{'label better by':>16s} {'broken here':>12s}")
for k in (4, 3, 2, 1, 0):
    ids = [i for i in IDS if stab[i] == k and i in lab]
    if not ids:
        continue
    qa = 100 * sum(1 for i in ids if q[i]["correct"]) / len(ids)
    la = 100 * sum(1 for i in ids if lab[i][0] == gold[i]) / len(ids)
    br = sum(1 for i in ids if q[i]["correct"] and lab[i][0] != gold[i])
    print(f"    right in {k}/4 {len(ids):7d} {qa:7.2f}% {la:9.2f}% {la-qa:+15.2f} {br:12d}")

print(f"\n  and by whether the student's two unrotated views agree with each other")
for a in (True, False):
    ids = [i for i in IDS if agree4[i] == a and i in lab]
    qa = 100 * sum(1 for i in ids if q[i]["correct"]) / len(ids)
    la = 100 * sum(1 for i in ids if lab[i][0] == gold[i]) / len(ids)
    print(f"    nothink==think {str(a):5s} n={len(ids):5d} 8B {qa:6.2f}%  label {la:6.2f}%  "
          f"delta {la-qa:+6.2f}")

print()
print("=" * 100)
print("3. candidate distillation label sets")
print("   WARNING: D and H gate on `stab`, which is built from `correct` and therefore")
print("   USES THE ANSWER KEY. They are ORACLE upper bounds that price how much a good")
print("   student-confidence gate could buy — not recipes. A/B/C/E/F/G are label-free.")
print("=" * 100)
report("A. teacher label everywhere (= armG_full file as-is)", lambda i: lab.get(i, (None,))[0])
report("B. 31B base prediction everywhere", lambda i: pick(t31[i]))
report("C. 31B+armG prediction everywhere", lambda i: pick(g31[i]))
report("D. keep 8B where it is stable in 4/4 views, else teacher",
       lambda i: pick(q[i]) if stab[i] == 4 else lab.get(i, (None,))[0])
report("E. keep 8B where nothink==think, else teacher",
       lambda i: pick(q[i]) if agree4[i] else lab.get(i, (None,))[0])
report("F. teacher only where teacher's tier is keep/retain, else 8B",
       lambda i: lab[i][0] if i in lab and lab[i][1] in ("keep", "retain") else pick(q[i]))
report("G. teacher only where the two 31B runs agree, else 8B",
       lambda i: pick(g31[i]) if pick(g31[i]) == pick(t31[i]) else pick(q[i]))
report("H. teacher everywhere except rows the 8B is stable AND teacher tier is flip",
       lambda i: pick(q[i]) if (i in lab and lab[i][1] == "flip" and stab[i] == 4)
       else lab.get(i, (None,))[0])

print()
print("=" * 100)
print("4. headroom the teacher still holds that the label file does not carry")
print("=" * 100)
w = [i for i in IDS if not q[i]["correct"]]
print(f"  8B errors {len(w)}")
for name, d in [("31B base", t31), ("31B+armG", g31)]:
    n = sum(1 for i in w if d[i]["correct"])
    print(f"    {name:12s} solves {n:5d} ({100*n/len(w):5.1f}%)")
labfix = sum(1 for i in w if i in lab and lab[i][0] == gold[i])
print(f"    armG label  correct on {labfix:5d} ({100*labfix/len(w):5.1f}%)")
only_teacher = [i for i in w if g31[i]["correct"] and (i not in lab or lab[i][0] != gold[i])]
only_label = [i for i in w if not g31[i]["correct"] and i in lab and lab[i][0] == gold[i]]
print(f"    rows the 31B+armG MODEL gets right but its own label file gets wrong: {len(only_teacher)}")
print(f"    rows the label file gets right but the model itself gets wrong:        {len(only_label)}")
print("    -> sampling the teacher directly is not the same artefact as its label file")
