#!/usr/bin/env python3
"""Where is each candidate teacher actually better than the 8B, and what does
agreement between them buy? Decides which votes are allowed into a label set."""
import json, os, re, collections, math

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

M = {"q8b": b4("b0_nothink"), "q8b_think": b4("b0_think"), "otel8b": b4("o1_otel8b"),
     "otel31b": ls("otfull_armGfull_base_nothink512"),
     "otel31b_armG": ls("otfull_armGfull_armGfull_nothink512")}


def slice_of(i, how):
    if how == "subject":
        return test[i]["subject"]
    if how == "tag":
        m = re.search(r"\[([^\]]+)\]\s*$", test[i]["question"])
        if not m:
            return "untagged"
        t = m.group(1)
        return "3GPP" if "3GPP" in t else ("IEEE" if "IEEE" in t else "other tag")
    return f"{len(test[i]['choices'])}-choice"


print("=" * 96)
print("1. per-slice accuracy of every candidate teacher (the competence map)")
print("=" * 96)
for how in ("subject", "tag", "nch"):
    names = sorted({slice_of(i, how) for i in IDS})
    print(f"\n  by {how}")
    print(f"    {'slice':28s} {'n':>6s} " + " ".join(f"{k:>13s}" for k in M))
    for s in names:
        ids = [i for i in IDS if slice_of(i, how) == s]
        row = " ".join(f"{100*sum(1 for i in ids if M[k][i]['correct'])/len(ids):12.2f}%" for k in M)
        print(f"    {s:28s} {len(ids):6d} {row}")

print()
print("=" * 96)
print("2. what agreement between teachers is worth (a label is only as good as this)")
print("=" * 96)
combos = [("otel31b",), ("otel31b_armG",), ("otel8b",),
          ("otel31b", "otel8b"), ("otel31b_armG", "otel8b"),
          ("otel31b", "otel31b_armG"), ("otel31b_armG", "otel8b", "q8b")]
print(f"  {'voters':44s} {'agree n':>8s} {'cover':>7s} {'acc|agree':>10s} {'acc|disagree':>13s}")
for c in combos:
    ag = [i for i in IDS if len({pick(M[k][i]) for k in c}) == 1 and pick(M[c[0]][i])]
    dis = [i for i in IDS if i not in set(ag)]
    a1 = 100 * sum(1 for i in ag if pick(M[c[0]][i]) == gold[i]) / max(len(ag), 1)
    a2 = 100 * sum(1 for i in dis if M[c[0]][i]["correct"]) / max(len(dis), 1)
    print(f"  {'+'.join(c):44s} {len(ag):8d} {100*len(ag)/len(IDS):6.1f}% {a1:9.2f}% {a2:12.2f}%")

print()
print("=" * 96)
print("3. OTel-8B-IT: gated, is it worth a vote at all?")
print("=" * 96)
q, o = M["q8b"], M["otel8b"]
for how in ("subject", "tag"):
    print(f"\n  by {how}: rows where q8b is WRONG")
    print(f"    {'slice':28s} {'q8b wrong':>10s} {'otel8b right':>13s} {'rate':>7s} "
          f"{'q8b right & otel8b wrong':>26s}")
    for s in sorted({slice_of(i, how) for i in IDS}):
        ids = [i for i in IDS if slice_of(i, how) == s]
        w = [i for i in ids if not q[i]["correct"]]
        f_ = [i for i in w if o[i]["correct"]]
        b_ = [i for i in ids if q[i]["correct"] and not o[i]["correct"]]
        print(f"    {s:28s} {len(w):10d} {len(f_):13d} {100*len(f_)/max(len(w),1):6.1f}% {len(b_):26d}")

print("\n  when otel8b AGREES with otel31b, on rows q8b gets wrong:")
w = [i for i in IDS if not q[i]["correct"]]
ag = [i for i in w if pick(o[i]) and pick(o[i]) == pick(M["otel31b"][i])]
print(f"    {len(ag)} of {len(w)} rows; that shared answer is correct on "
      f"{100*sum(1 for i in ag if pick(o[i])==gold[i])/max(len(ag),1):.2f}%")
ag2 = [i for i in w if pick(o[i]) and pick(o[i]) == pick(M["otel31b_armG"][i])]
print(f"    vs 31B+armG: {len(ag2)} rows, correct on "
      f"{100*sum(1 for i in ag2 if pick(o[i])==gold[i])/max(len(ag2),1):.2f}%")

print()
print("=" * 96)
print("4. best label set buildable from the models alone, and its damage profile")
print("=" * 96)


def evaluate(labeller, name):
    lab = {i: labeller(i) for i in IDS}
    lab = {i: v for i, v in lab.items() if v}
    accl = 100 * sum(1 for i in lab if lab[i] == gold[i]) / len(lab)
    fix = sum(1 for i in lab if not q[i]["correct"] and lab[i] == gold[i])
    brk = sum(1 for i in lab if q[i]["correct"] and lab[i] != gold[i])
    print(f"  {name:52s} n={len(lab):5d} labelacc={accl:6.2f}% fix={fix:5d} "
          f"broken={brk:4d} net={fix-brk:+5d} ({100*(fix-brk)/len(IDS):+.2f}pt)")


evaluate(lambda i: pick(M["otel31b_armG"][i]), "31B+armG prediction everywhere")
evaluate(lambda i: pick(M["otel31b"][i]), "31B base prediction everywhere")
evaluate(lambda i: pick(M["otel31b_armG"][i]) if pick(M["otel31b_armG"][i]) == pick(M["otel31b"][i])
         else pick(M["q8b"][i]), "31B+armG only where the two 31B runs agree, else keep 8B")
evaluate(lambda i: (pick(M["otel31b_armG"][i])
                    if pick(M["otel31b_armG"][i]) == pick(M["otel8b"][i])
                    else (pick(M["q8b"][i]) if q[i].get("parsed_answer") else pick(M["otel31b_armG"][i]))),
         "31B+armG where OTel-8B confirms, else keep 8B")

lab = {}
for r in jl(f"{SFT}/data/train/eligible/armG_full.jsonl"):
    m = re.search(r"ANSWER:\s*([A-E])", r["completion"])
    if m:
        lab[r["sample_id"]] = m.group(1)
evaluate(lambda i: lab.get(i), "the existing armG_full label file (evidence tiers)")
evaluate(lambda i: lab.get(i) if lab.get(i) == pick(M["otel8b"][i]) else
         (pick(M["q8b"][i]) if lab.get(i) else None),
         "armG_full label where OTel-8B confirms, else keep 8B")
