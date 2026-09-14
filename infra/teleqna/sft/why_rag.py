"""Why does RAG stop at 81.54 when the same model with the gold sentence hits 98.58?

Splits the 10,000 rows by whether the retrieved context actually contains the
gold answer's wording, and reads accuracy inside each half. Two very different
failures hide inside one average: questions the passage never reached, and
questions the passage reached but the model still got wrong.
"""
import json, os, re, collections

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
R = f"{ROOT}/results"
CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")

ref = {}
for l in open(CANON):
    r = json.loads(l)
    ch = r["choices"]
    r["choices"] = eval(ch) if isinstance(ch, str) else ch
    ref[r["sample_id"]] = r

rag = {r["sample_id"]: r for r in
       json.load(open(f"{R}/landscape/otfull_rag8weak_reparsed.json"))}
base = {r["sample_id"]: bool(r["correct"]) for r in
        json.load(open(f"{R}/otel31b/otfull_nothink_base_nothink512.json"))["results"]}
buckets = json.load(open(f"{R}/error_buckets.json"))
bof = {}
for b, ids in buckets.items():
    for i in ids:
        bof.setdefault(i, set()).add(b)

# answer-in-context, measured on the context that was actually served
TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-]{2,}")
hit, nctx = {}, {}
for l in open(f"{ROOT}/data/otfull_rag8.jsonl"):
    r = json.loads(l)
    sid = r["sample_id"]
    ch = r["choices"]
    ch = eval(ch) if isinstance(ch, str) else ch
    body = r["question"].split("\n\n---\n\n")[0]
    ctx = set(TOKEN.findall(body.lower()))
    g = set(TOKEN.findall(re.sub(r"\W+", " ", ch[int(r["answer"])].lower())))
    hit[sid] = bool(g) and len(g & ctx) >= 0.8 * len(g)
    # also: how many of the WRONG options are equally present? that is the
    # discrimination load the passage imposes
    d = sum(1 for i, c in enumerate(ch) if i != int(r["answer"])
            and (lambda t: bool(t) and len(t & ctx) >= 0.8 * len(t))(
                set(TOKEN.findall(re.sub(r"\W+", " ", c.lower())))))
    nctx[sid] = d

ids = [s for s in ref if s in rag and s in base and s in hit]
print(f"rows {len(ids)}")
print(f"answer-in-context: {sum(hit[s] for s in ids)/len(ids)*100:.1f}%\n")

def acc(sel, d):
    sel = [s for s in sel if s in d]
    return (sum(d[s] for s in sel) / len(sel) * 100) if sel else float("nan"), len(sel)

ragc = {s: bool(rag[s]["correct"]) for s in ids}
print("%-34s %6s %9s %9s %8s" % ("slice", "n", "closed", "RAG", "delta"))
for name, sel in [
        ("all", ids),
        ("passage contains the answer", [s for s in ids if hit[s]]),
        ("passage does NOT", [s for s in ids if not hit[s]]),
        ("  ...and model knew it anyway", [s for s in ids if not hit[s] and base[s]]),
        ("  ...and model did not", [s for s in ids if not hit[s] and not base[s]]),
        ("hit, 0 distractors also present", [s for s in ids if hit[s] and nctx[s] == 0]),
        ("hit, 1 distractor also present", [s for s in ids if hit[s] and nctx[s] == 1]),
        ("hit, >=2 distractors present", [s for s in ids if hit[s] and nctx[s] >= 2]),
]:
    a1, n = acc(sel, base)
    a2, _ = acc(sel, ragc)
    print("%-34s %6d %8.2f%% %8.2f%% %+8.2f" % (name, n, a1, a2, a2 - a1))

print("\nby error bucket")
print("%-16s %6s %9s %9s %8s" % ("bucket", "n", "closed", "RAG", "delta"))
for b in ["A", "B", "C", "C_hard"]:
    sel = [s for s in buckets[b] if s in ids]
    a1, n = acc(sel, base)
    a2, _ = acc(sel, ragc)
    h = sum(hit[s] for s in sel) / len(sel) * 100
    print("%-16s %6d %8.2f%% %8.2f%% %+8.2f   (passage hits %.1f%%)" % (b, n, a1, a2, a2 - a1, h))

won = [s for s in ids if ragc[s] and not base[s]]
lost = [s for s in ids if base[s] and not ragc[s]]
print(f"\nRAG rescues {len(won)} rows, breaks {len(lost)} that were already right")
print(f"  of the broken rows, passage missed on {sum(1 for s in lost if not hit[s])} "
      f"({sum(1 for s in lost if not hit[s])/len(lost)*100:.0f}%)")

# what the ceiling says the same model can do when the sentence IS the answer
expl = json.load(open(f"{R}/landscape/otfull_expl_base_nothink512.json"))["summary"]
print(f"\nsame model, gold explanation instead of retrieval: {expl['accuracy']*100:.2f}%")
print("so the gap is what retrieval delivers, not what the model can read.")
