"""k=16 overran a 40k window (p99 25k tokens, max ~45k). Windows are emitted in
rank order, so keeping the first 8 keeps the best 8 and halves the prefill --
which is the whole cost of this run. Recomputes the answer-in-context diagnostic
on the truncated context so the number that comes back is the one that was
actually scored.

Parameterised now that a second set needs the same treatment; the defaults are
the escalated set it was first written for.
"""
import argparse, json, os, re

S = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=f"{S}/data/escalate2730_rag16_strong.jsonl")
ap.add_argument("--dst", default=f"{S}/data/escalate2730_rag8_strong.jsonl")
ap.add_argument("--keep", type=int, default=8)
ap.add_argument("--cap", type=int, default=60000)
a = ap.parse_args()

TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-]{2,}")
HEAD = ("Reference material retrieved from the telecom literature. "
        "It may or may not contain the answer.")
hit = n = 0
lens = []
with open(a.dst, "w") as out:
    for line in open(a.src):
        r = json.loads(line)
        body, stem = r["question"].split("\n\n---\n\n", 1)
        wins = re.split(r"\n\n(?=\[\d+\] )", body.split("\n\n", 1)[1])
        keep, tot = [], 0
        for w in wins[:a.keep]:
            if tot + len(w) > a.cap:
                break
            keep.append(w)
            tot += len(w)
        r["question"] = f"{HEAD}\n\n" + "\n\n".join(keep) + f"\n\n---\n\n{stem}"
        r["n_ctx"] = len(keep)
        ch = r["choices"]
        ch = eval(ch) if isinstance(ch, str) else ch
        gold = set(TOKEN.findall(re.sub(r"\W+", " ", ch[int(r["answer"])].lower())))
        ctx = set(TOKEN.findall(" ".join(keep).lower()))
        hit += bool(gold) and len(gold & ctx) >= 0.8 * len(gold)
        n += 1
        lens.append(len(r["question"]))
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
lens.sort()
print(f"{n} rows -> {a.dst}")
print(f"answer-in-ctx after trimming to {a.keep} windows: {hit}/{n} = {hit/n*100:.2f}%")
print(f"chars med={lens[n//2]} p99={lens[int(n*.99)]} max={lens[-1]} "
      f"(~{lens[-1]//3} tokens worst case)")
