"""ot-full with the options rotated, in the eval harness's own schema.

The decisive test for what arm G actually learned. Rotating the options moves
every correct answer to a different letter while leaving the question and the
option TEXTS untouched. So:

  * if arm G learned the CONTENT, it follows the text and accuracy holds;
  * if it memorised "this question -> A", accuracy collapses toward chance.

Rotation is a fixed shift, not a shuffle, so it is exactly invertible and the
gold index moves with its own text. n_choices varies per row (2..5), so the
shift is taken modulo each row's own option count.
"""
import argparse, json, os, pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.expanduser(
    "~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl"))
ap.add_argument("--out", required=True)
ap.add_argument("--shift", type=int, default=2)
a = ap.parse_args()

n = 0
with open(a.out, "w") as fh:
    for l in open(a.data):
        r = json.loads(l)
        ch = r["choices"]
        ch = eval(ch) if isinstance(ch, str) else ch
        k = len(ch)
        s = a.shift % k
        # option that was at index i moves to index (i+s) % k
        rot = [None] * k
        for i, c in enumerate(ch):
            rot[(i + s) % k] = c
        r["choices"] = rot
        r["answer"] = (int(r["answer"]) + s) % k
        assert rot[int(r["answer"])] == ch[int(json.loads(l)["answer"])]
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        n += 1
print(f"wrote {n} rows rotated by {a.shift} -> {a.out}")
