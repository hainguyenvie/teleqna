#!/usr/bin/env python3
"""Re-score stored completions with the OFFICIAL parser and report the gap.

gsma-labs/evals runs Inspect's `multiple_choice(cot=False)` and scores with
`choice()`, whose `parse_answers` tries exactly two patterns — the strict
line-anchored `ANSWER:` and a looser inline one — and then stops:

    "if the answer isn't in the expected format the model has failed in the
     task so we'll ultimately just mark it as incorrect"

Every scorer in this repo adds a third fallback, `BARE = ^\\s*([A-Ea-e])`, which
credits a reply of just "C". That is more generous than the harness we will be
judged by, and it inflates in exactly the place it is least visible: models that
drift away from the required format still score.

This re-scores the completions already stored in each result file under the
official two-pattern cascade, so the size of the illusion is a measured number
rather than a worry. Report the official column.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

STRICT = re.compile(r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)", re.MULTILINE)
LOOSE = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")
BARE = re.compile(r"^\s*([A-Ea-e])(?:[).:,\s]|$)")


def official(completion: str) -> str:
    m = STRICT.findall(completion or "") or LOOSE.findall(completion or "")
    return m[-1].strip().rstrip(".").upper() if m else ""


def lenient(completion: str) -> str:
    m = (STRICT.findall(completion or "") or LOOSE.findall(completion or "")
         or BARE.findall(completion or ""))
    return m[-1].strip().rstrip(".").upper() if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--gold", required=True, help="dev jsonl, for the answers")
    ap.add_argument("--include", default="")
    args = ap.parse_args()

    gold, nch = {}, {}
    for line in open(args.gold, encoding="utf-8"):
        r = json.loads(line)
        gold[r["sample_id"]] = chr(65 + int(r["answer"]))
        nch[r["sample_id"]] = len(r["choices"])

    rows = []
    for d in args.dir:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            name = os.path.basename(f)[:-5]
            if args.include and not any(w and w in name
                                        for w in args.include.split(",")):
                continue
            try:
                j = json.load(open(f))
            except Exception:
                continue
            res = j.get("results") if isinstance(j, dict) else None
            if not res or j.get("summary", {}).get("total") != 1000:
                continue
            if "completion" not in res[0]:
                continue
            off = len_ = bare_only = 0
            for r in res:
                sid = r.get("sample_id")
                g = gold.get(sid)
                if g is None:
                    continue
                n = nch[sid]
                valid = {chr(65 + i) for i in range(n)}
                o = official(r.get("completion", ""))
                l = lenient(r.get("completion", ""))
                o = o if o in valid else ""
                l = l if l in valid else ""
                off += (o == g)
                len_ += (l == g)
                bare_only += (l == g and o != g)
            rows.append((name, j["summary"]["accuracy"], len_ / len(res),
                         off / len(res), bare_only))

    rows.sort(key=lambda x: -x[3])
    w = max(len(r[0]) for r in rows) + 2
    print(f"{'arm':<{w}}{'stored':>9}{'lenient':>9}{'OFFICIAL':>10}{'bare-only':>11}{'delta':>8}")
    for name, stored, len_, off, bo in rows:
        print(f"{name:<{w}}{stored*100:>9.2f}{len_*100:>9.2f}{off*100:>10.2f}"
              f"{bo:>11d}{(off-len_)*100:>8.2f}")
    print("\nOFFICIAL is what gsma-labs/evals would score. 'bare-only' counts "
          "rows credited solely by the extra fallback this repo added.")


if __name__ == "__main__":
    main()
