#!/usr/bin/env python3
"""Anatomy of what the current best arm still gets wrong.

Every direction proposed so far was argued from the errors of armCE or of the
base. S1 is a different model -- it keeps 96% of the base's correct answers and
recovers a fifth of its errors -- so its residual is a different population, and
the next experiment should be chosen against that residual rather than against
the one two arms ago.

The reachability columns are the point. An error that eight retrieved chunks fix
in context is an error whose fact exists in the corpus and can be found, so it
is addressable by anything that gets corpus content into the weights. An error
that survives even the benchmark's own explanation is not addressable by any
amount of corpus work, and counting those separately is what turns "the ceiling
is 97.5" into a number that can actually be planned against.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

AGG = re.compile(r"\b(all|none|both)\s+of\s+the\s+above\b", re.I)


def load(p: Path):
    d = json.loads(p.read_text())
    return {x["sample_id"]: x for x in d["results"]}, d.get("summary", {})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--arm", default="S1")
    args = ap.parse_args()
    R = args.root

    qs = {}
    for line in (R / "data/otfull10000.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        qs[r["sample_id"]] = r

    A, sa = load(R / f"results/otfull_arms/otfull_{args.arm}_nothink512.json")
    B, sb = load(R / "results/otfull_arms/otfull_base_nothink512.json")
    CE, _ = load(R / "results/otfull_arms/otfull_armCE_nothink512.json")
    RAG, _ = load(R / "results/rag/otfull_rag8_base_think.json")
    EXP, _ = load(R / "results/rag/otfull_rag8_expl_base_think.json")
    aimed = set(json.loads((R / "results/aimed_true.json").read_text()))

    err = [i for i in A if not A[i]["correct"]]
    print(f"{args.arm} {sa.get('accuracy',0)*100:.2f}  base {sb.get('accuracy',0)*100:.2f}"
          f"   errors {len(err)}\n")

    # where the residual came from
    reg = [i for i in err if B[i]["correct"]]
    keep = [i for i in err if not B[i]["correct"]]
    print("thanh phan cua residual")
    print(f"  hong do train (base dung -> {args.arm} sai)   {len(reg):5d}  "
          f"{len(reg)/len(err)*100:5.2f}%")
    print(f"  chua bao gio chua duoc (base cung sai)      {len(keep):5d}  "
          f"{len(keep)/len(err)*100:5.2f}%")

    # reachability of the residual
    print("\nkha nang cham toi cua residual")
    rows = [("8 chunk truy hoi chua duoc", [i for i in err if RAG.get(i, {}).get("correct")]),
            ("chi giai thich goc chua duoc",
             [i for i in err if EXP.get(i, {}).get("correct")
              and not RAG.get(i, {}).get("correct")]),
            ("khang ca giai thich goc",
             [i for i in err if not EXP.get(i, {}).get("correct")])]
    for nm, ids in rows:
        print(f"  {nm:32s} {len(ids):5d}  {len(ids)/len(err)*100:5.2f}%  "
              f"= {len(ids)/100:5.2f} diem")

    # was the aggressive arm able to do these?
    ce_ok = [i for i in err if CE[i]["correct"]]
    print(f"\n  armCE lam dung duoc {len(ce_ok)} cau trong so nay "
          f"({len(ce_ok)/len(err)*100:.2f}%) -- tran cua viec 'day manh tay hon'")

    # was the pipeline ever aimed here
    print(f"  da tung duoc nham fact toi          {len([i for i in err if i in aimed]):5d}  "
          f"{len([i for i in err if i in aimed])/len(err)*100:5.2f}%")

    # shape of the question
    agg = [i for i in err if AGG.search(" ".join(qs[i]["choices"]))]
    agg_all = [i for i in qs if AGG.search(" ".join(qs[i]["choices"]))]
    print(f"  co phuong an tong hop               {len(agg):5d}  "
          f"{len(agg)/len(err)*100:5.2f}%  (toan bo bench {len(agg_all)/len(qs)*100:.2f}%)")
    print(f"  khong parse duoc                    {sum(1 for i in err if not A[i].get('parsed')):5d}")

    # subject, ranked by how much of the deficit each carries
    print("\ntheo chu de")
    bysub = Counter(qs[i].get("subject", "?") for i in err)
    tot = Counter(qs[i].get("subject", "?") for i in qs)
    print(f"  {'subject':26s} {'n':>5s} {'acc':>7s} {'base':>7s} {'loi':>6s} {'%bench':>7s}")
    for s, n in sorted(bysub.items(), key=lambda kv: -kv[1]):
        ids = [i for i in qs if qs[i].get("subject") == s]
        acc = sum(A[i]["correct"] for i in ids) / len(ids) * 100
        bacc = sum(B[i]["correct"] for i in ids) / len(ids) * 100
        print(f"  {s:26s} {tot[s]:5d} {acc:7.2f} {bacc:7.2f} {n:6d} {n/len(qs)*100:7.2f}")

    out = R / f"results/err_{args.arm}.json"
    out.write_text(json.dumps({
        "arm": args.arm, "errors": len(err),
        "regression": reg, "never_fixed": keep,
        "rag_fixable": [i for i in err if RAG.get(i, {}).get("correct")],
        "expl_only": [i for i in err if EXP.get(i, {}).get("correct")
                      and not RAG.get(i, {}).get("correct")],
        "unreachable": [i for i in err if not EXP.get(i, {}).get("correct")],
    }, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
