#!/usr/bin/env python3
"""Per-question coverage funnel for the synthetic-fact pipeline.

The trained run gained +16.4pp on the questions its facts were aimed at and lost
11.6pp on the rest. Whether that trade can ever be won depends on how wide the
aimed set can be made, so this measures the width at every stage rather than at
the end: how many of the 10,000 benchmark questions still have at least one fact
alive after retrieval, generation, validation, the novelty gate, and the
subsample.

The join is structural rather than semantic, but it changes key halfway down the
pipeline. Up to the novelty gate a chunk records `pulled_by` and `item_id` is
`<chunk_id>_<n>`, so the chunk is readable off the id alone. The merge into
`train_items.jsonl` re-keys every fact to a hash and copies `pulled_by` onto the
row, and `expand_views.py` then sets `fact_id = item_id`. So the later stages
join through that file rather than through the id.

Coverage is also reported against base correctness, because a question the base
already answers is one where new facts can only do harm, and that is the half of
the funnel the previous run never looked at.
"""
from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path


def chunk_of(fact_id: str) -> str:
    return fact_id.rsplit("_", 1)[0]


def load_pulls(paths) -> dict[str, list[str]]:
    """chunk_id -> the questions that retrieved it."""
    pulls: dict[str, list[str]] = {}
    for p in paths:
        if not p.exists():
            continue
        n = 0
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                pb = r.get("pulled_by")
                if isinstance(pb, str):
                    try:
                        pb = ast.literal_eval(pb)
                    except Exception:
                        pb = []
                if r.get("chunk_id") and pb:
                    pulls[r["chunk_id"]] = list(pb)
                    n += 1
        print(f"  {p.name}: {n} chunks", flush=True)
    return pulls


def qs_of_stage(path: Path, pulls, id_field: str, byfact=None) -> tuple[set, int, set]:
    """Questions reachable from one stage file, plus row and unit counts."""
    qs, units, rows = set(), set(), 0
    if not path.exists():
        return qs, rows, units
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            rows += 1
            fid = r.get(id_field)
            if not fid:
                continue
            if byfact is not None:            # post-merge: hashed fact ids
                units.add(fid)
                qs.update(byfact.get(fid, ()))
            else:
                c = r.get("chunk_id") or chunk_of(fid)
                units.add(c)
                qs.update(pulls.get(c, ()))
    return qs, rows, units


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--otfull", type=Path, required=True)
    ap.add_argument("--base-results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    R = args.root

    all_q = [json.loads(l)["sample_id"] for l in args.otfull.open(encoding="utf-8")]
    base = {x["sample_id"]: x["correct"]
            for x in json.load(args.base_results.open(encoding="utf-8"))["results"]}
    wrong = {q for q in all_q if not base.get(q, True)}
    print(f"benchmark {len(all_q)} questions, base wrong on {len(wrong)}\n", flush=True)

    print("loading chunk provenance", flush=True)
    pulls = load_pulls([R / f for f in (
        "data/chunks_pilot2k.jsonl", "data/chunks_rest.jsonl",
        "data/chunks_extra31b.jsonl", "data/chunks_addressable.jsonl",
        "data/chunks_D_wrong.jsonl", "data/chunks_targeted_wrong.jsonl")])
    print(f"  total {len(pulls)} chunks with provenance\n", flush=True)

    # the merge re-keys facts to a hash and carries provenance on the row
    byfact: dict[str, list[str]] = {}
    with (R / "data/synth/train_items.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            pb = r.get("pulled_by")
            if isinstance(pb, str):
                try:
                    pb = ast.literal_eval(pb)
                except Exception:
                    pb = []
            if r.get("item_id"):
                byfact[r["item_id"]] = list(pb or [])
    print(f"  train_items: {len(byfact)} facts re-keyed\n", flush=True)

    stages = [
        ("1 retrieved",  ["data/chunks_pilot2k.jsonl", "data/chunks_rest.jsonl",
                          "data/chunks_extra31b.jsonl"], "chunk_id", None),
        ("2 generated",  ["data/synth/pilot2k_items.jsonl", "data/synth/rest_items.jsonl",
                          "data/synth/extra31b_items.jsonl"], "item_id", None),
        ("3 validated",  ["data/synth/pilot2k_kept.jsonl", "data/synth/rest_kept.jsonl",
                          "data/synth/extra31b_kept.jsonl"], "item_id", None),
        ("4 novel",      ["data/synth/rest_novel.jsonl",
                          "data/synth/extra31b_novel.jsonl"], "item_id", None),
        ("5 merged",     ["data/synth/train_items.jsonl"], "item_id", byfact),
        ("6 views",      ["data/synth/views.jsonl"], "fact_id", byfact),
        ("7 subsampled", ["data/synth/views_sub.jsonl"], "fact_id", byfact),
        ("8 trained",    ["data/synth/train_mix_100k.jsonl"], "fact_id", byfact),
    ]

    report = {"n_questions": len(all_q), "base_wrong": len(wrong), "stages": []}
    for name, files, field, bf in stages:
        qs, rows, chunks = set(), 0, set()
        for f in files:
            q, n, c = qs_of_stage(R / f, pulls, field, bf)
            qs |= q; rows += n; chunks |= c
        hit_w = len(qs & wrong)
        rec = {"stage": name, "rows": rows, "units": len(chunks),
               "questions": len(qs), "coverage": round(len(qs) / len(all_q), 4),
               "of_base_wrong": hit_w,
               "recall_on_wrong": round(hit_w / len(wrong), 4),
               "of_base_right": len(qs) - hit_w}
        report["stages"].append(rec)
        print(f"{name:14s} rows={rows:8d} units={len(chunks):7d} "
              f"questions={len(qs):5d} ({rec['coverage']*100:5.2f}%)  "
              f"of-wrong={hit_w:5d} ({rec['recall_on_wrong']*100:5.2f}%)", flush=True)

    # what the last surviving stage misses, and whether the base can do those
    trained, _, _ = qs_of_stage(R / "data/synth/train_mix_100k.jsonl", pulls,
                                "fact_id", byfact)
    missed_wrong = sorted(wrong - trained)
    report["missed_wrong"] = len(missed_wrong)
    report["missed_wrong_ids"] = missed_wrong
    report["trained_ids"] = sorted(trained)
    print(f"\nbase-wrong questions with no fact in the trained mix: "
          f"{len(missed_wrong)} of {len(wrong)}", flush=True)

    args.out.write_text(json.dumps(report, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
