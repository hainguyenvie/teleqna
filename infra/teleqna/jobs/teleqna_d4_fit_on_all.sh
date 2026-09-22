#!/bin/bash
# Deliberately fit the system prompt on all 10,000 rows and score it on the same
# 10,000 rows.
#
# WHAT THIS NUMBER IS
# -------------------
# Not a result. Not submittable. It is the ceiling of prompt-only fitting when
# the optimiser is handed the answer key: the largest number a system prompt can
# produce on this benchmark by any mixture of strategy and recall.
#
# WHY IT IS WORTH RUNNING ANYWAY
# ------------------------------
# Held against the honest run (fit on 1,000, scored on the disjoint 9,000), the
# difference is a direct measurement of how much of prompt optimisation on this
# benchmark is memorisation rather than generalisation. That quantity is not
# available any other way, and it sets the discount to apply to every future
# prompt-tuning number on this track.
#
#   honest      = fit(train 400 + val 600)   scored on heldout 9,000
#   ceiling     = fit(all 10,000)            scored on the same 10,000
#   leakage     = ceiling - honest
#
# Everything except the data is identical to the honest run — same optimiser,
# same seed prompt, same adapter, same no-think decoding, same scorer — so the
# gap is attributable to the split and to nothing else.
set -euo pipefail
ROOT=/workspace/telelogs-bench4/teleqna
PY=/workspace/telelogs-bench4/dspy/.venv/bin/python
cd "$ROOT/code"

# train = every row; val = a stratified 2,000-row sample of the same rows.
# The val set is in-sample on purpose — candidate selection is part of what is
# being allowed to cheat here.
"$PY" - <<'PYEOF'
import json, hashlib, random, pathlib
from collections import defaultdict
root = pathlib.Path("/workspace/telelogs-bench4/teleqna")
out = root / "data/splits_fitall"
out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in (root / "data/test.jsonl").open(encoding="utf-8")]
by = defaultdict(list)
for r in rows:
    by[r.get("subject") or "<none>"].append(r)
val = []
for subject in sorted(by):
    pool = by[subject]
    seed = int(hashlib.sha256(f"fitall|{subject}".encode()).hexdigest()[:16], 16)
    val.extend(random.Random(seed).sample(pool, max(1, round(2000 * len(pool) / len(rows)))))
val.sort(key=lambda r: r["sample_index"])
for name, part in (("train", rows), ("val", val)):
    with (out / f"{name}.jsonl").open("w", encoding="utf-8") as h:
        for r in part:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps({"train": len(rows), "val": len(val), "disjoint": False,
                  "note": "val is a subset of train by design"}))
PYEOF

"$PY" dspy_optimize_prompt.py \
  --splits "$ROOT/data/splits_fitall" \
  --out "$ROOT/results/gepa_fitall" \
  --optimizer gepa --max-metric-calls 40000 \
  --threads 16 --max-tokens 40

"$PY" run_baseline.py \
  --data "$ROOT/data/test.jsonl" \
  --out "$ROOT/results/gepa_fitall_alldata" \
  --system "$ROOT/results/gepa_fitall/system_prompt.txt" \
  --workers 32 --max-tokens 40 --temperature 0.0

echo "### attribution on all 10,000: no-system -> prompt fitted on all 10,000"
"$PY" attribute_gain.py \
  --data "$ROOT/data/test.jsonl" \
  --heldout "$ROOT/data/test.jsonl" \
  --baseline "$ROOT/results/b0_nothink" \
  --optimized "$ROOT/results/gepa_fitall_alldata" \
  --out "$ROOT/results/gepa_fitall_attribution.json"
