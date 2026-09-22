#!/bin/bash
# Score the GEPA-optimised system prompt on the held-out 9,000, then attribute
# whatever it gained.
#
# The evaluation runs through run_baseline.py, not through DSPy — same template,
# same Inspect-verbatim parser, same decoding as b0_nothink. The only difference
# from the baseline run is the injected system message, so the delta is
# attributable to the prompt and nothing else.
#
# The baseline side of the comparison is b0_nothink subset to the held-out ids
# by attribute_gain.py, so no rows are re-run and the pairing is exact.
#
# Read the attribution before the headline. GEPA's proposals were observed
# writing dev-set answers into the prompt verbatim ("Vehicle-to-everything
# (V2X)...", "IEEE 802.3af..."), which is a cheat sheet, not a strategy. If the
# gain does not survive on rows the shape heuristic gets wrong, it is not real.
#
# Three arms, not two. Without the middle one there is no way to tell "GEPA
# found something" from "any system prompt at all helps":
#   b0_nothink            no system message   (already run, subset to held-out)
#   seed_nothink_heldout  the hand-written seed prompt GEPA started from
#   gepa_nothink_heldout  GEPA's evolved prompt
set -euo pipefail
ROOT=/workspace/telelogs-bench4/teleqna
PY=/workspace/telelogs-bench4/dspy/.venv/bin/python
cd "$ROOT/code"

"$PY" - <<'PYEOF'
import json, pathlib
root = pathlib.Path("/workspace/telelogs-bench4/teleqna/results/gepa_nothink")
d = json.loads((root / "system_prompt.json").read_text(encoding="utf-8"))
(root / "seed_prompt.txt").write_text(d["seed_instructions"], encoding="utf-8")
print("seed prompt chars:", len(d["seed_instructions"]))
print("gepa prompt chars:", len(d["optimized_instructions"]))
PYEOF

"$PY" run_baseline.py \
  --data "$ROOT/data/splits/heldout.jsonl" \
  --out "$ROOT/results/seed_nothink_heldout" \
  --system "$ROOT/results/gepa_nothink/seed_prompt.txt" \
  --workers 32 --max-tokens 40 --temperature 0.0

"$PY" run_baseline.py \
  --data "$ROOT/data/splits/heldout.jsonl" \
  --out "$ROOT/results/gepa_nothink_heldout" \
  --system "$ROOT/results/gepa_nothink/system_prompt.txt" \
  --workers 32 --max-tokens 40 --temperature 0.0

echo "### attribution: no-system -> seed"
"$PY" attribute_gain.py \
  --data "$ROOT/data/test.jsonl" \
  --heldout "$ROOT/data/splits/heldout.jsonl" \
  --baseline "$ROOT/results/b0_nothink" \
  --optimized "$ROOT/results/seed_nothink_heldout" \
  --out "$ROOT/results/seed_attribution.json"

echo "### attribution: no-system -> GEPA"
"$PY" attribute_gain.py \
  --data "$ROOT/data/test.jsonl" \
  --heldout "$ROOT/data/splits/heldout.jsonl" \
  --baseline "$ROOT/results/b0_nothink" \
  --optimized "$ROOT/results/gepa_nothink_heldout" \
  --out "$ROOT/results/gepa_attribution.json"

echo "### attribution: seed -> GEPA (what GEPA itself added)"
"$PY" attribute_gain.py \
  --data "$ROOT/data/test.jsonl" \
  --heldout "$ROOT/data/splits/heldout.jsonl" \
  --baseline "$ROOT/results/seed_nothink_heldout" \
  --optimized "$ROOT/results/gepa_nothink_heldout" \
  --out "$ROOT/results/gepa_over_seed_attribution.json"
