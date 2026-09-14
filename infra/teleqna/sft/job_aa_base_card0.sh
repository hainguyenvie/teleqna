#!/usr/bin/env bash
# Re-measure the bare base on card 0 / spare4, to make the GRPO-full ladder
# legible.
#
# The ladder that job_grpo_full_card0.sh produces has no base reference in it —
# its sweep passes only --adapter arms. So every number it emits would have to be
# read against base thinking 75.20, which was measured in a different process, on
# card 4, in the teleqna-sft tree. Same venv (venvs/venv, vLLM 0.11.0) and the
# same scorer, so the comparison is probably fine; "probably" is the problem.
#
# This project's rule is that numbers from two eval stacks are never compared,
# and the reason that rule exists is that we got burned assuming equivalence. A
# different card and a different process is a weaker version of the same risk:
# vLLM batches greedily but continuous batching still changes how sequences are
# grouped, and small numeric differences follow. At an MDE of ~2.2pp on
# dev-1000, half a point of drift is enough to turn a real GRPO gain into a
# marginal one or the reverse.
#
# So spend four minutes and one model load to answer it directly. If this prints
# 75.20 the ladder can be read against the existing table as-is. If it does not,
# the ladder must be read against THIS number instead, and the gap itself is
# worth knowing.
#
# Deliberately no --adapter: enable_lora stays off, which is also the cleanest
# possible reproduction of how base was scored originally.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
BASE_ROOT=/workspace/telelogs-base
ROOT=/workspace/teleqna-spare4
SFT="$BASE_ROOT/runs/teleqna-sft"
INFRA="$SFT/infra"
PYV="$BASE_ROOT/venvs/venv/bin/python"     # vllm 0.11.0 — same stack as the table
BASE="$BASE_ROOT/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# Wait for the GRPO run and its own sweep to hand card 0 back, rather than
# fighting them for it. 45GB is what an 8B plus a working KV cache needs.
for i in $(seq 1 240); do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '1p')
  [ "$free_mib" -gt 45000 ] && break
  [ "$i" = 1 ] && echo "card 0 busy (${free_mib}MiB free) — waiting $(date -Iseconds)"
  sleep 30
done
[ "$free_mib" -gt 45000 ] || { echo "ABORT: card 0 still busy"; exit 14; }
echo "card 0 free=${free_mib}MiB  $(date -Iseconds)"

# tag=aacheck so it cannot collide with the dev1000_base_think.json that the
# earlier card-4 run already wrote into the results tree.
echo "#### A/A base thinking START $(date -Iseconds)"
timeout 7200 "$PYV" "$INFRA/eval_dev_vllm.py" \
  --base "$BASE" --data "$SFT/data/dev1000.jsonl" \
  --out-dir "$ROOT/results/vllm" --tag aacheck \
  --with-base --thinking --gpu-mem 0.85
echo "#### A/A base thinking DONE $(date -Iseconds)"

python3 - "$ROOT/results/vllm/aacheck_base_think.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
ref = 0.7520   # base thinking, card 4, teleqna-sft, same venv and scorer
got = s["accuracy"]
d = (got - ref) * 100
print(f"card0 base thinking = {got:.4f}   card4 reference = {ref:.4f}   drift = {d:+.2f}pp")
# 0.5pp is a quarter of dev-1000's MDE; past that, treating the two cards as one
# measurement stack is no longer defensible and the ladder needs its own base.
print("VERDICT:", "comparable — read the ladder against the existing table"
      if abs(d) <= 0.5 else
      "NOT comparable — read the GRPO ladder against THIS base, not 75.20")
PY
