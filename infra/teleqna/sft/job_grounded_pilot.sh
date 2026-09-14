#!/usr/bin/env bash
# Track B pilot: generate spec-grounded QA and run the double gate on 500 chunks,
# to measure keep rates before committing the full 11,296.
#
# Pinned to NODE card 0. The pod-level default is CUDA_VISIBLE_DEVICES=2,3,5,7;
# NVIDIA_VISIBLE_DEVICES=all puts all eight physical devices in the container, so
# overriding to 0 here selects node card 0 and nothing else. Card 4 is Track A's
# GRPO work and 6/7 belong to other tenants.
#
# telelogs-base is mounted readOnly in this pod: scripts and corpora are read
# from there, everything written goes to /workspace/teleqna-spare4.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
BASE=/workspace/telelogs-base
OUT=/workspace/teleqna-spare4/results
INFRA="$BASE/runs/teleqna-sft/infra"
PYV="$BASE/venvs/venv/bin/python"          # vllm 0.11.0
MODEL="$BASE/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
CHUNKS="$BASE/runs/bench4/synth/data/chunks.jsonl"
TEST="$BASE/runs/bench4/teleqna/data/test.jsonl"
N="${N:-500}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '1p')
echo "node card 0 free=${free_mib}MiB   chunks=$N"
[ "$free_mib" -gt 100000 ] || { echo "ABORT: node card 0 is not free"; exit 14; }

mkdir -p "$OUT"
echo "#### generate START $(date -Iseconds)"
timeout 14400 "$PYV" "$INFRA/gen_grounded_qa.py" \
  --chunks "$CHUNKS" \
  --out "$OUT/grounded_pilot.jsonl" \
  --model "$MODEL" -k 4 --limit "$N"
echo "#### generate DONE $(date -Iseconds)"
wc -l "$OUT/grounded_pilot.jsonl"

echo "#### validate START $(date -Iseconds)"
timeout 14400 "$PYV" "$INFRA/validate_grounded_qa.py" \
  --items "$OUT/grounded_pilot.jsonl" \
  --chunks "$CHUNKS" --test "$TEST" \
  --out "$OUT/grounded_pilot_kept.jsonl" \
  --model "$MODEL" -k 4
echo "#### validate DONE $(date -Iseconds)"

echo "#### sample of what survived ####"
head -3 "$OUT/grounded_pilot_kept.jsonl" 2>/dev/null || echo "(nothing survived)"
