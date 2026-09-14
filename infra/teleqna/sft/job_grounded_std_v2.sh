#!/usr/bin/env bash
# Grounded QA over the Tele-Data standard shard — the corpus rebuild.
#
# What changed from the run that produced distill_v1, and why. The method was
# not the problem. Measured on the same metric, same script:
#
#   Standards specifications rows with >=80% of their rare terms in one window
#     distill_v1 (GSMA marked/ tree, our chunking)     105 / 1,999
#     Tele-Data standard shard                       1,323 / 1,999
#
# distill1 moved Standards specifications 61.0 -> 64.0 while touching 5% of the
# rows it could have. This job points the identical generator at 12.6x the
# addressable surface: 50,815 chunks over 2,312 specifications, capped at 120
# chunks per spec so 38.331 does not eat the set.
#
# Run in slices, not one shot. vLLM buffers every completion and writes at the
# end, so a single 50k-chunk call returns nothing for five hours and loses
# everything if the card is taken. The generator skips chunk_ids already in
# --out, so each pass picks up the next slice and a preempted run costs one
# slice. The model reload between passes is ~90s against ~55min of generation.
#
# Card 4, this pod's own allocation, now genuinely free (143GB) — the leaked
# CUDA contexts from the killed evals were reclaimed. Sized dynamically anyway.
set -euo pipefail
ROOT=/workspace/teleqna-sft
BASE_ROOT=/workspace/telelogs-base
PYV="$BASE_ROOT/venvs/venv/bin/python"     # vllm 0.11.0
MODEL="$BASE_ROOT/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
CHUNKS="$ROOT/data/chunks_teledata_std.jsonl"
OUT="$ROOT/results/grounded_std_v2.jsonl"
SLICE="${SLICE:-10000}"
PASSES="${PASSES:-6}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$(dirname "$OUT")"
[ -s "$CHUNKS" ] || { echo "ABORT: no chunk file at $CHUNKS"; exit 15; }

TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n '1p')
for i in $(seq 1 60); do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '1p')
  [ "$free_mib" -gt 45000 ] && break
  [ "$i" = 1 ] && echo "card has only ${free_mib}MiB free — waiting $(date -Iseconds)"
  sleep 30
done
[ "$free_mib" -gt 45000 ] || { echo "ABORT: under 45GB free on the card"; exit 14; }
GPU_MEM=$(python3 -c "print(round(max(0.25, min(0.85, ($free_mib - 8000) / $TOTAL_MIB)), 2))")
echo "card free=${free_mib}MiB -> gpu_memory_utilization=$GPU_MEM  $(date -Iseconds)"
echo "chunks available: $(wc -l < "$CHUNKS")"

for p in $(seq 1 "$PASSES"); do
  echo "#### pass $p/$PASSES START $(date -Iseconds)  (out has $( [ -s "$OUT" ] && wc -l < "$OUT" || echo 0 ) items)"
  set +e
  timeout 14400 "$PYV" "$ROOT/infra/gen_grounded_qa.py" \
    --chunks "$CHUNKS" --out "$OUT" --model "$MODEL" \
    -k 4 --limit "$SLICE" --gpu-mem "$GPU_MEM" --max-tokens 1400
  rc=$?
  set -e
  # 124 is the timeout, 137 an OOM kill, anything else a real crash. Reported
  # rather than collapsed into one word, which cost a whole sweep once already.
  if [ $rc -ne 0 ]; then echo "#### pass $p FAILED rc=$rc $(date -Iseconds)"; break; fi
  echo "#### pass $p DONE $(date -Iseconds)"
done
echo "#### ALL DONE $(date -Iseconds)  items=$(wc -l < "$OUT")"
