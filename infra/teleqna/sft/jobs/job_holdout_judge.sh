#!/usr/bin/env bash
# 31B reads strong evidence on the 6,202 rows the margin router never escalated.
#
# This is the slice the gate is structurally blind to -- every one of the 497
# C_hard rows lives here, because C_hard means "confidently wrong", and a router
# keyed on confidence will never send those for a second opinion. It is also the
# slice arm V actively damaged.
#
# Waits for retrieval rather than being chained to it, so a retrieval hiccup does
# not silently produce a run over a half-written file.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${CARDS:?set CARDS}"
HOME_ROOT=/home/tensara
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
export PATH="$(dirname "$PY"):$PATH"
SFT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
MODEL="$HOME_ROOT/projects/telelogs/shared/models/OTel-2.0-31B-IT"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SET="$SFT/data/holdout1068_rag8_strong.jsonl"

for i in $(seq 1 240); do
  grep -aq "TRIM DONE" /tmp/holdout_retr.log 2>/dev/null && break
  [ "$i" = 1 ] && echo "waiting for retrieval+trim $(date -Iseconds)"
  sleep 30
done
grep -aq "TRIM DONE" /tmp/holdout_retr.log || { echo "ABORT: retrieval never finished"; exit 15; }
n=$(wc -l < "$SET")
echo "evidence set ready: $n rows $(date -Iseconds)"
[ "$n" -eq 1068 ] || { echo "ABORT: expected 1068 rows, got $n"; exit 16; }

for pid in $(pgrep -f "VLLM::EngineCore" 2>/dev/null || true); do
  [ "$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')" = "1" ] || continue
  echo "reaping orphaned EngineCore pid=$pid"; kill -9 "$pid" 2>/dev/null || true; sleep 8
done
for i in $(seq 1 360); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
  [ "$free" -gt 100000 ] && break
  [ "$i" = 1 ] && echo "card $CUDA_VISIBLE_DEVICES busy (${free}MiB) — waiting $(date -Iseconds)"
  sleep 20
done
[ "$free" -gt 100000 ] || { echo "ABORT: card stuck at ${free}MiB"; exit 14; }
echo "card $CUDA_VISIBLE_DEVICES free=${free}MiB $(date -Iseconds)"

# 32768, matching the escalated run: trimming to 8 windows caps the prompt near
# 20k tokens, and the earlier k=16 attempt died at 40960 for want of this.
echo "#### HOLDOUT JUDGE START $(date -Iseconds)"
"$PY" -u "$SFT/infra/eval_dev_vllm.py" \
  --base "$MODEL" --data "$SET" \
  --out-dir "$SFT/results/landscape" --tag holdout1068_ragstrong8 --with-base \
  --max-new 512 --tp 1 --gpu-mem 0.90 --max-model-len 32768
echo "#### HOLDOUT JUDGE DONE rc=$? $(date -Iseconds)"

# The strict parser will collapse again under this much context; recover the
# judgements, validating against the rows the harness did read.
"$PY" -u "$SFT/infra/reparse.py" \
  --result "$SFT/results/landscape/holdout1068_ragstrong8_base_nothink512.json" \
  --out "$SFT/results/landscape/holdout1068_ragstrong8_reparsed.json"
echo "#### HOLDOUT REPARSE DONE $(date -Iseconds)"
