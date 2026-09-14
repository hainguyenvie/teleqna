#!/usr/bin/env bash
# Arm E stage 1 on card 0. Card 7 is off limits by request; 1-3 and 5 belong to
# another user's jobs, 4 and 6 are held by rag8weak and armV2.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
HOME_ROOT=/home/tensara
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
export PATH="$(dirname "$PY"):$PATH"
SFT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
MODEL="$HOME_ROOT/projects/telelogs/shared/models/OTel-2.0-31B-IT"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Reap only engines whose parent is already gone; anything still supervised is
# somebody's running job.
for pid in $(pgrep -f "VLLM::EngineCore" 2>/dev/null || true); do
  [ "$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')" = "1" ] || continue
  echo "reaping orphaned EngineCore pid=$pid"; kill -9 "$pid" 2>/dev/null || true; sleep 8
done

for i in $(seq 1 180); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
  [ "$free" -gt 100000 ] && break
  [ "$i" = 1 ] && echo "card 0 busy (${free}MiB) — waiting $(date -Iseconds)"
  sleep 20
done
[ "$free" -gt 100000 ] || { echo "ABORT: card 0 stuck at ${free}MiB"; exit 14; }
echo "card 0 free=${free}MiB $(date -Iseconds)"

echo "#### armE roundtrip START $(date -Iseconds)"
"$PY" -u $SFT/infra/roundtrip_armE.py \
  --base "$MODEL" \
  --facts "$SFT/data/armE_facts.jsonl" \
  --out "$SFT/data/armE_facts_rt.jsonl" \
  
echo "#### armE roundtrip DONE $(date -Iseconds)"
