#!/bin/bash
# THE single held-out shot: merge dpo3@200 -> serve with vLLM -> score the
# frozen 9,000 with the exact b0_think harness config (temp 0.6, seed 42,
# max_tokens 6000). Two gates before the 9,000 is touched:
#   1. merged model must serve and answer;
#   2. dev-1000 through this stack must not collapse (<0.75 = merge bug).
# The dev-1000 pass also gives the tuned-vs-base comparison on the official
# vLLM stack (base think dev-1000 = 75.90 from b0_think's subset).
set -euo pipefail
ROOT=/workspace/teleqna-sft
BASE_ROOT=/workspace/telelogs-base
PYT="$BASE_ROOT/venvs/venv-train/bin/python"   # peft/transformers (merge)
PYV="$BASE_ROOT/venvs/venv/bin/python"         # vllm
BASE_MODEL="$BASE_ROOT/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
ADAPTER="$ROOT/models/teleqna-dpo3-checkpoints/checkpoint-200"
MERGED="$ROOT/models/teleqna-dpo3s200-merged"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "#### merge START $(date -Iseconds)"
if [ ! -e "$MERGED/config.json" ]; then
  timeout 3600 "$PYT" "$ROOT/infra/merge_adapter.py" \
    --base "$BASE_MODEL" --adapter "$ADAPTER" --out "$MERGED"
fi

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free=${free_mib}MiB"
[ "$free_mib" -gt 100000 ] || { echo "ABORT: GPU not free"; exit 14; }

echo "#### vllm serve START $(date -Iseconds)"
"$PYV" -m vllm.entrypoints.openai.api_server \
  --model "$MERGED" --served-model-name teleqna-dpo3s200 \
  --port 8010 --max-model-len 12288 --gpu-memory-utilization 0.9 \
  --disable-log-requests > "$ROOT/results/vllm_dpo3s200.log" 2>&1 &
VPID=$!
trap 'kill $VPID 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8010/health >/dev/null 2>&1 && break
  kill -0 $VPID 2>/dev/null || { echo "ABORT: vllm died"; tail -30 "$ROOT/results/vllm_dpo3s200.log"; exit 15; }
  sleep 10
done
curl -sf http://127.0.0.1:8010/health >/dev/null || { echo "ABORT: vllm never healthy"; exit 15; }

export TELEQNA_MODEL=teleqna-dpo3s200
export VLLM_CHAT_URL=http://127.0.0.1:8010/v1/chat/completions

echo "#### GATE: dev-1000 via vllm think START $(date -Iseconds)"
timeout 14400 "$PYV" "$ROOT/infra/run_baseline.py" \
  --data "$ROOT/data/dev1000.jsonl" \
  --out "$ROOT/results/vllm_dev1000_dpo3s200_think" \
  --thinking --max-tokens 6000 --temperature 0.6 --workers 48

acc=$("$PYV" -c "import json;print(json.load(open('$ROOT/results/vllm_dev1000_dpo3s200_think/summary.json'))['accuracy'])")
echo "#### dev-1000 vllm-think accuracy = $acc (base same-stack: 0.7590)"
ok=$("$PYV" -c "print(1 if $acc >= 0.75 else 0)")
[ "$ok" = "1" ] || { echo "ABORT: dev collapsed, not burning held-out"; exit 16; }

echo "#### HELD-OUT 9000 START $(date -Iseconds)"
timeout 43200 "$PYV" "$ROOT/infra/run_baseline.py" \
  --data "$ROOT/data/heldout9000.jsonl" \
  --out "$ROOT/results/vllm_heldout9000_dpo3s200_think" \
  --thinking --max-tokens 6000 --temperature 0.6 --workers 48
echo "#### heldout ALL DONE $(date -Iseconds)"
