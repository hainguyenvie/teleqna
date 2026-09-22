#!/usr/bin/env bash
# Inspect-faithful scoring (run_baseline.py = the GSMA harness contract) of a checkpoint served by vLLM's OpenAI server on
# card 0 (shared, gpu-mem 0.28). Usage: CKPT=models/kit/vd3/ep1 TAG=vd3_ep1 bash jobs/harness_eval.sh
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=${CARDS:-0}
cd "$ROOT"; CKPT="${CKPT:?}"; TAG="${TAG:?}"; PORT=${PORT:-8011}
for i in $(seq 1 480); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CUDA_VISIBLE_DEVICES); [ "$used" -le 100000 ] && break; sleep 30; done
echo "#### HARNESS $TAG START $(date -Iseconds)"
"$PY" -m vllm.entrypoints.openai.api_server --model "$CKPT" --served-model-name Qwen/Qwen3-8B --port $PORT --gpu-memory-utilization 0.28 --max-model-len 4096 --dtype bfloat16 > logs/vllm_serve_$TAG.log 2>&1 &
SP=$!
for i in $(seq 1 120); do curl -s http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; sleep 5; done
VLLM_CHAT_URL=http://127.0.0.1:$PORT/v1/chat/completions TELEQNA_MODEL=Qwen/Qwen3-8B python3 code/run_baseline.py --data data/eval/otfull10000.jsonl --out results/harness/$TAG --workers 32 --max-tokens 32 > logs/harness_$TAG.log 2>&1
kill $SP 2>/dev/null; sleep 5; pkill -f "vllm.entrypoints.openai.api_server --model $CKPT" 2>/dev/null
tail -25 logs/harness_$TAG.log | cut -c1-200; cat results/harness/$TAG/summary.json 2>/dev/null | head -c 800
echo; echo "#### HARNESS $TAG END $(date -Iseconds)"
