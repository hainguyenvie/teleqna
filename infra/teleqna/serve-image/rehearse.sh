#!/usr/bin/env bash
# Diễn tập hợp đồng của image trên H200 mà không cần docker: chạy đúng lệnh vLLM
# mà entrypoint.sh sẽ chạy, với đúng tên model và cờ, trên card 6.
set -u
cd ~/projects/teleqna/runs/teleqna-8b
export PATH=$HOME/venv-vllm-nightly/bin:$PATH
CARD=${CARD:-6}; PORT=${PORT:-8020}
CUDA_VISIBLE_DEVICES=$CARD python -m vllm.entrypoints.openai.api_server \
  --model models/kit/wise_o3 \
  --served-model-name Qwen3-8B-Telco teleqna-8b-closedbook wise-o3 \
  --host 0.0.0.0 --port "$PORT" \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 4096 --gpu-memory-utilization 0.85 --max-num-seqs 64 \
  --enable-prefix-caching
