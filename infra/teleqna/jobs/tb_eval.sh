#!/usr/bin/env bash
# After trace-back pass 2: build TB-8 eval set, score with base Qwen3-8B on a free card, report functional coverage.
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
until grep -q "TRACEBACK3 END" logs/traceback3_chain.log 2>/dev/null; do sleep 300; done
"$PY" -u code/kit/build_tb_eval.py || exit 1
for i in $(seq 1 720); do for c in 6 7 0 2; do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 30000 ] && { CARD=$c; break 2; }; done; sleep 60; done
echo "#### TB-8 EVAL START card=$CARD $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARD timeout --foreground 7200 "$PY" -u code/eval_dev_vllm.py --base "$HOME/projects/_shared/models/Qwen3-8B" --data data/eval/tb_rag8.jsonl --out-dir results/landscape --tag tb_rag8_q3_8b --with-base --max-new 512 --tp 1 --gpu-mem 0.75 --max-model-len 24576 > logs/eval_tb_rag8.log 2>&1
echo "#### TB-8 EVAL DONE rc=$? $(date -Iseconds)"
"$PY" code/kit/functional_coverage.py | tee logs/functional_coverage.log
echo "#### TB_EVAL END"
