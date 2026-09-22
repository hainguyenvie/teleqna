#!/usr/bin/env bash
# Letter probabilities for base and the tier-2 checkpoint on cards shared with the big run, then the partial-learning report.
cd ~/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python; export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6
CUDA_VISIBLE_DEVICES=1 $PY -u code/kit/letter_probs.py --ckpt $HOME/projects/_shared/models/Qwen3-8B --tag base > logs/letterprobs_base.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY -u code/kit/letter_probs.py --ckpt models/kit/tier2/ep1 --tag kit2_ep1 > logs/letterprobs_kit2.log 2>&1 &
wait; grep -h "chat logprob\|Error" logs/letterprobs_base.log logs/letterprobs_kit2.log
$PY code/kit/partial_learning.py base kit2_ep1 kit2_ep1
echo "#### LETTERPROBS END $(date -Iseconds)"
