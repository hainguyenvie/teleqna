#!/usr/bin/env bash
# After the vote-distillation stage: vote + raw-mode on the best big checkpoint (step24000) and raw-mode on vd/ep1.
cd ~/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python; export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6
until grep -q "VD_CHAIN_DONE\|ABORT" logs/vd_chain.log 2>/dev/null; do sleep 120; done
echo "#### POST-VD EVALS START $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=1 $PY -u code/kit/vote_eval.py --ckpt models/kit/big/step24000 --tag big_step24000 --gpu-mem 0.85 > logs/vote_big_step24000.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY -u code/kit/rawmode_eval.py --ckpt models/kit/big/step24000 --tag big_step24000 --gpu-mem 0.85 > logs/rawmode_big_step24000.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY -u code/kit/rawmode_eval.py --ckpt models/kit/vd/ep1 --tag vd_ep1 --gpu-mem 0.85 > logs/rawmode_vd_ep1.log 2>&1 &
wait
grep -hE "^big|greedy|raw-text|^vd" logs/vote_big_step24000.log logs/rawmode_big_step24000.log logs/rawmode_vd_ep1.log
echo "#### POST-VD EVALS END $(date -Iseconds)"
