#!/usr/bin/env bash
# Rerun the UTR selection pass on a dedicated card once big4 training has finished (never share a card with FSDP training).
cd ~/projects/teleqna/runs/teleqna-8b; export PATH=$HOME/venv-vllm-nightly/bin:$PATH HF_HUB_OFFLINE=1
until grep -q "TRAIN-BIG4 DONE rc=0" logs/big4r_chain.log 2>/dev/null; do sleep 60; done; sleep 30
CUDA_VISIBLE_DEVICES=2 python -u code/kit/utr_select.py --ckpt models/kit/merge_f --n 300000 --gpu-mem 0.85 > logs/utr_select.log 2>&1
grep -vE "^$|Processed|Rendering|INFO|WARNING|it/s" logs/utr_select.log | cut -c1-260 | tail -12
