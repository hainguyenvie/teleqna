#!/usr/bin/env bash
# First-pass audits with the base 8B as judge on cards shared with the big run: kit hallucination (card 1), label audit (card 2).
cd ~/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python; export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6
J=$HOME/projects/_shared/models/Qwen3-8B
CUDA_VISIBLE_DEVICES=1 $PY -u code/kit/kit_audit.py --judge $J --n 150 --gpu-mem 0.28 > logs/kit_audit_8b.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY -u code/kit/label_audit.py --judge $J --n 300 --gpu-mem 0.28 --tag q3_8b > logs/label_audit_8b.log 2>&1 &
wait; grep -vE "INFO|WARNING|it/s|^$|Loading" logs/kit_audit_8b.log | tail -25; grep -vE "INFO|WARNING|it/s|^$|Loading" logs/label_audit_8b.log | tail -25
echo "#### AUDITS-8B END $(date -Iseconds)"
