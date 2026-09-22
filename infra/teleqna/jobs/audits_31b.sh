#!/usr/bin/env bash
# Second-pass audits with OTel-2.0-31B-IT as judge (one full card), after the vote-distillation stage releases the cards:
# label audit on 600 random + broke/fixed samples, kit audit 200 rows/view. Measurement only.
cd ~/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python; export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
until grep -q "VD_CHAIN_DONE\|ABORT" logs/vd_chain.log 2>/dev/null; do sleep 300; done
for i in $(seq 1 240); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7); [ "$used" -le 1000 ] && break; sleep 30; done
J=$HOME/projects/_shared/models/OTel-2.0-31B-IT
echo "#### AUDITS-31B START $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=7 $PY -u code/kit/label_audit.py --judge $J --n 600 --gpu-mem 0.90 --tag otel31b > logs/label_audit_31b.log 2>&1; grep -vE "INFO|WARNING|it/s|^$|Loading" logs/label_audit_31b.log | tail -25
CUDA_VISIBLE_DEVICES=7 $PY -u code/kit/kit_audit.py --judge $J --n 200 --gpu-mem 0.90 > logs/kit_audit_31b.log 2>&1; grep -vE "INFO|WARNING|it/s|^$|Loading" logs/kit_audit_31b.log | tail -25
echo "#### AUDITS-31B END $(date -Iseconds)"
